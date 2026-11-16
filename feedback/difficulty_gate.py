"""
难度闭环：用 solver 模型对合成题跑 k 次，区分难度不合格和判分不完整。

复用评测链路（predict_records + judge_repeat），因此 prompt 排版、选项 shuffle、
\\boxed{} 抽取与正式评测一致。只有收到 k 个可判分的 MCQ 答案才使用分数；
判分不完整时默认拒绝。
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from src import runner
from src.evaluation.data import normalize_sample
from src.evaluation.judge import judge_repeat
from src.evaluation.prediction import predict_records
from src.evaluation.protocols import llm_overrides_for

logger = logging.getLogger(__name__)

_GATED_PROMPT_TYPES = {"mcq_single", "mcq_multi", "mcq_grouped"}
_DEFAULT_TRACE_TAIL_CHARS = 700
TRACE_MODES = ("tail_truncate", "anchor", "prompt_summary")

_OPTION_ANCHOR_RE = re.compile(
    r"(?:\bOption\s+[A-D]\b|(?<![A-Za-z])[A-D][.．):：]\s|"
    r"\b(?:eliminate|rule\s+out|eliminating|looking\s+at\s+(?:the\s+)?options)\b|"
    r"选项\s*[A-D]|来看选项|逐一分析|排除(?:选项)?\s*[A-D])",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？\n]+\s*")
_SUMMARY_FIELD_RE = re.compile(
    r"(?:^|\n)\s*(?:\[Summary\]|\[总结\])\s*[:：]\s*(.+?)(?=(?:\n|\s)*\\box(?:ed)?\s*\{|\n\s*\n|$)",
    re.DOTALL | re.IGNORECASE,
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_SUMMARY_DIRECTIVE_EN = (
    "\n\nBefore your final \\boxed{X}, include one brief line starting with "
    "`[Summary]:` (1-2 sentences) explaining why the wrong options are eliminated "
    "and why the chosen option is selected, then output \\boxed{X} on the last line."
)
_SUMMARY_DIRECTIVE_ZH = (
    "\n\n请在给出最终 \\boxed{X} 前，另起一行用 `[总结]：` 开头写出 1–2 句话，"
    "简要说明为什么排除错误选项以及为什么选择该选项，然后另起一行输出 \\boxed{X}。"
)


def _sentence_start_before(text: str, pos: int) -> int:
    last_end = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text[:pos]):
        last_end = m.end()
    return last_end


def _sentence_aligned_tail(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    raw_start = len(text) - max_chars
    m = _SENTENCE_BOUNDARY_RE.search(text, pos=raw_start)
    if m and m.end() < len(text):
        return text[m.end():].strip(), True
    return text[raw_start:].strip(), True


def _extract_anchored_reasoning(reasoning: str, max_tail_chars: int) -> str:
    if not reasoning:
        return ""
    max_anchor_chars = max_tail_chars * 2 if max_tail_chars > 0 else 0
    m = _OPTION_ANCHOR_RE.search(reasoning)
    if m:
        sent_start = _sentence_start_before(reasoning, m.start())
        anchored = reasoning[sent_start:].strip()
        if max_anchor_chars <= 0 or len(anchored) <= max_anchor_chars:
            return ("...[earlier reasoning omitted]...\n" + anchored) if sent_start > 0 else anchored
        tail, clipped = _sentence_aligned_tail(reasoning, max_anchor_chars)
        return ("...[earlier reasoning omitted]...\n" + tail) if clipped else tail
    tail, clipped = _sentence_aligned_tail(reasoning, max_tail_chars)
    return ("...[earlier reasoning omitted]...\n" + tail) if clipped else tail


def _extract_prompt_summary(content: str, reasoning: str) -> str:
    for text in (content, reasoning):
        if not text:
            continue
        m = _SUMMARY_FIELD_RE.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""


class _SummaryPromptClient:
    """Append bilingual [Summary]/[总结] directive to gate prompts in `prompt_summary` mode."""

    def __init__(self, inner: Any):
        self._inner = inner

    def batch_generate(self, prompts: List[str], desc: str = "", system_prompts: Optional[List[str]] = None):
        augmented = [
            p + (_SUMMARY_DIRECTIVE_ZH if _CJK_RE.search(p) else _SUMMARY_DIRECTIVE_EN)
            for p in prompts
        ]
        if system_prompts is not None:
            return self._inner.batch_generate(augmented, desc=desc, system_prompts=system_prompts)
        return self._inner.batch_generate(augmented, desc=desc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _format_trace_entry(
    record: Dict[str, Any],
    result: Dict[str, Any],
    max_tail_chars: int = _DEFAULT_TRACE_TAIL_CHARS,
    trace_mode: str = "anchor",
) -> Dict[str, Any]:
    """Extract a de-shuffled reasoning trace using `tail_truncate`, `anchor`, or `prompt_summary`."""
    if trace_mode not in TRACE_MODES:
        raise ValueError(f"Unknown trace_mode: {trace_mode!r}; expected one of {TRACE_MODES}")
    options = dict(record.get("options") or {})
    extracted = result.get("extracted")
    letters = [extracted] if isinstance(extracted, str) else list(extracted or [])
    selected_texts = [str(options.get(str(letter), letter)) for letter in letters]
    selected_option_text = " | ".join(text for text in selected_texts if text)
    pred = record.get("pred") or {}
    reasoning = str(pred.get("reasoning") or "").strip()
    content = str(pred.get("content") or "").strip()

    summary = ""
    if trace_mode == "tail_truncate":
        if max_tail_chars > 0 and len(reasoning) > max_tail_chars:
            reasoning_tail = "...[earlier reasoning omitted]...\n" + reasoning[-max_tail_chars:]
        else:
            reasoning_tail = reasoning
        content_out = (content[:400] + "...") if len(content) > 400 else content
    else:
        if trace_mode == "prompt_summary":
            summary = _extract_prompt_summary(content, reasoning)
        reasoning_tail = _extract_anchored_reasoning(reasoning, max_tail_chars)
        content_out = (content[:1500] + "...") if len(content) > 1500 else content

    entry: Dict[str, Any] = {
        "trace_mode": trace_mode,
        "is_correct": bool(result.get("is_correct")),
        "selected_letters": letters,
        "selected_option_text": selected_option_text,
        "option_map": options,
        "reasoning_tail": reasoning_tail,
        "content": content_out,
        "_raw_len": len(reasoning) + len(content),
    }
    if summary:
        entry["summary"] = summary
    return entry


class DifficultyGate:
    """pass@k solve-rate band gate. A question passes when min_solve_rate <= solve_rate <= max_solve_rate."""

    def __init__(
        self,
        solver_llm_config: Dict[str, Any],
        k: int = 4,
        max_solve_rate: float = 0.75,
        min_solve_rate: float = 0.0,
        protocol: str = "cot",
        fail_open: bool = False,
        trace_mode: str = "anchor",
        max_tail_chars: int = _DEFAULT_TRACE_TAIL_CHARS,
    ):
        if k < 1:
            raise ValueError("difficulty gate k must be >= 1")
        if trace_mode not in TRACE_MODES:
            raise ValueError(f"Unknown trace_mode: {trace_mode!r}; expected one of {TRACE_MODES}")
        self.k = int(k)
        self.min_solve_rate = float(min_solve_rate)
        self.max_solve_rate = float(max_solve_rate)
        if self.min_solve_rate > self.max_solve_rate:
            raise ValueError("difficulty gate min_solve_rate must be <= max_solve_rate")
        self.protocol = protocol
        self.fail_open = bool(fail_open)
        self.trace_mode = trace_mode
        self.max_tail_chars = int(max_tail_chars)
        config = dict(solver_llm_config)
        config.update(llm_overrides_for(protocol))
        self.model_name = str(config.get("model_name", "unknown"))
        self.client = runner.create_content_client(config)

    def evaluate(self, rows: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """Return {sample_id: {"correct": int, "total": int, "solve_rate": float}}."""
        samples = []
        for index, row in enumerate(rows):
            try:
                samples.append(normalize_sample(row, index))
            except ValueError as exc:
                logger.warning(f"[difficulty_gate] skip malformed row: {exc}")
        if not samples:
            return {}

        eval_client = _SummaryPromptClient(self.client) if self.trace_mode == "prompt_summary" else self.client
        records = predict_records(samples, dataset_name, eval_client, self.k, protocol=self.protocol)
        gated = [r for r in records if r["prompt_type"] in _GATED_PROMPT_TYPES]
        results = judge_repeat(gated) if gated else []

        counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        traces_by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record, result in zip(gated, results):
            if result.get("error_reason") not in (None, "wrong_answer"):
                continue
            options = record.get("options")
            if options:
                extracted = result["extracted"]
                letters = [extracted] if isinstance(extracted, str) else extracted
                if not letters or any(letter not in options for letter in letters):
                    continue
            sid = record["sample_id"]
            entry = counts[sid]
            entry[0] += int(bool(result.get("is_correct")))
            entry[1] += 1
            traces_by_sid[sid].append(
                _format_trace_entry(
                    record,
                    result,
                    max_tail_chars=self.max_tail_chars,
                    trace_mode=self.trace_mode,
                )
            )

        output: Dict[str, Dict[str, Any]] = {}
        for sid, (c, t) in counts.items():
            solve_rate = c / t if t else 0.0
            traces = traces_by_sid.get(sid, [])
            opt_counts = Counter(
                tr["selected_option_text"] for tr in traces if tr.get("selected_option_text")
            )
            correct_traces = [tr for tr in traces if tr["is_correct"]]
            wrong_traces = [tr for tr in traces if not tr["is_correct"]]
            rep_trace: Optional[Dict[str, Any]] = None
            if wrong_traces and (solve_rate < self.min_solve_rate or not correct_traces):
                wrong_counts = Counter(
                    tr["selected_option_text"] for tr in wrong_traces if tr.get("selected_option_text")
                )
                top_wrong = wrong_counts.most_common(1)[0][0] if wrong_counts else ""
                candidates = [tr for tr in wrong_traces if tr.get("selected_option_text") == top_wrong] or wrong_traces
                chosen = min(
                    candidates,
                    key=lambda x: (0 if (x["reasoning_tail"] or x["content"]) else 1, x["_raw_len"]),
                )
                rep_trace = {k: v for k, v in chosen.items() if k != "_raw_len"}
            elif correct_traces:
                chosen = min(
                    correct_traces,
                    key=lambda x: (0 if (x["reasoning_tail"] or x["content"]) else 1, x["_raw_len"]),
                )
                rep_trace = {k: v for k, v in chosen.items() if k != "_raw_len"}

            output[sid] = {
                "correct": c,
                "total": t,
                "solve_rate": solve_rate,
                "selected_option_counts": dict(opt_counts),
                "solver_trace": rep_trace,
            }
        return output

    def rejection_reason(self, stats: Optional[Dict[str, Any]]) -> Optional[str]:
        if not stats or stats.get("total") != self.k:
            return None if self.fail_open else "missing_eval_stats"
        rate = float(stats["solve_rate"])
        if rate > self.max_solve_rate:
            return "too_easy"
        if rate < self.min_solve_rate:
            return "too_hard_or_unanswerable"
        return None

    def passes(self, stats: Optional[Dict[str, Any]]) -> bool:
        return self.rejection_reason(stats) is None


def build_difficulty_gate(gate_config: Optional[Dict[str, Any]]) -> Optional[DifficultyGate]:
    """Build a gate from the resolved `roles.difficulty_gate` + `synthesis.difficulty_gate` config."""
    if not gate_config or not gate_config.get("enabled", False):
        return None
    llm = dict(gate_config.get("llm") or {})
    if not llm.get("model_name") or not llm.get("api_key"):
        raise ValueError("difficulty_gate is enabled but its solver model_name/api_key is missing")
    return DifficultyGate(
        solver_llm_config=llm,
        k=int(gate_config.get("k", 4)),
        max_solve_rate=float(gate_config.get("max_solve_rate", 0.75)),
        min_solve_rate=float(gate_config.get("min_solve_rate", 0.0)),
        protocol=str(gate_config.get("protocol", "cot")),
        fail_open=bool(gate_config.get("fail_open", False)),
        trace_mode=str(gate_config.get("trace_mode", "anchor")),
        max_tail_chars=int(gate_config.get("max_tail_chars", _DEFAULT_TRACE_TAIL_CHARS)),
    )
