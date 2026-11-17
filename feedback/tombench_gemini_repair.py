"""Gemini-native repair helpers for ToMBench."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from feedback.stage3_synthesis import _strip_mcq_footer, _validate_synthesized_question

SIGNPOST_RE = re.compile(
    r"\b(?:completely alone|without telling anyone|without mentioning|having spoken to neither|"
    r"spoke to neither|talked to neither|told neither|out of earshot)\b|"
    r"独自一人|没有告诉任何人|未告诉任何人|未向任何人提起|毫不知情|完全不知情",
    re.IGNORECASE,
)

_PARTIAL_ACCESS_MARKERS = (
    "heard", "saw", "noticed", "message", "text", "phone", "partial", "fragment",
    "before", "after", "while", "later", "earlier", "left", "returned", "stepped", "entered",
    "离开", "返回", "走后", "看到", "听到", "发现", "纸条", "消息", "之前", "之后",
)


def _repair_mode(stats: Dict[str, Any]) -> str:
    sc = stats.get("single_correct", stats.get("correct"))
    total = stats.get("total", 8)
    if sc is not None and total and int(sc) <= 1:
        return "answerability_fix"
    return "escalate_difficulty"


def build_repair_prompt(row: Dict[str, Any], stats: Dict[str, Any], last_error: str = "") -> str:
    answer = row.get("answer") or {}
    mode = _repair_mode(stats or {})
    if mode == "answerability_fix":
        mode_guidance = (
            "Mode: ANSWERABILITY FIX (solver solved 0/8 or 1/8 — likely missing premise, physical contradiction, or factually true distractor).\n"
            "- Keep the core ToM dilemma, but make the decisive clue observable/inferable BEFORE target_moment.\n"
            "- Resolve any physical obstacle (e.g. if transport/tools are missing, ensure the correct answer addresses it), and align the addressee spatially with the requested object.\n"
            "- If using approximate quantifiers ('nearly half', 'almost none', '将近一半', '几乎没有'), state total-count scope explicitly across rooms/categories and keep comparisons >= 4 units from the midpoint (no ±1/±2 overlapping vague estimates).\n"
            "- Do NOT leave an overheard confession unaccounted for when testing verified vs. tampered work, and keep any gap between a partial message preview and speech tight (1-2 minutes).\n"
            "- Verify EVERY wrong option is unambiguously false under the story (no factually true distractors)."
        )
    else:
        mode_guidance = (
            "Mode: DIFFICULTY ESCALATION (solver solved >= 7/8 — currently too easy for CoT).\n"
            "- Do NOT do a 1-to-1 structural re-skinning; rewrite the scenario and question (do NOT copy the old easy question verbatim) "
            "by adding a new reasoning hop (e.g., 2nd-order belief or two sequential updates with a partial rollback), a counter-directional partial cue "
            "that tempts a wrong option (NEVER let the partial cue reinforce the correct answer), and at least one neutral intervening beat before target_moment.\n"
            "- No Distractor-Killing Patches & No Front-Loaded Parables: NEVER insert convenience facts that make a strong distractor physically impossible, "
            "early explicit self-disqualification lines ('I only like X', 'Swimming, sorry'), camera-closeup spoilers on forgotten objects, "
            "or explanatory prior-event parables; keep distractors physically live and failing only on epistemic access or timing.\n"
            "- Avoid the 4-stage glass-wall vs. under-desk/solid-wall occlusion table, explicit calendar-day (Tuesday vs. Wednesday) tie-breakers, "
            "and objectively false 'because' clauses in emotion distractors; ensure at least two options match any stated two-part persuasion framing.\n"
            "- Ensure the strongest wrong option reflects what a CoT solver picks if it skips the target mental-state update."
        )
    retry_note = f"\nPrevious validation rejection: {last_error}. Fix this error strictly.\n" if last_error else ""
    return f"""Repair this ToMBench item using the hard-synthesis contracts.
{mode_guidance}

Core Story & Option Contracts:
- Use partial knowledge plus a true misleading cue, not zero information.
- Show access implicitly through timing, object location, partial messages, occlusion, or departures.
- NEVER use explicit isolation signposts ("completely alone", "without telling anyone", "without mentioning", "having spoken to neither", "privately without X knowing", "out of earshot", "独自一人", "毫不知情").
- NEVER narrate the queried belief/misinterpretation directly with narrator spoiler verbs ("assumed", "thought", "believed", "mistook", "misread", "unbeknownst to", "did not know/realize", "in reality", "误以为", "不知情", "实际上").
- Avoid any repeated gaze/smile template ("catches eye", "glances toward", "smiles") and whimsical decoy objects; rotate concrete evidence channels instead.
- Use a reasoning_trap_plan, an exact target_moment, and stop the story verbatim at target_moment with NO post-query spoiler.
- Keep all 4 options balanced in length/detail (0.80x-1.20x), single-claim (no bundled dual-claim clauses), grounded only in story entities, matched in certainty/hedging style, and avoid length-biased "cannot tell / pick at random" meta-options.
{retry_note}
Return only JSON with: reasoning_trap_plan, target_moment, story, question, options (A/B/C/D), correct_letter.
Do not embed options in question.

Existing item:
Story: {row.get('story', '')}
Question: {row.get('question', '')}
Correct: {answer.get('correct_answers', [])}
Wrong: {answer.get('wrong_answers', [])}
Pass@8 stats: {stats}
"""


def validate_repaired_row(row: Dict[str, Any]) -> Optional[str]:
    story = str(row.get("story") or "").strip()
    question = str(row.get("question") or "").strip()
    target = str(row.get("target_moment") or "").strip()
    options = row.get("options") or {}
    correct = str(row.get("correct_letter") or "").upper()
    if not story or not question or not target:
        return "missing_story_question_or_target_moment"
    if SIGNPOST_RE.search(story):
        return "explicit_isolation_signpost"
    compact_story = re.sub(r"\s+", " ", story).strip()
    compact_target = re.sub(r"\s+", " ", target).strip()
    if not compact_story.endswith(compact_target):
        return "story_does_not_end_at_target_moment"
    if set(k.upper() for k in options) != {"A", "B", "C", "D"} or correct not in {k.upper() for k in options}:
        return "invalid_options"
    norm_opts = {k.upper(): _strip_mcq_footer(v) for k, v in options.items()}
    values = [norm_opts[k] for k in "ABCD"]
    if any(not v for v in values) or len(set(v.casefold() for v in values)) != 4:
        return "duplicate_options"
    if re.search(r"(?<![A-Za-z])[A-D][.．)]\s+", question):
        return "question_contains_embedded_options"
    lower = story.lower()
    if not any(x in lower for x in _PARTIAL_ACCESS_MARKERS):
        return "no_partial_observation_marker"
    stage3_check = {
        "story": story,
        "question": _strip_mcq_footer(question),
        "answer": {
            "correct_answers": [norm_opts[correct]],
            "wrong_answers": [norm_opts[k] for k in "ABCD" if k != correct],
        },
        "_target_moment": target,
    }
    return _validate_synthesized_question(stage3_check, "ToMBench", difficulty="hard")


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text[len(fence):].lstrip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if text.endswith(fence):
            text = text[:-len(fence)].rstrip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    return value


class GeminiRepairClient:
    def __init__(self, api_url: str, api_key: str, model: str, timeout: float = 180.0):
        self.api_url, self.api_key, self.model = api_url.rstrip("/"), api_key, model
        self.timeout = timeout

    def native_url(self) -> str:
        origin = self.api_url[:-3] if self.api_url.endswith("/v1") else self.api_url
        return f"{origin}/v1beta/models/{self.model}:generateContent"

    def repair(self, row: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
        temperature = 0.35 if _repair_mode(stats or {}) == "answerability_fix" else 0.6
        last = ""
        for attempt in range(3):
            payload = {
                "contents": [{"role": "user", "parts": [{"text": build_repair_prompt(row, stats, last_error=last if attempt > 0 else "")}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192, "responseMimeType": "application/json"},
            }
            try:
                response = requests.post(self.native_url(), headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"}, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(str(part.get("text", "")) for part in parts if part.get("text") is not None)
                candidate = _extract_json(text)
                error = validate_repaired_row(candidate)
                if error is None:
                    return candidate
                last = error
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** (attempt + 1))
        raise RuntimeError(last or "repair failed")


def _existing_in_band_valid(row: Dict[str, Any]) -> bool:
    story = str(row.get("story") or "").strip()
    if not story or SIGNPOST_RE.search(story):
        return False
    return _validate_synthesized_question(row, "ToMBench", difficulty="") is None


def repair_file(
    input_path: Path,
    stats_path: Path,
    output_path: Path,
    api_config: Path,
    model: str,
    workers: int,
    keep_sweet_spot: bool = True,
    min_target_correct: int = 2,
    max_target_correct: int = 6,
) -> Dict[str, Any]:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if isinstance(raw_stats, dict) and "per_sample" in raw_stats:
        raw_stats = raw_stats["per_sample"]
    stats = {row["sample_id"]: row for row in raw_stats}
    config = tomllib.loads(api_config.read_text(encoding="utf-8"))["synthesis"]
    client = GeminiRepairClient(config["api_url"], config["api_key"], model)
    results, failures = [], []
    kept_in_band = 0

    def one(row: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(row.get("meta") or {})
        sid = str(meta.get("id", ""))
        st = stats.get(sid, {})
        sc = st.get("single_correct", st.get("correct"))
        if keep_sweet_spot and sc is not None and min_target_correct <= int(sc) <= max_target_correct and _existing_in_band_valid(row):
            kept = dict(row)
            meta["repair_source_id"] = sid
            meta["repair_action"] = "kept_in_band"
            kept["meta"] = meta
            return kept
        candidate = client.repair(row, st)
        raw_options = candidate.pop("options")
        options = {k.upper(): _strip_mcq_footer(v) for k, v in raw_options.items()}
        letter = str(candidate.pop("correct_letter")).upper()
        candidate.pop("reasoning_trap_plan", None)
        candidate.pop("target_moment", None)
        candidate["question"] = _strip_mcq_footer(candidate.get("question", ""))
        candidate["answer"] = {"correct_answers": [options[letter]], "wrong_answers": [options[k] for k in "ABCD" if k != letter]}
        meta["repair_source_id"], meta["repair_model"] = sid, model
        meta["repair_action"] = _repair_mode(st)
        candidate["meta"] = meta
        return candidate

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            sid = str((row.get("meta") or {}).get("id", ""))
            try:
                res = future.result()
                if (res.get("meta") or {}).get("repair_action") == "kept_in_band":
                    kept_in_band += 1
                results.append(res)
            except Exception as exc:
                failures.append({"sample_id": sid, "error": str(exc)})
    results.sort(key=lambda row: str((row.get("meta") or {}).get("id", "")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in results), encoding="utf-8")
    output_path.with_suffix(".failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"input": len(rows), "repaired": len(results), "kept_in_band": kept_in_band, "failed": len(failures), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, default=Path("api_config.toml"))
    parser.add_argument("--model", default="gemini-4-hydrogen")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite-in-band", action="store_true", help="Also rewrite items already in 2-6/8")
    args = parser.parse_args()
    print(json.dumps(
        repair_file(
            args.input, args.stats, args.output, args.api_config, args.model, args.workers,
            keep_sweet_spot=not args.overwrite_in_band,
        ),
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
