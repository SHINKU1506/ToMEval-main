"""Content quality review gate for synthesized ToM items.

Evaluates candidate items (typically after passing DifficultyGate) against
semantic, quantitative, epistemic, CoT-shortcut, and sibling-diversity criteria,
returning a structured accept/reject decision with actionable revision feedback.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from src import runner

logger = logging.getLogger(__name__)

ReviewDefectCategory = Literal[
    "competing_interpretation",
    "quantifier_or_boundary_ambiguity",
    "de_re_de_dicto_mismatch",
    "epistemic_leak_or_ungrounded_access",
    "cot_shortcut_bypass",
    "old_template_or_parable",
    "sibling_isomorphism",
    "cognitive_operation_mismatch",
]

_DEFAULT_REVIEW_TEMPERATURE = 0.2
_DEFAULT_REVIEW_MAX_TOKENS = 8192


class QualityReviewDecision(BaseModel):
    """Structured verdict emitted by the content quality reviewer."""

    decision: Literal["accept", "reject"] = Field(
        description="'accept' only if all content, determinacy, anti-shortcut, and anti-isomorphism checks pass; otherwise 'reject'."
    )
    defect_categories: List[ReviewDefectCategory] = Field(
        default_factory=list,
        description="Zero categories when accepted; one or more matching defect tags when rejected.",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific, evidence-cited flaws found in the story, options, or solver CoT traces.",
    )
    revision_guidance: str = Field(
        default="",
        description="Concrete fix instructions for the next synthesis retry when rejected; empty when accepted.",
    )
    summary: str = Field(
        default="",
        description="One-sentence overall assessment.",
    )


def _format_rollout_traces(gate_stats: Optional[Dict[str, Any]], max_traces: int = 4) -> str:
    """Render de-shuffled solver rollout CoTs so the reviewer can inspect shortcut bypasses."""
    if not isinstance(gate_stats, dict) or not gate_stats:
        return "(no solver rollout stats available)"
    lines = [
        f"- Solve rate: {gate_stats.get('correct', 0)}/{gate_stats.get('total', 0)} "
        f"({gate_stats.get('solve_rate', 0.0):.2f})"
    ]
    counts = gate_stats.get("selected_option_counts")
    if counts:
        lines.append(f"- Option counts: {json.dumps(counts, ensure_ascii=False)}")

    traces = list(gate_stats.get("rollout_traces") or [])
    if not traces and isinstance(gate_stats.get("solver_trace"), dict):
        traces = [gate_stats["solver_trace"]]

    for idx, tr in enumerate(traces[:max_traces], 1):
        if not isinstance(tr, dict):
            continue
        status = "CORRECT" if tr.get("is_correct") else "WRONG"
        chosen = tr.get("selected_option_text") or tr.get("selected_letters") or "(unknown)"
        tail = str(tr.get("reasoning_tail") or tr.get("content") or "").strip()
        if len(tail) > 900:
            tail = tail[:900] + "\n...[truncated]"
        lines.append(f"\n### Solver Rollout #{idx} [{status}] -> {chosen}")
        if tail:
            lines.append(f"```text\n{tail}\n```")
    return "\n".join(lines)


def _format_siblings_for_review(sibling_questions: Optional[Sequence[Dict[str, Any]]]) -> str:
    """Summarize accepted sibling variants for cross-variant isomorphism checks."""
    if not sibling_questions:
        return "(none)"
    lines: List[str] = []
    for idx, sib in enumerate(sibling_questions, 1):
        if not isinstance(sib, dict):
            continue
        q_text = str(sib.get("question") or "").strip()
        story = str(sib.get("story") or "").strip()
        plan = str(sib.get("_reasoning_trap_plan") or "").strip()
        lines.append(f"- Sibling #{idx} Question: {q_text[:200]}")
        if plan:
            lines.append(f"  Trap Plan: {plan[:260]}")
        elif story:
            lines.append(f"  Story Opening: {story[:220]}")
    return "\n".join(lines) if lines else "(none)"


def build_quality_review_prompt(
    row: Dict[str, Any],
    report: Dict[str, Any],
    dataset_name: str,
    gate_stats: Optional[Dict[str, Any]] = None,
    sibling_questions: Optional[Sequence[Dict[str, Any]]] = None,
    max_traces: int = 4,
) -> str:
    """Build a strict content-quality audit prompt for one candidate question."""
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    dimension = str((report.get("_meta") or {}).get("dimension") or report.get("dimension") or meta.get("report_dimension") or "")
    primary_op = str(report.get("primary_cognitive_operation") or meta.get("primary_cognitive_operation") or "")
    hard_constraints = report.get("hard_constraints") or []
    trap_plan = str(row.get("_reasoning_trap_plan") or "").strip() or "(none)"

    candidate_payload = {
        "story": row.get("story", ""),
        "question": row.get("question", ""),
        "correct_answers": answer.get("correct_answers", []),
        "wrong_answers": answer.get("wrong_answers", []),
        "lang": meta.get("lang", "en"),
        "difficulty": meta.get("difficulty", ""),
    }

    return f"""You are a strict Theory-of-Mind (ToM) dataset quality auditor.
Review the candidate {dataset_name} item below and decide whether to **accept** it into the training pool or **reject** it for targeted revision.
Note: passing the difficulty gate (e.g., 1-3/4 solve rate) does NOT mean an item is high quality — many 1-3/4 items have competing interpretations, boundary ambiguities, or CoT shortcuts.

## Target Dimension & Constraints
- Dimension: `{dimension}`
- Primary Cognitive Operation: `{primary_op}`
- Report Hard Constraints:
{json.dumps(hard_constraints, ensure_ascii=False, indent=2)}

## Candidate Item
```json
{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}
```

## Author's Reasoning Trap Plan
{trap_plan}

## Solver Rollout Traces
{_format_rollout_traces(gate_stats, max_traces=max_traces)}

## Already Accepted Sibling Variants for This Report
{_format_siblings_for_review(sibling_questions)}

## Audit Checklist (Reject if ANY item fails)
1. **competing_interpretation**: Is any distractor genuinely defensible under the story text or pragmatic norms? (Example failures: a restriction sign posted only on a sub-zone door that does not bar watching from outside; an indirect question that works equally well as a self-interest request when the speaker also lacks the item; multiple valid pragmatic readings.)
2. **quantifier_or_boundary_ambiguity**: Do numeric or vague-quantifier boundaries ("almost none", "nearly half", "几乎没有", "将近一半") hinge on a +/-1 edge (e.g. >0 vs =0) or leave intermediate sales/transfers unclosed so solvers can bypass the intended inference? Global determinacy (>=4-unit margin or exact post-update recount) overrides any vague-quantifier edge.
3. **de_re_de_dicto_mismatch**: If labels/lids/containers were swapped, do the question ("which X does A have in mind?") and options mix omniscient descriptors ("currently capped with...") with the agent's stale mental label so both the physical-target option and mental-label option are defensible?
4. **epistemic_leak_or_ungrounded_access**: Is it ambiguous whether an agent actually saw/heard a key preview or cue, or does an unopened/private message leak support to the correct answer rather than solely baiting a distractor?
5. **cot_shortcut_bypass**: Inspect the solver rollout traces above. Did the solver get the right answer by bypassing the planned reasoning hop (e.g., surface phrasing match, unclosed sales arithmetic, or ignoring the stale artifact) rather than reconciling the actual epistemic/desire update?
6. **old_template_or_parable**: Does the story rely on an overused skeleton (a 4-stage physical occlusion table, a child asking "are X only for Y?" / "do X only get given out once Y?", or a front-loaded explanatory prior-event parable that hands over the emotional/causal template)?
7. **sibling_isomorphism**: Does this candidate reuse the same domain/setting, historical anchor, and causal mechanism as an already-accepted sibling variant above?
8. **cognitive_operation_mismatch**: Does the question fail to genuinely test `{primary_op}` or violate any non-conflicting dimension constraint?

Return ONLY valid JSON matching `QualityReviewDecision` (`decision`, `defect_categories`, `issues`, `revision_guidance`, `summary`). Set `decision` to `"accept"` ONLY when all 8 checks cleanly pass."""


def _normalize_review_decision(parsed: Any) -> Optional[Dict[str, Any]]:
    """Normalize a parsed QualityReviewDecision into a consistent dict."""
    if parsed is None:
        return None
    data = parsed.model_dump() if isinstance(parsed, BaseModel) else (dict(parsed) if isinstance(parsed, dict) else None)
    if not data:
        return None

    decision = str(data.get("decision") or "").strip().lower()
    if decision not in ("accept", "reject"):
        return None

    categories = [str(c).strip() for c in (data.get("defect_categories") or []) if str(c).strip()]
    issues = [str(i).strip() for i in (data.get("issues") or []) if str(i).strip()]
    guidance = str(data.get("revision_guidance") or "").strip()
    summary = str(data.get("summary") or "").strip()

    # Enforce consistency: non-empty defect tags always force rejection.
    if categories and decision == "accept":
        decision = "reject"
    if decision == "reject" and not issues and summary:
        issues = [summary]
    if decision == "accept":
        categories = []

    return {
        "decision": decision,
        "defect_categories": categories,
        "issues": issues,
        "revision_guidance": guidance,
        "summary": summary,
    }


class QualityReviewer:
    """LLM-backed semantic content quality reviewer for synthesized candidates."""

    def __init__(
        self,
        reviewer_llm_config: Dict[str, Any],
        fail_open: bool = False,
        max_traces: int = 4,
    ):
        config = dict(reviewer_llm_config)
        config.setdefault("temperature", _DEFAULT_REVIEW_TEMPERATURE)
        config.setdefault("max_tokens", _DEFAULT_REVIEW_MAX_TOKENS)
        self.model_name = str(config.get("model_name", "unknown"))
        self.fail_open = bool(fail_open)
        self.max_traces = int(max_traces)
        self.client = runner.create_llm_client(config)

    def review_batch(
        self,
        items: Sequence[Dict[str, Any]],
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        """Review a batch of candidate items and return one normalized verdict dict per item."""
        if not items:
            return []

        prompts = [
            build_quality_review_prompt(
                row=item["row"],
                report=item["report"],
                dataset_name=dataset_name,
                gate_stats=item.get("gate_stats"),
                sibling_questions=item.get("sibling_questions"),
                max_traces=self.max_traces,
            )
            for item in items
        ]
        responses = self.client.batch_generate_structure(
            prompts,
            QualityReviewDecision,
            mode="create",
            desc=f"Reviewing {dataset_name}",
        )

        verdicts: List[Dict[str, Any]] = []
        for resp in responses:
            raw = resp.content if resp is not None else None
            normalized = _normalize_review_decision(raw)
            if normalized is None:
                if self.fail_open:
                    logger.warning("[quality_reviewer] invalid/empty review response; fail_open=True -> accept")
                    verdicts.append({
                        "decision": "accept",
                        "defect_categories": [],
                        "issues": [],
                        "revision_guidance": "",
                        "summary": "Reviewer response unavailable; accepted via fail_open.",
                        "error": True,
                    })
                else:
                    verdicts.append({
                        "decision": "error",
                        "defect_categories": [],
                        "issues": ["Quality reviewer did not return a valid structured decision."],
                        "revision_guidance": "",
                        "summary": "Reviewer response unavailable.",
                        "error": True,
                    })
                continue
            verdicts.append(normalized)
        return verdicts


def build_quality_reviewer(review_config: Optional[Dict[str, Any]]) -> Optional[QualityReviewer]:
    """Build a QualityReviewer from resolved `roles.quality_review` + `synthesis.quality_review` config."""
    if not review_config or not review_config.get("enabled", False):
        return None
    llm = dict(review_config.get("llm") or {})
    if not llm.get("model_name") or not llm.get("api_key"):
        raise ValueError("quality_review is enabled but its reviewer model_name/api_key is missing")
    return QualityReviewer(
        reviewer_llm_config=llm,
        fail_open=bool(review_config.get("fail_open", False)),
        max_traces=int(review_config.get("max_traces", 4)),
    )
