"""
Stage 3：从维度诊断报告合成新样本

读取 Stage 2 输出的 dimension_reports.jsonl，按报告生成新题目，
保存为与 train_data parquet 一致的格式（小写 story/question/answer/meta）。
"""

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, Field
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from .prompts import (
    build_stage2_generation_from_report_prompt,
    resolve_variant_difficulty,
    SYNTHESIS_FORMAT_REGISTRY,
    DATASET_SKILL_REGISTRY,
    _sombench_parse_dim,
)

logger = logging.getLogger(__name__)


_MCQ_FOOTER_RE = re.compile(
    r"(?:\n|\s)+(?:请选择(?:正确答案|最合适的一项)|Please\s+(?:select|choose)\s+the\s+(?:correct|best)\s+answer|正确答案\s*[:：]).*$",
    re.DOTALL | re.IGNORECASE,
)

_OPTION_MARKER_GUARD = r"(?<!Room\s)(?<!Grade\s)(?<!Type\s)(?<!Plan\s)(?<!Option\s)(?<![A-Za-z])"


def _strip_mcq_footer(text: str) -> str:
    """Remove trailing prompt-leak instructions such as '\\n请选择正确答案。'."""
    return _MCQ_FOOTER_RE.sub("", str(text or "")).strip()


def _parse_options_from_question(q_text: str, correct_letter: str):
    """从内嵌选项的 question 文本中解析正确答案和错误答案文本。

    支持 'A. ...' 或 'A．...' 格式（中英文句点）。
    返回 (correct_text, wrong_texts)。
    """
    correct_letter = correct_letter.strip().upper()
    options_map: dict = {}
    pattern = re.compile(
        rf"{_OPTION_MARKER_GUARD}([A-D])[.．)]\s*(.*?)(?=\s*{_OPTION_MARKER_GUARD}[A-D][.．)]|$)",
        re.DOTALL,
    )
    for m in pattern.finditer(q_text):
        options_map[m.group(1)] = _strip_mcq_footer(m.group(2))
    correct_text = options_map.get(correct_letter, correct_letter)
    wrong_texts = [v for k, v in sorted(options_map.items()) if k != correct_letter]
    return correct_text, wrong_texts


def _strip_embedded_options(q_text: str) -> str:
    """Remove the answer-choice suffix from a ToMBench question.

    The generator schema historically asked the model to place options in the
    question. The standardized evaluator renders options separately, so
    retaining that suffix creates two conflicting option blocks.
    """
    match = re.search(rf"{_OPTION_MARKER_GUARD}A[.．)]\s+", q_text, re.IGNORECASE)
    if not match:
        return _strip_mcq_footer(q_text)
    return q_text[: match.start()].rstrip()


def _parse_existing_pair_id(meta_id: str) -> Optional[Tuple[int, int]]:
    """Extract the report/sample pair from a generated question ID.

    Dataset-specific question types may add a final suffix, for example
    synthetic_tombench_000_0_0. The report and sample indices remain the two
    numeric components immediately before that optional suffix.
    """
    match = re.fullmatch(r"synthetic_.+?_(\d+)_(\d+)(?:_[^_]+)?", str(meta_id or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


_HARD_SIGNPOST_PATTERNS = (
    re.compile(r"\bcompletely alone\b", re.IGNORECASE),
    re.compile(r"\bwithout telling anyone\b", re.IGNORECASE),
    re.compile(r"\bwithout mentioning\b", re.IGNORECASE),
    re.compile(r"\bhaving spoken to neither\b", re.IGNORECASE),
    re.compile(r"\b(?:spoke|talked|told) to neither\b", re.IGNORECASE),
    re.compile(r"\btold neither\b", re.IGNORECASE),
    re.compile(r"\bprivately (?:discovered|noticed|learned|checked|knew)\b", re.IGNORECASE),
    re.compile(r"\bout of earshot\b", re.IGNORECASE),
    re.compile(r"\bunbeknownst to\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+\w+(?:'s)?\s+knowledge\b", re.IGNORECASE),
    re.compile(r"\bdid not (?:know|realize)\b", re.IGNORECASE),
    re.compile(r"\bnever (?:learned|realized|found out)\b", re.IGNORECASE),
    re.compile(r"\bin reality\b", re.IGNORECASE),
    re.compile(
        r"独自一人|没有告诉任何人|未告诉任何人|未向任何人提起|毫不知情|完全不知情|"
        r"误以为|浑然不知|并不知情|却不知道|全然不知|实际上|事实上"
    ),
)


def _find_explicit_signpost(story: str) -> Optional[str]:
    """Return the matched boolean isolation signpost if present, else None."""
    for pat in _HARD_SIGNPOST_PATTERNS:
        m = pat.search(story)
        if m:
            return m.group(0)
    return None


_GAZE_CUE_RE = re.compile(
    r"\b(?:caught|catches|catching)\s+\w+(?:'s)?\s+eye\b|\bglanc\w*\s+(?:pointedly\s+|briefly\s+)?(?:toward|towards|at)\b|"
    r"(?:眼神|目光|对视|瞥了[^，。！？]*一眼|使了个眼色)",
    re.IGNORECASE,
)
_SMILE_CUE_RE = re.compile(
    r"\b(?:smil\w*|grin\w*|wink\w*)\b|(?:微笑|笑了笑|嘴角[^，。！？]*笑|相视一笑|眨了眨眼)",
    re.IGNORECASE,
)
_WHIMSICAL_DECOY_RE = re.compile(
    r"\b(?:goofy|comical|whimsical|silly-looking|funny-looking|cartoonish)\b|(?:滑稽|搞笑|卡通)",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(?:unsure|uncertain|not sure|not yet sure|unclear|wonders? whether|undecided)\b|"
    r"(?:不确定|拿不准|吃不准|说不准|不太确定|无法确定|判断不出)",
    re.IGNORECASE,
)
_INDETERMINACY_META_RE = re.compile(
    r"\b(?:cannot|can't)\s+(?:tell|identify|uniquely\s+identify|determine)\b|\bpick\s+(?:one\s+)?at\s+random\b|"
    r"无法确定|无法判断|判断不出|随机选",
    re.IGNORECASE,
)
_OPTION_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_OPTION_NAME_STOPWORDS = {
    "She", "Her", "His", "They", "The", "This", "That", "Then", "When", "While", "Because",
    "Since", "After", "Before", "Only", "Both", "Neither", "Either", "Go", "Ask", "Take",
    "Head", "Walk", "Set", "Step", "Return", "Open", "Stay", "Wheel", "Plug", "Restart",
    "Inspect", "Shift", "Relieve", "Trade", "To", "Tell", "Keep", "Let", "Wait", "Leave",
}


def _find_template_issue(story: str, q_dict: Dict[str, Any]) -> Optional[str]:
    """Detect the over-used gaze/smile template and whimsical decoy objects in hard items."""
    sentences = [
        s
        for s in re.split(r"(?<=[.!?\"”])\s+|(?<=[。！？])\s*", story.strip())
        if s.strip()
    ]
    tail = " ".join(sentences[-2:])
    target = str(q_dict.get("_target_moment") or "")
    for segment in (tail, target):
        if segment and _GAZE_CUE_RE.search(segment) and _SMILE_CUE_RE.search(segment):
            return "gaze_smile_template_cue_rewrite_with_assigned_evidence_channel"
    if _WHIMSICAL_DECOY_RE.search(story):
        return "whimsical_decoy_object_remove_amusement_red_herring"
    return None


_CAUSAL_PREFIX_SPLIT_RE = re.compile(
    r"(?:[，,;；]\s*(?:因为|由于|since\b|because\b|as\b)|\s+(?:because|since)\b)",
    re.IGNORECASE,
)
_NUMERIC_QUANTITY_RE = re.compile(
    r"(?:\d+|[一二两三四五六七八九十百千]+|half|quarter|dozen|score)",
    re.IGNORECASE,
)
_FRACTION_GLOSS_RE = re.compile(
    r"(?:[，,]\s*|\s+)(?:略少于|略多于|超过了?|不到|不足|恰好是|正好是|也就是|即)(?:总数|全部)?的?(?:一半|二分之一|三分之一|四分之一)|"
    r",\s*(?:which\s+is\s+)?(?:just\s+under|just\s+over|exactly|precisely)\s+(?:one[- ])?(?:half|third|quarter)\b",
    re.IGNORECASE,
)


def _find_option_structural_leak(correct: List[str], wrong: List[str]) -> Optional[str]:
    """Detect shared leading prefixes (non-orthogonal 2-part options) and redundant fraction glosses."""
    options = correct + wrong
    prefixes = []
    for opt in options:
        parts = _CAUSAL_PREFIX_SPLIT_RE.split(opt, maxsplit=1)
        if len(parts) == 2 and parts[0].strip():
            prefixes.append(parts[0].strip().casefold())
    if len(prefixes) == len(options) and len(set(prefixes)) == 1:
        return "all_options_share_identical_leading_claim_use_2x2_orthogonal_options"

    glossed = sum(
        1 for opt in options
        if _NUMERIC_QUANTITY_RE.search(opt) and _FRACTION_GLOSS_RE.search(opt)
    )
    if glossed >= 2:
        return "numeric_options_contain_fraction_gloss_use_bare_estimates"
    return None


def _find_option_grounding_issue(story: str, correct: List[str], wrong: List[str]) -> Optional[str]:
    """Reject options that introduce unseen named entities or break hedging symmetry."""
    options = correct + wrong
    for opt in options:
        for m in _OPTION_NAME_RE.finditer(opt):
            name = m.group(0)
            # 句首大写词（如 "Proud and relieved"）不是专名，只检查句中的大写词。
            before = opt[: m.start()].rstrip()
            if not before or before[-1] in ".!?;:\"'“‘(—-":
                continue
            if name in _OPTION_NAME_STOPWORDS:
                continue
            if not re.search(rf"\b{re.escape(name)}\b", story):
                return "option_introduces_entity_absent_from_story"
    hedged = [bool(_HEDGE_RE.search(opt)) for opt in options]
    if sum(hedged) == 1:
        return "single_hedged_option_match_certainty_style_across_options"
    return None


def _validate_synthesized_question(
    q_dict: Dict[str, Any],
    dataset_name: str,
    difficulty: str = "",
) -> Optional[str]:
    """Validate structural integrity, anti-signpost rules, and option symmetry.

    Returns a short failure reason string for retry feedback, or None when valid.
    """
    story = str(q_dict.get("story") or "").strip()
    question = str(q_dict.get("question") or "").strip()
    if not story or not question:
        return "empty_story_or_question"

    if difficulty == "hard" and _find_explicit_signpost(story) is not None:
        return "explicit_isolation_signpost_detected_rewrite_boundaries_implicitly"

    if difficulty == "hard":
        target_moment = str(
            q_dict.get("_target_moment") or q_dict.get("target_moment") or ""
        ).strip()
        if not target_moment:
            return "missing_target_moment"
        story_compact = re.sub(r"\s+", " ", story).strip()
        target_compact = re.sub(r"\s+", " ", target_moment).strip()
        if not story_compact.endswith(target_compact):
            return "story_continues_after_target_moment"
        template_issue = _find_template_issue(story, q_dict)
        if template_issue:
            return template_issue

    if dataset_name == "ToMBench":
        answer = q_dict.get("answer") or {}
        correct = [str(x).strip() for x in (answer.get("correct_answers") or []) if str(x).strip()]
        wrong = [str(x).strip() for x in (answer.get("wrong_answers") or []) if str(x).strip()]
        if len(correct) != 1 or len(wrong) != 3:
            return "tombench_requires_1_correct_and_3_wrong_options"
        if re.fullmatch(r"[A-D]", correct[0], re.IGNORECASE):
            return "tombench_failed_to_parse_option_texts_from_question"
        if re.search(r"(?<![A-Za-z])[A-D][.．]\s+", question):
            return "tombench_question_still_contains_embedded_options"
        unique_opts = {opt.casefold() for opt in (correct + wrong)}
        if len(unique_opts) < 4:
            return "duplicate_answer_options"
        clen = len(correct[0])
        max_w = max(len(w) for w in wrong)
        avg_w = sum(len(w) for w in wrong) / len(wrong)
        is_cjk = bool(re.search(r"[\u4e00-\u9fff]", correct[0]))
        min_len_for_ratio = 10 if is_cjk else 24
        if clen >= min_len_for_ratio and max_w > 0 and clen > 1.45 * max_w and clen > 1.65 * avg_w:
            return "correct_option_significantly_longer_than_distractors_match_option_lengths"
        if _INDETERMINACY_META_RE.search(correct[0]) and avg_w > 0 and clen > 1.20 * avg_w:
            return "indeterminacy_meta_option_longer_than_distractors"
        grounding_issue = _find_option_grounding_issue(story, correct, wrong)
        if grounding_issue:
            return grounding_issue
        structural_leak = _find_option_structural_leak(correct, wrong)
        if structural_leak:
            return structural_leak

    return None


# ============ 数据合成Schema ============

# ---- ToMBench ----
class ToMBenchQuestionFlat(BaseModel):
    reasoning_trap_plan: str = Field(
        default="",
        description="Pre-drafting plan (stripped before storage): state (1) target agent's partial observations vs hidden updates, "
                    "(2) plausible misleading cue, (3) what flawed reasoning path each distractor exploits. Do NOT copy into story/question.",
    )
    target_moment: str = Field(
        description="Exact final sentence or utterance targeted by the question. The story must end with this text; do not include any subsequent action.",
    )
    story: str = Field(description="Complete story text as a plain string")
    question: str = Field(
        description="Question text with all 4 answer options embedded inline, "
                    "e.g. 'What does X think? A. ... B. ... C. ... D. ...'"
    )
    correct_letter: str = Field(
        description="The single correct answer letter: A, B, C, or D"
    )
    meta_ability: str = Field(
        description="The theory-of-mind ability tested, e.g. 'theory_of_mind', "
                    "'false_belief', 'perspective_taking'"
    )
    meta_id: str = Field(
        description="Unique identifier for this question, e.g. 'synthetic_0001'"
    )
    options: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional structured A/B/C/D options dict",
    )
    meta_difficulty: str = Field(
        default="",
        description="Target difficulty label: easy, medium, or hard",
    )

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        q_text = self.question
        letter = self.correct_letter.strip().upper()
        if isinstance(self.options, dict) and {k.upper() for k in self.options} >= {"A", "B", "C", "D"}:
            norm_opts = {k.upper(): _strip_mcq_footer(v) for k, v in self.options.items()}
            correct_text = norm_opts.get(letter, letter)
            wrong_texts = [norm_opts[k] for k in ("A", "B", "C", "D") if k != letter]
            clean_question = _strip_embedded_options(q_text)
        else:
            correct_text, wrong_texts = _parse_options_from_question(q_text, letter)
            clean_question = _strip_embedded_options(q_text)
        stored = {
            "story": self.story,
            "question": clean_question,
            "answer": {
                "correct_answers": [correct_text],
                "wrong_answers": wrong_texts,
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": [self.meta_ability],
            },
        }
        if self.meta_difficulty.strip():
            stored["meta"]["difficulty"] = self.meta_difficulty.strip()
        if self.target_moment.strip():
            stored["_target_moment"] = self.target_moment.strip()
        return stored

class ToMBenchSynthesisOutput(BaseModel):
    questions: List[ToMBenchQuestionFlat]


# ---- SocialIQA ----
class SocialIQAQuestionFlat(BaseModel):
    story_full: str = Field(description="Full story text describing a social situation")
    question: str = Field(description="Question about what someone wants, needs, or reacts to")
    correct_answer: str = Field(description="The correct answer text (full sentence)")
    wrong_answer_1: str = Field(description="First plausible but incorrect answer text")
    wrong_answer_2: str = Field(description="Second plausible but incorrect answer text")
    meta_id: str = Field(description="Unique identifier, e.g. 'synthetic_0001'")
    meta_dimension: str = Field(
        description="Social dimension being tested: xWant, xNeed, xReact, xEffect, xAttr, oWant, oReact, oEffect"
    )
    meta_difficulty: str = Field(description="Difficulty level: easy, medium, or hard")

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": [self.wrong_answer_1, self.wrong_answer_2],
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": [self.meta_dimension],
            },
        }

class SocialIQASynthesisOutput(BaseModel):
    questions: List[SocialIQAQuestionFlat]


# ---- BigToM ----
class BigToMQuestionFlat(BaseModel):
    story_full: str = Field(description="Full story text involving an agent's belief state")
    question: str = Field(description="Question about what the agent believes / will do / perceives")
    correct_answer: str = Field(description="The correct answer text")
    wrong_answer: str = Field(description="Exactly 1 wrong answer — what a model outputs when it skips the required belief operation")
    meta_id: str = Field(
        description="Unique identifier, e.g. 'synthetic_001__backward_belief'"
    )
    meta_condition_type: str = Field(
        description=(
            "The BigToM condition type. Must be one of: "
            "forward_belief (what does agent believe will happen), "
            "backward_belief (what does agent believe caused a past event), "
            "forward_action (what will agent do given their belief), "
            "percept_to_belief (what does agent believe based on perception), "
            "true_belief (agent has correct belief), "
            "false_belief (agent has incorrect belief). "
            "Use the condition type that matches the diagnosed error pattern in the report."
        )
    )
    meta_dimension: str = Field(
        description="Belief order: first_order or second_order"
    )

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": [self.wrong_answer],
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": self.meta_condition_type,
                "dimension": [self.meta_dimension],
            },
        }

class BigToMSynthesisOutput(BaseModel):
    questions: List[BigToMQuestionFlat]


# ---- EmoBench ----
class EmoBenchQuestionFlat(BaseModel):
    story_full: str = Field(description="Full story describing an emotional situation")
    question: str = Field(description="Question about emotion, cause, reaction, or action")
    correct_answer: str = Field(
        description="Correct emotion answer (exact text that will appear as an option)"
    )
    wrong_answer_1: str = Field(description="First wrong emotion option text")
    wrong_answer_2: str = Field(description="Second wrong emotion option text")
    wrong_answer_3: str = Field(
        default="",
        description="Third wrong option (leave empty string if not needed)"
    )
    wrong_answer_4: str = Field(
        default="",
        description="Fourth wrong option (leave empty string if not needed)"
    )
    wrong_answer_5: str = Field(
        default="",
        description="Fifth wrong option (leave empty string if not needed)"
    )
    meta_id: str = Field(description="Unique identifier, e.g. 'synthetic_0001'")
    meta_question_subtype: str = Field(
        description="Question subtype: cause, reaction, or desire"
    )
    meta_coarse_category: str = Field(
        description="Broad emotion category, e.g. 'joy', 'sadness', 'anger'"
    )
    meta_finegrained_category: str = Field(
        description="Specific emotion name, e.g. 'excitement', 'grief', 'frustration'"
    )
    meta_dimension: str = Field(
        description="Measured dimension, e.g. 'emotion_recognition', 'emotion_cause', or 'Action'"
    )
    meta_subset: str = Field(default="emotional_understanding", description="EmoBench subset: emotional_understanding or emotional_application")
    meta_language: str = Field(default="en", description="Language code: en or zh")
    meta_choice_texts: List[str] = Field(default_factory=list, description="All answer choices, including the correct choice")
    meta_paired_group_id: str = Field(default="", description="Shared EU pair identifier; empty for standalone EA questions")

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        wrong_answers = [self.wrong_answer_1, self.wrong_answer_2]
        for w in [self.wrong_answer_3, self.wrong_answer_4, self.wrong_answer_5]:
            if w.strip():
                wrong_answers.append(w)
        choice_texts = [str(x).strip() for x in self.meta_choice_texts if str(x).strip()]
        if not choice_texts:
            choice_texts = [self.correct_answer] + wrong_answers
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": wrong_answers,
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "subset": self.meta_subset,
                "language": self.meta_language,
                "question_subtype": self.meta_question_subtype,
                "coarse_category": self.meta_coarse_category,
                "finegrained_category": self.meta_finegrained_category,
                "dimension": [self.meta_dimension],
                "choice_texts": choice_texts,
                "paired_group_id": self.meta_paired_group_id or self.meta_id,
            },
        }

class EmoBenchSynthesisOutput(BaseModel):
    questions: List[EmoBenchQuestionFlat]


class EmoBenchGroupedQuestionFlat(BaseModel):
    """One EU generation containing the emotion and cause subquestions."""

    story_full: str
    emotion_question: str
    emotion_correct_answer: str
    emotion_wrong_answers: List[str] = Field(min_length=2, max_length=5)
    cause_question: str
    cause_correct_answer: str
    cause_wrong_answers: List[str] = Field(min_length=2, max_length=5)
    meta_id: str
    meta_coarse_category: str = ""
    meta_finegrained_category: str = ""
    meta_dimension: str = "emotion_recognition"
    meta_language: str = "en"
    meta_paired_group_id: str = ""

    def to_storage_rows(self) -> List[dict]:
        group_id = self.meta_paired_group_id or self.meta_id
        common = {
            "subset": "emotional_understanding",
            "language": self.meta_language,
            "coarse_category": self.meta_coarse_category,
            "finegrained_category": self.meta_finegrained_category,
            "paired_group_id": group_id,
        }
        emotion = {
            "story": self.story_full,
            "question": self.emotion_question,
            "answer": {
                "correct_answers": [self.emotion_correct_answer],
                "wrong_answers": list(self.emotion_wrong_answers),
            },
            "meta": {
                **common,
                "id": f"{group_id}_emotion",
                "question_subtype": "emotion",
                "dimension": ["emotional_understanding", "emotion_recognition", self.meta_coarse_category, self.meta_finegrained_category],
                "choice_texts": [self.emotion_correct_answer] + list(self.emotion_wrong_answers),
            },
        }
        cause = {
            "story": self.story_full,
            "question": self.cause_question,
            "answer": {
                "correct_answers": [self.cause_correct_answer],
                "wrong_answers": list(self.cause_wrong_answers),
            },
            "meta": {
                **common,
                "id": f"{group_id}_cause",
                "question_subtype": "cause",
                "dimension": ["emotional_understanding", "emotion_cause", self.meta_coarse_category, self.meta_finegrained_category],
                "choice_texts": [self.cause_correct_answer] + list(self.cause_wrong_answers),
            },
        }
        return [emotion, cause]


class EmoBenchGroupedSynthesisOutput(BaseModel):
    questions: List[EmoBenchGroupedQuestionFlat]


# ---- FanToM ----
_FANTOM_BINARY_TYPES = {
    "beliefQAs", "factQA", "answerabilityQAs_binary", "infoAccessibilityQAs_binary"
}
_FANTOM_LIST_TYPES = {"answerabilityQA_list", "infoAccessibilityQA_list"}


class FanToMQuestionFlat(BaseModel):
    story_full: str = Field(description="Full conversation story text with multiple speakers")
    story_summary: str = Field(
        default="",
        description="Brief summary of the story (can be empty string)"
    )
    question: str = Field(
        description="Question about what a character knows or believes"
    )
    # binary types: use correct_answer + wrong_answer
    correct_answer: str = Field(
        default="",
        description="For binary types: single correct answer text. For list types: leave empty."
    )
    wrong_answer: str = Field(
        default="",
        description="For binary types: single wrong answer text. For list types: leave empty."
    )
    # list types: use correct_names + wrong_names
    correct_names: List[str] = Field(
        default_factory=list,
        description="For list types (answerabilityQA_list / infoAccessibilityQA_list): "
                    "list of character names who CAN answer / DO know the info. "
                    "For binary types: leave empty."
    )
    wrong_names: List[str] = Field(
        default_factory=list,
        description="For list types: character names who CANNOT answer / do NOT know the info. "
                    "For binary types: leave empty."
    )
    meta_id: str = Field(
        description="ID in format '{topic}__{question_type}__{index}', "
                    "e.g. 'synthetic_001__infoAccessibilityQA_list__1'. "
                    "question_type must be one of: beliefQAs, factQA, "
                    "answerabilityQAs_binary, infoAccessibilityQAs_binary, "
                    "answerabilityQA_list, infoAccessibilityQA_list"
    )
    meta_question_type: str = Field(
        description="Question type: beliefQAs, factQA, answerabilityQAs_binary, "
                    "infoAccessibilityQAs_binary, answerabilityQA_list, or infoAccessibilityQA_list"
    )

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        if self.meta_question_type in _FANTOM_LIST_TYPES:
            answer = {
                "correct_answers": self.correct_names,
                "wrong_answers": self.wrong_names,
            }
        else:
            answer = {
                "correct_answers": [self.correct_answer] if self.correct_answer else [],
                "wrong_answers": [self.wrong_answer] if self.wrong_answer else [],
            }
        return {
            "story": self.story_full,
            "question": self.question,
            "answer": answer,
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": [self.meta_question_type],
                "question_type": self.meta_question_type,
            },
        }

class FanToMSynthesisOutput(BaseModel):
    questions: List[FanToMQuestionFlat]


# ---- HiToM ----
class HiToMQuestionFlat(BaseModel):
    story_full: str = Field(
        description="Full story involving agents with higher-order nested beliefs"
    )
    question: str = Field(
        description="Higher-order belief question (15-choice, A through O)"
    )
    correct_answer: str = Field(description="The single correct answer text")
    wrong_1: str = Field(description="Wrong option 1 (of 14 wrong options)")
    wrong_2: str = Field(description="Wrong option 2")
    wrong_3: str = Field(description="Wrong option 3")
    wrong_4: str = Field(description="Wrong option 4")
    wrong_5: str = Field(description="Wrong option 5")
    wrong_6: str = Field(description="Wrong option 6")
    wrong_7: str = Field(description="Wrong option 7")
    wrong_8: str = Field(description="Wrong option 8")
    wrong_9: str = Field(description="Wrong option 9")
    wrong_10: str = Field(description="Wrong option 10")
    wrong_11: str = Field(description="Wrong option 11")
    wrong_12: str = Field(description="Wrong option 12")
    wrong_13: str = Field(description="Wrong option 13")
    wrong_14: str = Field(description="Wrong option 14 (of 14 wrong options)")
    meta_id: str = Field(description="Unique identifier, e.g. 'synthetic_0001'")
    meta_order: int = Field(
        description="ToM recursion depth (belief order): 1, 2, 3, or 4"
    )

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        def _clean(text: str) -> str:
            # 去除括号注释，如 "shelf (what X believes)" -> "shelf"
            return re.sub(r'\s*\(.*?\)', '', text).strip()

        correct = _clean(self.correct_answer)
        raw_wrong = [
            self.wrong_1, self.wrong_2, self.wrong_3, self.wrong_4,
            self.wrong_5, self.wrong_6, self.wrong_7, self.wrong_8,
            self.wrong_9, self.wrong_10, self.wrong_11, self.wrong_12,
            self.wrong_13, self.wrong_14,
        ]
        seen: set = set()
        deduped = []
        for w in raw_wrong:
            cleaned = _clean(w)
            if cleaned and cleaned != correct and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)

        return {
            "story": self.story_full,
            "question": self.question,
            "answer": {
                "correct_answers": [correct],
                "wrong_answers": deduped,
            },
            "meta": {
                "id": self.meta_id,
                "condition_type": "",
                "dimension": ["belief"],
            },
        }

class HiToMSynthesisOutput(BaseModel):
    questions: List[HiToMQuestionFlat]


# ---- SoMBench（中文，4 题型异构，按 qtype 选 schema）----
_SM_VERDICTS = ["是", "否", "无法确定"]


def _sm_meta(meta_dim: str, perspective: str, domain: str,
             length_mode: str, qtype: str, q4_reference: Optional[List[str]] = None) -> dict:
    """构造 SoMBench 合成样本的 meta：dim1/dim2 由 dim 拆出，variant 标 synthetic。
    与 eval parquet 的 meta 对齐（id/lang/history 由 stage3 后处理写入）。"""
    parts = str(meta_dim).split(".")
    dim1 = parts[0] if parts else ""
    dim2 = ".".join(parts[:2]) if len(parts) >= 2 else dim1
    return {
        "dim": str(meta_dim), "dim1": dim1, "dim2": dim2, "qtype": qtype,
        "perspective": perspective or "third_person",
        "domain": domain or "", "length_mode": length_mode or "long",
        "variant": "synthetic", "lang": "zh",
        "q4_reference": list(q4_reference or []),
    }


class SoMBenchQ1Flat(BaseModel):
    story: str = Field(description="以①②③编号的完整中文故事")
    question: str = Field(description="中文问题")
    correct_answer: str = Field(description="正确答案中文句")
    wrong_answer_1: str = Field(description="干扰项1")
    wrong_answer_2: str = Field(description="干扰项2")
    wrong_answer_3: str = Field(description="干扰项3")
    meta_dim: str = Field(description="三级维度，如 1.1.1")
    meta_perspective: str = Field(default="third_person")
    meta_domain: str = Field(default="")
    meta_length_mode: str = Field(default="long")

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        return {
            "story": self.story,
            "question": self.question,
            "answer": {
                "correct_answers": [self.correct_answer],
                "wrong_answers": [self.wrong_answer_1, self.wrong_answer_2, self.wrong_answer_3],
            },
            "meta": _sm_meta(self.meta_dim, self.meta_perspective, self.meta_domain,
                             self.meta_length_mode, "Q1"),
        }


class SoMBenchQ1Output(BaseModel):
    questions: List[SoMBenchQ1Flat]


class SoMBenchQ2Flat(BaseModel):
    story: str = Field(description="以①②③编号的完整中文故事")
    question: str = Field(description="中文多选问题")
    correct_answers: List[str] = Field(description="正确答案（≥2 个）")
    wrong_answers: List[str] = Field(description="干扰项")
    meta_dim: str = Field(description="三级维度，如 1.1.1")
    meta_perspective: str = Field(default="third_person")
    meta_domain: str = Field(default="")
    meta_length_mode: str = Field(default="long")

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        correct = list(self.correct_answers or [])
        if len(correct) < 2:
            logger.warning(
                f"SoMBench Q2 正确项 <2（{len(correct)}），会被反推成 mcq_single: dim={self.meta_dim}"
            )
        return {
            "story": self.story,
            "question": self.question,
            "answer": {
                "correct_answers": correct,
                "wrong_answers": list(self.wrong_answers or []),
            },
            "meta": _sm_meta(self.meta_dim, self.meta_perspective, self.meta_domain,
                             self.meta_length_mode, "Q2"),
        }


class SoMBenchQ2Output(BaseModel):
    questions: List[SoMBenchQ2Flat]


class SoMBenchQ3Flat(BaseModel):
    story: str = Field(description="以①②③编号的完整中文故事")
    question: str = Field(description="判断题题面（含待判断陈述）")
    correct_verdict: str = Field(description="是 / 否 / 无法确定 之一")
    meta_dim: str = Field(description="三级维度，如 1.1.1")
    meta_perspective: str = Field(default="third_person")
    meta_domain: str = Field(default="")
    meta_length_mode: str = Field(default="long")

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        cv = str(self.correct_verdict).strip()
        if cv not in _SM_VERDICTS:
            cv = "无法确定"
        wrong = [v for v in _SM_VERDICTS if v != cv]  # 另两个判定作干扰
        return {
            "story": self.story,
            "question": self.question,
            "answer": {"correct_answers": [cv], "wrong_answers": wrong},
            "meta": _sm_meta(self.meta_dim, self.meta_perspective, self.meta_domain,
                             self.meta_length_mode, "Q3"),
        }


class SoMBenchQ3Output(BaseModel):
    questions: List[SoMBenchQ3Flat]


class SoMBenchQ4Flat(BaseModel):
    story: str = Field(description="以①②③编号的完整中文故事")
    question: str = Field(description="开放分析问题")
    reference_points: List[str] = Field(description="参考答案要点（3-8 条）")
    meta_dim: str = Field(description="三级维度，如 1.1.1（必须命中 q4 rubric 键）")
    meta_perspective: str = Field(default="third_person")
    meta_domain: str = Field(default="")
    meta_length_mode: str = Field(default="long")

    def to_storage_dict(self, synthesis_diagnosis=None) -> dict:
        pts = [str(p).strip() for p in (self.reference_points or []) if str(p).strip()]
        return {
            "story": self.story,
            "question": self.question,
            # 1 正 0 误 -> 下游 prompt_type() 判为 open
            "answer": {"correct_answers": ["\n".join(pts)], "wrong_answers": []},
            "meta": _sm_meta(self.meta_dim, self.meta_perspective, self.meta_domain,
                             self.meta_length_mode, "Q4", q4_reference=pts),
        }


class SoMBenchQ4Output(BaseModel):
    questions: List[SoMBenchQ4Flat]


SOMBENCH_SCHEMA_BY_QTYPE: Dict[str, Any] = {
    "Q1": SoMBenchQ1Output,
    "Q2": SoMBenchQ2Output,
    "Q3": SoMBenchQ3Output,
    "Q4": SoMBenchQ4Output,
}


SYNTHESIS_SCHEMA_REGISTRY: Dict[str, Any] = {
    "ToMBench": ToMBenchSynthesisOutput,
    "SocialIQA": SocialIQASynthesisOutput,
    "BigToM": BigToMSynthesisOutput,
    "EmoBench": EmoBenchSynthesisOutput,
    "EmoBenchGrouped": EmoBenchGroupedSynthesisOutput,
    "FanToM": FanToMSynthesisOutput,
    "HiToM": HiToMSynthesisOutput,
}

# 通用 fallback schema
class _FallbackSynthesisOutput(BaseModel):
    questions: List[Dict[str, Any]] = Field(description="Generated questions")


# ============ 从维度诊断报告合成新数据 ============

def synthesize_from_reports(
    reports: List[Dict[str, Any]],
    dataset_name: str,
    synthesis_llm_config: Dict[str, Any],
    samples_per_report: int = 2,
    max_retries: int = 3,
    existing_pairs: Optional[set] = None,
    batch_size: int = 10,
    output_file: Optional[Path] = None,
    target_difficulty: str = "",
    difficulty_weights: Optional[Dict[str, float]] = None,
    difficulty_gate: Optional[Any] = None,
    gate_difficulties: Tuple[str, ...] = ("hard",),
    gate_log_file: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """从维度级诊断报告合成新样本，每份报告生成 samples_per_report 条题目。

    跳过 pass@k 验证，直接合成后交守门员过滤。

    Args:
        reports: DimensionDiagnosisReport dict 列表
        dataset_name: 数据集名称
        synthesis_llm_config: 合成模型配置
        samples_per_report: 每份报告生成的样本数
        max_retries: 生成失败时最多重试次数
        batch_size: 每批并发请求数，每批完成后立即写盘
        output_file: 若提供，每批完成后追加写入该文件
        difficulty_gate: 若提供（DifficultyGate），gate_difficulties 内的题先跑 solver pass@k，
            solve_rate 超阈值的题打回重写；最后一轮仍过易则丢弃
        gate_log_file: 若提供，追加写入每次闸门判定（含被拒题目）

    Returns:
        (synthesized_data, stats) 元组
    """
    if not reports:
        logger.warning("No reports provided for synthesis")
        return [], {"total_reports": 0, "total_target": 0, "succeeded": 0, "failed": 0}

    output_schema = SYNTHESIS_SCHEMA_REGISTRY.get(dataset_name, _FallbackSynthesisOutput)

    def _schema_for_report(r_idx: int):
        """SoMBench 按 report 的 qtype 选 schema；其余数据集恒用 output_schema。
        用 _meta.dimension（stage2 程序化写入、权威），避免读 LLM 回显的顶层 dimension。"""
        if dataset_name == "EmoBench":
            key = str((reports[r_idx].get("_meta") or {}).get("dimension") or reports[r_idx].get("dimension", ""))
            if key.startswith("EU__"):
                return EmoBenchGroupedSynthesisOutput
            return EmoBenchSynthesisOutput
        if dataset_name != "SoMBench":
            return output_schema
        rep = reports[r_idx]
        key = str((rep.get("_meta") or {}).get("dimension") or rep.get("dimension", ""))
        _dim, qtype = _sombench_parse_dim(key)
        return SOMBENCH_SCHEMA_BY_QTYPE.get(qtype, SoMBenchQ1Output)

    total_target = len(reports) * samples_per_report
    logger.info(
        f"synthesize_from_reports: {len(reports)} reports × {samples_per_report} = "
        f"{total_target} target questions, schema={output_schema.__name__}"
    )

    # pending_pairs: list of (report_idx, sample_idx)，跳过已生成的 pair。
    # 按 sample_idx 外层、report_idx 内层排列：中断或部分运行时先覆盖所有报告（含 zh 报告），
    # 而不是把同一份报告的全部变体集中生成，避免候选集被少数报告的模板主导。
    _skip = existing_pairs or set()
    pending_pairs: List[Tuple[int, int]] = [
        (r_idx, s_idx)
        for s_idx in range(samples_per_report)
        for r_idx in range(len(reports))
        if (r_idx, s_idx) not in _skip
    ]
    results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    last_failed: Dict[Tuple[int, int], Dict[str, Any]] = {}
    generation_attempts: Dict[Tuple[int, int], int] = {p: 0 for p in pending_pairs}
    gate_rejected = 0

    for round_idx in range(max_retries):
        if not pending_pairs:
            break

        logger.info(f"  Round {round_idx + 1}/{max_retries}: generating {len(pending_pairs)} questions...")

        # 合成语言由 Stage 2 报告的 _meta.lang 决定（维度分组已按语言拆分）。
        gen_prompts = [
            build_stage2_generation_from_report_prompt(
                report=reports[r_idx],
                dataset_name=dataset_name,
                target_count=1,
                previous_question=last_failed.get((r_idx, s_idx)) if round_idx > 0 else None,
                lang=str((reports[r_idx].get("_meta") or {}).get("lang") or "en"),
                variant_index=s_idx,
                target_difficulty=target_difficulty,
                difficulty_weights=difficulty_weights,
                schedule_index=r_idx * samples_per_report + s_idx,
            ) + f"\n\nThis is variant {s_idx + 1} for the same diagnosis. Use a distinct scenario, names, and surface structure from the other variants."
            for (r_idx, s_idx) in pending_pairs
        ]

        success_this_round = 0
        client = runner.create_llm_client(synthesis_llm_config)
        for batch_start in range(0, len(gen_prompts), batch_size):
            batch_prompts = gen_prompts[batch_start: batch_start + batch_size]
            batch_pairs = pending_pairs[batch_start: batch_start + batch_size]

            # SoMBench 一个 batch 可能混不同 qtype（schema），按 schema 子分组各调一次，
            # 再按原顺序拼回，保持与 batch_pairs 对齐。其余数据集走单一 schema。
            if dataset_name in {"SoMBench", "EmoBench"}:
                groups: Dict[Any, List[int]] = {}
                for i, (r_idx, _s) in enumerate(batch_pairs):
                    groups.setdefault(_schema_for_report(r_idx), []).append(i)
                gen_results_raw = [None] * len(batch_pairs)
                for schema, idxs in groups.items():
                    sub = client.batch_generate_structure(
                        [batch_prompts[i] for i in idxs], schema,
                        mode="create",
                        desc=f"Synthesizing {dataset_name}[{schema.__name__}]"
                    )
                    for i, res in zip(idxs, sub):
                        gen_results_raw[i] = res
            else:
                gen_results_raw = client.batch_generate_structure(
                    batch_prompts, output_schema,
                    mode="create",
                    desc=f"Synthesizing {dataset_name}"
                )

            batch_new: List[Dict[str, Any]] = []
            # 通过结构校验、等待难度闸门判定的 pair → rows
            gate_candidates: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
            for pair, resp in zip(batch_pairs, gen_results_raw):
                r_idx, s_idx = pair
                generation_attempts[pair] = generation_attempts.get(pair, 0) + 1
                r = resp.content if resp else None
                if r is not None and hasattr(r, 'questions') and r.questions:
                    q = r.questions[0]
                    if dataset_name == "HiToM":
                        wrong_count = len({
                            str(getattr(q, f"wrong_{i}", "")).strip()
                            for i in range(1, 15)
                            if str(getattr(q, f"wrong_{i}", "")).strip()
                        })
                        if wrong_count < 14:
                            last_failed[pair] = {
                                "_failure_reason": f"HiToM requires 14 unique distractors; got {wrong_count}",
                            }
                            continue
                    if hasattr(q, "to_storage_rows"):
                        q_rows = q.to_storage_rows()
                    elif hasattr(q, "to_storage_dict"):
                        q_rows = [q.to_storage_dict()]
                    else:
                        q_rows = [q if isinstance(q, dict) else dict(q)]
                    pair_difficulty = resolve_variant_difficulty(
                        report=reports[r_idx],
                        variant_index=s_idx,
                        target_difficulty=target_difficulty,
                        difficulty_weights=difficulty_weights,
                        schedule_index=r_idx * samples_per_report + s_idx,
                    )
                    report_dim = str(
                        (reports[r_idx].get("_meta") or {}).get("dimension")
                        or reports[r_idx].get("dimension")
                        or ""
                    ).strip()
                    primary_op = str(reports[r_idx].get("primary_cognitive_operation") or "").strip()
                    validation_error = None
                    for row_idx, q_dict in enumerate(q_rows):
                        err = _validate_synthesized_question(q_dict, dataset_name, difficulty=pair_difficulty)
                        if err:
                            validation_error = err
                            break
                        q_dict.setdefault("meta", {})
                        meta = q_dict.get("meta", {})
                        if isinstance(meta, dict):
                            subtype = str(meta.get("question_subtype") or row_idx)
                            meta["id"] = f"synthetic_{dataset_name.lower()}_{r_idx:03d}_{s_idx}_{subtype}"
                            meta.setdefault("history", [])
                            meta["lang"] = str((reports[r_idx].get("_meta") or {}).get("lang") or "en")
                            meta["difficulty"] = pair_difficulty
                            if report_dim:
                                meta["report_dimension"] = report_dim
                                if meta.get("dimension") == ["theory_of_mind"] or not meta.get("dimension"):
                                    meta["dimension"] = [report_dim]
                            if primary_op:
                                meta["primary_cognitive_operation"] = primary_op
                    for q_dict in q_rows:
                        q_dict.pop("_target_moment", None)
                    if validation_error:
                        failed_payload = {"_failure_reason": validation_error}
                        if q_rows:
                            failed_payload.update({
                                "story": q_rows[0].get("story"),
                                "question": q_rows[0].get("question"),
                                "answer": q_rows[0].get("answer"),
                            })
                        last_failed[pair] = failed_payload
                        continue
                    if difficulty_gate is not None and pair_difficulty in gate_difficulties:
                        gate_candidates[pair] = q_rows
                        continue
                    results[pair] = q_rows
                    batch_new.extend(q_rows)
                    success_this_round += 1
                else:
                    last_failed[pair] = {"_failure_reason": "structured_output_failed"}

            if gate_candidates:
                gate_rows = [row for rows in gate_candidates.values() for row in rows]
                solve_stats = difficulty_gate.evaluate(gate_rows, dataset_name)
                gate_log: List[Dict[str, Any]] = []
                for pair, q_rows in gate_candidates.items():
                    row_stats = [solve_stats.get(str((r.get("meta") or {}).get("id"))) for r in q_rows]
                    passed = all(difficulty_gate.passes(st) for st in row_stats)
                    reasons = []
                    for st in row_stats:
                        if hasattr(difficulty_gate, "rejection_reason"):
                            r = difficulty_gate.rejection_reason(st)
                            if r:
                                reasons.append(r)
                        elif not difficulty_gate.passes(st):
                            reasons.append("too_easy")
                    gate_reason = (
                        "too_hard_or_unanswerable"
                        if "too_hard_or_unanswerable" in reasons
                        else ("too_easy" if "too_easy" in reasons else ("missing_eval_stats" if reasons else None))
                    )
                    gate_log.append({
                        "pair": list(pair),
                        "round": round_idx + 1,
                        "solver": difficulty_gate.model_name,
                        "stats": row_stats,
                        "passed": passed,
                        "reason": gate_reason,
                    })
                    if passed:
                        results[pair] = q_rows
                        batch_new.extend(q_rows)
                        success_this_round += 1
                        continue
                    gate_rejected += 1
                    valid_stats = [st for st in row_stats if st]
                    if valid_stats:
                        picked = (
                            min(valid_stats, key=lambda st: st["solve_rate"])
                            if gate_reason == "too_hard_or_unanswerable"
                            else max(valid_stats, key=lambda st: st["solve_rate"])
                        )
                        solver_result = f"{picked['correct']}/{picked['total']} correct by the target model"
                    else:
                        solver_result = "no readable evaluation stats returned"
                    last_failed[pair] = {
                        "story": q_rows[0].get("story"),
                        "question": q_rows[0].get("question"),
                        "answer": q_rows[0].get("answer"),
                        "gate_reason": gate_reason or "too_easy",
                        "solver_result": solver_result,
                    }
                    if gate_log_file is not None:
                        gate_log[-1]["rejected_rows"] = q_rows
                if gate_log_file is not None:
                    gate_log_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(gate_log_file, "a", encoding="utf-8") as _f:
                        for item in gate_log:
                            _f.write(json.dumps(item, ensure_ascii=False) + "\n")

            if output_file is not None and batch_new:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "a", encoding="utf-8") as _f:
                    for item in batch_new:
                        clean = {k: v for k, v in item.items() if not k.startswith("_")}
                        _f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        logger.info(f"  Round {round_idx + 1}: {success_this_round}/{len(pending_pairs)} generated")
        pending_pairs = [p for p in pending_pairs if p not in results]

    final_results = [row for rows in results.values() for row in rows]
    stats = {
        "total_reports": len(reports),
        "total_target": total_target,
        "succeeded": len(final_results),
        "failed": total_target - len(final_results),
        "total_generation_calls": sum(generation_attempts.values()),
        "difficulty_gate_rejected": gate_rejected,
    }

    logger.info(f"synthesize_from_reports complete: {len(final_results)}/{total_target} questions produced")
    return final_results, stats


# ============ 保存合成数据 ============

def save_synthesized_data(
    synthesized: List[Dict[str, Any]],
    dataset_name: str,
    output_path: str = "data_output/synth_raw",
    model_name: str = "unknown_model",
) -> Path:
    """保存合成数据为 JSONL（守门员过滤前的 raw 候选）

    Args:
        synthesized: 合成数据
        dataset_name: 数据集名称
        output_path: 输出目录
        model_name: 合成模型名称

    Returns:
        输出文件路径
    """
    output_dir = Path(output_path) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    output_file = output_dir / f"candidates_{safe_model}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for item in synthesized:
            # 清理内部追踪字段
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(synthesized)} synthesized questions (raw) to {output_file}")
    return output_file


# ============ 阶段三：从诊断报告合成数据 ============

def run_stage3_synthesis(
    reports_dir: str,
    dataset_name: str,
    synthesis_llm_config: Dict[str, Any],
    samples_per_report: int = 2,
    max_retries: int = 3,
    output_dir: str = "data_output/synth_raw",
    model_name: str = "unknown_model",
    batch_size: int = 10,
    target_difficulty: str = "",
    difficulty_weights: Optional[Dict[str, float]] = None,
    difficulty_gate: Optional[Any] = None,
    gate_difficulties: Tuple[str, ...] = ("hard",),
) -> Optional[Path]:
    """从维度诊断报告（dimension_reports.jsonl）生成并保存原始候选数据

    Args:
        reports_dir:           diagnosis_reports 输出目录（含 dimension_reports.jsonl）
        dataset_name:          数据集名称
        synthesis_llm_config:  合成/生成模型配置
        samples_per_report:    每份报告生成的新样本数
        max_retries:           生成失败最多重试次数
        output_dir:            输出根目录（synth_raw，守门员过滤前）
        model_name:            合成模型名称

    Returns:
        保存候选数据的文件 Path；若无报告则返回 None
    """
    reports_dir_ = Path(reports_dir)
    reports_path = reports_dir_ / "dimension_reports.jsonl"

    if not reports_path.exists():
        raise FileNotFoundError(f"dimension_reports.jsonl not found: {reports_path}")

    logger.info(f"Stage 3 Synthesis: {dataset_name}")
    logger.info(f"  Reading from: {reports_path}")

    reports: List[Dict[str, Any]] = []
    with open(reports_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reports.append(json.loads(line))

    if not reports:
        logger.warning("  No reports found, skipping synthesis")
        return None

    logger.info(f"  Loaded {len(reports)} dimension reports")

    # 读取已有候选文件，提取已完成的 (r_idx, s_idx) 对
    safe_model = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    output_file = Path(output_dir) / dataset_name / f"candidates_{safe_model}.jsonl"
    existing_pairs: set = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    mid = item.get("meta", {}).get("id", "")
                    pair = _parse_existing_pair_id(mid)
                    if pair is not None:
                        existing_pairs.add(pair)
                except Exception:
                    pass
        if existing_pairs:
            logger.info(f"  ⏭️  已有 {len(existing_pairs)} 条候选，只生成缺失的")

    total_target = len(reports) * samples_per_report
    if len(existing_pairs) >= total_target:
        logger.info(f"  ⏭️  [{dataset_name}] 所有 {total_target} 条已生成，跳过合成")
        return output_file

    synthesized, stats = synthesize_from_reports(
        reports=reports,
        dataset_name=dataset_name,
        synthesis_llm_config=synthesis_llm_config,
        samples_per_report=samples_per_report,
        max_retries=max_retries,
        existing_pairs=existing_pairs,
        batch_size=batch_size,
        output_file=output_file,
        target_difficulty=target_difficulty,
        difficulty_weights=difficulty_weights,
        difficulty_gate=difficulty_gate,
        gate_difficulties=gate_difficulties,
        gate_log_file=(output_file.parent / f"difficulty_gate_{safe_model}.jsonl") if difficulty_gate else None,
    )
    if difficulty_gate is not None:
        logger.info(f"  难度闸门（{difficulty_gate.model_name} pass@{difficulty_gate.k}）拒绝 {stats.get('difficulty_gate_rejected', 0)} 次")

    if not synthesized:
        logger.warning("  No new synthesized questions produced")
        return output_file if output_file.exists() else None

    total_written = len(existing_pairs) + len(synthesized)
    logger.info(f"Stage 3 done: {len(synthesized)} new + {len(existing_pairs)} existing = {total_written}/{total_target} saved to {output_file}")
    return output_file

