"""V3 Phase E：三种修复 prompt 模板（unanswerable / easy / shortcut）。

输出契约（强模型 StructureClient）：
  - story / question / correct_answers / wrong_answers 完整新样本
  - 保持原数据集风格（角色名/题型不变）

复用：
  - feedback.prompts.DATASET_SKILL_REGISTRY 提供数据集背景
"""

from __future__ import annotations

from typing import Dict, List

from feedback.prompts import DATASET_SKILL_REGISTRY


_SHARED_OUTPUT_SCHEMA = """\
Return ONLY valid JSON (no markdown fences) matching:
{{
  "story": "<full new story, self-contained>",
  "question": "<single question, exactly one '?' if interrogative>",
  "correct_answers": ["<correct option text>", ...],   # 1 element for single-choice; >1 for multi-choice
  "wrong_answers": ["<distractor text>", ...]          # keep the same count as the original sample
}}

Constraints:
  - {language_constraint}
  - Preserve the dataset's question style (length, agent names, register).
  - Keep correct_answers / wrong_answers as plain text (NOT letters).
  - Keep wrong_answers count identical to the original sample.
  - Do not include any commentary outside the JSON."""

# 修复内容的语言必须跟随原样本语言；JSON key 与结构约束保持英文。
_LANGUAGE_CONSTRAINTS = {
    "en": "Write all generated text (story, question, answers) in English, matching the original sample.",
    "zh": "所有生成的文本（story、question、answers 的内容）必须使用简体中文，与原样本语言一致。",
}


REPAIR_UNANSWERABLE_PROMPT = """\
You are repairing a Theory-of-Mind (ToM) question that an auditor flagged as NOT logically answerable.

Dataset: {dataset}
Dataset focus: {dataset_focus}

Failure label: {failure_label}
Auditor reason: {failure_reason}

Original sample (broken):
  Story: {story}
  Question: {question}
  Marked correct: {correct_answers}
  Marked wrong:   {wrong_answers}

Repair goal:
  - Fix the issue indicated by the failure label.
  - The new story MUST uniquely support exactly the marked correct answer(s).
  - All wrong answers must be unambiguously wrong given the new story (NEVER leave any distractor factually true or practically superior under the story).
  - Resolve any contradictions, ambiguity, missing info, physical obstacles (e.g. if transport/tools are missing, ensure the correct answer addresses them), or label errors.
  - If using approximate quantifiers ('nearly half', 'almost none', '将近一半', '几乎没有'), state total-count scope explicitly across rooms/categories and keep options >= 4 units away from the midpoint so numeric choices do not overlap within ±1/±2 rounding margins.
  - Do NOT leave an overheard confession (e.g. 'I rewrote one of hers') unaccounted for when testing verified vs. tampered work; keep any gap between a partial message preview and speech tight (1-2 minutes) and align the addressee spatially with the requested object.
  - Keep the ToM ability tested by the original question.

""" + _SHARED_OUTPUT_SCHEMA


REPAIR_EASY_PROMPT = """\
You are repairing a Theory-of-Mind (ToM) question that a small model answered correctly EVERY time \
(pass@k = k). It is too easy.

Dataset: {dataset}
Dataset focus: {dataset_focus}

Original sample (too easy):
  Story: {story}
  Question: {question}
  Marked correct: {correct_answers}
  Marked wrong:   {wrong_answers}

Repair goal:
  - Increase CoM/ToM reasoning depth against strong CoT models without 1-to-1 structural re-skinning: add a new reasoning hop \
(e.g. 2nd-order belief or two sequential updates with a partial rollback), a counter-directional partial observation/cue \
that tempts a wrong option (NEVER let the partial cue reinforce the correct answer), or a competing preference/norm/indirect-causal conflict.
  - No Distractor-Killing Patches & No Front-Loaded Parables: NEVER insert convenience sentences that render a strong distractor \
physically/factually impossible, and NEVER open with an explanatory prior-event parable that spells out the ending's causal mechanism; \
keep distractors physically live and failing only by epistemic access or timing.
  - Convey who witnessed what implicitly via physical timing, departures, or truncated messages — NEVER write \
explicit boolean signposts such as 'completely alone', 'without telling anyone', 'without mentioning', or 'having spoken to neither', \
and NEVER use narrator epistemic spoilers ('assumed', 'thought', 'believed', 'mistook', 'misread', 'unbeknownst to', 'did not know/realize', 'in reality', '误以为', '不知情', '实际上').
  - Avoid any repeated gaze/smile template ('catches eye', 'glances toward', 'smiles'), 4-stage glass-wall vs. under-desk/solid-wall occlusion tables, or camera-closeup spoilers on forgotten objects, and include at least one neutral intervening beat before the queried final sentence.
  - NEVER insert early explicit self-disqualification lines ('I only like X', 'Swimming, sorry', handing away one's tool right before asking), explicit calendar-day tie-breakers (e.g. Tuesday vs. Wednesday), or objectively false 'because' clauses in emotion distractors; for two-part persuasion framings, make at least two options match both parts.
  - End the story at the exact queried moment with NO subsequent spoiler action/reply, and keep only one unambiguous answer.
  - You may add an intermediary agent or restructure event order to deepen the reasoning dilemma while preserving the core characters/domain.
  - Distractors must be balanced in length/specificity (0.80x-1.20x), single-claim (no bundled dual-claim clauses), avoid length-biased 'cannot tell / pick at random' meta-options, and correspond to close-but-wrong CoT reasoning paths.

""" + _SHARED_OUTPUT_SCHEMA


REPAIR_SHORTCUT_PROMPT = """\
You are repairing a Theory-of-Mind (ToM) question that contains a SHORTCUT — the model can pick the right \
answer without genuinely reading the story (e.g. the answer is hinted in the question wording, the wrong \
options are obviously incoherent, or removing the story leaves the answer guessable from the options alone).

Dataset: {dataset}
Dataset focus: {dataset_focus}

Original sample (has shortcut):
  Story: {story}
  Question: {question}
  Marked correct: {correct_answers}
  Marked wrong:   {wrong_answers}

Repair goal:
  - Rewrite the story so the question MUST depend on a story-specific fact to be answerable.
  - Make distractors story-coherent: each distractor must be a fact that the story COULD have stated, \
and must remain wrong under the rewritten story.
  - Remove any cue in the question wording that gives the answer away.
  - Without the new story, the question must be unanswerable; with the new story, the answer must be unique.

""" + _SHARED_OUTPUT_SCHEMA


REPAIR_PROMPTS_BY_TYPE: Dict[str, str] = {
    "unanswerable": REPAIR_UNANSWERABLE_PROMPT,
    "easy": REPAIR_EASY_PROMPT,
    "shortcut": REPAIR_SHORTCUT_PROMPT,
}


def get_dataset_focus(dataset: str) -> str:
    return DATASET_SKILL_REGISTRY.get(dataset, "general Theory-of-Mind reasoning")


def build_repair_prompt(
    repair_type: str,
    dataset: str,
    story: str,
    question: str,
    correct_answers: List[str],
    wrong_answers: List[str],
    failure_label: str = "",
    failure_reason: str = "",
    lang: str = "en",
) -> str:
    template = REPAIR_PROMPTS_BY_TYPE.get(repair_type)
    if template is None:
        raise ValueError(f"unknown repair_type: {repair_type}")
    language_constraint = _LANGUAGE_CONSTRAINTS.get(lang, _LANGUAGE_CONSTRAINTS["en"])
    return template.format(
        dataset=dataset,
        dataset_focus=get_dataset_focus(dataset),
        story=story,
        question=question,
        correct_answers=", ".join(correct_answers),
        wrong_answers=", ".join(wrong_answers),
        failure_label=failure_label or "(unspecified)",
        failure_reason=failure_reason or "(unspecified)",
        language_constraint=language_constraint,
    )


__all__ = [
    "REPAIR_PROMPTS_BY_TYPE",
    "REPAIR_UNANSWERABLE_PROMPT",
    "REPAIR_EASY_PROMPT",
    "REPAIR_SHORTCUT_PROMPT",
    "build_repair_prompt",
    "get_dataset_focus",
]
