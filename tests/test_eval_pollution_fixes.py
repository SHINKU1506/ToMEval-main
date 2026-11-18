from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from feedback.stage3_synthesis import _strip_embedded_options
from filter.eval.eval_answerability_full import _build_prompt as build_answerability_prompt
from filter.eval.eval_shortcut import build_no_options_prompt, build_no_story_prompt
from filter.repair.repair_pipeline import RepairedSample, repair_samples
from filter.utils import row_to_sample
from src.evaluation.data import normalize_sample
from src.evaluation.prediction import predict_records
from src.evaluation.prompts import build_prompt, sanitize_mcq_question
from tasks.ToMBench.prompt import build_prompt as build_tombench_prompt


def test_sanitize_mcq_question_strips_embedded_options_without_false_positives():
    dirty_en = (
        "What does Nora think? "
        "A. She wants tea. B. She wants tarts. C. She forgot. D. None.\n"
        "Please select the best answer."
    )
    assert sanitize_mcq_question(dirty_en, True) == "What does Nora think?"

    dirty_zh = (
        "陈屿接下来打算做什么？A. 去面馆 B. 修收音机 C. 修感应灯 D. 去楼道\n请选择正确答案。"
    )
    assert sanitize_mcq_question(dirty_zh, True) == "陈屿接下来打算做什么？"

    # Open QA samples must remain intact even if they contain A. / B. prose
    assert sanitize_mcq_question(dirty_en, False) == dirty_en

    # Lowercase "a." or guarded labels like Room A / Plan A must not be truncated
    prose_with_lowercase_a = "Which plan a. works best compared to Room A. and Plan B.?"
    assert sanitize_mcq_question(prose_with_lowercase_a, True) == prose_with_lowercase_a
    assert _strip_embedded_options(prose_with_lowercase_a) == prose_with_lowercase_a


def test_normalize_sample_and_task_prompts_prevent_double_option_blocks():
    raw_row = {
        "story": "Nora and Felix visited the market.",
        "question": "Where is the tart? A. Stall one B. Stall two C. Bakery D. Home",
        "answer": {
            "correct_answers": ["Stall one"],
            "wrong_answers": ["Stall two", "Bakery", "Home"],
        },
        "meta": {"id": "tom_1", "lang": "en"},
    }
    sample = normalize_sample(raw_row, 0)
    assert sample["question"] == "Where is the tart?"

    # Direct calls bypassing normalize_sample must also strip embedded options
    dirty_sample = {
        "sample_id": "tom_1",
        "story": raw_row["story"],
        "question": raw_row["question"],
        "answer": raw_row["answer"],
        "meta": raw_row["meta"],
    }
    option_map = {"A": "Bakery", "B": "Stall one", "C": "Home", "D": "Stall two"}
    tom_prompt = build_tombench_prompt(dirty_sample, option_map, include_instruction=False)
    generic_prompt = build_prompt(dirty_sample, option_map, include_instruction=False)
    assert tom_prompt.count("Stall one") == 1
    assert generic_prompt.count("Stall one") == 1
    assert "A. Stall one" not in tom_prompt
    assert "A. Stall one" not in generic_prompt


def test_agent_mode_payload_sanitizes_embedded_options():
    captured_payloads = []

    class DummyAgentClient:
        def batch_generate(self, prompts, desc="", payloads=None, system_prompts=None):
            captured_payloads.extend(payloads or [])
            return [SimpleNamespace(content="A", reasoning="", raw=None) for _ in prompts]

    dirty_sample = {
        "sample_id": "tom_agent_1",
        "story": "Story.",
        "question": "Which room? A. Kitchen B. Hall C. Study D. Attic",
        "answer": {
            "correct_answers": ["Kitchen"],
            "wrong_answers": ["Hall", "Study", "Attic"],
        },
        "meta": {"id": "tom_agent_1", "lang": "en"},
    }
    predict_records([dirty_sample], "ToMBench", DummyAgentClient(), repeats=1, agent_mode=True)
    assert len(captured_payloads) == 1
    assert captured_payloads[0]["question"] == "Which room?"


def test_filter_shortcut_and_answerability_do_not_leak_embedded_options():
    row = {
        "story": "Alice hid the key.",
        "question": "Where does Bob look? A. Drawer B. Shelf C. Table D. Bag",
        "answer": {
            "correct_answers": ["Drawer"],
            "wrong_answers": ["Shelf", "Table", "Bag"],
        },
        "meta": {"id": "f_1", "lang": "en"},
    }
    sample = row_to_sample(row)
    assert sample["question"] == "Where does Bob look?"

    # Direct calls with unsanitized sample must still strip options in no_options & no_story
    dirty_sample = {**row, "sample_id": "f_1"}
    no_opts_prompt = build_no_options_prompt(dirty_sample)
    assert "Drawer" not in no_opts_prompt
    assert "Shelf" not in no_opts_prompt

    no_story_prompt = build_no_story_prompt(dirty_sample, {"A": "Table", "B": "Drawer"}, "mcq_single")
    assert "A. Drawer" not in no_story_prompt
    assert "\\boxed{}" in no_story_prompt

    ans_prompt = build_answerability_prompt(row, dataset="ToMBench")
    assert "Question: Where does Bob look?" in ans_prompt
    assert "Question: Where does Bob look? A." not in ans_prompt


def test_repair_samples_sanitizes_input_and_output_questions():
    df = pd.DataFrame([
        {
            "story": "Old story.",
            "question": "Old question? A. Opt1 B. Opt2 C. Opt3 D. Opt4",
            "answer": {"correct_answers": ["Opt1"], "wrong_answers": ["Opt2", "Opt3", "Opt4"]},
            "meta": {"id": "rep_1", "lang": "en"},
        }
    ])
    labels_df = pd.DataFrame([
        {
            "sample_id": "rep_1",
            "label": "bad",
            "repair_type": "unanswerable",
            "answerability_label": "missing_info",
            "answerability_reason": "Missing clue.",
        }
    ])

    captured_prompts = []

    class DummyRepairClient:
        def batch_generate_structure(self, prompts, schema, mode="create", desc=""):
            captured_prompts.extend(prompts)
            repaired = RepairedSample(
                story="Clean story.",
                question="Clean question? A. New1 B. New2 C. New3 D. New4",
                correct_answers=["New1"],
                wrong_answers=["New2", "New3", "New4"],
            )
            return [SimpleNamespace(content=repaired)]

    out_df = repair_samples(df, labels_df, "ToMBench", iter_n=1, repair_client=DummyRepairClient())
    assert "A. Opt1" not in captured_prompts[0]
    assert out_df.iloc[0]["question"] == "Clean question?"
