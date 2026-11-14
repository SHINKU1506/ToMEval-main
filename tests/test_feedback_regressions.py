from feedback.stage1_load_predictions import _extract_letters, _is_correct, get_dimension_key
from feedback.stage2_diagnosis import allocate_sombench_reports_by_qtype
from feedback.stage3_synthesis import EmoBenchGroupedQuestionFlat, EmoBenchQuestionFlat, ToMBenchQuestionFlat
from feedback.prompts import build_stage2_generation_from_report_prompt
from feedback.report_summary import generate_summary_report
from filter.eval.eval_answerability_full import _build_prompt as build_answerability_prompt


def test_sombench_report_allocation_preserves_available_qtypes():
    dim_counts = {
        "1.1.1__Q1__zh": 2,
        "1.1.2__Q2__zh": 35,
        "1.1.3__Q3__zh": 10,
        **{f"1.2.{i}__Q4__zh": 5 for i in range(1, 72)},
    }

    allocations = allocate_sombench_reports_by_qtype(dim_counts, total_reports=80)

    by_qtype = {}
    for key, value in allocations.items():
        qtype = key.split("__")[1]
        by_qtype[qtype] = by_qtype.get(qtype, 0) + value

    assert sum(allocations.values()) == 80
    assert set(by_qtype) == {"Q1", "Q2", "Q3", "Q4"}
    assert all(value >= 1 for value in by_qtype.values())
    assert by_qtype["Q4"] < 80


def test_tombench_storage_removes_embedded_options_from_question():
    question = ToMBenchQuestionFlat(
        story="A story.",
        question="What does A believe? A. The red box B. The blue box C. The green box D. The yellow box",
        correct_letter="B",
        meta_ability="false_belief",
        meta_id="synthetic-1",
    )

    stored = question.to_storage_dict()

    assert stored["question"] == "What does A believe?"
    assert stored["answer"]["correct_answers"] == ["The blue box"]
    assert len(stored["answer"]["wrong_answers"]) == 3


def test_emobench_storage_preserves_metadata_and_choice_texts():
    question = EmoBenchQuestionFlat(
        story_full="A story.",
        question="What should Alex do?",
        correct_answer="Talk to a teacher",
        wrong_answer_1="Keep the secret",
        wrong_answer_2="Confront the bully",
        meta_id="synthetic-2",
        meta_subset="emotional_application",
        meta_question_subtype="Action",
        meta_coarse_category="",
        meta_finegrained_category="",
        meta_dimension="emotional_application",
        meta_choice_texts=["Talk to a teacher", "Keep the secret", "Confront the bully"],
    )

    stored = question.to_storage_dict()

    assert stored["meta"]["subset"] == "emotional_application"
    assert stored["meta"]["question_subtype"] == "Action"
    assert stored["meta"]["choice_texts"] == [
        "Talk to a teacher", "Keep the secret", "Confront the bully"
    ]
    assert stored["meta"]["dimension"] == ["emotional_application"]


def test_emobench_grouped_schema_expands_to_paired_subquestions():
    question = EmoBenchGroupedQuestionFlat(
        story_full="A story.",
        emotion_question="How does Alex feel?",
        emotion_correct_answer="Relief",
        emotion_wrong_answers=["Anger", "Sadness", "Pride", "Fear", "Guilt"],
        cause_question="Why does Alex feel that way?",
        cause_correct_answer="The problem was resolved.",
        cause_wrong_answers=["The problem got worse.", "Alex ignored it.", "Alex left."],
        meta_id="synthetic-3",
        meta_coarse_category="relief",
        meta_finegrained_category="relief",
        meta_dimension="emotion_recognition",
    )

    rows = question.to_storage_rows()

    assert len(rows) == 2
    assert {row["meta"]["question_subtype"] for row in rows} == {"emotion", "cause"}
    assert len({row["meta"]["paired_group_id"] for row in rows}) == 1
    assert all(row["meta"]["subset"] == "emotional_understanding" for row in rows)


def test_emobench_dimension_key_keeps_subset_and_ea_eu_mode():
    assert "EU" in get_dimension_key(
        {"subset": "emotional_understanding", "dimension": ["emotion_recognition"], "language": "en"},
        "EmoBench",
    )
    assert "EA" in get_dimension_key(
        {"subset": "emotional_application", "dimension": ["Action"], "language": "en"},
        "EmoBench",
    )


def test_emobench_eu_prompt_requests_linked_grouped_questions():
    prompt = build_stage2_generation_from_report_prompt(
        report={
            "dimension": "EU__emotion_recognition",
            "common_error_patterns": [],
            "primary_cognitive_operation": "emotion_attribution",
            "difficulty_distribution": {},
            "_meta": {"dimension": "EU__emotion_recognition", "lang": "en"},
        },
        dataset_name="EmoBench",
    )

    assert "both linked EU subquestions" in prompt
    assert "cause answer" in prompt


def test_stage1_preserves_multiple_prediction_letters():
    assert _extract_letters(r"\boxed{A,C}") == ["A", "C"]
    assert _is_correct(r"\boxed{A,C}", ["A", "C"], prompt_type="mcq_multi")
    assert not _is_correct(r"\boxed{A}", ["A", "C"], prompt_type="mcq_multi")


def test_stage1_scores_grouped_predictions_against_each_subquestion():
    meta = {
        "sub_questions": [
            {"correct_letters": ["C"]},
            {"correct_letters": ["B"]},
        ]
    }
    assert _is_correct(r"\boxed{C}\n\boxed{B}", [], prompt_type="mcq_grouped", meta=meta)
    assert not _is_correct(r"\boxed{C}\n\boxed{A}", [], prompt_type="mcq_grouped", meta=meta)


def test_retry_prompt_reports_structured_output_failure():
    prompt = build_stage2_generation_from_report_prompt(
        report={
            "dimension": "false_belief",
            "common_error_patterns": [],
            "primary_cognitive_operation": "belief_state_update",
            "difficulty_distribution": {},
        },
        dataset_name="BigToM",
        previous_question={"_failure_reason": "structured_output_failed"},
    )

    assert "structured output failed" in prompt.lower()
    assert "too easy" not in prompt.lower()


def test_summary_report_uses_final_dataset_path(tmp_path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "SoMBench.parquet").write_bytes(b"placeholder")
    report = generate_summary_report({"SoMBench": {}}, ["SoMBench"], "all", "model", tmp_path)
    assert "SoMBench.parquet" in report


def test_answerability_prompt_explains_grouped_and_multi_answers():
    grouped = build_answerability_prompt({
        "story": "Story",
        "question": "Grouped",
        "answer": {"correct_answers": [], "wrong_answers": []},
        "meta": {
            "prompt_type_override": "mcq_grouped",
            "sub_questions": [{"question": "Q1", "options": {"A": "x"}, "correct_letters": ["A"]}],
        },
    }, "EmoBench")
    assert "Sub-question 1" in grouped
    multi = build_answerability_prompt({
        "story": "Story", "question": "Choose all",
        "answer": {"correct_answers": ["x", "y"], "wrong_answers": ["z"]},
        "meta": {},
    }, "SoMBench")
    assert "multiple correct answers" in multi
