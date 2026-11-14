from feedback.stage2_diagnosis import allocate_sombench_reports_by_qtype
from feedback.stage3_synthesis import ToMBenchQuestionFlat


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
