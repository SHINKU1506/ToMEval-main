import json
from feedback.stage1_load_predictions import _extract_letters, _is_correct, get_dimension_key
from feedback.stage2_diagnosis import allocate_sombench_reports_by_qtype
from feedback.stage3_synthesis import (
    EmoBenchGroupedQuestionFlat,
    EmoBenchQuestionFlat,
    ToMBenchQuestionFlat,
    _parse_existing_pair_id,
    _validate_synthesized_question,
    synthesize_from_reports,
)
from feedback.prompts import build_stage2_generation_from_report_prompt
from feedback.report_summary import generate_summary_report
from filter.eval.eval_answerability_full import _build_prompt as build_answerability_prompt
from filter.pipeline import prepare_filter_input
from filter.utils import is_correct_mcq, row_to_sample, sample_prompt_type
from src.evaluation.prompts import extract_prediction_value


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
    assert by_qtype["Q4"] <= 8
    assert by_qtype["Q3"] >= 20


def test_tombench_storage_removes_embedded_options_from_question():
    question = ToMBenchQuestionFlat(
        target_moment="A story.",
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


def test_sombench_q3_variants_rotate_verdict_targets():
    from feedback.prompts import build_stage2_generation_from_report_prompt

    report = {
        "dimension": "1.1.1__Q3__zh",
        "common_error_patterns": [],
        "primary_cognitive_operation": "social_inference",
        "difficulty_distribution": {},
        "_meta": {"dimension": "1.1.1__Q3__zh", "lang": "zh"},
    }
    prompts = [
        build_stage2_generation_from_report_prompt(report, "SoMBench", variant_index=i, lang="zh")
        for i in range(3)
    ]

    assert "correct_verdict\": \"是\"" in prompts[0]
    assert "correct_verdict\": \"否\"" in prompts[1]
    assert "correct_verdict\": \"无法确定\"" in prompts[2]


def test_hard_mix_schedules_hard_medium_and_easy_variants():
    import re
    from collections import Counter

    report = {
        "dimension": "false_belief",
        "common_error_patterns": [],
        "primary_cognitive_operation": "belief_state_update",
        "difficulty_distribution": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    }

    # Verify smooth interleaving across a 20-slot cycle and across 5-sample reports
    picked = []
    for idx in range(20):
        prompt = build_stage2_generation_from_report_prompt(
            report,
            "ToMBench",
            variant_index=idx % 5,
            schedule_index=idx,
            target_difficulty="hard_mix",
        )
        m = re.search(r"Target difficulty: \*\*(\w+)\*\*", prompt)
        assert m is not None
        picked.append(m.group(1))

    assert Counter(picked) == {"hard": 14, "medium": 5, "easy": 1}
    # First 5 variants (one report with samples_per_report=5) are no longer 100% hard
    assert "medium" in picked[:5]

    # Explicit target_difficulty="hard" forces 100% hard across all variants
    for idx in range(5):
        prompt_hard = build_stage2_generation_from_report_prompt(
            report,
            "ToMBench",
            variant_index=idx,
            schedule_index=idx,
            target_difficulty="hard",
        )
        assert "Target difficulty: **hard**" in prompt_hard


def test_tombench_synthesis_forces_create_mode_for_fenced_json(monkeypatch):
    from types import SimpleNamespace
    import feedback.stage3_synthesis as stage3

    calls = []

    class FakeClient:
        def batch_generate_structure(self, prompts, schema, mode="auto", desc=""):
            calls.append(mode)
            assert mode == "create"
            question = ToMBenchQuestionFlat(
                target_moment="A story.",
                story="A story.",
                question="What does Alex believe? A. Red B. Blue C. Green D. Yellow",
                correct_letter="A",
                meta_ability="false_belief",
                meta_id="synthetic-1",
            )
            return [SimpleNamespace(content=SimpleNamespace(questions=[question])) for _ in prompts]

    monkeypatch.setattr(stage3.runner, "create_llm_client", lambda config: FakeClient())

    results, stats = synthesize_from_reports(
        reports=[{
            "dimension": "false_belief",
            "common_error_patterns": [],
            "primary_cognitive_operation": "belief_state_update",
            "difficulty_distribution": {},
        }],
        dataset_name="ToMBench",
        synthesis_llm_config={"model_name": "test"},
        samples_per_report=1,
        max_retries=1,
    )

    assert calls == ["create"]
    assert stats["succeeded"] == 1
    assert len(results) == 1


def test_stage3_resume_recognizes_question_type_suffix_in_ids():
    assert _parse_existing_pair_id("synthetic_tombench_000_0_0") == (0, 0)
    assert _parse_existing_pair_id("synthetic_tombench_001_0_0") == (1, 0)
    assert _parse_existing_pair_id("synthetic_tombench_003_4_0") == (3, 4)
    assert _parse_existing_pair_id("synthetic_tombench_012_4") == (12, 4)
    assert _parse_existing_pair_id("not-a-synthetic-id") is None


def test_stage3_validator_rejects_length_bias_and_retries(monkeypatch):
    from types import SimpleNamespace
    import feedback.stage3_synthesis as stage3

    biased = {
        "story": "A concise story.",
        "question": "What does Alex believe?",
        "answer": {
            "correct_answers": ["She thinks the parcel arrived yesterday morning before noon and was moved upstairs."],
            "wrong_answers": ["Left early.", "Not sure.", "Stayed home."],
        },
    }
    assert _validate_synthesized_question(biased, "ToMBench") == (
        "correct_option_significantly_longer_than_distractors_match_option_lengths"
    )

    prompts_seen = []

    class FakeClient:
        def batch_generate_structure(self, prompts, schema, mode="auto", desc=""):
            prompts_seen.extend(prompts)
            if len(prompts_seen) == 1:
                q = ToMBenchQuestionFlat(
                    target_moment="Maya left her notebook on the desk before going out.",
                    story="Maya left her notebook on the desk before going out.",
                    question=(
                        "What does Maya think? "
                        "A. She believes the notebook is still sitting untouched on the same corner of her desk. "
                        "B. Upstairs. "
                        "C. Kitchen. "
                        "D. Gone."
                    ),
                    correct_letter="A",
                    meta_ability="false_belief",
                    meta_id="s1",
                )
            else:
                q = ToMBenchQuestionFlat(
                    target_moment="Maya left her notebook on the desk before going out.",
                    story="Maya left her notebook on the desk before going out.",
                    question=(
                        "What does Maya think? "
                        "A. The notebook is still on the study desk. "
                        "B. The notebook was moved to the kitchen. "
                        "C. The notebook is locked inside the cabinet. "
                        "D. The notebook was borrowed by her brother."
                    ),
                    correct_letter="A",
                    meta_ability="false_belief",
                    meta_id="s1",
                )
            return [SimpleNamespace(content=SimpleNamespace(questions=[q]))]

    monkeypatch.setattr(stage3.runner, "create_llm_client", lambda config: FakeClient())
    results, stats = synthesize_from_reports(
        reports=[{
            "dimension": "false_belief",
            "common_error_patterns": [],
            "primary_cognitive_operation": "belief_state_update",
            "difficulty_distribution": {},
        }],
        dataset_name="ToMBench",
        synthesis_llm_config={"model_name": "test"},
        samples_per_report=1,
        max_retries=2,
    )

    assert len(prompts_seen) == 2
    assert "correct option significantly longer than distractors" in prompts_seen[1]
    assert stats["succeeded"] == 1
    assert len(results) == 1


def test_hard_synthesis_rejects_explicit_isolation_signposts():
    sample = {
        "story": "Alex stayed in the room working completely alone and swapped the jars without telling anyone.",
        "question": "What does Maya expect inside the jar?",
        "answer": {
            "correct_answers": ["Sugar"],
            "wrong_answers": ["Salt", "Flour", "Rice"],
        },
    }
    err = _validate_synthesized_question(sample, "ToMBench", difficulty="hard")
    assert err == "explicit_isolation_signpost_detected_rewrite_boundaries_implicitly"
    assert _validate_synthesized_question(sample, "ToMBench", difficulty="medium") is None


def test_hard_prompt_and_easy_repair_enforce_cot_hardening_rules():
    from filter.repair.repair_prompts import REPAIR_EASY_PROMPT

    report = {
        "dimension": "Belief: Content false beliefs",
        "common_error_patterns": ["Agent A misreads a partial cue."],
        "primary_cognitive_operation": "belief_state_update",
        "difficulty_distribution": {"hard": 1.0},
    }
    prompt = build_stage2_generation_from_report_prompt(
        report,
        "ToMBench",
        target_difficulty="hard",
    )
    assert "Partial Knowledge + Misleading Attribution" in prompt
    assert "Competing Cross-Dimension Conflict" in prompt
    assert "NEVER write explicit boolean signposts" in prompt
    assert "NO SPOILER / STOP AT TARGET MOMENT" in prompt
    assert "truncated messages" in REPAIR_EASY_PROMPT
    assert "partial" in REPAIR_EASY_PROMPT.lower()


def test_tombench_schema_plans_before_story_and_strips_plan_on_storage():
    from feedback.prompts import SYNTHESIS_FORMAT_REGISTRY

    fields = list(ToMBenchQuestionFlat.model_fields.keys())
    assert fields[0] == "reasoning_trap_plan"
    skeleton_keys = list(SYNTHESIS_FORMAT_REGISTRY["ToMBench"]["json_skeleton"]["questions"][0].keys())
    assert skeleton_keys[0] == "reasoning_trap_plan"

    q = ToMBenchQuestionFlat(
        reasoning_trap_plan="Obs vs hidden swap; C=omniscient truth.",
        target_moment="Short natural story.",
        story="Short natural story.",
        question="What does Sam expect? A. Tea B. Coffee C. Cocoa D. Milk",
        correct_letter="A",
        meta_ability="false_belief",
        meta_id="s1",
    )
    stored = q.to_storage_dict()
    assert "reasoning_trap_plan" not in stored
    assert stored["answer"] == {
        "correct_answers": ["Tea"],
        "wrong_answers": ["Coffee", "Cocoa", "Milk"],
    }


def test_hard_tombench_rejects_story_continuing_after_target_moment():
    sample = ToMBenchQuestionFlat(
        reasoning_trap_plan="Partial cue; hidden update; distractors use world truth.",
        target_moment='Maya asks, "Did the key stay in the drawer?"',
        story=(
            'Maya asks, "Did the key stay in the drawer?" '
            "Liam immediately opens the drawer and points to the key."
        ),
        question='At that moment, what does Maya intend to find out? A. The key location. B. The weather. C. The time. D. The price.',
        correct_letter="A",
        meta_ability="pragmatic_inference",
        meta_id="s1",
    )
    stored = sample.to_storage_dict()
    stored["_target_moment"] = sample.target_moment
    assert _validate_synthesized_question(stored, "ToMBench", difficulty="hard") == (
        "story_continues_after_target_moment"
    )


def test_hard_tombench_accepts_story_ending_at_target_moment():
    sample = ToMBenchQuestionFlat(
        reasoning_trap_plan="Partial cue; hidden update; distractors use world truth.",
        target_moment='Maya asks, "Did the key stay in the drawer?"',
        story='Maya asks, "Did the key stay in the drawer?"',
        question='At that moment, what does Maya intend to find out? A. The key location. B. The weather. C. The time. D. The price.',
        correct_letter="A",
        meta_ability="pragmatic_inference",
        meta_id="s1",
    )
    stored = sample.to_storage_dict()
    stored["_target_moment"] = sample.target_moment
    assert _validate_synthesized_question(stored, "ToMBench", difficulty="hard") is None


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


def test_emobench_grouped_reports_use_grouped_schema_route():
    from feedback.stage3_synthesis import EmoBenchGroupedSynthesisOutput

    assert EmoBenchGroupedSynthesisOutput.__name__ == "EmoBenchGroupedSynthesisOutput"


def test_filter_merges_emobench_eu_rows_before_evaluation():
    rows = [
        {
            "story": "A story.", "question": "How does Alex feel?",
            "answer": {"correct_answers": ["Relief"], "wrong_answers": ["Anger", "Fear"]},
            "meta": {"id": "g_emotion", "subset": "emotional_understanding", "language": "en",
                     "question_subtype": "emotion", "paired_group_id": "group-1",
                     "dimension": ["emotion_recognition"]},
        },
        {
            "story": "A story.", "question": "Why does Alex feel that way?",
            "answer": {"correct_answers": ["The problem ended"], "wrong_answers": ["It worsened", "No reason"]},
            "meta": {"id": "g_cause", "subset": "emotional_understanding", "language": "en",
                     "question_subtype": "cause", "paired_group_id": "group-1",
                     "dimension": ["emotion_cause"]},
        },
    ]

    merged = prepare_filter_input(rows, "EmoBench")

    assert len(merged) == 1
    assert merged[0]["meta"]["prompt_type_override"] == "mcq_grouped"
    assert len(merged[0]["meta"]["sub_questions"]) == 2


def test_filter_grouped_rows_use_grouped_prompt_and_reward_parsing():
    merged = prepare_filter_input([
        {"story": "S", "question": "E", "answer": {"correct_answers": ["r"], "wrong_answers": ["a"]},
         "meta": {"id": "e", "subset": "emotional_understanding", "question_subtype": "emotion", "paired_group_id": "g", "language": "en"}},
        {"story": "S", "question": "C", "answer": {"correct_answers": ["why"], "wrong_answers": ["no"]},
         "meta": {"id": "c", "subset": "emotional_understanding", "question_subtype": "cause", "paired_group_id": "g", "language": "en"}},
    ], "EmoBench")
    sample = row_to_sample(merged[0])

    assert sample_prompt_type(sample) == "mcq_grouped"
    from types import SimpleNamespace
    response = SimpleNamespace(content="\\boxed{A}\n\\boxed{C}")
    assert extract_prediction_value("mcq_grouped", response) == ["A", "C"]
    assert is_correct_mcq("mcq_grouped", response, ["A", "C"])


def test_report_sanitizer_drops_post_target_confirmation_items():
    from feedback.prompts import _sanitize_report_items

    items = [
        "At least one relevant agent must respond in a way that confirms or disambiguates the implied intention.",
        "Agent A ignores the subsequent response of Agent C as evidence of a shared intention.",
        "Include a literal-meaning distractor that should trigger a word-by-word interpretation shortcut.",
        "Agent A's partial observation must precede the hidden update.",
    ]
    assert _sanitize_report_items(items) == items[2:]


def test_generation_prompt_omits_confirmation_constraints_from_report():
    report = {
        "dimension": "Intention: Intentions explanations",
        "primary_cognitive_operation": "pragmatic_inference",
        "common_error_patterns": ["Agent A reads the utterance literally."],
        "hard_constraints": [
            "At least one relevant agent must respond in a way that confirms the implied intention.",
            "The intended meaning must be inferable only from the preceding context.",
        ],
        "_meta": {"lang": "en"},
    }
    prompt = build_stage2_generation_from_report_prompt(report, "ToMBench", target_difficulty="hard")
    assert "respond in a way that confirms" not in prompt
    assert "inferable only from the preceding context" in prompt


def _hard_tombench_row(story, correct, wrong, target=None):
    return {
        "story": story,
        "question": "At that moment, what does Nora intend?",
        "answer": {"correct_answers": [correct], "wrong_answers": wrong},
        "meta": {"id": "s1"},
        "_target_moment": target if target is not None else story.split(". ")[-1],
    }


def test_hard_tombench_rejects_gaze_smile_template():
    story = (
        "Nora hid the watch in the owl box while Claire was downstairs. "
        'Nora caught Owen\'s eye, smiled toward the side table, and asked, "Isn\'t that box cheerful?"'
    )
    row = _hard_tombench_row(
        story,
        "Prompt Owen to bring the box holding the watch",
        ["Ask Owen whether the box looks cheerful", "Joke with Owen about the box design", "Ask Owen to move the box aside"],
        target='Nora caught Owen\'s eye, smiled toward the side table, and asked, "Isn\'t that box cheerful?"',
    )
    assert _validate_synthesized_question(row, "ToMBench", difficulty="hard") == (
        "gaze_smile_template_cue_rewrite_with_assigned_evidence_channel"
    )


def test_hard_tombench_rejects_whimsical_decoy_and_lone_hedge():
    story = "A goofy frog umbrella hung by the door. Nora read the revised schedule."
    row = _hard_tombench_row(
        story,
        "She expects the meeting at noon",
        ["She expects the meeting at ten", "She expects the meeting at two", "She expects the meeting at four"],
    )
    assert _validate_synthesized_question(row, "ToMBench", difficulty="hard") == (
        "whimsical_decoy_object_remove_amusement_red_herring"
    )

    story = "Nora read the first page of the revised schedule."
    row = _hard_tombench_row(
        story,
        "She is unsure whether the meeting moved",
        ["She is certain the meeting is at ten", "She is certain the meeting is at two", "She is certain it was cancelled"],
    )
    assert _validate_synthesized_question(row, "ToMBench", difficulty="hard") == (
        "single_hedged_option_match_certainty_style_across_options"
    )


def test_tombench_rejects_option_with_unseen_named_entity():
    story = "Nora read the first page of the revised schedule."
    row = _hard_tombench_row(
        story,
        "She thinks the meeting is at noon",
        ["She thinks Miriam cancelled it", "She thinks the meeting is at two", "She thinks the meeting is at four"],
    )
    assert _validate_synthesized_question(row, "ToMBench", difficulty="hard") == (
        "option_introduces_entity_absent_from_story"
    )


def test_hard_prompt_rotates_evidence_channel_and_forbids_templates():
    from feedback.prompts import _EVIDENCE_CHANNELS

    report = {"dimension": "d", "primary_cognitive_operation": "pragmatic_inference", "_meta": {"lang": "en"}}
    p0 = build_stage2_generation_from_report_prompt(report, "ToMBench", variant_index=0, target_difficulty="hard")
    p1 = build_stage2_generation_from_report_prompt(report, "ToMBench", variant_index=1, target_difficulty="hard")
    assert _EVIDENCE_CHANNELS[0] in p0 and _EVIDENCE_CHANNELS[1] not in p0
    assert _EVIDENCE_CHANNELS[1] in p1
    assert "catches eye / glances toward X / smiles" in p0
    assert "never invent a new character" in p0


def test_synthesis_covers_every_report_before_second_variant(monkeypatch):
    import feedback.stage3_synthesis as s3

    seen = []

    class FakeClient:
        def batch_generate_structure(self, prompts, schema, mode, desc):
            return [None for _ in prompts]

    def fake_prompt(**kwargs):
        seen.append((kwargs["report"]["id"], kwargs["variant_index"]))
        return "p"

    monkeypatch.setattr(s3.runner, "create_llm_client", lambda cfg: FakeClient())
    monkeypatch.setattr(s3, "build_stage2_generation_from_report_prompt", fake_prompt)
    reports = [{"id": i, "_meta": {"lang": "en"}} for i in range(3)]
    s3.synthesize_from_reports(reports, "ToMBench", {}, samples_per_report=2, max_retries=1)
    assert seen == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def test_difficulty_gate_rejects_easy_question_and_retries_with_too_easy_feedback(monkeypatch, tmp_path):
    import feedback.stage3_synthesis as s3
    from types import SimpleNamespace

    story = "Maya read the first page of the revised schedule."
    prompts_seen = []

    class FakeClient:
        def batch_generate_structure(self, prompts, schema, mode, desc):
            prompts_seen.extend(prompts)
            q = ToMBenchQuestionFlat(
                reasoning_trap_plan="plan",
                target_moment=story,
                story=story,
                question="When does Maya expect the meeting? A. At noon B. At ten C. At two D. At four",
                correct_letter="A",
                meta_ability="false_belief",
                meta_id="x",
            )
            return [SimpleNamespace(content=SimpleNamespace(questions=[q])) for _ in prompts]

    class FakeGate:
        model_name = "qwen3.8-flash"
        k = 4

        def __init__(self):
            self.calls = 0

        def evaluate(self, rows, dataset_name):
            self.calls += 1
            correct = 4 if self.calls == 1 else 1
            return {r["meta"]["id"]: {"correct": correct, "total": 4, "solve_rate": correct / 4} for r in rows}

        def passes(self, stats):
            return stats["solve_rate"] <= 0.75

    monkeypatch.setattr(s3.runner, "create_llm_client", lambda cfg: FakeClient())
    gate = FakeGate()
    log_file = tmp_path / "gate.jsonl"
    rows, stats = s3.synthesize_from_reports(
        [{"dimension": "d", "primary_cognitive_operation": "belief_state_update", "_meta": {"lang": "en"}}],
        "ToMBench", {}, samples_per_report=1, max_retries=2,
        target_difficulty="hard", difficulty_gate=gate, gate_log_file=log_file,
    )

    assert len(rows) == 1 and stats["difficulty_gate_rejected"] == 1
    assert "Previous Attempt Was Too Easy" in prompts_seen[1]
    assert "4/4 correct by the target model" in prompts_seen[1]
    log = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert [entry["passed"] for entry in log] == [False, True]


def test_difficulty_gate_drops_question_still_easy_after_last_round(monkeypatch):
    import feedback.stage3_synthesis as s3
    from types import SimpleNamespace

    story = "Maya read the first page of the revised schedule."

    class FakeClient:
        def batch_generate_structure(self, prompts, schema, mode, desc):
            q = ToMBenchQuestionFlat(
                target_moment=story, story=story,
                question="When does Maya expect the meeting? A. At noon B. At ten C. At two D. At four",
                correct_letter="A", meta_ability="false_belief", meta_id="x",
            )
            return [SimpleNamespace(content=SimpleNamespace(questions=[q])) for _ in prompts]

    class AlwaysEasyGate:
        model_name, k = "qwen3.8-flash", 4

        def evaluate(self, rows, dataset_name):
            return {r["meta"]["id"]: {"correct": 4, "total": 4, "solve_rate": 1.0} for r in rows}

        def passes(self, stats):
            return stats["solve_rate"] <= 0.75

    monkeypatch.setattr(s3.runner, "create_llm_client", lambda cfg: FakeClient())
    rows, stats = s3.synthesize_from_reports(
        [{"dimension": "d", "_meta": {"lang": "en"}}], "ToMBench", {},
        samples_per_report=1, max_retries=2, target_difficulty="hard", difficulty_gate=AlwaysEasyGate(),
    )
    assert rows == [] and stats["difficulty_gate_rejected"] == 2


def test_difficulty_gate_scores_with_eval_pipeline(monkeypatch):
    import feedback.difficulty_gate as dg
    from src.llm.client import LLMResponse

    class FakeContentClient:
        def batch_generate(self, prompts, desc="", system_prompts=None):
            return [LLMResponse(content="\boxed{A}", reasoning="") for _ in prompts]

    monkeypatch.setattr(dg.runner, "create_content_client", lambda cfg: FakeContentClient())
    gate = dg.DifficultyGate({"model_name": "qwen3.8-flash", "api_key": "k", "api_url": "u"}, k=4)
    row = {
        "story": "s", "question": "q",
        "answer": {"correct_answers": ["right"], "wrong_answers": ["w1", "w2", "w3"]},
        "meta": {"id": "synthetic_tombench_000_0_0", "lang": "en"},
    }
    stats = gate.evaluate([row], "ToMBench")["synthetic_tombench_000_0_0"]
    assert stats["total"] == 4 and 0 <= stats["correct"] <= 4
    assert gate.passes({"correct": 3, "total": 4, "solve_rate": 0.75})
    assert not gate.passes({"correct": 4, "total": 4, "solve_rate": 1.0})


def test_diagnosis_prompt_preserves_cot_head_and_tail_and_sanitizes_anti_patterns():
    from feedback.prompts import build_batch_diagnosis_prompt, _sanitize_report_items

    head = "HEAD_MARKER " + ("A" * 350)
    middle = "MIDDLE_FILLER " * 60
    tail = "TAIL_FINAL_BELIEF_ERROR " + ("Z" * 100)
    prompt = build_batch_diagnosis_prompt(
        [{"_actual_prompt": "Q?", "_wrong_reasoning": head + middle + tail, "_wrong_answers": ["B"], "_gold_answers": ["A"]}],
        "Belief: Content false beliefs",
        "ToMBench",
    )
    assert "HEAD_MARKER" in prompt
    assert "TAIL_FINAL_BELIEF_ERROR" in prompt
    assert "[middle truncated]" in prompt
    assert "NEVER prescribe explicit narrator belief/mental-state verbs" in prompt

    items = [
        "Use a glance and smile cue to hint at the hidden item.",
        "Make the correct answer 'pick one at random'.",
        "Enforce 'almost none != 0' when counting items.",
        "Place a genuine physical timing gap before the second inspection.",
    ]
    assert _sanitize_report_items(items) == [items[-1]]


def test_stage2_cyclic_sampling_prevents_quota_starvation_and_boosts_high_error_dims():
    from collections import Counter
    from feedback.stage2_diagnosis import sample_bad_cases_weighted
    from feedback.prompts import resolve_variant_difficulty

    cases = [{"sample_idx": 1, "_difficulty": 2}, {"sample_idx": 2, "_difficulty": 1}]
    sampled = sample_bad_cases_weighted(cases, n_samples=10, random_seed=7)
    assert len(sampled) == 10
    assert {c["sample_idx"] for c in sampled} == {1, 2}

    report = {
        "dimension": "Knowledge: Knowledge-attention links",
        "difficulty_distribution": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
        "difficulty_policy": "weighted",
        "_meta": {"dim_error_rate": 0.65, "hard_dimension_error_rate": 0.55, "hard_dimension_weight": 0.85},
    }
    picked = [resolve_variant_difficulty(report, variant_index=i % 5, schedule_index=i) for i in range(20)]
    assert Counter(picked)["hard"] >= 17


def test_stage3_prompt_encodes_review72_rules_and_rotates_by_schedule_index():
    from feedback.prompts import (
        _EVIDENCE_CHANNELS,
        _NAME_POOL_HINTS,
        build_stage2_generation_from_report_prompt,
    )

    report = {"dimension": "Knowledge: Information-knowledge links", "primary_cognitive_operation": "information_access", "_meta": {"lang": "zh"}}
    p_a = build_stage2_generation_from_report_prompt(
        report, "ToMBench", variant_index=0, schedule_index=0, target_difficulty="hard", lang="zh",
    )
    p_b = build_stage2_generation_from_report_prompt(
        report, "ToMBench", variant_index=0, schedule_index=3, target_difficulty="hard", lang="zh",
    )
    assert _EVIDENCE_CHANNELS[0] in p_a
    assert _EVIDENCE_CHANNELS[3] in p_b and _EVIDENCE_CHANNELS[0] not in p_b
    assert _NAME_POOL_HINTS["zh"][0] in p_a
    assert _NAME_POOL_HINTS["zh"][3] in p_b
    assert "NEVER narrate the queried mental state" in p_a
    assert "0.80x-1.20x" in p_a
    assert "bundle two independent claims" in p_a


def test_stage3_bilingual_validators_footer_strip_and_metadata_preservation(monkeypatch):
    from types import SimpleNamespace
    import feedback.stage3_synthesis as stage3

    q = ToMBenchQuestionFlat(
        target_moment="程悦看向桌上的罐子。",
        story="程悦看向桌上的罐子。",
        question=(
            "许文预期罐子里装的是什么？ "
            "A. Room B. 里的咖啡豆 "
            "B. 燕麦片 "
            "C. 绿茶包\n请选择正确答案。 "
            "D. 红茶包"
        ),
        correct_letter="A",
        meta_ability="theory_of_mind",
        meta_id="s1",
    )
    stored = q.to_storage_dict()
    assert stored["answer"]["correct_answers"] == ["Room B. 里的咖啡豆"]
    assert "请选择正确答案" not in stored["answer"]["wrong_answers"][1]

    zh_leak = {
        "story": "顾清离开后，沈舟换了标签，顾清误以为里面还是红茶。",
        "question": "顾清以为里面是什么？",
        "answer": {"correct_answers": ["红茶"], "wrong_answers": ["绿茶", "花茶", "白茶"]},
        "_target_moment": "顾清误以为里面还是红茶。",
    }
    assert _validate_synthesized_question(zh_leak, "ToMBench", difficulty="hard") == (
        "explicit_isolation_signpost_detected_rewrite_boundaries_implicitly"
    )

    zh_gaze = {
        "story": "顾清与沈舟对视一眼后微笑着指了指茶杯，问大家要不要加茶。",
        "question": "顾清的真实意图是什么？",
        "answer": {"correct_answers": ["提醒收拾茶杯"], "wrong_answers": ["夸奖茶杯好看", "询问茶水温度", "推荐新款茶叶"]},
        "_target_moment": "顾清与沈舟对视一眼后微笑着指了指茶杯，问大家要不要加茶。",
    }
    assert _validate_synthesized_question(zh_gaze, "ToMBench", difficulty="hard") == (
        "gaze_smile_template_cue_rewrite_with_assigned_evidence_channel"
    )

    meta_long = {
        "story": "Two jars stood on the shelf.",
        "question": "Which jar does she mean?",
        "answer": {
            "correct_answers": ["Cannot tell which jar; she must pick at random"],
            "wrong_answers": [
                "The left terracotta jar on the top",
                "The blue ceramic jar on the middle",
                "The green glass jar on the bottom",
            ],
        },
        "_target_moment": "Two jars stood on the shelf.",
    }
    assert _validate_synthesized_question(meta_long, "ToMBench", difficulty="hard") == (
        "indeterminacy_meta_option_longer_than_distractors"
    )

    class FakeClient:
        def batch_generate_structure(self, prompts, schema, mode="auto", desc=""):
            item = ToMBenchQuestionFlat(
                target_moment="沈舟把茶叶罐放回最上层。",
                story="顾清看到标签后走了出去，沈舟把茶叶罐放回最上层。",
                question="顾清认为罐子里是什么？",
                correct_letter="A",
                options={"A": "花茶", "B": "绿茶", "C": "红茶", "D": "白茶"},
                meta_ability="theory_of_mind",
                meta_difficulty="hard",
                meta_id="s1",
            )
            return [SimpleNamespace(content=SimpleNamespace(questions=[item])) for _ in prompts]

    monkeypatch.setattr(stage3.runner, "create_llm_client", lambda cfg: FakeClient())
    rows, stats = synthesize_from_reports(
        reports=[{
            "dimension": "Belief: Content false beliefs__zh",
            "primary_cognitive_operation": "belief_state_update",
            "_meta": {"dimension": "Belief: Content false beliefs__zh", "lang": "zh"},
        }],
        dataset_name="ToMBench",
        synthesis_llm_config={"model_name": "test"},
        samples_per_report=1,
        max_retries=1,
        target_difficulty="hard",
    )
    assert stats["succeeded"] == 1
    assert rows[0]["meta"]["dimension"] == ["Belief: Content false beliefs__zh"]
    assert rows[0]["meta"]["report_dimension"] == "Belief: Content false beliefs__zh"
    assert rows[0]["meta"]["primary_cognitive_operation"] == "belief_state_update"
    assert rows[0]["meta"]["difficulty"] == "hard"


def test_difficulty_gate_band_rejects_zero_solve_missing_stats_and_retries(monkeypatch, tmp_path):
    import feedback.difficulty_gate as dg
    import feedback.stage3_synthesis as stage3
    from types import SimpleNamespace

    gate = dg.build_difficulty_gate({
        "enabled": True,
        "k": 4,
        "min_solve_rate": 0.125,
        "max_solve_rate": 0.75,
        "fail_open": False,
        "llm": {"model_name": "checkpoint-124", "api_key": "k", "api_url": "u"},
    })
    assert gate is not None
    assert not gate.passes(None)
    assert gate.rejection_reason(None) == "missing_eval_stats"
    assert not gate.passes({"correct": 0, "total": 4, "solve_rate": 0.0})
    assert gate.rejection_reason({"correct": 0, "total": 4, "solve_rate": 0.0}) == "too_hard_or_unanswerable"
    assert gate.passes({"correct": 2, "total": 4, "solve_rate": 0.5})
    assert not gate.passes({"correct": 4, "total": 4, "solve_rate": 1.0})

    story = "Maya read the first page of the revised schedule."
    prompts_seen = []

    class FakeClient:
        def batch_generate_structure(self, prompts, schema, mode="auto", desc=""):
            prompts_seen.extend(prompts)
            q = ToMBenchQuestionFlat(
                reasoning_trap_plan="plan",
                target_moment=story,
                story=story,
                question="When does Maya expect the meeting? A. At noon B. At ten C. At two D. At four",
                correct_letter="A",
                meta_ability="false_belief",
                meta_id="x",
            )
            return [SimpleNamespace(content=SimpleNamespace(questions=[q])) for _ in prompts]

    calls = {"n": 0}

    def fake_evaluate(rows, dataset_name):
        calls["n"] += 1
        correct = 0 if calls["n"] == 1 else 2
        return {r["meta"]["id"]: {"correct": correct, "total": 4, "solve_rate": correct / 4} for r in rows}

    gate.evaluate = fake_evaluate
    monkeypatch.setattr(stage3.runner, "create_llm_client", lambda cfg: FakeClient())
    log_file = tmp_path / "gate_band.jsonl"
    rows, stats = stage3.synthesize_from_reports(
        [{"dimension": "d", "primary_cognitive_operation": "belief_state_update", "_meta": {"lang": "en"}}],
        "ToMBench", {}, samples_per_report=1, max_retries=2,
        target_difficulty="hard", difficulty_gate=gate, gate_log_file=log_file,
    )
    assert len(rows) == 1 and stats["difficulty_gate_rejected"] == 1
    assert "Unanswerable / Broken" in prompts_seen[1]
    assert "0/4 correct by the target model" in prompts_seen[1]
    log = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert [entry["reason"] for entry in log] == ["too_hard_or_unanswerable", None]


def test_filter_pipeline_guards_all_failed_and_repair_prompts_enforce_symmetry():
    import pandas as pd
    from filter.pipeline import _assign_one, assign_labels
    from filter.repair.repair_prompts import REPAIR_EASY_PROMPT, REPAIR_UNANSWERABLE_PROMPT

    assert _assign_one("all_failed", True, True, allow_all_failed_as_hard=True) == ("shortcut", "shortcut")
    assert _assign_one("all_failed", True, False, allow_all_failed_as_hard=False) == ("bad", "unanswerable")
    assert _assign_one("all_failed", True, False, allow_all_failed_as_hard=True) == ("hard", None)

    input_df = pd.DataFrame([
        {"story": "s1", "question": "q1", "answer": {"correct_answers": ["a"], "wrong_answers": ["b", "c", "d"]}, "meta": {"id": "id_1"}},
    ])
    passk_df = pd.DataFrame([{"sample_id": "id_1", "pass_at_k": 0.0, "bucket": "all_failed"}])
    ans_df = pd.DataFrame([{"sample_id": "id_1", "label": "answerable", "reason": "ok", "answerable": True}])
    shortcut_df = pd.DataFrame(columns=["sample_id", "no_story_pass", "no_question_pass", "no_options_pass", "is_shortcut"])
    labels_df = assign_labels(input_df, passk_df, ans_df, shortcut_df, iter_n=1, allow_all_failed_as_hard=False)
    assert labels_df.loc[0, "label"] == "bad"
    assert labels_df.loc[0, "repair_type"] == "unanswerable"

    assert "NEVER use narrator epistemic spoilers" in REPAIR_EASY_PROMPT
    assert "0.80x-1.20x" in REPAIR_EASY_PROMPT
    assert "factually true or practically superior" in REPAIR_UNANSWERABLE_PROMPT
