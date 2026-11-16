import json
from pathlib import Path


def test_repair_prompt_contains_recent_hardening_contracts():
    from feedback.tombench_gemini_repair import build_repair_prompt

    prompt = build_repair_prompt(
        {
            "story": "Alex moved a box.",
            "question": "What does Maya believe?",
            "answer": {"correct_answers": ["A"], "wrong_answers": ["B", "C", "D"]},
            "meta": {"id": "synthetic_tombench_000_0_0", "lang": "en"},
        },
        {"single_correct": 8, "total": 8, "pass_at_8": True},
        last_error="previous_error_hint",
    )

    assert "target_moment" in prompt
    assert "reasoning_trap_plan" in prompt
    assert "partial knowledge" in prompt.lower()
    assert "misleading" in prompt.lower()
    assert "without telling anyone" in prompt
    assert "gaze/smile" in prompt.lower()
    assert "DIFFICULTY ESCALATION" in prompt
    assert "No Distractor-Killing Patches" in prompt
    assert "counter-directional partial cue" in prompt.lower()
    assert "recent commits" not in prompt
    assert "previous_error_hint" in prompt

    ans_prompt = build_repair_prompt(
        {
            "story": "Alex moved a box.",
            "question": "What does Maya believe?",
            "answer": {"correct_answers": ["A"], "wrong_answers": ["B", "C", "D"]},
            "meta": {"id": "synthetic_tombench_001_3_0", "lang": "en"},
        },
        {"single_correct": 0, "total": 8, "pass_at_8": False},
    )
    assert "ANSWERABILITY FIX" in ans_prompt


def test_validate_repaired_row_rejects_signpost_gaze_and_accepts_non_spoiler_access():
    from feedback.tombench_gemini_repair import validate_repaired_row

    bad = {
        "reasoning_trap_plan": "partial cue",
        "target_moment": "Maya asks whether the box is still sealed.",
        "story": "Alex worked completely alone and moved the box. Maya asks whether the box is still sealed.",
        "question": "What does Maya believe?",
        "options": {"A": "It is sealed", "B": "It is open", "C": "It is missing", "D": "It is empty"},
        "correct_letter": "A",
    }
    assert validate_repaired_row(bad) == "explicit_isolation_signpost"

    # Must pass WITHOUT forcing narrator spoiler verbs like "assumed" or "thought"
    good = dict(bad)
    good["story"] = (
        "Alex moved the box to the upper locker while Maya stepped to the front desk to sign for a parcel. "
        "Maya returned carrying the invoice and saw the lower shelf empty before glancing at the brass tag. "
        "Maya asks whether the box is still sealed."
    )
    assert validate_repaired_row(good) is None

    # Gaze + smile template in final sentences must be rejected
    gaze_smile = dict(bad)
    gaze_smile["target_moment"] = "Alex caught Owen's eye, smiled toward the crate, and asked if it was full."
    gaze_smile["story"] = (
        "Alex moved the box before Owen walked back in from the porch. "
        "Alex caught Owen's eye, smiled toward the crate, and asked if it was full."
    )
    assert validate_repaired_row(gaze_smile) == "gaze_smile_template_cue_rewrite_with_assigned_evidence_channel"

    # Narrator belief spoiler must be rejected by Stage 3 validator
    spoiler = dict(bad)
    spoiler["story"] = (
        "Alex moved the box before Maya returned from the hallway. "
        "Unbeknownst to Maya, the brass key stayed inside. "
        "Maya asks whether the box is still sealed."
    )
    assert validate_repaired_row(spoiler) == "explicit_isolation_signpost_detected_rewrite_boundaries_implicitly"


def test_gemini_client_strips_openai_v1_suffix_before_native_route():
    from feedback.tombench_gemini_repair import GeminiRepairClient

    client = GeminiRepairClient("https://api.tokenkey.dev/v1", "key", "gemini-4-hydrogen")
    assert client.native_url() == "https://api.tokenkey.dev/v1beta/models/gemini-4-hydrogen:generateContent"


def test_repair_file_preserves_valid_in_band_samples_and_repairs_easy_ones(monkeypatch, tmp_path):
    import feedback.tombench_gemini_repair as repair_mod

    in_band_row = {
        "story": "Maya set the tea tin on the top shelf before stepping outside. Alex swapped the lids.",
        "question": "What does Maya expect inside the tea tin?",
        "answer": {"correct_answers": ["Jasmine tea"], "wrong_answers": ["Green tea", "Coffee beans", "Cocoa powder"]},
        "meta": {"id": "synthetic_tombench_000_0_0", "lang": "en"},
    }
    easy_row = {
        "story": "Alex put the key on the desk. Maya watched.",
        "question": "Where is the key?",
        "answer": {"correct_answers": ["On the desk"], "wrong_answers": ["In the bag", "On the chair", "In the drawer"]},
        "meta": {"id": "synthetic_tombench_000_1_0", "lang": "en"},
    }
    in_path = tmp_path / "input.jsonl"
    in_path.write_text(
        json.dumps(in_band_row) + "\n" + json.dumps(easy_row) + "\n",
        encoding="utf-8",
    )
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps([
            {"sample_id": "synthetic_tombench_000_0_0", "single_correct": 4, "total": 8},
            {"sample_id": "synthetic_tombench_000_1_0", "single_correct": 8, "total": 8},
        ]),
        encoding="utf-8",
    )
    api_cfg = tmp_path / "api.toml"
    api_cfg.write_text('[synthesis]\napi_url = "https://example.test/v1"\napi_key = "k"\n', encoding="utf-8")
    out_path = tmp_path / "repaired.jsonl"

    called_ids = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def repair(self, row, st):
            called_ids.append(row["meta"]["id"])
            return {
                "reasoning_trap_plan": "p",
                "target_moment": "Maya reached for the brass drawer.",
                "story": "Alex moved the key after Maya left to print invoices. Maya reached for the brass drawer.",
                "question": "Why did Maya open the brass drawer?",
                "options": {
                    "A": "To pick up the brass key",
                    "B": "To check the printer tray",
                    "C": "To lock the front window",
                    "D": "To find the spare ledger",
                },
                "correct_letter": "A",
            }

    monkeypatch.setattr(repair_mod, "GeminiRepairClient", FakeClient)
    summary = repair_mod.repair_file(in_path, stats_path, out_path, api_cfg, "gemini-4-hydrogen", workers=2)
    assert called_ids == ["synthetic_tombench_000_1_0"]
    assert summary["kept_in_band"] == 1
    assert summary["repaired"] == 2
    saved = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert saved[0]["meta"]["repair_action"] == "kept_in_band"
    assert saved[1]["meta"]["repair_action"] == "escalate_difficulty"
