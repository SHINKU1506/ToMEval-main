from pathlib import Path
import json
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.synthesize_grpo_data import (
    build_mutation_plan,
    candidate_to_grpo_record,
    validate_candidate,
)


def test_build_mutation_plan_has_stable_grouped_mutations():
    seeds = [
        {
            "sample_id": "seed-1",
            "dataset": "SoMBench",
            "prompt_type": "mcq_single",
            "language": "zh",
            "dimension": ["Q3"],
            "story": "故事",
            "question": "问题",
            "options": {"A": "是", "B": "否", "C": "无法确定"},
            "gold": "C",
        }
    ]

    plan = build_mutation_plan(seeds, variants_per_seed=5)

    assert len(plan) == 5
    assert [item["mutation_type"] for item in plan] == [
        "repair",
        "contrast",
        "contrast",
        "counterfactual",
        "harder",
    ]
    assert {item["group_id"] for item in plan} == {"seed-1"}
    assert plan[0]["sample_id"] == "seed-1__repair__00"


def test_validate_candidate_rejects_gold_outside_options():
    candidate = {
        "sample_id": "candidate-1",
        "group_id": "seed-1",
        "dataset": "ToMBench",
        "prompt_type": "mcq_single",
        "language": "en",
        "story": "A story.",
        "question": "What happened?",
        "options": {"A": "one", "B": "two"},
        "gold": "C",
        "dimension": ["faux_pas"],
        "mutation_type": "contrast",
        "parent_id": "seed-1",
    }

    with pytest.raises(ValueError, match="gold"):
        validate_candidate(candidate)


def test_candidate_to_grpo_record_separates_prompt_and_reward_fields():
    candidate = {
        "sample_id": "candidate-1",
        "group_id": "seed-1",
        "dataset": "ToMBench",
        "prompt_type": "mcq_single",
        "language": "en",
        "story": "A story.",
        "question": "What happened?",
        "options": {"A": "one", "B": "two"},
        "gold": "B",
        "dimension": ["faux_pas"],
        "mutation_type": "contrast",
        "parent_id": "seed-1",
    }

    record = candidate_to_grpo_record(candidate)

    assert record["prompt"].startswith("Story:")
    assert "B" in record["prompt"]
    assert "gold" not in record["prompt"].lower()
    assert record["gold"] == "B"
    assert record["group_id"] == "seed-1"


def test_script_entrypoint_supports_dry_run_from_repo_root(tmp_path):
    seed_path = tmp_path / "seeds.jsonl"
    output_path = tmp_path / "plan.jsonl"
    seed_path.write_text(
        json.dumps(
            {
                "sample_id": "seed-1",
                "dataset": "EmoBench",
                "prompt_type": "mcq_single",
                "story": "A story.",
                "question": "What happened?",
                "options": {"A": "one", "B": "two"},
                "gold": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/synthesize_grpo_data.py",
            "--seeds",
            str(seed_path),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 5
