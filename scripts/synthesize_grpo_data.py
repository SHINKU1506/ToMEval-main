#!/usr/bin/env python3
"""Generate GRPO candidates from reviewed seed samples.

The script separates three concerns:
1. build a deterministic mutation plan from seed records;
2. optionally call the configured structured-output API model;
3. validate and serialize candidates into the GRPO prompt/reward schema.

It intentionally writes raw candidates. Run the existing answerability,
shortcut and LSH filters before adding the output to a training manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml
from pydantic import BaseModel, Field

# When invoked as python scripts/synthesize_grpo_data.py, Python puts the
# scripts directory on sys.path, so add the repository root for src.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import runner


LOGGER = logging.getLogger("synthesize_grpo_data")
MUTATION_TYPES = ("repair", "contrast", "contrast", "counterfactual", "harder")
PROMPT_TYPES = {"mcq_single", "mcq_multi", "mcq_grouped", "open"}


class GeneratedSubQuestion(BaseModel):
    question: str
    options: Dict[str, str] = Field(default_factory=dict)
    gold: Any


class GeneratedCandidate(BaseModel):
    story: str
    question: str = ""
    prompt_type: str
    language: str = "en"
    options: Dict[str, str] = Field(default_factory=dict)
    gold: Any
    sub_questions: List[GeneratedSubQuestion] = Field(default_factory=list)
    dimension: List[str] = Field(default_factory=list)
    difficulty: str = "medium"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_mutation_plan(
    seeds: Sequence[Dict[str, Any]], variants_per_seed: int = 5
) -> List[Dict[str, Any]]:
    """Create a deterministic set of mutations for each seed.

    The first five mutations are deliberately stable so rerunning a job does
    not change sample IDs. Additional variants use contrast mutations.
    """
    if variants_per_seed < 1:
        raise ValueError("variants_per_seed must be positive")

    plan: List[Dict[str, Any]] = []
    for seed in seeds:
        parent_id = str(seed.get("sample_id") or seed.get("id") or "").strip()
        if not parent_id:
            raise ValueError("Every seed needs sample_id or id")
        mutation_types = list(MUTATION_TYPES)
        if variants_per_seed > len(mutation_types):
            mutation_types.extend(["contrast"] * (variants_per_seed - len(mutation_types)))
        for index, mutation_type in enumerate(mutation_types[:variants_per_seed]):
            plan.append(
                {
                    "sample_id": f"{parent_id}__{mutation_type}__{index:02d}",
                    "group_id": parent_id,
                    "parent_id": parent_id,
                    "mutation_type": mutation_type,
                    "variant_index": index,
                    "seed": seed,
                }
            )
    return plan


def _validate_single_or_multi(candidate: Dict[str, Any]) -> None:
    options = candidate.get("options") or {}
    gold = candidate.get("gold")
    if not isinstance(options, dict) or len(options) < 2:
        raise ValueError("options must contain at least two entries")
    if candidate["prompt_type"] == "mcq_single":
        if not isinstance(gold, str) or gold not in options:
            raise ValueError("gold must be an option label for mcq_single")
    else:
        if not isinstance(gold, list) or not gold:
            raise ValueError("gold must be a non-empty list for mcq_multi")
        if any(not isinstance(label, str) or label not in options for label in gold):
            raise ValueError("every multi-choice gold label must be an option label")


def validate_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a candidate before it enters the raw GRPO data file."""
    for field in ("sample_id", "group_id", "dataset", "story", "prompt_type"):
        if not str(candidate.get(field, "")).strip():
            raise ValueError(f"missing required field: {field}")

    prompt_type = candidate["prompt_type"]
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"unsupported prompt_type: {prompt_type}")
    if prompt_type in {"mcq_single", "mcq_multi"}:
        _validate_single_or_multi(candidate)
    elif prompt_type == "mcq_grouped":
        sub_questions = candidate.get("sub_questions") or []
        if len(sub_questions) < 2:
            raise ValueError("mcq_grouped requires at least two sub_questions")
        for sub in sub_questions:
            if not str(sub.get("question", "")).strip():
                raise ValueError("every grouped sub-question needs question text")
            options = sub.get("options") or {}
            gold = sub.get("gold")
            if len(options) < 2 or not isinstance(gold, str) or gold not in options:
                raise ValueError("grouped sub-question has invalid options or gold")
    elif not isinstance(candidate.get("gold"), str) or not candidate["gold"].strip():
        raise ValueError("open candidate needs non-empty gold text")

    return candidate


def _format_options(options: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in options.items())


def candidate_to_grpo_record(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Separate model-visible prompt text from reward metadata."""
    validate_candidate(candidate)
    prompt_type = candidate["prompt_type"]
    if prompt_type == "mcq_grouped":
        parts = [f"Scenario:\n{candidate['story']}"]
        for index, sub in enumerate(candidate["sub_questions"], start=1):
            parts.append(f"Question {index}:\n{sub['question']}\n{_format_options(sub['options'])}")
        parts.append("Answer each question in order using the required answer format.")
        prompt = "\n\n".join(parts)
        answer_schema = {"type": "array", "items": "option_label", "count": len(candidate["sub_questions"])}
    elif prompt_type in {"mcq_single", "mcq_multi"}:
        prompt = (
            f"Story:\n{candidate['story']}\n\nQuestion:\n{candidate.get('question', '')}\n\n"
            f"Options:\n{_format_options(candidate['options'])}\n\n"
            "Answer using the required option-label format."
        )
        answer_schema = {"type": "option_label" if prompt_type == "mcq_single" else "option_label_set"}
    else:
        prompt = f"Story:\n{candidate['story']}\n\nQuestion:\n{candidate.get('question', '')}\n\nAnswer the question concisely."
        answer_schema = {"type": "text"}

    return {
        "sample_id": candidate["sample_id"],
        "group_id": candidate["group_id"],
        "dataset": candidate["dataset"],
        "prompt_type": prompt_type,
        "prompt": prompt,
        "gold": candidate["gold"],
        "answer_schema": answer_schema,
        "dimension": candidate.get("dimension", []),
        "difficulty": candidate.get("difficulty", "medium"),
        "source_kind": "synthetic",
        "source_id": candidate.get("parent_id", candidate["group_id"]),
        "mutation_type": candidate.get("mutation_type", "contrast"),
        "quality_status": "raw_candidate",
    }


def build_generation_prompt(item: Dict[str, Any]) -> str:
    seed = item["seed"]
    return f"""Generate exactly one high-quality GRPO candidate.

Dataset: {seed.get('dataset', '')}
Prompt type: {seed.get('prompt_type', '')}
Language: {seed.get('language', 'en')}
Target dimensions: {json.dumps(seed.get('dimension', []), ensure_ascii=False)}
Mutation type: {item['mutation_type']}

Seed story:
{seed.get('story', '')}

Seed question:
{seed.get('question', '')}

Seed answer metadata, for generation only:
{json.dumps({'options': seed.get('options', {}), 'gold': seed.get('gold')}, ensure_ascii=False)}

Rules:
1. Change only the variable implied by the mutation type.
2. Make the gold answer uniquely supported by the story.
3. Keep all distractors plausible but false.
4. Do not include answer explanations, internal labels, or grading metadata in story/question.
5. Return one object matching the requested structured schema.
"""


def generate_candidates(
    plan: Sequence[Dict[str, Any]],
    client: Any,
    batch_size: int = 20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for start in range(0, len(plan), batch_size):
        batch = list(plan[start : start + batch_size])
        responses = client.batch_generate_structure(
            [build_generation_prompt(item) for item in batch],
            GeneratedCandidate,
            mode="auto",
            desc="Generating GRPO candidates",
        )
        for item, response in zip(batch, responses):
            content = response.content if response else None
            if content is None:
                rejected.append({"plan": item, "reason": "api_generation_failed"})
                continue
            candidate = content.model_dump()
            candidate.update(
                {
                    "sample_id": item["sample_id"],
                    "group_id": item["group_id"],
                    "parent_id": item["parent_id"],
                    "mutation_type": item["mutation_type"],
                    "dataset": item["seed"].get("dataset", ""),
                }
            )
            try:
                accepted.append(candidate_to_grpo_record(candidate))
            except (TypeError, ValueError) as exc:
                rejected.append({"candidate": candidate, "reason": str(exc)})
    return accepted, rejected


def load_synthesis_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model_config = config.get("synthesis_model") or config.get("llm")
    if not isinstance(model_config, dict):
        raise ValueError("config must contain synthesis_model or llm mapping")
    return dict(model_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True, help="Seed JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Accepted raw GRPO JSONL")
    parser.add_argument("--config", type=Path, default=Path("feedback/config.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rejected", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variants-per-seed", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    seeds = read_jsonl(args.seeds)
    if args.dataset:
        seeds = [seed for seed in seeds if seed.get("dataset") == args.dataset]
    if args.limit > 0:
        seeds = seeds[: args.limit]
    plan = build_mutation_plan(seeds, variants_per_seed=args.variants_per_seed)

    manifest = {
        "seed_file": str(args.seeds),
        "dataset": args.dataset,
        "seed_count": len(seeds),
        "plan_count": len(plan),
        "variants_per_seed": args.variants_per_seed,
        "mutation_types": list(MUTATION_TYPES),
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        write_jsonl(args.output, plan)
        LOGGER.info("Wrote dry-run plan: %s", args.output)
        return 0

    model_config = load_synthesis_config(args.config)
    client = runner.create_llm_client(model_config)
    accepted, rejected = generate_candidates(plan, client, batch_size=args.batch_size)
    write_jsonl(args.output, accepted)
    rejected_path = args.rejected or args.output.with_name(args.output.stem + ".rejected.jsonl")
    write_jsonl(rejected_path, rejected)
    LOGGER.info("accepted=%d rejected=%d output=%s", len(accepted), len(rejected), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
