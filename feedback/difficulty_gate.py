"""
难度闭环：用 solver 模型对合成题跑 pass@k，过易的题打回重写。

复用评测链路（predict_records + judge_repeat），因此 prompt 排版、选项 shuffle、
\\boxed{} 抽取与正式评测一致。只对 MCQ 题型判分；open 题和被 prepare_samples
改写 sample_id 的题（如 EmoBench EU 合并）视为通过。
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src import runner
from src.evaluation.data import normalize_sample
from src.evaluation.judge import judge_repeat
from src.evaluation.prediction import predict_records
from src.evaluation.protocols import llm_overrides_for

logger = logging.getLogger(__name__)

_GATED_PROMPT_TYPES = {"mcq_single", "mcq_multi", "mcq_grouped"}


class DifficultyGate:
    """pass@k solve-rate gate. A question passes when solve_rate <= max_solve_rate."""

    def __init__(
        self,
        solver_llm_config: Dict[str, Any],
        k: int = 4,
        max_solve_rate: float = 0.75,
        protocol: str = "cot",
    ):
        if k < 1:
            raise ValueError("difficulty gate k must be >= 1")
        self.k = int(k)
        self.max_solve_rate = float(max_solve_rate)
        self.protocol = protocol
        config = dict(solver_llm_config)
        config.update(llm_overrides_for(protocol))
        self.model_name = str(config.get("model_name", "unknown"))
        self.client = runner.create_content_client(config)

    def evaluate(self, rows: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """Return {sample_id: {"correct": int, "total": int, "solve_rate": float}}."""
        samples = []
        for index, row in enumerate(rows):
            try:
                samples.append(normalize_sample(row, index))
            except ValueError as exc:
                logger.warning(f"[difficulty_gate] skip malformed row: {exc}")
        if not samples:
            return {}

        records = predict_records(samples, dataset_name, self.client, self.k, protocol=self.protocol)
        gated = [r for r in records if r["prompt_type"] in _GATED_PROMPT_TYPES]
        results = judge_repeat(gated) if gated else []

        counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for record, result in zip(gated, results):
            entry = counts[record["sample_id"]]
            entry[0] += int(bool(result.get("is_correct")))
            entry[1] += 1
        return {
            sid: {"correct": c, "total": t, "solve_rate": c / t if t else 0.0}
            for sid, (c, t) in counts.items()
        }

    def passes(self, stats: Optional[Dict[str, Any]]) -> bool:
        if not stats or not stats.get("total"):
            return True
        return stats["solve_rate"] <= self.max_solve_rate


def build_difficulty_gate(gate_config: Optional[Dict[str, Any]]) -> Optional[DifficultyGate]:
    """Build a gate from the resolved `roles.difficulty_gate` + `synthesis.difficulty_gate` config."""
    if not gate_config or not gate_config.get("enabled", False):
        return None
    llm = dict(gate_config.get("llm") or {})
    if not llm.get("model_name") or not llm.get("api_key"):
        raise ValueError("difficulty_gate is enabled but its solver model_name/api_key is missing")
    return DifficultyGate(
        solver_llm_config=llm,
        k=int(gate_config.get("k", 4)),
        max_solve_rate=float(gate_config.get("max_solve_rate", 0.75)),
        protocol=str(gate_config.get("protocol", "cot")),
    )
