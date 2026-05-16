from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    score: float
    details: dict[str, Any]
    metabolic_burden_score: float = 1.0
    gate_count: int = 0
    complexity_penalty: float = 0.0
    robustness_score: float = 1.0
    signal_to_noise_ratio: float = 0.0
    monte_carlo_runs: int = 0
    temporal_score: float = 1.0
    rise_time: float | None = None
    orthogonality_score: float = 1.0
    cello_assignment_score: float = 0.0
    cello_buildable: bool = False
    semantic_faithfulness_score: float = 1.0
    missed_edge_cases: list[str] | None = None


class BaseEvaluator:
    def evaluate(self, candidate: dict[str, Any]) -> EvaluationResult:
        raise NotImplementedError
