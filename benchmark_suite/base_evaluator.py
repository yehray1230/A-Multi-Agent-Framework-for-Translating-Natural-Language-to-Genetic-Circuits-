from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    score: float
    details: dict


class BaseEvaluator:
    def evaluate(self, candidate: dict) -> EvaluationResult:
        raise NotImplementedError
