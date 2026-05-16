from __future__ import annotations

from benchmark_suite.base_evaluator import EvaluationResult


def score_static_plausibility(candidate: dict) -> EvaluationResult:
    score = float(candidate.get("plausibility_score", candidate.get("score", 0.0)))
    return EvaluationResult(score=score, details={"metric": "static_plausibility"})
