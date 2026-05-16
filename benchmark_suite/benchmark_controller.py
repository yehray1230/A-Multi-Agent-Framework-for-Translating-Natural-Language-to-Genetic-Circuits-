from __future__ import annotations

from typing import Any

from benchmark_suite.cello_constraint_evaluator import score_cello_constraints
from benchmark_suite.functional_scorer import score_functional
from benchmark_suite.kinetic_scorer import score_kinetic
from benchmark_suite.metabolic_scorer import score_metabolic_burden
from benchmark_suite.static_plausibility_evaluator import score_static_plausibility
from benchmark_suite.temporal_scorer import score_temporal

SCORE_WEIGHTS = {
    "functional": 0.22,
    "kinetic": 0.15,
    "static_plausibility": 0.08,
    "metabolic_burden": 0.15,
    "robustness": 0.15,
    "temporal": 0.05,
    "orthogonality": 0.10,
    "cello_assignment": 0.10,
}


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _candidate_float(candidate: dict[str, Any], key: str, default: float) -> float:
    try:
        return default if candidate.get(key) is None else float(candidate[key])
    except (TypeError, ValueError):
        return default


def _candidate_int(candidate: dict[str, Any], key: str, default: int) -> int:
    try:
        return default if candidate.get(key) is None else int(candidate[key])
    except (TypeError, ValueError):
        return default


def _candidate_bool(candidate: dict[str, Any], key: str, default: bool) -> bool:
    value = candidate.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
        return default
    return bool(value)


def _candidate_str_list(candidate: dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = candidate.get(key)
    if value is None:
        return default
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return default


def evaluate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    results = [
        score_functional(candidate),
        score_kinetic(candidate),
        score_static_plausibility(candidate),
        score_metabolic_burden(candidate),
        score_temporal(candidate),
        score_cello_constraints(candidate),
    ]
    component_scores = {
        str(result.details.get("metric", "unknown")): _clamp_score(result.score)
        for result in results
    }
    metabolic_result = next(
        result for result in results if result.details.get("metric") == "metabolic_burden"
    )
    kinetic_result = next(
        result for result in results if result.details.get("metric") == "kinetic"
    )
    cello_result = next(
        result for result in results if result.details.get("metric") == "cello_constraints"
    )
    temporal_result = next(
        result for result in results if result.details.get("metric") == "temporal"
    )
    robustness_score = _candidate_float(
        candidate,
        "robustness_score",
        kinetic_result.robustness_score,
    )
    orthogonality_score = cello_result.orthogonality_score
    cello_assignment_score = cello_result.cello_assignment_score
    cello_buildable = cello_result.cello_buildable
    temporal_score = temporal_result.temporal_score
    rise_time = temporal_result.rise_time
    semantic_faithfulness_score = _candidate_float(candidate, "semantic_faithfulness_score", 1.0)
    missed_edge_cases = _candidate_str_list(
        candidate,
        "missed_edge_cases",
        _candidate_str_list(candidate, "missed_conditions", []),
    )
    component_scores["robustness"] = _clamp_score(robustness_score)
    component_scores["temporal"] = _clamp_score(temporal_score)
    component_scores["orthogonality"] = _clamp_score(orthogonality_score)
    component_scores["cello_assignment"] = _clamp_score(cello_assignment_score)
    score = round(sum(
        component_scores.get(metric, 0.0) * weight
        for metric, weight in SCORE_WEIGHTS.items()
    ), 10)
    return {
        "score": score,
        "weighted_total_score": score,
        "grade": _grade(score),
        "metabolic_burden_score": metabolic_result.metabolic_burden_score,
        "gate_count": metabolic_result.gate_count,
        "complexity_penalty": metabolic_result.complexity_penalty,
        "robustness_score": robustness_score,
        "signal_to_noise_ratio": _candidate_float(
            candidate,
            "signal_to_noise_ratio",
            _candidate_float(candidate, "snr", 0.0),
        ),
        "monte_carlo_runs": _candidate_int(
            candidate,
            "monte_carlo_runs",
            _candidate_int(candidate, "monte_carlo_samples", 0),
        ),
        "temporal_score": temporal_score,
        "rise_time": rise_time,
        "orthogonality_score": orthogonality_score,
        "cello_assignment_score": cello_assignment_score,
        "cello_buildable": cello_buildable,
        "toxicity": cello_result.details.get("toxicity"),
        "toxicity_score": cello_result.details.get("toxicity_score"),
        "semantic_faithfulness_score": semantic_faithfulness_score,
        "missed_edge_cases": missed_edge_cases,
        "component_scores": component_scores,
        "score_weights": SCORE_WEIGHTS,
        "details": [
            result.details
            | {
                "score": result.score,
                "weight": SCORE_WEIGHTS.get(str(result.details.get("metric", "")), 0.0),
                "metabolic_burden_score": result.metabolic_burden_score,
                "gate_count": result.gate_count,
                "complexity_penalty": result.complexity_penalty,
                "robustness_score": result.robustness_score,
                "signal_to_noise_ratio": result.signal_to_noise_ratio,
                "monte_carlo_runs": result.monte_carlo_runs,
                "temporal_score": result.temporal_score,
                "rise_time": result.rise_time,
                "orthogonality_score": result.orthogonality_score,
                "cello_assignment_score": result.cello_assignment_score,
                "cello_buildable": result.cello_buildable,
                "semantic_faithfulness_score": result.semantic_faithfulness_score,
                "missed_edge_cases": result.missed_edge_cases or [],
            }
            for result in results
        ],
        "scoring_model": "weighted_total_score",
    }


def _grade(score: float) -> str:
    scaled = score * 100.0
    if scaled >= 80.0:
        return "Excellent"
    if scaled >= 60.0:
        return "Pass"
    return "Fail"
