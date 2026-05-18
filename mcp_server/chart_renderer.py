from __future__ import annotations

from pathlib import Path
from typing import Any


def render_score_chart(topology: dict[str, Any] | None, output_path: Path) -> Path | None:
    if not topology:
        return None
    metrics = _score_metrics(topology)
    if not metrics:
        return None
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(labels, values, color=["#32746d", "#7a6f9b", "#c58940", "#4d7ea8", "#8b5e5a"][: len(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Genetic Circuit Score Breakdown")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def render_ode_response_chart(topology: dict[str, Any] | None, output_path: Path) -> Path | None:
    if not topology or topology.get("ode_status") not in {"simulated", "disabled"}:
        return None
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return None

    # The current simulator stores aggregate ODE metrics, not the full trace.
    # This compact response curve makes those aggregates visible to Agent UIs.
    time = np.linspace(0, 600, 80)
    margin = _coerce_float(topology.get("dynamic_margin"), 1.0)
    snr = _coerce_float(topology.get("signal_to_noise_ratio"), 1.0)
    burden = _coerce_float(topology.get("metrics_max_burden"), 1.0)
    normalized_burden = burden / max(burden, 1.0)
    response = margin * (1.0 - np.exp(-time / 160.0))
    noise_floor = np.full_like(time, margin / max(snr, 1.0))
    burden_line = normalized_burden * (1.0 - np.exp(-time / 220.0))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(time, response, label="estimated output response", color="#32746d", linewidth=2)
    ax.plot(time, noise_floor, label="estimated noise floor", color="#8b5e5a", linestyle="--")
    ax.plot(time, burden_line, label="normalized burden", color="#4d7ea8", alpha=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized value")
    ax.set_title("ODE Simulation Summary")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def render_charts(topology: dict[str, Any] | None, run_dir: Path) -> list[Path]:
    charts: list[Path] = []
    score_chart = render_score_chart(topology, run_dir / "score_breakdown.png")
    if score_chart:
        charts.append(score_chart)
    ode_chart = render_ode_response_chart(topology, run_dir / "ode_summary.png")
    if ode_chart:
        charts.append(ode_chart)
    return charts


def _score_metrics(topology: dict[str, Any]) -> dict[str, float]:
    report = topology.get("benchmark_report")
    if not isinstance(report, dict):
        report = {}
    candidates = {
        "overall": topology.get("score", report.get("score")),
        "kinetic": topology.get("kinetic_score"),
        "robustness": topology.get("robustness_score", report.get("robustness_score")),
        "orthogonality": topology.get("orthogonality_score", report.get("orthogonality_score")),
        "assignment": topology.get("cello_assignment_score", report.get("cello_assignment_score")),
    }
    metrics: dict[str, float] = {}
    for key, value in candidates.items():
        number = _coerce_float(value, None)
        if number is not None:
            metrics[key] = max(0.0, min(1.0, number))
    return metrics


def _coerce_float(value: Any, default: float | None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

