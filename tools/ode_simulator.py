from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ModuleNotFoundError:
    solve_ivp = None

try:
    from joblib import Memory
except ModuleNotFoundError:
    Memory = None

from agents.data_miner_agent import DEFAULT_BIOKINETIC_PARAMETERS
from schemas.state import DesignState


@dataclass
class WarmStartResourceSolver:
    rnap_free: float
    ribosome_free: float
    fixed_point_iterations: int = 5
    bisection_iterations: int = 18
    tolerance: float = 1e-6

    def solve_rnap(self, total: float, demands: np.ndarray, km: float) -> tuple[float, float]:
        free = self._solve(total, demands, km, self.rnap_free)
        self.rnap_free = free
        occupancy = 1.0 - free / max(total, 1e-9)
        return free, _clamp01(occupancy)

    def solve_ribosome(self, total: float, mrna: np.ndarray, km: float) -> tuple[float, float]:
        free = self._solve(total, np.maximum(mrna, 0.0), km, self.ribosome_free)
        self.ribosome_free = free
        occupancy = 1.0 - free / max(total, 1e-9)
        return free, _clamp01(occupancy)

    def _solve(self, total: float, demands: np.ndarray, km: float, warm_start: float) -> float:
        total = max(float(total), 1e-9)
        km = max(float(km), 1e-9)
        x = min(max(float(warm_start), 0.0), total)

        for _ in range(self.fixed_point_iterations):
            f_value, derivative = self._constraint(x, total, demands, km)
            if abs(f_value) <= self.tolerance * total:
                return x
            step = f_value / max(derivative, 1e-9)
            x = min(max(x - step, 0.0), total)

        lo, hi = 0.0, total
        for _ in range(self.bisection_iterations):
            mid = 0.5 * (lo + hi)
            f_mid, _ = self._constraint(mid, total, demands, km)
            if f_mid > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _constraint(x: float, total: float, demands: np.ndarray, km: float) -> tuple[float, float]:
        denom = km + x
        bound = float(np.sum(demands * x / denom))
        derivative = 1.0 + float(np.sum(demands * km / (denom * denom)))
        return x + bound - total, derivative


@dataclass
class ResourceAwareSimulation:
    gene_count: int
    params: dict[str, float]
    solver: WarmStartResourceSolver
    resource_trace: list[dict[str, float]] = field(default_factory=list)

    def rhs(self, _time: float, y: np.ndarray) -> np.ndarray:
        n = self.gene_count
        mrna = np.maximum(y[:n], 0.0)
        protein = np.maximum(y[n:], 0.0)

        promoter_demand = np.full(n, self.params["promoter_resource_demand"])
        rnap_free, rnap_occupancy = self.solver.solve_rnap(
            self.params["rnap_total"], promoter_demand, self.params["km_rnap"]
        )
        ribo_free, ribo_occupancy = self.solver.solve_ribosome(
            self.params["ribosome_total"], mrna, self.params["km_ribosome"]
        )

        rnap_factor = rnap_free / (self.params["km_rnap"] + rnap_free)
        ribo_factor = ribo_free / (self.params["km_ribosome"] + ribo_free)
        regulation = self._hill_regulation(protein)

        d_mrna = (
            self.params["transcription_rate"] * regulation * rnap_factor
            - self.params["mrna_degradation_rate"] * mrna
        )
        d_protein = (
            self.params["translation_rate"] * mrna * ribo_factor
            - self.params["protein_degradation_rate"] * protein
        )

        self.resource_trace.append(
            {
                "rnap_free": rnap_free,
                "ribosome_free": ribo_free,
                "rnap_occupancy": rnap_occupancy,
                "ribosome_occupancy": ribo_occupancy,
                "burden_nM": float(np.sum(mrna) + np.sum(protein)),
            }
        )
        return np.concatenate([d_mrna, d_protein])

    def _hill_regulation(self, protein: np.ndarray) -> np.ndarray:
        n = self.gene_count
        regulation = np.ones(n)
        if n <= 1:
            return regulation
        kd = max(self.params["kd"], 1e-9)
        hill = max(self.params["hill_coefficient"], 1.0)
        leak = _clamp01(self.params["leak_fraction"])
        repressors = protein[:-1]
        regulation[1:] = leak + (1.0 - leak) / (1.0 + np.power(repressors / kd, hill))
        return regulation


class BatchODESimulator:
    def __init__(
        self,
        simulation_time: float = 600.0,
        sample_count: int = 80,
        monte_carlo_samples: int = 1,
        noise_fraction: float = 0.15,
        noise_level: float | None = None,
        cache_dir: str | None = None,
    ):
        self.simulation_time = simulation_time
        self.sample_count = sample_count
        self.monte_carlo_samples = max(1, int(monte_carlo_samples))
        selected_noise = noise_fraction if noise_level is None else noise_level
        self.noise_fraction = max(0.0, float(selected_noise))
        self._memory = Memory(cache_dir, verbose=0) if cache_dir and Memory is not None else None
        self._local_cache: dict[tuple, dict[str, Any]] = {}

    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes.get(state.current_node_id) if state.current_node_id else None
        topologies = node.candidate_topologies if node else state.candidate_topologies
        for index, topology in enumerate(topologies):
            self._simulate_topology(topology, index)
        if node:
            node.candidate_topologies = topologies
        state.candidate_topologies = topologies
        return state

    def _simulate_topology(self, topology: dict[str, Any], index: int) -> None:
        gene_count = _infer_gene_count(topology)
        biokinetic_parameters = topology.get("biokinetic_parameters", {})
        params = _flatten_parameters(biokinetic_parameters)
        parameter_provenance = _parameter_provenance(biokinetic_parameters)
        params["promoter_resource_demand"] = max(1.0, params["km_rnap"] * 0.35)
        physical_assignment_metrics = _physical_assignment_metrics(topology)

        cache_key = _cache_key(
            topology,
            index,
            gene_count,
            params,
            self.simulation_time,
            self.sample_count,
            self.monte_carlo_samples,
            self.noise_fraction,
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            topology.update(deepcopy(cached))
            return

        t_eval = np.linspace(0.0, self.simulation_time, self.sample_count)
        simulation, result = self._run_single_simulation(gene_count, params, t_eval)

        if result is None or not result.success:
            updates = {
                "ode_status": "failed",
                "kinetic_score": 0.0,
                "robustness_score": 0.0,
                "signal_to_noise_ratio": 0.0,
                "monte_carlo_runs": self.monte_carlo_samples,
                "score": 0.0,
                "benchmark_report": {
                    "score": 0.0,
                    "robustness_score": 0.0,
                    "signal_to_noise_ratio": 0.0,
                    "monte_carlo_runs": self.monte_carlo_samples,
                    **physical_assignment_metrics,
                    "details": [{"metric": "kinetic", "status": "ode_failed"}],
                },
            }
            topology.update(updates)
            self._set_cached(cache_key, updates)
            return

        metrics = _simulation_metrics(result.y, simulation.resource_trace, params)
        if self.monte_carlo_samples > 1:
            metrics.update(self._monte_carlo_metrics(gene_count, params, t_eval))
        kinetic_score = _kinetic_score(metrics)
        robustness_score = kinetic_score
        base_score = float(topology.get("score", 0.65 + index * 0.02))
        final_score = max(0.0, min(1.0, 0.35 * base_score + 0.65 * kinetic_score))
        resource_occupancy = {
            "rnap_max": metrics["rnap_occupancy_max"],
            "ribosome_max": metrics["ribosome_occupancy_max"],
            "rnap_free_min": metrics["rnap_free_min"],
            "ribosome_free_min": metrics["ribosome_free_min"],
        }

        details = [
            {"metric": "kinetic", "score": kinetic_score},
            {"metric": "robustness", "score": robustness_score},
            {"metric": "signal_to_noise_ratio", "value": metrics["signal_to_noise_ratio"]},
            {"metric": "max_burden", "value": metrics["max_burden_nM"], "unit": "nM"},
            {"metric": "output_cv", "value": metrics["output_cv"]},
            {"metric": "resource_occupancy", "value": resource_occupancy},
            {"metric": "parameter_provenance", "value": parameter_provenance},
        ]
        if self.monte_carlo_samples > 1:
            details.append(
                {
                    "metric": "monte_carlo",
                    "samples": self.monte_carlo_samples,
                    "noise_fraction": self.noise_fraction,
                    "terminal_output_cv": metrics["monte_carlo_terminal_output_cv"],
                    "failure_rate": metrics["monte_carlo_failure_rate"],
                }
            )
        updates = {
            "ode_status": "simulated",
            "gene_count": gene_count,
            "kinetic_score": kinetic_score,
            "robustness_score": robustness_score,
            "signal_to_noise_ratio": metrics["signal_to_noise_ratio"],
            "monte_carlo_runs": self.monte_carlo_samples,
            "score": final_score,
            "metrics_max_burden": metrics["max_burden_nM"],
            "metrics_cv": metrics["output_cv"],
            "dynamic_margin": metrics["dynamic_margin"],
            "resource_occupancy": resource_occupancy,
            "parameter_provenance": parameter_provenance,
            "monte_carlo_samples": self.monte_carlo_samples,
            "monte_carlo_noise_fraction": self.noise_fraction,
            "benchmark_report": {
                "score": final_score,
                "robustness_score": robustness_score,
                "signal_to_noise_ratio": metrics["signal_to_noise_ratio"],
                "monte_carlo_runs": self.monte_carlo_samples,
                **physical_assignment_metrics,
                "details": details,
            },
        }
        if self.monte_carlo_samples > 1:
            updates["monte_carlo_failure_rate"] = metrics["monte_carlo_failure_rate"]
            updates["monte_carlo_terminal_output_cv"] = metrics["monte_carlo_terminal_output_cv"]
        topology.update(updates)
        self._set_cached(cache_key, updates)

    def _run_single_simulation(
        self,
        gene_count: int,
        params: dict[str, float],
        t_eval: np.ndarray,
    ) -> tuple[ResourceAwareSimulation, Any]:
        solver = WarmStartResourceSolver(
            rnap_free=params["rnap_total"],
            ribosome_free=params["ribosome_total"],
        )
        simulation = ResourceAwareSimulation(gene_count=gene_count, params=params, solver=solver)
        initial_state = np.zeros(gene_count * 2)
        return simulation, self._integrate(simulation, initial_state, t_eval)

    def simulate_noisy_response(
        self,
        topology: dict[str, Any],
        noise_level: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float | bool | str]:
        gene_count = _infer_gene_count(topology)
        params = _flatten_parameters(topology.get("biokinetic_parameters", {}))
        params["promoter_resource_demand"] = max(1.0, params["km_rnap"] * 0.35)
        sample_params = _perturb_biokinetic_parameters(
            params,
            self.noise_fraction if noise_level is None else max(0.0, float(noise_level)),
            rng or np.random.default_rng(),
        )
        t_eval = np.linspace(0.0, self.simulation_time, self.sample_count)
        simulation, result = self._run_single_simulation(gene_count, sample_params, t_eval)
        if result is None or not result.success:
            return {
                "success": False,
                "on_value": 0.0,
                "off_value": float("inf"),
                "error": getattr(result, "message", "ODE simulation failed."),
            }
        protein = np.maximum(result.y[result.y.shape[0] // 2 :, :], 0.0)
        output = protein[-1, :] if protein.size else np.zeros(result.y.shape[1])
        off_window = max(1, min(len(output), int(math.ceil(len(output) * 0.1))))
        on_value = float(output[-1]) if len(output) else 0.0
        off_value = float(np.max(output[:off_window])) if len(output) else 0.0
        metrics = _simulation_metrics(result.y, simulation.resource_trace, sample_params)
        return {
            "success": True,
            "on_value": on_value,
            "off_value": off_value,
            "signal_to_noise_ratio": metrics["signal_to_noise_ratio"],
        }

    def _monte_carlo_metrics(
        self,
        gene_count: int,
        params: dict[str, float],
        t_eval: np.ndarray,
    ) -> dict[str, float]:
        rng = np.random.default_rng(_stable_seed(gene_count, params, self.simulation_time, self.sample_count))
        terminal_outputs: list[float] = []
        failures = 0
        perturbable = [
            "transcription_rate",
            "translation_rate",
            "kd",
            "hill_coefficient",
            "leak_fraction",
            "mrna_degradation_rate",
            "protein_degradation_rate",
            "y_min",
            "ymax",
            "y_max",
        ]
        for _ in range(self.monte_carlo_samples):
            sample_params = _perturb_biokinetic_parameters(
                params,
                self.noise_fraction,
                rng,
                perturbable,
            )
            simulation, result = self._run_single_simulation(gene_count, sample_params, t_eval)
            if result is None or not result.success:
                failures += 1
                continue
            protein = np.maximum(result.y[result.y.shape[0] // 2 :, :], 0.0)
            terminal_outputs.append(float(protein[-1, -1]) if protein.size else 0.0)
        mean_output = float(np.mean(terminal_outputs)) if terminal_outputs else 0.0
        cv = float(np.std(terminal_outputs)) / max(mean_output, 1e-9) if terminal_outputs else 1.0
        return {
            "monte_carlo_terminal_output_cv": cv,
            "monte_carlo_failure_rate": failures / max(1, self.monte_carlo_samples),
        }

    def _integrate(
        self,
        simulation: ResourceAwareSimulation,
        initial_state: np.ndarray,
        t_eval: np.ndarray,
    ) -> Any:
        if solve_ivp is not None:
            for method in ("BDF", "Radau"):
                try:
                    result = solve_ivp(
                        simulation.rhs,
                        (0.0, self.simulation_time),
                        initial_state,
                        method=method,
                        t_eval=t_eval,
                        rtol=1e-5,
                        atol=1e-8,
                    )
                except Exception:
                    result = None
                if result is not None and result.success:
                    return result
        return _rk4_integrate(simulation.rhs, initial_state, t_eval)

    def _get_cached(self, key: tuple) -> dict[str, Any] | None:
        if self._memory is None:
            return deepcopy(self._local_cache.get(key))
        return _cached_lookup(self._memory, key, self._local_cache)

    def _set_cached(self, key: tuple, value: dict[str, Any]) -> None:
        self._local_cache[key] = deepcopy(value)


def _flatten_parameters(raw: dict[str, Any]) -> dict[str, float]:
    params = {key: float(value["value"]) for key, value in DEFAULT_BIOKINETIC_PARAMETERS.items()}
    records = raw.get("parameters", raw) if isinstance(raw, dict) else {}
    for key, value in records.items():
        if isinstance(value, dict) and "value" in value:
            try:
                params[key] = float(value["value"])
            except (TypeError, ValueError):
                continue
        else:
            try:
                params[key] = float(value)
            except (TypeError, ValueError):
                continue
    return params


def _parameter_provenance(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"source_summary": {"conservative_default": len(DEFAULT_BIOKINETIC_PARAMETERS)}, "unit_system": "nM and seconds"}
    summary = raw.get("mining_summary", {})
    if isinstance(summary, dict) and summary:
        return {
            "source_summary": summary.get("source_summary", {}),
            "records_used": summary.get("records_used", []),
            "all_parameters_have_external_source": summary.get("all_parameters_have_external_source", False),
            "unit_system": summary.get("unit_system", "nM and seconds"),
        }
    parameters = raw.get("parameters", raw)
    source_summary: dict[str, int] = {}
    if isinstance(parameters, dict):
        for parameter in parameters.values():
            source = parameter.get("source", "unknown") if isinstance(parameter, dict) else "unknown"
            source_summary[str(source)] = source_summary.get(str(source), 0) + 1
    return {"source_summary": source_summary, "unit_system": "nM and seconds"}


def _perturb_biokinetic_parameters(
    params: dict[str, float],
    noise_level: float,
    rng: np.random.Generator,
    perturbable: list[str] | None = None,
) -> dict[str, float]:
    sample_params = params.copy()
    keys = perturbable or [
        "transcription_rate",
        "translation_rate",
        "kd",
        "hill_coefficient",
        "leak_fraction",
        "mrna_degradation_rate",
        "protein_degradation_rate",
        "y_min",
        "ymax",
        "y_max",
    ]
    for key in keys:
        if key not in sample_params:
            continue
        original = float(sample_params[key])
        sigma = abs(original) * max(0.0, float(noise_level))
        sample_params[key] = max(0.0, float(rng.normal(original, sigma)))
    return sample_params


def _rk4_integrate(rhs, initial_state: np.ndarray, t_eval: np.ndarray) -> Any:
    y = np.zeros((len(initial_state), len(t_eval)))
    y[:, 0] = initial_state
    current = initial_state.copy()
    for index in range(1, len(t_eval)):
        t0 = float(t_eval[index - 1])
        dt = float(t_eval[index] - t_eval[index - 1])
        k1 = rhs(t0, current)
        k2 = rhs(t0 + 0.5 * dt, np.maximum(current + 0.5 * dt * k1, 0.0))
        k3 = rhs(t0 + 0.5 * dt, np.maximum(current + 0.5 * dt * k2, 0.0))
        k4 = rhs(t0 + dt, np.maximum(current + dt * k3, 0.0))
        current = np.maximum(current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
        y[:, index] = current
    return SimpleNamespace(success=True, y=y, message="internal RK4 fallback")


def _infer_gene_count(topology: dict[str, Any]) -> int:
    if topology.get("biokinetic_parameters", {}).get("gene_count"):
        return max(1, int(topology["biokinetic_parameters"]["gene_count"]))
    if topology.get("gate_count"):
        return max(1, int(topology["gate_count"]))
    verilog = str(topology.get("verilog") or "")
    primitive_count = len(re.findall(r"\b(and|or|not|nand|nor|xor|xnor)\s*\(", verilog))
    assign_count = len(re.findall(r"\bassign\b", verilog))
    return max(1, primitive_count + assign_count)


def _simulation_metrics(y: np.ndarray, trace: list[dict[str, float]], params: dict[str, float]) -> dict[str, float]:
    protein = np.maximum(y[y.shape[0] // 2 :, :], 0.0)
    output = protein[-1, :] if protein.size else np.zeros(y.shape[1])
    output_mean = float(np.mean(output))
    output_std = float(np.std(output))
    burden_values = [entry["burden_nM"] for entry in trace] or [0.0]
    rnap_occupancies = [entry["rnap_occupancy"] for entry in trace] or [0.0]
    ribo_occupancies = [entry["ribosome_occupancy"] for entry in trace] or [0.0]
    rnap_free = [entry["rnap_free"] for entry in trace] or [params["rnap_total"]]
    ribo_free = [entry["ribosome_free"] for entry in trace] or [params["ribosome_total"]]
    max_burden = float(max(burden_values))
    dynamic_margin = output_mean / (1.0 + float(np.max(protein[:-1, :])) if protein.shape[0] > 1 else 1.0)
    return {
        "max_burden_nM": max_burden,
        "output_cv": output_std / max(output_mean, 1e-9),
        "signal_to_noise_ratio": output_mean / max(output_std, 1e-9),
        "dynamic_margin": dynamic_margin,
        "rnap_occupancy_max": float(max(rnap_occupancies)),
        "ribosome_occupancy_max": float(max(ribo_occupancies)),
        "rnap_free_min": float(min(rnap_free)),
        "ribosome_free_min": float(min(ribo_free)),
        "resource_capacity_factor": min(
            1.0,
            0.5 * params["rnap_total"] / DEFAULT_BIOKINETIC_PARAMETERS["rnap_total"]["value"]
            + 0.5 * params["ribosome_total"] / DEFAULT_BIOKINETIC_PARAMETERS["ribosome_total"]["value"],
        ),
        "burden_penalty": _sigmoid_penalty(max_burden, params["burden_soft_limit"], 0.00018),
        "toxicity_penalty": _sigmoid_penalty(max_burden, params["toxicity_threshold"], 0.00022),
    }


def _kinetic_score(metrics: dict[str, float]) -> float:
    stability = 1.0 / (1.0 + metrics["output_cv"])
    if "monte_carlo_terminal_output_cv" in metrics:
        stability *= 1.0 / (1.0 + metrics["monte_carlo_terminal_output_cv"])
    margin = _clamp01(metrics["dynamic_margin"] / 80.0)
    resource_penalty = 1.0 - 0.5 * (metrics["rnap_occupancy_max"] + metrics["ribosome_occupancy_max"])
    failure_penalty = 1.0 - metrics.get("monte_carlo_failure_rate", 0.0)
    score = (
        0.25 * stability
        + 0.20 * margin
        + 0.25 * metrics["burden_penalty"]
        + 0.20 * metrics["toxicity_penalty"]
        + 0.10 * _clamp01(resource_penalty)
    )
    return _clamp01(score * failure_penalty * (0.35 + 0.65 * metrics["resource_capacity_factor"]))


def _sigmoid_penalty(value: float, soft_limit: float, steepness: float) -> float:
    exponent = steepness * (value - soft_limit)
    exponent = min(60.0, max(-60.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _coerce_float(value: Any, default: float) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
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


def _physical_assignment_metrics(topology: dict[str, Any]) -> dict[str, float | bool]:
    benchmark_report = topology.get("benchmark_report")
    if not isinstance(benchmark_report, dict):
        benchmark_report = {}
    return {
        "orthogonality_score": _coerce_float(
            topology.get(
                "orthogonality_score",
                benchmark_report.get("orthogonality_score"),
            ),
            1.0,
        ),
        "cello_assignment_score": _coerce_float(
            topology.get(
                "cello_assignment_score",
                benchmark_report.get("cello_assignment_score"),
            ),
            0.0,
        ),
        "cello_buildable": _coerce_bool(
            topology.get("cello_buildable", benchmark_report.get("cello_buildable")),
            False,
        ),
        "toxicity": _coerce_float(
            topology.get("toxicity", benchmark_report.get("toxicity")),
            0.0,
        ),
        "toxicity_score": _coerce_float(
            topology.get("toxicity_score", benchmark_report.get("toxicity_score")),
            1.0,
        ),
    }


def _cache_key(
    topology: dict[str, Any],
    index: int,
    gene_count: int,
    params: dict[str, float],
    simulation_time: float,
    sample_count: int,
    monte_carlo_samples: int,
    noise_fraction: float,
) -> tuple:
    return (
        str(topology.get("verilog", "")),
        int(topology.get("gate_count", gene_count)),
        index,
        tuple(sorted((key, round(float(value), 9)) for key, value in params.items())),
        tuple(sorted(_physical_assignment_metrics(topology).items())),
        round(float(simulation_time), 6),
        int(sample_count),
        int(monte_carlo_samples),
        round(float(noise_fraction), 6),
    )


def _stable_seed(gene_count: int, params: dict[str, float], simulation_time: float, sample_count: int) -> int:
    seed_text = repr(
        (
            gene_count,
            tuple(sorted((key, round(float(value), 6)) for key, value in params.items())),
            round(float(simulation_time), 6),
            sample_count,
        )
    )
    return abs(hash(seed_text)) % (2**32)


def _cached_lookup(_memory, key: tuple, local_cache: dict[tuple, dict[str, Any]]) -> dict[str, Any] | None:
    # The Memory object is retained so callers can opt into a joblib-backed cache
    # directory later without changing the public constructor. The current cache
    # is intentionally conservative and stores only process-local simulation
    # payloads, avoiding stale scientific results across code changes.
    return deepcopy(local_cache.get(key))
