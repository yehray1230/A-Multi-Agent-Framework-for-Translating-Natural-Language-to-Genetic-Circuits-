from __future__ import annotations

import numpy as np

from benchmark_suite.kinetic_scorer import score_kinetic
from agents.data_miner_agent import DataMinerAgent
from schemas.state import DesignState, SearchNode
from tools.ode_simulator import BatchODESimulator, _perturb_biokinetic_parameters
from utils.unit_conversion import normalize_biokinetic_value
from workflows.reflexion_controller import run_reflexion_workflow


def _state_with_topologies(topologies: list[dict]) -> DesignState:
    state = DesignState(user_intent="A and not B")
    state.current_node_id = "root"
    state.tree_nodes["root"] = SearchNode(node_id="root", candidate_topologies=topologies)
    return state


def test_ode_simulator_marks_topology_simulated_and_outputs_metrics() -> None:
    state = _state_with_topologies([{"verilog": "module c(input A, output Y); assign Y = A; endmodule"}])

    result = BatchODESimulator(simulation_time=120.0, sample_count=24).run(state)
    topology = result.tree_nodes["root"].candidate_topologies[0]

    assert topology["ode_status"] == "simulated"
    assert 0.0 <= topology["kinetic_score"] <= 1.0
    assert 0.0 <= topology["robustness_score"] <= 1.0
    assert topology["signal_to_noise_ratio"] >= 0.0
    assert topology["monte_carlo_runs"] == 1
    assert "metrics_max_burden" in topology
    assert "metrics_cv" in topology
    assert "resource_occupancy" in topology
    assert "benchmark_report" in topology
    assert topology["benchmark_report"]["robustness_score"] == topology["robustness_score"]
    assert topology["benchmark_report"]["signal_to_noise_ratio"] == topology["signal_to_noise_ratio"]
    assert topology["benchmark_report"]["monte_carlo_runs"] == 1
    assert topology["benchmark_report"]["orthogonality_score"] == 1.0
    assert topology["benchmark_report"]["cello_assignment_score"] == 0.0
    assert topology["benchmark_report"]["cello_buildable"] is False


def test_burden_increases_and_score_drops_with_more_genes() -> None:
    state = _state_with_topologies(
        [
            {"gate_count": 1, "score": 0.8},
            {"gate_count": 8, "score": 0.8},
        ]
    )

    result = BatchODESimulator(simulation_time=150.0, sample_count=28).run(state)
    simple, complex_ = result.tree_nodes["root"].candidate_topologies

    assert complex_["metrics_max_burden"] > simple["metrics_max_burden"]
    assert complex_["resource_occupancy"]["rnap_max"] > simple["resource_occupancy"]["rnap_max"]
    assert complex_["score"] < simple["score"]


def test_lower_resource_totals_increase_resource_pressure() -> None:
    plentiful = {
        "gate_count": 5,
        "biokinetic_parameters": {
            "parameters": {
                "rnap_total": {"value": 8000.0},
                "ribosome_total": {"value": 40000.0},
            }
        },
    }
    scarce = {
        "gate_count": 5,
        "biokinetic_parameters": {
            "parameters": {
                "rnap_total": {"value": 900.0},
                "ribosome_total": {"value": 4500.0},
            }
        },
    }
    state = _state_with_topologies([plentiful, scarce])

    result = BatchODESimulator(simulation_time=150.0, sample_count=28).run(state)
    plentiful_result, scarce_result = result.tree_nodes["root"].candidate_topologies

    assert scarce_result["resource_occupancy"]["rnap_free_min"] < plentiful_result["resource_occupancy"]["rnap_free_min"]
    assert scarce_result["resource_occupancy"]["ribosome_free_min"] < plentiful_result["resource_occupancy"]["ribosome_free_min"]
    assert scarce_result["score"] < plentiful_result["score"]


def test_data_miner_writes_normalized_parameters_to_topologies() -> None:
    state = _state_with_topologies([{"gate_count": 2}])

    result = DataMinerAgent().run(state)
    parameters = result.tree_nodes["root"].candidate_topologies[0]["biokinetic_parameters"]["parameters"]

    assert parameters["rnap_total"]["unit"] == "nM"
    assert parameters["translation_rate"]["unit"] == "1/s"
    assert parameters["rnap_total"]["source"] == "conservative_default"
    assert result.tree_nodes["root"].candidate_topologies[0]["biokinetic_parameters"]["mining_summary"]["source_summary"]["conservative_default"] > 0
    assert result.biokinetic_context["unit_system"] == "nM and seconds"
    assert result.biokinetic_context["data_miner_enabled"] is True


def test_data_miner_can_use_vector_records_without_being_vector_retriever() -> None:
    class Retriever:
        def search(self, _query: str, k: int = 5) -> list[dict]:
            return [
                {
                    "parameter": "kd",
                    "value": 0.075,
                    "unit": "uM",
                    "source": "local_bionumbers_record",
                    "confidence": 0.8,
                }
            ][:k]

    state = _state_with_topologies([{"gate_count": 2}])
    result = DataMinerAgent(vector_retriever=Retriever()).run(state)
    kd = result.tree_nodes["root"].candidate_topologies[0]["biokinetic_parameters"]["parameters"]["kd"]

    assert kd["value"] == 75.0
    assert kd["unit"] == "nM"
    assert kd["raw_value"] == 0.075
    assert kd["raw_unit"] == "uM"
    assert kd["source"] == "local_bionumbers_record"


def test_unit_conversion_utility_handles_common_biokinetic_units() -> None:
    assert normalize_biokinetic_value(0.075, "uM").value == 75.0
    assert normalize_biokinetic_value(0.075, "µM").value == 75.0
    assert normalize_biokinetic_value(3.0, "1/min").value == 0.05
    assert normalize_biokinetic_value(120.0, "nM/min").value == 2.0


def test_ode_benchmark_report_includes_parameter_provenance() -> None:
    state = _state_with_topologies([{"gate_count": 2}])
    state = DataMinerAgent().run(state)

    result = BatchODESimulator(simulation_time=120.0, sample_count=24).run(state)
    topology = result.tree_nodes["root"].candidate_topologies[0]

    assert "parameter_provenance" in topology
    assert topology["parameter_provenance"]["source_summary"]["conservative_default"] > 0
    details = topology["benchmark_report"]["details"]
    assert any(detail["metric"] == "parameter_provenance" for detail in details)


def test_ode_simulator_can_run_monte_carlo_stress_test() -> None:
    state = _state_with_topologies([{"gate_count": 2, "score": 0.8}])

    result = BatchODESimulator(
        simulation_time=90.0,
        sample_count=18,
        monte_carlo_samples=3,
        noise_fraction=0.1,
    ).run(state)
    topology = result.tree_nodes["root"].candidate_topologies[0]

    assert topology["monte_carlo_samples"] == 3
    assert topology["monte_carlo_noise_fraction"] == 0.1
    assert "monte_carlo_terminal_output_cv" in topology
    assert any(detail["metric"] == "monte_carlo" for detail in topology["benchmark_report"]["details"])


def test_noise_perturbation_keeps_biochemical_parameters_non_negative() -> None:
    perturbed = _perturb_biokinetic_parameters(
        {
            "transcription_rate": 0.08,
            "translation_rate": 0.045,
            "kd": 50.0,
            "hill_coefficient": 2.0,
            "y_min": 0.02,
            "y_max": 1.0,
        },
        noise_level=10.0,
        rng=np.random.default_rng(7),
    )

    assert all(value >= 0.0 for value in perturbed.values())


def test_kinetic_scorer_runs_noisy_monte_carlo_robustness() -> None:
    result = score_kinetic(
        {
            "verilog": "module c(input A, output Y); assign Y = A; endmodule",
            "gate_count": 1,
            "monte_carlo_runs": 4,
            "noise_level": 0.1,
            "simulation_time": 80.0,
            "sample_count": 16,
        }
    )

    assert result.details["status"] == "ok"
    assert result.monte_carlo_runs == 4
    assert result.signal_to_noise_ratio >= 0.0
    assert 0.0 <= result.robustness_score <= 1.0
    assert result.score == result.robustness_score


class _NoopAgent:
    kwargs: dict = {}

    def run(self, state: DesignState) -> DesignState:
        return state


class _BuilderStub:
    kwargs: dict = {}

    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes[state.current_node_id]
        node.logic_proposals = ["Y = A"]
        state.last_error = None
        return state


class _TranslatorStub:
    kwargs: dict = {}

    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes[state.current_node_id]
        node.verilog_codes = ["module c(input A, output Y); assign Y = A; endmodule"]
        state.last_error = None
        return state


class _CelloStub:
    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes[state.current_node_id]
        node.candidate_topologies = [{"gate_count": 2, "score": 0.7}]
        state.last_error = None
        return state


class _DataMinerRecorder:
    def __init__(self):
        self.called = False

    def run(self, state: DesignState) -> DesignState:
        self.called = True
        node = state.tree_nodes[state.current_node_id]
        for topology in node.candidate_topologies:
            topology["biokinetic_parameters"] = {"parameters": {"rnap_total": {"value": 7000.0}}}
        state.last_error = None
        return state


class _OdeRecorder:
    def __init__(self):
        self.saw_parameters = False

    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes[state.current_node_id]
        self.saw_parameters = "biokinetic_parameters" in node.candidate_topologies[0]
        node.candidate_topologies[0]["score"] = 0.4
        return state


class _CriticStub:
    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes[state.current_node_id]
        node.is_approved = False
        node.error_type = "PART_ERROR"
        node.critic_feedbacks.append("physical route")
        return state


def test_workflow_calls_data_miner_between_cello_and_ode() -> None:
    data_miner = _DataMinerRecorder()
    ode = _OdeRecorder()

    state = run_reflexion_workflow(
        DesignState(user_intent="A", compute_budget=1),
        builder=_BuilderStub(),
        translator=_TranslatorStub(),
        cello_wrapper=_CelloStub(),
        batch_ode_simulator=ode,
        critic=_CriticStub(),
        consolidator=_NoopAgent(),
        skill_retriever=None,
        data_miner=data_miner,
    )

    assert data_miner.called is True
    assert ode.saw_parameters is True
    topology = state.tree_nodes["root"].candidate_topologies[0]
    assert topology["scoring_model"] == "weighted_total_score"
    assert "weighted_total_score" in topology
    assert topology["benchmark_report"]["scoring_model"] == "weighted_total_score"


def test_workflow_remains_compatible_without_data_miner() -> None:
    ode = _OdeRecorder()

    state = run_reflexion_workflow(
        DesignState(user_intent="A", compute_budget=1),
        builder=_BuilderStub(),
        translator=_TranslatorStub(),
        cello_wrapper=_CelloStub(),
        batch_ode_simulator=ode,
        critic=_CriticStub(),
        consolidator=_NoopAgent(),
        skill_retriever=None,
    )

    assert ode.saw_parameters is False
    assert state.failed_attempts[0]["error_type"] == "PART_ERROR"
    assert "orthogonality_score" in state.failed_attempts[0]
    assert "cello_buildable" in state.failed_attempts[0]
