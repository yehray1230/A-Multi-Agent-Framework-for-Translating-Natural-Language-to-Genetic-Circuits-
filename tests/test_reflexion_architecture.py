from __future__ import annotations

import json
import sys
import types

litellm_stub = types.ModuleType("litellm")
litellm_stub.completion = lambda **_: None
litellm_exceptions_stub = types.ModuleType("litellm.exceptions")
for name in ["AuthenticationError", "RateLimitError", "BadRequestError", "APIError"]:
    setattr(litellm_exceptions_stub, name, type(name, (Exception,), {}))
litellm_caching_stub = types.ModuleType("litellm.caching")
litellm_caching_stub.Cache = lambda **_: object()
litellm_stub.exceptions = litellm_exceptions_stub
sys.modules.setdefault("litellm", litellm_stub)
sys.modules.setdefault("litellm.exceptions", litellm_exceptions_stub)
sys.modules.setdefault("litellm.caching", litellm_caching_stub)

from agents.builder_agent import call_builder
from agents.critic_agent import (
    CELLO_UCF_GUIDANCE,
    SEMANTIC_FAITHFULNESS_GUIDANCE_TEMPLATE,
    CriticAgent,
)
from schemas.state import DesignState, SearchNode
from workflows.reflexion_controller import run_reflexion_workflow


def _builder_payload() -> str:
    proposal = {
        "strategy_name": "Gate-Count Optimization",
        "optimization_goal": "minimize gates",
        "truth_table_or_logic_matrix": [{"A": 1, "B": 0, "Y": 1}],
        "logic_blueprint": "Y = A AND NOT B",
        "verilog_draft": (
            "module genetic_circuit(input A, input B, output Y); "
            "wire not_b; not(not_b, B); and(Y, A, not_b); endmodule"
        ),
        "translator_directives": ["MINIMIZE_GATE_COUNT"],
    }
    payload = {
        "gate_count_optimization": proposal,
        "depth_optimization": {
            **proposal,
            "strategy_name": "Depth Optimization",
            "optimization_goal": "minimize depth",
        },
        "robustness_strategy": {
            **proposal,
            "strategy_name": "Robustness Strategy",
            "optimization_goal": "maximize part compatibility",
        },
    }
    return json.dumps(payload)


def test_state_defaults_include_hitl_and_failed_memory() -> None:
    state = DesignState()
    node = SearchNode(node_id="root")

    assert state.failed_attempts == []
    assert node.failed_attempts == []
    assert state.requires_human_input is False
    assert state.human_feedback_prompt is None
    assert state.pause_reason is None
    assert state.human_constraints == []
    assert node.metabolic_burden_score == 1.0
    assert node.gate_count == 0
    assert node.complexity_penalty == 0.0
    assert node.robustness_score == 1.0
    assert node.signal_to_noise_ratio == 0.0
    assert node.monte_carlo_runs == 0
    assert node.temporal_score == 1.0
    assert node.rise_time is None
    assert node.orthogonality_score == 1.0
    assert node.cello_assignment_score == 0.0
    assert node.cello_buildable is False
    assert node.semantic_faithfulness_score == 1.0
    assert node.missed_edge_cases == []


def test_search_node_syncs_evaluation_metrics_from_topology() -> None:
    node = SearchNode(node_id="root")

    node.sync_evaluation_metrics(
        {
            "score": "0.72",
            "metabolic_burden_score": "0.88",
            "gate_count": "3",
            "complexity_penalty": "0.15",
            "robustness_score": "0.91",
            "snr": "12.5",
            "monte_carlo_runs": "8",
        }
    )

    assert node.score == 0.72
    assert node.metabolic_burden_score == 0.88
    assert node.gate_count == 3
    assert node.complexity_penalty == 0.15
    assert node.robustness_score == 0.91
    assert node.signal_to_noise_ratio == 12.5
    assert node.monte_carlo_runs == 8

    node.sync_evaluation_metrics(
        {
            "benchmark_report": {
                "metabolic_burden_score": "0.66",
                "gate_count": "5",
                "complexity_penalty": "0.34",
                "robustness_score": "0.73",
                "signal_to_noise_ratio": "9.25",
                "monte_carlo_samples": "3",
                "temporal_score": "0.77",
                "rise_time": "140.0",
                "semantic_faithfulness_score": "0.81",
                "missed_conditions": ["missing low input case"],
            }
        }
    )

    assert node.metabolic_burden_score == 0.66
    assert node.gate_count == 5
    assert node.complexity_penalty == 0.34
    assert node.robustness_score == 0.73
    assert node.signal_to_noise_ratio == 9.25
    assert node.monte_carlo_runs == 3
    assert node.temporal_score == 0.77
    assert node.rise_time == 140.0
    assert node.semantic_faithfulness_score == 0.81
    assert node.missed_edge_cases == ["missing low input case"]


def test_builder_accepts_exactly_three_required_strategies(monkeypatch) -> None:
    monkeypatch.setattr("agents.builder_agent.call_llm", lambda **_: _builder_payload())

    state = DesignState(user_intent="A and not B")
    state.tree_nodes["root"] = SearchNode(node_id="root")
    state.current_node_id = "root"

    result = call_builder(state, api_key=None, model_name="mock")

    assert result.last_error is None
    assert len(result.logic_proposals) == 3
    assert len(result.tree_nodes["root"].logic_proposals) == 3
    assert "Gate-Count Optimization" in result.logic_proposals[0]


def test_builder_rejects_missing_strategy(monkeypatch) -> None:
    data = json.loads(_builder_payload())
    data.pop("robustness_strategy")
    monkeypatch.setattr("agents.builder_agent.call_llm", lambda **_: json.dumps(data))

    state = call_builder(DesignState(user_intent="A and not B"), api_key=None, model_name="mock")

    assert state.last_error is not None
    assert "three-strategy JSON schema" in state.last_error


def test_critic_routes_low_functional_score_to_logic_error(monkeypatch) -> None:
    response = {
        "reasoning": "Functional behavior misses the requested truth table.",
        "score": 0.42,
        "benchmark_summary": "functional score is low",
        "is_approved": False,
        "error_type": "LOGIC_ERROR",
        "routing_target": "Builder",
        "recoverable": True,
        "requires_human_input": False,
        "feedback": "Repair the Boolean logic.",
    }
    monkeypatch.setattr("agents.critic_agent.call_llm", lambda **_: json.dumps(response))

    state = DesignState(user_intent="A and not B")
    state.current_node_id = "root"
    state.tree_nodes["root"] = SearchNode(
        node_id="root",
        logic_proposals=["Y = A OR B"],
        best_topology={
            "score": 0.42,
            "benchmark_report": {
                "score": 0.42,
                "details": [{"metric": "functional", "score": 0.2}],
            },
        },
    )

    result = CriticAgent(api_key=None, model_name="mock").run(state)

    node = result.tree_nodes["root"]
    assert node.is_approved is False
    assert node.error_type == "LOGIC_ERROR"
    assert result.error_type == "LOGIC_ERROR"


def test_critic_routes_mapping_failure_to_builder_when_cello_is_not_buildable(monkeypatch) -> None:
    response = {
        "reasoning": "Logic is plausible but mapping failed.",
        "score": 0.55,
        "benchmark_summary": "mapping failed and kinetic score is low",
        "is_approved": False,
        "error_type": "PART_ERROR",
        "routing_target": "Translator",
        "recoverable": True,
        "requires_human_input": False,
        "feedback": "Adjust part constraints and mapping hints.",
    }
    monkeypatch.setattr("agents.critic_agent.call_llm", lambda **_: json.dumps(response))

    state = DesignState(user_intent="A and not B")
    state.current_node_id = "root"
    state.tree_nodes["root"] = SearchNode(
        node_id="root",
        logic_proposals=["Y = A AND NOT B"],
        best_topology={"score": 0.55, "mapping_status": "failed"},
    )

    result = CriticAgent(api_key=None, model_name="mock").run(state)

    assert result.tree_nodes["root"].error_type == "LOGIC_ERROR"
    assert result.error_type == "LOGIC_ERROR"
    assert CELLO_UCF_GUIDANCE in result.tree_nodes["root"].critic_feedbacks[-1]


def test_critic_forces_builder_feedback_for_low_metabolic_burden(monkeypatch) -> None:
    response = {
        "reasoning": "Looks functionally correct.",
        "score": 0.82,
        "benchmark_summary": "weighted score is acceptable but burden is high",
        "is_approved": True,
        "error_type": "NONE",
        "routing_target": "Consolidator",
        "recoverable": True,
        "requires_human_input": False,
        "feedback": "Simplify the implementation.",
    }
    captured = {}

    def fake_call_llm(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["user_content"] = kwargs["user_content"]
        return json.dumps(response)

    monkeypatch.setattr("agents.critic_agent.call_llm", fake_call_llm)

    state = DesignState(user_intent="A and not B")
    state.current_node_id = "root"
    state.tree_nodes["root"] = SearchNode(
        node_id="root",
        logic_proposals=["Y = A AND NOT B"],
        best_topology={
            "score": 0.82,
            "metabolic_burden_score": 0.55,
            "gate_count": 8,
            "complexity_penalty": 0.45,
            "benchmark_report": {
                "score": 0.82,
                "metabolic_burden_score": 0.55,
                "details": [{"metric": "metabolic_burden", "score": 0.55}],
            },
        },
    )

    result = CriticAgent(api_key=None, model_name="mock").run(state)

    feedback = result.tree_nodes["root"].critic_feedbacks[-1]
    assert result.tree_nodes["root"].is_approved is False
    assert result.tree_nodes["root"].error_type == "LOGIC_ERROR"
    assert "metabolic_burden_score" in captured["system_prompt"]
    assert "metabolic_burden_score" in captured["user_content"]
    assert (
        "當前基因電路設計過於複雜，會對宿主細胞造成過高的代謝負擔，"
        "請嘗試使用卡諾圖化簡或合併邏輯閘來精簡 Verilog 代碼"
    ) in feedback


def test_critic_forces_builder_feedback_for_low_robustness(monkeypatch) -> None:
    response = {
        "reasoning": "Functionally correct but dynamics are fragile.",
        "score": 0.83,
        "benchmark_summary": "robustness is low",
        "is_approved": True,
        "error_type": "NONE",
        "routing_target": "Consolidator",
        "recoverable": True,
        "requires_human_input": False,
        "feedback": "Increase dynamic margin.",
    }
    captured = {}

    def fake_call_llm(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["user_content"] = kwargs["user_content"]
        return json.dumps(response)

    monkeypatch.setattr("agents.critic_agent.call_llm", fake_call_llm)

    state = DesignState(user_intent="A and not B")
    state.current_node_id = "root"
    state.tree_nodes["root"] = SearchNode(
        node_id="root",
        logic_proposals=["Y = A AND NOT B"],
        best_topology={
            "score": 0.83,
            "robustness_score": 0.62,
            "signal_to_noise_ratio": 1.4,
            "benchmark_report": {
                "score": 0.83,
                "robustness_score": 0.62,
                "details": [
                    {
                        "metric": "kinetic",
                        "score": 0.62,
                        "robustness_score": 0.62,
                        "collapsed": False,
                    }
                ],
            },
        },
    )

    result = CriticAgent(api_key=None, model_name="mock").run(state)

    feedback = result.tree_nodes["root"].critic_feedbacks[-1]
    assert result.tree_nodes["root"].is_approved is False
    assert result.tree_nodes["root"].error_type == "LOGIC_ERROR"
    assert "robustness_score" in captured["system_prompt"]
    assert "robustness_score" in captured["user_content"]
    assert (
        "動態強健性測試失敗：當前電路在生化參數發生高斯擾動時，ON/OFF 狀態的訊號邊界會模糊甚至重疊。"
        "請考慮：(1) 更換具有更陡峭 Hill Function（非線性更強）的邏輯閘元件，"
        "(2) 確保上游推動下游的訊號餘裕（Margin）足夠大，避免級聯訊號衰減。"
    ) in feedback


def test_critic_forces_builder_feedback_for_signal_overlap(monkeypatch) -> None:
    response = {
        "reasoning": "Aggregate score is high but ON/OFF overlap occurred.",
        "score": 0.84,
        "benchmark_summary": "collapsed noisy trial",
        "is_approved": True,
        "error_type": "NONE",
        "routing_target": "Consolidator",
        "recoverable": True,
        "requires_human_input": False,
        "feedback": "Review dynamic behavior.",
    }
    monkeypatch.setattr("agents.critic_agent.call_llm", lambda **_: json.dumps(response))

    state = DesignState(user_intent="A and not B")
    state.current_node_id = "root"
    state.tree_nodes["root"] = SearchNode(
        node_id="root",
        logic_proposals=["Y = A AND NOT B"],
        best_topology={
            "score": 0.84,
            "robustness_score": 0.9,
            "benchmark_report": {
                "score": 0.84,
                "robustness_score": 0.9,
                "details": [
                    {
                        "metric": "kinetic",
                        "collapsed": True,
                        "min_signal": 12.0,
                        "max_noise": 13.0,
                    }
                ],
            },
        },
    )

    result = CriticAgent(api_key=None, model_name="mock").run(state)

    assert result.tree_nodes["root"].is_approved is False
    assert result.tree_nodes["root"].error_type == "LOGIC_ERROR"
    assert "動態強健性測試失敗" in result.tree_nodes["root"].critic_feedbacks[-1]


def test_critic_forces_builder_feedback_for_cello_ucf_violation(monkeypatch) -> None:
    response = {
        "reasoning": "Cello assignment is not buildable.",
        "score": 0.86,
        "benchmark_summary": "UCF mapping failed",
        "is_approved": True,
        "error_type": "NONE",
        "routing_target": "Consolidator",
        "recoverable": True,
        "requires_human_input": False,
        "feedback": "Try a different mapping.",
    }
    captured = {}

    def fake_call_llm(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["user_content"] = kwargs["user_content"]
        return json.dumps(response)

    monkeypatch.setattr("agents.critic_agent.call_llm", fake_call_llm)

    state = DesignState(user_intent="A and not B")
    state.current_node_id = "root"
    state.tree_nodes["root"] = SearchNode(
        node_id="root",
        logic_proposals=["Y = A AND NOT B"],
        best_topology={
            "score": 0.86,
            "mapping_status": "MAPPING_FAILED",
            "benchmark_report": {
                "score": 0.86,
                "orthogonality_score": 0.05,
                "cello_assignment_score": 0.0,
                "cello_buildable": False,
            },
        },
    )

    result = CriticAgent(api_key=None, model_name="mock").run(state)

    feedback = result.tree_nodes["root"].critic_feedbacks[-1]
    assert result.tree_nodes["root"].is_approved is False
    assert result.tree_nodes["root"].error_type == "LOGIC_ERROR"
    assert "orthogonality_score" in captured["system_prompt"]
    assert "cello_buildable" in captured["user_content"]
    assert CELLO_UCF_GUIDANCE in feedback


def test_critic_forces_precise_feedback_for_low_semantic_faithfulness(monkeypatch) -> None:
    response = {
        "reasoning": "Physical implementation may work but misses part of the prompt.",
        "score": 0.91,
        "benchmark_summary": "semantic coverage is incomplete",
        "is_approved": True,
        "error_type": "NONE",
        "routing_target": "Consolidator",
        "recoverable": True,
        "requires_human_input": False,
        "feedback": "Looks buildable.",
    }
    monkeypatch.setattr("agents.critic_agent.call_llm", lambda **_: json.dumps(response))
    missed_edge_cases = ["A=0,B=1 must force Y=0", "invalid input should fail closed"]

    state = DesignState(user_intent="A and not B, fail closed on invalid input")
    state.current_node_id = "root"
    state.tree_nodes["root"] = SearchNode(
        node_id="root",
        logic_proposals=["Y = A AND NOT B"],
        best_topology={
            "score": 0.91,
            "benchmark_report": {
                "score": 0.91,
                "semantic_faithfulness_score": 0.84,
                "missed_edge_cases": missed_edge_cases,
            },
        },
    )

    result = CriticAgent(api_key=None, model_name="mock").run(state)

    expected_guidance = SEMANTIC_FAITHFULNESS_GUIDANCE_TEMPLATE.format(
        missed_edge_cases="；".join(missed_edge_cases)
    )
    feedback = result.tree_nodes["root"].critic_feedbacks[-1]
    assert result.tree_nodes["root"].is_approved is False
    assert result.tree_nodes["root"].error_type == "LOGIC_ERROR"
    assert result.error_type == "LOGIC_ERROR"
    assert expected_guidance in feedback


class _NoopAgent:
    kwargs: dict = {}

    def run(self, state: DesignState) -> DesignState:
        return state


class _BuilderStub:
    kwargs: dict = {}

    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes[state.current_node_id]
        node.logic_proposals = ["Y = A AND NOT B"]
        state.logic_proposals = node.logic_proposals
        state.last_error = None
        return state


class _TranslatorStub:
    kwargs: dict = {}

    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes[state.current_node_id]
        node.verilog_codes = ["module genetic_circuit(input A, input B, output Y); assign Y = A & ~B; endmodule"]
        state.last_error = None
        return state


class _CelloStub:
    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes[state.current_node_id]
        node.candidate_topologies = [{"score": 0.45, "mapping_status": "mapped"}]
        state.last_error = None
        return state


class _CriticStub:
    def __init__(self, error_type: str):
        self.error_type = error_type

    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes[state.current_node_id]
        node.is_approved = False
        node.error_type = self.error_type
        node.critic_feedbacks.append(f"Route via {self.error_type}")
        state.error_type = self.error_type
        return state


def _run_controller_with(error_type: str) -> DesignState:
    return run_reflexion_workflow(
        DesignState(user_intent="A and not B", compute_budget=1),
        builder=_BuilderStub(),
        translator=_TranslatorStub(),
        cello_wrapper=_CelloStub(),
        batch_ode_simulator=_NoopAgent(),
        critic=_CriticStub(error_type),
        consolidator=_NoopAgent(),
        skill_retriever=None,
    )


def test_controller_budget_pause_records_hitl_state() -> None:
    state = run_reflexion_workflow(
        DesignState(user_intent="A and not B", compute_budget=0),
        builder=_BuilderStub(),
        translator=_TranslatorStub(),
        cello_wrapper=_CelloStub(),
        batch_ode_simulator=_NoopAgent(),
        critic=_CriticStub("LOGIC_ERROR"),
        consolidator=_NoopAgent(),
        skill_retriever=None,
    )

    assert state.requires_human_input is True
    assert state.pause_reason == "compute_budget_exceeded"


def test_controller_logic_error_creates_repair_and_failed_attempt() -> None:
    state = _run_controller_with("LOGIC_ERROR")

    assert state.requires_human_input is True
    assert state.failed_attempts[0]["error_type"] == "LOGIC_ERROR"
    assert any(node.search_mode == "Repair" for node in state.tree_nodes.values())


def test_controller_part_error_creates_exploitation_and_failed_attempt() -> None:
    state = _run_controller_with("PART_ERROR")

    assert state.requires_human_input is True
    assert state.failed_attempts[0]["error_type"] == "PART_ERROR"
    assert any(node.search_mode == "Exploitation" for node in state.tree_nodes.values())
