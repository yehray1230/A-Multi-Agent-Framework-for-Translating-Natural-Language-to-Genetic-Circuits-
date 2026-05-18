from __future__ import annotations

import json
import sys
import types
from pathlib import Path

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
from agents.skill_extractor_agent import SkillExtractorAgent
from benchmark_suite.benchmark_controller import evaluate_candidate
from benchmark_suite.cello_constraint_evaluator import evaluate_cello_constraints
from benchmark_suite.functional_scorer import score_functional
from benchmark_suite.metabolic_scorer import MetabolicBurdenEvaluator, count_logic_gates
from benchmark_suite.semantic_evaluator import evaluate_semantic_faithfulness
from benchmark_suite.static_plausibility_evaluator import score_static_plausibility
from benchmark_suite.temporal_scorer import score_temporal
from oracle_evaluator import export_best_verilog
from schemas.state import DesignState, SearchNode
from tools.cello_wrapper import CelloWrapper, _truncate_error_log
from tools.skill_retriever import SkillRetriever
from vector_db import InMemoryVectorDB


def _builder_payload() -> str:
    proposal = {
        "strategy_name": "Gate-Count Optimization",
        "optimization_goal": "minimize gates",
        "truth_table_or_logic_matrix": [{"A": 1, "Y": 1}],
        "logic_blueprint": "Y = A",
        "verilog_draft": "module c(input A, output Y); assign Y = A; endmodule",
        "translator_directives": [],
    }
    return json.dumps(
        {
            "gate_count_optimization": proposal,
            "depth_optimization": {**proposal, "strategy_name": "Depth Optimization"},
            "robustness_strategy": {**proposal, "strategy_name": "Robustness Strategy"},
        }
    )


def test_oracle_exports_best_verilog_file(tmp_path: Path) -> None:
    topology = {"score": 0.9, "verilog": "module winner(input A, output Y); assign Y = A; endmodule"}
    state = DesignState(user_intent="A passthrough", best_topology=topology)

    result = export_best_verilog(state, output_dir=tmp_path)

    assert result["ok"] is True
    assert result["module_name"] == "winner"
    assert Path(result["path"]).read_text(encoding="utf-8") == topology["verilog"] + "\n"
    assert topology["verilog_export_path"] == result["path"]


def test_oracle_returns_error_without_verilog(tmp_path: Path) -> None:
    result = export_best_verilog(DesignState(user_intent="missing"), output_dir=tmp_path)

    assert result["ok"] is False
    assert result["error"]


def test_skill_retriever_loads_json_and_returns_motif_snippets() -> None:
    retriever = SkillRetriever.from_json_file("邏輯設計skill.json")

    xor = retriever.retrieve_skills("xor boolean decomposition", k=2)
    nor = retriever.retrieve_skills("nor promoter gate", k=2)

    assert len(retriever.skills) == 13
    assert "XOR_GATE" in xor
    assert "NOR_GATE" in nor
    assert "Boolean template" in xor


def test_skill_retriever_default_path_is_repo_relative(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    retriever = SkillRetriever.from_json_file()

    assert len(retriever.skills) == 13


def test_core_text_files_are_valid_utf8_without_replacement_characters() -> None:
    for path in [
        Path("README.md"),
        Path("app.py"),
        Path("agents/translator_agent.py"),
        Path("agents/critic_agent.py"),
        Path("tools/skill_retriever.py"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text


def test_skill_retriever_uses_graph_tags_and_mode_pruning() -> None:
    retriever = SkillRetriever(
        skills=[
            {
                "title": "Avoid weak mapping",
                "summary": "Dead-end part choice",
                "tags": ["failure/part-error", "mapping/failed"],
                "confidence_score": 0.9,
                "search_text": "avoid dead-end mapping",
            },
            {
                "title": "Physical tuning",
                "summary": "Use alternate gates for mapping recovery",
                "tags": ["mapping/failed", "mode/exploitation"],
                "confidence_score": 0.7,
                "search_text": "physical mapping recovery",
            },
        ]
    )

    result = retriever.retrieve_skills("alternate mapping", mode="Exploitation", k=1)

    assert "Physical tuning" in result
    assert "Avoid weak mapping" not in result


def test_skill_extractor_writes_obsidian_card_and_vector_record(tmp_path: Path) -> None:
    vector_db = InMemoryVectorDB()
    state = DesignState(user_intent="A and not B", host_organism="E coli")
    state.current_node_id = "root"
    state.tree_nodes["root"] = SearchNode(
        node_id="root",
        status="Pass",
        score=0.84,
        best_topology={"score": 0.84, "gate_count": 2, "ode_status": "simulated"},
        critic_feedbacks=["Design passed with good dynamic margin."],
    )

    result = SkillExtractorAgent(vault_dir=tmp_path, vector_db=vector_db).run(state)

    assert len(result.extracted_skills) == 1
    assert len(vector_db.all()) == 1
    card_path = Path(result.extracted_skills[0]["obsidian_path"])
    assert card_path.exists()
    assert "confidence_score: 0.84" in card_path.read_text(encoding="utf-8")


def test_benchmark_controller_uses_weighted_total_score() -> None:
    result = evaluate_candidate(
        {
            "functional_score": 0.8,
            "kinetic_score": 0.5,
            "plausibility_score": 0.5,
            "robustness_score": 0.77,
            "snr": 4.2,
            "monte_carlo_runs": 6,
            "semantic_faithfulness_score": 0.9,
            "missed_edge_cases": ["B=unknown not specified"],
        }
    )

    assert result["score"] == 0.6315
    assert result["weighted_total_score"] == 0.6315
    assert result["scoring_model"] == "weighted_total_score"
    assert result["grade"] == "Pass"
    assert result["metabolic_burden_score"] == 1.0
    assert result["gate_count"] == 0
    assert result["robustness_score"] == 0.77
    assert result["signal_to_noise_ratio"] == 4.2
    assert result["monte_carlo_runs"] == 6
    assert result["orthogonality_score"] == 0.25
    assert result["cello_assignment_score"] == 0.0
    assert result["cello_buildable"] is False
    assert result["semantic_faithfulness_score"] == 0.9
    assert result["missed_edge_cases"] == ["B=unknown not specified"]
    assert result["score_weights"]["metabolic_burden"] == 0.15
    assert result["score_weights"]["robustness"] == 0.15
    assert result["score_weights"]["temporal"] == 0.05
    assert result["score_weights"]["orthogonality"] == 0.10
    assert result["score_weights"]["cello_assignment"] == 0.10
    assert result["component_scores"]["robustness"] == 0.77
    assert result["component_scores"]["temporal"] == 1.0
    assert result["component_scores"]["orthogonality"] == 0.25


def test_functional_scorer_checks_truth_table_against_verilog() -> None:
    result = score_functional(
        {
            "verilog": "module c(input A, input B, output Y); assign Y = A & ~B; endmodule",
            "truth_table": [
                {"A": 0, "B": 0, "Y": 0},
                {"A": 1, "B": 0, "Y": 1},
                {"A": 1, "B": 1, "Y": 0},
            ],
        }
    )

    assert result.score == 1.0
    assert result.details["logic_compliance_score"] == 1.0
    assert result.details["truth_table_rows_checked"] == 3


def test_functional_scorer_settles_gate_outputs_before_assigns() -> None:
    result = score_functional(
        {
            "verilog": "module c(input A, output Y); wire n; not(n, A); assign Y = n; endmodule",
            "truth_table": [
                {"A": 0, "Y": 1},
                {"A": 1, "Y": 0},
            ],
        }
    )

    assert result.score == 1.0
    assert result.details["logic_failures"] == []


def test_functional_scorer_penalizes_logic_mismatch() -> None:
    result = score_functional(
        {
            "verilog": "module c(input A, input B, output Y); assign Y = A | B; endmodule",
            "truth_table": [
                {"A": 0, "B": 0, "Y": 0},
                {"A": 1, "B": 0, "Y": 1},
                {"A": 0, "B": 1, "Y": 0},
            ],
        }
    )

    assert result.score == 2 / 3
    assert len(result.details["logic_failures"]) == 1


def test_static_plausibility_penalizes_repeated_parts_and_depth() -> None:
    result = score_static_plausibility(
        {
            "part_ids": ["pA", "pA", "pB", "pB"],
            "logic_depth": 7,
        }
    )

    assert 0.0 < result.score < 1.0
    assert result.details["repeated_part_count"] == 2
    assert result.details["logic_depth"] == 7


def test_temporal_scorer_uses_trace_rise_time() -> None:
    result = score_temporal(
        {
            "time": [0.0, 30.0, 60.0, 90.0],
            "output": [0.1, 0.3, 0.55, 0.8],
            "threshold_on": 0.5,
            "target_rise_time": 60.0,
        }
    )

    assert result.rise_time == 60.0
    assert result.temporal_score == 1.0


def test_semantic_evaluator_parses_llm_json(monkeypatch) -> None:
    captured = {}

    def fake_call_llm(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["user_content"] = kwargs["user_content"]
        return json.dumps(
            {
                "score": 0.72,
                "missed_conditions": ["Missing behavior for A=0, B=1"],
            }
        )

    monkeypatch.setattr("benchmark_suite.semantic_evaluator.call_llm", fake_call_llm)

    result = evaluate_semantic_faithfulness(
        {
            "user_intent": "Output Y only when A is high and B is low.",
            "verilog": "module c(input A, input B, output Y); assign Y = A; endmodule",
            "model_name": "mock",
        }
    )

    assert result["semantic_faithfulness_score"] == 0.72
    assert result["missed_edge_cases"] == ["Missing behavior for A=0, B=1"]
    assert "strict test engineer" in captured["system_prompt"]
    assert "Original natural-language prompt" in captured["user_content"]
    assert "Builder/Translator Verilog" in captured["user_content"]


def test_semantic_evaluator_handles_non_json_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmark_suite.semantic_evaluator.call_llm",
        lambda **_: "not json",
    )

    result = evaluate_semantic_faithfulness(
        {
            "user_intent": "Output Y only when A is high.",
            "verilog": "module c(input A, output Y); assign Y = A; endmodule",
            "model_name": "mock",
        }
    )

    assert result["semantic_faithfulness_score"] == 0.0
    assert result["missed_edge_cases"]


def test_cello_constraint_evaluator_parses_json_report(tmp_path: Path) -> None:
    report_path = tmp_path / "cello_report.json"
    report_path.write_text(
        json.dumps({"Gate Assignment Score": 87.5, "Toxicity": 0.12}),
        encoding="utf-8",
    )

    result = evaluate_cello_constraints(
        {
            "mapping_status": "mapped",
            "cello_report_path": str(report_path),
            "cello_buildable": True,
        }
    )

    assert result["status"] == "ok"
    assert result["cello_buildable"] is True
    assert result["cello_assignment_score"] == 0.875
    assert result["orthogonality_score"] == 1.0
    assert result["toxicity"] == 0.12
    assert result["toxicity_score"] == 0.88


def test_cello_constraint_evaluator_penalizes_crosstalk_and_missing_gates() -> None:
    result = evaluate_cello_constraints(
        {
            "mapping_status": "MAPPING_FAILED",
            "raw_error_log": "ERROR: Not enough gates available; severe Crosstalk detected",
            "return_code": 2,
        }
    )

    assert result["status"] == "constraint_failed"
    assert result["cello_buildable"] is False
    assert result["orthogonality_score"] == 0.05
    assert result["cello_assignment_score"] == 0.0


def test_metabolic_scorer_counts_instantiated_logic_gates() -> None:
    verilog = """
    module c(input A, input B, output Y);
      wire n1, n2, n3;
      // and ignored_comment_gate(Y, A, B);
      not(n1, A);
      and g1(n2, A, B);
      NOR u2(n3, n1, n2);
      xor #(.delay(1)) u3(Y, n2, n3);
    endmodule
    """

    result = MetabolicBurdenEvaluator().evaluate({"verilog": verilog})

    assert count_logic_gates(verilog) == 4
    assert result.gate_count == 4
    assert 0.0 < result.metabolic_burden_score < 1.0
    assert result.complexity_penalty == 1.0 - result.metabolic_burden_score


def test_metabolic_scorer_reads_verilog_file(tmp_path: Path) -> None:
    verilog_path = tmp_path / "candidate.v"
    verilog_path.write_text(
        "module c(input A, input B, output Y); nor g1(Y, A, B); endmodule\n",
        encoding="utf-8",
    )

    result = MetabolicBurdenEvaluator().evaluate({"verilog_path": str(verilog_path)})

    assert result.details["status"] == "ok"
    assert result.details["source"] == str(verilog_path)
    assert result.gate_count == 1
    assert result.metabolic_burden_score == 1.0


def test_metabolic_scorer_handles_file_read_failure(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.v"

    result = MetabolicBurdenEvaluator().evaluate({"verilog_path": str(missing_path)})

    assert result.details["status"] == "error"
    assert result.metabolic_burden_score == 0.0
    assert result.gate_count == 0
    assert result.complexity_penalty == 1.0


def test_builder_prompt_includes_retrieved_skills_and_apply_instruction(monkeypatch) -> None:
    captured = {}

    class Retriever:
        def retrieve_skills(self, query: str, mode: str = "Exploration", k: int = 5) -> str:
            return "Motif: XOR_GATE\nBoolean template: A XOR B\nKnown risks: logic hazard"

    def fake_call_llm(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        return _builder_payload()

    monkeypatch.setattr("agents.builder_agent.call_llm", fake_call_llm)

    state = DesignState(user_intent="build xor circuit")
    result = call_builder(state, api_key=None, model_name="mock", skill_retriever=Retriever())

    assert result.last_error is None
    assert "Retrieved Successful Design Skills" in captured["system_prompt"]
    assert "Motif: XOR_GATE" in captured["system_prompt"]
    assert "Please apply the successful skills above" in captured["system_prompt"]


def test_cello_wrapper_mock_mode_is_unchanged() -> None:
    state = DesignState()
    state.verilog_codes = ["module c(input A, output Y); assign Y = A; endmodule"]

    result = CelloWrapper().run(state)

    topology = result.candidate_topologies[0]
    assert topology["source"] == "mock_cello_wrapper"
    assert topology["mapping_status"] == "unmapped"
    assert topology["orthogonality_score"] == 1.0
    assert topology["cello_assignment_score"] == 0.0
    assert topology["cello_buildable"] is False


def test_cello_wrapper_external_failure_becomes_mapping_failed_topology() -> None:
    state = DesignState()
    state.verilog_codes = ["module c(input A, output Y); assign Y = A; endmodule"]
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('UCF constraint mismatch\\njava.lang.RuntimeException: no gate found\\n'); sys.exit(2)",
    ]

    result = CelloWrapper(cello_command=command, timeout_seconds=5).run(state)
    topology = result.candidate_topologies[0]

    assert result.last_error is None
    assert topology["mapping_status"] == "MAPPING_FAILED"
    assert topology["error_type"] == "PART_ERROR"
    assert topology["cello_buildable"] is False
    assert topology["mapping_error_category"] == "UCF_INCOMPATIBLE"
    assert "RuntimeException" in topology["raw_error_log"]


def test_cello_wrapper_external_success_marks_topology_buildable() -> None:
    state = DesignState()
    state.verilog_codes = ["module c(input A, output Y); assign Y = A; endmodule"]
    command = [sys.executable, "-c", "print('Gate Assignment Score: 92\\nToxicity: 0.08')"]

    result = CelloWrapper(cello_command=command, timeout_seconds=5).run(state)
    topology = result.candidate_topologies[0]

    assert topology["mapping_status"] == "mapped"
    assert topology["cello_buildable"] is True
    assert topology["cello_assignment_score"] == 0.92
    assert topology["orthogonality_score"] == 1.0
    assert topology["toxicity"] == 0.08


def test_cello_wrapper_timeout_becomes_mapping_failed_topology() -> None:
    state = DesignState()
    state.verilog_codes = ["module c(input A, output Y); assign Y = A; endmodule"]
    command = [sys.executable, "-c", "import time; time.sleep(2)"]

    result = CelloWrapper(cello_command=command, timeout_seconds=1).run(state)
    topology = result.candidate_topologies[0]

    assert topology["mapping_status"] == "MAPPING_FAILED"
    assert topology["mapping_error_category"] == "TIMEOUT"
    assert "timed out" in topology["mapping_error_summary"]


def test_search_node_syncs_cello_evaluation_metrics() -> None:
    node = SearchNode(node_id="root")

    node.sync_evaluation_metrics(
        {
            "benchmark_report": {
                "orthogonality_score": "0.82",
                "cello_assignment_score": "0.63",
                "cello_buildable": "true",
            }
        }
    )

    assert node.orthogonality_score == 0.82
    assert node.cello_assignment_score == 0.63
    assert node.cello_buildable is True


def test_cello_log_truncation_preserves_head_and_tail() -> None:
    long_log = "ERROR: initial declaration\n" + "\n".join(f"stack frame {i}" for i in range(500)) + "\nFinalException: concrete cause"

    truncated = _truncate_error_log(long_log, max_chars=1000)

    assert "ERROR: initial declaration" in truncated
    assert "FinalException: concrete cause" in truncated
    assert "truncated" in truncated
    assert len(truncated) < len(long_log)
