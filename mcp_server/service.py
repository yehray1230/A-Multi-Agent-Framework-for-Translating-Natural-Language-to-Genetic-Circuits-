from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agents.consolidator_agent import ConsolidatorAgent
from agents.data_miner_agent import DataMinerAgent
from benchmark_suite.benchmark_controller import evaluate_candidate
from mcp_server.artifact_writer import create_run_dir, write_json, write_state_artifacts
from mcp_server.chart_renderer import render_charts
from mcp_server.run_store import RunStore
from mcp_server.serializers import summarize_state, summarize_topology
from schemas.state import DesignState, SearchNode
from tools.cello_wrapper import CelloWrapper
from tools.ode_simulator import BatchODESimulator
from vector_db import InMemoryVectorDB
from workflows.reflexion_controller import run_reflexion_workflow


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_RUN_STORE = RunStore()


@dataclass
class WorkflowOptions:
    enable_rag: bool = True
    enable_ode: bool = True
    enable_skill_extraction: bool = True
    compute_budget: int = 2
    monte_carlo_samples: int = 1
    output_dir: str | None = None
    cello_command: str | None = None
    ucf_path: str | None = None
    model_name: str | None = None
    api_base: str | None = None
    api_key: str | None = None


class TranslatorRunner:
    def __init__(self, api_key: str | None, model_name: str, api_base: str | None):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        self.kwargs: dict[str, Any] = {}

    def run(self, state: DesignState) -> DesignState:
        from agents.translator_agent import call_translator

        return call_translator(
            state,
            api_key=self.api_key,
            model_name=self.model_name,
            api_base=self.api_base,
            **self.kwargs,
        )


class NoOpODESimulator:
    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes.get(state.current_node_id) if state.current_node_id else None
        topologies = node.candidate_topologies if node else state.candidate_topologies
        for topology in topologies:
            topology["ode_status"] = "disabled"
        state.candidate_topologies = topologies
        return state


def design_circuit_quick(
    user_intent: str,
    host_organism: str = "Escherichia coli",
    compute_budget: int = 2,
    enable_rag: bool = True,
    enable_ode: bool = True,
    enable_skill_extraction: bool = True,
    monte_carlo_samples: int = 1,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    output_dir: str | None = None,
    cello_command: str | None = None,
    ucf_path: str | None = None,
) -> dict[str, Any]:
    options = WorkflowOptions(
        enable_rag=enable_rag,
        enable_ode=enable_ode,
        enable_skill_extraction=enable_skill_extraction,
        compute_budget=compute_budget,
        monte_carlo_samples=monte_carlo_samples,
        output_dir=output_dir,
        cello_command=cello_command,
        ucf_path=ucf_path,
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
    )
    resolved_model = _resolve_model(options.model_name)
    resolved_api_key = _resolve_api_key(options.api_key)
    resolved_api_base = options.api_base or os.getenv("LITELLM_API_BASE") or None

    if not user_intent.strip():
        return {"status": "error", "error": "user_intent is required."}
    if not resolved_model:
        return {"status": "error", "error": "model_name is required via argument or LITELLM_MODEL."}

    try:
        from agents.builder_agent import BuilderAgent
        from agents.critic_agent import CriticAgent
        from agents.skill_extractor_agent import SkillExtractorAgent
        from tools.skill_retriever import SkillRetriever
    except ModuleNotFoundError as exc:
        return {
            "status": "error",
            "error": (
                "design_circuit_quick requires the LLM workflow dependencies. "
                f"Missing module: {exc.name}"
            ),
        }

    state = DesignState(
        user_intent=user_intent.strip(),
        host_organism=host_organism.strip() or "Escherichia coli",
        compute_budget=max(1, int(options.compute_budget)),
    )

    batch_ode_simulator = (
        BatchODESimulator(monte_carlo_samples=max(1, int(options.monte_carlo_samples)))
        if options.enable_ode
        else NoOpODESimulator()
    )
    skill_retriever = SkillRetriever.from_json_file() if options.enable_rag else None
    skill_extractor = (
        SkillExtractorAgent(vault_dir="outputs/obsidian_skills", vector_db=InMemoryVectorDB())
        if options.enable_skill_extraction
        else None
    )

    try:
        result_state = run_reflexion_workflow(
            state=state,
            builder=BuilderAgent(api_key=resolved_api_key, model_name=resolved_model, api_base=resolved_api_base),
            translator=TranslatorRunner(api_key=resolved_api_key, model_name=resolved_model, api_base=resolved_api_base),
            cello_wrapper=CelloWrapper(cello_command=options.cello_command, ucf_path=options.ucf_path),
            batch_ode_simulator=batch_ode_simulator,
            critic=CriticAgent(api_key=resolved_api_key, model_name=resolved_model, api_base=resolved_api_base),
            consolidator=ConsolidatorAgent(),
            skill_retriever=skill_retriever,
            data_miner=DataMinerAgent() if options.enable_ode else None,
            skill_extractor=skill_extractor,
        )
    except Exception as exc:
        return {"status": "error", "error": f"workflow failed: {exc}"}

    run_dir = create_run_dir(options.output_dir)
    charts = render_charts(result_state.best_topology, run_dir)
    artifacts = write_state_artifacts(result_state, run_dir, charts)
    summary = summarize_state(result_state)
    return {
        "status": _status_from_state(result_state),
        "run_dir": str(run_dir.resolve()),
        "summary": summary,
        "artifacts": artifacts,
    }


def start_design_run(
    user_intent: str,
    host_organism: str = "Escherichia coli",
    compute_budget: int = 6,
    enable_rag: bool = True,
    enable_ode: bool = True,
    enable_skill_extraction: bool = True,
    monte_carlo_samples: int = 1,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    output_dir: str | None = None,
    cello_command: str | None = None,
    ucf_path: str | None = None,
    run_store: RunStore | None = None,
) -> dict[str, Any]:
    request = {
        "user_intent": user_intent,
        "host_organism": host_organism,
        "compute_budget": compute_budget,
        "enable_rag": enable_rag,
        "enable_ode": enable_ode,
        "enable_skill_extraction": enable_skill_extraction,
        "monte_carlo_samples": monte_carlo_samples,
        "model_name": model_name,
        "api_base": api_base,
        "api_key": api_key,
        "output_dir": output_dir,
        "cello_command": cello_command,
        "ucf_path": ucf_path,
    }
    selected_store = run_store or DEFAULT_RUN_STORE
    return selected_store.start(
        task=lambda: design_circuit_quick(
            user_intent=user_intent,
            host_organism=host_organism,
            compute_budget=compute_budget,
            enable_rag=enable_rag,
            enable_ode=enable_ode,
            enable_skill_extraction=enable_skill_extraction,
            monte_carlo_samples=monte_carlo_samples,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            output_dir=output_dir,
            cello_command=cello_command,
            ucf_path=ucf_path,
        ),
        request=request,
    )


def get_design_run_status(run_id: str, run_store: RunStore | None = None) -> dict[str, Any]:
    selected_store = run_store or DEFAULT_RUN_STORE
    return selected_store.status(run_id)


def get_design_run_result(run_id: str, run_store: RunStore | None = None) -> dict[str, Any]:
    selected_store = run_store or DEFAULT_RUN_STORE
    return selected_store.result(run_id)


def evaluate_verilog(
    verilog: str,
    user_intent: str = "Evaluate a Cello-compatible genetic circuit.",
    host_organism: str = "Escherichia coli",
    enable_ode: bool = True,
    monte_carlo_samples: int = 1,
    output_dir: str | None = None,
    cello_command: str | None = None,
    ucf_path: str | None = None,
) -> dict[str, Any]:
    if not verilog.strip():
        return {"status": "error", "error": "verilog is required."}

    state = DesignState(user_intent=user_intent, host_organism=host_organism, compute_budget=1)
    node = SearchNode(node_id="root", search_mode="Exploration")
    node.verilog_codes = [verilog.strip()]
    state.tree_nodes["root"] = node
    state.current_node_id = "root"
    state.verilog_codes = [verilog.strip()]

    try:
        state = CelloWrapper(cello_command=cello_command, ucf_path=ucf_path).run(state)
        if state.last_error:
            return {"status": "error", "error": state.last_error}
        state = DataMinerAgent().run(state) if enable_ode else state
        state = (
            BatchODESimulator(monte_carlo_samples=max(1, int(monte_carlo_samples))).run(state)
            if enable_ode
            else NoOpODESimulator().run(state)
        )
        for topology in node.candidate_topologies:
            topology.update(evaluate_candidate(topology))
        best_topology = max(node.candidate_topologies, key=lambda item: float(item.get("score", -9999)), default=None)
        node.best_topology = best_topology
        node.sync_evaluation_metrics(best_topology)
        state.best_topology = best_topology
    except Exception as exc:
        return {"status": "error", "error": f"evaluation failed: {exc}"}

    run_dir = create_run_dir(output_dir)
    charts = render_charts(state.best_topology, run_dir)
    artifacts = write_state_artifacts(state, run_dir, charts)
    write_json(run_dir / "input_verilog.json", {"verilog": verilog})
    return {
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "summary": summarize_state(state),
        "best_topology": summarize_topology(state.best_topology),
        "artifacts": artifacts,
    }


def summarize_design_state(state_json: dict[str, Any]) -> dict[str, Any]:
    """Summarize a saved state JSON produced by this adapter."""
    return {
        "status": "completed",
        "summary": {
            "user_intent": state_json.get("user_intent"),
            "host_organism": state_json.get("host_organism"),
            "is_completed": state_json.get("is_completed"),
            "is_approved": state_json.get("is_approved"),
            "requires_human_input": state_json.get("requires_human_input"),
            "pause_reason": state_json.get("pause_reason"),
            "current_node_id": state_json.get("current_node_id"),
            "best_topology": summarize_topology(state_json.get("best_topology")),
        },
    }


def _resolve_model(model_name: str | None) -> str:
    return model_name or os.getenv("LITELLM_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


def _resolve_api_key(api_key: str | None) -> str | None:
    return api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None


def _status_from_state(state: DesignState) -> str:
    if state.last_error:
        return "error"
    if state.requires_human_input:
        return "needs_human_input"
    if state.is_completed or state.best_topology:
        return "completed"
    return "stopped"
