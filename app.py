from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict
from typing import Any

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from schemas.state import DesignState, SearchNode


MODE_COLORS = {
    "Exploration": "#2563eb",
    "Repair": "#d97706",
    "Exploitation": "#059669",
}

STATUS_COLORS = {
    "Pending": "#64748b",
    "Evaluated": "#2563eb",
    "Pass": "#059669",
    "Dead_End": "#dc2626",
}

ERROR_COLORS = {
    "NONE": "#059669",
    "LOGIC_ERROR": "#d97706",
    "PART_ERROR": "#7c3aed",
    "BOTH": "#dc2626",
}


def main() -> None:
    if st is None:
        print("Streamlit is not installed. Install requirements and run `streamlit run app.py`.")
        return

    st.set_page_config(page_title="Genetic Circuit Designer", layout="wide")
    _inject_styles()
    _ensure_session_state()

    state = st.session_state.design_state
    _render_sidebar(state)

    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1>Genetic Circuit Designer</h1>
                <p>Natural language to Cello-compatible genetic circuits with tree search, Graph RAG, and critic-guided refinement.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_status_strip(state)

    work_col, inspector_col = st.columns([1.45, 1], gap="large")
    with work_col:
        _render_pipeline(state)
        _render_chart_overview(state)
        _render_tree_workspace(state)

    with inspector_col:
        _render_inspector(state)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
            }
            .app-header {
                border-bottom: 1px solid #e2e8f0;
                margin-bottom: 1rem;
                padding-bottom: 0.75rem;
            }
            .app-header h1 {
                color: #0f172a;
                font-size: 2rem;
                font-weight: 760;
                letter-spacing: 0;
                margin: 0;
            }
            .app-header p {
                color: #475569;
                font-size: 0.95rem;
                margin: 0.25rem 0 0 0;
            }
            .section-title {
                color: #0f172a;
                font-size: 1rem;
                font-weight: 720;
                margin: 1rem 0 0.45rem 0;
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                min-height: 88px;
                padding: 0.85rem 0.95rem;
            }
            .metric-label {
                color: #64748b;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }
            .metric-value {
                color: #0f172a;
                font-size: 1.25rem;
                font-weight: 760;
                line-height: 1.25;
                margin-top: 0.3rem;
                overflow-wrap: anywhere;
            }
            .pill {
                border-radius: 999px;
                color: #ffffff;
                display: inline-block;
                font-size: 0.75rem;
                font-weight: 720;
                line-height: 1;
                padding: 0.38rem 0.58rem;
                white-space: nowrap;
            }
            .step-grid {
                display: grid;
                gap: 0.45rem;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin-bottom: 0.8rem;
            }
            .step {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                min-height: 70px;
                padding: 0.65rem;
            }
            .step.active {
                border-color: #2563eb;
                box-shadow: inset 0 0 0 1px #2563eb;
            }
            .step.done {
                border-color: #059669;
            }
            .step-label {
                color: #0f172a;
                font-size: 0.82rem;
                font-weight: 720;
            }
            .step-caption {
                color: #64748b;
                font-size: 0.74rem;
                margin-top: 0.2rem;
            }
            .empty-state {
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                border-radius: 8px;
                color: #475569;
                padding: 1rem;
            }
            .node-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-bottom: 0.55rem;
                padding: 0.75rem;
            }
            .node-title {
                color: #0f172a;
                font-size: 0.9rem;
                font-weight: 760;
                overflow-wrap: anywhere;
            }
            .node-meta {
                color: #64748b;
                font-size: 0.75rem;
                margin-top: 0.25rem;
            }
            .code-panel {
                background: #0f172a;
                border-radius: 8px;
                color: #e2e8f0;
                font-family: Consolas, monospace;
                font-size: 0.8rem;
                overflow-x: auto;
                padding: 0.8rem;
                white-space: pre;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_session_state() -> None:
    if "design_state" not in st.session_state:
        st.session_state.design_state = DesignState()
    if "selected_node_id" not in st.session_state:
        st.session_state.selected_node_id = None
    if "ui_options" not in st.session_state:
        st.session_state.ui_options = {
            "enable_rag": True,
            "enable_ode": True,
            "enable_tree_search": True,
            "enable_cache": True,
        }
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = {
            "provider": "OpenAI",
            "model_name": "gpt-4o-mini",
            "api_base": "",
            "api_key": "",
        }
    if "run_message" not in st.session_state:
        st.session_state.run_message = None


def _render_sidebar(state: DesignState) -> None:
    with st.sidebar:
        st.header("Design Control")

        state.user_intent = st.text_area(
            "User intent",
            value=state.user_intent,
            height=140,
            placeholder="Example: Design a circuit that turns on GFP only when A is high and B is low.",
        )
        state.host_organism = st.selectbox(
            "Host organism",
            ["Escherichia coli", "Saccharomyces cerevisiae", "Bacillus subtilis", "Custom"],
            index=_safe_index(
                ["Escherichia coli", "Saccharomyces cerevisiae", "Bacillus subtilis", "Custom"],
                state.host_organism,
            ),
        )
        if state.host_organism == "Custom":
            state.host_organism = st.text_input("Custom host", value="Custom host")

        state.compute_budget = st.slider("Compute budget", min_value=1, max_value=20, value=state.compute_budget)

        st.subheader("Workflow Options")
        options = st.session_state.ui_options
        options["enable_rag"] = st.toggle("Graph RAG", value=options["enable_rag"])
        options["enable_ode"] = st.toggle("ODE simulation", value=options["enable_ode"])
        options["enable_tree_search"] = st.toggle("Multi-agent tree search", value=options["enable_tree_search"])
        options["enable_cache"] = st.toggle("Caching", value=options["enable_cache"])

        _render_byok_controls()

        st.subheader("Execution")
        if st.button("Run Demo Iteration", type="primary", use_container_width=True):
            if not state.user_intent.strip():
                st.session_state.run_message = ("warning", "Please enter a design intent before running the workflow.")
            else:
                _run_demo_iteration(state)
            st.rerun()

        if st.button("Run Demo Search", use_container_width=True):
            if not state.user_intent.strip():
                st.session_state.run_message = ("warning", "Please enter a design intent before running the workflow.")
            else:
                if not state.tree_nodes:
                    _run_demo_iteration(state)
                while state.active_frontier and not state.is_completed and state.used_budget < state.compute_budget:
                    _run_demo_iteration(state)
            st.rerun()

        if st.button("Run BYOK Workflow", use_container_width=True):
            _run_byok_workflow(state)
            st.rerun()

        if st.button("Reset", use_container_width=True):
            st.session_state.design_state = DesignState()
            st.session_state.selected_node_id = None
            st.session_state.run_message = None
            st.rerun()

        st.download_button(
            "Export State JSON",
            data=json.dumps(asdict(state), indent=2, ensure_ascii=False),
            file_name="genetic_circuit_design_state.json",
            mime="application/json",
            use_container_width=True,
        )

        _render_run_message()
        st.caption("Demo execution creates deterministic sample nodes. BYOK execution uses your API key only for the current session and does not export it.")


def _render_status_strip(state: DesignState) -> None:
    best_score = _best_score(state)
    active_node = state.current_node_id or "Not started"
    status = "Completed" if state.is_completed else ("Running" if state.tree_nodes else "Ready")
    budget_text = f"{state.used_budget} / {state.compute_budget}"
    error_type = state.error_type
    if state.current_node_id and state.current_node_id in state.tree_nodes:
        error_type = state.tree_nodes[state.current_node_id].error_type

    cols = st.columns(5, gap="small")
    cards = [
        ("Status", status),
        ("Budget", budget_text),
        ("Current Node", active_node),
        ("Best Score", "N/A" if best_score is None else f"{best_score:.2f}"),
        ("Latest Error", error_type),
    ]
    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_byok_controls() -> None:
    config = st.session_state.llm_config
    model_presets = {
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        "OpenRouter": ["openrouter/openai/gpt-4o-mini", "openrouter/anthropic/claude-3.5-sonnet"],
        "Anthropic": ["anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-5-haiku-20241022"],
        "Google": ["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro"],
        "Custom LiteLLM": [config.get("model_name", "custom/model") or "custom/model"],
    }

    with st.expander("BYOK Model Settings", expanded=False):
        provider_names = list(model_presets)
        config["provider"] = st.selectbox(
            "Provider",
            provider_names,
            index=_safe_index(provider_names, config.get("provider")),
        )
        preset_models = model_presets[config["provider"]]
        selected_model = st.selectbox(
            "Model preset",
            preset_models,
            index=_safe_index(preset_models, config.get("model_name")),
        )
        config["model_name"] = st.text_input(
            "LiteLLM model name",
            value=config.get("model_name") or selected_model,
            placeholder=selected_model,
        )
        config["api_key"] = st.text_input(
            "API key",
            value=config.get("api_key", ""),
            type="password",
            placeholder="Paste your provider key for this session",
        )
        config["api_base"] = st.text_input(
            "API base URL",
            value=config.get("api_base", ""),
            placeholder="Optional, for OpenRouter or self-hosted endpoints",
        )


def _render_run_message() -> None:
    message = st.session_state.get("run_message")
    if not message:
        return
    level, text = message
    if level == "success":
        st.success(text)
    elif level == "error":
        st.error(text)
    else:
        st.warning(text)


def _render_pipeline(state: DesignState) -> None:
    st.markdown('<div class="section-title">Workflow Progress</div>', unsafe_allow_html=True)
    current_step = _current_step(state)
    steps = [
        ("Intent", bool(state.user_intent.strip()), "Natural language goal"),
        ("RAG Retrieval", bool(state.rag_context), "Historical rules"),
        ("Builder", bool(state.logic_proposals), "Logic proposals"),
        ("Translator", bool(state.verilog_codes), "Cello Verilog"),
        ("Cello Mapping", bool(state.candidate_topologies), "Topology candidates"),
        ("ODE Simulation", any("ode_status" in topo for topo in state.candidate_topologies), "Dynamic score"),
        ("Critic", bool(state.critic_feedbacks), "Feedback routing"),
        ("Consolidator", state.best_topology is not None, "Best result"),
    ]
    step_html = ['<div class="step-grid">']
    for index, (label, done, caption) in enumerate(steps):
        class_name = "step done" if done else "step"
        if index == current_step:
            class_name += " active"
        step_html.append(
            f"""
            <div class="{class_name}">
                <div class="step-label">{label}</div>
                <div class="step-caption">{caption}</div>
            </div>
            """
        )
    step_html.append("</div>")
    st.markdown("".join(step_html), unsafe_allow_html=True)


def _render_chart_overview(state: DesignState) -> None:
    st.markdown('<div class="section-title">Design Analytics</div>', unsafe_allow_html=True)
    if not state.tree_nodes:
        st.markdown(
            '<div class="empty-state">Charts appear after the workflow produces evaluated nodes or topology candidates.</div>',
            unsafe_allow_html=True,
        )
        return

    score_rows = _node_score_rows(state)
    topology_rows = _topology_chart_rows(state)
    left, right = st.columns(2, gap="medium")
    with left:
        st.caption("Node score progression")
        if pd is not None and score_rows:
            chart_df = pd.DataFrame(score_rows).set_index("node")
            st.line_chart(chart_df[["score"]], use_container_width=True)
        else:
            st.info("No finite node scores yet.")
    with right:
        st.caption("Candidate topology scores")
        if pd is not None and topology_rows:
            chart_df = pd.DataFrame(topology_rows).set_index("candidate")
            st.bar_chart(chart_df[["score"]], use_container_width=True)
        else:
            st.info("No topology scores yet.")


def _render_tree_workspace(state: DesignState) -> None:
    st.markdown('<div class="section-title">Tree Search Workspace</div>', unsafe_allow_html=True)

    if not state.tree_nodes:
        st.markdown(
            '<div class="empty-state">Enter a design intent, then run a demo iteration to create the root search node.</div>',
            unsafe_allow_html=True,
        )
        return

    table_rows = []
    for node in state.tree_nodes.values():
        table_rows.append(
            {
                "Node": node.node_id,
                "Parent": node.parent_id or "-",
                "Mode": node.search_mode,
                "Status": node.status,
                "Score": None if not math.isfinite(node.score) else round(node.score, 3),
                "Error": node.error_type,
                "Children": len(node.children_ids),
            }
        )

    if pd is not None:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.table(table_rows)

    node_ids = list(state.tree_nodes.keys())
    default_node = st.session_state.selected_node_id or state.current_node_id or node_ids[0]
    selected = st.selectbox("Inspect node", node_ids, index=_safe_index(node_ids, default_node))
    st.session_state.selected_node_id = selected

    node = state.tree_nodes[selected]
    mode_color = MODE_COLORS.get(node.search_mode, "#64748b")
    status_color = STATUS_COLORS.get(node.status, "#64748b")
    error_color = ERROR_COLORS.get(node.error_type, "#64748b")
    st.markdown(
        f"""
        <div class="node-card">
            <div class="node-title">{node.node_id}</div>
            <div class="node-meta">
                <span class="pill" style="background:{mode_color};">{node.search_mode}</span>
                <span class="pill" style="background:{status_color};">{node.status}</span>
                <span class="pill" style="background:{error_color};">{node.error_type}</span>
            </div>
            <div class="node-meta">Parent: {node.parent_id or "None"} | Children: {", ".join(node.children_ids) or "None"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_inspector(state: DesignState) -> None:
    st.markdown('<div class="section-title">Result Inspector</div>', unsafe_allow_html=True)
    node = _selected_node(state)

    if node is None:
        st.markdown(
            '<div class="empty-state">No node selected yet. Run the workflow to inspect proposals, Verilog, topology, and critic feedback.</div>',
            unsafe_allow_html=True,
        )
        return

    proposal_tab, verilog_tab, topology_tab, charts_tab, critic_tab, rag_tab, raw_tab = st.tabs(
        ["Proposals", "Verilog", "Topology", "Charts", "Critic", "RAG Context", "Raw State"]
    )

    with proposal_tab:
        proposals = node.logic_proposals or state.logic_proposals
        if proposals:
            for index, proposal in enumerate(proposals, start=1):
                with st.expander(f"Proposal {index}", expanded=index == 1):
                    _render_json_or_text(proposal)
        else:
            st.info("No logic proposals generated for this node yet.")

    with verilog_tab:
        codes = node.verilog_codes or state.verilog_codes
        if codes:
            selected_code = st.radio("Verilog candidate", [f"Candidate {i + 1}" for i in range(len(codes))], horizontal=True)
            code_index = int(selected_code.split()[-1]) - 1
            st.markdown(f'<div class="code-panel">{_escape_html(codes[code_index])}</div>', unsafe_allow_html=True)
        else:
            st.info("No Verilog generated for this node yet.")

    with topology_tab:
        topologies = node.candidate_topologies or state.candidate_topologies
        if topologies:
            rows = [
                {
                    key: value
                    for key, value in topology.items()
                    if key not in {"verilog"}
                }
                for topology in topologies
            ]
            if pd is not None:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.table(rows)
            st.subheader("Best topology")
            st.json(node.best_topology or state.best_topology or {})
        else:
            st.info("No topology candidates available yet.")

    with charts_tab:
        _render_topology_charts(node, state)

    with critic_tab:
        cols = st.columns(3)
        cols[0].metric("Approved", "Yes" if node.is_approved else "No")
        cols[1].metric("Error Type", node.error_type)
        cols[2].metric("Score", "N/A" if not math.isfinite(node.score) else f"{node.score:.2f}")
        if node.critic_feedbacks:
            for feedback in node.critic_feedbacks:
                st.warning(feedback)
        elif state.critic_feedbacks:
            for feedback in state.critic_feedbacks:
                st.warning(feedback)
        else:
            st.info("No critic feedback yet.")
        if node.last_error:
            st.error(node.last_error)

    with rag_tab:
        if state.rag_context:
            st.text_area("Retrieved context", value=state.rag_context, height=260)
        else:
            st.info("No RAG context retrieved yet.")

    with raw_tab:
        st.json(asdict(node))
        with st.expander("Full DesignState"):
            st.json(asdict(state))


def _render_topology_charts(node: SearchNode, state: DesignState) -> None:
    if pd is None:
        st.info("Install pandas to enable chart rendering in Streamlit.")
        return

    topologies = node.candidate_topologies or state.candidate_topologies
    if not topologies:
        st.info("No topology candidates available for charts yet.")
        return

    chart_df = pd.DataFrame(_topology_rows(topologies)).set_index("candidate")
    st.caption("Score by topology candidate")
    st.bar_chart(chart_df[["score"]], use_container_width=True)

    metric_cols = [column for column in ["gate_count", "dynamic_margin"] if column in chart_df.columns]
    if metric_cols:
        st.caption("Implementation complexity and dynamic margin")
        st.line_chart(chart_df[metric_cols], use_container_width=True)


def _run_demo_iteration(state: DesignState) -> None:
    options = st.session_state.ui_options
    if not state.tree_nodes:
        root = SearchNode(node_id="root", search_mode="Exploration")
        state.tree_nodes[root.node_id] = root
        state.active_frontier = [root.node_id]

    if not state.active_frontier or state.used_budget >= state.compute_budget:
        _select_best_fallback(state)
        return

    current_node_id = state.active_frontier.pop(0)
    node = state.tree_nodes[current_node_id]
    state.current_node_id = current_node_id

    mode = node.search_mode
    if options["enable_rag"] and mode in {"Exploration", "Repair"}:
        state.rag_context = _demo_rag_context(state.user_intent, mode)

    if mode != "Exploitation":
        node.logic_proposals = _demo_proposals(state, node)
    elif not node.logic_proposals:
        node.logic_proposals = state.logic_proposals[:]

    node.current_topology = node.logic_proposals[0] if node.logic_proposals else ""
    state.logic_proposals = node.logic_proposals[:]
    state.current_topology = node.current_topology

    node.verilog_codes = [_demo_verilog(index, proposal, mode) for index, proposal in enumerate(node.logic_proposals)]
    state.verilog_codes = node.verilog_codes[:]

    node.candidate_topologies = _demo_topologies(node, options["enable_ode"])
    state.candidate_topologies = node.candidate_topologies[:]

    best_topology = max(node.candidate_topologies, key=lambda item: float(item.get("score", -9999)), default=None)
    node.best_topology = best_topology
    node.score = float(best_topology.get("score", -9999)) if best_topology else -float("inf")
    node.sync_evaluation_metrics(best_topology)
    state.best_topology = best_topology

    _demo_critic_and_branch(state, node, options["enable_tree_search"])
    node.status = "Pass" if node.is_approved else "Evaluated"
    state.error_type = node.error_type
    state.is_approved = node.is_approved
    state.critic_feedbacks = node.critic_feedbacks[:]
    state.last_error = None
    state.iteration_count += 1
    st.session_state.selected_node_id = node.node_id

    if not state.is_completed and not state.active_frontier and state.used_budget >= state.compute_budget:
        _select_best_fallback(state)


def _run_byok_workflow(state: DesignState) -> None:
    config = st.session_state.llm_config
    if not state.user_intent.strip():
        st.session_state.run_message = ("warning", "Please enter a design intent before running the BYOK workflow.")
        return
    if not config.get("api_key", "").strip():
        st.session_state.run_message = ("warning", "Please enter an API key in BYOK Model Settings.")
        return
    if not config.get("model_name", "").strip():
        st.session_state.run_message = ("warning", "Please choose or enter a LiteLLM model name.")
        return

    try:
        from agents.builder_agent import BuilderAgent
        from agents.consolidator_agent import ConsolidatorAgent
        from agents.critic_agent import CriticAgent
        from agents.data_miner_agent import DataMinerAgent
        from agents.skill_extractor_agent import SkillExtractorAgent
        from agents.translator_agent import call_translator
        from tools.cello_wrapper import CelloWrapper
        from tools.ode_simulator import BatchODESimulator
        from tools.skill_retriever import SkillRetriever
        from utils import llm_utils
        from vector_db import InMemoryVectorDB
        from workflows.reflexion_controller import run_reflexion_workflow
    except Exception as exc:
        st.session_state.run_message = ("error", f"Could not load workflow components: {exc}")
        return

    class TranslatorRunner:
        def __init__(self, api_key: str, model_name: str, api_base: str | None):
            self.api_key = api_key
            self.model_name = model_name
            self.api_base = api_base
            self.kwargs: dict[str, Any] = {}

        def run(self, workflow_state: DesignState) -> DesignState:
            return call_translator(
                workflow_state,
                api_key=self.api_key,
                model_name=self.model_name,
                api_base=self.api_base,
                **self.kwargs,
            )

    options = st.session_state.ui_options
    api_key = config["api_key"].strip()
    model_name = config["model_name"].strip()
    api_base = config.get("api_base", "").strip() or None

    llm_utils.ENABLE_LLM_CACHE = bool(options["enable_cache"])
    state.last_error = None
    try:
        result_state = run_reflexion_workflow(
            state=state,
            builder=BuilderAgent(api_key=api_key, model_name=model_name, api_base=api_base),
            translator=TranslatorRunner(api_key=api_key, model_name=model_name, api_base=api_base),
            cello_wrapper=CelloWrapper(),
            batch_ode_simulator=BatchODESimulator() if options["enable_ode"] else _NoOpODESimulator(),
            critic=CriticAgent(api_key=api_key, model_name=model_name, api_base=api_base),
            consolidator=ConsolidatorAgent(),
            skill_retriever=SkillRetriever.from_json_file() if options["enable_rag"] else None,
            data_miner=DataMinerAgent() if options["enable_ode"] else None,
            skill_extractor=SkillExtractorAgent(vault_dir="outputs/obsidian_skills", vector_db=InMemoryVectorDB()),
        )
    except Exception as exc:
        state.last_error = f"ERROR: BYOK workflow failed: {exc}"
        st.session_state.run_message = ("error", state.last_error)
        return

    st.session_state.design_state = result_state
    st.session_state.selected_node_id = result_state.current_node_id
    if result_state.last_error:
        st.session_state.run_message = ("error", result_state.last_error)
    else:
        st.session_state.run_message = ("success", "BYOK workflow completed. Inspect the generated nodes, Verilog, topology, and charts.")


class _NoOpODESimulator:
    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes.get(state.current_node_id) if state.current_node_id else None
        topologies = node.candidate_topologies if node else state.candidate_topologies
        for topology in topologies:
            topology["ode_status"] = "disabled"
        return state


def _demo_proposals(state: DesignState, node: SearchNode) -> list[str]:
    base_intent = state.user_intent.strip() or "Design a genetic logic circuit"
    repair_hint = " incorporating critic feedback" if node.search_mode == "Repair" else ""
    blueprints = [
        ("proposal_a", "Minimize biological part cost", "Y = A AND NOT B", 2, 3, ["PRESERVE_SIMPLE_GATES"]),
        ("proposal_b", "Minimize logic depth and delay", "Y = A OR B", 1, 4, []),
        ("proposal_c", "Prioritize robustness under noisy inputs", "Y = (A AND NOT B) OR (A AND C)", 3, 5, ["USE_STRUCTURAL_INSTANTIATION"]),
    ]
    proposals = []
    for key, strategy, blueprint, depth, cost, directives in blueprints:
        proposals.append(
            json.dumps(
                {
                    "id": key,
                    "strategy_description": f"{strategy}{repair_hint} for: {base_intent}",
                    "total_logic_depth": depth,
                    "total_repressor_cost": cost,
                    "logic_blueprint": blueprint,
                    "translator_directives": directives,
                },
                ensure_ascii=False,
            )
        )
    return proposals


def _demo_verilog(index: int, proposal: str, mode: str) -> str:
    module_name = f"genetic_circuit_{mode.lower()}_{index + 1}"
    if index == 1:
        return f"""module {module_name}(input A, input B, output Y);
  assign Y = A | B;
endmodule"""
    if index == 2:
        return f"""module {module_name}(input A, input B, input C, output Y);
  wire not_b;
  wire arm_a;
  wire arm_b;
  not(not_b, B);
  and(arm_a, A, not_b);
  and(arm_b, A, C);
  or(Y, arm_a, arm_b);
endmodule"""
    return f"""module {module_name}(input A, input B, output Y);
  wire not_b;
  not(not_b, B);
  and(Y, A, not_b);
endmodule"""


def _demo_topologies(node: SearchNode, enable_ode: bool) -> list[dict[str, Any]]:
    mode_bonus = {"Exploration": 0.0, "Repair": 0.08, "Exploitation": 0.14}.get(node.search_mode, 0.0)
    topologies = []
    for index, code in enumerate(node.verilog_codes):
        score = min(0.96, 0.58 + mode_bonus + index * 0.08 + len(node.critic_feedbacks) * 0.03)
        topology = {
            "source": "demo_cello_wrapper",
            "verilog_index": index,
            "mapping_status": "mapped",
            "gate_count": 2 + index,
            "score": round(score, 3),
            "verilog": code,
        }
        if enable_ode:
            topology["ode_status"] = "simulated"
            topology["dynamic_margin"] = round(0.31 + index * 0.07 + mode_bonus, 3)
        else:
            topology["ode_status"] = "disabled"
        topologies.append(topology)
    return topologies


def _demo_critic_and_branch(state: DesignState, node: SearchNode, enable_tree_search: bool) -> None:
    approved = node.score >= 0.82 or state.used_budget >= state.compute_budget - 1
    node.is_approved = approved

    if approved:
        node.error_type = "NONE"
        node.critic_feedbacks.append("Design passes the demo threshold. Consolidate this topology as the current best result.")
        state.is_completed = True
        return

    state.used_budget += 1
    if node.search_mode == "Exploration":
        node.error_type = "LOGIC_ERROR"
        node.critic_feedbacks.append("Logic is plausible but underspecified. Add a repair branch that preserves the intent and tightens Boolean behavior.")
    elif node.search_mode == "Repair":
        node.error_type = "PART_ERROR"
        node.critic_feedbacks.append("Logic is now acceptable, but part mapping can improve. Try exploitation without changing the architecture.")
    else:
        node.error_type = "PART_ERROR"
        node.critic_feedbacks.append("Part assignment still needs tuning. Keep the best-scoring topology as fallback.")

    if not enable_tree_search:
        return

    if node.error_type in {"LOGIC_ERROR", "BOTH"}:
        repair_id = _child_id(node.node_id, "repair")
        repair_node = SearchNode(
            node_id=repair_id,
            parent_id=node.node_id,
            search_mode="Repair",
            critic_feedbacks=node.critic_feedbacks[:],
            error_type=node.error_type,
        )
        node.children_ids.append(repair_id)
        state.tree_nodes[repair_id] = repair_node
        state.active_frontier.append(repair_id)

        if state.used_budget < state.compute_budget - 1:
            explore_id = _child_id(node.node_id, "explore")
            explore_node = SearchNode(
                node_id=explore_id,
                parent_id=node.node_id,
                search_mode="Exploration",
                critic_feedbacks=node.critic_feedbacks[:],
                error_type=node.error_type,
            )
            node.children_ids.append(explore_id)
            state.tree_nodes[explore_id] = explore_node
            state.active_frontier.append(explore_id)
    elif node.error_type == "PART_ERROR":
        exploit_id = _child_id(node.node_id, "exploit")
        exploit_node = SearchNode(
            node_id=exploit_id,
            parent_id=node.node_id,
            search_mode="Exploitation",
            logic_proposals=node.logic_proposals[:],
            critic_feedbacks=node.critic_feedbacks[:],
            error_type=node.error_type,
        )
        node.children_ids.append(exploit_id)
        state.tree_nodes[exploit_id] = exploit_node
        state.active_frontier.append(exploit_id)


def _select_best_fallback(state: DesignState) -> None:
    best_node = None
    for node in state.tree_nodes.values():
        if not node.best_topology:
            continue
        if best_node is None or node.score > best_node.score:
            best_node = node
    if best_node:
        state.current_node_id = best_node.node_id
        state.best_topology = best_node.best_topology
        state.logic_proposals = best_node.logic_proposals[:]
        state.verilog_codes = best_node.verilog_codes[:]
        state.candidate_topologies = best_node.candidate_topologies[:]
        state.error_type = best_node.error_type
        state.critic_feedbacks = best_node.critic_feedbacks[:]
        st.session_state.selected_node_id = best_node.node_id


def _demo_rag_context(intent: str, mode: str) -> str:
    return "\n".join(
        [
            f"Mode-aware retrieval: {mode}",
            "Prefer Cello-compatible combinational logic: primitive gates, wire, assign.",
            "Avoid always blocks, registers, clocks, latches, memories, and delay syntax.",
            f"Intent keywords: {', '.join(intent.lower().split()[:8]) or 'not provided'}",
        ]
    )


def _node_score_rows(state: DesignState) -> list[dict[str, Any]]:
    rows = []
    for index, node in enumerate(state.tree_nodes.values(), start=1):
        if not math.isfinite(node.score):
            continue
        rows.append(
            {
                "node": f"{index}. {node.node_id}",
                "score": round(float(node.score), 3),
            }
        )
    return rows


def _topology_chart_rows(state: DesignState) -> list[dict[str, Any]]:
    node = _selected_node(state)
    topologies = []
    if node is not None:
        topologies = node.candidate_topologies
    if not topologies:
        topologies = state.candidate_topologies
    return _topology_rows(topologies)


def _topology_rows(topologies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, topology in enumerate(topologies):
        candidate = f"Candidate {int(topology.get('verilog_index', index)) + 1}"
        rows.append(
            {
                "candidate": candidate,
                "score": float(topology.get("score", 0.0)),
                "gate_count": int(topology.get("gate_count", 0)),
                "dynamic_margin": float(topology.get("dynamic_margin", 0.0)),
            }
        )
    return rows


def _current_step(state: DesignState) -> int:
    checks = [
        bool(state.user_intent.strip()),
        bool(state.rag_context),
        bool(state.logic_proposals),
        bool(state.verilog_codes),
        bool(state.candidate_topologies),
        any("ode_status" in topo for topo in state.candidate_topologies),
        bool(state.critic_feedbacks),
        state.best_topology is not None,
    ]
    for index, done in enumerate(checks):
        if not done:
            return max(0, index - 1)
    return len(checks) - 1


def _selected_node(state: DesignState) -> SearchNode | None:
    selected = st.session_state.selected_node_id or state.current_node_id
    if selected and selected in state.tree_nodes:
        return state.tree_nodes[selected]
    if state.tree_nodes:
        return next(iter(state.tree_nodes.values()))
    return None


def _best_score(state: DesignState) -> float | None:
    scores = [node.score for node in state.tree_nodes.values() if math.isfinite(node.score)]
    return max(scores) if scores else None


def _render_json_or_text(value: str) -> None:
    try:
        st.json(json.loads(value))
    except Exception:
        st.write(value)


def _safe_index(items: list[str], value: str | None) -> int:
    try:
        return items.index(value or "")
    except ValueError:
        return 0


def _child_id(parent_id: str, mode: str) -> str:
    return f"{parent_id}_{mode}_{uuid.uuid4().hex[:4]}"


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
