from __future__ import annotations

import json
import math
import re
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

MODE_LABELS = {
    "Exploration": "探索",
    "Repair": "修正",
    "Exploitation": "最佳化",
}

STATUS_COLORS = {
    "Pending": "#64748b",
    "Evaluated": "#2563eb",
    "Pass": "#059669",
    "Dead_End": "#dc2626",
    "Needs_Human_Input": "#ea580c",
}

STATUS_LABELS = {
    "Pending": "待處理",
    "Evaluated": "已評估",
    "Pass": "通過",
    "Dead_End": "無可行分支",
    "Needs_Human_Input": "需要人工輸入",
}

ERROR_COLORS = {
    "NONE": "#059669",
    "LOGIC_ERROR": "#d97706",
    "PART_ERROR": "#7c3aed",
    "BOTH": "#dc2626",
}

ERROR_LABELS = {
    "NONE": "無",
    "LOGIC_ERROR": "邏輯問題",
    "PART_ERROR": "元件問題",
    "BOTH": "邏輯與元件問題",
}


def main() -> None:
    if st is None:
        print("尚未安裝 Streamlit。請先安裝 requirements，然後執行 `streamlit run app.py`。")
        return

    st.set_page_config(page_title="基因電路設計器", layout="wide")
    _inject_styles()
    _ensure_session_state()
    _render_tutorial()

    state = st.session_state.design_state
    _render_sidebar(state)

    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1>基因電路設計器</h1>
                <p>將自然語言需求轉換為 Cello 相容基因電路，整合樹狀搜尋、圖譜 RAG 與評審回饋修正。</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_status_strip(state)
    _render_human_loop_panel(state)

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
            .topology-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-bottom: 0.85rem;
                padding: 0.85rem;
            }
            .topology-card.best {
                border-color: #059669;
                box-shadow: inset 0 0 0 1px #059669;
            }
            .topology-title {
                color: #0f172a;
                font-size: 0.95rem;
                font-weight: 760;
            }
            .topology-subtitle {
                color: #64748b;
                font-size: 0.75rem;
                margin-top: 0.2rem;
            }
            .topology-metrics {
                display: grid;
                gap: 0.4rem;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin-top: 0.7rem;
            }
            .topology-metric {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                min-height: 58px;
                padding: 0.5rem;
            }
            .topology-metric-label {
                color: #64748b;
                font-size: 0.68rem;
                font-weight: 700;
                text-transform: uppercase;
            }
            .topology-metric-value {
                color: #0f172a;
                font-size: 0.88rem;
                font-weight: 760;
                margin-top: 0.18rem;
                overflow-wrap: anywhere;
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
            .hitl-panel {
                background: #fff7ed;
                border: 1px solid #fed7aa;
                border-radius: 8px;
                margin: 0.9rem 0 1rem 0;
                padding: 0.95rem;
            }
            .hitl-panel h2 {
                color: #9a3412;
                font-size: 1rem;
                font-weight: 760;
                letter-spacing: 0;
                margin: 0 0 0.35rem 0;
            }
            .hitl-panel p {
                color: #7c2d12;
                font-size: 0.88rem;
                margin: 0.25rem 0;
            }
            .hitl-meta {
                color: #9a3412;
                font-size: 0.78rem;
                font-weight: 700;
                margin-top: 0.55rem;
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
            "model_name": "gpt-5.4-mini",
            "api_base": "",
            "api_key": "",
        }
    if "run_message" not in st.session_state:
        st.session_state.run_message = None
    if "show_tutorial" not in st.session_state:
        st.session_state.show_tutorial = False


def _render_tutorial() -> None:
    if not st.session_state.get("show_tutorial", False):
        return

    if hasattr(st, "dialog"):
        @st.dialog("📖 系統使用導覽")
        def tutorial_dialog():
            st.markdown(
                """
                ### 歡迎使用基因電路設計器！
                這是一個將自然語言轉換為基因電路的自動化工具。以下是簡單的使用步驟：
                
                1. **輸入需求**：在左側的「設計需求」框中，用自然語言描述想要的基因電路功能（例如：A 和 B 同時存在時輸出 Y）。
                2. **設定參數**：選擇宿主生物、調整計算預算，並開關 RAG、ODE 模擬等功能。
                3. **執行生成**：
                   - **示範模式**：點擊「執行示範迭代」或「執行示範搜尋」，體驗系統流程。
                   - **自備金鑰**：於「自備金鑰模型設定」輸入 API Key 後，點擊「執行自備金鑰工作流程」。
                4. **檢視與分析**：在右側「結果檢視器」切換分頁，查看邏輯提案、Verilog、拓樸與評審回饋。
                5. **下載狀態**：點擊左側底部的「匯出狀態 JSON」保存您的設計進度。
                """
            )
            if st.button("開始使用", use_container_width=True):
                st.session_state.show_tutorial = False
                st.rerun()
        tutorial_dialog()
    else:
        st.info(
            "**📖 系統使用導覽**\n\n"
            "歡迎使用基因電路設計器！以下是簡單的使用步驟：\n\n"
            "1. **輸入需求**：在左側的「設計需求」框中，用自然語言描述想要的基因電路功能。\n"
            "2. **設定參數**：選擇宿主生物、調整計算預算，並開關 RAG、ODE 模擬等功能。\n"
            "3. **執行生成**：點擊「執行示範迭代」體驗系統流程，或於設定 API Key 後「執行自備金鑰工作流程」。\n"
            "4. **檢視與分析**：在右側「結果檢視器」切換分頁，查看邏輯提案、Verilog、拓樸與評審回饋。\n"
            "5. **下載狀態**：點擊左側底部的「匯出狀態 JSON」保存您的設計進度。"
        )
        if st.button("我知道了", key="close_tutorial_inline"):
            st.session_state.show_tutorial = False
            st.rerun()


def _render_sidebar(state: DesignState) -> None:
    with st.sidebar:
        st.header("設計控制")

        st.button(
            "📖 使用導覽",
            use_container_width=True,
            on_click=lambda: st.session_state.update(show_tutorial=True),
        )

        state.user_intent = st.text_area(
            "設計需求",
            value=state.user_intent,
            height=140,
            placeholder="範例：設計一個只有在 A 高、B 低時才啟動 GFP 的基因電路。",
        )
        host_options = ["Escherichia coli", "Saccharomyces cerevisiae", "Bacillus subtilis", "自訂"]
        state.host_organism = st.selectbox(
            "宿主生物",
            host_options,
            index=_safe_index(host_options, state.host_organism),
        )
        if state.host_organism == "自訂":
            state.host_organism = st.text_input("自訂宿主", value="自訂宿主")

        state.compute_budget = st.slider("計算預算", min_value=1, max_value=20, value=state.compute_budget)

        st.subheader("工作流程選項")
        options = st.session_state.ui_options
        options["enable_rag"] = st.toggle("圖譜 RAG", value=options["enable_rag"])
        options["enable_ode"] = st.toggle("ODE 模擬", value=options["enable_ode"])
        options["enable_tree_search"] = st.toggle("多代理樹狀搜尋", value=options["enable_tree_search"])
        options["enable_cache"] = st.toggle("快取", value=options["enable_cache"])

        _render_byok_controls()

        st.subheader("執行")
        if st.button("執行示範迭代", type="primary", use_container_width=True):
            if not state.user_intent.strip():
                st.session_state.run_message = ("warning", "請先輸入設計需求再執行工作流程。")
            else:
                _run_demo_iteration(state)
            st.rerun()

        if st.button("執行示範搜尋", use_container_width=True):
            if not state.user_intent.strip():
                st.session_state.run_message = ("warning", "請先輸入設計需求再執行工作流程。")
            else:
                if not state.tree_nodes:
                    _run_demo_iteration(state)
                while state.active_frontier and not state.is_completed and state.used_budget < state.compute_budget:
                    _run_demo_iteration(state)
            st.rerun()

        if st.button("執行自備金鑰工作流程", use_container_width=True):
            _run_byok_workflow(state)
            st.rerun()

        if st.button("重設", use_container_width=True):
            st.session_state.design_state = DesignState()
            st.session_state.selected_node_id = None
            st.session_state.run_message = None
            st.rerun()

        st.download_button(
            "匯出狀態 JSON",
            data=json.dumps(asdict(state), indent=2, ensure_ascii=False),
            file_name="genetic_circuit_design_state.json",
            mime="application/json",
            use_container_width=True,
        )

        _render_run_message()
        st.caption("示範執行會產生可重現的範例節點。自備金鑰執行只會在目前工作階段使用你的 API key，不會匯出。")


def _render_status_strip(state: DesignState) -> None:
    best_score = _best_score(state)
    active_node = state.current_node_id or "尚未開始"
    status = "已完成" if state.is_completed else ("執行中" if state.tree_nodes else "待命")
    budget_text = f"{state.used_budget} / {state.compute_budget}"
    error_type = state.error_type
    if state.current_node_id and state.current_node_id in state.tree_nodes:
        error_type = state.tree_nodes[state.current_node_id].error_type

    cols = st.columns(5, gap="small")
    cards = [
        ("狀態", status),
        ("預算", budget_text),
        ("目前節點", active_node),
        ("最佳分數", "無資料" if best_score is None else f"{best_score:.2f}"),
        ("最新問題", ERROR_LABELS.get(error_type, error_type)),
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
        "OpenAI": ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"],
        "OpenRouter": [
            "openrouter/openai/gpt-5.4",
            "openrouter/anthropic/claude-sonnet-4-6",
            "openrouter/google/gemini-3.1-pro-preview",
        ],
        "Anthropic": [
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-opus-4-7",
            "anthropic/claude-haiku-4-5-20251001",
        ],
        "Google": [
            "gemini/gemini-3.5-flash",
            "gemini/gemini-3.1-pro-preview",
            "gemini/gemini-3.1-flash-lite",
            "gemini/gemini-2.5-pro",
        ],
        "Groq": [
            "groq/meta-llama/llama-4-scout-17b-16e-instruct",
            "groq/llama-3.1-8b-instant",
            "groq/qwen/qwen3-32b",
        ],
        "Custom LiteLLM": [config.get("model_name", "custom/model") or "custom/model"],
    }

    with st.expander("自備金鑰模型設定", expanded=False):
        provider_names = list(model_presets)
        config["provider"] = st.selectbox(
            "服務提供者",
            provider_names,
            index=_safe_index(provider_names, config.get("provider")),
        )
        preset_models = model_presets[config["provider"]]
        selected_model = st.selectbox(
            "模型預設",
            preset_models,
            index=_safe_index(preset_models, config.get("model_name")),
        )
        config["model_name"] = st.text_input(
            "LiteLLM 模型名稱",
            value=config.get("model_name") or selected_model,
            placeholder=selected_model,
        )
        config["api_key"] = st.text_input(
            "API key",
            value=config.get("api_key", ""),
            type="password",
            placeholder="貼上此工作階段要使用的服務金鑰",
        )
        config["api_base"] = st.text_input(
            "API base URL",
            value=config.get("api_base", ""),
            placeholder="選填，供 OpenRouter 或自架端點使用",
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


def _render_human_loop_panel(state: DesignState) -> None:
    if not state.requires_human_input:
        return

    selected_node = _selected_node(state)
    prompt = state.human_feedback_prompt or state.latest_critic_feedback or "系統需要更多限制或偏好，才能安全地繼續搜尋。"
    reason_label = _pause_reason_label(state.pause_reason)
    best_score = _best_score(state)
    fallback_text = "目前尚無可用的備用拓樸。"
    if state.best_topology:
        fallback_score = state.best_topology.get("score", best_score)
        fallback_mapping = state.best_topology.get("mapping_status", "unknown")
        fallback_text = f"目前最佳備用拓樸分數 {fallback_score}，mapping 狀態為 {fallback_mapping}。"
    escaped_fallback_text = _escape_html(fallback_text)

    st.markdown(
        f"""
        <div class="hitl-panel">
            <h2>需要人工介入</h2>
            <p><strong>暫停原因：</strong>{_escape_html(reason_label)}</p>
            <p><strong>系統請求：</strong>{_escape_html(prompt)}</p>
            <div class="hitl-meta">目前節點：{_escape_html(selected_node.node_id if selected_node else "無")} | {escaped_fallback_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if state.human_constraints:
        with st.expander("已套用的人工限制", expanded=False):
            for constraint in state.human_constraints:
                st.write(f"- {constraint}")

    with st.form("human_loop_guidance_form"):
        guidance = st.text_area(
            "補充給系統的限制或偏好",
            height=110,
            placeholder="例如：優先降低 gate count；允許犧牲一點動態裕度；避免使用特定 promoter；保留目前最佳拓樸作為 fallback。",
        )
        action = st.radio(
            "下一步",
            [
                "加入限制並接續既有搜尋",
                "加入限制並建立修正分支",
                "加入限制並建立元件最佳化分支",
                "接受目前最佳拓樸作為結果",
            ],
            horizontal=False,
        )
        extra_budget = st.number_input("追加計算預算", min_value=0, max_value=20, value=2, step=1)
        submitted = st.form_submit_button("套用人工回饋", type="primary", use_container_width=True)

    if submitted:
        _apply_human_guidance(state, guidance, action, int(extra_budget))
        st.rerun()


def _pause_reason_label(reason: str | None) -> str:
    labels = {
        "compute_budget_exceeded": "計算預算已用完，尚未得到通過結果",
        "critic_requested_human_input": "評審代理判斷需要人工補充限制",
        "critic_unrecoverable": "評審代理判斷目前條件不足以自動修復",
        "repeated_error_type": "同類問題重複出現，需要改變搜尋方向",
        "no_recoverable_route": "評審回饋沒有明確可自動修復的路線",
        "frontier_exhausted": "所有搜尋分支都已耗盡",
    }
    return labels.get(reason or "", reason or "系統等待人工確認")


def _apply_human_guidance(state: DesignState, guidance: str, action: str, extra_budget: int) -> None:
    constraints = [line.strip("- ").strip() for line in guidance.splitlines() if line.strip()]
    state.human_constraints.extend(constraint for constraint in constraints if constraint)
    if extra_budget:
        state.compute_budget += extra_budget

    if action == "接受目前最佳拓樸作為結果":
        _select_best_fallback(state)
        state.is_completed = state.best_topology is not None
        state.requires_human_input = False
        state.pause_reason = None
        state.human_feedback_prompt = None
        st.session_state.run_message = (
            "success",
            "已接受目前最佳拓樸作為結果。" if state.best_topology else "目前沒有可接受的最佳拓樸，請改用修正或最佳化分支繼續搜尋。",
        )
        return

    if action == "加入限制並建立修正分支":
        _create_guided_child(state, "Repair")
    elif action == "加入限制並建立元件最佳化分支":
        _create_guided_child(state, "Exploitation")

    current_node = state.tree_nodes.get(state.current_node_id) if state.current_node_id else None
    if current_node and current_node.status == "Needs_Human_Input":
        current_node.status = "Evaluated"
    state.requires_human_input = False
    state.pause_reason = None
    state.human_feedback_prompt = None
    st.session_state.run_message = ("success", "已套用人工回饋。請繼續執行下一輪搜尋。")


def _create_guided_child(state: DesignState, search_mode: str) -> None:
    parent = state.tree_nodes.get(state.current_node_id) if state.current_node_id else _selected_node(state)
    if parent is None:
        return
    child_id = _child_id(parent.node_id, "repair" if search_mode == "Repair" else "exploit")
    child = SearchNode(
        node_id=child_id,
        parent_id=parent.node_id,
        search_mode=search_mode,
        logic_proposals=parent.logic_proposals[:] if search_mode == "Exploitation" else [],
        critic_feedbacks=parent.critic_feedbacks[:],
        failed_attempts=parent.failed_attempts[:],
        error_type=parent.error_type,
    )
    parent.children_ids.append(child_id)
    state.tree_nodes[child_id] = child
    state.active_frontier.insert(0, child_id)
    st.session_state.selected_node_id = child_id


def _render_pipeline(state: DesignState) -> None:
    st.markdown('<div class="section-title">工作流程進度</div>', unsafe_allow_html=True)
    current_step = _current_step(state)
    steps = [
        ("需求", bool(state.user_intent.strip()), "自然語言目標"),
        ("RAG 檢索", bool(state.rag_context), "歷史規則"),
        ("設計生成器", bool(state.logic_proposals), "邏輯提案"),
        ("轉譯器", bool(state.verilog_codes), "Cello Verilog"),
        ("Cello 映射", bool(state.candidate_topologies), "拓樸候選"),
        ("ODE 模擬", any("ode_status" in topo for topo in state.candidate_topologies), "動態分數"),
        ("評審代理", bool(state.critic_feedbacks), "回饋分流"),
        ("整合器", state.best_topology is not None, "最佳結果"),
    ]
    step_html = ['<div class="step-grid">']
    for index, (label, done, caption) in enumerate(steps):
        class_name = "step done" if done else "step"
        if index == current_step:
            class_name += " active"
        step_html.append(
            f'<div class="{class_name}">'
            f'<div class="step-label">{label}</div>'
            f'<div class="step-caption">{caption}</div>'
            "</div>"
        )
    step_html.append("</div>")
    st.markdown("".join(step_html), unsafe_allow_html=True)


def _render_chart_overview(state: DesignState) -> None:
    st.markdown('<div class="section-title">設計分析</div>', unsafe_allow_html=True)
    if not state.tree_nodes:
        st.markdown(
            '<div class="empty-state">工作流程產生已評估節點或拓樸候選後，這裡會顯示圖表。</div>',
            unsafe_allow_html=True,
        )
        return

    score_rows = _node_score_rows(state)
    topology_rows = _topology_chart_rows(state)
    left, right = st.columns(2, gap="medium")
    with left:
        st.caption("節點分數變化")
        if pd is not None and score_rows:
            chart_df = pd.DataFrame(score_rows).set_index("node")
            st.line_chart(chart_df[["score"]], use_container_width=True)
        else:
            st.info("目前尚無有效的節點分數。")
    with right:
        st.caption("候選拓樸分數")
        if pd is not None and topology_rows:
            chart_df = pd.DataFrame(topology_rows).set_index("candidate")
            st.bar_chart(chart_df[["score"]], use_container_width=True)
        else:
            st.info("目前尚無拓樸分數。")


def _render_tree_workspace(state: DesignState) -> None:
    st.markdown('<div class="section-title">樹狀搜尋工作區</div>', unsafe_allow_html=True)

    if not state.tree_nodes:
        st.markdown(
            '<div class="empty-state">請輸入設計需求，接著執行一次示範迭代來建立根搜尋節點。</div>',
            unsafe_allow_html=True,
        )
        return

    node_ids = list(state.tree_nodes.keys())
    default_node = st.session_state.selected_node_id or state.current_node_id or node_ids[0]
    selected = st.selectbox("檢視節點", node_ids, index=_safe_index(node_ids, default_node))
    st.session_state.selected_node_id = selected
    _render_search_path_panel(state, selected)

    table_rows = []
    for node in state.tree_nodes.values():
        table_rows.append(
            {
                "節點": node.node_id,
                "父節點": node.parent_id or "-",
                "模式": MODE_LABELS.get(node.search_mode, node.search_mode),
                "狀態": STATUS_LABELS.get(node.status, node.status),
                "分數": None if not math.isfinite(node.score) else round(node.score, 3),
                "問題": ERROR_LABELS.get(node.error_type, node.error_type),
                "子節點數": len(node.children_ids),
            }
        )

    if pd is not None:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.table(table_rows)

    node = state.tree_nodes[selected]
    mode_color = MODE_COLORS.get(node.search_mode, "#64748b")
    status_color = STATUS_COLORS.get(node.status, "#64748b")
    error_color = ERROR_COLORS.get(node.error_type, "#64748b")
    st.markdown(
        f"""
        <div class="node-card">
            <div class="node-title">{node.node_id}</div>
            <div class="node-meta">
                <span class="pill" style="background:{mode_color};">{MODE_LABELS.get(node.search_mode, node.search_mode)}</span>
                <span class="pill" style="background:{status_color};">{STATUS_LABELS.get(node.status, node.status)}</span>
                <span class="pill" style="background:{error_color};">{ERROR_LABELS.get(node.error_type, node.error_type)}</span>
            </div>
            <div class="node-meta">父節點：{node.parent_id or "無"} | 子節點：{", ".join(node.children_ids) or "無"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_search_path_panel(state: DesignState, selected_node_id: str) -> None:
    st.markdown('<div class="section-title">搜尋路徑與分支原因</div>', unsafe_allow_html=True)
    summary = _search_next_step_summary(state)
    if summary["level"] == "success":
        st.success(summary["text"])
    elif summary["level"] == "warning":
        st.warning(summary["text"])
    else:
        st.info(summary["text"])

    dot = _build_search_tree_dot(state, selected_node_id)
    if dot:
        st.graphviz_chart(dot, use_container_width=True)

    path = _search_path_to_node(state, selected_node_id)
    if not path:
        st.info("目前沒有可顯示的搜尋路徑。")
        return

    st.caption("目前路徑")
    for depth, node in enumerate(path):
        _render_path_node_card(state, node, depth)


def _render_path_node_card(state: DesignState, node: SearchNode, depth: int) -> None:
    reason = _branch_reason_for_node(state, node.node_id)
    feedback = _summarize_feedback(node.critic_feedbacks[-1] if node.critic_feedbacks else "")
    score = "無資料" if not math.isfinite(node.score) else f"{node.score:.2f}"
    badges = _node_state_badges(state, node)
    mode_color = MODE_COLORS.get(node.search_mode, "#64748b")
    status_color = STATUS_COLORS.get(node.status, "#64748b")
    error_color = ERROR_COLORS.get(node.error_type, "#64748b")
    mode_label = MODE_LABELS.get(node.search_mode, node.search_mode)
    status_label = STATUS_LABELS.get(node.status, node.status)
    error_label = ERROR_LABELS.get(node.error_type, node.error_type)
    st.markdown(
        f"""
        <div class="node-card">
            <div class="node-title">{depth + 1}. {_escape_html(node.node_id)}</div>
            <div class="node-meta">
                <span class="pill" style="background:{mode_color};">{mode_label}</span>
                <span class="pill" style="background:{status_color};">{status_label}</span>
                <span class="pill" style="background:{error_color};">{error_label}</span>
                {badges}
            </div>
            <div class="node-meta">分數：{score} | 建立原因：{_escape_html(reason)}</div>
            <div class="node-meta">評審摘要：{_escape_html(feedback or "尚無評審摘要")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_inspector(state: DesignState) -> None:
    st.markdown('<div class="section-title">結果檢視器</div>', unsafe_allow_html=True)
    node = _selected_node(state)

    if node is None:
        st.markdown(
            '<div class="empty-state">尚未選取節點。執行工作流程後即可檢視提案、Verilog、拓樸與評審回饋。</div>',
            unsafe_allow_html=True,
        )
        return

    proposal_tab, verilog_tab, topology_tab, ode_tab, charts_tab, critic_tab, rag_tab, raw_tab = st.tabs(
        ["提案", "Verilog", "拓樸", "ODE 模擬", "圖表", "評審", "RAG 內容", "原始狀態"]
    )

    with proposal_tab:
        proposals = node.logic_proposals or state.logic_proposals
        if proposals:
            for index, proposal in enumerate(proposals, start=1):
                with st.expander(f"提案 {index}", expanded=index == 1):
                    _render_json_or_text(proposal)
        else:
            st.info("這個節點尚未產生邏輯提案。")

    with verilog_tab:
        codes = node.verilog_codes or state.verilog_codes
        if codes:
            selected_code = st.radio("Verilog 候選", [f"候選 {i + 1}" for i in range(len(codes))], horizontal=True)
            code_index = int(selected_code.split()[-1]) - 1
            st.markdown(f'<div class="code-panel">{_escape_html(codes[code_index])}</div>', unsafe_allow_html=True)
        else:
            st.info("這個節點尚未產生 Verilog。")

    with topology_tab:
        topologies = node.candidate_topologies or state.candidate_topologies
        if topologies:
            best_topology = node.best_topology or state.best_topology
            for index, topology in enumerate(topologies):
                _render_topology_card(index, topology, best_topology)
        else:
            st.info("目前尚無拓樸候選。")

    with ode_tab:
        _render_ode_simulation_tab(node, state)

    with charts_tab:
        _render_topology_charts(node, state)

    with critic_tab:
        cols = st.columns(3)
        cols[0].metric("是否通過", "是" if node.is_approved else "否")
        cols[1].metric("問題類型", ERROR_LABELS.get(node.error_type, node.error_type))
        cols[2].metric("分數", "無資料" if not math.isfinite(node.score) else f"{node.score:.2f}")
        if node.critic_feedbacks:
            for feedback in node.critic_feedbacks:
                st.warning(feedback)
        elif state.critic_feedbacks:
            for feedback in state.critic_feedbacks:
                st.warning(feedback)
        else:
            st.info("目前尚無評審回饋。")
        if node.last_error:
            st.error(node.last_error)

    with rag_tab:
        if state.rag_context:
            st.text_area("檢索內容", value=state.rag_context, height=260)
        else:
            st.info("目前尚未檢索 RAG 內容。")

    with raw_tab:
        st.json(asdict(node))
        with st.expander("完整 DesignState"):
            st.json(asdict(state))


def _render_topology_card(index: int, topology: dict[str, Any], best_topology: dict[str, Any] | None) -> None:
    candidate_label = f"候選 {int(topology.get('verilog_index', index)) + 1}"
    is_best = topology is best_topology or (
        best_topology is not None
        and topology.get("verilog_index") == best_topology.get("verilog_index")
        and topology.get("score") == best_topology.get("score")
    )
    status = str(topology.get("mapping_status", "unknown"))
    best_label = " · 最佳拓樸" if is_best else ""
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="topology-title">{_escape_html(candidate_label)}{best_label}</div>
            <div class="topology-subtitle">Mapping：{_escape_html(status)} · Source：{_escape_html(str(topology.get("source", "unknown")))}</div>
            {_topology_metrics_html(topology)}
            """,
            unsafe_allow_html=True,
        )
        if is_best:
            st.success("目前分數最高或已選定的最佳拓樸。")

        graph = _verilog_to_gate_graph(str(topology.get("verilog", "") or ""))
        if graph["ok"]:
            st.graphviz_chart(str(graph["dot"]), use_container_width=True)
        else:
            st.warning(str(graph["message"]))

        if _is_failed_mapping(status) and topology.get("mapping_error_summary"):
            st.error(str(topology["mapping_error_summary"]))

        with st.expander(f"{candidate_label} 原始資料", expanded=False):
            raw_topology = {key: value for key, value in topology.items() if key != "verilog"}
            st.json(raw_topology)
            if topology.get("verilog"):
                st.markdown(f'<div class="code-panel">{_escape_html(str(topology["verilog"]))}</div>', unsafe_allow_html=True)


def _render_ode_simulation_tab(node: SearchNode, state: DesignState) -> None:
    topologies = node.candidate_topologies or state.candidate_topologies
    if not topologies:
        st.info("目前尚無拓樸候選，因此沒有 ODE 模擬圖表。")
        return

    labels = [_topology_candidate_label(index, topology) for index, topology in enumerate(topologies)]
    selected_label = st.selectbox("檢視 ODE 候選", labels, key=f"ode_candidate_{node.node_id}")
    selected_index = labels.index(selected_label)
    topology = topologies[selected_index]
    trace = topology.get("ode_trace")

    status = str(topology.get("ode_status", "unknown"))
    if status == "disabled":
        st.info("此候選的 ODE 模擬已停用。請在左側開啟 ODE 模擬後重新執行。")
        return
    if status == "failed":
        st.error("此候選的 ODE 模擬失敗，無法顯示時間序列圖。")
        return
    if not _valid_ode_trace(trace):
        st.warning("此候選尚未保存 ODE 時間序列。請重新執行 ODE 模擬以產生圖表資料。")
        _render_ode_metric_summary(topology)
        return

    _render_ode_metric_summary(topology)
    trace_rows = _ode_trace_rows(trace)
    if pd is not None:
        trace_df = pd.DataFrame(trace_rows).set_index("time")
        left, right = st.columns(2, gap="medium")
        with left:
            st.caption("輸出蛋白濃度")
            st.line_chart(trace_df[["output_protein"]], use_container_width=True)
        with right:
            st.caption("mRNA / protein 總負擔")
            burden_cols = [column for column in ["total_mrna", "total_protein"] if column in trace_df.columns]
            st.line_chart(trace_df[burden_cols], use_container_width=True)
        st.caption("資源佔用率")
        occupancy_cols = [column for column in ["rnap_occupancy", "ribosome_occupancy"] if column in trace_df.columns]
        st.line_chart(trace_df[occupancy_cols], use_container_width=True)
    else:
        st.table(trace_rows)

    with st.expander("ODE trace 原始資料", expanded=False):
        st.json(trace)


def _render_ode_metric_summary(topology: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("ODE 狀態", str(topology.get("ode_status", "unknown")))
    cols[1].metric("Dynamic margin", _format_metric(topology.get("dynamic_margin")))
    cols[2].metric("SNR", _format_metric(topology.get("signal_to_noise_ratio")))
    cols[3].metric("Output CV", _format_metric(topology.get("metrics_cv")))


def _valid_ode_trace(trace: Any) -> bool:
    if not isinstance(trace, dict):
        return False
    time_values = trace.get("time")
    output_values = trace.get("output_protein")
    return isinstance(time_values, list) and isinstance(output_values, list) and len(time_values) == len(output_values) and len(time_values) > 0


def _ode_trace_rows(trace: dict[str, list[float]]) -> list[dict[str, float]]:
    time_values = trace.get("time", [])
    rows = []
    for index, time_value in enumerate(time_values):
        row = {"time": float(time_value)}
        for key in ["output_protein", "total_mrna", "total_protein", "rnap_occupancy", "ribosome_occupancy"]:
            values = trace.get(key, [])
            if index < len(values):
                row[key] = float(values[index])
        rows.append(row)
    return rows


def _topology_candidate_label(index: int, topology: dict[str, Any]) -> str:
    candidate_number = int(topology.get("verilog_index", index)) + 1
    score = topology.get("score")
    status = topology.get("ode_status", topology.get("mapping_status", "unknown"))
    score_text = f" · score {float(score):.2f}" if isinstance(score, int | float) else ""
    return f"候選 {candidate_number}{score_text} · {status}"


def _topology_metrics_html(topology: dict[str, Any]) -> str:
    metric_keys = [
        ("score", "Score"),
        ("gate_count", "Gates"),
        ("dynamic_margin", "Dynamic"),
        ("robustness_score", "Robustness"),
        ("orthogonality_score", "Orthogonality"),
        ("cello_assignment_score", "Cello"),
    ]
    blocks = []
    for key, label in metric_keys:
        if key not in topology:
            continue
        blocks.append(
            '<div class="topology-metric">'
            f'<div class="topology-metric-label">{label}</div>'
            f'<div class="topology-metric-value">{_escape_html(_format_metric(topology.get(key)))}</div>'
            "</div>"
        )
    return f'<div class="topology-metrics">{"".join(blocks)}</div>' if blocks else ""


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "無資料" if value is None else str(value)


def _is_failed_mapping(status: str) -> bool:
    return status.strip().lower() in {"failed", "mapping_failed", "unmapped", "error"}


def _render_topology_charts(node: SearchNode, state: DesignState) -> None:
    if pd is None:
        st.info("請安裝 pandas 以啟用 Streamlit 圖表渲染。")
        return

    topologies = node.candidate_topologies or state.candidate_topologies
    if not topologies:
        st.info("目前沒有可用於圖表的拓樸候選。")
        return

    chart_df = pd.DataFrame(_topology_rows(topologies)).set_index("candidate")
    st.caption("各拓樸候選分數")
    st.bar_chart(chart_df[["score"]], use_container_width=True)

    metric_cols = [column for column in ["gate_count", "dynamic_margin"] if column in chart_df.columns]
    if metric_cols:
        st.caption("實作複雜度與動態裕度")
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
        st.session_state.run_message = ("warning", "請先輸入設計需求再執行自備金鑰工作流程。")
        return
    if not config.get("api_key", "").strip():
        st.session_state.run_message = ("warning", "請在自備金鑰模型設定中輸入 API key。")
        return
    if not config.get("model_name", "").strip():
        st.session_state.run_message = ("warning", "請選擇或輸入 LiteLLM 模型名稱。")
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
        st.session_state.run_message = ("error", f"無法載入工作流程元件：{exc}")
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
        state.last_error = f"錯誤：自備金鑰工作流程失敗：{exc}"
        st.session_state.run_message = ("error", state.last_error)
        return

    st.session_state.design_state = result_state
    st.session_state.selected_node_id = result_state.current_node_id
    if result_state.last_error:
        st.session_state.run_message = ("error", result_state.last_error)
    else:
        st.session_state.run_message = ("success", "自備金鑰工作流程已完成。請檢視產生的節點、Verilog、拓樸與圖表。")


class _NoOpODESimulator:
    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes.get(state.current_node_id) if state.current_node_id else None
        topologies = node.candidate_topologies if node else state.candidate_topologies
        for topology in topologies:
            topology["ode_status"] = "disabled"
        return state


def _demo_proposals(state: DesignState, node: SearchNode) -> list[str]:
    base_intent = state.user_intent.strip() or "設計一個基因邏輯電路"
    repair_hint = "，並納入評審回饋" if node.search_mode == "Repair" else ""
    blueprints = [
        ("proposal_a", "降低生物元件成本", "Y = A AND NOT B", 2, 3, ["PRESERVE_SIMPLE_GATES"]),
        ("proposal_b", "降低邏輯深度與延遲", "Y = A OR B", 1, 4, []),
        ("proposal_c", "優先提高雜訊輸入下的穩健性", "Y = (A AND NOT B) OR (A AND C)", 3, 5, ["USE_STRUCTURAL_INSTANTIATION"]),
    ]
    proposals = []
    for key, strategy, blueprint, depth, cost, directives in blueprints:
        proposals.append(
            json.dumps(
                {
                    "id": key,
                    "strategy_description": f"{strategy}{repair_hint}；需求：{base_intent}",
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
            topology["ode_trace"] = _demo_ode_trace(index, mode_bonus)
        else:
            topology["ode_status"] = "disabled"
        topologies.append(topology)
    return topologies


def _demo_ode_trace(index: int, mode_bonus: float) -> dict[str, list[float]]:
    time_points = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
    scale = 1.0 + index * 0.28 + mode_bonus
    output = [round(scale * 85.0 * (1.0 - math.exp(-point / 210.0)), 3) for point in time_points]
    total_mrna = [round(scale * 14.0 * (1.0 - math.exp(-point / 130.0)), 3) for point in time_points]
    total_protein = [round(scale * 120.0 * (1.0 - math.exp(-point / 260.0)), 3) for point in time_points]
    rnap_occupancy = [round(min(0.95, 0.12 + scale * 0.22 * (1.0 - math.exp(-point / 180.0))), 4) for point in time_points]
    ribosome_occupancy = [round(min(0.95, 0.16 + scale * 0.26 * (1.0 - math.exp(-point / 220.0))), 4) for point in time_points]
    return {
        "time": time_points,
        "output_protein": output,
        "total_mrna": total_mrna,
        "total_protein": total_protein,
        "rnap_occupancy": rnap_occupancy,
        "ribosome_occupancy": ribosome_occupancy,
    }


def _demo_critic_and_branch(state: DesignState, node: SearchNode, enable_tree_search: bool) -> None:
    approved = node.score >= 0.82 or state.used_budget >= state.compute_budget - 1
    node.is_approved = approved

    if approved:
        node.error_type = "NONE"
        node.critic_feedbacks.append("設計已通過示範門檻。請將此拓樸整合為目前最佳結果。")
        state.is_completed = True
        return

    state.used_budget += 1
    if node.search_mode == "Exploration":
        node.error_type = "LOGIC_ERROR"
        node.critic_feedbacks.append("邏輯方向合理，但規格仍不夠明確。請新增修正分支，在保留需求的前提下收斂布林行為。")
    elif node.search_mode == "Repair":
        node.error_type = "PART_ERROR"
        node.critic_feedbacks.append("邏輯目前可接受，但元件 mapping 仍可改善。請在不改變架構的前提下進行最佳化。")
    else:
        node.error_type = "PART_ERROR"
        node.critic_feedbacks.append("元件指派仍需要微調。請保留最高分拓樸作為備用結果。")

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
            f"模式感知檢索：{MODE_LABELS.get(mode, mode)}",
            "優先使用 Cello 相容的組合邏輯：primitive gates、wire、assign。",
            "避免 always blocks、registers、clocks、latches、memories 與 delay syntax。",
            f"需求關鍵字：{', '.join(intent.lower().split()[:8]) or '未提供'}",
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
        candidate = f"候選 {int(topology.get('verilog_index', index)) + 1}"
        rows.append(
            {
                "candidate": candidate,
                "score": float(topology.get("score", 0.0)),
                "gate_count": int(topology.get("gate_count", 0)),
                "dynamic_margin": float(topology.get("dynamic_margin", 0.0)),
            }
        )
    return rows


def _verilog_to_gate_graph(verilog: str) -> dict[str, Any]:
    code = _strip_verilog_comments(verilog)
    if not code.strip():
        return {"ok": False, "dot": "", "message": "此拓樸沒有 Verilog，無法產生 gate graph。"}

    inputs, outputs, wires = _extract_verilog_signals(code)
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    gate_index = 0

    for signal in sorted(inputs):
        nodes[signal] = "input"
    for signal in sorted(outputs):
        nodes[signal] = "output"
    for signal in sorted(wires):
        nodes.setdefault(signal, "wire")

    for gate, body in re.findall(r"\b(and|or|not|nand|nor|xor|xnor)\s*\(([^;]+?)\)\s*;", code, flags=re.IGNORECASE | re.DOTALL):
        parts = [_clean_signal_name(part) for part in body.split(",")]
        parts = [part for part in parts if part]
        if len(parts) < 2:
            continue
        output, gate_inputs = parts[0], parts[1:]
        gate_index += 1
        gate_node = f"{gate.upper()}_{gate_index}"
        nodes[gate_node] = "gate"
        nodes.setdefault(output, "wire")
        for signal in gate_inputs:
            nodes.setdefault(signal, "unknown")
            edges.append((signal, gate_node))
        edges.append((gate_node, output))

    for lhs, rhs in re.findall(r"\bassign\s+([^=;]+?)\s*=\s*([^;]+?)\s*;", code, flags=re.IGNORECASE | re.DOTALL):
        output = _clean_signal_name(lhs)
        if not output:
            continue
        nodes.setdefault(output, "output" if output in outputs else "wire")
        gate_index = _add_assign_expression_edges(output, rhs, nodes, edges, gate_index)

    if not edges:
        return {"ok": False, "dot": "", "message": "無法解析 gate graph；請展開原始資料查看 Verilog。"}

    return {"ok": True, "dot": _build_gate_graph_dot(nodes, edges, inputs, outputs), "message": ""}


def _strip_verilog_comments(verilog: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", "", verilog, flags=re.DOTALL)
    return re.sub(r"//.*", "", without_block)


def _extract_verilog_signals(code: str) -> tuple[set[str], set[str], set[str]]:
    signals: dict[str, set[str]] = {"input": set(), "output": set(), "wire": set()}
    for keyword in signals:
        for match in re.finditer(rf"\b{keyword}\b\s*(?:\[[^\]]+\]\s*)?([^;);]+)", code, flags=re.IGNORECASE):
            declaration = re.split(r"\b(?:input|output|wire|module|endmodule)\b", match.group(1), flags=re.IGNORECASE)[0]
            for name in re.split(r",", declaration):
                signal = _clean_signal_name(name)
                if signal:
                    signals[keyword].add(signal)
        for match in re.finditer(rf"\b{keyword}\b\s*(?:\[[^\]]+\]\s*)?([A-Za-z_]\w*)", code, flags=re.IGNORECASE):
            signals[keyword].add(match.group(1))
    return signals["input"], signals["output"], signals["wire"]


def _add_assign_expression_edges(
    output: str,
    expression: str,
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    gate_index: int,
) -> int:
    expression = _strip_outer_parens(expression.strip())
    direct = _clean_signal_name(expression)
    if direct and direct == expression.strip():
        nodes.setdefault(direct, "unknown")
        edges.append((direct, output))
        return gate_index

    for operator, gate in [("&", "AND"), ("|", "OR"), ("^", "XOR")]:
        parts = _split_expression(expression, operator)
        if len(parts) > 1:
            gate_index += 1
            gate_node = f"{gate}_{gate_index}"
            nodes[gate_node] = "gate"
            for part in parts:
                source, gate_index = _expression_source_node(part, nodes, edges, gate_index)
                if source:
                    edges.append((source, gate_node))
            edges.append((gate_node, output))
            return gate_index

    source, gate_index = _expression_source_node(expression, nodes, edges, gate_index)
    if source:
        edges.append((source, output))
    return gate_index


def _expression_source_node(
    expression: str,
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    gate_index: int,
) -> tuple[str | None, int]:
    expression = _strip_outer_parens(expression.strip())
    if expression.startswith("~") or expression.startswith("!"):
        source = _clean_signal_name(expression[1:])
        if not source:
            return None, gate_index
        gate_index += 1
        gate_node = f"NOT_{gate_index}"
        nodes.setdefault(source, "unknown")
        nodes[gate_node] = "gate"
        edges.append((source, gate_node))
        return gate_node, gate_index

    signal = _clean_signal_name(expression)
    if signal:
        nodes.setdefault(signal, "unknown")
        return signal, gate_index
    return None, gate_index


def _split_expression(expression: str, operator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in expression:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == operator and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def _strip_outer_parens(value: str) -> str:
    value = value.strip()
    while value.startswith("(") and value.endswith(")"):
        inner = value[1:-1].strip()
        if not inner:
            break
        value = inner
    return value


def _clean_signal_name(value: str) -> str:
    value = re.sub(r"\b(?:input|output|wire|reg)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\[[^\]]+\]", "", value)
    match = re.search(r"[A-Za-z_]\w*", value.strip())
    return match.group(0) if match else ""


def _build_gate_graph_dot(nodes: dict[str, str], edges: list[tuple[str, str]], inputs: set[str], outputs: set[str]) -> str:
    lines = [
        "digraph GateGraph {",
        "  graph [rankdir=LR, bgcolor=\"transparent\", pad=\"0.2\", nodesep=\"0.45\", ranksep=\"0.65\"];",
        "  node [fontname=\"Arial\", fontsize=10, margin=\"0.08,0.05\"];",
        "  edge [color=\"#64748b\", arrowsize=0.7];",
    ]
    for name, kind in sorted(nodes.items()):
        normalized_kind = "input" if name in inputs else "output" if name in outputs else kind
        lines.append(f"  {_dot_id(name)} [{_dot_node_attrs(name, normalized_kind)}];")
    for source, target in edges:
        lines.append(f"  {_dot_id(source)} -> {_dot_id(target)};")
    lines.append("}")
    return "\n".join(lines)


def _dot_node_attrs(name: str, kind: str) -> str:
    label = _dot_escape(_gate_label(name))
    if kind == "input":
        return f'label="{label}", shape=box, style="rounded,filled", fillcolor="#dbeafe", color="#60a5fa"'
    if kind == "output":
        return f'label="{label}", shape=box, style="rounded,filled", fillcolor="#dcfce7", color="#34d399"'
    if kind == "gate":
        return f'label="{label}", shape=box, style="rounded,filled", fillcolor="#ffffff", color="#334155"'
    return f'label="{label}", shape=ellipse, style=filled, fillcolor="#f1f5f9", color="#cbd5e1"'


def _gate_label(name: str) -> str:
    return re.sub(r"_\d+$", "", name) if re.match(r"^(?:AND|OR|NOT|NAND|NOR|XOR|XNOR)_\d+$", name) else name


def _dot_id(name: str) -> str:
    return f'"{_dot_escape(name)}"'


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _search_path_to_node(state: DesignState, node_id: str | None) -> list[SearchNode]:
    if not node_id or node_id not in state.tree_nodes:
        return []
    path: list[SearchNode] = []
    seen: set[str] = set()
    current_id: str | None = node_id
    while current_id and current_id in state.tree_nodes and current_id not in seen:
        seen.add(current_id)
        node = state.tree_nodes[current_id]
        path.append(node)
        current_id = node.parent_id
    path.reverse()
    return path


def _branch_reason_for_node(state: DesignState, node_id: str) -> str:
    node = state.tree_nodes.get(node_id)
    if node is None:
        return "找不到節點"
    parent = state.tree_nodes.get(node.parent_id or "")
    if parent is None:
        return "搜尋起點"
    if node.search_mode == "Repair":
        return f"{ERROR_LABELS.get(parent.error_type, parent.error_type)} -> 修正"
    if node.search_mode == "Exploitation":
        return f"{ERROR_LABELS.get(parent.error_type, parent.error_type)} -> 元件最佳化"
    if node.search_mode == "Exploration":
        return f"{ERROR_LABELS.get(parent.error_type, parent.error_type)} -> 重新探索"
    return f"{ERROR_LABELS.get(parent.error_type, parent.error_type)} -> {MODE_LABELS.get(node.search_mode, node.search_mode)}"


def _search_next_step_summary(state: DesignState) -> dict[str, str]:
    if state.requires_human_input:
        return {"level": "warning", "text": "工作流程正在等待人工回饋。請先處理上方的人工介入面板，再繼續執行搜尋。"}
    if state.is_completed:
        return {"level": "success", "text": "搜尋已完成，最佳拓樸已可在結果檢視器中查看。"}
    if state.active_frontier:
        next_id = state.active_frontier[0]
        reason = _branch_reason_for_node(state, next_id)
        return {"level": "info", "text": f"下一個待處理節點：{next_id}。原因：{reason}。"}
    return {"level": "warning", "text": "目前沒有待處理節點；若尚未完成，請補充人工限制或選擇 fallback 拓樸。"}


def _build_search_tree_dot(state: DesignState, selected_node_id: str | None) -> str:
    if not state.tree_nodes:
        return ""
    lines = [
        "digraph SearchTree {",
        "  graph [rankdir=TB, bgcolor=\"transparent\", pad=\"0.2\", nodesep=\"0.35\", ranksep=\"0.55\"];",
        "  node [fontname=\"Arial\", fontsize=10, margin=\"0.10,0.06\"];",
        "  edge [fontname=\"Arial\", fontsize=9, color=\"#94a3b8\", arrowsize=0.7];",
    ]
    frontier = set(state.active_frontier)
    for node_id, node in state.tree_nodes.items():
        lines.append(f"  {_dot_id(node_id)} [{_search_tree_node_attrs(state, node, selected_node_id, frontier)}];")
    for node_id, node in state.tree_nodes.items():
        for child_id in node.children_ids:
            if child_id not in state.tree_nodes:
                continue
            reason = _branch_reason_for_node(state, child_id)
            lines.append(f"  {_dot_id(node_id)} -> {_dot_id(child_id)} [label=\"{_dot_escape(reason)}\"];")
    lines.append("}")
    return "\n".join(lines)


def _search_tree_node_attrs(
    state: DesignState,
    node: SearchNode,
    selected_node_id: str | None,
    frontier: set[str],
) -> str:
    label_parts = [node.node_id, MODE_LABELS.get(node.search_mode, node.search_mode)]
    if math.isfinite(node.score):
        label_parts.append(f"{node.score:.2f}")
    if node.node_id == state.current_node_id:
        label_parts.append("current")
    if node.node_id in frontier:
        label_parts.append("frontier")
    label = _dot_escape("\\n".join(label_parts))
    fill = {
        "Exploration": "#dbeafe",
        "Repair": "#ffedd5",
        "Exploitation": "#dcfce7",
    }.get(node.search_mode, "#f1f5f9")
    color = STATUS_COLORS.get(node.status, MODE_COLORS.get(node.search_mode, "#64748b"))
    penwidth = "3" if node.node_id == selected_node_id else "1.6"
    style = "rounded,filled,bold" if node.node_id in frontier or node.node_id == selected_node_id else "rounded,filled"
    return f'label="{label}", shape=box, style="{style}", fillcolor="{fill}", color="{color}", penwidth={penwidth}'


def _node_state_badges(state: DesignState, node: SearchNode) -> str:
    badges = []
    if node.node_id == state.current_node_id:
        badges.append(("#0f172a", "目前"))
    if node.node_id in state.active_frontier:
        badges.append(("#0891b2", "待處理"))
    if node.node_id == st.session_state.get("selected_node_id"):
        badges.append(("#6366f1", "選取"))
    return "".join(
        f'<span class="pill" style="background:{color};">{label}</span>'
        for color, label in badges
    )


def _summarize_feedback(feedback: str, limit: int = 110) -> str:
    compact = " ".join(feedback.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


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
