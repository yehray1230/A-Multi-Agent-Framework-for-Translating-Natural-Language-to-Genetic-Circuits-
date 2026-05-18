from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp_server.serializers import summarize_state, summarize_topology, to_jsonable


DEFAULT_OUTPUT_DIR = Path("outputs") / "mcp_runs"


def create_run_dir(output_dir: str | Path | None = None, run_id: str | None = None) -> Path:
    base_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    selected_run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = base_dir / selected_run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base_dir / f"{selected_run_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_state_artifacts(state: Any, run_dir: Path, charts: list[Path] | None = None) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    full_state_path = run_dir / "state.json"
    summary_path = run_dir / "summary.json"
    topology_path = run_dir / "best_topology.json"
    verilog_path = run_dir / "best_design.v"
    markdown_path = run_dir / "run_summary.md"

    write_json(full_state_path, state)
    write_json(summary_path, summarize_state(state))
    write_json(topology_path, summarize_topology(state.best_topology))
    artifacts["state_json"] = str(full_state_path.resolve())
    artifacts["summary_json"] = str(summary_path.resolve())
    artifacts["best_topology_json"] = str(topology_path.resolve())

    best_verilog = ""
    if state.best_topology:
        best_verilog = str(state.best_topology.get("verilog") or "")
    if not best_verilog and state.verilog_codes:
        best_verilog = str(state.verilog_codes[0])
    if best_verilog:
        write_text(verilog_path, best_verilog)
        artifacts["best_verilog"] = str(verilog_path.resolve())

    write_text(markdown_path, _summary_markdown(state))
    artifacts["run_summary_md"] = str(markdown_path.resolve())

    for chart in charts or []:
        artifacts[chart.stem] = str(chart.resolve())
    return artifacts


def _summary_markdown(state: Any) -> str:
    best = summarize_topology(state.best_topology)
    lines = [
        "# MCP Genetic Circuit Run",
        "",
        f"- Intent: {state.user_intent}",
        f"- Host: {state.host_organism}",
        f"- Completed: {state.is_completed}",
        f"- Approved: {state.is_approved}",
        f"- Requires human input: {state.requires_human_input}",
        f"- Pause reason: {state.pause_reason or ''}",
        f"- Score: {best.get('score', '')}",
        f"- Mapping status: {best.get('mapping_status', '')}",
        f"- ODE status: {best.get('ode_status', '')}",
        f"- Critic feedback: {state.latest_critic_feedback}",
        "",
    ]
    if state.human_feedback_prompt:
        lines.extend(["## Human Feedback Prompt", "", state.human_feedback_prompt, ""])
    if best.get("verilog"):
        lines.extend(["## Verilog", "", "```verilog", str(best["verilog"]), "```", ""])
    return "\n".join(lines)

