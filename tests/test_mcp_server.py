from __future__ import annotations

import time
from pathlib import Path

from mcp_server.run_store import RunStore
from mcp_server.service import evaluate_verilog, get_design_run_result, get_design_run_status, start_design_run, summarize_design_state


def _wait_for_completed(fetch_result):
    result = fetch_result()
    for _ in range(100):
        if result["status"] == "completed":
            return result
        time.sleep(0.02)
        result = fetch_result()
    return result


def test_evaluate_verilog_writes_agent_artifacts(tmp_path: Path) -> None:
    result = evaluate_verilog(
        "module genetic_circuit(input A, input B, output Y); assign Y = A & ~B; endmodule",
        enable_ode=False,
        output_dir=str(tmp_path),
    )

    assert result["status"] == "completed"
    artifacts = result["artifacts"]
    assert Path(artifacts["summary_json"]).exists()
    assert Path(artifacts["best_topology_json"]).exists()
    assert Path(artifacts["best_verilog"]).exists()
    assert Path(artifacts["run_summary_md"]).exists()
    assert result["best_topology"]["mapping_status"] == "unmapped"


def test_summarize_design_state_accepts_saved_state_shape() -> None:
    result = summarize_design_state(
        {
            "user_intent": "A and not B",
            "host_organism": "Escherichia coli",
            "is_completed": True,
            "is_approved": False,
            "requires_human_input": False,
            "pause_reason": None,
            "current_node_id": "root",
            "best_topology": {"score": 0.72, "mapping_status": "mapped"},
        }
    )

    assert result["status"] == "completed"
    assert result["summary"]["best_topology"]["score"] == 0.72


def test_run_store_background_task_persists_result(tmp_path: Path) -> None:
    store = RunStore(base_dir=tmp_path, max_workers=1)
    started = store.start(
        task=lambda: {
            "status": "completed",
            "summary": {"user_intent": "A and not B", "best_topology": {"score": 0.8}},
            "artifacts": {"summary_json": "summary.json"},
        },
        request={"user_intent": "A and not B", "api_key": "secret"},
    )

    run_id = started["run_id"]
    result = _wait_for_completed(lambda: store.result(run_id))

    status = store.status(run_id)
    assert status["status"] == "completed"
    assert status["summary"]["score"] == 0.8
    assert result["async_run_id"] == run_id
    assert Path(status["result_path"]).exists()


def test_service_async_design_run_uses_background_store(monkeypatch, tmp_path: Path) -> None:
    store = RunStore(base_dir=tmp_path, max_workers=1)

    def fake_design_circuit_quick(**kwargs):
        return {
            "status": "completed",
            "run_dir": str(tmp_path / "workflow"),
            "summary": {
                "user_intent": kwargs["user_intent"],
                "host_organism": kwargs["host_organism"],
                "is_completed": True,
                "best_topology": {"score": 0.91, "mapping_status": "unmapped"},
            },
            "artifacts": {"summary_json": str(tmp_path / "summary.json")},
        }

    monkeypatch.setattr("mcp_server.service.design_circuit_quick", fake_design_circuit_quick)

    started = start_design_run(
        user_intent="A and not B",
        host_organism="E. coli",
        run_store=store,
    )
    run_id = started["run_id"]
    result = _wait_for_completed(lambda: get_design_run_result(run_id, run_store=store))

    status = get_design_run_status(run_id, run_store=store)
    assert status["status"] == "completed"
    assert status["workflow_run_dir"] == str(tmp_path / "workflow")
    assert result["summary"]["best_topology"]["score"] == 0.91
