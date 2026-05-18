from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mcp_server.artifact_writer import DEFAULT_OUTPUT_DIR, write_json
from mcp_server.serializers import to_jsonable


TERMINAL_STATUSES = {"completed", "needs_human_input", "error", "failed", "cancelled"}


class RunStore:
    """Small in-process run manager for the MCP prototype."""

    def __init__(self, base_dir: str | Path | None = None, max_workers: int = 2):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_OUTPUT_DIR / "async_runs"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mcp-design-run")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def start(
        self,
        task: Callable[[], dict[str, Any]],
        request: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        selected_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        run_dir = self.base_dir / selected_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "run_id": selected_run_id,
            "status": "queued",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "request": _redact_request(request),
            "run_dir": str(run_dir.resolve()),
            "result_path": str((run_dir / "result.json").resolve()),
            "error": None,
        }
        self._write_metadata(run_dir, metadata)

        future = self._executor.submit(self._run_task, selected_run_id, run_dir, task)
        with self._lock:
            self._futures[selected_run_id] = future

        return self.status(selected_run_id)

    def status(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return {"status": "not_found", "run_id": run_id, "error": f"Unknown run_id: {run_id}"}
        metadata = self._read_metadata(run_dir)
        future = self._futures.get(run_id)
        if future and not future.done() and metadata.get("status") == "queued":
            metadata["status"] = "queued"
        return _public_status(metadata)

    def result(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return {"status": "not_found", "run_id": run_id, "error": f"Unknown run_id: {run_id}"}
        metadata = self._read_metadata(run_dir)
        status = str(metadata.get("status") or "unknown")
        if status not in TERMINAL_STATUSES:
            return {
                **_public_status(metadata),
                "message": "Run is not finished yet. Call get_design_run_status again later.",
            }

        result_path = Path(str(metadata.get("result_path") or run_dir / "result.json"))
        if not result_path.exists():
            return {
                **_public_status(metadata),
                "error": metadata.get("error") or "Run finished but result.json was not written.",
            }
        return json.loads(result_path.read_text(encoding="utf-8"))

    def _run_task(self, run_id: str, run_dir: Path, task: Callable[[], dict[str, Any]]) -> None:
        metadata = self._read_metadata(run_dir)
        metadata.update({"status": "running", "started_at": _now_iso(), "updated_at": _now_iso()})
        self._write_metadata(run_dir, metadata)

        try:
            result = task()
            result = to_jsonable(result)
            status = str(result.get("status") or "completed")
            if status not in TERMINAL_STATUSES:
                status = "completed"
            result["async_run_id"] = run_id
            result["async_run_dir"] = str(run_dir.resolve())
            write_json(run_dir / "result.json", result)
            metadata.update(
                {
                    "status": status,
                    "finished_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "result_status": result.get("status"),
                    "workflow_run_dir": result.get("run_dir"),
                    "artifacts": result.get("artifacts", {}),
                    "summary": _compact_summary(result.get("summary", {})),
                }
            )
        except Exception as exc:
            error_result = {
                "status": "failed",
                "async_run_id": run_id,
                "async_run_dir": str(run_dir.resolve()),
                "error": str(exc),
            }
            write_json(run_dir / "result.json", error_result)
            metadata.update(
                {
                    "status": "failed",
                    "finished_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "error": str(exc),
                }
            )
        finally:
            self._write_metadata(run_dir, metadata)

    def _run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def _read_metadata(self, run_dir: Path) -> dict[str, Any]:
        metadata_path = run_dir / "metadata.json"
        last_error: Exception | None = None
        for _ in range(5):
            try:
                return json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as exc:
                last_error = exc
                time.sleep(0.02)
        raise RuntimeError(f"Could not read run metadata at {metadata_path}: {last_error}")

    def _write_metadata(self, run_dir: Path, metadata: dict[str, Any]) -> None:
        metadata_path = run_dir / "metadata.json"
        temp_path = run_dir / "metadata.json.tmp"
        write_json(temp_path, metadata)
        temp_path.replace(metadata_path)


def _public_status(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": metadata.get("run_id"),
        "status": metadata.get("status"),
        "created_at": metadata.get("created_at"),
        "started_at": metadata.get("started_at"),
        "finished_at": metadata.get("finished_at"),
        "updated_at": metadata.get("updated_at"),
        "run_dir": metadata.get("run_dir"),
        "workflow_run_dir": metadata.get("workflow_run_dir"),
        "result_path": metadata.get("result_path"),
        "summary": metadata.get("summary", {}),
        "artifacts": metadata.get("artifacts", {}),
        "error": metadata.get("error"),
    }


def _compact_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    best_topology = summary.get("best_topology")
    if not isinstance(best_topology, dict):
        best_topology = {}
    return {
        "user_intent": summary.get("user_intent"),
        "host_organism": summary.get("host_organism"),
        "is_completed": summary.get("is_completed"),
        "is_approved": summary.get("is_approved"),
        "requires_human_input": summary.get("requires_human_input"),
        "pause_reason": summary.get("pause_reason"),
        "current_node_id": summary.get("current_node_id"),
        "current_node_status": summary.get("current_node_status"),
        "used_budget": summary.get("used_budget"),
        "compute_budget": summary.get("compute_budget"),
        "latest_critic_feedback": summary.get("latest_critic_feedback"),
        "score": best_topology.get("score"),
        "mapping_status": best_topology.get("mapping_status"),
        "ode_status": best_topology.get("ode_status"),
    }


def _redact_request(request: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(request)
    if redacted.get("api_key"):
        redacted["api_key"] = "***"
    return to_jsonable(redacted)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
