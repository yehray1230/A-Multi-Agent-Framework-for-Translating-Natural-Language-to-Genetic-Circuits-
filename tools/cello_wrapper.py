from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from benchmark_suite.cello_constraint_evaluator import evaluate_cello_constraints
from schemas.state import DesignState


class CelloWrapper:
    def __init__(
        self,
        cello_command: str | list[str] | None = None,
        ucf_path: str | None = None,
        work_dir: str | Path | None = None,
        timeout_seconds: int = 120,
        max_log_chars: int = 4000,
    ):
        self.cello_command = cello_command
        self.ucf_path = ucf_path
        self.work_dir = Path(work_dir) if work_dir else None
        self.timeout_seconds = timeout_seconds
        self.max_log_chars = max_log_chars

    def run(self, state: DesignState) -> DesignState:
        node = state.tree_nodes.get(state.current_node_id) if state.current_node_id else None
        codes = node.verilog_codes if node else state.verilog_codes
        valid_codes = [code for code in codes if code and not code.startswith("ERROR:")]
        if not valid_codes:
            state.last_error = "ERROR: CelloWrapper received no valid Verilog."
            return state

        if self.cello_command is None:
            topologies = [self._mock_topology(index, code) for index, code in enumerate(valid_codes)]
        else:
            topologies = [self._run_external_cello(index, code) for index, code in enumerate(valid_codes)]

        if node:
            node.candidate_topologies = topologies
        state.candidate_topologies = topologies
        state.last_error = None
        return state

    def _mock_topology(self, index: int, code: str) -> dict[str, Any]:
        return {
            "source": "mock_cello_wrapper",
            "verilog_index": index,
            "verilog": code,
            "score": 0.0,
            "mapping_status": "unmapped",
            "orthogonality_score": 1.0,
            "cello_assignment_score": 0.0,
            "cello_buildable": False,
        }

    def _run_external_cello(self, index: int, code: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(dir=self.work_dir) as temp_dir:
            temp_path = Path(temp_dir)
            netlist_path = temp_path / f"candidate_{index}.v"
            output_dir = temp_path / "output"
            output_dir.mkdir(exist_ok=True)
            netlist_path.write_text(code, encoding="utf-8")
            command = self._build_command(netlist_path, output_dir)
            try:
                completed = subprocess.run(
                    command,
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return self._failed_topology(
                    index,
                    code,
                    "TIMEOUT",
                    f"Cello mapping timed out after {self.timeout_seconds} seconds.",
                    (exc.stdout or "") + "\n" + (exc.stderr or ""),
                )
            except Exception as exc:
                return self._failed_topology(
                    index,
                    code,
                    "WRAPPER_EXCEPTION",
                    f"Cello wrapper crashed before mapping: {exc}",
                    repr(exc),
                )

            log = f"STDOUT:\n{completed.stdout or ''}\nSTDERR:\n{completed.stderr or ''}"
            if completed.returncode != 0:
                category = _classify_error_log(log)
                return self._failed_topology(
                    index,
                    code,
                    category,
                    _summarize_error(category, log),
                    log,
                    return_code=completed.returncode,
                )

            topology = {
                "source": "external_cello_wrapper",
                "verilog_index": index,
                "verilog": code,
                "score": 0.0,
                "mapping_status": "mapped",
                "orthogonality_score": 1.0,
                "cello_assignment_score": 0.0,
                "cello_buildable": True,
                "cello_stdout": _truncate_error_log(completed.stdout or "", self.max_log_chars),
            }
            topology.update(_topology_cello_metrics(topology))
            return topology

    def _build_command(self, netlist_path: Path, output_dir: Path) -> list[str]:
        command = shlex.split(self.cello_command) if isinstance(self.cello_command, str) else list(self.cello_command or [])
        expanded: list[str] = []
        for part in command:
            expanded.append(
                part.format(
                    input_netlist=str(netlist_path),
                    output_dir=str(output_dir),
                    ucf_path=self.ucf_path or "",
                )
            )
        return expanded

    def _failed_topology(
        self,
        index: int,
        code: str,
        category: str,
        summary: str,
        raw_log: str,
        return_code: int | None = None,
    ) -> dict[str, Any]:
        topology = {
            "source": "external_cello_wrapper",
            "verilog_index": index,
            "verilog": code,
            "score": 0.0,
            "mapping_status": "MAPPING_FAILED",
            "error_type": "PART_ERROR",
            "orthogonality_score": 1.0,
            "cello_assignment_score": 0.0,
            "cello_buildable": False,
            "mapping_error_category": category,
            "mapping_error_summary": summary,
            "raw_error_log": _truncate_error_log(raw_log, self.max_log_chars),
            "return_code": return_code,
        }
        topology.update(_topology_cello_metrics(topology))
        return topology


def _topology_cello_metrics(topology: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluate_cello_constraints(topology)
    return {
        "orthogonality_score": metrics["orthogonality_score"],
        "cello_assignment_score": metrics["cello_assignment_score"],
        "cello_buildable": metrics["cello_buildable"],
        "toxicity": metrics["toxicity"],
        "toxicity_score": metrics["toxicity_score"],
        "cello_constraint_report": metrics,
    }


def _classify_error_log(log: str) -> str:
    lowered = log.lower()
    if "ucf" in lowered and any(token in lowered for token in ("missing", "mismatch", "incompatible", "constraint")):
        return "UCF_INCOMPATIBLE"
    if any(token in lowered for token in ("syntax error", "parse error", "verilog")):
        return "VERILOG_SYNTAX_ERROR"
    if any(token in lowered for token in ("unsupported gate", "unsupported primitive", "cannot map")):
        return "UNSUPPORTED_GATE"
    if any(token in lowered for token in ("part unavailable", "no gate", "not found in library")):
        return "PART_UNAVAILABLE"
    if "timeout" in lowered:
        return "TIMEOUT"
    return "MAPPING_FAILED"


def _summarize_error(category: str, log: str) -> str:
    first_line = next((line.strip() for line in log.splitlines() if line.strip()), "")
    last_exception = next(
        (
            line.strip()
            for line in reversed(log.splitlines())
            if any(token in line.lower() for token in ("exception", "error", "failed"))
        ),
        "",
    )
    detail = last_exception or first_line or "Cello mapping failed."
    return f"{category}: {detail[:500]}"


def _truncate_error_log(log: str, max_chars: int = 4000) -> str:
    text = (log or "").strip()
    if len(text) <= max_chars:
        return text
    head_len = max(500, max_chars // 2)
    tail_len = max(500, max_chars - head_len - 120)
    head = text[:head_len].rstrip()
    tail = text[-tail_len:].lstrip()
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n... [truncated {omitted} chars of middle stack trace/log] ...\n\n{tail}"
