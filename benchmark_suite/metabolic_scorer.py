from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from benchmark_suite.base_evaluator import BaseEvaluator, EvaluationResult

LOGIC_GATE_NAMES = ("and", "nand", "or", "nor", "xor", "xnor", "not", "buf")
DEFAULT_IDEAL_GATE_LIMIT = 3
DEFAULT_DECAY_RATE = 0.35


def _strip_verilog_comments(verilog: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", "", verilog, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", without_block_comments, flags=re.MULTILINE)


def count_logic_gates(verilog: str) -> int:
    if not verilog.strip():
        raise ValueError("Verilog content is empty.")

    source = _strip_verilog_comments(verilog)
    gate_pattern = "|".join(LOGIC_GATE_NAMES)
    primitive_without_instance = re.compile(rf"\b(?:{gate_pattern})\s*\(", flags=re.IGNORECASE)
    primitive_with_instance = re.compile(
        rf"\b(?:{gate_pattern})\s+(?:#\s*\([^;]*?\)\s*)?[A-Za-z_][\w$]*\s*\(",
        flags=re.IGNORECASE,
    )
    return len(primitive_without_instance.findall(source)) + len(primitive_with_instance.findall(source))


def metabolic_burden_score(
    gate_count: int,
    ideal_gate_limit: int = DEFAULT_IDEAL_GATE_LIMIT,
    decay_rate: float = DEFAULT_DECAY_RATE,
) -> float:
    if gate_count < 0:
        raise ValueError("gate_count must be non-negative.")
    if ideal_gate_limit < 0:
        raise ValueError("ideal_gate_limit must be non-negative.")
    if decay_rate < 0:
        raise ValueError("decay_rate must be non-negative.")
    excess_gates = max(0, gate_count - ideal_gate_limit)
    return float(math.exp(-decay_rate * excess_gates))


def _read_verilog_source(candidate: dict[str, Any]) -> tuple[str | None, str | None]:
    inline_verilog = candidate.get("verilog") or candidate.get("verilog_code")
    if inline_verilog is not None:
        return str(inline_verilog), "inline_verilog"

    for key in ("verilog_path", "cello_output_path", "output_path", "path"):
        raw_path = candidate.get(key)
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if path.is_dir():
            verilog_files = sorted(path.glob("*.v"))
            if not verilog_files:
                raise FileNotFoundError(f"No .v files found in directory: {path}")
            path = verilog_files[0]
        try:
            return path.read_text(encoding="utf-8"), str(path)
        except OSError as exc:
            raise OSError(f"Failed to read Verilog source from {path}: {exc}") from exc

    return None, None


class MetabolicBurdenEvaluator(BaseEvaluator):
    def __init__(
        self,
        ideal_gate_limit: int = DEFAULT_IDEAL_GATE_LIMIT,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ):
        self.ideal_gate_limit = ideal_gate_limit
        self.decay_rate = decay_rate

    def evaluate(self, candidate: dict[str, Any]) -> EvaluationResult:
        try:
            verilog, source = _read_verilog_source(candidate)
            if verilog is not None:
                gate_count = count_logic_gates(verilog)
            elif candidate.get("gate_count") is not None:
                gate_count = int(candidate["gate_count"])
                source = "candidate.gate_count"
            else:
                return EvaluationResult(
                    score=1.0,
                    details={
                        "metric": "metabolic_burden",
                        "status": "skipped",
                        "reason": "No Verilog source or gate_count was provided.",
                    },
                    metabolic_burden_score=1.0,
                    gate_count=0,
                    complexity_penalty=0.0,
                )

            score = metabolic_burden_score(
                gate_count,
                ideal_gate_limit=self.ideal_gate_limit,
                decay_rate=self.decay_rate,
            )
            complexity_penalty = 1.0 - score
            return EvaluationResult(
                score=score,
                details={
                    "metric": "metabolic_burden",
                    "status": "ok",
                    "source": source,
                    "ideal_gate_limit": self.ideal_gate_limit,
                    "decay_rate": self.decay_rate,
                },
                metabolic_burden_score=score,
                gate_count=gate_count,
                complexity_penalty=complexity_penalty,
            )
        except (OSError, ValueError, TypeError) as exc:
            return EvaluationResult(
                score=0.0,
                details={
                    "metric": "metabolic_burden",
                    "status": "error",
                    "error": str(exc),
                },
                metabolic_burden_score=0.0,
                gate_count=0,
                complexity_penalty=1.0,
            )


def score_metabolic_burden(candidate: dict[str, Any]) -> EvaluationResult:
    return MetabolicBurdenEvaluator().evaluate(candidate)
