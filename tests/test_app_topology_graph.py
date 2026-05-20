from __future__ import annotations

from app import _verilog_to_gate_graph


def test_gate_graph_parses_primitive_gates() -> None:
    graph = _verilog_to_gate_graph(
        "module c(input A, input B, output Y); wire n; not(n, B); and(Y, A, n); endmodule"
    )

    assert graph["ok"] is True
    dot = graph["dot"]
    assert '"B" -> "NOT_1"' in dot
    assert '"NOT_1" -> "n"' in dot
    assert '"A" -> "AND_2"' in dot
    assert '"n" -> "AND_2"' in dot
    assert '"AND_2" -> "Y"' in dot


def test_gate_graph_parses_simple_assign() -> None:
    graph = _verilog_to_gate_graph("module c(input A, output Y); assign Y = A; endmodule")

    assert graph["ok"] is True
    assert '"A" -> "Y"' in graph["dot"]


def test_gate_graph_parses_expression_assign() -> None:
    graph = _verilog_to_gate_graph("module c(input A, input B, output Y); assign Y = A & ~B; endmodule")

    assert graph["ok"] is True
    dot = graph["dot"]
    assert '"B" -> "NOT_2"' in dot
    assert '"NOT_2" -> "AND_1"' in dot
    assert '"A" -> "AND_1"' in dot
    assert '"AND_1" -> "Y"' in dot


def test_gate_graph_handles_malformed_verilog_without_exception() -> None:
    graph = _verilog_to_gate_graph("module c(input A, output Y); this is not supported endmodule")

    assert graph["ok"] is False
    assert "無法解析 gate graph" in graph["message"]


def test_gate_graph_handles_missing_verilog() -> None:
    graph = _verilog_to_gate_graph("")

    assert graph["ok"] is False
    assert "沒有 Verilog" in graph["message"]
