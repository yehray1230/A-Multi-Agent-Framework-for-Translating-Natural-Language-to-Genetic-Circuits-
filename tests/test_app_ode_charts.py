from __future__ import annotations

from app import _ode_trace_rows, _valid_ode_trace


def test_valid_ode_trace_requires_time_and_output_series() -> None:
    assert _valid_ode_trace({"time": [0.0, 1.0], "output_protein": [0.0, 2.0]}) is True
    assert _valid_ode_trace({"time": [0.0], "output_protein": []}) is False
    assert _valid_ode_trace({}) is False


def test_ode_trace_rows_aligns_available_series() -> None:
    rows = _ode_trace_rows(
        {
            "time": [0.0, 1.0],
            "output_protein": [0.0, 10.0],
            "total_mrna": [1.0, 2.0],
            "rnap_occupancy": [0.2, 0.4],
        }
    )

    assert rows == [
        {"time": 0.0, "output_protein": 0.0, "total_mrna": 1.0, "rnap_occupancy": 0.2},
        {"time": 1.0, "output_protein": 10.0, "total_mrna": 2.0, "rnap_occupancy": 0.4},
    ]
