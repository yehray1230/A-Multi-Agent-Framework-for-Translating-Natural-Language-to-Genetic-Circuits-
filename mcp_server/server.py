from __future__ import annotations

from typing import Any

from mcp_server.service import (
    design_circuit_quick,
    evaluate_verilog,
    get_design_run_result as service_get_design_run_result,
    get_design_run_status as service_get_design_run_status,
    start_design_run as service_start_design_run,
    summarize_design_state,
)


try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional runtime package
    FastMCP = None  # type: ignore[assignment]
    MCP_IMPORT_ERROR = exc
else:
    MCP_IMPORT_ERROR = None


if FastMCP is not None:
    mcp = FastMCP("genetic-circuit-workflow")

    @mcp.tool()
    def design_genetic_circuit_quick(
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
        """Run a compact genetic circuit design workflow and write artifacts."""
        return design_circuit_quick(
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
        )

    @mcp.tool()
    def evaluate_cello_verilog(
        verilog: str,
        user_intent: str = "Evaluate a Cello-compatible genetic circuit.",
        host_organism: str = "Escherichia coli",
        enable_ode: bool = True,
        monte_carlo_samples: int = 1,
        output_dir: str | None = None,
        cello_command: str | None = None,
        ucf_path: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate existing Cello-compatible Verilog without calling an LLM."""
        return evaluate_verilog(
            verilog=verilog,
            user_intent=user_intent,
            host_organism=host_organism,
            enable_ode=enable_ode,
            monte_carlo_samples=monte_carlo_samples,
            output_dir=output_dir,
            cello_command=cello_command,
            ucf_path=ucf_path,
        )

    @mcp.tool()
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
    ) -> dict[str, Any]:
        """Start a background genetic circuit design run and return a run_id."""
        return service_start_design_run(
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
        )

    @mcp.tool()
    def get_design_run_status(run_id: str) -> dict[str, Any]:
        """Return queued/running/completed status for a background design run."""
        return service_get_design_run_status(run_id)

    @mcp.tool()
    def get_design_run_result(run_id: str) -> dict[str, Any]:
        """Return the final result and artifacts for a background design run."""
        return service_get_design_run_result(run_id)

    @mcp.tool()
    def summarize_mcp_design_state(state_json: dict[str, Any]) -> dict[str, Any]:
        """Summarize a state JSON previously produced by this MCP adapter."""
        return summarize_design_state(state_json)


def main() -> None:
    if FastMCP is None:
        raise SystemExit(
            "The optional 'mcp' package is not installed. Install it with `pip install mcp` "
            "or add `mcp>=1.0` to your environment before running this server."
        ) from MCP_IMPORT_ERROR
    mcp.run()


if __name__ == "__main__":
    main()
