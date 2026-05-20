# MCP Prototype

This folder is an additive MCP adapter for the existing genetic circuit workflow.
It does not replace or modify the Streamlit UI.

## Tools

- `design_genetic_circuit_quick`: runs a compact Builder -> Translator -> Cello -> ODE -> Critic workflow.
- `start_design_run`: starts a longer background design run and returns `run_id`.
- `get_design_run_status`: polls a background run.
- `get_design_run_result`: returns a finished background run result.
- `evaluate_cello_verilog`: evaluates existing Cello-compatible Verilog without calling an LLM.
- `summarize_mcp_design_state`: compresses a saved state JSON into an Agent-friendly summary.

## Runtime Configuration

The adapter reads model settings from arguments or environment variables:

```powershell
$env:OPENAI_API_KEY="..."
$env:LITELLM_MODEL="gpt-5.4-mini"
python -m mcp_server.server
```

The `mcp` Python package is optional for local tests, but required to run the MCP server:

```powershell
pip install mcp
```

## Artifacts

Each run writes files under `outputs/mcp_runs` by default:

- `state.json`
- `summary.json`
- `best_topology.json`
- `best_design.v`
- `run_summary.md`
- `score_breakdown.png`
- `ode_summary.png` when ODE metrics are available

Background run metadata and results are written under `outputs/mcp_runs/async_runs`.
The actual workflow artifacts remain in the normal run folder referenced by `workflow_run_dir`.

## Notes

The asynchronous prototype uses an in-process thread pool. It is suitable for a local MCP server
session, but it is not a durable distributed queue. If the MCP server process stops, completed
results remain readable from disk, while actively running jobs stop with the process.
