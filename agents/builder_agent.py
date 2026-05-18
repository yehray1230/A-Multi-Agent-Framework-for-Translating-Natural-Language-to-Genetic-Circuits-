from __future__ import annotations

import json

from agents.base import AgentProtocol
from pydantic import BaseModel, Field, ValidationError
from schemas.state import DesignState
from tools.skill_retriever import SkillRetriever
from utils.llm_utils import call_llm


REQUIRED_STRATEGIES = (
    "gate_count_optimization",
    "depth_optimization",
    "robustness_strategy",
)


class BuilderProposal(BaseModel):
    strategy_name: str
    optimization_goal: str
    truth_table_or_logic_matrix: list[dict[str, str | int | bool]] | str
    logic_blueprint: str
    verilog_draft: str
    translator_directives: list[str] = Field(default_factory=list)


class BuilderOutput(BaseModel):
    gate_count_optimization: BuilderProposal
    depth_optimization: BuilderProposal
    robustness_strategy: BuilderProposal


def _build_system_prompt(
    state: DesignState,
    skill_retriever: SkillRetriever | None = None,
    skill_file_path: str = "邏輯設計skill.json",
) -> str:
    system_prompt = f"""You are Bio-Logic Architect, a synthetic biology design agent.

Goal:
- Convert the user's intent into exactly three Cello-compatible genetic circuit design proposals.
- Keep designs combinational and suitable for Cello CAD technology mapping.
- Prefer designs that can later be translated to simple Verilog gates (`and`, `or`, `not`, `nand`, `nor`, `xor`, `xnor`, or `assign`).

Target host:
{state.host_organism}

User intent:
{state.user_intent}

Design constraints:
- Return exactly three top-level strategies: `gate_count_optimization`, `depth_optimization`, and `robustness_strategy`.
- `gate_count_optimization` must minimize the number of Boolean gates and biological repressors.
- `depth_optimization` must minimize logic depth and signal propagation delay.
- `robustness_strategy` must prioritize biological part compatibility, toxicity avoidance, and dynamic robustness while remaining Cello-compatible.
- Each strategy must include a truth table or logic matrix and a raw Verilog draft.
- Avoid sequential logic, delay syntax, clocks, memory elements, or Verilog constructs Cello cannot map.
- If you use motifs such as pulse-like behavior or feed-forward logic, describe them in a way the Translator can preserve as structural combinational logic.
- Include translator directives when a design requires structural instantiation rather than a simplified Boolean expression.

Output only valid JSON. No Markdown, no comments.

Required schema:
{{
  "gate_count_optimization": {{
    "strategy_name": "Gate-Count Optimization",
    "optimization_goal": "minimize Boolean gates and repressor count",
    "truth_table_or_logic_matrix": [
      {{"A": 0, "B": 0, "Y": 0}},
      {{"A": 0, "B": 1, "Y": 0}},
      {{"A": 1, "B": 0, "Y": 1}},
      {{"A": 1, "B": 1, "Y": 0}}
    ],
    "logic_blueprint": "Y = A AND NOT B",
    "verilog_draft": "module genetic_circuit(input A, input B, output Y); wire not_b; not(not_b, B); and(Y, A, not_b); endmodule",
    "translator_directives": ["MINIMIZE_GATE_COUNT"]
  }},
  "depth_optimization": {{
    "strategy_name": "Depth Optimization",
    "optimization_goal": "minimize logic depth and delay",
    "truth_table_or_logic_matrix": "compact truth table or Karnaugh-map-style matrix",
    "logic_blueprint": "Y = A OR B",
    "verilog_draft": "module genetic_circuit(input A, input B, output Y); or(Y, A, B); endmodule",
    "translator_directives": []
  }},
  "robustness_strategy": {{
    "strategy_name": "Robustness Strategy",
    "optimization_goal": "maximize part compatibility and dynamic robustness",
    "truth_table_or_logic_matrix": "compact truth table or logic matrix",
    "logic_blueprint": "Y = robust combinational motif using A and B",
    "verilog_draft": "module genetic_circuit(input A, input B, output Y); wire not_b; not(not_b, B); and(Y, A, not_b); endmodule",
    "translator_directives": ["PRESERVE_LOGIC"]
  }}
}}
"""
    if state.human_constraints:
        system_prompt += (
            "\nHuman-in-the-loop constraints to satisfy in this revision:\n"
            + "\n".join(f"- {constraint}" for constraint in state.human_constraints)
            + "\n"
        )

    skill_context = _retrieve_skill_context(state, skill_retriever, skill_file_path)
    if skill_context:
        system_prompt += (
            "\n=== Retrieved Successful Design Skills ===\n"
            f"{skill_context}\n"
            "Please apply the successful skills above, including Boolean decomposition strategies "
            "and known risks, to this topology generation.\n"
        )
    elif getattr(state, "skill_library_context", None):
        system_prompt += f"\nReusable design skill context:\n{state.skill_library_context}\n"

    if getattr(state, "seed_debate_transcript", None):
        system_prompt += (
            "\nPrevious debate and design iterations (for context):\n"
            f"{state.seed_debate_transcript}\n"
        )

    if state.latest_critic_feedback:
        system_prompt += (
            "\nCritic feedback to address in this revision:\n"
            f"{state.latest_critic_feedback}\n"
        )

    node_id = state.current_node_id
    if node_id and node_id in state.tree_nodes:
        mode = state.tree_nodes[node_id].search_mode
        system_prompt += f"\nCurrent Search Mode: {mode}\n"
        if mode == "Repair":
            system_prompt += "You are in REPAIR mode. You must rethink the logic based on the critic feedback.\n"

    return system_prompt


def _retrieve_skill_context(
    state: DesignState,
    skill_retriever: SkillRetriever | None,
    skill_file_path: str,
) -> str:
    snippets: list[str] = []
    if state.rag_context:
        snippets.append(state.rag_context)

    retriever = skill_retriever
    if retriever is None:
        try:
            retriever = SkillRetriever.from_json_file(skill_file_path)
        except Exception:
            retriever = None

    if retriever:
        query = " ".join(
            value
            for value in [
                state.user_intent,
                state.current_topology,
                state.latest_critic_feedback,
                " ".join(state.logic_proposals[:2]),
            ]
            if value
        )
        retrieved = retriever.retrieve_skills(query or state.user_intent, mode="Exploration", k=5)
        if retrieved and retrieved not in snippets:
            snippets.append(retrieved)
    return "\n".join(snippets)


def call_builder(
    state: DesignState,
    api_key: str | None,
    model_name: str,
    api_base: str | None = None,
    force_zero_shot: bool = False,
    temperature: float = 0.7,
    skill_retriever: SkillRetriever | None = None,
    skill_file_path: str = "邏輯設計skill.json",
    **kwargs,
) -> DesignState:
    system_prompt = _build_system_prompt(
        state,
        skill_retriever=skill_retriever,
        skill_file_path=skill_file_path,
    )
    user_content = (
        "Generate three alternative Cello-compatible genetic circuit proposals "
        "for the target intent. Output only the JSON object described in the system prompt."
    )

    response = call_llm(
        api_key=api_key,
        model_name=model_name,
        system_prompt=system_prompt,
        user_content=user_content,
        api_base=api_base,
        temperature=temperature,
        **kwargs,
    )

    if response.startswith("ERROR:"):
        state.last_error = response
        return state

    try:
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            json_str = response[start : end + 1]
            data = json.loads(json_str)
            if set(data.keys()) != set(REQUIRED_STRATEGIES):
                missing = sorted(set(REQUIRED_STRATEGIES) - set(data.keys()))
                extra = sorted(set(data.keys()) - set(REQUIRED_STRATEGIES))
                raise ValueError(
                    f"expected exactly {list(REQUIRED_STRATEGIES)}; "
                    f"missing={missing}, extra={extra}"
                )
            validated = BuilderOutput.model_validate(data)
            proposals = [
                json.dumps(getattr(validated, strategy).model_dump(), ensure_ascii=False)
                for strategy in REQUIRED_STRATEGIES
            ]

            node_id = state.current_node_id
            if node_id and node_id in state.tree_nodes:
                node = state.tree_nodes[node_id]
                node.logic_proposals = proposals or [json_str]
                node.current_topology = node.logic_proposals[0] if node.logic_proposals else ""
                state.logic_proposals = node.logic_proposals
                state.current_topology = node.current_topology
            else:
                state.logic_proposals = proposals or [json_str]
                state.current_topology = state.logic_proposals[0] if state.logic_proposals else ""
        else:
            state.logic_proposals = [response]
            if state.current_node_id and state.current_node_id in state.tree_nodes:
                state.tree_nodes[state.current_node_id].logic_proposals = state.logic_proposals

        state.last_error = None

    except Exception as exc:
        detail = exc.errors() if isinstance(exc, ValidationError) else str(exc)
        state.last_error = (
            f"ERROR: Builder response failed the three-strategy JSON schema: {detail}\n"
            f"Raw response:\n{response}"
        )

    return state


class BuilderAgent(AgentProtocol):
    def __init__(
        self,
        api_key: str | None,
        model_name: str,
        api_base: str | None = None,
        force_zero_shot: bool = False,
        skill_retriever: SkillRetriever | None = None,
        skill_file_path: str = "邏輯設計skill.json",
        **kwargs,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        self.force_zero_shot = force_zero_shot
        self.skill_retriever = skill_retriever
        self.skill_file_path = skill_file_path
        self.kwargs = kwargs

    def run(self, state: DesignState) -> DesignState:
        state.iteration_count += 1
        state = call_builder(
            state=state,
            api_key=self.api_key,
            model_name=self.model_name,
            api_base=self.api_base,
            force_zero_shot=self.force_zero_shot,
            skill_retriever=self.skill_retriever,
            skill_file_path=self.skill_file_path,
            **self.kwargs,
        )
        if state.last_error:
            state.is_completed = False
        return state
