from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SearchMode = Literal["Exploration", "Repair", "Exploitation"]
NodeStatus = Literal["Pending", "Evaluated", "Pass", "Dead_End"]
ErrorType = Literal["LOGIC_ERROR", "PART_ERROR", "BOTH", "NONE"]


@dataclass
class SearchNode:
    node_id: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    search_mode: SearchMode = "Exploration"
    status: NodeStatus = "Pending"

    logic_proposals: list[str] = field(default_factory=list)
    verilog_codes: list[str] = field(default_factory=list)
    candidate_topologies: list[dict[str, Any]] = field(default_factory=list)
    current_topology: str = ""
    best_topology: dict[str, Any] | None = None
    score: float = -float("inf")

    critic_feedbacks: list[str] = field(default_factory=list)
    is_approved: bool = False
    error_type: ErrorType = "NONE"
    last_error: str | None = None


@dataclass
class DesignState:
    user_intent: str = ""
    host_organism: str = "Escherichia coli"

    tree_nodes: dict[str, SearchNode] = field(default_factory=dict)
    active_frontier: list[str] = field(default_factory=list)
    current_node_id: str | None = None
    compute_budget: int = 6
    used_budget: int = 0

    rag_context: str = ""
    skill_library_context: str = ""
    seed_debate_transcript: str = ""

    logic_proposals: list[str] = field(default_factory=list)
    verilog_codes: list[str] = field(default_factory=list)
    candidate_topologies: list[dict[str, Any]] = field(default_factory=list)
    current_topology: str = ""
    best_topology: dict[str, Any] | None = None

    critic_feedbacks: list[str] = field(default_factory=list)
    is_approved: bool = False
    error_type: ErrorType = "NONE"
    is_completed: bool = False
    iteration_count: int = 0
    last_error: str | None = None

    @property
    def latest_critic_feedback(self) -> str:
        if self.current_node_id and self.current_node_id in self.tree_nodes:
            feedbacks = self.tree_nodes[self.current_node_id].critic_feedbacks
            if feedbacks:
                return feedbacks[-1]
        return self.critic_feedbacks[-1] if self.critic_feedbacks else ""
