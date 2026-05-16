from __future__ import annotations

from schemas.state import DesignState
from agents.base import AgentProtocol


class ConsolidatorAgent(AgentProtocol):
    def run(self, state: DesignState) -> DesignState:
        if state.current_node_id and state.current_node_id in state.tree_nodes:
            node = state.tree_nodes[state.current_node_id]
            if node.best_topology:
                state.best_topology = node.best_topology
            state.is_approved = node.is_approved
            state.error_type = node.error_type
            if node.critic_feedbacks:
                state.critic_feedbacks = node.critic_feedbacks.copy()
        return state
