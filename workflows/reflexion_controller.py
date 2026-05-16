import logging
import uuid
from typing import Any

from benchmark_suite.benchmark_controller import evaluate_candidate
from schemas.state import DesignState, SearchNode

# Assuming agent imports are handled by an app or factory, 
# but we will define the workflow loop here.

MAX_CONSECUTIVE_ERROR_TYPE = 3


def _latest_feedback(node: SearchNode) -> str:
    return node.critic_feedbacks[-1] if node.critic_feedbacks else ""


def _topology_summary(topology: dict | None) -> dict:
    if not topology:
        return {}
    summary_keys = [
        "score",
        "mapping_status",
        "ode_status",
        "dynamic_margin",
        "gate_count",
        "metabolic_burden_score",
        "complexity_penalty",
        "robustness_score",
        "collapsed",
        "robustness_collapsed",
        "signal_to_noise_ratio",
        "snr",
        "monte_carlo_runs",
        "monte_carlo_samples",
        "temporal_score",
        "rise_time",
        "weighted_total_score",
        "orthogonality_score",
        "cello_assignment_score",
        "cello_buildable",
        "toxicity",
        "toxicity_score",
        "semantic_faithfulness_score",
        "missed_edge_cases",
        "scoring_model",
        "verilog_index",
    ]
    return {key: topology[key] for key in summary_keys if key in topology}


def _apply_weighted_benchmark(topology: dict[str, Any]) -> None:
    existing_report = topology.get("benchmark_report")
    if not isinstance(existing_report, dict):
        existing_report = {}
    weighted_report = evaluate_candidate(topology)
    existing_details = existing_report.get("details", [])
    weighted_details = weighted_report.get("details", [])
    if not isinstance(existing_details, list):
        existing_details = []
    if not isinstance(weighted_details, list):
        weighted_details = []

    topology.update(weighted_report)
    topology["benchmark_report"] = {
        **weighted_report,
        "details": existing_details + weighted_details,
        "ode_report": existing_report,
    }


def _record_failed_attempt(state: DesignState, node: SearchNode) -> None:
    attempt = {
        "node_id": node.node_id,
        "search_mode": node.search_mode,
        "score": node.score,
        "metabolic_burden_score": node.metabolic_burden_score,
        "gate_count": node.gate_count,
        "complexity_penalty": node.complexity_penalty,
        "robustness_score": node.robustness_score,
        "signal_to_noise_ratio": node.signal_to_noise_ratio,
        "monte_carlo_runs": node.monte_carlo_runs,
        "temporal_score": node.temporal_score,
        "rise_time": node.rise_time,
        "orthogonality_score": node.orthogonality_score,
        "cello_assignment_score": node.cello_assignment_score,
        "cello_buildable": node.cello_buildable,
        "semantic_faithfulness_score": node.semantic_faithfulness_score,
        "missed_edge_cases": node.missed_edge_cases,
        "error_type": node.error_type,
        "feedback": _latest_feedback(node),
        "best_topology": _topology_summary(node.best_topology),
    }
    node.failed_attempts.append(attempt)
    state.failed_attempts.append(attempt)


def _consecutive_error_count(state: DesignState, error_type: str) -> int:
    count = 0
    for attempt in reversed(state.failed_attempts):
        if attempt.get("error_type") != error_type:
            break
        count += 1
    return count


def _pause_for_human_input(
    state: DesignState,
    node: SearchNode | None,
    reason: str,
    prompt: str,
) -> DesignState:
    if node:
        node.status = "Needs_Human_Input"
    state.requires_human_input = True
    state.pause_reason = reason
    state.human_feedback_prompt = prompt
    state.is_completed = False
    logging.warning("Workflow paused for human input: %s", reason)
    return state

def run_reflexion_workflow(
    state: DesignState,
    builder,
    translator,
    cello_wrapper,
    batch_ode_simulator,
    critic,
    consolidator,
    skill_retriever,
    data_miner=None,
    skill_extractor=None,
) -> DesignState:
    state.requires_human_input = False
    state.pause_reason = None
    state.human_feedback_prompt = None
    
    if not state.active_frontier and not state.tree_nodes:
        root_node = SearchNode(node_id="root", search_mode="Exploration")
        state.tree_nodes["root"] = root_node
        state.active_frontier.append("root")
    
    while state.active_frontier:
        if state.used_budget >= state.compute_budget:
            return _pause_for_human_input(
                state,
                state.tree_nodes.get(state.current_node_id) if state.current_node_id else None,
                "compute_budget_exceeded",
                (
                    "The reflexion workflow reached the compute budget before approval. "
                    "Please provide additional design constraints, acceptable trade-offs, "
                    "or a preferred fallback topology before resuming."
                ),
            )
            
        current_node_id = state.active_frontier.pop(0)
        node = state.tree_nodes[current_node_id]
        state.current_node_id = current_node_id
        
        logging.info(f"Processing Node: {current_node_id} (Mode: {node.search_mode})")
        
        # Determine parameters based on search_mode
        if node.search_mode == "Exploration":
            temperature = 0.7
            if skill_retriever:
                state.rag_context = skill_retriever.retrieve_skills(state.user_intent, mode="Exploration")
        elif node.search_mode == "Repair":
            temperature = 0.1
            if skill_retriever:
                state.rag_context = skill_retriever.retrieve_skills(state.user_intent, mode="Repair")
        elif node.search_mode == "Exploitation":
            temperature = 0.1
            # Inherit rag_context from parent if possible, or just default
            
        # 1. Builder
        if node.search_mode != "Exploitation":
            builder.kwargs["temperature"] = temperature
            state = builder.run(state)
            if state.last_error:
                node.status = "Dead_End"
                node.last_error = state.last_error
                continue
                
        # 2. Translator
        translator.kwargs["temperature"] = temperature
        state = translator.run(state)
        if state.last_error:
            node.status = "Dead_End"
            node.last_error = state.last_error
            continue
            
        # 3. Cello (assuming it reads node.verilog_codes via state)
        state = cello_wrapper.run(state)
        if state.last_error:
            node.status = "Dead_End"
            node.last_error = state.last_error
            continue
            
        # 4. Optional biokinetic data mining, then batch ODE/DAE simulation
        if data_miner:
            state = data_miner.run(state)
            if state.last_error:
                node.status = "Dead_End"
                node.last_error = state.last_error
                continue

        state = batch_ode_simulator.run(state)

        for topo in node.candidate_topologies:
            _apply_weighted_benchmark(topo)
        
        # Evaluate Best Topology inside the node
        best_topo = None
        best_score = -9999.0
        for topo in node.candidate_topologies:
            if topo.get("score", -9999) > best_score:
                best_score = topo.get("score", -9999)
                best_topo = topo
                
        node.best_topology = best_topo
        node.score = best_score
        node.sync_evaluation_metrics(best_topo)
        if best_topo:
            state.best_topology = best_topo
            
        # 5. Critic
        state = critic.run(state)
        node.status = "Evaluated"
        
        # Branching based on Critic evaluation
        if node.is_approved:
            logging.info(f"Node {current_node_id} PASS! Goal reached.")
            node.status = "Pass"
            state.is_completed = True
            break
            
        # If not approved, handle errors
        state.used_budget += 1
        _record_failed_attempt(state, node)

        if state.requires_human_input:
            return _pause_for_human_input(
                state,
                node,
                "critic_requested_human_input",
                _latest_feedback(node)
                or "The critic marked this design as requiring human guidance.",
            )

        if _consecutive_error_count(state, node.error_type) >= MAX_CONSECUTIVE_ERROR_TYPE:
            return _pause_for_human_input(
                state,
                node,
                "repeated_error_type",
                (
                    f"The workflow encountered {node.error_type} repeatedly. "
                    "Please add constraints that clarify the intended logic, allowed parts, "
                    "or acceptable physical trade-offs."
                ),
            )
        
        if node.error_type in ["LOGIC_ERROR", "BOTH"]:
            # Generate Repair child
            repair_id = f"{current_node_id}_repair_{uuid.uuid4().hex[:4]}"
            repair_node = SearchNode(
                node_id=repair_id, 
                parent_id=current_node_id, 
                search_mode="Repair",
                critic_feedbacks=node.critic_feedbacks.copy(), # Pass historical context
                failed_attempts=node.failed_attempts.copy(),
                error_type=node.error_type
            )
            node.children_ids.append(repair_id)
            state.tree_nodes[repair_id] = repair_node
            state.active_frontier.append(repair_id)
            
            # If budget permits, generate Exploration child
            if state.used_budget < state.compute_budget - 1:
                explore_id = f"{current_node_id}_explore_{uuid.uuid4().hex[:4]}"
                explore_node = SearchNode(
                    node_id=explore_id,
                    parent_id=current_node_id,
                    search_mode="Exploration",
                    critic_feedbacks=node.critic_feedbacks.copy(),
                    failed_attempts=node.failed_attempts.copy(),
                    error_type=node.error_type
                )
                node.children_ids.append(explore_id)
                state.tree_nodes[explore_id] = explore_node
                state.active_frontier.append(explore_id)
                
        elif node.error_type == "PART_ERROR":
            # If score is high but mapping failed, generate Exploitation
            if state.used_budget <= state.compute_budget:
                exploit_id = f"{current_node_id}_exploit_{uuid.uuid4().hex[:4]}"
                exploit_node = SearchNode(
                    node_id=exploit_id,
                    parent_id=current_node_id,
                    search_mode="Exploitation",
                    logic_proposals=node.logic_proposals.copy(), # Inherit logic
                    critic_feedbacks=node.critic_feedbacks.copy(),
                    failed_attempts=node.failed_attempts.copy(),
                    error_type=node.error_type
                )
                node.children_ids.append(exploit_id)
                state.tree_nodes[exploit_id] = exploit_node
                state.active_frontier.append(exploit_id)
        else:
            return _pause_for_human_input(
                state,
                node,
                "no_recoverable_route",
                (
                    "The critic rejected the design but did not provide a recoverable "
                    "LOGIC_ERROR or PART_ERROR route. Please provide additional guidance."
                ),
            )

    # Degradation / Fallback selection
    if not state.is_completed:
        if not state.requires_human_input:
            return _pause_for_human_input(
                state,
                state.tree_nodes.get(state.current_node_id) if state.current_node_id else None,
                "frontier_exhausted",
                (
                    "All reflexion branches were exhausted before approval. "
                    "Please provide extra constraints or select a fallback candidate."
                ),
            )
        logging.warning("Workflow paused before approval. Best topology remains available as fallback.")
        best_node = None
        highest_score = -float('inf')
        for nid, n in state.tree_nodes.items():
            if n.score > highest_score and n.best_topology:
                highest_score = n.score
                best_node = n
        if best_node and best_node.best_topology:
            state.best_topology = best_node.best_topology
            state.current_node_id = best_node.node_id
            
    # 6. Consolidator
    state = consolidator.run(state)
    if skill_extractor:
        state = skill_extractor.run(state)
    return state
