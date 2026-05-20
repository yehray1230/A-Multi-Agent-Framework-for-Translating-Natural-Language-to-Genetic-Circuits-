from __future__ import annotations

from app import _branch_reason_for_node, _search_next_step_summary, _search_path_to_node
from schemas.state import DesignState, SearchNode


def _state_with_path() -> DesignState:
    state = DesignState()
    root = SearchNode(node_id="root", search_mode="Exploration", error_type="LOGIC_ERROR")
    repair = SearchNode(
        node_id="root_repair_1",
        parent_id="root",
        search_mode="Repair",
        error_type="PART_ERROR",
    )
    exploit = SearchNode(
        node_id="root_repair_1_exploit_1",
        parent_id="root_repair_1",
        search_mode="Exploitation",
    )
    root.children_ids.append(repair.node_id)
    repair.children_ids.append(exploit.node_id)
    state.tree_nodes = {
        root.node_id: root,
        repair.node_id: repair,
        exploit.node_id: exploit,
    }
    state.current_node_id = repair.node_id
    return state


def test_search_path_to_node_returns_ordered_path() -> None:
    state = _state_with_path()

    path = _search_path_to_node(state, "root_repair_1_exploit_1")

    assert [node.node_id for node in path] == ["root", "root_repair_1", "root_repair_1_exploit_1"]


def test_branch_reason_for_logic_repair() -> None:
    state = _state_with_path()

    assert _branch_reason_for_node(state, "root_repair_1") == "邏輯問題 -> 修正"


def test_branch_reason_for_part_exploitation() -> None:
    state = _state_with_path()

    assert _branch_reason_for_node(state, "root_repair_1_exploit_1") == "元件問題 -> 元件最佳化"


def test_next_step_summary_uses_first_frontier_node() -> None:
    state = _state_with_path()
    state.active_frontier = ["root_repair_1_exploit_1"]

    summary = _search_next_step_summary(state)

    assert summary["level"] == "info"
    assert "root_repair_1_exploit_1" in summary["text"]
    assert "元件問題 -> 元件最佳化" in summary["text"]


def test_search_helpers_handle_empty_and_orphan_nodes() -> None:
    empty_state = DesignState()
    assert _search_path_to_node(empty_state, "missing") == []
    assert _branch_reason_for_node(empty_state, "missing") == "找不到節點"

    orphan_state = DesignState()
    orphan_state.tree_nodes["orphan"] = SearchNode(node_id="orphan", parent_id="missing")
    assert [node.node_id for node in _search_path_to_node(orphan_state, "orphan")] == ["orphan"]
    assert _branch_reason_for_node(orphan_state, "orphan") == "搜尋起點"
