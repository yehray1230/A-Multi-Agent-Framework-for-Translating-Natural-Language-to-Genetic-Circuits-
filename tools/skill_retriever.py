from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SKILL_LIBRARY_PATH = "邏輯設計skill.json"


@dataclass
class SkillRetriever:
    skills: list[dict[str, Any]] = field(default_factory=list)
    min_confidence: float = 0.5
    tag_hops: int = 1

    @classmethod
    def from_json_file(
        cls,
        path: str | Path = DEFAULT_SKILL_LIBRARY_PATH,
        min_confidence: float = 0.5,
    ) -> "SkillRetriever":
        skill_path = Path(path)
        if not skill_path.exists() and not skill_path.is_absolute():
            repo_relative_path = Path(__file__).resolve().parents[1] / skill_path
            if repo_relative_path.exists():
                skill_path = repo_relative_path
        data = json.loads(skill_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Skill library must be a JSON list: {skill_path}")
        return cls(skills=[_normalize_skill(record) for record in data if isinstance(record, dict)], min_confidence=min_confidence)

    def retrieve_skills(self, query: str, mode: str = "Exploration", k: int = 5) -> str:
        query_terms = set(query.lower().split())
        tag_index = _build_tag_index(self.skills)
        backlink_index = _build_backlink_index(self.skills)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for skill in self.skills:
            confidence = float(skill.get("confidence_score", 1.0))
            if confidence < self.min_confidence:
                continue
            text = _searchable_text(skill)
            overlap = len(query_terms.intersection(text.split()))
            mode_bonus = _mode_bonus(skill, mode)
            graph_bonus = _graph_bonus(skill, query_terms, tag_index, backlink_index, self.tag_hops)
            negative_penalty = _negative_memory_penalty(skill, mode)
            recency_bonus = float(skill.get("recency_score", 0.0)) * 0.15
            ranked.append((overlap + confidence + mode_bonus + graph_bonus + recency_bonus - negative_penalty, skill))
        ranked.sort(key=lambda item: item[0], reverse=True)
        snippets = [_format_skill_snippet(skill) for _, skill in ranked[:k]]
        return "\n".join(snippets)


def _normalize_skill(record: dict[str, Any]) -> dict[str, Any]:
    title = str(record.get("title") or record.get("motif_name") or "Unnamed skill")
    category = str(record.get("category") or "")
    boolean_template = str(record.get("boolean_template") or "")
    decomposition = str(record.get("decomposition_strategy") or "")
    risks = str(record.get("known_risks") or "")
    tradeoffs = str(record.get("trade_offs") or "")
    purpose = str(record.get("purpose") or record.get("summary") or "")
    tags = _normalize_tags(record.get("tags", [])) + [title, category, boolean_template]
    summary = (
        f"Motif: {title}\n"
        f"Category: {category}\n"
        f"Boolean template: {boolean_template}\n"
        f"Decomposition strategy: {decomposition}\n"
        f"Trade-offs: {tradeoffs}\n"
        f"Known risks: {risks}"
    )
    search_text = " ".join([title, category, purpose, boolean_template, decomposition, tradeoffs, risks])
    return {
        **record,
        "title": title,
        "summary": summary,
        "tags": [tag for tag in tags if tag],
        "search_text": search_text,
        "confidence_score": float(record.get("confidence_score", 1.0)),
        "backlinks": record.get("backlinks", []),
    }


def _format_skill_snippet(skill: dict[str, Any]) -> str:
    title = str(skill.get("title") or "")
    summary = str(skill.get("summary") or "")
    if title and summary and title not in summary:
        return f"Skill: {title}\n{summary}"
    return summary or title or str(skill)


def _searchable_text(skill: dict[str, Any]) -> str:
    return " ".join(
        str(skill.get(key, ""))
        for key in ("title", "summary", "tags", "search_text", "backlinks", "source_node")
    ).lower()


def _normalize_tags(tags: Any) -> list[str]:
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, list):
        return [str(tag) for tag in tags]
    return []


def _build_tag_index(skills: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for skill in skills:
        for tag in _normalize_tags(skill.get("tags", [])):
            index.setdefault(tag.lower(), []).append(skill)
    return index


def _build_backlink_index(skills: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for skill in skills:
        title = str(skill.get("title", "")).lower()
        source = str(skill.get("source_node", "")).lower()
        for key in (title, source):
            if key:
                index.setdefault(key, []).append(skill)
    return index


def _graph_bonus(
    skill: dict[str, Any],
    query_terms: set[str],
    tag_index: dict[str, list[dict[str, Any]]],
    backlink_index: dict[str, list[dict[str, Any]]],
    tag_hops: int,
) -> float:
    bonus = 0.0
    seen_titles = {str(skill.get("title", "")).lower()}
    frontier = [tag.lower() for tag in _normalize_tags(skill.get("tags", []))]
    for _ in range(max(0, tag_hops)):
        next_frontier: list[str] = []
        for tag in frontier:
            related = tag_index.get(tag, [])
            for neighbor in related:
                title = str(neighbor.get("title", "")).lower()
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                text = _searchable_text(neighbor)
                if query_terms.intersection(text.split()):
                    bonus += 0.15 * float(neighbor.get("confidence_score", 1.0))
                next_frontier.extend(t.lower() for t in _normalize_tags(neighbor.get("tags", [])))
        frontier = next_frontier

    for backlink in _normalize_tags(skill.get("backlinks", [])):
        key = backlink.strip("[]").lower()
        for neighbor in backlink_index.get(key, []):
            text = _searchable_text(neighbor)
            if query_terms.intersection(text.split()):
                bonus += 0.1 * float(neighbor.get("confidence_score", 1.0))
    return min(bonus, 1.0)


def _mode_bonus(skill: dict[str, Any], mode: str) -> float:
    text = _searchable_text(skill)
    normalized_mode = mode.lower()
    if normalized_mode in text:
        return 0.3
    if normalized_mode == "repair" and any(token in text for token in ("failure", "logic-error", "recovery")):
        return 0.25
    if normalized_mode == "exploitation" and any(token in text for token in ("part-error", "mapping", "ode", "physical")):
        return 0.25
    if normalized_mode == "exploration" and any(token in text for token in ("motif", "success", "template")):
        return 0.2
    return 0.0


def _negative_memory_penalty(skill: dict[str, Any], mode: str) -> float:
    text = _searchable_text(skill)
    if mode.lower() == "repair":
        return 0.0
    if "dead_end" in text or "dead-end" in text or "avoid" in text:
        return 0.45
    return 0.0
