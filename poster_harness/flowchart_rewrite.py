from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from .llm import ChatGPTAccountResponsesProvider
from .prompt import sanitize_public_text
from .schemas import JSON_SCHEMA_DRAFT


FLOWCHART_REWRITER_SYSTEM_PROMPT = (
    "You rewrite analysis flowcharts for a scientific poster. "
    "Every node must carry concrete paper-grounded content that an expert could not guess without reading the paper. "
    "Return JSON only. Never invent thresholds, variables, regions, or results."
)


FLOWCHART_REWRITE_RULES: list[str] = [
    "Rewrite only the supplied sections that already have a flowchart.",
    "Each node is one information capsule, not a generic stage label.",
    "Every node MUST contain at least one specific number, variable, threshold, paper-specific region/category label, fitted observable, uncertainty label, or statistical-method detail.",
    "Good: '>=2 same-sign muons: pT>30 GeV, |eta|<2.4'. Bad: 'preselection'.",
    "Good: 'VBF tag: mjj>750 GeV, |Delta eta_jj|>2.5'. Bad: 'VBF topology'.",
    "Good: 'HMN SR binned in Delta phi(ll) | Weinberg SR binned in pTmiss'. Bad: 'signal regions'.",
    "Use a '|' character inside a node label to denote a true branch or parallel sub-flow; do not use it as decoration.",
    "Order nodes in true data-flow order: dataset -> object/event selection -> categorization -> SR/CR definition -> fit/model -> output result.",
    "Produce 4-5 nodes; fewer if the analysis is short. Never pad with generic stages.",
    "Use the paper's own symbols and unit suffixes exactly when present (e.g. mjj, pTmiss, Delta phi(ll), GeV, fb^-1).",
    "Keep each label under 22 words; ideally 10-15.",
    "For every node, include evidence: a short verbatim or near-verbatim quote from the supplied paper text supporting the label.",
    "If you cannot find evidence for a node, omit that node.",
]


def flowchart_rewrite_schema() -> dict[str, Any]:
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "type": "object",
        "additionalProperties": True,
        "required": ["sections"],
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["section_id", "nodes"],
                    "properties": {
                        "section_id": {"type": "integer"},
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": True,
                                "required": ["label", "evidence"],
                                "properties": {
                                    "label": {"type": "string"},
                                    "evidence": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def rewrite_flowcharts_from_paper(
    spec: Mapping[str, Any],
    paper_text: str,
    *,
    provider: ChatGPTAccountResponsesProvider | None = None,
    text_char_limit: int = 24000,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """Rewrite existing spec flowcharts into concrete, evidenced HEP schematics.

    The stage is intentionally strict and non-fallback: if candidate sections exist
    but the LLM call fails, the exception bubbles to the caller.  If the spec has no
    flowchart fields, the stage is skipped without making an LLM call.
    """

    provider = provider or ChatGPTAccountResponsesProvider()
    candidates = _candidate_sections(spec)
    if not candidates:
        return {"result": {"sections": []}, "_skipped": True}

    truncated_text = str(paper_text or "")
    if len(truncated_text) > text_char_limit:
        truncated_text = truncated_text[:text_char_limit] + "\n... [truncated]"
    prompt = _compose_prompt(
        header="Rewrite each analysis flowchart so every node carries concrete, paper-grounded content.",
        instructions=[*FLOWCHART_REWRITE_RULES, extra_instructions or ""],
        context={
            "sections_to_rewrite": candidates,
            "paper_text": truncated_text,
        },
    )
    return provider.generate_json(
        stage_name="flowchart_rewrite",
        prompt=prompt,
        schema=flowchart_rewrite_schema(),
        system_prompt=FLOWCHART_REWRITER_SYSTEM_PROMPT,
    )


def apply_flowchart_rewrites(
    spec: Mapping[str, Any],
    rewrite_envelope: Mapping[str, Any],
    *,
    require_evidence: bool = True,
) -> dict[str, Any]:
    """Return a copy of *spec* with evidenced flowchart node rewrites applied."""

    payload = rewrite_envelope.get("result") if isinstance(rewrite_envelope.get("result"), Mapping) else rewrite_envelope
    sections_patch = list((payload or {}).get("sections") or [])
    new_spec = copy.deepcopy(dict(spec))
    by_id: dict[int, dict[str, Any]] = {}
    for sec in new_spec.get("sections") or []:
        if isinstance(sec, dict) and sec.get("id") is not None:
            try:
                by_id[int(sec["id"])] = sec
            except (TypeError, ValueError):
                continue

    rewritten_records: list[dict[str, Any]] = []
    for entry in sections_patch:
        if not isinstance(entry, Mapping):
            continue
        try:
            sid = int(entry.get("section_id"))
        except (TypeError, ValueError):
            continue
        section = by_id.get(sid)
        if section is None:
            continue
        labels: list[str] = []
        evidences: list[str] = []
        for node in entry.get("nodes") or []:
            if not isinstance(node, Mapping):
                continue
            label = _clean_node_label(node.get("label"))
            evidence = sanitize_public_text(str(node.get("evidence") or "")).strip()
            if not label:
                continue
            if require_evidence and not evidence:
                continue
            if len(labels) >= 5:
                continue
            labels.append(label)
            evidences.append(evidence)
        if labels:
            section["flowchart"] = labels
            rewritten_records.append(
                {
                    "section_id": sid,
                    "nodes": labels,
                    "evidence": evidences,
                }
            )
        else:
            # The rewriter was asked about this section but found no evidenced
            # concrete nodes (or all nodes were dropped).  Clear the old generic
            # pipeline rather than letting a "selection -> fit" schematic leak
            # back into the image prompt.
            section["flowchart"] = []
            rewritten_records.append({"section_id": sid, "nodes": [], "evidence": [], "cleared": True})

    new_spec.setdefault("_flowchart_rewrite", {}).update(
        {
            "enabled": True,
            "rewritten_sections": rewritten_records,
            "skipped": bool(rewrite_envelope.get("_skipped")),
        }
    )
    return new_spec


def _candidate_sections(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for sec in spec.get("sections") or []:
        if not isinstance(sec, Mapping):
            continue
        flow = [str(item).strip() for item in sec.get("flowchart") or [] if str(item).strip()]
        if not flow:
            continue
        try:
            sid = int(sec.get("id"))
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "section_id": sid,
                "section_title": str(sec.get("title") or ""),
                "current_nodes": flow,
            }
        )
    return candidates


def _clean_node_label(value: Any) -> str:
    text = sanitize_public_text(str(value or "")).strip()
    text = " ".join(text.split())
    # These phrases tend to be prompt/system residue rather than poster copy.
    banned = ("placeholder", "replacement", "workflow note", "todo", "draw ", "place ")
    lowered = text.lower()
    if any(item in lowered for item in banned):
        return ""
    return text[:180]


def _compose_prompt(*, header: str, instructions: Sequence[str], context: Mapping[str, Any]) -> str:
    lines = [header, "", "Instructions:"]
    for item in instructions:
        item = str(item).strip()
        if item:
            lines.append(f"- {item}")
    lines += ["", "Context JSON:", json.dumps(context, ensure_ascii=False, indent=2)]
    return "\n".join(lines)
