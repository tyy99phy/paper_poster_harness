"""Observer: assemble a standardized, auditable record of one PosterHarness run.

This module is a *read-only observer* over a completed run directory. It reads the
artifacts a run already persists (``run_manifest.yaml`` plus the ``specs/``, ``qa/``
and ``scratch/`` files it references) and emits two views of the same content:

* ``run_record.json`` -- the machine-readable corpus unit (the source of truth).
* ``run_record.md``   -- a human-readable rendering of that same record.

It never re-invokes the pipeline and never mutates run inputs, so it is safe to run
on any finished run (including failed ones) and cheap to run repeatedly. The
Markdown is rendered *from* the JSON, so the two never drift.

Design intent: the record separates OBJECTIVE checks -- provenance, aspect contract,
containment, public-text cleanliness, recovery accounting -- which are computed from
artifacts or hold by construction, from SUBJECTIVE model-as-judge scores (template
critic / final QA). The subjective scores are fenced off and must not be read as
ground truth: the judge shares a model family with the generator, so self-preference
bias cannot be excluded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

# Reuse the pipeline's own geometry math so the observer's aspect check matches the
# contract the pipeline actually enforces, rather than re-deriving (and risking drift).
from poster_harness.replace import (
    _box_ratio,
    _parse_placeholder_aspect,
    _ratio_relative_error,
)

SCHEMA_VERSION = "1.2"

# replace.py uses 0.20 as its base placeholder ratio tolerance; match it.
ASPECT_TOLERANCE = 0.20

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_TEXT_SUFFIXES = {".txt", ".md", ".yaml", ".yml", ".json", ".tex", ".log"}

# The capability lens: each pipeline stage exercises one model competency, in order.
# (stage_key, human label, capability tag)
STAGE_LENS: tuple[tuple[str, str, str], ...] = (
    ("extract", "Read paper text & figures", "reading"),
    ("domain_profile", "Infer scientific domain", "reading"),
    ("copy_deck", "Plan poster text", "text_planning"),
    ("storyboard", "Plan poster layout", "layout_planning"),
    ("figure_selection", "Select & ground figures", "figure_grounding"),
    ("generate", "Generate placeholder template", "image_generation"),
    ("template_critic", "Review template", "image_review"),
    ("detect", "Detect placeholder geometry", "geometry"),
    ("placeholder_qa", "QA placeholder layout", "image_review"),
    ("replace", "Insert real figures", "figure_replacement"),
    ("final_qa", "Final QA", "image_review"),
    ("physics_quiz", "Domain comprehension probe", "domain_knowledge"),
)
_LENS = {key: (label, cap) for key, label, cap in STAGE_LENS}


# --------------------------------------------------------------------------- IO

def _load_yaml(path: Path | None) -> Any:
    """Best-effort YAML load; returns ``None`` for missing/unreadable files."""
    if path is None:
        return None
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return None


def _reroot(raw: Any, out: str, run_dir: Path) -> Path | None:
    """Map a manifest path (stored as ``<out>/sub/x`` at run time) under ``run_dir``.

    Manifest paths are relative to the repo root at the moment the run executed; the
    run may later be inspected from a different working directory or after being
    moved, so we re-root every per-run artifact onto the absolute ``run_dir`` we were
    handed. Falls back to an existing absolute path or a same-name file in the run.
    """
    if not raw:
        return None
    p = Path(str(raw))
    if out:
        try:
            return run_dir / p.relative_to(out)
        except ValueError:
            pass
    if p.is_absolute() and p.exists():
        return p
    by_name = run_dir / p.name
    return by_name if by_name.exists() else (run_dir / p)


def _stem(path_like: Any) -> str:
    return Path(str(path_like)).stem


def _rel(path: Path | None, run_dir: Path) -> str | None:
    """Return a stable, human-readable path for records."""
    if path is None:
        return None
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _short(text: Any, limit: int = 140) -> str:
    """Single-line, bounded summary for tables."""
    value = " ".join(str(text or "").split())
    return value[: limit - 1] + "…" if len(value) > limit else value


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _read_text(path: Path) -> str | None:
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _artifact_record(path: Path, run_dir: Path, *, role: str, include_content: bool = True) -> dict[str, Any]:
    exists = path.exists()
    content = _read_text(path) if exists and include_content else None
    return {
        "role": role,
        "path": _rel(path, run_dir),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": _sha256(path) if exists else None,
        "language": path.suffix.lower().lstrip(".") or "text",
        "content": content,
    }


def _add_artifact(
    out: list[dict[str, Any]],
    seen: set[str],
    path: Path | None,
    run_dir: Path,
    *,
    role: str,
    include_content: bool = True,
) -> None:
    if path is None:
        return
    key = str(path.resolve()) if path.exists() else str(path)
    if key in seen:
        return
    seen.add(key)
    out.append(_artifact_record(path, run_dir, role=role, include_content=include_content))


def _collect_workflow_artifacts(
    manifest: Mapping[str, Any],
    out: str,
    run_dir: Path,
    *,
    stem: str | None,
    text_path: Path | None,
) -> list[dict[str, Any]]:
    """Full text artifacts for the detailed workflow trace.

    This intentionally goes beyond the compact scoreboard: it captures the exact
    persisted prompt(s), planning artifacts, detections, QA judgments, and manifest
    content needed to audit what the harness did without re-running the pipeline.
    Binary images/PDFs are indexed elsewhere by path/hash and are not embedded.
    """
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    _add_artifact(artifacts, seen, run_dir / "run_manifest.yaml", run_dir, role="run_manifest")
    _add_artifact(artifacts, seen, text_path, run_dir, role="source_text")

    for key in (
        "assets_manifest",
        "domain_profile",
        "content_outline",
        "draft_spec",
        "storyboard",
        "physics_quiz",
        "figure_selection",
        "copy_deck",
        "flowchart_rewrite",
        "layout_contract",
        "poster_spec",
        "postprocess_manifest",
    ):
        _add_artifact(artifacts, seen, _reroot(manifest.get(key), out, run_dir), run_dir, role=f"planning:{key}")

    for path in sorted((run_dir / "prompts").glob("*")):
        if path.is_file():
            _add_artifact(artifacts, seen, path, run_dir, role="prompt")
    for path in sorted((run_dir / "traces").glob("*")):
        if path.is_file():
            _add_artifact(artifacts, seen, path, run_dir, role="llm_envelope_trace")
    for path in sorted((run_dir / "scratch").glob("*.yaml")):
        _add_artifact(artifacts, seen, path, run_dir, role="detection_or_scratch")
    for path in sorted((run_dir / "qa").glob("*.yaml")):
        _add_artifact(artifacts, seen, path, run_dir, role="qa_or_judgment")
    for path in sorted((run_dir / "specs").glob("poster_spec.*.with_placements*.yaml")):
        _add_artifact(artifacts, seen, path, run_dir, role="placed_spec")
    for path in sorted((run_dir / "specs").glob("postprocess*.yaml")):
        _add_artifact(artifacts, seen, path, run_dir, role="postprocess_manifest")

    # If a future manifest references exact LLM envelopes, include them even if they
    # are outside the conventional traces/ directory.
    for raw in manifest.get("llm_traces") or []:
        _add_artifact(artifacts, seen, _reroot(raw, out, run_dir), run_dir, role="llm_envelope_trace")

    if stem:
        _add_artifact(
            artifacts,
            seen,
            run_dir / "scratch" / f"{stem}.detections.yaml",
            run_dir,
            role="accepted_detection",
        )
    return artifacts


# --------------------------------------------------------------- objective checks

def _aspect_fields(declared: str, bbox: Any) -> dict[str, Any]:
    """Compare a declared aspect ("2.5:1 wide", "1:1 square") to a detected bbox."""
    want = _parse_placeholder_aspect(declared)
    if not bbox:
        return {"detected_ratio": None, "aspect_match": None}
    got = _box_ratio(tuple(bbox))
    if want is None:
        return {"detected_ratio": round(got, 3), "aspect_match": None}
    return {
        "detected_ratio": round(got, 3),
        "aspect_match": _ratio_relative_error(got, want) <= ASPECT_TOLERANCE,
    }


def _provenance_ledger(
    figure_selection: Any, detections: Any, out: str, run_dir: Path
) -> list[dict[str, Any]]:
    """One row per selected figure: is it a real extracted asset, and does its
    detected placeholder match the declared aspect contract?"""
    figs = (figure_selection or {}).get("selected_figures") or []
    placements = (detections or {}).get("placements") or {}
    ledger: list[dict[str, Any]] = []
    for fig in figs:
        pid = str(fig.get("placeholder_id") or fig.get("id") or "")
        declared = str(fig.get("aspect") or "")
        src = _reroot(fig.get("source_path") or fig.get("asset"), out, run_dir)
        if (src is None or not src.exists()) and fig.get("asset"):
            copied = run_dir / "assets" / Path(str(fig.get("asset"))).name
            if copied.exists():
                src = copied
        bbox = placements.get(pid)
        ledger.append(
            {
                "placeholder_id": pid,
                "label": str(fig.get("label") or ""),
                "role": str(fig.get("role") or ""),
                "priority": fig.get("priority"),
                "source_asset": (src.name if src else None),
                "source_exists": bool(src and src.exists()),
                "declared_aspect": declared,
                "detected_bbox": list(bbox) if bbox else None,
                **_aspect_fields(declared, bbox),
            }
        )
    return ledger


def _qa_kind(path: Path) -> str:
    name = path.name
    for suffix, kind in (
        (".template-critic.qa.yaml", "template_critic"),
        (".layout-contract.qa.yaml", "layout_contract"),
        (".placeholder.qa.yaml", "placeholder_qa"),
        (".final.qa.yaml", "final_qa"),
    ):
        if name.endswith(suffix):
            return kind
    return "qa"


def _collect_qa_reports(manifest: Mapping[str, Any], out: str, run_dir: Path) -> list[dict[str, Any]]:
    """Summarize every QA/critic artifact referenced by the run manifest."""
    reports: list[dict[str, Any]] = []
    for raw in manifest.get("qa") or []:
        path = _reroot(raw, out, run_dir)
        data = _load_yaml(path)
        if not isinstance(data, dict):
            reports.append(
                {
                    "path": _rel(path, run_dir),
                    "kind": _qa_kind(path or Path(str(raw))),
                    "exists": bool(path and path.exists()),
                    "passes": None,
                    "score": None,
                    "issue_count": None,
                    "summary": "unreadable or missing QA artifact",
                }
            )
            continue
        issues = data.get("issues") or []
        reports.append(
            {
                "path": _rel(path, run_dir),
                "kind": _qa_kind(path or Path(str(raw))),
                "exists": True,
                "passes": data.get("passes"),
                "score": data.get("score") if data.get("score") is not None else (data.get("scores") or {}).get("overall"),
                "issue_count": len(issues) if isinstance(issues, list) else None,
                "summary": _short(data.get("summary"), 220),
            }
        )
    return reports


# ------------------------------------------------------------------- build record

def build_record(run_dir: str | Path) -> dict[str, Any]:
    """Assemble the structured run record from a completed run directory."""
    run_dir = Path(run_dir).resolve()
    manifest = _load_yaml(run_dir / "run_manifest.yaml") or {}
    out = str(manifest.get("out") or "")

    poster_sets = manifest.get("poster_sets") or []
    accepted = bool(poster_sets)
    stem = _stem(poster_sets[0].get("template")) if poster_sets else None

    figure_selection = _load_yaml(_reroot(manifest.get("figure_selection"), out, run_dir))
    domain_profile = _load_yaml(_reroot(manifest.get("domain_profile"), out, run_dir))
    detections = _load_yaml(run_dir / "scratch" / f"{stem}.detections.yaml") if stem else None
    final_qa = _load_yaml(run_dir / "qa" / f"{stem}.final.qa.yaml") if stem else None
    placeholder_qa = _load_yaml(run_dir / "qa" / f"{stem}.placeholder.qa.yaml") if stem else None
    accepted_critic = _load_yaml(run_dir / "qa" / f"{stem}.template-critic.qa.yaml") if stem else None

    ledger = _provenance_ledger(figure_selection, detections, out, run_dir)
    qa_reports = _collect_qa_reports(manifest, out, run_dir)

    # Source text: prefer the conventional in-run location (runs that reuse another
    # run's extracted text point text_source elsewhere).
    text_path = run_dir / "input" / "extracted_text.txt"
    if not text_path.exists():
        text_path = _reroot(manifest.get("text_source"), out, run_dir)
    text_chars = 0
    if text_path and text_path.exists():
        try:
            text_chars = len(text_path.read_text(encoding="utf-8"))
        except OSError:
            text_chars = 0
    assets_dir = run_dir / "assets"
    n_assets = (
        len(
            [
                p
                for p in assets_dir.glob("*")
                if p.suffix.lower() in _IMAGE_SUFFIXES and "contact" not in p.name.lower()
            ]
        )
        if assets_dir.exists()
        else 0
    )

    # Recovery accounting: how many template candidates, how many critic rejections.
    candidates = manifest.get("generated_all") or manifest.get("generated_candidates") or []
    critic_rejections = 0
    for crit in manifest.get("template_critiques") or []:
        cq = _load_yaml(_reroot(crit, out, run_dir))
        if isinstance(cq, dict) and cq.get("passes") is False:
            critic_rejections += 1

    detections_list = (detections or {}).get("placeholders") or []
    exports = manifest.get("exports") or []
    checks = (final_qa or {}).get("checks") or {}
    domain = manifest.get("domain_profile_name") or (domain_profile or {}).get("name")
    qa_failure_count = sum(1 for report in qa_reports if report.get("passes") is False)

    # ---- capability stage spine ---------------------------------------------
    def row(key: str, status: str, objective: bool, detail: str) -> dict[str, Any]:
        label, cap = _LENS.get(key, (key, "other"))
        return {
            "id": key,
            "label": label,
            "capability": cap,
            "status": status,
            "objective": objective,
            "detail": detail,
        }

    n_fig = len(ledger)
    n_prov = sum(1 for e in ledger if e["source_exists"])
    confs = [p.get("confidence") for p in detections_list if isinstance(p.get("confidence"), (int, float))]
    nsec = checks.get("section_count")

    stages = [
        row("extract", "done" if text_chars else "missing", True,
            f"{text_chars} chars of source text; {n_assets} source figure(s) extracted"),
        row("domain_profile", "done" if domain else "missing", True,
            f"domain inferred: {domain}" if domain else "no domain profile"),
        row("copy_deck", "done" if manifest.get("copy_deck") else "n/a", True,
            "poster text planned"),
        row("storyboard", "done" if manifest.get("storyboard") else "n/a", True,
            f"{nsec} section(s) planned" if nsec else "layout planned"),
        row("figure_selection", "done" if ledger else "n/a", True,
            f"{n_fig} figure(s) selected; {n_prov}/{n_fig} resolve to real extracted assets"),
        row("generate", "done" if candidates else "missing", True,
            f"{len(candidates)} template candidate(s) generated"),
        row("template_critic",
            ("pass" if accepted_critic.get("passes") else "fail") if accepted_critic else "n/a",
            False,
            (f"accepted template critic passes={accepted_critic.get('passes')}; "
             f"{critic_rejections} earlier rejection(s)") if accepted_critic else "no critic report"),
        row("detect", "done" if detections_list else "n/a", True,
            f"{len(detections_list)} placeholder(s) detected"
            + (f"; min confidence {min(confs):.2f}" if confs else "")),
        row("placeholder_qa",
            ("pass" if placeholder_qa.get("passes") else "fail") if placeholder_qa else "n/a",
            False,
            f"placeholder QA passes={placeholder_qa.get('passes')}" if placeholder_qa else "no report"),
        row("replace", "done" if exports else "missing", True,
            f"{len(exports)} export(s) produced; containment hard-enforced (margin=0)"),
        row("final_qa",
            ("pass" if final_qa.get("passes") else "fail") if final_qa else "n/a",
            False,
            f"final QA passes={final_qa.get('passes')} score={final_qa.get('score')}"
            if final_qa else "no final QA report"),
        row("physics_quiz", "done" if manifest.get("physics_quiz") else "n/a", True,
            "domain comprehension quiz generated"),
    ]

    # ---- objective scoreboard (the corpus-grade signal) ---------------------
    aspect_known = [e for e in ledger if e["aspect_match"] is not None]
    n_aspect_ok = sum(1 for e in aspect_known if e["aspect_match"])
    scoreboard = {
        "provenance": f"{n_prov}/{n_fig}" if n_fig else "n/a",
        "provenance_all_real": (n_fig > 0 and n_prov == n_fig),
        "aspect_contract": f"{n_aspect_ok}/{len(aspect_known)}" if aspect_known else "n/a",
        "containment": (
            f"by construction (margin=0 hard error on violation); {len(exports)} export(s) produced"
            if exports else "no exports produced"
        ),
        "public_text_clean": checks.get("public_text_clean"),
        "placeholders_accounted_for": checks.get("placeholders_accounted_for"),
        "domain_inferred": domain,
        "recovery": (
            f"accepted {len(poster_sets)} of {max(len(candidates), len(poster_sets))} "
            f"candidate template(s); {critic_rejections} critic rejection(s), "
            f"{qa_failure_count} failed QA/contract report(s)"
        ),
    }

    # ---- subjective scores (fenced: model-as-judge, not ground truth) --------
    subjective = {
        "caveat": (
            "Model-as-judge scores. The judge shares a model family with the generator, "
            "so self-preference bias cannot be excluded; treat as indicative, not ground truth."
        ),
        "template_critic_scores": (accepted_critic or {}).get("scores"),
        "final_qa_score": (final_qa or {}).get("score"),
    }
    artifacts = {
        "manifest": "run_manifest.yaml",
        "source_text": _rel(text_path if text_path and text_path.exists() else None, run_dir),
        "prompt": _rel(_reroot(manifest.get("prompt"), out, run_dir), run_dir),
        "poster_spec": _rel(_reroot(manifest.get("poster_spec"), out, run_dir), run_dir),
        "postprocess_manifest": _rel(_reroot(manifest.get("postprocess_manifest"), out, run_dir), run_dir),
        "figure_selection": _rel(_reroot(manifest.get("figure_selection"), out, run_dir), run_dir),
        "detections": _rel(run_dir / "scratch" / f"{stem}.detections.yaml", run_dir) if stem else None,
        "exports": [_rel(_reroot(item, out, run_dir), run_dir) for item in exports],
    }
    workflow_artifacts = _collect_workflow_artifacts(
        manifest,
        out,
        run_dir,
        stem=stem,
        text_path=text_path if text_path and text_path.exists() else None,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "paper": {
            "source_pdf": manifest.get("paper"),
            "arxiv_id": _stem(manifest.get("paper")) if manifest.get("paper") else None,
            "text_chars": text_chars,
            "source_figures": n_assets,
        },
        "config": {
            "content_mode": manifest.get("content_mode"),
            "domain": domain,
            "required_successes": manifest.get("required_successes"),
            "max_candidate_batches": manifest.get("max_candidate_batches"),
            "template_critic": manifest.get("template_critic"),
        },
        "outcome": {
            "accepted": accepted,
            "n_candidates": len(candidates),
            "n_accepted_sets": len(poster_sets),
            "critic_rejections": critic_rejections,
            "qa_failures": qa_failure_count,
        },
        "capability_stages": stages,
        "provenance_ledger": ledger,
        "objective_scoreboard": scoreboard,
        "subjective_scores": subjective,
        "qa_reports": qa_reports,
        "artifact_index": artifacts,
        "workflow_artifacts": workflow_artifacts,
        "trace_note": (
            "This record embeds every text artifact persisted by the run. Historical runs "
            "may not contain exact LLM-stage prompts unless traces/*.envelope.yaml files "
            "were saved at generation time; the image-generation prompt is persisted under prompts/."
        ),
    }


# ----------------------------------------------------------------- markdown view

def _yn(value: Any) -> str:
    return "✅" if value is True else ("❌" if value is False else "—")


def _fmt(value: Any) -> str:
    if value is True:
        return "✅ yes"
    if value is False:
        return "❌ no"
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _md_cell(value: Any) -> str:
    """Escape a value for a Markdown table cell."""
    return str(value if value is not None else "—").replace("\n", " ").replace("|", r"\|")


def _artifact_heading(item: Mapping[str, Any]) -> str:
    status = "present" if item.get("exists") else "missing"
    size = _fmt(item.get("bytes"))
    digest = str(item.get("sha256") or "")
    digest = f", sha256={digest[:12]}…" if digest else ""
    return f"`{item.get('path')}` ({status}, {size} bytes{digest})"


def _append_artifact_block(lines: list[str], item: Mapping[str, Any]) -> None:
    lines.append(f"#### {item.get('role')} — {_artifact_heading(item)}")
    lines.append("")
    content = item.get("content")
    if content is None:
        lines.append("_Binary, missing, or intentionally not embedded; see path/hash above._")
        lines.append("")
        return
    lang = str(item.get("language") or "text")
    if lang == "yml":
        lang = "yaml"
    lines.append(f"````{lang}")
    lines.append(str(content).rstrip())
    lines.append("````")
    lines.append("")


_STATUS_BADGE = {
    "pass": "✅ pass",
    "fail": "❌ fail",
    "done": "✅ done",
    "missing": "❌ missing",
    "n/a": "— n/a",
}


def render_markdown(record: dict[str, Any]) -> str:
    """Render the structured record as a human-readable Markdown report."""
    paper = record["paper"]
    cfg = record["config"]
    out = record["outcome"]
    lines: list[str] = []

    lines += [
        f"# PosterHarness run record — `{record['run_id']}`",
        "",
        f"*Schema v{record['schema_version']}. Assembled by a read-only observer from "
        "run artifacts. Objective checks and model-judge scores are kept separate — "
        "see the final two sections.*",
        "",
        "## Run card",
        "",
        f"- **Paper:** `{paper.get('source_pdf')}` "
        f"({paper.get('text_chars')} chars text, {paper.get('source_figures')} source figures)",
        f"- **Mode / domain:** {cfg.get('content_mode')} / {cfg.get('domain')}",
        "- **Outcome:** "
        + ("✅ accepted" if out["accepted"] else "❌ no accepted poster")
        + f" — {out['n_accepted_sets']} set(s) from {out['n_candidates']} candidate(s), "
        f"{out['critic_rejections']} critic rejection(s), "
        f"{out.get('qa_failures', 0)} failed QA/contract report(s)",
        "",
        "## Capability stages",
        "",
        "| # | Stage | Capability | Status | Check | Detail |",
        "|---|-------|------------|--------|-------|--------|",
    ]
    for i, stage in enumerate(record["capability_stages"], 1):
        kind = "objective" if stage["objective"] else "model-judge"
        badge = _STATUS_BADGE.get(stage["status"], stage["status"])
        lines.append(
            f"| {i} | {_md_cell(stage['label'])} | `{_md_cell(stage['capability'])}` "
            f"| {badge} | {kind} | {_md_cell(stage['detail'])} |"
        )

    if record.get("qa_reports"):
        lines += [
            "",
            "## QA / failure accounting",
            "",
            "*Every persisted QA or critic report referenced by the run manifest. "
            "This is the process trace used to reconstruct retries and failure modes.*",
            "",
            "| Report | Type | Verdict | Score | Issues | Summary |",
            "|--------|------|---------|-------|--------|---------|",
        ]
        for report in record["qa_reports"]:
            verdict = _yn(report.get("passes"))
            score = _fmt(report.get("score"))
            lines.append(
                f"| `{_md_cell(report.get('path'))}` | `{_md_cell(report.get('kind'))}` "
                f"| {verdict} | {_md_cell(score)} | {_fmt(report.get('issue_count'))} "
                f"| {_md_cell(report.get('summary'))} |"
            )

    lines += [
        "",
        "## Provenance ledger",
        "",
        "*Every selected figure traced to a real extracted asset, with its detected "
        "placeholder checked against the declared aspect contract.*",
        "",
        "| Placeholder | Figure | Real asset? | Declared | Detected | Aspect OK? |",
        "|-------------|--------|-------------|----------|----------|------------|",
    ]
    for e in record["provenance_ledger"]:
        label = e["label"][:44] + ("…" if len(e["label"]) > 44 else "")
        detected = e.get("detected_ratio")
        detected = f"{detected:.2f}" if isinstance(detected, (int, float)) else "—"
        lines.append(
            f"| {_md_cell(e['placeholder_id'])} | {_md_cell(label)} | {_yn(e['source_exists'])} "
            f"| {_md_cell(e['declared_aspect'] or '—')} | {detected} | {_yn(e['aspect_match'])} |"
        )

    sb = record["objective_scoreboard"]
    lines += [
        "",
        "## Objective scoreboard",
        "",
        "*Computed from artifacts or true by construction — the corpus-grade signal "
        "that aggregates across runs.*",
        "",
    ]
    lines += [f"- **{key}:** {_fmt(val)}" for key, val in sb.items()]

    artifacts = record.get("artifact_index") or {}
    if artifacts:
        lines += [
            "",
            "## Artifact index",
            "",
            "*Key files that make the record auditable without re-running the pipeline.*",
            "",
        ]
        for key, val in artifacts.items():
            if isinstance(val, list):
                rendered = ", ".join(f"`{item}`" for item in val if item) or "—"
            else:
                rendered = f"`{val}`" if val else "—"
            lines.append(f"- **{key}:** {rendered}")

    workflow_artifacts = record.get("workflow_artifacts") or []
    if workflow_artifacts:
        lines += [
            "",
            "## Complete workflow trace",
            "",
            f"> {record.get('trace_note') or ''}",
            "",
            "This section embeds the persisted text artifacts themselves: exact image-generation "
            "prompt(s), planning YAML, detection YAML, QA/critic judgments, and placement specs. "
            "Binary images/PDFs are referenced by path/hash rather than embedded.",
            "",
        ]
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for item in workflow_artifacts:
            role = str(item.get("role") or "other")
            group = role.split(":", 1)[0]
            grouped.setdefault(group, []).append(item)
        labels = [
            ("run_manifest", "Run manifest"),
            ("source_text", "Source text"),
            ("planning", "Planning artifacts and structured outputs"),
            ("prompt", "Exact prompt files"),
            ("llm_envelope_trace", "Exact LLM-stage envelopes / prompts"),
            ("detection_or_scratch", "Detections and scratch geometry"),
            ("placed_spec", "Specs after detected placements"),
            ("accepted_detection", "Accepted detection file"),
            ("qa_or_judgment", "QA, critic, and failure judgments"),
        ]
        emitted: set[str] = set()
        for group, title in labels:
            items = grouped.get(group) or []
            if not items:
                if group == "llm_envelope_trace":
                    lines += [
                        f"### {title}",
                        "",
                        "_No `traces/*.envelope.yaml` files are present in this historical run. "
                        "Future runs can persist exact LLM-stage envelopes; this run still includes "
                        "the exact image-generation prompt under `prompts/` and all output/judgment artifacts._",
                        "",
                    ]
                continue
            emitted.add(group)
            lines += [f"### {title}", ""]
            for item in items:
                _append_artifact_block(lines, item)
        for group, items in grouped.items():
            if group in emitted:
                continue
            lines += [f"### Other artifacts: {group}", ""]
            for item in items:
                _append_artifact_block(lines, item)

    su = record["subjective_scores"]
    lines += [
        "",
        "## Model-judge scores (not ground truth)",
        "",
        f"> {su['caveat']}",
        "",
        f"- Final QA score: {_fmt(su.get('final_qa_score'))}",
    ]
    for key, val in (su.get("template_critic_scores") or {}).items():
        lines.append(f"- Template critic · {key}: {_fmt(val)}")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------- write

def write_record(run_dir: str | Path) -> tuple[Path, Path]:
    """Build the record and write ``run_record.json`` + ``run_record.md`` into the run.

    Returns the (json_path, md_path) pair. The JSON is the source of truth; the
    Markdown is rendered from it.
    """
    run_dir = Path(run_dir).resolve()
    record = build_record(run_dir)
    json_path = run_dir / "run_record.json"
    md_path = run_dir / "run_record.md"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(record), encoding="utf-8")
    return json_path, md_path
