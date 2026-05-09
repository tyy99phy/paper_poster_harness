from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from .extract import make_contact_sheet
from .layout_contract import (
    attach_layout_contract_boxes,
    evaluate_layout_contract_alignment,
)
from .schemas import normalize_placeholder_id


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
VECTOR_FIGURE_EXTENSIONS = {".pdf"}
ASSET_EXTENSIONS = IMAGE_EXTENSIONS | VECTOR_FIGURE_EXTENSIONS
IGNORED_NAME_PARTS = {
    "contact_sheet",
    "thumbnail",
    "thumb",
    "preview",
    "backup",
    ".before-",
    "logo",
    "orcid",
    "funding",
    "authorlist",
    "cms_paper",
}


def infer_asset_roots(
    paper_path: str | Path,
    *,
    explicit_roots: Sequence[str | Path] | None = None,
    extracted_roots: Sequence[str | Path] | None = None,
) -> list[Path]:
    """Infer useful figure/image roots for a paper project.

    The function prefers explicit roots and obvious paper-source figure
    directories. It does not recursively scan the entire home tree.
    """
    paper = Path(paper_path)
    roots: list[Path] = []
    roots.extend(Path(p) for p in explicit_roots or [])
    roots.extend(Path(p) for p in extracted_roots or [])

    parent = paper.parent if paper.parent != Path("") else Path.cwd()
    common_names = [
        "assets",
        "asset",
        "figures",
        "figure",
        "figs",
        "fig",
        "figures_png",
        "images",
        "image",
        "media",
        "plots",
        "plot",
        "graphics",
    ]
    for name in common_names:
        candidate = parent / name
        if candidate.exists():
            roots.append(candidate)

    # Last resort: if the paper sits in a compact source directory that already
    # contains images, scanning that directory is often useful for arXiv sources.
    if any(child.suffix.lower() in IMAGE_EXTENSIONS for child in parent.iterdir() if child.is_file()):
        roots.append(parent)

    return _dedupe_existing_dirs(roots)


def build_assets_manifest(
    source_roots: Sequence[str | Path],
    *,
    out_dir: str | Path | None = None,
    copy_assets: bool = True,
    recursive: bool = True,
    max_assets: int | None = None,
    min_width: int = 48,
    min_height: int = 48,
    contact_sheet: bool = True,
) -> dict[str, Any]:
    """Scan figure roots, optionally copy/render them into ``out_dir``, and manifest them."""
    dest_dir = Path(out_dir) if out_dir else None
    if copy_assets and dest_dir:
        dest_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()
    seen_sources: set[Path] = set()
    for src in iter_image_files(source_roots, recursive=recursive):
        try:
            real = src.resolve()
        except Exception:
            real = src.absolute()
        if real in seen_sources or _ignored_asset_name(src.name):
            continue
        seen_sources.add(real)
        materialized = _materialize_asset(src, dest_dir=dest_dir if copy_assets else None, used_names=used_names)
        if materialized is None:
            continue
        asset_name, asset_path, width, height = materialized
        if width < min_width or height < min_height:
            continue

        rows.append(
            {
                "name": asset_name,
                "asset": asset_name,
                "path": str(asset_path),
                "source_path": str(src),
                "kind": "figure",
                "label": infer_asset_label(src.name),
                "width": width,
                "height": height,
                "aspect": aspect_label(width, height),
                "notes": [],
            }
        )
        if max_assets is not None and len(rows) >= max_assets:
            break

    if contact_sheet and dest_dir and rows:
        make_contact_sheet([Path(row["path"]) for row in rows], dest_dir / "contact_sheet.jpg")
    return {"assets": rows}


def iter_image_files(source_roots: Sequence[str | Path], *, recursive: bool = True) -> Iterable[Path]:
    for root_like in source_roots:
        root = Path(root_like)
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in ASSET_EXTENSIONS:
                yield root
            continue
        pattern = "**/*" if recursive else "*"
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.suffix.lower() in ASSET_EXTENSIONS:
                yield path


def _materialize_asset(src: Path, *, dest_dir: Path | None, used_names: set[str]) -> tuple[str, Path, int, int] | None:
    suffix = src.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        try:
            with Image.open(src) as im:
                width, height = im.size
        except Exception:
            return None
        if dest_dir:
            asset_name = _unique_asset_name(src.name, used_names)
            asset_path = dest_dir / asset_name
            try:
                same = asset_path.resolve() == src.resolve()
            except Exception:
                same = False
            if not same:
                shutil.copy2(src, asset_path)
        else:
            asset_name = _unique_asset_name(src.name, used_names)
            asset_path = src
        return asset_name, asset_path, width, height

    if suffix == ".pdf":
        if not dest_dir:
            return None
        try:
            import fitz
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyMuPDF/fitz is required to render PDF figure assets from arXiv sources") from exc
        doc = fitz.open(src)
        if len(doc) < 1:
            return None
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(240 / 72.0, 240 / 72.0), alpha=False)
        asset_name = _unique_asset_name(f"{src.stem}.png", used_names)
        asset_path = dest_dir / asset_name
        pix.save(asset_path)
        return asset_name, asset_path, int(pix.width), int(pix.height)
    return None


def apply_figure_selection_to_spec(
    spec: Mapping[str, Any],
    figure_selection: Mapping[str, Any],
    *,
    prune_unselected: bool = True,
) -> dict[str, Any]:
    """Update placeholder assets/labels from an LLM selection result."""
    updated = copy.deepcopy(dict(spec))
    selected = list(figure_selection.get("selected_figures") or [])
    if not selected:
        return updated

    selected = sorted(selected, key=lambda item: int(_safe_int(item.get("priority"), 999)))
    existing = {_canonical_placeholder_id(p.get("id")): dict(p) for p in updated.get("placeholders") or []}
    new_placeholders: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        placeholder_id = _canonical_placeholder_id(item.get("placeholder_id") or item.get("id") or normalize_placeholder_id(index))
        row = existing.get(placeholder_id, {"id": placeholder_id})
        row["id"] = placeholder_id
        section_value = item.get("section") if item.get("section") is not None else item.get("target_section")
        if section_value is not None:
            row["section"] = int(_safe_int(section_value, row.get("section") or 1))
            row["target_section"] = row["section"]
        if item.get("role"):
            row["role"] = str(item["role"])
        if item.get("reason") or item.get("rationale"):
            row["reason"] = str(item.get("reason") or item.get("rationale"))
            row["rationale"] = str(item.get("rationale") or item.get("reason"))
        if item.get("label"):
            row["label"] = str(item["label"])
        if item.get("aspect"):
            row["aspect"] = str(item["aspect"])
        if item.get("asset"):
            row["asset"] = str(item["asset"])
        new_placeholders.append(row)

    if prune_unselected:
        updated["placeholders"] = new_placeholders
    else:
        replacement = {row["id"]: row for row in new_placeholders}
        merged: list[dict[str, Any]] = []
        for old in updated.get("placeholders") or []:
            key = _canonical_placeholder_id(old.get("id"))
            merged.append(replacement.pop(key, dict(old)))
        merged.extend(replacement.values())
        updated["placeholders"] = merged
    updated["placements"] = {
        _canonical_placeholder_id(key): value
        for key, value in (updated.get("placements") or {}).items()
        if not prune_unselected or _canonical_placeholder_id(key) in {p["id"] for p in updated["placeholders"]}
    }
    return updated


def apply_detections_to_spec(
    spec: Mapping[str, Any],
    detections: Mapping[str, Any],
    *,
    min_confidence: float = 0.15,
) -> dict[str, Any]:
    """Write detected placeholder boxes into ``spec['placements']``."""
    updated = copy.deepcopy(dict(spec))
    expected_ids = {_canonical_placeholder_id(p.get("id")) for p in updated.get("placeholders") or []}
    placements: dict[str, list[int]] = dict(updated.get("placements") or {})
    for item in detections.get("placeholders") or []:
        placeholder_id = _canonical_placeholder_id(item.get("id"))
        if expected_ids and placeholder_id not in expected_ids:
            continue
        bbox = item.get("bbox")
        confidence = float(item.get("confidence") or 0)
        if confidence < min_confidence or not _valid_bbox(bbox):
            continue
        placements[placeholder_id] = [int(round(v)) for v in bbox]
    updated["placements"] = placements
    image_size = detections.get("image_size") if isinstance(detections, Mapping) else None
    if image_size and isinstance(updated.get("layout_contract"), Mapping):
        updated = attach_layout_contract_boxes(updated, image_size)
        issues = evaluate_layout_contract_alignment(
            updated.get("layout_contract"),
            placements,
            image_size,
        )
        if issues:
            updated["_layout_contract_issues"] = issues
    return updated


def aspect_label(width: int, height: int) -> str:
    if height <= 0:
        return "1:1 square"
    ratio = width / height
    if 0.92 <= ratio <= 1.08:
        return "1:1 square"
    if ratio >= 2.2:
        return f"{ratio:.1f}:1 wide"
    if ratio >= 1.25:
        return f"{ratio:.2f}:1 landscape"
    if ratio <= 0.45:
        return f"1:{1 / ratio:.1f} tall"
    if ratio <= 0.8:
        return f"1:{1 / ratio:.2f} portrait"
    return f"{ratio:.2f}:1"


def infer_asset_label(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"(?i)^pdf-page\d+-img\d+", "PDF extracted figure", stem)
    stem = re.sub(r"(?i)^image\d+", "image", stem)
    stem = re.sub(r"(?i)^fig(?:ure)?[_\-\s]*\d*[a-z]?", "", stem).strip("_- ")
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem[:1].upper() + stem[1:] if stem else Path(filename).stem


def _canonical_placeholder_id(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = text.strip("[](){}")
    match = re.search(r"FIG(?:URE)?\s*0*([0-9]+)", text)
    if match:
        return normalize_placeholder_id(int(match.group(1)))
    return text or "FIG 01"


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 4:
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in value]
    except Exception:
        return False
    return x1 > x0 and y1 > y0


def _ignored_asset_name(name: str) -> bool:
    low = name.lower()
    return any(part in low for part in IGNORED_NAME_PARTS)


def _unique_asset_name(name: str, used: set[str]) -> str:
    src = Path(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", src.stem).strip("-._") or "asset"
    suffix = src.suffix.lower() or ".png"
    candidate = f"{stem}{suffix}"
    index = 2
    while candidate in used:
        candidate = f"{stem}-{index}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def _dedupe_existing_dirs(roots: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root.absolute()
        if not resolved.exists() or not resolved.is_dir() or resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
