from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops

from .fonts import load_font


def replace_placeholders(
    *,
    base_image: str | Path,
    spec: dict[str, Any],
    asset_dir: str | Path,
    out_path: str | Path,
    scale: float = 1.0,
    dry_run: bool = False,
) -> Path:
    base = Image.open(base_image).convert('RGB')
    canvas = base.copy()
    assets = Path(asset_dir)
    mapping = spec.get('placements') or {}
    placeholders = {p['id']: p for p in spec.get('placeholders', [])}
    clear_mapping = spec.get('_replacement_clear_boxes') or {}
    render_items: list[tuple[str, Path | None, tuple[int, int, int, int], tuple[int, int, int, int] | None]] = []

    for fig_id, box in mapping.items():
        ph = placeholders.get(fig_id, {})
        asset = ph.get('asset')
        if not asset:
            continue
        x0, y0, x1, y1 = [int(round(v * scale)) for v in box]
        bleed = int(max(2, 3 * scale))
        x0 = max(0, x0 - bleed)
        y0 = max(0, y0 - bleed)
        x1 = min(canvas.width, x1 + bleed)
        y1 = min(canvas.height, y1 + bleed)
        clear_box = None
        if fig_id in clear_mapping:
            try:
                cx0, cy0, cx1, cy1 = [int(round(float(v) * scale)) for v in clear_mapping[fig_id]]
                clear_box = (
                    max(0, min(canvas.width, cx0)),
                    max(0, min(canvas.height, cy0)),
                    max(0, min(canvas.width, cx1)),
                    max(0, min(canvas.height, cy1)),
                )
            except Exception:
                clear_box = None
        render_items.append((str(fig_id), assets / asset, (x0, y0, x1, y1), clear_box))

    if not dry_run:
        for _, _, _, clear_box in render_items:
            if clear_box and clear_box[2] > clear_box[0] and clear_box[3] > clear_box[1]:
                _clear_replacement_region(canvas, clear_box)

    for fig_id, asset_path, box, _ in render_items:
        if dry_run:
            _draw_debug_box(canvas, box, fig_id)
            continue
        assert asset_path is not None
        paste_fit(canvas, asset_path, box, pad=int(max(2, 4 * scale)))


    for overlay in spec.get('text_overlays', []):
        _draw_text_overlay(canvas, overlay, scale=scale)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)
    return out


def paste_fit(canvas: Image.Image, asset_path: Path, box: tuple[int, int, int, int], pad: int = 4) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    # First clear the whole target rectangle.  The placeholder artwork uses
    # dashed borders; a rounded alpha panel alone can leave those dashes visible
    # in transparent corners after replacement.
    canvas.paste(Image.new("RGB", (w, h), (255, 255, 255)), (x0, y0))
    panel = _panel((w, h), radius=max(4, min(w, h)//18))
    canvas.paste(panel.convert('RGB'), (x0, y0), panel.split()[-1])
    im = Image.open(asset_path).convert('RGBA')
    im = _trim_uniform_border(im)
    thumb = ImageOps.contain(im, (max(1, w - 2*pad), max(1, h - 2*pad)), Image.Resampling.LANCZOS)
    bg = Image.new('RGBA', thumb.size, (255,255,255,255))
    bg.alpha_composite(thumb)
    canvas.paste(bg.convert('RGB'), (x0 + (w-thumb.width)//2, y0 + (h-thumb.height)//2))


def _clear_replacement_region(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    canvas.paste(Image.new("RGB", (w, h), (255, 255, 255)), (x0, y0))


def normalize_placeholder_geometry(
    *,
    base_image: str | Path,
    spec: dict[str, Any],
    out_path: str | Path,
    scale: float = 1.0,
    redraw: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Redraw blank placeholder boxes with their declared source aspect ratios.

    Image generation is good at overall poster art but can drift on exact box
    geometry. This deterministic production stage keeps the generated design while
    making replacement boxes match the real source figures before strict QA and
    replacement.
    """
    canvas = Image.open(base_image).convert("RGB")
    updated = copy.deepcopy(spec)
    placements = dict(updated.get("placements") or {})
    placeholders = {str(p.get("id")): p for p in updated.get("placeholders", [])}
    new_placements: dict[str, list[int]] = {}
    clear_boxes: dict[str, list[int]] = {}
    draw_plans: list[tuple[str, str, str, tuple[int, int, int, int], tuple[int, int, int, int]]] = []

    for fig_id, raw_box in placements.items():
        ph = placeholders.get(str(fig_id), {})
        if not ph:
            continue
        try:
            x0, y0, x1, y1 = [int(round(float(v) * scale)) for v in raw_box]
        except Exception:
            continue
        ratio = _parse_placeholder_aspect(str(ph.get("aspect") or ""))
        if not ratio:
            ratio = max(0.1, (x1 - x0) / max(1, y1 - y0))
        original_box = (x0, y0, x1, y1)
        label = str(ph.get("label") or fig_id)
        source_box = _find_enclosing_placeholder_panel(canvas, original_box, ratio) or original_box
        min_size = _minimum_placeholder_size(ratio, label=label)
        box = _fit_box_to_ratio(source_box, ratio, min_size=min_size, canvas_size=canvas.size)
        box = _shrink_to_avoid_busy_overlap(canvas, box, source_box, ratio)
        clear_box = _pad_box(_union_box(source_box, box), 4, canvas.size)
        draw_plans.append((str(fig_id), label, str(ph.get("aspect") or ""), clear_box, box))
        if scale != 1:
            new_placements[str(fig_id)] = [int(round(v / scale)) for v in box]
            clear_boxes[str(fig_id)] = [int(round(v / scale)) for v in clear_box]
        else:
            new_placements[str(fig_id)] = list(box)
            clear_boxes[str(fig_id)] = list(clear_box)

    if redraw:
        for fig_id, label, aspect, clear_box, box in draw_plans:
            _erase_placeholder_region(canvas, clear_box)
            _draw_clean_placeholder(canvas, box, fig_id, label, aspect)

    updated["placements"] = new_placements or placements
    if clear_boxes:
        updated["_replacement_clear_boxes"] = clear_boxes
    if not redraw:
        return Path(base_image), updated
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)
    return out, updated


def audit_generated_placeholder_geometry(
    *,
    base_image: str | Path,
    spec: dict[str, Any],
    scale: float = 1.0,
    ratio_tolerance: float = 0.20,
) -> list[dict[str, Any]]:
    """Deterministically verify visible placeholder aspect ratios.

    LLM QA can be fooled by placeholder label text. This audit checks the pixel
    geometry before any hidden normalization. If the image model drew a 2.5:1
    placeholder as a ribbon-like 6:1 strip, strict autoposter should reject the
    variant instead of trying to hide the mismatch with post-processing.
    """
    canvas = Image.open(base_image).convert("RGB")
    placements = dict(spec.get("placements") or {})
    placeholders = {str(p.get("id")): p for p in spec.get("placeholders", [])}
    issues: list[dict[str, Any]] = []
    for fig_id, raw_box in placements.items():
        ph = placeholders.get(str(fig_id), {})
        if not ph:
            continue
        expected = _parse_placeholder_aspect(str(ph.get("aspect") or ""))
        if not expected:
            continue
        try:
            detected_box = tuple(int(round(float(v) * scale)) for v in raw_box)  # type: ignore[arg-type]
        except Exception:
            continue
        if len(detected_box) != 4:
            continue
        visible_box = _find_enclosing_placeholder_panel(canvas, detected_box, expected) or detected_box
        actual = _box_ratio(visible_box)
        rel_error = _ratio_relative_error(actual, expected)
        if rel_error > ratio_tolerance:
            issues.append(
                {
                    "id": str(fig_id),
                    "expected_ratio": round(expected, 4),
                    "actual_ratio": round(actual, 4),
                    "relative_error": round(rel_error, 4),
                    "detected_box": list(detected_box),
                    "visible_box": list(visible_box),
                    "message": (
                        f"{fig_id} visible placeholder ratio {actual:.2f}:1 does not match "
                        f"declared ratio {expected:.2f}:1 within {ratio_tolerance:.0%}"
                    ),
                }
            )
    return issues


def _parse_placeholder_aspect(aspect: str) -> float | None:
    text = str(aspect or "").strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[:/]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        den = float(match.group(2))
        return float(match.group(1)) / den if den else None
    if "square" in text:
        return 1.0
    return None


def _fit_box_to_ratio(
    box: tuple[int, int, int, int],
    ratio: float,
    *,
    min_size: tuple[int, int] | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Return a ratio-correct replacement box near ``box``.

    The image model often draws the right *region* but not the exact aspect
    ratio.  A shrink-only correction makes wide scientific figures unreadably
    tiny, so this helper is allowed to grow the corrected box to a conservative
    minimum size while keeping the declared source aspect exact.
    """
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    current = width / height
    min_w, min_h = min_size or (1, 1)
    if abs(current / max(ratio, 0.01) - 1.0) <= 0.12:
        # If the generated panel is already close to the declared ratio, keep
        # the visual footprint and trim the excess dimension inside the panel.
        # This avoids growing into neighboring placeholders in dense rows.
        shrunk = _shrink_box_to_ratio(box, ratio)
        if (shrunk[2] - shrunk[0]) >= min_w * 0.90 and (shrunk[3] - shrunk[1]) >= min_h * 0.90:
            return shrunk
    if current > ratio:
        # The generated slot is too wide/thin.  Prefer growing height to avoid
        # making the pasted scientific plot a smaller inset inside the original
        # placeholder panel.
        new_w = max(width, int(round(min_w)))
        new_h = int(round(new_w / ratio))
        if new_h < min_h:
            new_h = int(round(min_h))
            new_w = int(round(new_h * ratio))
    else:
        # The generated slot is too narrow/tall.  Prefer growing width.
        new_h = max(height, int(round(min_h)))
        new_w = int(round(new_h * ratio))
        if new_w < min_w:
            new_w = int(round(min_w))
            new_h = int(round(new_w / ratio))

    if canvas_size:
        canvas_w, canvas_h = canvas_size
        max_w = max(1, canvas_w - 8)
        max_h = max(1, canvas_h - 8)
        if new_w > max_w:
            new_w = max_w
            new_h = int(round(new_w / ratio))
        if new_h > max_h:
            new_h = max_h
            new_w = int(round(new_h * ratio))

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    nx0 = int(round(cx - new_w / 2))
    ny0 = int(round(cy - new_h / 2))
    nx1 = nx0 + new_w
    ny1 = ny0 + new_h
    if canvas_size:
        nx0, ny0, nx1, ny1 = _clamp_box((nx0, ny0, nx1, ny1), canvas_size)
    return nx0, ny0, max(nx0 + 1, nx1), max(ny0 + 1, ny1)


def _minimum_placeholder_size(ratio: float, *, label: str) -> tuple[int, int]:
    """Conservative readable minima for a 1024×1536 generated poster."""
    low = str(label or "").lower()
    hero = any(
        key in low
        for key in (
            "limit",
            "95%",
            "result",
            "constraint",
            "cross section",
            "significance",
            "exclusion",
            "observed",
            "measurement",
        )
    )
    if 0.92 <= ratio <= 1.08:
        side = 285 if hero else 220
        return side, side
    if ratio >= 2.2:
        height = 200
        return int(round(height * ratio)), height
    if ratio > 1.08:
        width = 320 if hero else 240
        return width, max(120, int(round(width / ratio)))
    height = 320 if hero else 250
    return max(120, int(round(height * ratio))), height


def _clamp_box(box: tuple[int, int, int, int], canvas_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    canvas_w, canvas_h = canvas_size
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    if x0 < 4:
        x1 += 4 - x0
        x0 = 4
    if y0 < 4:
        y1 += 4 - y0
        y0 = 4
    if x1 > canvas_w - 4:
        shift = x1 - (canvas_w - 4)
        x0 -= shift
        x1 -= shift
    if y1 > canvas_h - 4:
        shift = y1 - (canvas_h - 4)
        y0 -= shift
        y1 -= shift
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(canvas_w, max(x0 + 1, x1))
    y1 = min(canvas_h, max(y0 + 1, y1))
    # Preserve exact fitted dimensions unless the canvas is too small.
    if x1 - x0 < w and w <= canvas_w:
        x1 = min(canvas_w, x0 + w)
    if y1 - y0 < h and h <= canvas_h:
        y1 = min(canvas_h, y0 + h)
    return x0, y0, x1, y1


def _union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _pad_box(box: tuple[int, int, int, int], pad: int, canvas_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    w, h = canvas_size
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _find_enclosing_placeholder_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int] | None:
    """Find the visible dashed/outlined placeholder panel around an LLM box.

    The detector sometimes returns the text cluster inside a placeholder rather
    than the whole dashed panel.  This lightweight image pass expands to the
    enclosing colored placeholder border when one is visible; if it is uncertain
    it returns ``None`` and the caller keeps the LLM box.
    """
    x0, y0, x1, y1 = box
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    if ratio >= 1.8:
        dx = max(460, int(width * 3.2))
        dy = max(180, int(height * 2.4))
    else:
        dx = max(220, int(width * 1.6))
        dy = max(180, int(height * 1.6))
    sx0, sy0 = max(0, cx - dx), max(0, cy - dy)
    sx1, sy1 = min(canvas.width, cx + dx), min(canvas.height, cy + dy)
    if sx1 <= sx0 + 8 or sy1 <= sy0 + 8:
        return None

    pixels = canvas.load()
    search_w = sx1 - sx0
    row_scores: list[tuple[int, int]] = []
    row_threshold = max(35, int(search_w * 0.18))
    for y in range(sy0, sy1):
        count = 0
        for x in range(sx0, sx1):
            if _is_placeholder_border_pixel(pixels[x, y]):
                count += 1
        if count >= row_threshold:
            row_scores.append((y, count))
    if not row_scores:
        return None
    row_groups = _cluster_line_positions([y for y, _ in row_scores])
    row_group_max = _group_max_counts(row_groups, row_scores)
    top_candidates = [g for g in row_groups if g[1] <= cy - 3]
    bottom_candidates = [g for g in row_groups if g[0] >= cy + 3]
    if not top_candidates or not bottom_candidates:
        return None
    row_limit = max(60, int(height * 0.30))
    row_slop = max(6, int(height * 0.08))
    top_outer_candidates = [
        g
        for g in top_candidates
        if y0 - row_limit <= g[0] <= y0 + row_slop
        and row_group_max.get(g, 0) < search_w * 0.85
    ]
    bottom_outer_candidates = [
        g
        for g in bottom_candidates
        if y1 - row_slop <= g[1] <= y1 + row_limit
        and row_group_max.get(g, 0) < search_w * 0.85
    ]
    top_group = min(top_outer_candidates, key=lambda g: g[0]) if top_outer_candidates else max(top_candidates, key=lambda g: g[1])
    bottom_group = min(bottom_outer_candidates, key=lambda g: g[1]) if bottom_outer_candidates else min(bottom_candidates, key=lambda g: g[0])
    # A dashed border often forms a thick cluster because the vertical sides are
    # counted on rows near the corner.  Use the *outer* edge of that cluster so
    # the replacement panel matches the generated placeholder, not an inset text
    # or corner cluster.
    top = top_group[0]
    bottom = bottom_group[1]
    if bottom - top < max(24, height * 0.55):
        return None

    col_scores: list[tuple[int, int]] = []
    col_threshold = max(18, int((bottom - top + 1) * 0.22))
    for x in range(sx0, sx1):
        count = 0
        for y in range(top, bottom + 1):
            if _is_placeholder_border_pixel(pixels[x, y]):
                count += 1
        if count >= col_threshold:
            col_scores.append((x, count))
    if not col_scores:
        return None
    col_groups = _cluster_line_positions([x for x, _ in col_scores])
    col_group_max = _group_max_counts(col_groups, col_scores)
    left_candidates = [g for g in col_groups if g[1] <= cx - 3]
    right_candidates = [g for g in col_groups if g[0] >= cx + 3]
    if not left_candidates or not right_candidates:
        return None
    panel_h = max(1, bottom - top + 1)
    col_limit = max(60, int(width * 0.30))
    col_slop = max(6, int(width * 0.08))
    left_outer_candidates = [
        g
        for g in left_candidates
        if x0 - col_limit <= g[0] <= x0 + col_slop
        and col_group_max.get(g, 0) < panel_h * 0.90
    ]
    right_outer_candidates = [
        g
        for g in right_candidates
        if x1 - col_slop <= g[1] <= x1 + col_limit
        and col_group_max.get(g, 0) < panel_h * 0.90
    ]
    near_left_candidates = [g for g in left_outer_candidates if abs(g[0] - x0) <= col_slop]
    if near_left_candidates:
        left_group = min(near_left_candidates, key=lambda g: abs(g[0] - x0))
    else:
        left_group = min(left_outer_candidates, key=lambda g: g[0]) if left_outer_candidates else max(left_candidates, key=lambda g: g[1])
    right_group = min(right_outer_candidates, key=lambda g: g[1]) if right_outer_candidates else min(right_candidates, key=lambda g: g[0])
    left = left_group[0]
    right = right_group[1]
    if right - left < max(24, width * 0.55):
        return None

    panel = (left, top, right + 1, bottom + 1)
    # Only accept a genuine expansion/close match.  Very small unrelated text
    # edges can otherwise be mistaken for a panel.
    if _box_area(panel) < _box_area(box) * 0.75:
        return None
    # Do not let a surrounding section/card border masquerade as the placeholder
    # itself.  This happened for artistic result cards that contain a nearly
    # square dashed placeholder: the long card outline was colorful enough to be
    # detected as a "panel", producing a 2:1 replacement region.  Keep the LLM
    # seed box when it is already closer to the declared source ratio and the
    # pixel expansion grows into a much worse geometry.
    raw_error = _ratio_relative_error(_box_ratio(box), ratio)
    panel_error = _ratio_relative_error(_box_ratio(panel), ratio)
    area_growth = _box_area(panel) / max(1, _box_area(box))
    if area_growth > 1.15 and panel_error > raw_error + 0.08:
        return None
    return panel


def _box_ratio(box: tuple[int, int, int, int]) -> float:
    return max(1, box[2] - box[0]) / max(1, box[3] - box[1])


def _ratio_relative_error(actual: float, expected: float) -> float:
    return abs(actual / max(expected, 0.01) - 1.0)


def _is_placeholder_border_pixel(pixel: tuple[int, int, int]) -> bool:
    r, g, b = pixel
    mx, mn = max(pixel), min(pixel)
    if mx < 45 or mn > 248:
        return False
    # Dashed placeholder borders are intentionally colored; neutral black text
    # and pale white backgrounds should not dominate this mask.
    return (mx - mn) >= 28


def _cluster_line_positions(values: list[int], *, max_gap: int = 2) -> list[tuple[int, int]]:
    if not values:
        return []
    values = sorted(set(values))
    groups: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value - prev <= max_gap:
            prev = value
        else:
            groups.append((start, prev))
            start = prev = value
    groups.append((start, prev))
    return groups


def _group_max_counts(groups: list[tuple[int, int]], scores: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for group in groups:
        values = [count for pos, count in scores if group[0] <= pos <= group[1]]
        out[group] = max(values) if values else 0
    return out


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _shrink_to_avoid_busy_overlap(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    source_box: tuple[int, int, int, int],
    ratio: float,
    *,
    max_busy_density: float = 0.055,
) -> tuple[int, int, int, int]:
    """Reduce deterministic expansion if it would cover existing public text.

    The normalizer can enlarge incorrect placeholder boxes, but it should not
    bulldoze a generated section full of readable text or another module.  We
    inspect only the newly-added area outside the original visible placeholder.
    If that area is visually busy, we smoothly scale back toward the largest
    ratio-correct box that fits inside the source panel.
    """
    if _busy_overlap_density(canvas, box, source_box) <= max_busy_density:
        return box
    min_box = _shrink_box_to_ratio(source_box, ratio)
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    min_w = max(1, min_box[2] - min_box[0])
    min_h = max(1, min_box[3] - min_box[1])
    w = max(1, box[2] - box[0])
    h = max(1, box[3] - box[1])
    best = min_box
    for _ in range(18):
        w = max(min_w, int(round(w * 0.90)))
        h = max(min_h, int(round(w / ratio)))
        if h < min_h:
            h = min_h
            w = int(round(h * ratio))
        candidate = (
            int(round(cx - w / 2)),
            int(round(cy - h / 2)),
            int(round(cx - w / 2)) + w,
            int(round(cy - h / 2)) + h,
        )
        candidate = _clamp_box(candidate, canvas.size)
        if _busy_overlap_density(canvas, candidate, source_box) <= max_busy_density:
            return candidate
        best = candidate
        if w <= min_w + 1 and h <= min_h + 1:
            break
    return _clamp_box(best, canvas.size)


def _shrink_box_to_ratio(box: tuple[int, int, int, int], ratio: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    current = width / height
    if current > ratio:
        new_h = height
        new_w = int(round(height * ratio))
    else:
        new_w = width
        new_h = int(round(width / ratio))
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    nx0 = int(round(cx - new_w / 2))
    ny0 = int(round(cy - new_h / 2))
    return nx0, ny0, nx0 + new_w, ny0 + new_h


def _busy_overlap_density(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    source_box: tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = _pad_box(box, 0, canvas.size)
    sx0, sy0, sx1, sy1 = source_box
    pixels = canvas.load()
    busy = 0
    total = 0
    # Sample every other pixel to keep this cheap while still catching text
    # blocks, section borders, icons, and other figures.
    for y in range(y0, y1, 2):
        for x in range(x0, x1, 2):
            if sx0 <= x < sx1 and sy0 <= y < sy1:
                continue
            total += 1
            if _is_busy_pixel(pixels[x, y]):
                busy += 1
    if total == 0:
        return 0.0
    return busy / total


def _is_busy_pixel(pixel: tuple[int, int, int]) -> bool:
    r, g, b = pixel
    mx, mn = max(pixel), min(pixel)
    lum = (r + g + b) / 3
    if lum < 170:
        return True
    return (mx - mn) > 35 and mx < 250


def _erase_placeholder_region(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    panel = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    d = ImageDraw.Draw(panel)
    radius = max(4, min(w, h) // 18)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=(255, 255, 255, 255), outline=(210, 220, 235, 255), width=1)
    canvas.paste(panel.convert("RGB"), (x0, y0), panel.split()[-1])


def _draw_clean_placeholder(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    fig_id: str,
    label: str,
    aspect: str,
) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    overlay = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    d = ImageDraw.Draw(overlay)
    radius = max(6, min(w, h) // 20)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=(255, 255, 255, 255), outline=(95, 95, 120, 255), width=2)
    _draw_dashed_rect(d, [4, 4, w - 5, h - 5], fill=(105, 70, 140, 255), width=2, dash=10)
    bold_size = max(14, min(28, h // 8))
    body_size = max(8, min(18, h // 12))
    small_size = max(8, min(15, h // 14))
    bold = load_font(bold_size, bold=True)
    font = load_font(body_size)
    small = load_font(small_size)
    lines, fonts = _placeholder_text_layout(
        d,
        fig_id=fig_id,
        label=label,
        aspect=aspect,
        max_width=max(40, w - 36),
        max_height=max(40, h - 24),
        bold=bold,
        font=font,
        small=small,
    )
    line_heights = [(getattr(f, "size", 14) if f is not None else 14) + 4 for f in fonts[: len(lines)]]
    total_h = sum(line_heights)
    cur_y = max(12, (h - total_h) // 2)
    for line, f, lh in zip(lines, fonts, line_heights):
        try:
            bbox = d.textbbox((0, 0), line, font=f)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(line) * 7
        d.text(((w - tw) / 2, cur_y), line, fill=(35, 35, 55, 255), font=f)
        cur_y += lh
    canvas.paste(overlay.convert("RGB"), (x0, y0), overlay.split()[-1])


def _placeholder_text_layout(
    draw: ImageDraw.ImageDraw,
    *,
    fig_id: str,
    label: str,
    aspect: str,
    max_width: int,
    max_height: int,
    bold,
    font,
    small,
) -> tuple[list[str], list[Any]]:
    if font is None:
        lines = [f"[{fig_id}]", *_wrap_text(label, max_width, draw, None)]
        if aspect:
            lines.append(f"aspect {aspect}")
        return lines, [None] * len(lines)
    body_start = getattr(font, "size", 10)
    for body_size in range(body_start, 7, -1):
        body = load_font(body_size)
        title = load_font(max(body_size + 3, min(getattr(bold, "size", body_size + 4), body_size + 8)), bold=True)
        tiny = load_font(max(8, body_size - 1))
        label_lines = _wrap_text(label, max_width, draw, body)
        lines = [f"[{fig_id}]", *label_lines]
        if aspect:
            lines.append(f"aspect {aspect}")
        fonts = [title] + [body] * len(label_lines) + ([tiny] if aspect else [])
        total = sum((getattr(f, "size", body_size) + 4) for f in fonts)
        if total <= max_height:
            return lines, fonts
    body = load_font(8)
    title = load_font(11, bold=True)
    tiny = load_font(8)
    label_lines = _wrap_text(label, max_width, draw, body)
    lines = [f"[{fig_id}]", *label_lines]
    if aspect:
        lines.append(f"aspect {aspect}")
    return lines, [title] + [body] * len(label_lines) + ([tiny] if aspect else [])


def _draw_dashed_rect(draw: ImageDraw.ImageDraw, box: list[int], *, fill, width: int = 1, dash: int = 8) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0, x1, dash * 2):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=fill, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=fill, width=width)
    for y in range(y0, y1, dash * 2):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=fill, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=fill, width=width)


def _trim_uniform_border(im: Image.Image, *, tolerance: int = 12, margin: int = 8) -> Image.Image:
    """Trim near-uniform white/transparent margins before figure replacement.

    This does not alter scientific content; it only removes empty border area so
    axes/legends use more of the placeholder box.
    """
    rgba = im.convert('RGBA')
    # Composite on white so transparent PDF render margins behave like paper.
    white = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    rgb = white.convert('RGB')
    bg = Image.new('RGB', rgb.size, rgb.getpixel((0, 0)))
    diff = ImageChops.difference(rgb, bg).convert('L')
    mask = diff.point(lambda px: 255 if px > tolerance else 0)
    bbox = mask.getbbox()
    if not bbox:
        return im
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(im.width, x1 + margin)
    y1 = min(im.height, y1 + margin)
    # Avoid pathological crops caused by a single non-white speck.
    if (x1 - x0) < im.width * 0.2 or (y1 - y0) < im.height * 0.2:
        return im
    return im.crop((x0, y0, x1, y1))


def upscale_image(input_path: str | Path, out_path: str | Path, factor: float = 2.0, sharpen: bool = True) -> Path:
    """Deterministic high-res export helper.

    This does not invent scientific detail; it makes text/edges cleaner for review.
    Production-grade output should use a higher native image-generation size when
    available, then run this only as a final polish.
    """
    im = Image.open(input_path).convert('RGB')
    w, h = im.size
    up = im.resize((int(w*factor), int(h*factor)), Image.Resampling.LANCZOS)
    if sharpen:
        up = up.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    up.save(out, quality=95)
    return out


def _panel(size: tuple[int, int], radius: int = 7, fill=(255,255,255), outline=(150,175,205)) -> Image.Image:
    img = Image.new('RGBA', size, (0,0,0,0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0,0,size[0]-1,size[1]-1], radius=radius, fill=fill+(255,), outline=outline+(255,), width=1)
    return img


def _draw_debug_box(canvas: Image.Image, box: tuple[int, int, int, int], label: str) -> None:
    d = ImageDraw.Draw(canvas)
    d.rectangle(box, outline=(255, 0, 0), width=3)
    d.text((box[0]+5, box[1]+5), label, fill=(255, 0, 0))


def _draw_text_overlay(canvas: Image.Image, overlay: dict[str, Any], scale: float = 1.0) -> None:
    box = overlay.get('box')
    if not box:
        return
    x0, y0, x1, y1 = [int(round(v * scale)) for v in box]
    fill = tuple(overlay.get('fill', [5, 30, 65]))
    text_color = tuple(overlay.get('color', [255, 255, 255]))
    d = ImageDraw.Draw(canvas)
    radius = int(overlay.get('radius', 8) * scale)
    d.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)
    font_path = overlay.get('font') or None
    font_size = max(8, int(overlay.get('font_size', 12) * scale))
    small_size = max(8, int(overlay.get('small_font_size', font_size) * scale))
    heading_size = max(font_size, int(overlay.get('heading_font_size', font_size + 2) * scale))
    font = load_font(font_size, preferred=font_path)
    small = load_font(small_size, preferred=font_path)
    heading_font = load_font(heading_size, bold=True)
    pad = int(overlay.get('padding', 10) * scale)
    cur_y = y0 + pad
    max_w = x1 - x0 - 2 * pad
    line_gap = max(1, int(overlay.get('line_gap', 3) * scale))
    for line in overlay.get('lines', []):
        raw = str(line)
        is_heading = raw.startswith('__heading__:')
        is_closing = raw.startswith('__closing__:')
        text = raw.replace('__heading__:', '', 1).replace('__closing__:', '', 1)
        f = heading_font if is_heading else (small if is_closing else font)
        prefix = '' if (is_heading or is_closing) else '• '
        wrapped_lines = _wrap_text(text, max_w - int(10*scale), d, f)
        for i, wrapped in enumerate(wrapped_lines):
            d.text((x0 + pad, cur_y), (prefix if i == 0 else '  ') + wrapped, fill=text_color, font=f)
            cur_y += (getattr(f, 'size', font_size) if f is not None else font_size) + line_gap
        cur_y += int(overlay.get('paragraph_gap', 2) * scale)

def _wrap_text(text: str, max_width: int, draw: ImageDraw.ImageDraw, font) -> list[str]:
    words = text.split()
    if not words:
        return ['']
    lines: list[str] = []
    cur = ''
    for word in words:
        trial = word if not cur else cur + ' ' + word
        try:
            width = draw.textbbox((0, 0), trial, font=font)[2]
        except Exception:
            width = len(trial) * 7
        if width <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines
