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
    erase_mapping = spec.get('_replacement_erase_boxes') or {}
    frame_mapping = spec.get('_replacement_frame_boxes') or {}
    render_items: list[
        tuple[
            str,
            Path | None,
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            bool,
        ]
    ] = []

    for fig_id, box in mapping.items():
        ph = placeholders.get(fig_id, {})
        asset = ph.get('asset')
        if not asset:
            continue
        x0, y0, x1, y1 = [int(round(v * scale)) for v in box]
        x0 = max(0, min(canvas.width, x0))
        y0 = max(0, min(canvas.height, y0))
        x1 = max(0, min(canvas.width, x1))
        y1 = max(0, min(canvas.height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
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
        has_clear_box = bool(clear_box and clear_box[2] > clear_box[0] and clear_box[3] > clear_box[1])
        if has_clear_box and not _is_contained((x0, y0, x1, y1), clear_box, margin=0):
            raise ValueError(
                f"{fig_id} target box {[x0, y0, x1, y1]} is not fully contained by "
                f"its detected placeholder boundary {list(clear_box)}"
            )
        # The cleanup/final-frame rectangle is the hard visual envelope for the
        # replacement.  Do not run outward border scans or pad outward at export
        # time: they can mistake nearby section/card borders for placeholder
        # borders, and even a tiny downward pad can intrude into a neighboring
        # conclusion strip on dense posters.
        cleanup_box = clear_box if has_clear_box else (x0, y0, x1, y1)
        erase_box = _read_optional_box(
            erase_mapping,
            str(fig_id),
            scale=scale,
            canvas_size=canvas.size,
        ) or _default_erase_box(cleanup_box, ph, canvas.size, scale=scale)
        frame_box = _read_optional_box(
            frame_mapping,
            str(fig_id),
            scale=scale,
            canvas_size=canvas.size,
        ) or cleanup_box
        aspect_ratio = _parse_placeholder_aspect(str(ph.get("aspect") or "")) or _box_ratio((x0, y0, x1, y1))
        is_square_result = _is_result_like_square_placeholder(ph, aspect_ratio)
        if is_square_result:
            erase_box = _intersect_box(erase_box, cleanup_box) or cleanup_box
            frame_box = _intersect_box(frame_box, cleanup_box) or cleanup_box
        if not _is_contained((x0, y0, x1, y1), frame_box, margin=0):
            frame_box = _union_box(frame_box, (x0, y0, x1, y1))
            frame_box = _clamp_box(frame_box, canvas.size)
        render_items.append((str(fig_id), assets / asset, (x0, y0, x1, y1), cleanup_box, erase_box, frame_box, is_square_result))

    _validate_render_plan(render_items, canvas_size=canvas.size)

    if not dry_run:
        for _, _, _, _, erase_box, frame_box, is_square_result in render_items:
            if is_square_result:
                _erase_placeholder_region_with_sampled_fill(canvas, erase_box)
            else:
                _erase_final_placeholder_region(canvas, erase_box)
            _draw_final_figure_frame(canvas, frame_box)

    for fig_id, asset_path, box, _, _, _, _ in render_items:
        if dry_run:
            _draw_debug_box(canvas, box, fig_id)
            continue
        assert asset_path is not None
        paste_fit(canvas, asset_path, box, pad=0)


    for overlay in spec.get('text_overlays', []):
        _draw_text_overlay(canvas, overlay, scale=scale)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)
    return out


def paste_fit(canvas: Image.Image, asset_path: Path, box: tuple[int, int, int, int], pad: int = 4) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    # The outer cleanup/final frame is drawn before this call.  Here we only
    # clear the exact image target and paste the contained scientific figure.
    canvas.paste(Image.new("RGB", (w, h), (255, 255, 255)), (x0, y0))
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


def _read_optional_box(
    mapping: Any,
    fig_id: str,
    *,
    scale: float,
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(mapping, dict) or fig_id not in mapping:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v) * scale)) for v in mapping[fig_id]]
    except Exception:
        return None
    box = (
        max(0, min(canvas_size[0], x0)),
        max(0, min(canvas_size[1], y0)),
        max(0, min(canvas_size[0], x1)),
        max(0, min(canvas_size[1], y1)),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


def _default_erase_box(
    cleanup_box: tuple[int, int, int, int],
    ph: dict[str, Any],
    canvas_size: tuple[int, int],
    *,
    scale: float,
) -> tuple[int, int, int, int]:
    visual_scale = max(float(scale), canvas_size[0] / 1024.0, canvas_size[1] / 1536.0)
    pad = int(round(6 * visual_scale))
    aspect_ratio = _parse_placeholder_aspect(str(ph.get("aspect") or "")) or _box_ratio(cleanup_box)
    if _is_result_like_square_placeholder(ph, aspect_ratio):
        # Square headline/result plots often sit immediately above another
        # summary/conclusion module.  A symmetric erase pad is visually unsafe:
        # it creates a plain white rectangle outside the real placeholder even
        # when the plotted image itself is contained.  For these dense hero
        # result cards, the final visible envelope must be exactly the detected
        # placeholder, not a padded cleanup rectangle.
        return cleanup_box
    if _is_lower_hero_placeholder(
        ph,
        aspect_ratio,
        cleanup_box,
        canvas_size,
    ):
        # Never erase downward into a bottom conclusion strip.  The lower-hero
        # planner supplies a custom erase box when it needs to remove old top
        # placeholder text.
        x0, y0, x1, y1 = cleanup_box
        return _clamp_box((x0 - pad, y0 - pad, x1 + pad, y1), canvas_size)
    # Generic plot placeholders are often followed immediately by bullet rows.
    # A symmetric erase pad can blank the first bullet even when the real figure
    # is perfectly contained.  The final publication frame already covers the
    # detected placeholder border, so keep the lower edge fixed and only allow a
    # small top/side cleanup margin.
    x0, y0, x1, y1 = cleanup_box
    return _clamp_box((x0 - pad, y0 - pad, x1 + pad, y1), canvas_size)


def _erase_final_placeholder_region(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    """Erase old placeholder text/dashes without drawing a visible final frame."""
    x0, y0, x1, y1 = box
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    # This is an eraser, not the final visible frame.  Use a full rectangular
    # clear so antialiased dashed placeholder strokes just outside/at the
    # detected edge cannot survive in the rounded-corner cutouts of the final
    # publication frame.
    canvas.paste(Image.new("RGB", (w, h), (255, 255, 255)), (x0, y0))


def _erase_placeholder_region_with_sampled_fill(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    """Erase square-result placeholder edge artifacts without blurring text.

    For dense headline result cards, the original image-generation placeholder
    is often a dashed rectangle on a cream/gradient card.  Clearing that whole
    region to pure white makes a giant "white board" that looks larger than the
    intended placeholder.  Earlier versions used a blurred local repair mask;
    that looked especially bad when a public callout was close to the result
    slot because the callout text became smeared.  Instead, only repaint
    line-like placeholder border artifacts with a sampled card fill.  The
    smaller publication frame is drawn separately around the real figure.
    """
    x0, y0, x1, y1 = box
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    crop = canvas.crop((x0, y0, x1, y1)).convert("RGB")
    mask = _placeholder_artifact_mask(crop)
    if mask.getbbox() is None:
        return
    fill = Image.new("RGB", (w, h), _sample_light_card_fill(canvas, box))
    canvas.paste(fill, (x0, y0), mask)


def _placeholder_artifact_mask(crop: Image.Image) -> Image.Image:
    """Mask only line-like placeholder dashes, not nearby public text."""
    rgb = crop.convert("RGB")
    mask = Image.new("L", rgb.size, 0)
    px = rgb.load()
    mp = mask.load()
    if rgb.width <= 0 or rgb.height <= 0:
        return mask

    artifact: list[list[bool]] = []
    row_counts: list[int] = []
    col_counts = [0 for _ in range(rgb.width)]
    for y in range(rgb.height):
        row: list[bool] = []
        count = 0
        for x in range(rgb.width):
            is_artifact = _is_square_placeholder_artifact_pixel(px[x, y])
            row.append(is_artifact)
            if is_artifact:
                count += 1
                col_counts[x] += 1
        artifact.append(row)
        row_counts.append(count)

    # Placeholder dashes are line-like structures near the replacement boundary.
    # Public callout text may also be dark, but it should not form both border
    # rows/columns at the outer edge of the placeholder cleanup box.
    edge_band_y = max(8, int(round(rgb.height * 0.16)))
    edge_band_x = max(8, int(round(rgb.width * 0.16)))
    row_threshold = max(12, int(round(rgb.width * 0.10)))
    col_threshold = max(12, int(round(rgb.height * 0.10)))
    x_probe = max(4, int(round(rgb.width * 0.14)))
    y_probe = max(4, int(round(rgb.height * 0.14)))

    def row_has_edge_hits(y: int) -> bool:
        row = artifact[y]
        return any(row[x] for x in range(0, min(rgb.width, x_probe))) and any(
            row[x] for x in range(max(0, rgb.width - x_probe), rgb.width)
        )

    def col_has_edge_hits(x: int) -> bool:
        return any(artifact[y][x] for y in range(0, min(rgb.height, y_probe))) and any(
            artifact[y][x] for y in range(max(0, rgb.height - y_probe), rgb.height)
        )

    candidate_rows = {
        y
        for y, count in enumerate(row_counts)
        if count >= row_threshold
        and (y <= edge_band_y or y >= rgb.height - edge_band_y - 1)
        and row_has_edge_hits(y)
    }
    candidate_cols = {
        x
        for x, count in enumerate(col_counts)
        if count >= col_threshold
        and (x <= edge_band_x or x >= rgb.width - edge_band_x - 1)
        and col_has_edge_hits(x)
    }
    if not candidate_rows and not candidate_cols:
        return mask

    radius = max(1, min(4, int(round(min(rgb.width, rgb.height) * 0.006))))
    for y, row in enumerate(artifact):
        near_row = any(abs(y - yy) <= radius for yy in candidate_rows)
        for x, is_artifact in enumerate(row):
            if not is_artifact:
                continue
            if near_row or any(abs(x - xx) <= radius for xx in candidate_cols):
                mp[x, y] = 255
    # Slightly dilate to catch antialiasing, but do not blur: blur was the cause
    # of smeared public text in dense result cards.
    return mask.filter(ImageFilter.MaxFilter(3))


def _is_square_placeholder_artifact_pixel(pixel: tuple[int, int, int]) -> bool:
    r, g, b = pixel
    mx, mn = max(pixel), min(pixel)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    chroma = mx - mn
    # Placeholder text/dashes are neutral dark gray/black in the generated
    # layouts.  Avoid broad saturated blue/gold detector artwork in the card.
    if lum < 225 and chroma < 80:
        return True
    if lum < 145:
        return True
    return False


def _sample_light_card_fill(canvas: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    x0, y0, x1, y1 = _pad_box(box, 0, canvas.size)
    pixels = canvas.load()
    step = max(1, int(round(min(max(1, x1 - x0), max(1, y1 - y0)) / 90)))
    samples: list[tuple[int, int, int]] = []
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            pixel = pixels[x, y]
            if _is_placeholder_line_pixel(pixel):
                continue
            r, g, b = pixel
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if lum < 190:
                continue
            if max(pixel) - min(pixel) > 95:
                continue
            samples.append(pixel)
    if not samples:
        return (248, 246, 240)
    samples.sort(key=lambda p: 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
    # Use a lower-middle bright sample rather than the maximum; this avoids
    # turning a cream CMS card into a stark white rectangle.
    return samples[int(len(samples) * 0.35)]


def _validate_render_plan(
    items: list[
        tuple[
            str,
            Path | None,
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            bool,
        ]
    ],
    *,
    canvas_size: tuple[int, int] | None = None,
) -> None:
    """Reject unsafe replacement plans instead of silently deforming figures.

    Replacement is a deterministic production stage, not a layout engine.  If
    two figure targets overlap, shrinking one of them after the fact produces
    unreadable "stickers" and can hide scientific content.  The correct strict
    behavior is to reject the generated/detected geometry so the pipeline can
    choose another variant or report a real error.
    """
    for fig_id, _asset, target, cleanup, _erase, _frame, _is_square_result in items:
        if not _is_contained(target, cleanup, margin=0):
            raise ValueError(
                f"{fig_id} target box {list(target)} is not fully contained by "
                f"its placeholder cleanup box {list(cleanup)}"
            )
        if canvas_size is not None:
            margin_x = max(4, int(round(canvas_size[0] * 0.015)))
            margin_y = max(4, int(round(canvas_size[1] * 0.010)))
            if (
                target[0] < margin_x
                or cleanup[0] < margin_x
                or target[1] < margin_y
                or cleanup[1] < margin_y
                or target[2] > canvas_size[0] - margin_x
                or cleanup[2] > canvas_size[0] - margin_x
                or target[3] > canvas_size[1] - margin_y
                or cleanup[3] > canvas_size[1] - margin_y
            ):
                raise ValueError(
                    f"{fig_id} replacement box target={list(target)} cleanup={list(cleanup)} "
                    f"is too close to the canvas edge; leave a safe poster gutter"
                )
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            id_i, _asset_i, box_i, _cleanup_i, _erase_i, _frame_i, _is_square_i = items[i]
            id_j, _asset_j, box_j, _cleanup_j, _erase_j, _frame_j, _is_square_j = items[j]
            overlap = _box_overlap(box_i, box_j)
            if overlap <= 0:
                continue
            min_area = max(1, min(_box_area(box_i), _box_area(box_j)))
            ratio = overlap / min_area
            if ratio > 0.03:
                raise ValueError(
                    f"{id_i} target box {list(box_i)} overlaps {id_j} "
                    f"target box {list(box_j)} by {ratio:.1%} of the smaller figure"
                )


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Return overlap area in pixels, or 0 if no overlap."""
    ox0 = max(a[0], b[0])
    oy0 = max(a[1], b[1])
    ox1 = min(a[2], b[2])
    oy1 = min(a[3], b[3])
    if ox1 <= ox0 or oy1 <= oy0:
        return 0
    return (ox1 - ox0) * (oy1 - oy0)


def _enforce_containment(
    box: tuple[int, int, int, int],
    container: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    """Ensure ``box`` is strictly inside ``container``, preserving ratio.

    If ``box`` already fits, return it unchanged.  Otherwise, fit the largest
    ratio-correct rectangle inside ``container`` centered on ``box``'s center.
    """
    if _is_contained(box, container):
        return box
    return _fit_box_to_ratio_inside(container, ratio)


def _is_contained(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
    margin: int = 2,
) -> bool:
    """Check if ``inner`` is strictly inside ``outer`` with a small margin."""
    return (
        inner[0] >= outer[0] + margin
        and inner[1] >= outer[1] + margin
        and inner[2] <= outer[2] - margin
        and inner[3] <= outer[3] - margin
    )


def _draw_final_figure_frame(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    """Replace placeholder styling with a clean final figure panel.

    This deliberately uses a solid, publication-style frame rather than dashed
    placeholder cues.  It is drawn inside the cleanup box only; the real figure
    itself is pasted later into its stricter contained placement box.
    """
    x0, y0, x1, y1 = box
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    overlay = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    d = ImageDraw.Draw(overlay)
    radius = max(4, min(22, min(w, h) // 18))
    # A light solid frame gives the replacement a finished look and covers
    # dashed placeholder borders without extending the pasted figure.
    d.rounded_rectangle(
        [0, 0, w - 1, h - 1],
        radius=radius,
        fill=(255, 255, 255, 255),
        outline=(188, 198, 215, 255),
        width=max(1, min(3, min(w, h) // 180)),
    )
    inset = max(2, min(8, min(w, h) // 45))
    if w > inset * 2 + 2 and h > inset * 2 + 2:
        d.rounded_rectangle(
            [inset, inset, w - inset - 1, h - inset - 1],
            radius=max(2, radius - inset),
            outline=(232, 236, 244, 255),
            width=1,
        )
    canvas.paste(overlay.convert("RGB"), (x0, y0), overlay.split()[-1])


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
    erase_boxes: dict[str, list[int]] = {}
    frame_boxes: dict[str, list[int]] = {}
    preexisting_clear_mapping = updated.get("_replacement_clear_boxes") or {}
    contract_search_mapping = updated.get("_layout_contract_search_boxes") or {}
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
        source_box = _repair_edge_wide_source_box(source_box, original_box, ratio, canvas.size)
        source_box = _enforce_canvas_gutter(source_box, canvas.size)
        contract_search_box = _read_optional_box(
            contract_search_mapping,
            str(fig_id),
            scale=scale,
            canvas_size=canvas.size,
        )
        source_box = _apply_layout_contract_search_constraint(
            source_box,
            original_box,
            contract_search_box,
            ratio,
        )
        source_box = _refine_square_result_source_to_dashed_placeholder(
            canvas,
            source_box,
            ph,
            ratio,
        )
        raw_source_box = source_box
        source_box = _constrain_lower_hero_placeholder_box(source_box, ratio, ph, canvas.size)
        lower_hero = _is_lower_hero_placeholder(ph, ratio, source_box, canvas.size)
        if redraw:
            min_size = _minimum_placeholder_size(ratio, label=label)
            box = _fit_box_to_ratio(source_box, ratio, min_size=min_size, canvas_size=canvas.size)
            box = _shrink_to_avoid_busy_overlap(canvas, box, source_box, ratio)
            clear_box = _pad_box(_union_box(source_box, box), 4, canvas.size)
        else:
            # Hidden production planning must not move the final real figure
            # outside the original generated placeholder.  Maximize overlap by
            # taking the largest ratio-correct rectangle that fits *inside* the
            # visible placeholder panel.  Then enforce strict containment so
            # the pasted figure never protrudes beyond the dashed border.
            box = _fit_box_to_ratio_inside(source_box, ratio)
            box = _enforce_containment(box, source_box, ratio)
            clear_box = source_box
            erase_box = _default_erase_box(source_box, ph, canvas.size, scale=scale)
            frame_box = source_box
            if lower_hero:
                # A lower hero comparison/result slot is often adjacent to a
                # bottom conclusion band.  It still needs enough erase surface
                # to remove the old dashed placeholder top/label, but the final
                # visible frame must not grow downward into the next block.
                erase_box = (
                    max(0, raw_source_box[0] - int(round((raw_source_box[2] - raw_source_box[0]) * 0.04))),
                    raw_source_box[1],
                    min(canvas.size[0], raw_source_box[2] + int(round((raw_source_box[2] - raw_source_box[0]) * 0.04))),
                    source_box[3],
                )
                frame_pad_x = int(round(max(10, (box[2] - box[0]) * 0.025)))
                frame_box = _clamp_box(
                    (
                        box[0] - frame_pad_x,
                        box[1],
                        box[2] + frame_pad_x,
                        box[3],
                    ),
                    canvas.size,
                )
            if _is_lower_hero_placeholder(ph, ratio, source_box, canvas.size):
                pad_x = int(round(max(12, (source_box[2] - source_box[0]) * 0.04)))
                clear_box = (
                    max(0, clear_box[0] - pad_x),
                    clear_box[1],
                    min(canvas.size[0], clear_box[2] + pad_x),
                    clear_box[3],
                )
            if _needs_supporting_plot_text_clearance(ph, ratio) and _busy_content_below_placeholder(canvas, source_box):
                # Wide fit/distribution plots often sit immediately above
                # analysis-strategy bullets.  Preserve the detected placeholder
                # as the hard containment envelope, but draw/paste a slightly
                # shorter publication frame so it cannot blanket the first bullet
                # row.  Still erase the *full* detected placeholder first;
                # otherwise old dashed borders/labels/aspect text can remain
                # visible below the shorter final frame.  This is a deterministic
                # single-figure repair, not a fallback layout rewrite.
                box = _inset_supporting_wide_target_box(source_box, ratio)
                frame_box = _supporting_wide_frame_around_target(box, source_box)
                erase_box = source_box
            # For square result plots, LLM detection often captures a good
            # ratio-correct seed that is visually just a little too low/right
            # inside a decorative result block.  Do not expand to the full
            # decorative dashed panel (that can cover the conclusion block);
            # instead apply a small top-left nudge while clearing the original
            # seed area to avoid leftover placeholder text.
            source_ratio_error = _ratio_relative_error(_box_ratio(source_box), ratio)
            if (
                source_box == original_box
                and source_ratio_error <= 0.12
                and str(fig_id) not in preexisting_clear_mapping
            ):
                nudged = _nudge_square_result_box(box, ratio, label, canvas.size)
                if nudged != box:
                    # Move the final white publication frame with the square
                    # plot.  Keeping the old lower edge creates a visibly large
                    # right/bottom white mat and makes the plot look shifted
                    # up-left; the frame is a visual replacement envelope, not
                    # a separate eraser.
                    dx = nudged[0] - box[0]
                    dy = nudged[1] - box[1]
                    clear_box = _clamp_box(
                        (
                            clear_box[0] + dx,
                            clear_box[1] + dy,
                            clear_box[2] + dx,
                            clear_box[3] + dy,
                        ),
                        canvas.size,
                    )
                    frame_box = _clamp_box(
                        (
                            frame_box[0] + dx,
                            frame_box[1] + dy,
                            frame_box[2] + dx,
                            frame_box[3] + dy,
                        ),
                        canvas.size,
                    )
                    # Keep erasing the old detected placeholder region as well
                    # as the nudged frame, otherwise dashed placeholder strokes
                    # can remain at the original bottom/right edges.
                    erase_box = _clamp_box(_union_box(erase_box, frame_box), canvas.size)
                    box = nudged
            box, clear_box, erase_box, frame_box = _repair_square_result_replacement_plan(
                box=box,
                clear_box=clear_box,
                erase_box=erase_box,
                frame_box=frame_box,
                ph=ph,
                ratio=ratio,
                canvas_size=canvas.size,
            )
        draw_plans.append((str(fig_id), label, str(ph.get("aspect") or ""), clear_box, box))
        if scale != 1:
            new_placements[str(fig_id)] = [int(round(v / scale)) for v in box]
            clear_boxes[str(fig_id)] = [int(round(v / scale)) for v in clear_box]
            if not redraw:
                erase_boxes[str(fig_id)] = [int(round(v / scale)) for v in erase_box]
                frame_boxes[str(fig_id)] = [int(round(v / scale)) for v in frame_box]
        else:
            new_placements[str(fig_id)] = list(box)
            clear_boxes[str(fig_id)] = list(clear_box)
            if not redraw:
                erase_boxes[str(fig_id)] = list(erase_box)
                frame_boxes[str(fig_id)] = list(frame_box)

    if redraw:
        for fig_id, label, aspect, clear_box, box in draw_plans:
            _erase_placeholder_region(canvas, clear_box)
            _draw_clean_placeholder(canvas, box, fig_id, label, aspect)

    updated["placements"] = new_placements or placements
    if clear_boxes:
        updated["_replacement_clear_boxes"] = clear_boxes
    if erase_boxes:
        updated["_replacement_erase_boxes"] = erase_boxes
    if frame_boxes:
        updated["_replacement_frame_boxes"] = frame_boxes
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
        effective_tolerance = _effective_placeholder_ratio_tolerance(expected, ratio_tolerance)
        if rel_error > effective_tolerance:
            issues.append(
                {
                    "id": str(fig_id),
                    "expected_ratio": round(expected, 4),
                    "actual_ratio": round(actual, 4),
                    "relative_error": round(rel_error, 4),
                    "ratio_tolerance": round(effective_tolerance, 4),
                    "detected_box": list(detected_box),
                    "visible_box": list(visible_box),
                    "message": (
                        f"{fig_id} visible placeholder ratio {actual:.2f}:1 does not match "
                        f"declared ratio {expected:.2f}:1 within {effective_tolerance:.0%}"
                    ),
                }
            )
        if _is_result_like_square_placeholder(ph, expected) and _box_too_large_for_result_square(
            visible_box,
            canvas.size,
        ):
            side = min(visible_box[2] - visible_box[0], visible_box[3] - visible_box[1])
            issues.append(
                {
                    "id": str(fig_id),
                    "expected_ratio": round(expected, 4),
                    "actual_ratio": round(actual, 4),
                    "relative_size": round(side / max(1, canvas.width), 4),
                    "size_tolerance": 0.35,
                    "detected_box": list(detected_box),
                    "visible_box": list(visible_box),
                    "message": (
                        f"{fig_id} square result placeholder is too large "
                        f"({side / max(1, canvas.width):.0%} of canvas width); "
                        "strict mode requires a moderate hero figure slot with surrounding breathing room"
                    ),
                }
            )
        if _is_result_like_square_placeholder(ph, expected) and _box_too_low_for_result_square(
            visible_box,
            canvas.size,
        ):
            bottom_limit = _RESULT_SQUARE_MAX_BOTTOM_FRACTION
            issues.append(
                {
                    "id": str(fig_id),
                    "expected_ratio": round(expected, 4),
                    "actual_ratio": round(actual, 4),
                    "bottom_fraction": round(visible_box[3] / max(1, canvas.height), 4),
                    "bottom_limit": bottom_limit,
                    "detected_box": list(detected_box),
                    "visible_box": list(visible_box),
                    "message": (
                        f"{fig_id} square result placeholder sits too low "
                        f"(bottom at {visible_box[3] / max(1, canvas.height):.0%} of canvas height); "
                        "strict mode requires a clear gutter above the bottom summary/conclusion modules"
                    ),
                }
            )
        surface_metrics = _figure_surface_dark_metrics(canvas, visible_box, seed_box=detected_box)
        if surface_metrics and surface_metrics["dark_fraction"] > 0.55 and surface_metrics["light_fraction"] < 0.30:
            issues.append(
                {
                    "id": str(fig_id),
                    "category": "figure_surface",
                    "dark_fraction": round(surface_metrics["dark_fraction"], 4),
                    "light_fraction": round(surface_metrics["light_fraction"], 4),
                    "detected_box": list(detected_box),
                    "visible_box": list(visible_box),
                    "sample_box": list(surface_metrics["sample_box"]),
                    "message": (
                        f"{fig_id} figure-card surface around the placeholder is too dark "
                        f"({surface_metrics['dark_fraction']:.0%} dark pixels in the immediate mat/card band); "
                        "strict mode requires plot/diagram placeholders to sit on light paper-like cards, "
                        "with dark colors used only as outer accents"
                    ),
                }
            )
    return issues


def _is_result_like_square_placeholder(ph: dict[str, Any], ratio: float) -> bool:
    if not (0.90 <= ratio <= 1.10):
        return False
    low = str(ph.get("label") or "").lower()
    return any(word in low for word in ("limit", "result", "constraint", "upper", "exclusion"))


def _box_too_large_for_result_square(
    box: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
    *,
    max_canvas_width_fraction: float = 0.35,
) -> bool:
    side = min(max(1, box[2] - box[0]), max(1, box[3] - box[1]))
    return side > canvas_size[0] * max_canvas_width_fraction


_RESULT_SQUARE_MAX_BOTTOM_FRACTION = 0.89


def _box_too_low_for_result_square(
    box: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
    *,
    max_bottom_fraction: float = _RESULT_SQUARE_MAX_BOTTOM_FRACTION,
) -> bool:
    return box[3] > canvas_size[1] * max_bottom_fraction


def _figure_surface_dark_metrics(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    seed_box: tuple[int, int, int, int] | None = None,
) -> dict[str, Any] | None:
    """Estimate whether the immediate figure-card band is dark.

    A template can satisfy the blank-placeholder rule while placing that slot in
    a dramatic dark hero block.  Since real CMS/HEP plots commonly have white
    backgrounds, final replacement then looks like a pasted sticker.  We sample
    a ring outside the detected placeholder; good figure-card designs keep this
    ring light and paper-like even when the outer poster background is dark.
    """
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    if w < 16 or h < 16:
        return None
    if seed_box and _box_area(box) > _box_area(seed_box) * 1.55 and _box_light_fraction(canvas, box) >= 0.70:
        # The recovered visible box is already the larger light card/mat around
        # the seed.  Dark artwork outside that complete mat is acceptable.
        return None
    pad = int(round(max(12, min(96, min(w, h) * 0.10))))
    inner_pad = int(round(max(2, min(12, min(w, h) * 0.015))))
    outer = _pad_box(box, pad, canvas.size)
    inner = _pad_box(box, inner_pad, canvas.size)
    if _box_area(outer) <= _box_area(inner):
        return None
    pixels = canvas.load()
    step = max(1, int(round(max(canvas.size) / 1600)))
    dark = 0
    light = 0
    total = 0
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    for y in range(oy0, oy1, step):
        for x in range(ox0, ox1, step):
            if ix0 <= x < ix1 and iy0 <= y < iy1:
                continue
            total += 1
            pixel = pixels[x, y]
            if _is_dark_figure_surface_pixel(pixel):
                dark += 1
            if _is_light_figure_surface_pixel(pixel):
                light += 1
    if total == 0:
        return None
    return {
        "dark_fraction": dark / total,
        "light_fraction": light / total,
        "sample_box": outer,
    }


def _is_dark_figure_surface_pixel(pixel: tuple[int, int, int]) -> bool:
    r, g, b = pixel
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return lum < 135


def _is_light_figure_surface_pixel(pixel: tuple[int, int, int]) -> bool:
    r, g, b = pixel
    mx, mn = max(pixel), min(pixel)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return lum >= 205 and (mx - mn) <= 80


def _box_light_fraction(canvas: Image.Image, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = _pad_box(box, 0, canvas.size)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    pixels = canvas.load()
    step = max(1, int(round(max(canvas.size) / 1600)))
    light = 0
    total = 0
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            total += 1
            if _is_light_figure_surface_pixel(pixels[x, y]):
                light += 1
    return light / total if total else 0.0


def _effective_placeholder_ratio_tolerance(expected_ratio: float, base_tolerance: float) -> float:
    """Aspect-ratio tolerance for generated placeholder QA.

    Wide scientific plots must remain strict because a 2.5:1 placeholder can
    become unreadable when drawn as a ribbon or generic wide card.  Square
    result plots are more forgiving: the final scientific image is still pasted
    as a true square inside the detected panel, and a slightly landscape
    decorative panel is acceptable as long as it stays close.
    """
    base = max(0.0, float(base_tolerance))
    if 0.92 <= expected_ratio <= 1.08:
        return max(base, 0.30)
    if 1.08 < expected_ratio < 1.50:
        return max(base, 0.25)
    if expected_ratio >= 2.0:
        return max(base, 0.30)
    return base


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


def _intersect_box(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _pad_box(box: tuple[int, int, int, int], pad: int, canvas_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    w, h = canvas_size
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _enforce_canvas_gutter(
    box: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Keep replacement planning away from canvas edges.

    LLM placeholder detection sometimes reports a box that starts at x=0 when
    the visible placeholder is actually inset inside a section card.  If we use
    that box directly, pasted paper figures can be clipped at the canvas edge.
    This deterministic inset preserves strict no-fallback behavior while keeping
    final scientific figures inside a safe poster gutter.
    """
    x0, y0, x1, y1 = box
    canvas_w, canvas_h = canvas_size
    margin_x = max(8, min(160, int(round(canvas_w * 0.030))))
    margin_y = max(8, min(180, int(round(canvas_h * 0.015))))
    nx0 = max(x0, margin_x)
    ny0 = max(y0, margin_y)
    nx1 = min(x1, canvas_w - margin_x)
    ny1 = min(y1, canvas_h - margin_y)
    # Do not create a degenerate box.  If a malformed detection is smaller than
    # the safe gutters, keep the original so later validation rejects it.
    if nx1 <= nx0 + 8 or ny1 <= ny0 + 8:
        return box
    return nx0, ny0, nx1, ny1


def _repair_edge_wide_source_box(
    source_box: tuple[int, int, int, int],
    original_box: tuple[int, int, int, int],
    ratio: float,
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Recover vertical extent when a wide edge detection is truncated.

    The LLM sometimes reports a wide placeholder starting at x=0; border
    recovery then misses the true bottom dashed edge and leaves a visible blank
    placeholder strip below the pasted figure.  When a wide figure seed touches
    the canvas edge, keep a conservative vertical envelope around the original
    detection while the later canvas-gutter step moves the left edge inward.
    """
    if ratio < 1.8:
        return source_box
    canvas_w, canvas_h = canvas_size
    edge_margin = max(8, int(round(canvas_w * 0.02)))
    if source_box[0] > edge_margin and original_box[0] > edge_margin:
        return source_box
    ox0, oy0, ox1, oy1 = original_box
    original_h = max(1, oy1 - oy0)
    y_pad = int(round(original_h * 0.22))
    return (
        min(source_box[0], ox0),
        max(0, min(source_box[1], oy0 - y_pad)),
        max(source_box[2], ox1),
        min(canvas_h, max(source_box[3], oy1 + y_pad)),
    )


def _apply_layout_contract_search_constraint(
    source_box: tuple[int, int, int, int],
    original_box: tuple[int, int, int, int],
    search_box: tuple[int, int, int, int] | None,
    ratio: float,
) -> tuple[int, int, int, int]:
    """Use the planned layout search zone to reject obvious over-expansions.

    This is not a fallback placement: the LLM-detected/or pixel-recovered box
    remains the evidence source.  The contract only prevents border recovery
    from expanding a placeholder seed into a whole neighboring card/section when
    the generated art has many similar light/dashed edges.
    """
    if search_box is None:
        return source_box
    if _box_area(source_box) <= 0 or _box_area(search_box) <= 0:
        return source_box
    if _box_area(source_box) <= _box_area(search_box) * 1.12:
        return source_box

    ix0 = max(source_box[0], search_box[0])
    iy0 = max(source_box[1], search_box[1])
    ix1 = min(source_box[2], search_box[2])
    iy1 = min(source_box[3], search_box[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return source_box
    clipped = (ix0, iy0, ix1, iy1)
    # Only clip when the original detection seed still has meaningful support
    # inside the clipped region; otherwise the contract is probably stale for
    # this artistic variant and strict QA should reject later.
    seed_overlap = _box_overlap(clipped, original_box) / max(1, _box_area(original_box))
    if seed_overlap < 0.35:
        return source_box
    source_ratio_error = _ratio_relative_error(_box_ratio(source_box), ratio)
    clipped_ratio_error = _ratio_relative_error(_box_ratio(clipped), ratio)
    if clipped_ratio_error > max(0.45, source_ratio_error + 0.25):
        return source_box
    if _box_area(clipped) < _box_area(original_box) * 0.75:
        return source_box
    return clipped


def _refine_square_result_source_to_dashed_placeholder(
    canvas: Image.Image,
    source_box: tuple[int, int, int, int],
    ph: dict[str, Any],
    ratio: float,
) -> tuple[int, int, int, int]:
    """Recover the original dashed placeholder inside a generated result card.

    Vision detection sometimes returns the whole light result-card surface rather
    than the actual dashed [FIG NN] placeholder drawn by image_generation.  Later
    replacement then treats our self-drawn white publication panel as the
    "placeholder", which is exactly the failure mode the user noticed.  This pass
    re-anchors square result figures to the initial dashed border pixels in the
    generated layout before any final frame or real image is pasted.
    """
    if not _is_result_like_square_placeholder(ph, ratio):
        return source_box
    x0, y0, x1, y1 = source_box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if width < 80 or height < 80:
        return source_box

    row_groups = _placeholder_line_groups(
        canvas,
        source_box,
        axis="row",
        threshold=max(80, int(round(width * 0.16))),
    )
    if len(row_groups) < 2:
        return source_box
    cy = (y0 + y1) / 2
    top_candidates = [item for item in row_groups if item[0][1] < cy - height * 0.05]
    bottom_candidates = [item for item in row_groups if item[0][0] > cy + height * 0.05]
    if not top_candidates or not bottom_candidates:
        return source_box

    # Ignore an outer card edge at the very top when there is an inner dashed
    # placeholder row below it.
    inner_top = [item for item in top_candidates if item[0][0] >= y0 + max(8, height * 0.030)]
    top_group, _top_count = min(inner_top or top_candidates, key=lambda item: item[0][0])
    # Ignore an outer card edge at the very bottom when an inner dashed
    # placeholder row is available.  Among inner candidates, the dashed bottom
    # line usually has the strongest long support; card/arc edges below it tend
    # to be shorter/weaker.
    inner_bottom = [item for item in bottom_candidates if item[0][1] <= y1 - max(8, height * 0.030)]
    bottom_group, _bottom_count = max(inner_bottom or bottom_candidates, key=lambda item: (item[1], item[0][1]))
    top = top_group[0]
    bottom = bottom_group[1] + 1
    if bottom <= top + max(40, height * 0.40):
        return source_box

    row_refined = (x0, top, x1, bottom)
    col_groups = _placeholder_line_groups(
        canvas,
        row_refined,
        axis="col",
        threshold=max(80, int(round((bottom - top) * 0.14))),
    )
    if len(col_groups) < 2:
        return source_box
    cx = (x0 + x1) / 2
    left_candidates = [item for item in col_groups if item[0][1] < cx - width * 0.05]
    right_candidates = [item for item in col_groups if item[0][0] > cx + width * 0.05]
    if not left_candidates or not right_candidates:
        return source_box
    panel = _choose_square_result_column_pair(
        left_candidates,
        right_candidates,
        top=top,
        bottom=bottom,
        source_box=source_box,
        ratio=ratio,
    )
    if panel is None:
        return source_box
    if _box_area(panel) < _box_area(source_box) * 0.45:
        return source_box
    if _box_overlap(panel, source_box) / max(1, _box_area(panel)) < 0.95:
        return source_box
    # The image model may draw a slightly landscape dashed result slot.  That is
    # acceptable as a source placeholder; the real plot is still fitted as a
    # square inside it.  Reject only clearly unrelated long lines.
    if _ratio_relative_error(_box_ratio(panel), ratio) > 0.35:
        return source_box
    # Require a meaningful correction; one-pixel line jitter should not churn.
    edge_delta = max(abs(panel[0] - x0), abs(panel[1] - y0), abs(panel[2] - x1), abs(panel[3] - y1))
    if edge_delta < max(6, min(width, height) * 0.015):
        return source_box
    return panel


def _choose_square_result_column_pair(
    left_candidates: list[tuple[tuple[int, int], int]],
    right_candidates: list[tuple[tuple[int, int], int]],
    *,
    top: int,
    bottom: int,
    source_box: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int] | None:
    x0, _y0, x1, _y1 = source_box
    source_w = max(1, x1 - x0)
    panel_h = max(1, bottom - top)
    edge_tol = max(8, source_w * 0.030)
    best: tuple[float, tuple[int, int, int, int]] | None = None
    has_inner_left = any(item[0][0] >= x0 + edge_tol for item in left_candidates)
    has_inner_right = any(item[0][1] <= x1 - edge_tol for item in right_candidates)
    for left_group, left_count in left_candidates:
        for right_group, right_count in right_candidates:
            left = left_group[0]
            right = right_group[1] + 1
            width = right - left
            if width <= 0:
                continue
            if width < source_w * 0.55:
                continue
            if width * panel_h < _box_area(source_box) * 0.45:
                continue
            rel = _ratio_relative_error(width / panel_h, ratio)
            if rel > 0.45:
                continue
            edge_penalty = 0.0
            if has_inner_left and left <= x0 + edge_tol:
                edge_penalty += 0.03
            if has_inner_right and right >= x1 - edge_tol:
                edge_penalty += 0.03
            # Prefer longer continuous line evidence when geometry is otherwise
            # similar, but don't let an outer card border win purely by strength.
            strength_bonus = min(0.04, (left_count + right_count) / max(1, panel_h * 2) * 0.03)
            score = rel + edge_penalty - strength_bonus
            candidate = (left, top, right, bottom)
            if best is None or score < best[0]:
                best = (score, candidate)
    return best[1] if best else None


def _placeholder_line_groups(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    axis: str,
    threshold: int,
) -> list[tuple[tuple[int, int], int]]:
    x0, y0, x1, y1 = _pad_box(box, 0, canvas.size)
    pixels = canvas.load()
    scores: list[tuple[int, int]] = []
    if axis == "row":
        for y in range(y0, y1):
            count = 0
            for x in range(x0, x1, 2):
                if _is_placeholder_line_pixel(pixels[x, y]):
                    count += 2
            if count >= threshold:
                scores.append((y, count))
    elif axis == "col":
        for x in range(x0, x1):
            count = 0
            for y in range(y0, y1, 2):
                if _is_placeholder_line_pixel(pixels[x, y]):
                    count += 2
            if count >= threshold:
                scores.append((x, count))
    else:
        return []
    groups = _cluster_line_positions([pos for pos, _ in scores], max_gap=2)
    counts = _group_max_counts(groups, scores)
    return [(group, counts.get(group, 0)) for group in groups]


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
    raw_error = _ratio_relative_error(_box_ratio(box), ratio)
    if raw_error <= _effective_placeholder_ratio_tolerance(ratio, 0.20) and _box_has_placeholder_edge_evidence(canvas, box):
        # When the vision model already reports a ratio-correct box with visible
        # placeholder border evidence, trust that seed.  Expanding a good seed to
        # the surrounding light card/section is what made wide post-fit plots
        # cover the bullet rows below and made square result erasers reach into
        # public callouts.
        return None

    dashed_panel = _find_dashed_placeholder_panel(canvas, box, ratio)
    if dashed_panel is not None:
        return dashed_panel

    light_panel = _find_light_placeholder_panel(canvas, box, ratio)
    if light_panel is not None:
        return light_panel

    color_panel = _find_color_consistent_placeholder_panel(canvas, box, ratio)
    if color_panel is not None:
        return color_panel

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
    if 0.90 <= ratio <= 1.10 and area_growth > 1.35:
        if (
            panel[1] < y0 - height * 0.18
            or panel[3] > y1 + height * 0.18
            or panel[0] < x0 - width * 0.25
            or panel[2] > x1 + width * 0.25
        ):
            return None
    if area_growth > 1.15 and panel_error > raw_error + 0.08:
        return None
    return panel


def _find_light_placeholder_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int] | None:
    """Recover a light placeholder card around a square result seed.

    Some generated result placeholders use a light rounded rectangle with a
    dashed border on a dark hero card.  Vision detection can lock onto only the
    central label area, missing the true right edge; border-only detection can
    also fail when the dashed line is broken by text/glow.  This pass detects
    the contiguous light placeholder panel so geometry QA can reject landscape
    or too-low result boxes before replacement.
    """
    if not (0.90 <= ratio <= 1.10):
        return None
    x0, y0, x1, y1 = box
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    cx = (x0 + x1) // 2
    scale_hint = max(canvas.width / 1024.0, canvas.height / 1536.0, 1.0)
    pad_x = int(round(max(140 * scale_hint, width * 0.90)))
    pad_y = int(round(max(90 * scale_hint, height * 0.45)))
    sx0 = max(0, x0 - pad_x)
    sx1 = min(canvas.width, x1 + pad_x)
    sy0 = max(0, y0 - pad_y)
    sy1 = min(canvas.height, y1 + pad_y)
    if sx1 <= sx0 + 12 or sy1 <= sy0 + 12:
        return None

    step = max(2, int(round(scale_hint)))
    gap = int(round(max(18, 10 * scale_hint)))
    min_segment = int(round(max(48 * scale_hint, width * 0.18)))
    rows: list[tuple[int, tuple[int, int], int]] = []
    pixels = canvas.load()
    for y in range(sy0, sy1, step):
        xs: list[int] = []
        for x in range(sx0, sx1, step):
            if _is_light_placeholder_pixel(pixels[x, y]):
                xs.append(x)
        for segment in _cluster_segments(xs, gap=gap, min_length=min_segment):
            overlap = min(segment[1], x1) - max(segment[0], x0)
            if segment[0] <= cx <= segment[1] or overlap >= min_segment * 0.5:
                rows.append((y, segment, segment[1] - segment[0]))
    if not rows:
        return None

    row_gap = max(step * 4, int(round(6 * scale_hint)))
    clusters: list[list[tuple[int, tuple[int, int], int]]] = []
    for row in rows:
        if not clusters or row[0] - clusters[-1][-1][0] > row_gap:
            clusters.append([row])
        else:
            clusters[-1].append(row)

    seed_area = max(1, width * height)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for cluster in clusters:
        top = cluster[0][0]
        bottom = min(canvas.height, cluster[-1][0] + step)
        left = min(item[1][0] for item in cluster)
        right = max(item[1][1] for item in cluster)
        if right <= left or bottom <= top:
            continue
        panel = _pad_box((left, top, right, bottom), 0, canvas.size)
        overlap = _box_overlap(panel, box) / seed_area
        area_ratio = _box_area(panel) / seed_area
        if overlap < 0.30 or area_ratio < 0.35:
            continue
        # Ignore enormous pale section backgrounds; this is only for the figure
        # placeholder card itself.
        if area_ratio > 4.0:
            continue
        center_penalty = abs(((panel[0] + panel[2]) / 2) - cx) / max(1, width)
        score = overlap + 0.15 * min(area_ratio, 2.0) - 0.25 * center_penalty
        candidates.append((score, panel))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _box_has_placeholder_edge_evidence(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
) -> bool:
    x0, y0, x1, y1 = _pad_box(box, 0, canvas.size)
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    if w < 24 or h < 24:
        return False
    pixels = canvas.load()
    tol = max(3, min(18, int(round(min(w, h) * 0.025))))
    min_h = max(24, int(round(w * 0.18)))
    min_v = max(24, int(round(h * 0.18)))

    def row_score(y: int) -> int:
        if y < 0 or y >= canvas.height:
            return 0
        return sum(1 for x in range(x0, x1, 2) if _is_placeholder_line_pixel(pixels[x, y])) * 2

    def col_score(x: int) -> int:
        if x < 0 or x >= canvas.width:
            return 0
        return sum(1 for y in range(y0, y1, 2) if _is_placeholder_line_pixel(pixels[x, y])) * 2

    top_ok = max(row_score(y) for y in range(max(0, y0 - tol), min(canvas.height, y0 + tol + 1))) >= min_h
    bottom_ok = max(row_score(y) for y in range(max(0, y1 - tol), min(canvas.height, y1 + tol + 1))) >= min_h
    left_ok = max(col_score(x) for x in range(max(0, x0 - tol), min(canvas.width, x0 + tol + 1))) >= min_v
    right_ok = max(col_score(x) for x in range(max(0, x1 - tol), min(canvas.width, x1 + tol + 1))) >= min_v
    # This helper is used to decide whether an LLM/vision box is already the
    # actual placeholder.  Require evidence on all sides; an over-tall detection
    # may share the real left/right dashed edges while extending below the
    # placeholder into the next text block.
    return top_ok and bottom_ok and left_ok and right_ok


def _find_dashed_placeholder_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int] | None:
    """Detect neutral/gray dashed placeholder rectangles around a seed box.

    LLM vision can occasionally return the whole surrounding section card
    instead of the actual dashed placeholder, especially for wide result slots
    with text on the left.  The colored-border detector below intentionally
    ignores neutral gray borders, so this pass searches for long dashed
    horizontal/vertical line pairs (gray or colored) that enclose the seed
    center and have a plausible source aspect.  It is conservative: uncertain
    cases return ``None`` so strict QA can still reject the variant.
    """
    x0, y0, x1, y1 = box
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    scale_hint = max(canvas.width / 1024.0, canvas.height / 1536.0, 1.0)
    gap = int(round(max(20, 9 * scale_hint)))
    row_cluster_gap = int(round(max(5, 3 * scale_hint)))
    col_cluster_gap = int(round(max(5, 3 * scale_hint)))
    if 0.90 <= ratio <= 1.10:
        # Vision detections for square result placeholders often lock onto the
        # central label/text region or only the left part of a wide decorative
        # placeholder.  Search farther horizontally so the real dashed border
        # can be recovered and rejected if it is actually landscape/oversized.
        pad_x = int(round(max(140 * scale_hint, width * 0.80)))
        pad_y = int(round(max(90 * scale_hint, height * 0.35)))
    else:
        pad_x = int(round(max(90 * scale_hint, width * 0.20)))
        pad_y = int(round(max(70 * scale_hint, height * 0.20)))
    sx0 = max(0, x0 - pad_x)
    sx1 = min(canvas.width, x1 + pad_x)
    sy0 = max(0, y0 - pad_y)
    sy1 = min(canvas.height, y1 + pad_y)
    if sx1 <= sx0 + 12 or sy1 <= sy0 + 12:
        return None

    min_h_segment = int(round(max(44 * scale_hint, min(width, sx1 - sx0) * 0.12)))
    rows: list[tuple[int, tuple[int, int], int]] = []
    for y in range(sy0, sy1):
        segments = _line_segments_on_row(canvas, y, sx0, sx1, gap=gap, min_length=min_h_segment)
        for segment in segments:
            overlap = min(segment[1], x1) - max(segment[0], x0)
            if segment[0] <= cx <= segment[1] or overlap >= min_h_segment:
                rows.append((y, segment, segment[1] - segment[0]))

    if len(rows) < 2:
        return None

    row_groups: list[list[tuple[int, tuple[int, int], int]]] = []
    for rec in rows:
        if not row_groups or rec[0] - row_groups[-1][-1][0] > row_cluster_gap:
            row_groups.append([rec])
        else:
            row_groups[-1].append(rec)

    horizontal: list[tuple[int, int, tuple[int, int], int]] = []
    for group in row_groups:
        # Prefer segments that actually pass through the seed center; otherwise
        # take the longest nearby segment in the row cluster.
        best = max(
            group,
            key=lambda item: (
                1 if item[1][0] <= cx <= item[1][1] else 0,
                item[2],
                -abs(((item[1][0] + item[1][1]) / 2) - cx),
            ),
        )
        horizontal.append((group[0][0], group[-1][0], best[1], best[2]))

    raw_ratio_error = _ratio_relative_error(width / height, ratio)
    max_ratio_error = max(_effective_placeholder_ratio_tolerance(ratio, 0.20) + 0.12, 0.36)
    seed_area = max(1, width * height)
    pair_candidates: list[tuple[float, tuple[int, int, int, int], tuple[int, int]]] = []

    for i, top_group in enumerate(horizontal):
        if top_group[1] >= cy - 3:
            continue
        for bottom_group in horizontal[i + 1 :]:
            if bottom_group[0] <= cy + 3:
                continue
            top = top_group[0]
            bottom = bottom_group[1] + 1
            panel_h = bottom - top
            if panel_h < max(int(36 * scale_hint), height * 0.18):
                continue
            if panel_h > max(int(760 * scale_hint), height * 1.85):
                continue

            # A real dashed placeholder has top/bottom horizontal evidence that
            # overlaps around the seed center.  This prevents section-card
            # borders or left-column text from being paired with a placeholder
            # bottom edge.
            ix0 = max(top_group[2][0], bottom_group[2][0])
            ix1 = min(top_group[2][1], bottom_group[2][1])
            if ix1 - ix0 < min_h_segment:
                continue
            if not (ix0 - gap <= cx <= ix1 + gap):
                continue
            prelim_ratio = (ix1 - ix0) / max(1, panel_h)
            prelim_ratio_error = _ratio_relative_error(prelim_ratio, ratio)
            if prelim_ratio_error > max_ratio_error + 0.18:
                continue
            prelim_area_ratio = ((ix1 - ix0) * panel_h) / seed_area
            prelim_center_penalty = (
                abs(((ix0 + ix1) / 2) - cx) / max(1, width)
                + abs(((top + bottom) / 2) - cy) / max(1, height)
            )
            safe_square_growth = (
                0.90 <= ratio <= 1.10
                and prelim_ratio_error <= 0.34
                and prelim_center_penalty <= 0.35
                and prelim_area_ratio <= 3.25
            )
            if raw_ratio_error <= 0.20 and prelim_area_ratio > 1.18 and not safe_square_growth:
                continue
            prelim_edge_penalty = (
                abs(top - y0) / max(1, height)
                + 0.25 * abs(bottom - y1) / max(1, height)
            )
            prelim_score = (
                -0.70 * prelim_ratio_error
                -0.30 * prelim_center_penalty
                -0.35 * prelim_edge_penalty
                -0.04 * abs(prelim_area_ratio - 1.0)
            )
            x_window = (max(sx0, ix0 - gap), min(sx1, ix1 + gap))
            pair_candidates.append((prelim_score, (top, bottom, ix0, ix1), x_window))

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    pair_candidates.sort(key=lambda item: item[0], reverse=True)
    for _prelim_score, pair, x_window in pair_candidates[:12]:
        top, bottom, _ix0, _ix1 = pair
        panel_h = bottom - top
        vx0, vx1 = x_window
        min_v_segment = int(round(max(36 * scale_hint, panel_h * 0.28)))
        col_hits: list[int] = []
        for x in range(vx0, vx1):
            if _has_vertical_line_segment(
                canvas,
                x,
                top,
                bottom,
                gap=gap,
                min_length=min_v_segment,
            ):
                col_hits.append(x)
        if not col_hits:
            continue
        col_groups = _cluster_line_positions(col_hits, max_gap=col_cluster_gap)
        left_groups = [g for g in col_groups if g[1] <= cx + gap]
        right_groups = [g for g in col_groups if g[0] >= cx - gap]
        if not left_groups or not right_groups:
            continue

        for left_group in left_groups:
            for right_group in right_groups:
                if right_group[0] <= left_group[1] + min_h_segment:
                    continue
                left = left_group[0]
                right = right_group[1] + 1
                panel = _pad_box((left, top, right, bottom), 0, canvas.size)
                panel_ratio = _box_ratio(panel)
                ratio_error = _ratio_relative_error(panel_ratio, ratio)
                if ratio_error > max_ratio_error:
                    continue
                area_ratio = _box_area(panel) / seed_area
                center_penalty = (
                    abs(((panel[0] + panel[2]) / 2) - cx) / max(1, width)
                    + abs(((panel[1] + panel[3]) / 2) - cy) / max(1, height)
                )
                # When the LLM seed already has roughly the declared
                # source ratio, this detector is meant to *refine inward*
                # to the true dashed box, not expand to a surrounding card.
                if raw_ratio_error <= 0.20 and area_ratio > 1.15:
                    if 0.90 <= ratio <= 1.10 and raw_ratio_error <= 0.08 and 1.35 < area_ratio < 2.00 and ratio_error <= 0.15:
                        # If a square result seed is already ratio-correct,
                        # a much larger near-square dashed/card outline is
                        # usually the surrounding decorative mat rather than a
                        # better replacement target.  Expanding to it creates a
                        # giant low figure and excessive white border.
                        continue
                    safe_square_growth = (
                        0.90 <= ratio <= 1.10
                        and ratio_error <= 0.34
                        and center_penalty <= 0.35
                        and area_ratio <= 3.25
                    )
                    if not safe_square_growth:
                        continue
                seed_edge_penalty = (
                    abs(panel[1] - y0) / max(1, height)
                    + 0.25 * abs(panel[3] - y1) / max(1, height)
                )
                # Ratio is a guide, not the only objective: generated
                # placeholders are allowed some visual looseness, but the
                # replacement panel must stay near the actual dashed region.
                score = (
                    -0.70 * ratio_error
                    -0.30 * center_penalty
                    -0.35 * seed_edge_penalty
                    -0.04 * abs(area_ratio - 1.0)
                )
                candidates.append((score, panel))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0][1]
    if _box_area(best) < seed_area * 0.18:
        return None
    return best


def _line_segments_on_row(
    canvas: Image.Image,
    y: int,
    x0: int,
    x1: int,
    *,
    gap: int,
    min_length: int,
) -> list[tuple[int, int]]:
    pixels = canvas.load()
    xs: list[int] = []
    for x in range(max(0, x0), min(canvas.width, x1), 2):
        if _is_placeholder_line_pixel(pixels[x, y]):
            xs.append(x)
    return _cluster_segments(xs, gap=gap, min_length=min_length)


def _has_vertical_line_segment(
    canvas: Image.Image,
    x: int,
    y0: int,
    y1: int,
    *,
    gap: int,
    min_length: int,
) -> bool:
    pixels = canvas.load()
    ys: list[int] = []
    for y in range(max(0, y0), min(canvas.height, y1), 2):
        if _is_placeholder_line_pixel(pixels[x, y]):
            ys.append(y)
    return bool(_cluster_segments(ys, gap=gap, min_length=min_length))


def _cluster_segments(values: list[int], *, gap: int, min_length: int) -> list[tuple[int, int]]:
    if not values:
        return []
    out: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value - prev <= gap:
            prev = value
        else:
            if prev - start + 1 >= min_length:
                out.append((start, prev + 1))
            start = prev = value
    if prev - start + 1 >= min_length:
        out.append((start, prev + 1))
    return out


def _find_color_consistent_placeholder_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int] | None:
    """Recover the visible placeholder rectangle from its colored border.

    The LLM detector is often directionally correct but can include nearby card
    edges or section gutters in the y direction.  For generated placeholders the
    top and bottom dashed borders have a consistent accent color (purple, teal,
    gold, etc.).  This pass looks for same-color horizontal border pairs that
    enclose the detected center, then expands x to the visible colored line
    extent.  It intentionally returns ``None`` when uncertain; callers then use
    the original LLM box and strict QA can reject the variant if needed.
    """
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if 0.90 <= ratio <= 1.10 and _ratio_relative_error(width / height, ratio) <= 0.20:
        # Square result placeholders are commonly detected as a useful seed.
        # A full-panel expansion can create an oversized pasted figure that
        # intrudes into adjacent blocks; keep the seed and let hidden planning
        # apply only a tiny conservative nudge if needed.
        return None
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    score_pad_x = max(8, int(round(width * 0.04)))
    sx0 = max(0, x0 - score_pad_x)
    sx1 = min(canvas.width, x1 + score_pad_x)
    y_pad = max(120, int(round(height * 0.90)))
    sy0 = max(0, y0 - y_pad)
    sy1 = min(canvas.height, y1 + y_pad)
    if sx1 <= sx0 + 8 or sy1 <= sy0 + 8:
        return None

    pixels = canvas.load()
    row_scores: list[tuple[int, int]] = []
    row_threshold = max(18, int(round((sx1 - sx0) * 0.08)))
    for y in range(sy0, sy1):
        count = 0
        for x in range(sx0, sx1, 2):
            if _is_placeholder_border_pixel(pixels[x, y]):
                count += 2
        if count >= row_threshold:
            row_scores.append((y, count))
    if not row_scores:
        return None

    row_groups = _cluster_line_positions([y for y, _ in row_scores], max_gap=6)
    row_group_max = _group_max_counts(row_groups, row_scores)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for top_group in row_groups:
        if top_group[1] >= cy - 4:
            continue
        for bottom_group in row_groups:
            if bottom_group[0] <= cy + 4:
                continue
            top = top_group[0]
            bottom = bottom_group[1] + 1
            panel_h = bottom - top
            if panel_h < max(28, height * 0.30):
                continue
            # Allow a detected inner text box to expand, but reject enormous
            # spans that clearly include another section.
            if panel_h > max(80, height * 2.4):
                continue

            top_color = _colored_group_mean(canvas, top_group, sx0, sx1)
            bottom_color = _colored_group_mean(canvas, bottom_group, sx0, sx1)
            if top_color is None or bottom_color is None:
                continue
            color_distance = _rgb_distance(top_color, bottom_color)
            if color_distance > 75:
                continue

            accent = tuple((top_color[i] + bottom_color[i]) / 2 for i in range(3))
            top_extent = _colored_line_extent(canvas, top_group, accent, box)
            bottom_extent = _colored_line_extent(canvas, bottom_group, accent, box)
            if top_extent is None or bottom_extent is None:
                continue

            left = min(x0, top_extent[0], bottom_extent[0])
            right = max(x1, top_extent[1], bottom_extent[1])
            if right <= left + 8:
                continue
            panel = _pad_box((left, top, right, bottom), 0, canvas.size)
            panel_ratio = _box_ratio(panel)
            ratio_error = _ratio_relative_error(panel_ratio, ratio)
            if ratio_error > 1.20:
                continue
            raw_error = _ratio_relative_error(_box_ratio(box), ratio)
            area_growth = _box_area(panel) / max(1, _box_area(box))
            if 1.08 <= ratio <= 1.50 and raw_error <= 0.12 and area_growth > 1.40:
                # Near-square process placeholders are often detected very
                # accurately by vision.  Large color-consistent expansions can
                # jump to the surrounding card/module border and overlap the
                # adjacent process placeholder; keep the seed in that case.
                continue
            if 1.08 <= ratio <= 1.50 and raw_error <= 0.12 and 0.85 <= area_growth <= 1.25:
                # If a near-square seed already has the right ratio and the
                # candidate is about the same size, a color match is more likely
                # a neighboring decorative/card line than a useful correction.
                # Keep the seed; still allow clear shrink corrections for
                # over-tall detections (area_growth < 0.85).
                continue

            center_penalty = (
                abs(((panel[1] + panel[3]) / 2) - cy) / max(1, height)
                + abs(((panel[0] + panel[2]) / 2) - cx) / max(1, width)
            )
            near_square_placeholder = 0.90 <= ratio <= 1.10
            safe_square_expansion = (
                near_square_placeholder
                and raw_error <= 0.20
                and ratio_error <= 0.18
                and center_penalty <= 0.30
                and area_growth <= 2.10
            )
            if (
                area_growth > 1.15
                and ratio_error > raw_error + 0.08
                and not safe_square_expansion
            ):
                continue
            support = (
                row_group_max.get(top_group, 0)
                + row_group_max.get(bottom_group, 0)
            ) / max(1, (sx1 - sx0) * 2)
            # Prefer color-consistent border pairs and the nearest enclosing
            # pair.  Ratio matters, but visible placeholder boxes are allowed
            # to be looser than the declared source-image ratio; the hidden
            # placement stage fits the real figure inside afterward.
            score = support - 0.45 * center_penalty - 0.20 * ratio_error - 0.004 * color_distance
            candidates.append((score, panel))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0][1]
    if _box_area(best) < _box_area(box) * 0.35:
        return None
    return best


def _colored_group_mean(
    canvas: Image.Image,
    group: tuple[int, int],
    x0: int,
    x1: int,
) -> tuple[float, float, float] | None:
    pixels = canvas.load()
    red = green = blue = count = 0
    for y in range(max(0, group[0]), min(canvas.height - 1, group[1]) + 1):
        for x in range(max(0, x0), min(canvas.width, x1), 2):
            pixel = pixels[x, y]
            if _is_placeholder_border_pixel(pixel):
                red += pixel[0]
                green += pixel[1]
                blue += pixel[2]
                count += 1
    if count < 8:
        return None
    return red / count, green / count, blue / count


def _colored_line_extent(
    canvas: Image.Image,
    group: tuple[int, int],
    accent: tuple[float, float, float],
    seed_box: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    x0, _y0, x1, _y1 = seed_box
    width = max(1, x1 - x0)
    cx = (x0 + x1) // 2
    search_pad = max(120, int(round(width * 0.35)))
    sx0 = max(0, x0 - search_pad)
    sx1 = min(canvas.width, x1 + search_pad)
    pixels = canvas.load()
    xs: list[int] = []
    for x in range(sx0, sx1):
        for y in range(max(0, group[0]), min(canvas.height - 1, group[1]) + 1):
            pixel = pixels[x, y]
            if _is_placeholder_border_pixel(pixel) and _rgb_distance(pixel, accent) <= 105:
                xs.append(x)
                break
    if not xs:
        return None
    groups = _cluster_line_positions(xs, max_gap=25)
    if not groups:
        return None
    # Pick the same-colored horizontal segment that overlaps the LLM seed most.
    # This expands inner detections to full dashed borders without jumping to
    # unrelated detector-art lines in the margins.
    def key(group: tuple[int, int]) -> tuple[int, int, int]:
        overlap = max(0, min(group[1], x1) - max(group[0], x0))
        contains_center = 1 if group[0] <= cx <= group[1] else 0
        length = group[1] - group[0] + 1
        return overlap, contains_center, length

    best = max(groups, key=key)
    if key(best)[0] < min(width * 0.25, 80):
        return None
    return best[0], best[1] + 1


def _rgb_distance(
    a: tuple[float, float, float] | tuple[int, int, int],
    b: tuple[float, float, float] | tuple[int, int, int],
) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2) ** 0.5


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


def _is_placeholder_line_pixel(pixel: tuple[int, int, int]) -> bool:
    """Pixel predicate for dashed placeholder lines.

    Some image-generation variants draw placeholder borders in neutral gray
    rather than colored accent strokes.  For line-pair detection we therefore
    include both colored border pixels and moderately dark neutral gray pixels.
    Text also satisfies this predicate, but the caller requires long horizontal
    and vertical dashed-line support, which filters ordinary words/bullets out.
    """
    if _is_placeholder_border_pixel(pixel):
        return True
    r, g, b = pixel
    lum = (r + g + b) / 3
    if lum < 185:
        return True
    return False


def _is_light_placeholder_pixel(pixel: tuple[int, int, int]) -> bool:
    r, g, b = pixel
    mx, mn = max(pixel), min(pixel)
    lum = (r + g + b) / 3
    return lum >= 205 and (mx - mn) <= 65


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


def _fit_box_to_ratio_inside(box: tuple[int, int, int, int], ratio: float) -> tuple[int, int, int, int]:
    """Largest ratio-correct box fully contained by ``box``.

    Used for post-generation replacement planning: real scientific figures
    should overlap the generated placeholder as much as possible, but never
    protrude beyond the original placeholder rectangle.
    """
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if ratio <= 0:
        return x0, y0, x1, y1
    if width / height > ratio:
        new_h = height
        new_w = min(width, max(1, int(round(new_h * ratio))))
    else:
        new_w = width
        new_h = min(height, max(1, int(round(new_w / ratio))))
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    nx0 = int(round(cx - new_w / 2))
    ny0 = int(round(cy - new_h / 2))
    nx1 = nx0 + new_w
    ny1 = ny0 + new_h
    # Rounding can push an edge out by one pixel; clamp inward only.
    if nx0 < x0:
        nx1 += x0 - nx0
        nx0 = x0
    if ny0 < y0:
        ny1 += y0 - ny0
        ny0 = y0
    if nx1 > x1:
        nx0 -= nx1 - x1
        nx1 = x1
    if ny1 > y1:
        ny0 -= ny1 - y1
        ny1 = y1
    return max(x0, nx0), max(y0, ny0), min(x1, max(nx0 + 1, nx1)), min(y1, max(ny0 + 1, ny1))


def _nudge_square_result_box(
    box: tuple[int, int, int, int],
    ratio: float,
    label: str,
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Conservatively move square result plots slightly upward.

    This fixes a common visual issue where a square limits/result plot detected
    by the LLM is ratio-correct but sits a little low in the artistic result
    block.  Older versions also shrank and moved the plot left, which produced
    an obvious oversized right/bottom white mat.  Keep the size and x-position
    stable; only make a small upward move when the plot is in the lower part of
    the poster.
    """
    if not (0.90 <= ratio <= 1.10):
        return box
    low = str(label or "").lower()
    if not any(word in low for word in ("limit", "result", "constraint", "upper")):
        return box
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])
    if _ratio_relative_error(width / height, 1.0) > 0.08:
        return box
    bottom_fraction = box[3] / max(1, canvas_size[1])
    if bottom_fraction < 0.74:
        return box
    side = min(width, height)
    shift = int(round(min(70, max(24, side * 0.055))))
    ny0 = max(0, box[1] - shift)
    return box[0], ny0, box[2], ny0 + height


def _repair_square_result_replacement_plan(
    *,
    box: tuple[int, int, int, int],
    clear_box: tuple[int, int, int, int],
    erase_box: tuple[int, int, int, int],
    frame_box: tuple[int, int, int, int],
    ph: dict[str, Any],
    ratio: float,
    canvas_size: tuple[int, int],
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
]:
    """Keep square result replacement plans visually contained.

    This is the deterministic single-figure repair pass used after hidden
    normalization.  It does not invent a new layout; it only makes the final
    target/frame/erase rectangles self-consistent with the detected placeholder.
    The visible white frame/erase envelope stays exactly inside the placeholder,
    while the real scientific plot is inset slightly.  This is deliberately
    different from "erasing a protruding white border": the real plot itself is
    made smaller so the poster keeps a clean margin inside the placeholder.
    """
    if not _is_result_like_square_placeholder(ph, ratio):
        return box, clear_box, erase_box, frame_box

    if not _is_contained(box, clear_box, margin=0):
        box = _fit_box_to_ratio_inside(clear_box, ratio)

    box = _inset_square_result_target_box(clear_box, ratio)

    # For square result cards, *any* visible white envelope outside the original
    # placeholder reads as a protruding block.  Clip every edge to the detected
    # placeholder; do not keep the earlier side/top over-erase.
    frame_box = _square_result_frame_around_target(box, clear_box)
    erase_box = _intersect_box(erase_box, clear_box) or clear_box
    return box, clear_box, erase_box, frame_box


def _needs_supporting_plot_text_clearance(ph: dict[str, Any], ratio: float) -> bool:
    if ratio < 1.9:
        return False
    role = str(ph.get("role") or "").lower()
    label = str(ph.get("label") or "").lower()
    target = " ".join([role, label])
    return any(
        word in target
        for word in (
            "fit",
            "post-fit",
            "distribution",
            "template",
            "validation",
            "control region",
            "signal region",
        )
    )


def _busy_content_below_placeholder(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
) -> bool:
    x0, _y0, x1, y1 = _pad_box(box, 0, canvas.size)
    width = max(1, x1 - x0)
    height = max(1, box[3] - box[1])
    pad_x = int(round(width * 0.04))
    sx0 = max(0, x0 + pad_x)
    sx1 = min(canvas.width, x1 - pad_x)
    sy0 = min(canvas.height, y1)
    sy1 = min(canvas.height, y1 + max(24, min(180, int(round(height * 0.24)))))
    if sx1 <= sx0 or sy1 <= sy0:
        return False
    pixels = canvas.load()
    step = max(1, int(round(max(canvas.size) / 1800)))
    busy = 0
    total = 0
    for y in range(sy0, sy1, step):
        for x in range(sx0, sx1, step):
            total += 1
            if _is_busy_pixel(pixels[x, y]):
                busy += 1
    return bool(total and busy / total >= 0.025)


def _inset_supporting_wide_target_box(
    source_box: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = source_box
    h = max(1, y1 - y0)
    top_inset = max(6, min(28, int(round(h * 0.025))))
    bottom_reserve = max(24, min(120, int(round(h * 0.120))))
    inner = (x0, y0 + top_inset, x1, max(y0 + top_inset + 1, y1 - bottom_reserve))
    return _fit_box_to_ratio_inside(inner, ratio)


def _supporting_wide_frame_around_target(
    box: tuple[int, int, int, int],
    source_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = source_box
    h = max(1, y1 - y0)
    bottom_pad = max(4, min(18, int(round(h * 0.020))))
    return x0, y0, x1, min(y1, box[3] + bottom_pad)


def _square_result_frame_around_target(
    box: tuple[int, int, int, int],
    clear_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    side = min(max(1, box[2] - box[0]), max(1, box[3] - box[1]))
    pad = max(10, min(42, int(round(side * 0.040))))
    return _intersect_box(_pad_box(box, pad, (10**9, 10**9)), clear_box) or box


def _inset_square_result_target_box(
    clear_box: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    """Return a slightly smaller real-figure target inside a result placeholder."""
    x0, y0, x1, y1 = clear_box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    side = min(w, h)
    inset = max(2, min(96, int(round(side * _RESULT_SQUARE_TARGET_INSET_FRACTION))))
    if w <= inset * 2 + 12 or h <= inset * 2 + 12:
        return _fit_box_to_ratio_inside(clear_box, ratio)
    inner = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
    return _fit_box_to_ratio_inside(inner, ratio)


_RESULT_SQUARE_TARGET_INSET_FRACTION = 0.085
_RESULT_SQUARE_MIN_INSET_FRACTION = 0.070


def _square_result_downward_erase_tolerance(
    box: tuple[int, int, int, int],
    canvas_size: tuple[int, int] | None = None,
) -> int:
    """Small tolerance for anti-aliased bottom strokes, not visible protrusions."""
    side = min(max(1, box[2] - box[0]), max(1, box[3] - box[1]))
    canvas_hint = min(canvas_size or (side, side))
    return max(3, min(24, int(round(min(side * 0.015, canvas_hint * 0.004)))))


def _box_exceeds_outer(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
    *,
    tolerance: int = 0,
) -> bool:
    return (
        inner[0] < outer[0] - tolerance
        or inner[1] < outer[1] - tolerance
        or inner[2] > outer[2] + tolerance
        or inner[3] > outer[3] + tolerance
    )


def _square_result_inner_margin_issue(
    fig_id: str,
    box: tuple[int, int, int, int],
    clear: tuple[int, int, int, int],
) -> dict[str, Any] | None:
    side = min(max(1, clear[2] - clear[0]), max(1, clear[3] - clear[1]))
    required = int(round(side * _RESULT_SQUARE_MIN_INSET_FRACTION))
    insets = {
        "left": box[0] - clear[0],
        "top": box[1] - clear[1],
        "right": clear[2] - box[2],
        "bottom": clear[3] - box[3],
    }
    min_inset = min(insets.values())
    if min_inset >= required:
        return None
    return {
        "severity": "warning",
        "category": "figure_inner_margin",
        "id": fig_id,
        "message": (
            f"{fig_id} real square-result figure is too close to the placeholder edge "
            f"(minimum inset {min_inset}px; required at least {required}px)"
        ),
        "location": f"spec.placements[{fig_id}]",
        "insets": insets,
        "required_inset": required,
        "suggested_fix": (
            "Shrink this square result figure target inside its placeholder so the "
            "final plot has a visible safety margin on all four sides."
        ),
    }


def _square_result_frame_size_issue(
    fig_id: str,
    box: tuple[int, int, int, int],
    frame: tuple[int, int, int, int],
    clear: tuple[int, int, int, int],
) -> dict[str, Any] | None:
    target_area = max(1, _box_area(box))
    frame_area = max(1, _box_area(frame))
    area_ratio = frame_area / target_area
    clear_area_ratio = frame_area / max(1, _box_area(clear))
    max_ratio = 1.28
    max_clear_fraction = 0.90
    if area_ratio <= max_ratio and clear_area_ratio <= max_clear_fraction:
        return None
    return {
        "severity": "warning",
        "category": "figure_frame_oversized",
        "id": fig_id,
        "message": (
            f"{fig_id} visible white figure board is too large relative to the "
            f"real plot target (frame/target area ratio {area_ratio:.2f})"
        ),
        "location": f"spec._replacement_frame_boxes[{fig_id}]",
        "frame_area_ratio": round(area_ratio, 3),
        "frame_clear_fraction": round(clear_area_ratio, 3),
        "suggested_fix": (
            "Keep the broad placeholder erase separate from the visible frame; "
            "draw the white publication board tightly around the real plot."
        ),
    }


def _constrain_lower_hero_placeholder_box(
    box: tuple[int, int, int, int],
    ratio: float,
    ph: dict[str, Any],
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Keep lower hero comparison/result figures out of conclusion strips.

    Image models often place the global-comparison/result placeholder in a lower
    Section 5 card.  Vision detection can include the lower card/conclusion
    boundary, yielding a replacement frame that visually protrudes into the
    conclusion strip even though its nominal box is ratio-correct.  For hero
    result/comparison figures in the lower poster, clip the planning envelope
    above the lower conclusion zone and let ratio-preserving fit shrink the
    scientific figure instead of allowing overlap.
    """
    if not _is_lower_hero_placeholder(ph, ratio, box, canvas_size):
        return box
    canvas_h = max(1, canvas_size[1])
    bottom_limit = int(round(canvas_h * 0.890))
    height = max(1, box[3] - box[1])
    top_inset = int(round(min(120, max(64, height * 0.07))))
    target_top = box[1] + top_inset
    target_bottom = min(box[3], bottom_limit)
    if target_bottom <= target_top + 1:
        target_bottom = min(canvas_h, target_top + 1)
    if box[3] <= bottom_limit and target_top <= box[1] + 1:
        return box
    clipped = (box[0], target_top, box[2], target_bottom)
    # If clipping would leave an unusably thin figure, shift the same box upward
    # instead of crushing the plot.
    min_h = max(120, int(round((box[2] - box[0]) / max(ratio, 0.1) * 0.55)))
    if clipped[3] - clipped[1] >= min_h:
        return clipped
    shift = box[3] - bottom_limit
    return _clamp_box((box[0], box[1] - shift, box[2], box[3] - shift), canvas_size)


def _is_lower_hero_placeholder(
    ph: dict[str, Any],
    ratio: float,
    box: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> bool:
    if not (1.20 <= ratio <= 2.20):
        return False
    label = str(ph.get("label") or "").lower()
    group = str(ph.get("group") or "").lower()
    if not (
        "hero" in group
        or "result" in group
        or any(word in label for word in ("comparison", "result", "mass measurement", "electroweak"))
    ):
        return False
    return box[1] >= max(1, canvas_size[1]) * 0.66


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


def _trim_uniform_border(im: Image.Image, *, tolerance: int = 18, margin: int = 3) -> Image.Image:
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


def audit_figure_containment(
    *,
    spec: dict[str, Any],
    scale: float = 1.0,
    max_overlap_ratio: float = 0.05,
) -> list[dict[str, Any]]:
    """Verify that every figure placement is strictly inside its placeholder.

    Returns a list of issues found:
    - containment violations: figure box extends beyond placeholder box
    - overlap violations: two figure boxes overlap beyond the threshold
    """
    issues: list[dict[str, Any]] = []
    placements = spec.get("placements") or {}
    clear_boxes = spec.get("_replacement_clear_boxes") or {}
    erase_boxes = spec.get("_replacement_erase_boxes") or {}
    frame_boxes = spec.get("_replacement_frame_boxes") or {}
    placeholders = {str(p.get("id")): p for p in spec.get("placeholders", [])}

    fig_boxes: dict[str, tuple[int, int, int, int]] = {}
    for fig_id, raw_box in placements.items():
        try:
            box = tuple(int(round(float(v) * scale)) for v in raw_box)
        except Exception:
            continue
        fig_boxes[str(fig_id)] = box

    for fig_id, box in fig_boxes.items():
        ph = placeholders.get(fig_id, {})
        # Use the clear box (the detected placeholder boundary) as the
        # containment envelope.  If no clear box exists, skip the check.
        raw_clear = clear_boxes.get(fig_id)
        if raw_clear:
            try:
                clear = tuple(int(round(float(v) * scale)) for v in raw_clear)
            except Exception:
                continue
            if not _is_contained(box, clear, margin=0):
                issues.append({
                    "severity": "warning",
                    "category": "figure_containment",
                    "message": (
                        f"{fig_id} figure box {list(box)} extends beyond "
                        f"placeholder boundary {list(clear)}"
                    ),
                    "location": f"Section {ph.get('section', '?')} / {fig_id}",
                    "suggested_fix": (
                        "Ensure the figure is pasted strictly inside the "
                        "detected placeholder boundary."
                    ),
                })
            expected = _parse_placeholder_aspect(str(ph.get("aspect") or "")) or _box_ratio(box)
            if _is_result_like_square_placeholder(ph, expected):
                margin_issue = _square_result_inner_margin_issue(fig_id, box, clear)
                if margin_issue:
                    issues.append(margin_issue)
                tolerance = _square_result_downward_erase_tolerance(clear)
                raw_erase = erase_boxes.get(fig_id)
                if raw_erase:
                    try:
                        erase = tuple(int(round(float(v) * scale)) for v in raw_erase)
                    except Exception:
                        erase = None
                    if erase and _box_exceeds_outer(erase, clear, tolerance=tolerance):
                        issues.append({
                            "severity": "warning",
                            "category": "figure_visual_envelope",
                            "id": fig_id,
                            "message": (
                                f"{fig_id} erase box {list(erase)} protrudes outside "
                                f"the detected square-result placeholder {list(clear)}"
                            ),
                            "location": f"spec._replacement_erase_boxes[{fig_id}]",
                            "suggested_fix": (
                                "Clip the visible erase envelope to the placeholder and "
                                "repair only this figure's replacement geometry."
                            ),
                        })
                raw_frame = frame_boxes.get(fig_id)
                if raw_frame:
                    try:
                        frame = tuple(int(round(float(v) * scale)) for v in raw_frame)
                    except Exception:
                        frame = None
                    if frame and _box_exceeds_outer(frame, clear, tolerance=tolerance):
                        issues.append({
                            "severity": "warning",
                            "category": "figure_visual_envelope",
                            "id": fig_id,
                            "message": (
                                f"{fig_id} frame box {list(frame)} protrudes outside "
                                f"the detected square-result placeholder {list(clear)}"
                            ),
                            "location": f"spec._replacement_frame_boxes[{fig_id}]",
                            "suggested_fix": (
                                "Keep the final square-result publication frame inside "
                                "the placeholder envelope."
                            ),
                        })
                    if frame:
                        frame_issue = _square_result_frame_size_issue(fig_id, box, frame, clear)
                        if frame_issue:
                            issues.append(frame_issue)

    # Check for figure-figure overlaps
    fig_ids = list(fig_boxes.keys())
    for i in range(len(fig_ids)):
        for j in range(i + 1, len(fig_ids)):
            id_i, id_j = fig_ids[i], fig_ids[j]
            box_i, box_j = fig_boxes[id_i], fig_boxes[id_j]
            overlap = _box_overlap(box_i, box_j)
            if overlap <= 0:
                continue
            area_i = _box_area(box_i)
            area_j = _box_area(box_j)
            min_area = max(1, min(area_i, area_j))
            ratio = overlap / min_area
            if ratio > max_overlap_ratio:
                issues.append({
                    "severity": "warning",
                    "category": "figure_overlap",
                    "message": (
                        f"{id_i} and {id_j} overlap by {overlap} pixels "
                        f"({ratio:.1%} of the smaller figure)"
                    ),
                    "location": f"{id_i} ↔ {id_j}",
                    "suggested_fix": (
                        "Adjust placeholder placements to add a clear gutter "
                        "between figures, or reduce figure sizes."
                    ),
                })

    return issues


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
