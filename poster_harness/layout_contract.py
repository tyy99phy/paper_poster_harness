from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence


NormBox = tuple[float, float, float, float]
PixelBox = tuple[int, int, int, int]


DEFAULT_CANVAS = {"width": 1024, "height": 1536, "coordinate_system": "normalized_xyxy"}


def build_layout_contract(
    spec: Mapping[str, Any],
    *,
    canvas_width: int = 1024,
    canvas_height: int = 1536,
) -> dict[str, Any]:
    """Build a deterministic Paper2Poster-style layout prior.

    The image model still owns the final artistic rendering.  This contract is a
    structured, normalized prior that says where each scientific placeholder is
    expected to live and which source aspect it must preserve.  Downstream
    detection/replacement stages can use it as a sanity check instead of
    trusting a single vision-detected pixel box unconditionally.
    """

    sections = _sections_in_reading_order(spec)
    section_zones = _section_zones(sections)
    placeholders = [dict(item) for item in spec.get("placeholders") or [] if isinstance(item, Mapping)]

    by_section: dict[int, list[dict[str, Any]]] = {}
    for item in placeholders:
        try:
            sid = int(item.get("section") or item.get("target_section") or 1)
        except Exception:
            sid = 1
        by_section.setdefault(sid, []).append(item)

    planned_placeholders: list[dict[str, Any]] = []
    for sid, figs in by_section.items():
        section_zone = section_zones.get(sid) or _fallback_section_zone(sid)
        slots = _slots_for_section(section_zone, figs)
        for fig, slot in zip(figs, slots):
            ratio = _parse_aspect_ratio(str(fig.get("aspect") or "")) or 1.0
            search_zone = _intersection_norm(
                _pad_norm(slot, 0.045, 0.035),
                _pad_norm(section_zone, 0.025, 0.018),
            )
            planned_placeholders.append(
                {
                    "id": str(fig.get("id") or ""),
                    "section": sid,
                    "role": str(fig.get("role") or ""),
                    "label": str(fig.get("label") or fig.get("asset") or fig.get("id") or ""),
                    "aspect": str(fig.get("aspect") or ""),
                    "expected_aspect": round(ratio, 4),
                    "zone": _round_box(slot),
                    "search_zone": _round_box(search_zone),
                    "placement_policy": "detect_near_planned_zone_then_fit_source_aspect_inside_placeholder",
                }
            )

    return {
        "version": 1,
        "source": "deterministic_layout_contract",
        "canvas": {
            "width": int(canvas_width),
            "height": int(canvas_height),
            "coordinate_system": "normalized_xyxy",
        },
        "policy": {
            "purpose": "soft visual prior for image-generation plus strict sanity check for placeholder detection",
            "planned_zone_tolerance": "generated placeholders may move modestly for artistic balance, but must remain in the same section and overlap their planned/search zone",
            "replacement_rule": "real source figures are fitted inside detected placeholders; planned zones are never used as a fallback when detection fails",
            "min_iou": 0.06,
            "max_center_distance_fraction": 0.28,
        },
        "sections": [
            {"id": int(section.get("id")), "title": str(section.get("title") or ""), "zone": _round_box(section_zones[int(section.get("id"))])}
            for section in sections
            if section.get("id") in section_zones
        ],
        "placeholders": planned_placeholders,
    }


def contract_boxes_for_image(
    contract: Mapping[str, Any] | None,
    image_size: Mapping[str, Any] | Sequence[int] | tuple[int, int],
    *,
    key: str = "zone",
) -> dict[str, list[int]]:
    """Convert normalized contract boxes into pixel boxes for an image."""

    if not contract:
        return {}
    width, height = _image_size_tuple(image_size, contract)
    out: dict[str, list[int]] = {}
    for item in contract.get("placeholders") or []:
        if not isinstance(item, Mapping):
            continue
        fig_id = str(item.get("id") or "")
        raw_box = item.get(key) or item.get("zone")
        if not fig_id or not isinstance(raw_box, Sequence) or len(raw_box) < 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in raw_box[:4]]
        out[fig_id] = [
            max(0, min(width, int(round(x0 * width)))),
            max(0, min(height, int(round(y0 * height)))),
            max(0, min(width, int(round(x1 * width)))),
            max(0, min(height, int(round(y1 * height)))),
        ]
    return out


def evaluate_layout_contract_alignment(
    contract: Mapping[str, Any] | None,
    placements: Mapping[str, Any],
    image_size: Mapping[str, Any] | Sequence[int] | tuple[int, int],
    *,
    min_iou: float | None = None,
    max_center_distance_fraction: float | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic issues when detections drift far from the prior."""

    if not contract or not placements:
        return []
    policy = contract.get("policy") if isinstance(contract.get("policy"), Mapping) else {}
    min_iou = float(min_iou if min_iou is not None else policy.get("min_iou", 0.06))
    max_center_distance_fraction = float(
        max_center_distance_fraction
        if max_center_distance_fraction is not None
        else policy.get("max_center_distance_fraction", 0.28)
    )
    planned = contract_boxes_for_image(contract, image_size, key="zone")
    search = contract_boxes_for_image(contract, image_size, key="search_zone")
    width, height = _image_size_tuple(image_size, contract)
    diag = max(1.0, (width * width + height * height) ** 0.5)
    issues: list[dict[str, Any]] = []
    for fig_id, raw in placements.items():
        detected = _coerce_pixel_box(raw, width=width, height=height)
        plan = planned.get(str(fig_id))
        search_box = search.get(str(fig_id))
        if not detected or not plan:
            continue
        detected_t = tuple(detected)  # type: ignore[arg-type]
        plan_t = tuple(plan)  # type: ignore[arg-type]
        search_t = tuple(search_box) if search_box else plan_t  # type: ignore[arg-type]
        iou = _iou(detected_t, plan_t)
        search_iou = _iou(detected_t, search_t)
        dist = _center_distance(detected_t, plan_t) / diag
        if iou < min_iou and search_iou < min_iou and dist > max_center_distance_fraction:
            issues.append(
                {
                    "severity": "critical",
                    "category": "layout_contract_alignment",
                    "id": str(fig_id),
                    "detected_box": list(detected_t),
                    "planned_box": list(plan_t),
                    "search_box": list(search_t),
                    "iou": round(iou, 4),
                    "search_iou": round(search_iou, 4),
                    "center_distance_fraction": round(dist, 4),
                    "message": (
                        f"{fig_id} detected placeholder is far from its planned layout-contract zone; "
                        "reject this generated variant rather than using the plan as a fallback."
                    ),
                }
            )
    return issues


def attach_layout_contract_boxes(
    spec: Mapping[str, Any],
    image_size: Mapping[str, Any] | Sequence[int] | tuple[int, int],
) -> dict[str, Any]:
    """Attach pixel-space planned/search boxes to a spec for QA/debugging."""

    updated = copy.deepcopy(dict(spec))
    contract = updated.get("layout_contract") if isinstance(updated.get("layout_contract"), Mapping) else None
    if not contract:
        return updated
    updated["_layout_contract_boxes"] = contract_boxes_for_image(contract, image_size, key="zone")
    updated["_layout_contract_search_boxes"] = contract_boxes_for_image(contract, image_size, key="search_zone")
    return updated


def _sections_in_reading_order(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    sections_by_id: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(spec.get("sections") or [], start=1):
        if not isinstance(item, Mapping):
            continue
        try:
            sid = int(item.get("id") or idx)
        except Exception:
            sid = idx
        row = dict(item)
        row["id"] = sid
        sections_by_id[sid] = row
    reading_order: list[int] = []
    storyboard = spec.get("storyboard") if isinstance(spec.get("storyboard"), Mapping) else {}
    layout_tree = storyboard.get("layout_tree") if isinstance(storyboard.get("layout_tree"), Mapping) else {}
    for value in layout_tree.get("reading_order") or []:
        try:
            sid = int(value)
        except Exception:
            continue
        if sid in sections_by_id and sid not in reading_order:
            reading_order.append(sid)
    for sid in sorted(sections_by_id):
        if sid not in reading_order:
            reading_order.append(sid)
    return [sections_by_id[sid] for sid in reading_order]


def _section_zones(sections: Sequence[Mapping[str, Any]]) -> dict[int, NormBox]:
    base_zones: list[NormBox] = [
        (0.035, 0.170, 0.485, 0.500),
        (0.515, 0.170, 0.965, 0.500),
        (0.035, 0.515, 0.485, 0.755),
        (0.515, 0.515, 0.965, 0.755),
        (0.035, 0.770, 0.965, 0.925),
        (0.035, 0.930, 0.965, 0.985),
    ]
    out: dict[int, NormBox] = {}
    for order, section in enumerate(sections):
        try:
            sid = int(section.get("id"))
        except Exception:
            continue
        if order < len(base_zones):
            out[sid] = base_zones[order]
        else:
            out[sid] = _fallback_section_zone(sid)
    return out


def _fallback_section_zone(section_id: int) -> NormBox:
    idx = max(0, int(section_id) - 1)
    row = idx // 2
    col = idx % 2
    y0 = 0.17 + row * 0.17
    y1 = min(0.95, y0 + 0.145)
    x0 = 0.035 if col == 0 else 0.515
    x1 = 0.485 if col == 0 else 0.965
    return x0, y0, x1, y1


def _slots_for_section(section_zone: NormBox, figs: Sequence[Mapping[str, Any]]) -> list[NormBox]:
    if not figs:
        return []
    n = len(figs)
    if n == 1:
        return [_single_slot(section_zone, figs[0])]
    if n == 2:
        x0, y0, x1, y1 = section_zone
        w = x1 - x0
        h = y1 - y0
        lower = (x0 + 0.035 * w, y0 + 0.44 * h, x1 - 0.035 * w, y1 - 0.08 * h)
        lx0, ly0, lx1, ly1 = lower
        gap = 0.035 * w
        mid = (lx0 + lx1) / 2
        containers = [(lx0, ly0, mid - gap / 2, ly1), (mid + gap / 2, ly0, lx1, ly1)]
        return [_fit_norm_box(container, _parse_aspect_ratio(str(fig.get("aspect") or "")) or 1.0, fill=0.92) for container, fig in zip(containers, figs)]

    x0, y0, x1, y1 = section_zone
    w = x1 - x0
    h = y1 - y0
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols
    grid = (x0 + 0.04 * w, y0 + 0.36 * h, x1 - 0.04 * w, y1 - 0.06 * h)
    gx0, gy0, gx1, gy1 = grid
    gap_x = 0.025 * w
    gap_y = 0.025 * h
    cell_w = (gx1 - gx0 - gap_x * (cols - 1)) / cols
    cell_h = (gy1 - gy0 - gap_y * (rows - 1)) / rows
    out: list[NormBox] = []
    for idx, fig in enumerate(figs):
        r = idx // cols
        c = idx % cols
        cell = (
            gx0 + c * (cell_w + gap_x),
            gy0 + r * (cell_h + gap_y),
            gx0 + c * (cell_w + gap_x) + cell_w,
            gy0 + r * (cell_h + gap_y) + cell_h,
        )
        out.append(_fit_norm_box(cell, _parse_aspect_ratio(str(fig.get("aspect") or "")) or 1.0, fill=0.90))
    return out


def _single_slot(section_zone: NormBox, fig: Mapping[str, Any]) -> NormBox:
    x0, y0, x1, y1 = section_zone
    w = x1 - x0
    h = y1 - y0
    ratio = _parse_aspect_ratio(str(fig.get("aspect") or "")) or 1.0
    label = str(fig.get("label") or "").lower()
    role = str(fig.get("role") or "").lower()
    hero = "hero" in role or any(word in label for word in ("limit", "result", "constraint", "measurement", "exclusion"))
    if hero and 0.75 <= ratio <= 1.35 and w > h:
        container = (x0 + 0.46 * w, y0 + 0.18 * h, x1 - 0.045 * w, y1 - 0.08 * h)
        return _fit_norm_box(container, ratio, fill=0.96)
    if ratio >= 2.0:
        container = (x0 + 0.04 * w, y0 + 0.50 * h, x1 - 0.04 * w, y1 - 0.08 * h)
        return _fit_norm_box(container, ratio, fill=0.98)
    container = (x0 + 0.08 * w, y0 + 0.40 * h, x1 - 0.08 * w, y1 - 0.08 * h)
    return _fit_norm_box(container, ratio, fill=0.94)


def _fit_norm_box(container: NormBox, ratio: float, *, fill: float = 0.94) -> NormBox:
    x0, y0, x1, y1 = container
    cw = max(0.001, x1 - x0)
    ch = max(0.001, y1 - y0)
    width = cw * fill
    height = width / max(0.05, ratio)
    if height > ch * fill:
        height = ch * fill
        width = height * max(0.05, ratio)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return _clamp_norm((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))


def _parse_aspect_ratio(aspect: str) -> float | None:
    text = str(aspect or "").strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[:/]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        den = float(match.group(2))
        return float(match.group(1)) / den if den else None
    if "square" in text:
        return 1.0
    return None


def _image_size_tuple(
    image_size: Mapping[str, Any] | Sequence[int] | tuple[int, int],
    contract: Mapping[str, Any] | None = None,
) -> tuple[int, int]:
    if isinstance(image_size, Mapping):
        width = int(image_size.get("width") or image_size.get("w") or 0)
        height = int(image_size.get("height") or image_size.get("h") or 0)
    elif isinstance(image_size, Sequence) and len(image_size) >= 2:
        width, height = int(image_size[0]), int(image_size[1])
    else:
        width = height = 0
    if (width <= 0 or height <= 0) and contract:
        canvas = contract.get("canvas") if isinstance(contract.get("canvas"), Mapping) else {}
        width = int(canvas.get("width") or 1024)
        height = int(canvas.get("height") or 1536)
    return max(1, width or 1024), max(1, height or 1536)


def _coerce_pixel_box(value: Any, *, width: int, height: int) -> PixelBox | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) < 4:
        return None
    x0, y0, x1, y1 = [int(round(float(v))) for v in value[:4]]
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _iou(a: PixelBox, b: PixelBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, area_a + area_b - inter)


def _center_distance(a: PixelBox, b: PixelBox) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _pad_norm(box: NormBox, pad_x: float, pad_y: float) -> NormBox:
    return _clamp_norm((box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y))


def _intersection_norm(a: NormBox, b: NormBox) -> NormBox:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return _clamp_norm(a)
    return _clamp_norm((x0, y0, x1, y1))


def _clamp_norm(box: NormBox) -> NormBox:
    x0, y0, x1, y1 = box
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    if x1 <= x0:
        x1 = min(1.0, x0 + 0.001)
    if y1 <= y0:
        y1 = min(1.0, y0 + 0.001)
    return x0, y0, x1, y1


def _round_box(box: NormBox) -> list[float]:
    return [round(float(v), 4) for v in box]
