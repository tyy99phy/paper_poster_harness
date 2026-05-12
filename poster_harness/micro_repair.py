from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFilter

from .fonts import load_font


ITALIC_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Oblique.ttf",
    "/Library/Fonts/Arial Italic.ttf",
    "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "C:/Windows/Fonts/ariali.ttf",
]

BOLD_ITALIC_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-BoldOblique.ttf",
    "/Library/Fonts/Arial Bold Italic.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
    "C:/Windows/Fonts/arialbi.ttf",
]


def apply_micro_repairs(
    *,
    image_path: str | Path,
    out_path: str | Path,
    repairs: Sequence[Mapping[str, Any]],
    scale: float = 1.0,
) -> Path:
    """Apply small deterministic poster repairs.

    Micro-repairs are for monotonic, local fixes after template/final QA: typos,
    unsupported short phrases, and tiny cosmetic cleanup.  They deliberately do
    not regenerate the whole poster and they never touch areas outside the
    explicit repair boxes.
    """

    canvas = Image.open(image_path).convert("RGB")
    for repair in repairs:
        kind = str(repair.get("type") or repair.get("kind") or "text_patch")
        if kind not in {"text_patch", "text_box", "glyph_patch"}:
            raise ValueError(f"unsupported micro-repair type: {kind}")
        _apply_text_patch(canvas, repair, scale=scale)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)
    return out


def _apply_text_patch(canvas: Image.Image, repair: Mapping[str, Any], *, scale: float) -> None:
    box = _scaled_box(repair.get("box"), scale=scale, canvas_size=canvas.size)
    if box is None:
        raise ValueError(f"micro-repair {repair.get('id') or ''} is missing a valid box")

    erase_mode = str(repair.get("erase") or repair.get("erase_mode") or "box").lower()
    fill = _parse_color(repair.get("fill"))
    if fill is None:
        fill = _sample_fill(canvas, box, prefer=str(repair.get("fill_prefer") or "auto"))

    if erase_mode in {"box", "rectangle", "rect"}:
        _fill_box(canvas, box, fill=fill, radius=int(round(float(repair.get("radius", 0) or 0) * scale)))
    elif erase_mode in {"text_mask", "glyph", "mask"}:
        _erase_text_mask(
            canvas,
            box,
            fill=fill,
            threshold=str(repair.get("text_threshold") or "auto"),
            dilate=int(round(float(repair.get("dilate", 3) or 0) * scale)),
        )
    elif erase_mode in {"none", "false"}:
        pass
    else:
        raise ValueError(f"unsupported micro-repair erase mode: {erase_mode}")

    lines = repair.get("lines")
    if lines is None:
        text = repair.get("text", "")
        lines = str(text).splitlines() if text is not None else []
    lines = [str(line) for line in lines if str(line)]
    if not lines:
        return

    font_size = max(4, int(round(float(repair.get("font_size", 24)) * scale)))
    style = str(repair.get("font_style") or repair.get("style") or "regular").lower()
    bold = "bold" in style
    preferred = _preferred_font(style)
    font = load_font(font_size, bold=bold, preferred=preferred)
    color = _parse_color(repair.get("color")) or (0, 0, 0)
    stroke_fill = _parse_color(repair.get("stroke_fill"))
    stroke_width = int(round(float(repair.get("stroke_width", 0) or 0) * scale))
    shadow = _parse_color(repair.get("shadow"))
    shadow_offset_raw = repair.get("shadow_offset") or [2, 2]
    try:
        shadow_offset = (
            int(round(float(shadow_offset_raw[0]) * scale)),
            int(round(float(shadow_offset_raw[1]) * scale)),
        )
    except Exception:
        shadow_offset = (int(round(2 * scale)), int(round(2 * scale)))

    draw = ImageDraw.Draw(canvas)
    padding = _padding(repair.get("padding"), scale=scale)
    x = box[0] + padding[0]
    y = box[1] + padding[1]
    max_width = max(1, box[2] - box[0] - padding[0] - padding[2])
    line_gap = int(round(float(repair.get("line_gap", font_size * 0.22)) * scale))
    align = str(repair.get("align") or "left").lower()
    wrapped: list[str] = []
    wrap = bool(repair.get("wrap", False))
    for line in lines:
        wrapped.extend(_wrap_text(line, max_width, draw, font) if wrap else [line])

    for line in wrapped:
        text_x = x
        try:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(line) * font_size // 2
            text_h = font_size
        if align == "center":
            text_x = box[0] + (box[2] - box[0] - text_w) // 2
        elif align == "right":
            text_x = box[2] - padding[2] - text_w
        if shadow:
            draw.text(
                (text_x + shadow_offset[0], y + shadow_offset[1]),
                line,
                fill=shadow,
                font=font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
        draw.text(
            (text_x, y),
            line,
            fill=color,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        y += text_h + line_gap


def _scaled_box(value: Any, *, scale: float, canvas_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v) * scale)) for v in value[:4]]
    except Exception:
        return None
    x0 = max(0, min(canvas_size[0], x0))
    x1 = max(0, min(canvas_size[0], x1))
    y0 = max(0, min(canvas_size[1], y0))
    y1 = max(0, min(canvas_size[1], y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _padding(value: Any, *, scale: float) -> tuple[int, int, int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = list(value)
        if len(raw) == 2:
            px, py = raw
            return (
                int(round(float(px) * scale)),
                int(round(float(py) * scale)),
                int(round(float(px) * scale)),
                int(round(float(py) * scale)),
            )
        if len(raw) >= 4:
            return tuple(int(round(float(item) * scale)) for item in raw[:4])  # type: ignore[return-value]
    pad = int(round(float(value or 0) * scale))
    return pad, pad, pad, pad


def _parse_color(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) < 3:
        return None
    try:
        return tuple(max(0, min(255, int(round(float(v))))) for v in value[:3])  # type: ignore[return-value]
    except Exception:
        return None


def _preferred_font(style: str) -> str | None:
    candidates = BOLD_ITALIC_FONT_CANDIDATES if "bold" in style and "italic" in style else ITALIC_FONT_CANDIDATES
    if "italic" not in style and "oblique" not in style:
        return None
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _sample_fill(canvas: Image.Image, box: tuple[int, int, int, int], *, prefer: str = "auto") -> tuple[int, int, int]:
    x0, y0, x1, y1 = box
    pad = max(8, int(round(min(x1 - x0, y1 - y0) * 0.30)))
    outer = (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(canvas.width, x1 + pad),
        min(canvas.height, y1 + pad),
    )
    pixels = canvas.load()
    samples: list[tuple[int, int, int]] = []
    step = max(1, int(round(max(canvas.size) / 1800)))
    for y in range(outer[1], outer[3], step):
        for x in range(outer[0], outer[2], step):
            if x0 <= x < x1 and y0 <= y < y1:
                continue
            r, g, b = pixels[x, y]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            chroma = max(r, g, b) - min(r, g, b)
            if prefer == "light" and lum < 190:
                continue
            if prefer == "dark" and lum > 120:
                continue
            if prefer == "auto":
                # Avoid sampling foreground text/accent strokes when possible.
                if lum < 20 or (lum > 235 and chroma < 18):
                    continue
            samples.append((r, g, b))
    if not samples:
        crop = canvas.crop(box).resize((1, 1), Image.Resampling.BOX)
        return crop.getpixel((0, 0))
    return tuple(int(median(channel)) for channel in zip(*samples))  # type: ignore[return-value]


def _fill_box(canvas: Image.Image, box: tuple[int, int, int, int], *, fill: tuple[int, int, int], radius: int) -> None:
    draw = ImageDraw.Draw(canvas)
    if radius > 0:
        draw.rounded_rectangle([box[0], box[1], box[2], box[3]], radius=radius, fill=fill)
    else:
        draw.rectangle([box[0], box[1], box[2], box[3]], fill=fill)


def _erase_text_mask(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    threshold: str,
    dilate: int,
) -> None:
    crop = canvas.crop(box).convert("RGB")
    mask = Image.new("L", crop.size, 0)
    px = crop.load()
    mp = mask.load()
    for y in range(crop.height):
        for x in range(crop.width):
            r, g, b = px[x, y]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            chroma = max(r, g, b) - min(r, g, b)
            erase = False
            if threshold in {"light", "white"}:
                erase = lum > 145 and chroma < 85
            elif threshold in {"dark", "black"}:
                erase = lum < 110 and chroma < 90
            else:
                erase = (lum > 150 and chroma < 95) or (lum < 70 and chroma < 80)
            if erase:
                mp[x, y] = 255
    if dilate > 0:
        size = max(3, dilate * 2 + 1)
        if size % 2 == 0:
            size += 1
        mask = mask.filter(ImageFilter.MaxFilter(size))
    fill_img = Image.new("RGB", crop.size, fill)
    canvas.paste(fill_img, (box[0], box[1]), mask)


def _wrap_text(text: str, max_width: int, draw: ImageDraw.ImageDraw, font) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
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
