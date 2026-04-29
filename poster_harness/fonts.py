from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PIL import ImageFont


REGULAR_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

BOLD_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def load_font(size: int, *, bold: bool = False, preferred: str | Path | None = None):
    """Load a cross-platform TrueType font, falling back to PIL's default.

    The harness should run on a clean user's machine, not only on Linux images
    with DejaVu installed under /usr/share.  Callers may pass an explicit font
    path from config/overlay metadata, but missing fonts must never crash image
    export.
    """

    for candidate in _font_candidates(bold=bold, preferred=preferred):
        try:
            if candidate and Path(candidate).exists():
                return ImageFont.truetype(str(candidate), max(1, int(size)))
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=max(1, int(size)))
    except TypeError:  # Pillow < 10
        return ImageFont.load_default()


def _font_candidates(*, bold: bool, preferred: str | Path | None = None) -> Iterable[str | Path]:
    if preferred:
        yield preferred
    env_key = "POSTER_HARNESS_FONT_BOLD" if bold else "POSTER_HARNESS_FONT_REGULAR"
    if os.getenv(env_key):
        yield os.environ[env_key]
    yield from (BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES)
