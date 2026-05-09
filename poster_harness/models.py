from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FigureSlot:
    id: str
    label: str
    aspect: str = "1:1 square"
    role: str = "plot"
    asset: str | None = None
    group: str | None = None
    box: list[int] | None = None  # x0,y0,x1,y1 in generated image pixels


@dataclass
class TextBlock:
    title: str
    body: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)


@dataclass
class Section:
    id: int
    title: str
    layout: str = "card"
    text: list[TextBlock] = field(default_factory=list)
    figures: list[FigureSlot] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PosterSpec:
    title: str
    subtitle: str = ""
    authors: str = ""
    affiliation: str = ""
    style: str = "premium scientific conference poster"
    palette: dict[str, str] = field(default_factory=dict)
    forbidden_phrases: list[str] = field(default_factory=list)
    conclusion: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


ASPECT_HINTS = {
    "1:1": "1:1 square",
    "square": "1:1 square",
    "2:1": "2:1 wide",
    "2.4:1": "2.4:1 wide",
    "4:3": "4:3",
    "16:9": "16:9 wide",
    "3:2": "3:2 wide",
}
