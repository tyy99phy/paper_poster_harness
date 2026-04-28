from __future__ import annotations

import gzip
import html
import json
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import request

from .config import cfg_get
from .llm import ChatGPTAccountResponsesProvider
from .schemas import arxiv_resolution_schema


ARXIV_ID_RE = re.compile(r"(?P<id>(?:arXiv:)?\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", re.I)


@dataclass(slots=True)
class ArxivDownload:
    arxiv_id: str
    abs_url: str
    pdf_url: str
    source_url: str
    pdf_path: Path
    eprint_path: Path
    source_dir: Path
    main_tex: Path
    asset_roots: list[Path]
    metadata_path: Path


def resolve_arxiv_with_llm(
    query: str,
    *,
    provider: ChatGPTAccountResponsesProvider,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    web_cfg = dict(cfg_get(dict(config), "llm.web_search", {}) or {})
    tool_type = str(web_cfg.get("tool_type") or "web_search")
    allowed_domains = list(web_cfg.get("allowed_domains") or ["arxiv.org"])
    tool: dict[str, Any] = {"type": tool_type}
    if allowed_domains:
        tool["filters"] = {"allowed_domains": allowed_domains}

    include = list(web_cfg.get("include") or ["web_search_call.action.sources"]) if web_cfg.get("include_sources", True) else None
    reasoning_effort = web_cfg.get("reasoning_effort")
    reasoning = {"effort": reasoning_effort} if reasoning_effort else None
    prompt = (
        "Use web search to find the single best matching arXiv paper for the user query.\n"
        "Return only a JSON object. The arxiv_id must be the canonical arXiv identifier, "
        "without the 'arXiv:' prefix. Prefer arxiv.org/abs and arxiv.org/pdf URLs. "
        "If multiple papers are plausible, choose the highest-confidence exact match and explain why.\n\n"
        f"User query: {query}"
    )
    envelope = provider.generate_json(
        stage_name="resolve_arxiv_with_web_search",
        prompt=prompt,
        schema=arxiv_resolution_schema(),
        system_prompt="You are an arXiv resolver. You must use web search and return grounded JSON only.",
        tools=[tool],
        tool_choice="auto",
        include=include,
        reasoning=reasoning,
    )
    result = dict(envelope["result"])
    return normalize_arxiv_resolution(result)


def normalize_arxiv_resolution(result: Mapping[str, Any]) -> dict[str, Any]:
    arxiv_id = normalize_arxiv_id(str(result.get("arxiv_id") or result.get("id") or ""))
    if not arxiv_id:
        for value in (result.get("abs_url"), result.get("pdf_url"), result.get("source_url"), result.get("title")):
            arxiv_id = normalize_arxiv_id(str(value or ""))
            if arxiv_id:
                break
    if not arxiv_id:
        raise RuntimeError("arXiv resolver did not return a valid arxiv_id")
    base = arxiv_id_versionless(arxiv_id)
    abs_url = str(result.get("abs_url") or f"https://arxiv.org/abs/{arxiv_id}")
    pdf_url = str(result.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}")
    source_url = str(result.get("source_url") or f"https://arxiv.org/e-print/{arxiv_id}")
    if "arxiv.org" not in abs_url or "arxiv.org" not in pdf_url:
        raise RuntimeError(f"arXiv resolver returned non-arxiv URLs: {abs_url}, {pdf_url}")
    return {
        "arxiv_id": arxiv_id,
        "arxiv_id_versionless": base,
        "title": str(result.get("title") or ""),
        "authors": list(result.get("authors") or []),
        "abstract": str(result.get("abstract") or ""),
        "published": str(result.get("published") or ""),
        "abs_url": abs_url,
        "pdf_url": pdf_url,
        "source_url": source_url,
        "confidence": float(result.get("confidence") or 0.0),
        "rationale": str(result.get("rationale") or ""),
        "sources": list(result.get("sources") or []),
    }


def normalize_arxiv_id(value: str) -> str:
    text = value.strip()
    text = text.replace("https://arxiv.org/abs/", "").replace("https://arxiv.org/pdf/", "")
    text = text.replace("http://arxiv.org/abs/", "").replace("http://arxiv.org/pdf/", "")
    text = text.removesuffix(".pdf").strip("/")
    match = ARXIV_ID_RE.search(text)
    if not match:
        return ""
    return match.group("id").replace("arXiv:", "")


def arxiv_id_versionless(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def download_arxiv_bundle(
    resolution: Mapping[str, Any],
    *,
    out_dir: str | Path,
    config: Mapping[str, Any],
) -> ArxivDownload:
    arxiv_id = normalize_arxiv_id(str(resolution.get("arxiv_id") or ""))
    if not arxiv_id:
        raise RuntimeError("download_arxiv_bundle requires a valid arxiv_id")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_id = arxiv_id.replace("/", "_")
    pdf_url = str(resolution.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}")
    source_url = str(resolution.get("source_url") or f"https://arxiv.org/e-print/{arxiv_id}")
    abs_url = str(resolution.get("abs_url") or f"https://arxiv.org/abs/{arxiv_id}")

    pdf_path = out / f"{safe_id}.pdf"
    eprint_path = out / f"{safe_id}.eprint"
    source_dir = out / "source"
    metadata_path = out / "arxiv_resolution.json"

    if cfg_get(dict(config), "arxiv.download_pdf", True):
        download_url(pdf_url, pdf_path)
    else:
        raise RuntimeError("arxiv.download_pdf=false is incompatible with strict autoposter")

    if cfg_get(dict(config), "arxiv.download_source", True):
        download_arxiv_source(source_url, eprint_path, arxiv_id=arxiv_id)
        extract_arxiv_source(eprint_path, source_dir)
    else:
        raise RuntimeError("arxiv.download_source=false is incompatible with strict autoposter")

    main_tex = find_main_tex(source_dir)
    asset_roots = find_source_asset_roots(source_dir, cfg_get(dict(config), "arxiv.source_asset_roots", []))
    metadata = dict(resolution)
    metadata.update(
        {
            "arxiv_id": arxiv_id,
            "abs_url": abs_url,
            "pdf_url": pdf_url,
            "source_url": source_url,
            "pdf_path": str(pdf_path),
            "eprint_path": str(eprint_path),
            "source_dir": str(source_dir),
            "main_tex": str(main_tex),
            "asset_roots": [str(path) for path in asset_roots],
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return ArxivDownload(
        arxiv_id=arxiv_id,
        abs_url=abs_url,
        pdf_url=pdf_url,
        source_url=source_url,
        pdf_path=pdf_path,
        eprint_path=eprint_path,
        source_dir=source_dir,
        main_tex=main_tex,
        asset_roots=asset_roots,
        metadata_path=metadata_path,
    )


def download_url(url: str, dest: str | Path, *, timeout: int = 180) -> Path:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    req = request.Request(url, headers={"User-Agent": "poster-harness/0.1"})
    with request.urlopen(req, timeout=timeout) as resp, path.open("wb") as handle:
        shutil.copyfileobj(resp, handle)
    if path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file from {url}")
    return path


def download_arxiv_source(url: str, dest: str | Path, *, arxiv_id: str, timeout: int = 180) -> Path:
    path = download_url(url, dest, timeout=timeout)
    if looks_like_html(path):
        text = path.read_text(encoding="utf-8", errors="replace")
        links = extract_eprint_links(text)
        candidates = links + [
            f"https://arxiv.org/e-print/{arxiv_id}",
            f"https://export.arxiv.org/e-print/{arxiv_id}",
            f"https://arxiv.org/src/{arxiv_id}",
        ]
        for candidate in candidates:
            if candidate == url:
                continue
            try:
                path = download_url(candidate, dest, timeout=timeout)
            except Exception:
                continue
            if not looks_like_html(path):
                return path
        raise RuntimeError(f"arXiv source download returned HTML instead of an e-print archive: {url}")
    return path


def looks_like_html(path: str | Path) -> bool:
    try:
        head = Path(path).read_bytes()[:512].lstrip().lower()
    except Exception:
        return False
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<title>" in head[:200]


def extract_eprint_links(text: str) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r'href=["\']([^"\']*(?:e-print|src)[^"\']*)["\']', text, flags=re.I):
        href = html.unescape(match.group(1))
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://arxiv.org" + href
        if href.startswith("http"):
            links.append(href)
    return links


def extract_arxiv_source(eprint_path: str | Path, out_dir: str | Path) -> Path:
    src = Path(eprint_path)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(src):
        with tarfile.open(src) as tf:
            safe_extract_tar(tf, out)
        return out
    data = src.read_bytes()
    try:
        data = gzip.decompress(data)
    except Exception:
        pass
    if data.lstrip().startswith(b"\\"):
        (out / "main.tex").write_bytes(data)
        return out
    raise RuntimeError(f"Unsupported arXiv source archive format: {src}")


def safe_extract_tar(tf: tarfile.TarFile, out_dir: Path) -> None:
    root = out_dir.resolve()
    for member in tf.getmembers():
        target = (out_dir / member.name).resolve()
        if root not in [target, *target.parents]:
            raise RuntimeError(f"Unsafe path in arXiv source archive: {member.name}")
    tf.extractall(out_dir)


def find_main_tex(source_dir: str | Path) -> Path:
    source = Path(source_dir)
    tex_files = sorted(p for p in source.rglob("*.tex") if p.is_file())
    if not tex_files:
        raise RuntimeError(f"No .tex files found in arXiv source directory: {source}")
    scored: list[tuple[int, Path]] = []
    for path in tex_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:120000]
        except Exception:
            continue
        score = 0
        if "\\documentclass" in text:
            score += 100
        if "\\begin{document}" in text:
            score += 100
        if "\\title" in text:
            score += 40
        if "\\abstract" in text or "\\begin{abstract}" in text:
            score += 40
        if "authorlist" in path.name.lower():
            score -= 200
        score += min(len(text) // 1000, 50)
        scored.append((score, path))
    if not scored:
        raise RuntimeError(f"Could not read candidate .tex files in {source}")
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] < 100:
        raise RuntimeError(f"No convincing main TeX file found in {source}")
    return scored[0][1]


def find_source_asset_roots(source_dir: str | Path, configured_names: Sequence[str]) -> list[Path]:
    source = Path(source_dir)
    roots: list[Path] = []
    names = [str(name) for name in configured_names]
    for name in names:
        candidate = source / name
        if candidate.exists() and candidate.is_dir():
            roots.append(candidate)
    if any(path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".pdf"} for path in source.iterdir() if path.is_file()):
        roots.append(source)
    seen: set[Path] = set()
    deduped: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(root)
    if not deduped:
        raise RuntimeError(f"No source asset roots found in arXiv source directory: {source}")
    return deduped
