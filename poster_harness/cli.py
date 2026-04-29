from __future__ import annotations

import argparse
import copy
import shutil
import re
from pathlib import Path
from typing import Any, Mapping

from .arxiv import download_arxiv_bundle, normalize_arxiv_id, resolve_arxiv_with_llm
from .assets import (
    apply_detections_to_spec,
    apply_figure_selection_to_spec,
    build_assets_manifest,
    infer_asset_roots,
)
from .auth_login import DEFAULT_CALLBACK_PORT, run_browser_login
from .config import (
    cfg_get,
    load_config,
    dump_config,
    load_harness_config,
    write_default_harness_config,
)
from .extract import extract_pdf_images, extract_text, extract_pptx_media, render_pdf_pages
from .latex_utils import clean_latex_inline, extract_latex_braced
from .prompt import build_prompt, sanitize_public_text
from .replace import audit_generated_placeholder_geometry, normalize_placeholder_geometry, replace_placeholders, upscale_image
from .image_backend import generate_images_from_config
from .llm import ChatGPTAccountResponsesProvider
from .llm_stages import (
    detect_placeholders_from_image,
    draft_spec_from_text,
    qa_poster,
    select_figures,
)
from .schemas import DEFAULT_MODEL, default_poster_spec


def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.project_dir)
    for sub in ["input", "assets", "generated", "exports", "specs", "prompts", "scratch"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    spec = default_poster_spec(title=args.title or "Untitled Scientific Poster")
    dump_config(spec, root / "specs" / "poster_spec.yaml")
    print(root)
    print(root / "specs" / "poster_spec.yaml")


def cmd_init_config(args: argparse.Namespace) -> None:
    write_default_harness_config(args.out)
    print(args.out)
    if getattr(args, "login", False):
        result = run_browser_login(
            out_dir=args.auth_dir,
            out_file=args.auth_file,
            callback_port=args.callback_port,
            open_browser=not args.no_browser,
            timeout_s=args.login_timeout,
            force=args.force_auth,
        )
        print(result.path)


def cmd_login(args: argparse.Namespace) -> None:
    result = run_browser_login(
        out_dir=args.auth_dir,
        out_file=args.auth_file,
        callback_port=args.callback_port,
        open_browser=not args.no_browser,
        timeout_s=args.timeout,
        force=args.force,
    )
    summary = result.public_summary()
    print(summary["path"])
    print(f"email: {summary['email']}")
    print(f"account_id: {summary['account_id']}")


def cmd_resolve_arxiv(args: argparse.Namespace) -> None:
    config = load_harness_config(args.config)
    if args.arxiv_id:
        arxiv_id = normalize_arxiv_id(args.arxiv_id)
        if not arxiv_id:
            raise SystemExit(f"invalid --arxiv-id: {args.arxiv_id}")
        resolution = {
            "arxiv_id": arxiv_id,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "source_url": f"https://arxiv.org/e-print/{arxiv_id}",
            "confidence": 1.0,
            "sources": [{"title": f"arXiv:{arxiv_id}", "url": f"https://arxiv.org/abs/{arxiv_id}"}],
        }
    else:
        if not args.query:
            raise SystemExit("resolve-arxiv requires --query or --arxiv-id")
        provider = _provider_from_config(config, model=args.model or cfg_get(config, "llm.web_search.model") or cfg_get(config, "llm.model"))
        resolution = resolve_arxiv_with_llm(args.query, provider=provider, config=config)
    dump_config(resolution, args.out)
    print(args.out)


def cmd_extract(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inp = Path(args.input)
    text_path = out / "extracted_text.txt"
    text = extract_text(inp, text_path)
    print(text_path)
    if inp.suffix.lower() == ".pptx":
        media_dir = out / "media"
        extract_pptx_media(inp, media_dir)
        print(media_dir)
    elif inp.suffix.lower() == ".pdf" and args.render_pages:
        pages_dir = out / "pages"
        render_pdf_pages(inp, pages_dir, dpi=args.dpi, max_pages=args.max_pages)
        print(pages_dir)
    if inp.suffix.lower() == ".pdf" and args.extract_images:
        images_dir = out / "pdf_images"
        extract_pdf_images(
            inp,
            images_dir,
            min_width=args.min_image_width,
            min_height=args.min_image_height,
            max_images=args.max_images,
        )
        print(images_dir)


def cmd_draft_spec(args: argparse.Namespace) -> None:
    text = Path(args.text).read_text(encoding="utf-8", errors="replace")
    title = args.title or _guess_title(text)
    spec = default_poster_spec(title=title)
    # Fill with a conservative manual starter; strict autoposter uses LLM stages instead.
    abstract = _guess_abstract(text)
    if abstract:
        spec["sections"][0]["text"][0]["body"] = [abstract]
    if args.authors:
        spec["project"]["authors"] = args.authors
    if args.topic:
        spec["project"]["topic"] = args.topic
    dump_config(spec, args.out)
    print(args.out)


def cmd_prompt(args: argparse.Namespace) -> None:
    spec = load_config(args.spec)
    prompt = build_prompt(spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(prompt, encoding="utf-8")
    print(out)


def cmd_generate(args: argparse.Namespace) -> None:
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    config = load_harness_config(args.config)
    paths = generate_images_from_config(
        prompt=prompt,
        out_dir=args.out_dir,
        basename=args.basename,
        config=config,
        model=args.model,
        size=args.size,
        quality=args.quality,
        account=args.account,
        n=args.n,
    )
    for p in paths:
        print(p)


def cmd_replace(args: argparse.Namespace) -> None:
    spec = load_config(args.spec)
    out = replace_placeholders(
        base_image=args.base_image,
        spec=spec,
        asset_dir=args.asset_dir,
        out_path=args.out,
        scale=args.scale,
        dry_run=args.dry_run,
    )
    print(out)


def cmd_upscale(args: argparse.Namespace) -> None:
    out = upscale_image(args.input, args.out, factor=args.factor, sharpen=not args.no_sharpen)
    print(out)


def cmd_sanitize(args: argparse.Namespace) -> None:
    text = Path(args.input).read_text(encoding="utf-8")
    spec = load_config(args.spec) if args.spec else {}
    cleaned = sanitize_public_text(text, spec.get("forbidden_phrases", []))
    Path(args.out).write_text(cleaned, encoding="utf-8")
    print(args.out)


def cmd_manifest(args: argparse.Namespace) -> None:
    manifest = build_assets_manifest(
        args.assets_dir,
        out_dir=args.copy_to,
        copy_assets=bool(args.copy_to),
        recursive=not args.no_recursive,
        max_assets=args.max_assets,
        min_width=args.min_width,
        min_height=args.min_height,
        contact_sheet=not args.no_contact_sheet,
    )
    dump_config(manifest, args.out)
    print(args.out)
    if args.copy_to:
        print(args.copy_to)


def cmd_llm_draft_spec(args: argparse.Namespace) -> None:
    text = _read_text_file(args.text)
    envelope = draft_spec_from_text(
        text,
        assets_manifest=_load_optional_config(args.assets_manifest),
        provider=_llm_provider(args),
        project_overrides=_load_mapping_arg(args.project_json, root_key="project"),
        style_overrides=_load_mapping_arg(args.style_json, root_key="style"),
    )
    _dump_llm_result(envelope, args.out)
    print(args.out)


def cmd_llm_select_figures(args: argparse.Namespace) -> None:
    text = _read_text_file(args.text)
    spec = _load_spec_arg(args.spec) if args.spec else None
    envelope = select_figures(
        text,
        _load_required_config(args.assets_manifest, "--assets-manifest"),
        spec=spec,
        provider=_llm_provider(args),
        max_figures=args.max_figures,
    )
    _dump_llm_result(envelope, args.out)
    print(args.out)


def cmd_llm_detect_placeholders(args: argparse.Namespace) -> None:
    expected = None
    if args.spec:
        spec = _load_spec_arg(args.spec)
        expected = spec.get("placeholders") or []
    elif args.expected_json:
        expected = _extract_expected_placeholders(_load_required_config(args.expected_json, "--expected-json"))
    envelope = detect_placeholders_from_image(
        args.image,
        expected_placeholders=expected,
        provider=_llm_provider(args),
    )
    _dump_llm_result(envelope, args.out)
    print(args.out)


def cmd_llm_qa(args: argparse.Namespace) -> None:
    envelope = qa_poster(
        _load_spec_arg(args.spec),
        prompt=_read_prompt_arg(args.prompt),
        image_path=args.image,
        detected_placeholders=_load_detection_arg(args.detections),
        provider=_llm_provider(args),
        qa_mode=args.mode,
    )
    _dump_llm_result(envelope, args.out)
    print(args.out)


def cmd_autoposter(args: argparse.Namespace) -> None:
    config = load_harness_config(args.config)
    provider = _provider_from_config(config, model=args.model or cfg_get(config, "llm.model", DEFAULT_MODEL))
    resolution: dict[str, Any] | None = None
    arxiv_bundle = None
    query = args.query or cfg_get(config, "paper.query", "")
    arxiv_id_arg = args.arxiv_id or cfg_get(config, "paper.arxiv_id", "")
    paper_arg = args.paper or cfg_get(config, "paper.paper", "")
    text_source_arg = args.text_source or cfg_get(config, "paper.text_source", "")

    if query:
        search_provider = _provider_from_config(
            config,
            model=args.search_model or cfg_get(config, "llm.web_search.model") or cfg_get(config, "llm.model", DEFAULT_MODEL),
        )
        resolution = resolve_arxiv_with_llm(query, provider=search_provider, config=config)
    elif arxiv_id_arg:
        arxiv_id = normalize_arxiv_id(str(arxiv_id_arg))
        if not arxiv_id:
            raise SystemExit(f"invalid --arxiv-id: {args.arxiv_id}")
        resolution = {
            "arxiv_id": arxiv_id,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "source_url": f"https://arxiv.org/e-print/{arxiv_id}",
            "confidence": 1.0,
            "sources": [{"title": f"arXiv:{arxiv_id}", "url": f"https://arxiv.org/abs/{arxiv_id}"}],
        }

    root = Path(args.out or cfg_get(config, "paper.out", "") or _default_run_dir(paper_arg, config, resolution))
    dirs = _ensure_project_dirs(root)

    config_assets = cfg_get(config, "paper.assets_dir", []) or []
    source_roots: list[Path] = [Path(p) for p in [*config_assets, *(args.assets_dir or [])] if p]
    if resolution:
        arxiv_bundle = download_arxiv_bundle(resolution, out_dir=dirs["input"] / "arxiv", config=config)
        paper = arxiv_bundle.pdf_path
        text_source = Path(text_source_arg) if text_source_arg else arxiv_bundle.main_tex
        source_roots.extend(arxiv_bundle.asset_roots)
    else:
        if not paper_arg:
            raise SystemExit("autoposter requires --query, --arxiv-id, or --paper")
        paper = Path(str(paper_arg))
        if not paper.exists():
            raise SystemExit(f"--paper does not exist: {paper}")
        text_source = Path(text_source_arg) if text_source_arg else paper

    copied_paper = dirs["input"] / paper.name
    if paper.resolve() != copied_paper.resolve():
        shutil.copy2(paper, copied_paper)

    text_path = dirs["input"] / "extracted_text.txt"
    text = _extract_text_strict(text_source, text_path)

    extracted_roots: list[Path] = []
    if paper.suffix.lower() == ".pptx":
        media_dir = dirs["input"] / "media"
        extract_pptx_media(paper, media_dir)
        extracted_roots.append(media_dir)
    if paper.suffix.lower() == ".pdf":
        if _opt_bool(args.render_pages, config, "autoposter.render_pages", True):
            render_pdf_pages(
                paper,
                dirs["input"] / "pages",
                dpi=int(_opt(args.dpi, config, "autoposter.pdf_render_dpi", 220)),
                max_pages=_opt(args.max_pages, config, "autoposter.max_pages", 12),
            )
        if _opt_bool(args.extract_pdf_images, config, "autoposter.extract_pdf_images", True):
            pdf_images = dirs["input"] / "pdf_images"
            extract_pdf_images(
                paper,
                pdf_images,
                min_width=int(_opt(args.min_image_width, config, "autoposter.min_image_width", 96)),
                min_height=int(_opt(args.min_image_height, config, "autoposter.min_image_height", 96)),
                max_images=_opt(args.max_assets, config, "autoposter.max_assets", 48),
            )
            extracted_roots.append(pdf_images)

    if _opt_bool(args.auto_assets, config, "autoposter.auto_assets", True):
        source_roots = infer_asset_roots(paper, explicit_roots=source_roots, extracted_roots=extracted_roots)
    elif extracted_roots:
        source_roots.extend(extracted_roots)
    source_roots = _dedupe_paths(source_roots)

    assets_manifest = build_assets_manifest(
        source_roots,
        out_dir=dirs["assets"],
        copy_assets=True,
        recursive=_opt_bool(args.recursive_assets, config, "autoposter.recursive_assets", True),
        max_assets=_opt(args.max_assets, config, "autoposter.max_assets", 48),
        min_width=int(_opt(args.min_image_width, config, "autoposter.min_image_width", 96)),
        min_height=int(_opt(args.min_image_height, config, "autoposter.min_image_height", 96)),
    )
    _attach_latex_captions(assets_manifest, text_source)
    assets_manifest_path = dirs["specs"] / "assets_manifest.yaml"
    dump_config(assets_manifest, assets_manifest_path)

    style_name = args.style or cfg_get(config, "autoposter.style", "generic")
    project_overrides, style_overrides, spec_extras = _style_preset(style_name, config)
    draft_envelope = draft_spec_from_text(
        text,
        assets_manifest=assets_manifest,
        provider=provider,
        project_overrides=project_overrides,
        style_overrides=style_overrides,
    )
    draft_spec = _apply_spec_extras(dict(draft_envelope["result"]), spec_extras)
    draft_spec_path = dirs["specs"] / "poster_spec.draft.yaml"
    dump_config(draft_spec, draft_spec_path)

    selection_envelope = select_figures(
        text,
        assets_manifest,
        spec=draft_spec,
        provider=provider,
        max_figures=_opt(args.max_figures, config, "autoposter.max_figures", None),
        extra_instructions=str(cfg_get(config, "autoposter.figure_layout_policy", "") or ""),
    )
    figure_selection = dict(selection_envelope["result"])
    figure_selection_path = dirs["specs"] / "figure_selection.yaml"
    dump_config(figure_selection, figure_selection_path)

    final_spec = apply_figure_selection_to_spec(
        draft_spec,
        figure_selection,
        prune_unselected=not _opt_bool(args.keep_unselected_placeholders, config, "autoposter.keep_unselected_placeholders", False),
    )
    final_spec = _apply_spec_extras(final_spec, spec_extras)
    spec_path = dirs["specs"] / "poster_spec.yaml"
    dump_config(final_spec, spec_path)

    prompt = build_prompt(final_spec)
    prompt_path = dirs["prompts"] / "poster_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    run_manifest: dict[str, Any] = {
        "paper": str(paper),
        "out": str(root),
        "style": style_name,
        "config": str(args.config) if args.config else "",
        "arxiv": resolution or {},
        "arxiv_bundle": {
            "metadata": str(arxiv_bundle.metadata_path),
            "source_dir": str(arxiv_bundle.source_dir),
            "main_tex": str(arxiv_bundle.main_tex),
        } if arxiv_bundle else {},
        "text_source": str(text_source),
        "source_roots": [str(p) for p in source_roots],
        "assets_manifest": str(assets_manifest_path),
        "draft_spec": str(draft_spec_path),
        "figure_selection": str(figure_selection_path),
        "poster_spec": str(spec_path),
        "prompt": str(prompt_path),
        "generated": [],
        "exports": [],
        "qa": [],
    }
    if not assets_manifest.get("assets"):
        raise RuntimeError("autoposter: no usable image assets found; strict mode will not continue without real figure assets")

    try:
        generated = generate_images_from_config(
            prompt=prompt,
            out_dir=dirs["generated"],
            basename=args.basename or f"{paper.stem}-placeholder-layout",
            config=config,
            model=args.image_model or cfg_get(config, "image_generation.model", "gpt-5.5"),
            size=args.size or cfg_get(config, "image_generation.size", "1024x1536"),
            quality=args.quality or cfg_get(config, "image_generation.quality", "high"),
            account=args.account or cfg_get(config, "image_generation.account.account", ""),
            n=int(_opt(args.variants, config, "image_generation.variants", 1)),
        )
    except Exception:
        dump_config(run_manifest, root / "run_manifest.yaml")
        raise
    generated_scale = float(cfg_get(config, "image_generation.generated_scale", 1.0) or 1.0)
    native_generated: list[Path] = []
    if generated_scale > 1:
        generated, native_generated = _promote_generated_templates_to_scale(generated, factor=generated_scale)
        if native_generated:
            run_manifest["generated_native"] = [str(p) for p in native_generated]
        run_manifest["generated_scale"] = generated_scale
    run_manifest["generated"] = [str(p) for p in generated]
    if not generated:
        raise RuntimeError("autoposter: image generation returned no images")

    placeholder_failures: list[str] = []
    for index, image_path in enumerate(generated, start=1):
        stem = Path(image_path).stem
        detection_envelope = detect_placeholders_from_image(
            image_path,
            expected_placeholders=final_spec.get("placeholders") or [],
            provider=provider,
        )
        detections = dict(detection_envelope["result"])
        detections_path = dirs["scratch"] / f"{stem}.detections.yaml"
        dump_config(detections, detections_path)

        spec_with_placements = apply_detections_to_spec(
            copy.deepcopy(final_spec),
            detections,
            min_confidence=float(_opt(args.min_detection_confidence, config, "autoposter.min_detection_confidence", 0.15)),
        )
        geometry_issues = audit_generated_placeholder_geometry(
            base_image=image_path,
            spec=spec_with_placements,
            ratio_tolerance=float(cfg_get(config, "autoposter.placeholder_aspect_tolerance", 0.20)),
        )
        if geometry_issues:
            geometry_issue_path = dirs["qa"] / f"{stem}.placeholder-geometry.qa.yaml"
            dump_config({"passes": False, "issues": geometry_issues}, geometry_issue_path)
            run_manifest["qa"].append(str(geometry_issue_path))
            placeholder_failures.append(
                f"{image_path}: failed deterministic placeholder geometry QA; see {geometry_issue_path}"
            )
            continue
        qa_image_path = Path(image_path)
        if bool(cfg_get(config, "autoposter.normalize_placeholder_geometry", True)):
            redraw_geometry = bool(cfg_get(config, "autoposter.redraw_normalized_placeholders", False))
            qa_image_path, spec_with_placements = normalize_placeholder_geometry(
                base_image=image_path,
                spec=spec_with_placements,
                out_path=dirs["generated"] / f"{stem}-geometry-normalized.png",
                redraw=redraw_geometry,
            )
            _overwrite_detection_boxes(detections, spec_with_placements.get("placements") or {})
            dump_config(detections, detections_path)
        placed_spec_path = dirs["specs"] / f"poster_spec.{index:02d}.with_placements.yaml"
        dump_config(spec_with_placements, placed_spec_path)
        placements = spec_with_placements.get("placements") or {}
        if not placements:
            raise RuntimeError(f"autoposter: no valid placeholder placements detected for {image_path}")

        placeholder_qa_envelope = qa_poster(
            spec_with_placements,
            prompt=prompt,
            image_path=qa_image_path,
            detected_placeholders=detections,
            provider=provider,
            qa_mode="placeholder",
        )
        placeholder_qa_result = dict(placeholder_qa_envelope["result"])
        placeholder_qa_path = dirs["qa"] / f"{stem}.placeholder.qa.yaml"
        dump_config(placeholder_qa_result, placeholder_qa_path)
        run_manifest["qa"].append(str(placeholder_qa_path))
        if not bool(placeholder_qa_result.get("passes")):
            placeholder_failures.append(
                f"{image_path}: failed strict placeholder QA; see {placeholder_qa_path}"
            )
            continue

        export_path = dirs["exports"] / f"{stem}-realfigures.png"
        # Use the original generated poster as the final visual base.  Geometry
        # normalization is a hidden replacement plan, not a second visible
        # placeholder layer; this avoids white normalized boxes that do not align
        # with the model's original card art.
        replacement_base_path = Path(image_path)
        replace_placeholders(
            base_image=replacement_base_path,
            spec=spec_with_placements,
            asset_dir=dirs["assets"],
            out_path=export_path,
        )
        run_manifest["exports"].append(str(export_path))
        qa_image = export_path
        upscale_factor = float(_opt(args.upscale_factor, config, "image_generation.upscale_factor", 4.0) or 0)
        if upscale_factor and upscale_factor > 1:
            upscaled_path = dirs["exports"] / f"{stem}-realfigures-{int(upscale_factor)}x.png"
            if generated_scale >= upscale_factor:
                _link_or_copy(export_path, upscaled_path)
            else:
                # For the production high-res export, upscale the generated
                # template first and paste real scientific figures directly at
                # the target coordinates. This keeps paper figures sharper than
                # enlarging the already-composited result.
                extra_scale = upscale_factor / max(1.0, generated_scale)
                upscaled_base_path = dirs["generated"] / f"{stem}-production-base-{int(upscale_factor)}x.png"
                upscale_image(replacement_base_path, upscaled_base_path, factor=extra_scale)
                replace_placeholders(
                    base_image=upscaled_base_path,
                    spec=spec_with_placements,
                    asset_dir=dirs["assets"],
                    out_path=upscaled_path,
                    scale=extra_scale,
                )
            run_manifest["exports"].append(str(upscaled_path))
            qa_image = upscaled_path

        qa_envelope = qa_poster(
            spec_with_placements,
            prompt=prompt,
            image_path=qa_image,
            detected_placeholders=detections,
            provider=provider,
            qa_mode="final",
        )
        qa_result = dict(qa_envelope["result"])
        qa_path = dirs["qa"] / f"{stem}.final.qa.yaml"
        dump_config(qa_result, qa_path)
        run_manifest["qa"].append(str(qa_path))

    if not run_manifest["exports"]:
        run_manifest_path = root / "run_manifest.yaml"
        dump_config(run_manifest, run_manifest_path)
        detail = "; ".join(placeholder_failures) if placeholder_failures else "no exportable generated image"
        raise RuntimeError(f"autoposter: no generated poster passed strict placeholder QA ({detail})")

    run_manifest_path = root / "run_manifest.yaml"
    dump_config(run_manifest, run_manifest_path)
    print(run_manifest_path)
    for item in run_manifest["exports"] or run_manifest["generated"]:
        print(item)


def _llm_provider(args: argparse.Namespace):
    config_path = getattr(args, "config", None)
    if config_path:
        return _provider_from_config(load_harness_config(config_path), model=getattr(args, "model", None))
    # The command-line LLM subcommands are still strict. Without a config they
    # use the built-in default config, whose default backend is chatgpt_account.
    return _provider_from_config(load_harness_config(None), model=getattr(args, "model", None))


def _overwrite_detection_boxes(detections: dict[str, Any], placements: Mapping[str, Any]) -> None:
    normalized = {str(key): [int(round(float(v))) for v in value] for key, value in dict(placements).items()}
    for item in detections.get("placeholders") or []:
        if isinstance(item, dict) and str(item.get("id")) in normalized:
            item["bbox"] = normalized[str(item.get("id"))]
    if normalized:
        detections["placements"] = normalized


def _promote_generated_templates_to_scale(paths: list[Path], *, factor: float) -> tuple[list[Path], list[Path]]:
    promoted: list[Path] = []
    native_paths: list[Path] = []
    for path_like in paths:
        path = Path(path_like)
        native = _unique_sibling(path, suffix="-native")
        path.rename(native)
        upscale_image(native, path, factor=factor)
        promoted.append(path)
        native_paths.append(native)
    return promoted, native_paths


def _unique_sibling(path: Path, *, suffix: str) -> Path:
    candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}{suffix}-{idx}{path.suffix}")
        idx += 1
    return candidate


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _provider_from_config(config: Mapping[str, Any], *, model: str | None = None):
    cfg = dict(config)
    backend = str(cfg_get(cfg, "llm.backend", "chatgpt_account") or "chatgpt_account")
    llm_model = model or cfg_get(cfg, "llm.model", None) or DEFAULT_MODEL
    timeout = int(cfg_get(cfg, "llm.timeout", 180))

    if backend == "chatgpt_account":
        account_cfg = dict(cfg_get(cfg, "llm.account", {}) or {})
        return ChatGPTAccountResponsesProvider(
            model=str(llm_model),
            timeout=timeout,
            account=str(account_cfg.get("account") or "") or None,
            auth_dir=account_cfg.get("auth_dir") or None,
            auth_file=account_cfg.get("auth_file") or None,
            endpoint=str(account_cfg.get("endpoint") or "https://chatgpt.com/backend-api/codex/responses"),
            min_token_seconds=int(account_cfg.get("min_token_seconds") or 60),
            proxy=str(account_cfg.get("proxy") or "") or None,
        )

    raise RuntimeError(f"unsupported llm.backend={backend!r}. Supported: chatgpt_account")


def _opt(value: Any, config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    return value if value is not None else cfg_get(dict(config), dotted, default)


def _opt_bool(value: Any, config: Mapping[str, Any], dotted: str, default: bool) -> bool:
    return bool(value if value is not None else cfg_get(dict(config), dotted, default))


def _default_run_dir(paper: str | None, config: Mapping[str, Any], resolution: Mapping[str, Any] | None) -> str:
    runs_dir = Path(str(cfg_get(dict(config), "paths.runs_dir", "runs")))
    if resolution and resolution.get("arxiv_id"):
        return str(runs_dir / f"arxiv_{str(resolution['arxiv_id']).replace('/', '_')}")
    if paper:
        return str(runs_dir / Path(paper).stem)
    return str(runs_dir / "poster_run")


def _read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _read_prompt_arg(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return value


def _load_optional_config(path: str | None) -> Any:
    return load_config(path) if path else None


def _load_required_config(path: str, label: str) -> Any:
    try:
        return load_config(path)
    except Exception as exc:
        raise SystemExit(f"failed to load {label} {path!r}: {exc}") from exc


def _load_mapping_arg(path: str | None, *, root_key: str | None = None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = _unwrap_stage_result(_load_required_config(path, f"--{root_key}-json" if root_key else "config"))
    if root_key and isinstance(payload, Mapping) and isinstance(payload.get(root_key), Mapping):
        payload = payload[root_key]
    if not isinstance(payload, Mapping):
        raise SystemExit(f"{path!r} must contain a JSON/YAML object")
    return dict(payload)


def _load_spec_arg(path: str) -> dict[str, Any]:
    payload = _unwrap_stage_result(_load_required_config(path, "--spec"))
    if not isinstance(payload, Mapping):
        raise SystemExit(f"--spec {path!r} must contain a JSON/YAML object")
    if "project" not in payload and isinstance(payload.get("poster_spec"), Mapping):
        payload = payload["poster_spec"]
    return dict(payload)


def _load_detection_arg(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = _unwrap_stage_result(_load_required_config(path, "--detections"))
    if not isinstance(payload, Mapping):
        raise SystemExit(f"--detections {path!r} must contain a JSON/YAML object")
    return dict(payload)


def _extract_expected_placeholders(payload: Any) -> Any:
    payload = _unwrap_stage_result(payload)
    if isinstance(payload, Mapping):
        if isinstance(payload.get("placeholders"), list):
            return payload["placeholders"]
        if isinstance(payload.get("expected_placeholders"), list):
            return payload["expected_placeholders"]
        if isinstance(payload.get("poster_spec"), Mapping):
            return payload["poster_spec"].get("placeholders") or []
    return payload


def _unwrap_stage_result(payload: Any) -> Any:
    if isinstance(payload, Mapping) and "result" in payload and (
        "stage" in payload or "mode" in payload or "provider" in payload
    ):
        return payload["result"]
    return payload


def _dump_llm_result(envelope: Mapping[str, Any], path: str | Path) -> None:
    result = envelope.get("result") if isinstance(envelope, Mapping) else envelope
    if not isinstance(result, Mapping):
        raise SystemExit("LLM stage did not return a JSON/YAML object result")
    dump_config(dict(result), path)


def _ensure_project_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        name: root / name
        for name in ["input", "assets", "generated", "exports", "specs", "prompts", "scratch", "qa"]
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            key = path.resolve()
        except Exception:
            key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _extract_text_strict(text_source: Path, text_path: Path) -> str:
    """Extract text from the explicit source only; never switch sources implicitly."""
    if not text_source.exists():
        raise SystemExit(f"--text-source/--paper does not exist: {text_source}")
    text = extract_text(text_source, text_path)
    words = _word_count(text)
    if words < 120:
        raise RuntimeError(
            f"autoposter: extracted text from {text_source} is too short ({words} words). "
            "Strict mode will not infer another source; pass a richer source explicitly with --text-source."
        )
    return text


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _attach_latex_captions(manifest: dict[str, Any], text_source: Path) -> None:
    if text_source.suffix.lower() != ".tex":
        return
    try:
        text = text_source.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
    captions = _latex_asset_caption_map(text)
    if not captions:
        return
    for asset in manifest.get("assets") or []:
        stem = Path(str(asset.get("source_path") or asset.get("asset") or asset.get("name") or "")).stem
        caption = captions.get(stem)
        if not caption:
            continue
        asset["caption"] = caption
        asset["label"] = _caption_to_label(caption, default_label=str(asset.get("label") or stem))


def _latex_asset_caption_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    figure_blocks = re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", text, flags=re.S)
    if not figure_blocks:
        figure_blocks = [text]
    for block in figure_blocks:
        includes = [
            Path(match.group(1).strip()).stem
            for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", block)
        ]
        if not includes:
            continue
        caption = extract_latex_braced(block, "caption") or extract_latex_braced(block, "topcaption")
        if not caption:
            continue
        cleaned = clean_latex_inline(caption)
        for idx, stem in enumerate(includes):
            label = cleaned
            if len(includes) == 2 and re.search(r"\bleft\b", cleaned, flags=re.I) and re.search(r"\bright\b", cleaned, flags=re.I):
                label = f"{cleaned} ({'left' if idx == 0 else 'right'} panel)"
            mapping[stem] = label
    return mapping


def _caption_to_label(caption: str, *, default_label: str) -> str:
    low = caption.lower()
    if "feynman" in low or "diagram" in low:
        if "left panel" in low and "heavy majorana" in low:
            return "Heavy Majorana neutrino VBF diagram"
        if "right panel" in low and "weinberg" in low:
            return "Weinberg-operator VBF diagram"
    text = re.split(r"(?<=[.!?])\s+", caption.strip())[0] or default_label
    text = re.sub(r"^Example\s+", "", text, flags=re.I)
    return text[:117].rstrip() + "..." if len(text) > 120 else text


def _style_preset(name: str, config: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    normalized = (name or "generic").lower().replace("_", "-")
    if config:
        styles = cfg_get(dict(config), "styles", {}) or {}
        style_cfg = styles.get(normalized) or styles.get(name) or {}
        if style_cfg:
            return (
                dict(style_cfg.get("project") or {}),
                dict(style_cfg.get("style") or {}),
                dict(style_cfg.get("extras") or {}),
            )
    if normalized in {"cms", "cms-hep", "hep"}:
        return (
            {
                "audience": "high-energy physics conference audience",
            },
            {
                "summary": "premium high-energy-physics conference poster, CERN/LHCC-inspired, CMS-style detector aesthetic, luminous beamline abstraction, artistic but readable; not a collage",
                "aspect": "A0 vertical / 2:3 ratio",
                "top_band": "dark navy title band with identity area on left and abstract detector/beam art on right",
                "body_layout": "five major modules on a pale scientific background with one dominant key-result figure card that preserves source aspect",
                "color_grammar": "primary result = CMS blue; secondary interpretation = magenta/purple; limits/results = gold and black accents",
            },
            {
                "decorative_art_constraints": [
                    "The title/header artwork must be abstract detector or beamline art only.",
                    "Do not draw Feynman diagrams in decorative areas.",
                    "Do not put particle labels such as mu, nu, j, q, W, N in decorative header artwork.",
                    "Do not add physics arrows, interaction vertices, or process diagrams outside labeled placeholders.",
                    "Only [FIG 01]-style placeholders may contain diagram slots.",
                ],
                "forbidden_phrases": [
                    "internal workflow",
                    "production workflow",
                    "production-process",
                    "replacement",
                    "validated source",
                    "placeholder explanation",
                ],
            },
        )
    return (
        {},
        {
            "summary": "premium CERN/LHCC-inspired scientific poster, modern editorial HEP design, artistic but readable, not a collage",
            "aspect": "A0 vertical / 2:3 ratio",
            "top_band": "strong title banner with concise identity text and abstract scientific artwork",
            "body_layout": "4-6 large numbered modules with one dominant result region, generous gutters, and varied card shapes",
            "color_grammar": "one primary accent color for headline results and one secondary accent color for contrasts",
        },
        {
            "decorative_art_constraints": [
                "Decorative artwork must remain abstract.",
                "Do not draw fake plots, fake tables, or unlabeled scientific diagrams outside placeholders.",
            ],
            "forbidden_phrases": ["internal workflow", "production workflow", "production-process", "replacement", "placeholder explanation"],
        },
    )


def _apply_spec_extras(spec: dict[str, Any], extras: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    if extras.get("decorative_art_constraints"):
        existing = [str(item) for item in out.get("decorative_art_constraints") or []]
        for item in extras["decorative_art_constraints"]:
            if str(item) not in existing:
                existing.append(str(item))
        out["decorative_art_constraints"] = existing
    if extras.get("forbidden_phrases"):
        existing = [str(item) for item in out.get("forbidden_phrases") or []]
        low = {item.lower() for item in existing}
        for item in extras["forbidden_phrases"]:
            text = str(item)
            if text.lower() not in low:
                existing.append(text)
                low.add(text.lower())
        out["forbidden_phrases"] = existing
    if out.get("conclusion"):
        out["conclusion"] = [
            cleaned
            for cleaned in (
                sanitize_public_text(str(item), out.get("forbidden_phrases") or []).strip()
                for item in out.get("conclusion") or []
            )
            if cleaned
        ]
    return out


def _guess_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        if 12 <= len(line) <= 180:
            return line
    return "Untitled Scientific Poster"


def _guess_abstract(text: str) -> str:
    low = text.lower()
    idx = low.find("abstract")
    if idx >= 0:
        snippet = text[idx:idx+1600]
    else:
        snippet = text[:1600]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    sentences = re.split(r"(?<=[.!?])\s+", snippet)
    picked = " ".join(sentences[:2])
    return picked[:700]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poster-harness", description="Article-to-poster placeholder harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a project workspace and starter spec")
    p.add_argument("project_dir")
    p.add_argument("--title")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("init-config", help="write a strict harness configuration template")
    p.add_argument("--out", required=True)
    p.add_argument("--login", action="store_true", help="after writing config, run the interactive OpenAI/ChatGPT login bootstrap")
    p.add_argument("--auth-dir", help="directory for the generated account-auth JSON")
    p.add_argument("--auth-file", help="explicit output path for the generated account-auth JSON")
    p.add_argument("--callback-port", type=int, default=DEFAULT_CALLBACK_PORT)
    p.add_argument("--login-timeout", type=int, default=900)
    p.add_argument("--no-browser", action="store_true", help="print the login URL instead of opening a browser")
    p.add_argument("--force-auth", action="store_true", help="overwrite an existing auth JSON if the login output path already exists")
    p.set_defaults(func=cmd_init_config)

    p = sub.add_parser("login", help="interactive browser login that writes a local account-auth JSON")
    p.add_argument("--auth-dir", help="directory for the generated account-auth JSON")
    p.add_argument("--auth-file", help="explicit output path for the generated account-auth JSON")
    p.add_argument("--callback-port", type=int, default=DEFAULT_CALLBACK_PORT)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--no-browser", action="store_true", help="print the login URL instead of opening a browser")
    p.add_argument("--force", action="store_true", help="overwrite an existing auth JSON if the output path already exists")
    p.set_defaults(func=cmd_login)

    p = sub.add_parser("resolve-arxiv", help="use LLM web_search to resolve a query to an arXiv paper")
    p.add_argument("--query")
    p.add_argument("--arxiv-id")
    p.add_argument("--out", required=True)
    p.add_argument("--config")
    p.add_argument("--model")
    p.set_defaults(func=cmd_resolve_arxiv)

    p = sub.add_parser("extract", help="extract text and figure inventory from PDF/PPTX/TXT")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--render-pages", action="store_true")
    p.add_argument("--extract-images", action="store_true", help="also extract embedded raster images from PDFs")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--max-pages", type=int)
    p.add_argument("--max-images", type=int)
    p.add_argument("--min-image-width", type=int, default=96)
    p.add_argument("--min-image-height", type=int, default=96)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("draft-spec", help="make a conservative starter poster spec from extracted text")
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title")
    p.add_argument("--topic")
    p.add_argument("--authors")
    p.set_defaults(func=cmd_draft_spec)

    p = sub.add_parser("prompt", help="render image-generation prompt from poster spec")
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("generate", help="generate placeholder poster via configured image backend")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--basename", default="poster-placeholder-layout")
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--size", default="1024x1536")
    p.add_argument("--quality", default="high")
    p.add_argument("--account", default="", help="optional ChatGPT account email; empty auto-discovers local auth file")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--config", help="YAML/JSON harness config; defaults to POSTER_HARNESS_CONFIG or built-in strict config")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("replace", help="replace placeholder boxes with real assets using spec placements")
    p.add_argument("--base-image", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--asset-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_replace)

    p = sub.add_parser("upscale", help="deterministic high-resolution review export")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--factor", type=float, default=2.0)
    p.add_argument("--no-sharpen", action="store_true")
    p.set_defaults(func=cmd_upscale)

    p = sub.add_parser("sanitize", help="remove internal/workflow lines from text")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--spec")
    p.set_defaults(func=cmd_sanitize)

    p = sub.add_parser("manifest", help="scan/copy image assets and write an assets_manifest file")
    p.add_argument("--assets-dir", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--copy-to")
    p.add_argument("--max-assets", type=int)
    p.add_argument("--min-width", type=int, default=48)
    p.add_argument("--min-height", type=int, default=48)
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--no-contact-sheet", action="store_true")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("llm-draft-spec", help="draft a poster spec from source text via the LLM stage")
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--assets-manifest")
    p.add_argument("--project-json")
    p.add_argument("--style-json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config")
    p.set_defaults(func=cmd_llm_draft_spec)

    p = sub.add_parser("llm-select-figures", help="select and map figure assets onto poster placeholders")
    p.add_argument("--text", required=True)
    p.add_argument("--assets-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--spec")
    p.add_argument("--max-figures", type=int)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config")
    p.set_defaults(func=cmd_llm_select_figures)

    p = sub.add_parser("llm-detect-placeholders", help="detect placeholder bounding boxes in a rendered poster image")
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    expected = p.add_mutually_exclusive_group()
    expected.add_argument("--spec")
    expected.add_argument("--expected-json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config")
    p.set_defaults(func=cmd_llm_detect_placeholders)

    p = sub.add_parser("llm-qa", help="QA a poster spec and optional rendered image/detections")
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image")
    p.add_argument("--prompt")
    p.add_argument("--detections")
    p.add_argument("--mode", choices=["placeholder", "final"], default="placeholder")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config")
    p.set_defaults(func=cmd_llm_qa)

    p = sub.add_parser("autoposter", help="one-command paper/assets → spec → prompt → optional generation/replacement pipeline")
    p.add_argument("--config", help="YAML/JSON harness config; defaults to POSTER_HARNESS_CONFIG or built-in strict config")
    p.add_argument("--query", help="natural-language paper query; LLM uses web_search to resolve arXiv")
    p.add_argument("--arxiv-id", help="known arXiv id; skips search but still downloads arXiv paper/source")
    p.add_argument("--paper", help="local PDF/PPTX/TXT/MD/TEX source")
    p.add_argument("--text-source", help="explicit rich text source; use this instead of implicit source switching")
    p.add_argument("--out", help="run directory; defaults to config paths.runs_dir plus paper/arXiv id")
    p.add_argument("--assets-dir", action="append", default=[], help="extra directory of real figure assets; can be repeated")
    p.add_argument("--style", help="style preset from config, e.g. cms-hep or generic")
    p.add_argument("--variants", type=int, help="number of image-generation variants")
    p.add_argument("--max-figures", type=int)
    p.add_argument("--max-assets", type=int)
    p.add_argument("--min-image-width", type=int)
    p.add_argument("--min-image-height", type=int)
    p.add_argument("--dpi", type=int)
    p.add_argument("--max-pages", type=int)
    p.add_argument("--auto-assets", dest="auto_assets", action="store_true", default=None, help="infer sibling/source figure directories")
    p.add_argument("--no-auto-assets", dest="auto_assets", action="store_false")
    p.add_argument("--recursive-assets", dest="recursive_assets", action="store_true", default=None)
    p.add_argument("--no-recursive-assets", dest="recursive_assets", action="store_false")
    p.add_argument("--render-pages", dest="render_pages", action="store_true", default=None)
    p.add_argument("--no-render-pages", dest="render_pages", action="store_false")
    p.add_argument("--extract-pdf-images", dest="extract_pdf_images", action="store_true", default=None)
    p.add_argument("--no-extract-pdf-images", dest="extract_pdf_images", action="store_false")
    p.add_argument("--keep-unselected-placeholders", action="store_true", default=None)
    p.add_argument("--min-detection-confidence", type=float)
    p.add_argument("--model", help="LLM model for spec/selection/detection/QA stages")
    p.add_argument("--search-model", help="LLM model for arXiv web_search resolution")
    p.add_argument("--image-model", help="local image-generation model")
    p.add_argument("--size")
    p.add_argument("--quality")
    p.add_argument("--account", help="ChatGPT account email for image_generation.backend=chatgpt_account")
    p.add_argument("--basename")
    p.add_argument("--upscale-factor", type=float)
    p.set_defaults(func=cmd_autoposter)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
