from __future__ import annotations

import argparse
import copy
import shutil
import re
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

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
    deep_merge,
    load_config,
    dump_config,
    load_autoposter_config,
    load_harness_config,
    write_default_harness_config,
)
from .extract import extract_pdf_images, extract_text, extract_pptx_media, render_pdf_pages
from .flowchart_rewrite import apply_flowchart_rewrites, rewrite_flowcharts_from_paper
from .latex_utils import clean_latex_inline, extract_latex_braced
from .layout_contract import build_layout_contract
from .micro_repair import apply_micro_repairs
from .prompt import build_prompt, sanitize_public_text
from .replace import audit_generated_placeholder_geometry, audit_figure_containment, normalize_placeholder_geometry, replace_placeholders, upscale_image
from .image_backend import edit_image_from_config, generate_images_from_config
from .llm import ChatGPTAccountResponsesProvider, OpenAICompatibleResponsesProvider
from .llm_stages import (
    copy_deck_from_text,
    critique_poster_template,
    detect_placeholders_from_image,
    draft_spec_from_text,
    paper_content_outline_from_text,
    paper_domain_profile_from_text,
    physics_quiz_from_text,
    qa_poster,
    select_figures,
    storyboard_from_text,
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
    extract_text(inp, text_path)
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


def cmd_micro_repair(args: argparse.Namespace) -> None:
    plan = load_config(args.plan)
    repairs = plan.get("repairs") if isinstance(plan, Mapping) else None
    if not isinstance(repairs, list):
        raise SystemExit("--plan must contain a top-level repairs: list")
    out = apply_micro_repairs(
        image_path=args.input,
        out_path=args.out,
        repairs=repairs,
        scale=args.scale,
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


def cmd_llm_storyboard(args: argparse.Namespace) -> None:
    text = _read_text_file(args.text)
    spec = _load_spec_arg(args.spec) if args.spec else None
    envelope = storyboard_from_text(
        text,
        assets_manifest=_load_optional_config(args.assets_manifest),
        spec=spec,
        provider=_llm_provider(args),
        extra_instructions=args.extra_instructions or "",
    )
    _dump_llm_result(envelope, args.out)
    print(args.out)


def cmd_llm_content_outline(args: argparse.Namespace) -> None:
    text = _read_text_file(args.text)
    envelope = paper_content_outline_from_text(
        text,
        assets_manifest=_load_optional_config(args.assets_manifest),
        provider=_llm_provider(args),
        project_overrides=_load_mapping_arg(args.project_json, root_key="project"),
        max_sections=args.max_sections,
        max_facts=args.max_facts,
        max_formulas=args.max_formulas,
        extra_instructions=args.extra_instructions or "",
    )
    _dump_llm_result(envelope, args.out)
    print(args.out)


def cmd_llm_domain_profile(args: argparse.Namespace) -> None:
    text = _read_text_file(args.text)
    config = load_harness_config(args.config)
    profiles = list((cfg_get(config, "domain_profiles", {}) or {}).keys())
    classifier_cfg = cfg_get(config, "autoposter.domain_classifier", {}) or {}
    envelope = paper_domain_profile_from_text(
        text,
        assets_manifest=_load_optional_config(args.assets_manifest),
        provider=_llm_provider(args),
        arxiv_metadata=_load_optional_config(args.arxiv_metadata),
        available_profiles=profiles,
        max_text_chars=args.max_text_chars or int(cfg_get(config, "autoposter.domain_classifier.max_text_chars", 12000) or 12000),
        extra_instructions=args.extra_instructions or str((classifier_cfg or {}).get("extra_instructions") or ""),
    )
    _dump_llm_result(envelope, args.out)
    print(args.out)


def cmd_llm_select_figures(args: argparse.Namespace) -> None:
    text = _read_text_file(args.text)
    spec = _load_spec_arg(args.spec) if args.spec else None
    storyboard = _load_optional_config(args.storyboard) if getattr(args, "storyboard", None) else None
    envelope = select_figures(
        text,
        _load_required_config(args.assets_manifest, "--assets-manifest"),
        spec=spec,
        storyboard=storyboard,
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


def cmd_llm_template_critic(args: argparse.Namespace) -> None:
    envelope = critique_poster_template(
        _load_spec_arg(args.spec),
        prompt=_read_prompt_arg(args.prompt) or "",
        image_path=args.image,
        provider=_llm_provider(args),
        extra_instructions=args.extra or "",
    )
    _dump_llm_result(envelope, args.out)
    print(args.out)


def cmd_llm_rewrite_flowcharts(args: argparse.Namespace) -> None:
    spec = _load_spec_arg(args.spec)
    text = Path(args.text).read_text(encoding="utf-8")
    envelope = rewrite_flowcharts_from_paper(
        spec,
        text,
        provider=_llm_provider(args),
        text_char_limit=args.text_char_limit,
        extra_instructions=args.extra or "",
    )
    if args.raw_out:
        dump_config(envelope, args.raw_out)
    new_spec = apply_flowchart_rewrites(spec, envelope, require_evidence=not args.allow_unevidenced)
    dump_config(new_spec, args.out)
    print(args.out)


def cmd_autoposter(args: argparse.Namespace) -> None:
    config = load_autoposter_config(args.config, content_mode=getattr(args, "content_mode", None))
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
        text_source = Path(text_source_arg) if text_source_arg else (arxiv_bundle.main_tex or arxiv_bundle.pdf_path)
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

    domain_profile: dict[str, Any] | None = None
    domain_profile_path: Path | None = None
    domain_profile = _resolve_domain_profile(
        text=text,
        assets_manifest=assets_manifest,
        provider=provider,
        config=config,
        dirs=dirs,
        args=args,
        resolution=resolution,
        arxiv_bundle=arxiv_bundle,
    )
    if domain_profile:
        domain_profile_path = dirs["specs"] / "domain_profile.yaml"
        dump_config(domain_profile, domain_profile_path)
        domain_project, domain_style, domain_extras = _domain_profile_preset(
            str(domain_profile.get("domain_profile") or "generic"),
            config,
            detected=domain_profile,
        )
        project_overrides = deep_merge(project_overrides, domain_project)
        style_overrides = deep_merge(style_overrides, domain_style)
        spec_extras = _merge_spec_extras(spec_extras, domain_extras)

    content_outline: dict[str, Any] | None = None
    content_outline_path: Path | None = None
    outline_cfg = cfg_get(config, "autoposter.content_outline", {})
    outline_enabled = True
    outline_extra = ""
    outline_max_sections = int(cfg_get(config, "autoposter.content_outline.max_sections", 6) or 6)
    outline_max_facts = int(cfg_get(config, "autoposter.content_outline.max_facts", 28) or 28)
    outline_max_formulas = int(cfg_get(config, "autoposter.content_outline.max_formulas", 6) or 6)
    if isinstance(outline_cfg, Mapping):
        outline_enabled = bool(outline_cfg.get("enabled", True))
        outline_extra = str(outline_cfg.get("extra_instructions") or "")
        outline_max_sections = int(outline_cfg.get("max_sections") or outline_max_sections)
        outline_max_facts = int(outline_cfg.get("max_facts") or outline_max_facts)
        outline_max_formulas = int(outline_cfg.get("max_formulas") or outline_max_formulas)
    elif outline_cfg is not None:
        outline_enabled = bool(outline_cfg)
    if outline_enabled:
        outline_envelope = _call_llm_stage_with_retries(
            "paper_content_outline_from_text",
            config,
            paper_content_outline_from_text,
            text,
            assets_manifest=assets_manifest,
            provider=provider,
            project_overrides=project_overrides,
            max_sections=outline_max_sections,
            max_facts=outline_max_facts,
            max_formulas=outline_max_formulas,
            extra_instructions=outline_extra,
        )
        content_outline = dict(outline_envelope["result"])
        content_outline_path = dirs["specs"] / "content_outline.yaml"
        dump_config(content_outline, content_outline_path)

    draft_envelope = _call_llm_stage_with_retries(
        "draft_spec_from_text",
        config,
        draft_spec_from_text,
        text,
        assets_manifest=assets_manifest,
        provider=provider,
        project_overrides=project_overrides,
        style_overrides=style_overrides,
        content_outline=content_outline,
        domain_profile=domain_profile,
    )
    draft_spec = _apply_spec_extras(dict(draft_envelope["result"]), spec_extras)
    draft_spec_path = dirs["specs"] / "poster_spec.draft.yaml"
    dump_config(draft_spec, draft_spec_path)

    storyboard: dict[str, Any] | None = None
    storyboard_path: Path | None = None
    storyboard_cfg = cfg_get(config, "autoposter.storyboard", {})
    storyboard_enabled = True
    storyboard_extra = ""
    if isinstance(storyboard_cfg, Mapping):
        storyboard_enabled = bool(storyboard_cfg.get("enabled", True))
        storyboard_extra = str(storyboard_cfg.get("extra_instructions") or "")
    elif storyboard_cfg is not None:
        storyboard_enabled = bool(storyboard_cfg)
    if storyboard_enabled:
        storyboard_envelope = _call_llm_stage_with_retries(
            "storyboard_from_text",
            config,
            storyboard_from_text,
            text,
            assets_manifest=assets_manifest,
            spec=draft_spec,
            content_outline=content_outline,
            domain_profile=domain_profile,
            provider=provider,
            extra_instructions=storyboard_extra,
        )
        storyboard = dict(storyboard_envelope["result"])
        storyboard_path = dirs["specs"] / "storyboard.yaml"
        dump_config(storyboard, storyboard_path)
        draft_spec["storyboard"] = storyboard
    if content_outline:
        draft_spec["content_outline"] = content_outline

    physics_quiz: dict[str, Any] | None = None
    physics_quiz_path: Path | None = None
    quiz_cfg = cfg_get(config, "autoposter.physics_quiz", {})
    quiz_enabled = True
    quiz_extra = ""
    quiz_max = int(cfg_get(config, "autoposter.physics_quiz.max_questions", 16) or 16)
    if isinstance(quiz_cfg, Mapping):
        quiz_enabled = bool(quiz_cfg.get("enabled", True))
        quiz_extra = str(quiz_cfg.get("extra_instructions") or "")
        quiz_max = int(quiz_cfg.get("max_questions") or quiz_max)
    elif quiz_cfg is not None:
        quiz_enabled = bool(quiz_cfg)
    if quiz_enabled:
        quiz_envelope = _call_llm_stage_with_retries(
            "physics_quiz_from_text",
            config,
            physics_quiz_from_text,
            text,
            assets_manifest=assets_manifest,
            spec=draft_spec,
            storyboard=storyboard,
            content_outline=content_outline,
            domain_profile=domain_profile,
            provider=provider,
            max_questions=quiz_max,
            extra_instructions=quiz_extra,
        )
        physics_quiz = dict(quiz_envelope["result"])
        physics_quiz_path = dirs["specs"] / "physics_quiz.yaml"
        dump_config(physics_quiz, physics_quiz_path)
        draft_spec["physics_quiz"] = physics_quiz

    selection_envelope = _call_llm_stage_with_retries(
        "select_figures",
        config,
        select_figures,
        text,
        assets_manifest,
        spec=draft_spec,
        storyboard=storyboard,
        content_outline=content_outline,
        domain_profile=domain_profile,
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
    if storyboard:
        final_spec["storyboard"] = storyboard
    if physics_quiz:
        final_spec["physics_quiz"] = physics_quiz
    if content_outline:
        final_spec["content_outline"] = content_outline
    if domain_profile:
        final_spec["domain_profile"] = domain_profile
    final_spec = _apply_spec_extras(final_spec, spec_extras)
    copy_deck: dict[str, Any] | None = None
    copy_deck_path: Path | None = None
    copy_cfg = cfg_get(config, "autoposter.copy_deck", {})
    copy_enabled = True
    copy_extra = ""
    copy_max = int(cfg_get(config, "autoposter.copy_deck.max_units", 28) or 28)
    if isinstance(copy_cfg, Mapping):
        copy_enabled = bool(copy_cfg.get("enabled", True))
        copy_extra = str(copy_cfg.get("extra_instructions") or "")
        copy_max = int(copy_cfg.get("max_units") or copy_max)
    elif copy_cfg is not None:
        copy_enabled = bool(copy_cfg)
    if copy_enabled:
        copy_envelope = _call_llm_stage_with_retries(
            "copy_deck_from_text",
            config,
            copy_deck_from_text,
            text,
            assets_manifest=assets_manifest,
            spec=final_spec,
            storyboard=storyboard,
            physics_quiz=physics_quiz,
            figure_selection=figure_selection,
            content_outline=content_outline,
            domain_profile=domain_profile,
            provider=provider,
            max_units=copy_max,
            extra_instructions=copy_extra,
        )
        copy_deck = dict(copy_envelope["result"])
        copy_deck_path = dirs["specs"] / "copy_deck.yaml"
        dump_config(copy_deck, copy_deck_path)
        final_spec["copy_deck"] = copy_deck
    flowchart_rewrite_path: Path | None = None
    flowchart_cfg = cfg_get(config, "autoposter.flowchart_rewrite", {})
    flowchart_enabled = True
    flowchart_extra = ""
    flowchart_limit = int(cfg_get(config, "autoposter.flowchart_rewrite.text_char_limit", 24000) or 24000)
    if isinstance(flowchart_cfg, Mapping):
        flowchart_enabled = bool(flowchart_cfg.get("enabled", True))
        flowchart_extra = str(flowchart_cfg.get("extra_instructions") or "")
        flowchart_limit = int(flowchart_cfg.get("text_char_limit") or flowchart_limit)
    elif flowchart_cfg is not None:
        flowchart_enabled = bool(flowchart_cfg)
    if flowchart_enabled:
        flowchart_envelope = _call_llm_stage_with_retries(
            "flowchart_rewrite",
            config,
            rewrite_flowcharts_from_paper,
            final_spec,
            text,
            provider=provider,
            text_char_limit=flowchart_limit,
            extra_instructions=flowchart_extra,
        )
        flowchart_rewrite_path = dirs["specs"] / "flowchart_rewrite.yaml"
        dump_config(flowchart_envelope, flowchart_rewrite_path)
        final_spec = apply_flowchart_rewrites(final_spec, flowchart_envelope)
    layout_contract_path: Path | None = None
    if bool(cfg_get(config, "autoposter.layout_contract.enabled", True)):
        contract_w, contract_h = _image_size_from_config(args.size or cfg_get(config, "image_generation.size", "1024x1536"))
        layout_contract = build_layout_contract(
            final_spec,
            canvas_width=contract_w,
            canvas_height=contract_h,
        )
        final_spec["layout_contract"] = layout_contract
        layout_contract_path = dirs["specs"] / "layout_contract.yaml"
        dump_config(layout_contract, layout_contract_path)
    spec_path = dirs["specs"] / "poster_spec.yaml"
    dump_config(final_spec, spec_path)

    prompt = build_prompt(final_spec)
    prompt_path = dirs["prompts"] / "poster_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    run_manifest: dict[str, Any] = {
        "paper": str(paper),
        "out": str(root),
        "style": style_name,
        "content_mode": str(cfg_get(config, "autoposter.content_mode", "standard")),
        "domain_profile": str(domain_profile_path) if domain_profile_path else "",
        "domain_profile_name": str(domain_profile.get("domain_profile") if domain_profile else ""),
        "config": str(args.config) if args.config else "",
        "arxiv": resolution or {},
        "arxiv_bundle": {
            "metadata": str(arxiv_bundle.metadata_path),
            "source_dir": str(arxiv_bundle.source_dir),
            "main_tex": str(arxiv_bundle.main_tex) if arxiv_bundle.main_tex else "",
        } if arxiv_bundle else {},
        "text_source": str(text_source),
        "source_roots": [str(p) for p in source_roots],
        "assets_manifest": str(assets_manifest_path),
        "content_outline": str(content_outline_path) if content_outline_path else "",
        "draft_spec": str(draft_spec_path),
        "storyboard": str(storyboard_path) if storyboard_path else "",
        "physics_quiz": str(physics_quiz_path) if physics_quiz_path else "",
        "figure_selection": str(figure_selection_path),
        "copy_deck": str(copy_deck_path) if copy_deck_path else "",
        "flowchart_rewrite": str(flowchart_rewrite_path) if flowchart_rewrite_path else "",
        "layout_contract": str(layout_contract_path) if layout_contract_path else "",
        "poster_spec": str(spec_path),
        "prompt": str(prompt_path),
        "generated": [],
        "exports": [],
        "qa": [],
    }
    if not assets_manifest.get("assets"):
        raise RuntimeError("autoposter: no usable image assets found; strict mode will not continue without real figure assets")

    required_successes = max(
        1,
        int(_opt(getattr(args, "poster_sets", None), config, "autoposter.required_successes", 2) or 2),
    )
    max_candidate_batches = max(1, int(cfg_get(config, "autoposter.max_candidate_batches", 3) or 3))
    run_manifest["required_successes"] = required_successes
    run_manifest["max_candidate_batches"] = max_candidate_batches
    run_manifest["poster_sets"] = []
    run_manifest.setdefault("generated_candidates", [])
    generated_scale = float(cfg_get(config, "image_generation.generated_scale", 1.0) or 1.0)

    placeholder_failures: list[str] = []
    template_failures: list[str] = []
    candidate_index = 0
    base_basename = args.basename or f"{paper.stem}-placeholder-layout"
    try:
        for batch_index in range(max_candidate_batches):
            if len(run_manifest["poster_sets"]) >= required_successes:
                break
            batch_args = copy.copy(args)
            if batch_index > 0:
                batch_args.basename = f"{base_basename}-batch{batch_index + 1}"
            generated, prompt, batch_template_failures = _generate_templates_with_critic(
                prompt=prompt,
                final_spec=final_spec,
                paper_stem=paper.stem,
                dirs=dirs,
                config=config,
                args=batch_args,
                provider=provider,
                run_manifest=run_manifest,
                root=root,
            )
            template_failures.extend(batch_template_failures)
            if not generated:
                detail = "; ".join(batch_template_failures) if batch_template_failures else "no image-generation template passed template critic"
                placeholder_failures.append(f"batch {batch_index + 1}: {detail}")
                continue
            run_manifest["generated_candidates"].extend(str(p) for p in generated)
            for image_path in generated:
                if len(run_manifest["poster_sets"]) >= required_successes:
                    break
                candidate_index += 1
                candidate_exports = _process_generated_template_candidate(
                    image_path=Path(image_path),
                    index=candidate_index,
                    final_spec=final_spec,
                    prompt=prompt,
                    dirs=dirs,
                    config=config,
                    args=args,
                    provider=provider,
                    generated_scale=generated_scale,
                    run_manifest=run_manifest,
                    placeholder_failures=placeholder_failures,
                )
                if not candidate_exports:
                    continue
                poster_set = {
                    "index": len(run_manifest["poster_sets"]) + 1,
                    "template": str(image_path),
                    "exports": candidate_exports,
                }
                run_manifest["poster_sets"].append(poster_set)
                run_manifest["exports"].extend(candidate_exports)
    except Exception:
        dump_config(run_manifest, root / "run_manifest.yaml")
        raise

    if len(run_manifest["poster_sets"]) < required_successes:
        run_manifest_path = root / "run_manifest.yaml"
        dump_config(run_manifest, run_manifest_path)
        details = [*template_failures, *placeholder_failures]
        detail = "; ".join(details[-12:]) if details else "no exportable generated image"
        raise RuntimeError(
            f"autoposter: collected only {len(run_manifest['poster_sets'])}/{required_successes} "
            f"strict-QA poster sets ({detail})"
        )

    run_manifest_path = root / "run_manifest.yaml"
    run_manifest["generated"] = [str(item["template"]) for item in run_manifest["poster_sets"]]
    dump_config(run_manifest, run_manifest_path)
    print(run_manifest_path)
    for poster_set in run_manifest["poster_sets"]:
        print(f"poster_set_{poster_set['index']}: {poster_set['template']}")
        for item in poster_set["exports"]:
            print(item)


def _process_generated_template_candidate(
    *,
    image_path: Path,
    index: int,
    final_spec: Mapping[str, Any],
    prompt: str,
    dirs: Mapping[str, Path],
    config: Mapping[str, Any],
    args: argparse.Namespace,
    provider: ChatGPTAccountResponsesProvider,
    generated_scale: float,
    run_manifest: dict[str, Any],
    placeholder_failures: list[str],
    geometry_repair_depth: int = 0,
) -> list[str] | None:
    """Run detection, replacement, and QA for one generated placeholder layout."""
    stem = Path(image_path).stem
    effective_generated_scale = _effective_generated_scale_for_layout(image_path, generated_scale, config)
    detection_envelope = _call_llm_stage_with_retries(
        "detect_placeholders_from_image",
        config,
        detect_placeholders_from_image,
        image_path,
        expected_placeholders=final_spec.get("placeholders") or [],
        provider=provider,
    )
    detections = dict(detection_envelope["result"])
    detections_path = dirs["scratch"] / f"{stem}.detections.yaml"
    dump_config(detections, detections_path)

    spec_with_placements = apply_detections_to_spec(
        copy.deepcopy(dict(final_spec)),
        detections,
        min_confidence=float(_opt(args.min_detection_confidence, config, "autoposter.min_detection_confidence", 0.15)),
    )
    layout_contract_issues = list(spec_with_placements.get("_layout_contract_issues") or [])
    if layout_contract_issues and bool(cfg_get(config, "autoposter.layout_contract.reject_misaligned", True)):
        layout_contract_qa_path = dirs["qa"] / f"{stem}.layout-contract.qa.yaml"
        dump_config({"passes": False, "issues": layout_contract_issues}, layout_contract_qa_path)
        run_manifest["qa"].append(str(layout_contract_qa_path))
        placeholder_failures.append(
            f"{image_path}: failed deterministic layout-contract QA; see {layout_contract_qa_path}"
        )
        return None
    geometry_issues = audit_generated_placeholder_geometry(
        base_image=image_path,
        spec=spec_with_placements,
        ratio_tolerance=float(cfg_get(config, "autoposter.placeholder_aspect_tolerance", 0.20)),
    )
    if geometry_issues:
        geometry_qa = {"passes": False, "issues": geometry_issues}
        geometry_issue_path = dirs["qa"] / f"{stem}.placeholder-geometry.qa.yaml"
        dump_config(geometry_qa, geometry_issue_path)
        run_manifest["qa"].append(str(geometry_issue_path))
        repaired_layout = _attempt_placeholder_geometry_micro_repair(
            image_path=image_path,
            geometry_qa=geometry_qa,
            final_spec=final_spec,
            prompt=prompt,
            dirs=dirs,
            config=config,
            args=args,
            run_manifest=run_manifest,
            repair_depth=geometry_repair_depth,
        )
        if repaired_layout is not None:
            return _process_generated_template_candidate(
                image_path=repaired_layout,
                index=index,
                final_spec=final_spec,
                prompt=prompt,
                dirs=dirs,
                config=config,
                args=args,
                provider=provider,
                generated_scale=generated_scale,
                run_manifest=run_manifest,
                placeholder_failures=placeholder_failures,
                geometry_repair_depth=geometry_repair_depth + 1,
            )
        placeholder_failures.append(
            f"{image_path}: failed deterministic placeholder geometry QA; see {geometry_issue_path}"
        )
        return None
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

    placeholder_qa_envelope = _call_llm_stage_with_retries(
        "qa_poster_placeholder",
        config,
        qa_poster,
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
    if not _qa_result_passes_strict(placeholder_qa_result, qa_mode="placeholder"):
        placeholder_failures.append(
            f"{image_path}: failed strict placeholder QA; see {placeholder_qa_path}"
        )
        return None

    export_result = _export_realfigures_from_layout(
        stem=stem,
        layout_path=Path(image_path),
        spec_with_placements=spec_with_placements,
        dirs=dirs,
        config=config,
        args=args,
        generated_scale=effective_generated_scale,
        run_manifest=run_manifest,
        placeholder_failures=placeholder_failures,
    )
    if export_result is None:
        return None
    candidate_exports, qa_image = export_result

    qa_envelope = _call_llm_stage_with_retries(
        "qa_poster_final",
        config,
        qa_poster,
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
    if not _qa_result_passes_strict(qa_result, qa_mode="final"):
        repaired_exports = _attempt_final_micro_repair(
            stem=stem,
            final_spec=spec_with_placements,
            prompt=prompt,
            dirs=dirs,
            config=config,
            provider=provider,
            qa_result=qa_result,
            candidate_exports=[Path(item) for item in candidate_exports],
            layout_base_path=Path(image_path),
            args=args,
            generated_scale=effective_generated_scale,
            run_manifest=run_manifest,
            placeholder_failures=placeholder_failures,
        )
        if repaired_exports:
            return [str(item) for item in repaired_exports]
        for failed_export in candidate_exports:
            try:
                Path(failed_export).unlink(missing_ok=True)
            except Exception:
                pass
        failed_target = candidate_exports[0] if candidate_exports else str(image_path)
        placeholder_failures.append(
            f"{failed_target}: failed strict final QA; see {qa_path}"
        )
        return None
    return candidate_exports


def _export_realfigures_from_layout(
    *,
    stem: str,
    layout_path: Path,
    spec_with_placements: Mapping[str, Any],
    dirs: Mapping[str, Path],
    config: Mapping[str, Any],
    args: argparse.Namespace,
    generated_scale: float,
    run_manifest: dict[str, Any],
    placeholder_failures: list[str],
) -> tuple[list[str], Path] | None:
    """Paste real source figures into a placeholder layout and build exports."""
    export_path = dirs["exports"] / f"{stem}-realfigures.png"
    # Use the visible generated/edited layout as the final visual base.  Any
    # geometry normalization is a hidden replacement plan, not a second visible
    # placeholder layer; this avoids normalized white boxes that do not align
    # with the model's original card art.
    try:
        replace_placeholders(
            base_image=layout_path,
            spec=spec_with_placements,
            asset_dir=dirs["assets"],
            out_path=export_path,
        )
    except Exception as exc:
        placeholder_failures.append(
            f"{layout_path}: failed deterministic replacement; {exc}"
        )
        return None

    # Run deterministic containment audit on the replacement plan.
    containment_issues = audit_figure_containment(spec=spec_with_placements)
    if containment_issues:
        containment_path = dirs["qa"] / f"{stem}.containment.qa.yaml"
        dump_config({"passes": False, "issues": containment_issues}, containment_path)
        run_manifest["qa"].append(str(containment_path))
        placeholder_failures.append(
            f"{layout_path}: failed deterministic figure containment QA; see {containment_path}"
        )
        export_path.unlink(missing_ok=True)
        return None

    candidate_exports = [str(export_path)]
    qa_image = export_path
    upscale_factor = float(_opt(args.upscale_factor, config, "image_generation.upscale_factor", 4.0) or 0)
    if upscale_factor and upscale_factor > 1:
        upscaled_path = dirs["exports"] / f"{stem}-realfigures-{int(upscale_factor)}x.png"
        if generated_scale >= upscale_factor:
            _link_or_copy(export_path, upscaled_path)
        else:
            # For the production high-res export, upscale the generated/edited
            # template first and paste real scientific figures directly at the
            # target coordinates. This keeps paper figures sharper than
            # enlarging the already-composited result.
            extra_scale = upscale_factor / max(1.0, generated_scale)
            upscaled_base_path = dirs["generated"] / f"{stem}-production-base-{int(upscale_factor)}x.png"
            upscale_image(layout_path, upscaled_base_path, factor=extra_scale)
            try:
                replace_placeholders(
                    base_image=upscaled_base_path,
                    spec=spec_with_placements,
                    asset_dir=dirs["assets"],
                    out_path=upscaled_path,
                    scale=extra_scale,
                )
            except Exception as exc:
                placeholder_failures.append(
                    f"{layout_path}: failed deterministic high-resolution replacement; {exc}"
                )
                export_path.unlink(missing_ok=True)
                return None
        candidate_exports.append(str(upscaled_path))
        qa_image = upscaled_path
    return candidate_exports, qa_image


def _attempt_placeholder_geometry_micro_repair(
    *,
    image_path: Path,
    geometry_qa: Mapping[str, Any],
    final_spec: Mapping[str, Any],
    prompt: str,
    dirs: Mapping[str, Path],
    config: Mapping[str, Any],
    args: argparse.Namespace,
    run_manifest: dict[str, Any],
    repair_depth: int,
) -> Path | None:
    repair_cfg = cfg_get(dict(config), "autoposter.micro_repair", {}) or {}
    if isinstance(repair_cfg, Mapping):
        enabled = bool(repair_cfg.get("enabled", True))
        backend = str(repair_cfg.get("backend") or "image_edit").strip().lower()
        max_rounds = int(repair_cfg.get("placeholder_geometry_max_rounds") or repair_cfg.get("max_rounds") or 1)
        extra = str(repair_cfg.get("extra_instructions") or "")
    else:
        enabled = bool(repair_cfg)
        backend = "image_edit"
        max_rounds = 1
        extra = ""
    if not enabled or backend not in {"image_edit", "image_generation_edit", "generation_edit"}:
        return None
    if repair_depth >= max_rounds:
        return None
    issues = [item for item in geometry_qa.get("issues") or [] if isinstance(item, Mapping)]
    if not issues:
        return None
    stem = image_path.stem
    round_index = repair_depth + 1
    repair_prompt = _build_placeholder_geometry_image_edit_prompt(
        geometry_qa=geometry_qa,
        spec=final_spec,
        extra_instructions=extra,
    )
    prompt_path = dirs["prompts"] / f"{stem}.placeholder-geometry-microrepair{round_index}.image_edit_prompt.txt"
    prompt_path.write_text(repair_prompt, encoding="utf-8")
    run_manifest.setdefault("placeholder_geometry_micro_repairs", []).append(
        {
            "backend": "image_edit",
            "stage": "placeholder_geometry",
            "prompt": str(prompt_path),
            "source_layout": str(image_path),
            "repaired_layout": "",
        }
    )
    edit_input = _prepare_micro_repair_edit_input(
        source=image_path,
        scratch_dir=dirs["scratch"],
        stem=f"{stem}.placeholder-geometry-microrepair{round_index}",
        round_index=round_index,
        config=config,
    )
    try:
        edited_source = edit_image_from_config(
            image_path=edit_input,
            prompt=repair_prompt,
            out_dir=dirs["generated"],
            basename=f"{stem}-placeholder-geometry-microrepair{round_index}-native",
            config=config,
            model=args.image_model or cfg_get(dict(config), "image_generation.model", "gpt-5.5"),
            size=args.size or cfg_get(dict(config), "image_generation.size", "1024x1536"),
            quality=args.quality or cfg_get(dict(config), "image_generation.quality", "high"),
            account=args.account or cfg_get(dict(config), "image_generation.account.account", ""),
        )
        repaired_layout = _materialize_image_edit_layout(
            edited_source=Path(edited_source),
            previous_layout=image_path,
            out_path=dirs["generated"] / f"{stem}-placeholder-geometry-microrepair{round_index}.png",
        )
    except Exception:
        return None
    if run_manifest.get("placeholder_geometry_micro_repairs"):
        last = run_manifest["placeholder_geometry_micro_repairs"][-1]
        if isinstance(last, dict):
            last["repaired_layout"] = str(repaired_layout)
            if edit_input.resolve() != image_path.resolve():
                last["edit_input"] = str(edit_input)
    return repaired_layout


def _build_placeholder_geometry_image_edit_prompt(
    *,
    geometry_qa: Mapping[str, Any],
    spec: Mapping[str, Any],
    extra_instructions: str = "",
) -> str:
    project = spec.get("project") if isinstance(spec.get("project"), Mapping) else {}
    title = str(project.get("title") or spec.get("title") or "scientific poster")
    issues = [item for item in geometry_qa.get("issues") or [] if isinstance(item, Mapping)]
    issue_lines: list[str] = []
    for idx, issue in enumerate(issues[:8], start=1):
        fig_id = str(issue.get("id") or f"FIG {idx:02d}")
        expected = issue.get("expected_ratio")
        actual = issue.get("actual_ratio")
        msg = str(issue.get("message") or "")
        issue_lines.append(
            f"{idx}. {fig_id}: expected visible dashed-placeholder ratio {expected}:1, currently about {actual}:1. {msg}"
        )
    parts = [
        "Perform a local placeholder-geometry MICRO-REPAIR on the provided scientific poster template.",
        f"Poster title/topic: {title}",
        "",
        "The input is a placeholder-layout image BEFORE real figure replacement.",
        "Do NOT redraw the full poster. Preserve the overall design, colors, text, section cards, and all [FIG NN] IDs.",
        "Do NOT insert real or fake scientific figures. Every figure area must remain a blank dashed placeholder.",
        "",
        "Repair only these placeholder rectangle proportions:",
        *issue_lines,
        "",
        "For each listed [FIG NN], locally resize the dashed rectangle and its light figure mat so the visible placeholder box matches the expected aspect ratio. Keep the ID, label, and aspect-ratio text inside the placeholder. It is acceptable to move nearby callout text slightly within the same card to make room, but do not change the reading order or overall layout.",
        "Aspect-ratio intuition: 2.7:1 is a shallow wide strip; 1.32:1 is a modest landscape rectangle; 1.22:1 is close to square but still landscape. Avoid panoramic boxes for 1.22:1 or 1.32:1.",
        "Return the complete edited placeholder-layout poster image, not a crop.",
    ]
    if extra_instructions.strip():
        parts.extend(["", "Additional strict user instructions:", extra_instructions.strip()])
    return "\n".join(parts)


def _attempt_final_micro_repair(
    *,
    stem: str,
    final_spec: Mapping[str, Any],
    prompt: str,
    dirs: Mapping[str, Path],
    config: Mapping[str, Any],
    provider: ChatGPTAccountResponsesProvider,
    qa_result: Mapping[str, Any],
    candidate_exports: list[Path],
    layout_base_path: Path,
    args: argparse.Namespace,
    generated_scale: float,
    run_manifest: dict[str, Any],
    placeholder_failures: list[str],
) -> list[Path] | None:
    repair_cfg = cfg_get(dict(config), "autoposter.micro_repair", {}) or {}
    if isinstance(repair_cfg, Mapping):
        enabled = bool(repair_cfg.get("enabled", True))
        backend = str(repair_cfg.get("backend") or "image_edit").strip().lower()
        max_rounds = int(repair_cfg.get("max_rounds") or 1)
        extra = str(repair_cfg.get("extra_instructions") or "")
    else:
        enabled = bool(repair_cfg)
        backend = "image_edit"
        max_rounds = 1
        extra = ""
    if not enabled or max_rounds <= 0 or not _final_qa_is_micro_repairable(qa_result):
        return None
    if backend not in {"image_edit", "image_generation_edit", "generation_edit"}:
        raise RuntimeError(
            "autoposter.micro_repair only supports backend: image_edit in the strict autoposter pipeline; "
            "deterministic box patching is available only via the standalone micro-repair CLI command."
        )

    current_layout = layout_base_path
    current_exports = [Path(item) for item in candidate_exports]
    current_qa = dict(qa_result)
    produced_artifacts: list[Path] = []
    for round_index in range(1, max_rounds + 1):
        repair_prompt = _build_image_edit_micro_repair_prompt(
            final_qa=current_qa,
            spec=final_spec,
            extra_instructions=extra,
        )
        prompt_path = dirs["prompts"] / f"{stem}.microrepair{round_index}.image_edit_prompt.txt"
        prompt_path.write_text(repair_prompt, encoding="utf-8")
        run_manifest.setdefault("micro_repairs", []).append(
            {
                "backend": "image_edit",
                "stage": "layout",
                "prompt": str(prompt_path),
                "source_layout": "",
                "repaired_layout": "",
                "exports": [],
                "qa": "",
            }
        )

        edit_input = _prepare_micro_repair_edit_input(
            source=current_layout,
            scratch_dir=dirs["scratch"],
            stem=stem,
            round_index=round_index,
            config=config,
        )
        try:
            edited_source = edit_image_from_config(
                image_path=edit_input,
                prompt=repair_prompt,
                out_dir=dirs["generated"],
                basename=f"{stem}-microrepair{round_index}-layout-native",
                config=config,
                model=args.image_model or cfg_get(dict(config), "image_generation.model", "gpt-5.5"),
                size=args.size or cfg_get(dict(config), "image_generation.size", "1024x1536"),
                quality=args.quality or cfg_get(dict(config), "image_generation.quality", "high"),
                account=args.account or cfg_get(dict(config), "image_generation.account.account", ""),
            )
        except Exception:
            for item in produced_artifacts:
                item.unlink(missing_ok=True)
            return None
        produced_artifacts.append(Path(edited_source))

        try:
            repaired_layout = _materialize_image_edit_layout(
                edited_source=Path(edited_source),
                previous_layout=current_layout,
                out_path=dirs["generated"] / f"{stem}-microrepair{round_index}-layout.png",
            )
        except Exception:
            Path(edited_source).unlink(missing_ok=True)
            for item in produced_artifacts:
                item.unlink(missing_ok=True)
            return None
        produced_artifacts.append(repaired_layout)

        repaired_detection_envelope = _call_llm_stage_with_retries(
            "detect_placeholders_from_image_microrepair",
            config,
            detect_placeholders_from_image,
            repaired_layout,
            expected_placeholders=final_spec.get("placeholders") or [],
            provider=provider,
        )
        repaired_detections = dict(repaired_detection_envelope["result"])
        repaired_detections_path = dirs["scratch"] / f"{stem}.microrepair{round_index}.detections.yaml"
        dump_config(repaired_detections, repaired_detections_path)

        repaired_spec = apply_detections_to_spec(
            copy.deepcopy(dict(final_spec)),
            repaired_detections,
            min_confidence=float(_opt(args.min_detection_confidence, config, "autoposter.min_detection_confidence", 0.15)),
        )
        layout_contract_issues = list(repaired_spec.get("_layout_contract_issues") or [])
        if layout_contract_issues and bool(cfg_get(config, "autoposter.layout_contract.reject_misaligned", True)):
            layout_contract_qa_path = dirs["qa"] / f"{stem}.microrepair{round_index}.layout-contract.qa.yaml"
            dump_config({"passes": False, "issues": layout_contract_issues}, layout_contract_qa_path)
            run_manifest["qa"].append(str(layout_contract_qa_path))
            for item in produced_artifacts:
                item.unlink(missing_ok=True)
            return None
        geometry_issues = audit_generated_placeholder_geometry(
            base_image=repaired_layout,
            spec=repaired_spec,
            ratio_tolerance=float(cfg_get(config, "autoposter.placeholder_aspect_tolerance", 0.20)),
        )
        if geometry_issues:
            geometry_issue_path = dirs["qa"] / f"{stem}.microrepair{round_index}.placeholder-geometry.qa.yaml"
            dump_config({"passes": False, "issues": geometry_issues}, geometry_issue_path)
            run_manifest["qa"].append(str(geometry_issue_path))
            for item in produced_artifacts:
                item.unlink(missing_ok=True)
            return None

        repaired_placeholder_qa_image = repaired_layout
        if bool(cfg_get(config, "autoposter.normalize_placeholder_geometry", True)):
            redraw_geometry = bool(cfg_get(config, "autoposter.redraw_normalized_placeholders", False))
            repaired_placeholder_qa_image, repaired_spec = normalize_placeholder_geometry(
                base_image=repaired_layout,
                spec=repaired_spec,
                out_path=dirs["generated"] / f"{stem}-microrepair{round_index}-geometry-normalized.png",
                redraw=redraw_geometry,
            )
            produced_artifacts.append(Path(repaired_placeholder_qa_image))
            _overwrite_detection_boxes(repaired_detections, repaired_spec.get("placements") or {})
            dump_config(repaired_detections, repaired_detections_path)
        repaired_spec_path = dirs["specs"] / f"{stem}.microrepair{round_index}.with_placements.yaml"
        dump_config(repaired_spec, repaired_spec_path)

        placeholder_qa_envelope = _call_llm_stage_with_retries(
            "qa_poster_placeholder_microrepair",
            config,
            qa_poster,
            repaired_spec,
            prompt=prompt,
            image_path=repaired_placeholder_qa_image,
            detected_placeholders=repaired_detections,
            provider=provider,
            qa_mode="placeholder",
        )
        placeholder_qa_result = dict(placeholder_qa_envelope["result"])
        placeholder_qa_path = dirs["qa"] / f"{stem}.microrepair{round_index}.placeholder.qa.yaml"
        dump_config(placeholder_qa_result, placeholder_qa_path)
        run_manifest["qa"].append(str(placeholder_qa_path))
        if not _qa_result_passes_strict(placeholder_qa_result, qa_mode="placeholder"):
            for item in produced_artifacts:
                item.unlink(missing_ok=True)
            return None

        if run_manifest.get("micro_repairs"):
            last = run_manifest["micro_repairs"][-1]
            if isinstance(last, dict):
                last["source_layout"] = str(current_layout)
                last["repaired_layout"] = str(repaired_layout)
                if edit_input.resolve() != current_layout.resolve():
                    last["edit_input"] = str(edit_input)

        export_result = _export_realfigures_from_layout(
            stem=f"{stem}-microrepair{round_index}",
            layout_path=repaired_layout,
            spec_with_placements=repaired_spec,
            dirs=dirs,
            config=config,
            args=args,
            generated_scale=generated_scale,
            run_manifest=run_manifest,
            placeholder_failures=placeholder_failures,
        )
        if export_result is None:
            for item in produced_artifacts:
                item.unlink(missing_ok=True)
            return None
        repaired_exports_str, repaired_qa_image = export_result
        repaired_exports = [Path(item) for item in repaired_exports_str]
        produced_artifacts.extend(repaired_exports)
        if run_manifest.get("micro_repairs"):
            last = run_manifest["micro_repairs"][-1]
            if isinstance(last, dict):
                last["exports"] = [str(item) for item in repaired_exports]

        qa_envelope = _call_llm_stage_with_retries(
            "qa_poster_final_microrepair",
            config,
            qa_poster,
            repaired_spec,
            prompt=prompt,
            image_path=repaired_qa_image,
            detected_placeholders=repaired_detections,
            provider=provider,
            qa_mode="final",
        )
        current_qa = dict(qa_envelope["result"])
        qa_path = dirs["qa"] / f"{stem}.microrepair{round_index}.final.qa.yaml"
        dump_config(current_qa, qa_path)
        run_manifest["qa"].append(str(qa_path))
        if run_manifest.get("micro_repairs"):
            last = run_manifest["micro_repairs"][-1]
            if isinstance(last, dict):
                last["qa"] = str(qa_path)
        if _qa_result_passes_strict(current_qa, qa_mode="final"):
            for old in current_exports:
                try:
                    old.unlink(missing_ok=True)
                except Exception:
                    pass
            return repaired_exports
        current_exports = repaired_exports
        current_layout = repaired_layout
    for item in produced_artifacts:
        item.unlink(missing_ok=True)
    return None


def _build_image_edit_micro_repair_prompt(
    *,
    final_qa: Mapping[str, Any],
    spec: Mapping[str, Any],
    extra_instructions: str = "",
) -> str:
    project = spec.get("project") if isinstance(spec.get("project"), Mapping) else {}
    title = str(project.get("title") or spec.get("title") or "scientific poster")
    issues = final_qa.get("issues") if isinstance(final_qa.get("issues"), list) else []
    repairs = final_qa.get("recommended_repairs") if isinstance(final_qa.get("recommended_repairs"), list) else []
    qa_blob = " ".join(
        str(item)
        for item in [
            final_qa.get("summary"),
            *(issues or []),
            *(repairs or []),
        ]
    ).lower()
    issue_lines: list[str] = []
    for idx, issue in enumerate(issues[:8], start=1):
        if not isinstance(issue, Mapping):
            continue
        severity = str(issue.get("severity") or "issue")
        category = str(issue.get("category") or "")
        message = str(issue.get("message") or "")
        location = str(issue.get("location") or "")
        suggested = str(issue.get("suggested_fix") or "")
        line = f"{idx}. [{severity}/{category}] {message}"
        if location:
            line += f" Location: {location}."
        if suggested:
            line += f" Fix: {suggested}"
        issue_lines.append(line)
    repair_lines = [f"- {str(item)}" for item in repairs[:10] if str(item).strip()]
    if not issue_lines and not repair_lines:
        issue_lines.append("1. Correct only the small local public-text problems identified by the final QA.")

    parts = [
        "Perform a MICRO-REPAIR image edit on the provided placeholder-layout poster template.",
        f"Poster title/topic: {title}",
        "",
        "This image is the source layout BEFORE real scientific figure replacement. It intentionally contains blank [FIG NN] placeholders.",
        "Keep the same canvas size/aspect, overall layout, section cards, colors, background art, typography style, placeholder IDs, and placeholder aspect ratios.",
        "Do NOT redraw the poster from scratch. Do NOT cover areas with opaque patch rectangles. Do NOT insert real or fake scientific plots.",
        "Preserve every [FIG NN] as a blank dashed placeholder with its ID, label, and aspect-ratio text. If QA says a final real figure/frame covered public text, you may make only a tiny local layout repair within that same section: move nearby public text/callouts out from under the placeholder, or nudge/resize the blank placeholder slightly while preserving its aspect ratio and section association.",
        "",
        "Allowed edit scope: tiny local public-text/glyph corrections and tiny local placeholder/text spacing repairs requested by QA. Do not change the science story or section order.",
        "After this edit, the harness will re-detect placeholders, paste the real paper figures again, and rerun final QA.",
        "Required corrections from QA:",
        *issue_lines,
    ]
    if repair_lines:
        parts.extend(["", "Recommended repairs to apply exactly:", *repair_lines])
    parts.extend(
        [
            "",
            "Preserve evidence-vs-observation wording exactly unless QA asks to fix a typo.",
            "Return the complete edited placeholder-layout poster image, not a crop.",
        ]
    )
    if any(marker in qa_blob for marker in ("gamma", "h→z", "h->z", "mzy", "zγ", "zy")):
        parts.append(
            "QA-specific notation hint: if a Latin y is visibly substituting for Greek gamma, replace only that erroneous glyph/string with γ/gamma as requested by QA."
        )
    if any(marker in qa_blob for marker in ("m_ee", "dilepton", "mℓℓ", "mll")):
        parts.append(
            "QA-specific notation hint: if QA asks for dilepton mass notation, correct only that local text to the requested mℓℓ/mll form."
        )
    if extra_instructions.strip():
        parts.extend(["", "Additional strict user instructions:", extra_instructions.strip()])
    return "\n".join(parts)


def _prepare_micro_repair_edit_input(
    *,
    source: Path,
    scratch_dir: Path,
    stem: str,
    round_index: int,
    config: Mapping[str, Any],
) -> Path:
    max_w, max_h = _image_size_from_config(
        cfg_get(dict(config), "autoposter.micro_repair.edit_input_size", None)
        or cfg_get(dict(config), "autoposter.micro_repair.size", None)
        or cfg_get(dict(config), "image_generation.size", "1024x1536")
    )
    try:
        with Image.open(source) as im:
            width, height = im.size
            scale = min(max_w / width, max_h / height, 1.0)
            if scale >= 0.999:
                return source
            resized = im.convert("RGB").resize(
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                Image.Resampling.LANCZOS,
            )
    except Exception:
        return source
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out = scratch_dir / f"{stem}.microrepair{round_index}.edit-input.png"
    resized.save(out)
    return out


def _materialize_image_edit_layout(
    *,
    edited_source: Path,
    previous_layout: Path,
    out_path: Path,
) -> Path:
    try:
        with Image.open(edited_source) as im:
            edited_w, edited_h = im.size
    except Exception:
        edited_w, edited_h = 0, 0
    with Image.open(previous_layout) as im:
        target_w, target_h = im.size
    if edited_w <= 0 or edited_h <= 0:
        raise RuntimeError(f"cannot read edited image dimensions: {edited_source}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if (target_w, target_h) == (edited_w, edited_h):
        _link_or_copy(edited_source, out_path)
        return out_path
    factor_w = target_w / edited_w
    factor_h = target_h / edited_h
    if abs(factor_w - factor_h) > 0.03:
        raise RuntimeError(
            f"micro_repair image_edit layout aspect mismatch: edited={edited_w}x{edited_h}, target={target_w}x{target_h}"
        )
    upscale_image(edited_source, out_path, factor=factor_w)
    return out_path


def _final_qa_is_micro_repairable(qa_result: Mapping[str, Any]) -> bool:
    checks = qa_result.get("checks") if isinstance(qa_result.get("checks"), Mapping) else {}
    if checks and checks.get("placeholders_accounted_for") is False:
        return False
    critical = [
        item for item in qa_result.get("issues") or []
        if isinstance(item, Mapping) and str(item.get("severity") or "").lower() == "critical"
    ]
    if not critical:
        return True
    repairable_markers = (
        "text",
        "typo",
        "glyph",
        "notation",
        "symbol",
        "gamma",
        "public_text",
        "claim_precision",
        "scientific_claim",
    )
    blocking_markers = (
        "figure",
        "crop",
        "overlap",
        "containment",
        "placeholder",
        "missing",
        "fake_science",
        "fake_scientific_content",
        "layout",
        "geometry",
    )
    for issue in critical:
        blob = " ".join(str(issue.get(key) or "").lower() for key in ("category", "message", "suggested_fix"))
        if any(marker in blob for marker in blocking_markers) and not any(marker in blob for marker in repairable_markers):
            return False
        if not any(marker in blob for marker in repairable_markers):
            return False
    return True


def _qa_result_passes_strict(qa_result: Mapping[str, Any], *, qa_mode: str) -> bool:
    """Interpret VLM QA conservatively for automated release gating.

    Some VLM answers contain ``passes: true`` while also setting authoritative
    boolean checks such as ``public_text_clean: false`` or reporting release-
    blocking warning categories.  The harness should not accept those as final
    posters; it should run the same strict repair loop instead.
    """
    if not bool(qa_result.get("passes")):
        return False
    checks = qa_result.get("checks") if isinstance(qa_result.get("checks"), Mapping) else {}
    common_required = ("placeholders_accounted_for", "no_internal_text")
    for key in common_required:
        if checks.get(key) is False:
            return False
    if qa_mode == "placeholder":
        for key in ("placeholder_contract_clean", "no_fake_science"):
            if checks.get(key) is False:
                return False
    if qa_mode == "final":
        if checks.get("public_text_clean") is False:
            return False
        release_blocking_warning_categories = {
            "public_text_cleanliness",
            "public_text_readability",
            "text_readability",
            "text_accuracy",
            "readability",
            "unreplaced_placeholder",
            "placeholder_artifact",
            "final_cleanup_artifact",
            "figure_placement",
            "figure_overlap",
            "figure_containment",
        }
        for issue in qa_result.get("issues") or []:
            if not isinstance(issue, Mapping):
                continue
            severity = str(issue.get("severity") or "").lower()
            category = str(issue.get("category") or "").lower()
            if severity == "critical":
                return False
            if severity == "warning" and category in release_blocking_warning_categories:
                return False
    return True


def _call_llm_stage_with_retries(
    stage_name: str,
    harness_config: Mapping[str, Any],
    func,
    *args,
    **kwargs,
):
    """Retry the same LLM stage on transient transport errors.

    This is not a fallback or a weaker path: the exact same LLM stage is called
    again when the local ChatGPT/Codex transport drops a connection.
    """
    retries = max(0, int(cfg_get(dict(harness_config), "autoposter.llm_stage_retries", 2) or 0))
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries or not _looks_transient_llm_error(exc):
                raise
            delay = min(20.0, 2.5 * (attempt + 1))
            print(f"WARNING: {stage_name} transient error on attempt {attempt + 1}/{retries + 1}: {exc}; retrying in {delay:.1f}s")
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _looks_transient_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "connection reset",
            "connection aborted",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "network error",
            "urlopen error",
            "remote end closed",
            "incompleteread",
            "incomplete read",
            "405",
            "method not allowed",
            "502",
            "503",
            "504",
        )
    )


def _generate_templates_with_critic(
    *,
    prompt: str,
    final_spec: Mapping[str, Any],
    paper_stem: str,
    dirs: Mapping[str, Path],
    config: Mapping[str, Any],
    args: argparse.Namespace,
    provider: ChatGPTAccountResponsesProvider,
    run_manifest: dict[str, Any],
    root: Path,
) -> tuple[list[Path], str, list[str]]:
    critic_enabled = _opt_bool(
        getattr(args, "template_critic", None),
        config,
        "autoposter.template_critic.enabled",
        True,
    )
    max_regen_rounds = 0
    if critic_enabled:
        max_regen_rounds = int(
            _opt(
                getattr(args, "template_regens", None),
                config,
                "autoposter.template_critic.max_regen_rounds",
                1,
            )
            or 0
        )
    generated_scale = float(cfg_get(dict(config), "image_generation.generated_scale", 1.0) or 1.0)
    base_basename = args.basename or f"{paper_stem}-placeholder-layout"
    current_prompt = prompt
    template_failures: list[str] = []
    run_manifest["generated_scale"] = generated_scale
    run_manifest.setdefault("generated_all", [])
    run_manifest.setdefault("template_critiques", [])
    if critic_enabled:
        run_manifest["template_critic"] = {
            "enabled": True,
            "max_regen_rounds": max_regen_rounds,
            "require_pass": bool(cfg_get(dict(config), "autoposter.template_critic.require_pass", True)),
        }

    for round_index in range(max_regen_rounds + 1):
        round_basename = base_basename if round_index == 0 else f"{base_basename}-regen{round_index}"
        if round_index > 0:
            prompt_path = dirs["prompts"] / f"poster_prompt.regen{round_index}.txt"
            prompt_path.write_text(current_prompt, encoding="utf-8")
            run_manifest.setdefault("regen_prompts", []).append(str(prompt_path))
        generated = _call_llm_stage_with_retries(
            "image_generation",
            config,
            generate_images_from_config,
            prompt=current_prompt,
            out_dir=dirs["generated"],
            basename=round_basename,
            config=config,
            model=args.image_model or cfg_get(dict(config), "image_generation.model", "gpt-5.5"),
            size=args.size or cfg_get(dict(config), "image_generation.size", "1024x1536"),
            quality=args.quality or cfg_get(dict(config), "image_generation.quality", "high"),
            account=args.account or cfg_get(dict(config), "image_generation.account.account", ""),
            n=int(_opt(args.variants, config, "image_generation.variants", 1)),
        )
        if not generated:
            raise RuntimeError("autoposter: image generation returned no images")
        native_generated: list[Path] = []
        if generated_scale > 1:
            generated, native_generated = _promote_generated_templates_to_scale(generated, factor=generated_scale)
            if native_generated:
                run_manifest.setdefault("generated_native", []).extend(str(p) for p in native_generated)
        run_manifest["generated_all"].extend(str(p) for p in generated)

        if not critic_enabled:
            run_manifest["generated"] = [str(p) for p in generated]
            return generated, current_prompt, template_failures

        accepted: list[Path] = []
        round_repairs: list[str] = []
        for image_path in generated:
            image_path = Path(image_path)
            critic_image_path = _native_sibling_for_promoted_layout(image_path) or image_path
            stem = critic_image_path.stem
            critique_envelope = _call_llm_stage_with_retries(
                "critique_poster_template",
                config,
                critique_poster_template,
                final_spec,
                prompt=current_prompt,
                image_path=critic_image_path,
                provider=provider,
                extra_instructions=str(cfg_get(dict(config), "autoposter.template_critic.extra_instructions", "") or ""),
            )
            critique = dict(critique_envelope["result"])
            critique_path = dirs["qa"] / f"{stem}.template-critic.qa.yaml"
            dump_config(critique, critique_path)
            run_manifest["qa"].append(str(critique_path))
            run_manifest["template_critiques"].append(str(critique_path))
            if _template_critic_accepts(critique, config):
                accepted.append(critic_image_path)
            else:
                repaired = _attempt_template_critic_micro_repair(
                    image_path=critic_image_path,
                    critique=critique,
                    final_spec=final_spec,
                    prompt=current_prompt,
                    dirs=dirs,
                    config=config,
                    args=args,
                    provider=provider,
                    run_manifest=run_manifest,
                )
                if repaired is not None:
                    accepted.append(repaired)
                    continue
                if _template_critique_can_defer_to_downstream(critique):
                    run_manifest.setdefault("template_critic_deferred", []).append(
                        {
                            "image": str(critic_image_path),
                            "qa": str(critique_path),
                            "reason": "repairable template issues deferred to deterministic placeholder QA/final layout micro-repair",
                        }
                    )
                    accepted.append(critic_image_path)
                    continue
                summary = str(critique.get("summary") or "template critic rejected generated poster")
                template_failures.append(f"{image_path}: failed template critic; see {critique_path}; {summary}")
                round_repairs.extend(_template_critic_repairs(critique))
        if accepted:
            run_manifest["generated"] = [str(p) for p in accepted]
            return accepted, current_prompt, template_failures

        if round_index < max_regen_rounds:
            current_prompt = _prompt_with_template_critic_repairs(
                prompt,
                repairs=round_repairs,
                round_index=round_index + 1,
            )
            dump_config(run_manifest, root / "run_manifest.yaml")

    run_manifest["generated"] = []
    return [], current_prompt, template_failures


def _attempt_template_critic_micro_repair(
    *,
    image_path: Path,
    critique: Mapping[str, Any],
    final_spec: Mapping[str, Any],
    prompt: str,
    dirs: Mapping[str, Path],
    config: Mapping[str, Any],
    args: argparse.Namespace,
    provider: ChatGPTAccountResponsesProvider,
    run_manifest: dict[str, Any],
) -> Path | None:
    repair_cfg = cfg_get(dict(config), "autoposter.micro_repair", {}) or {}
    if isinstance(repair_cfg, Mapping):
        enabled = bool(repair_cfg.get("enabled", True))
        backend = str(repair_cfg.get("backend") or "image_edit").strip().lower()
        max_rounds = int(repair_cfg.get("template_critic_max_rounds") or repair_cfg.get("max_rounds") or 1)
        extra = str(repair_cfg.get("extra_instructions") or "")
    else:
        enabled = bool(repair_cfg)
        backend = "image_edit"
        max_rounds = 1
        extra = ""
    if not enabled or max_rounds <= 0 or backend not in {"image_edit", "image_generation_edit", "generation_edit"}:
        return None
    if not _template_critique_is_micro_repairable(critique):
        return None

    current_layout = image_path
    current_critique = dict(critique)
    for round_index in range(1, max_rounds + 1):
        stem = current_layout.stem
        repair_prompt = _build_template_image_edit_micro_repair_prompt(
            critique=current_critique,
            spec=final_spec,
            extra_instructions=extra,
        )
        prompt_path = dirs["prompts"] / f"{stem}.template-microrepair{round_index}.image_edit_prompt.txt"
        prompt_path.write_text(repair_prompt, encoding="utf-8")
        run_manifest.setdefault("template_micro_repairs", []).append(
            {
                "backend": "image_edit",
                "stage": "template_critic",
                "prompt": str(prompt_path),
                "source_layout": str(current_layout),
                "repaired_layout": "",
                "qa": "",
            }
        )
        edit_input = _prepare_micro_repair_edit_input(
            source=current_layout,
            scratch_dir=dirs["scratch"],
            stem=f"{stem}.template-microrepair{round_index}",
            round_index=round_index,
            config=config,
        )
        try:
            edited_source = edit_image_from_config(
                image_path=edit_input,
                prompt=repair_prompt,
                out_dir=dirs["generated"],
                basename=f"{stem}-template-microrepair{round_index}-native",
                config=config,
                model=args.image_model or cfg_get(dict(config), "image_generation.model", "gpt-5.5"),
                size=args.size or cfg_get(dict(config), "image_generation.size", "1024x1536"),
                quality=args.quality or cfg_get(dict(config), "image_generation.quality", "high"),
                account=args.account or cfg_get(dict(config), "image_generation.account.account", ""),
            )
            repaired_layout = _materialize_image_edit_layout(
                edited_source=Path(edited_source),
                previous_layout=current_layout,
                out_path=dirs["generated"] / f"{stem}-template-microrepair{round_index}.png",
            )
        except Exception:
            return None

        critique_envelope = _call_llm_stage_with_retries(
            "critique_poster_template_microrepair",
            config,
            critique_poster_template,
            final_spec,
            prompt=prompt,
            image_path=repaired_layout,
            provider=provider,
            extra_instructions=str(cfg_get(dict(config), "autoposter.template_critic.extra_instructions", "") or ""),
        )
        repaired_critique = dict(critique_envelope["result"])
        qa_path = dirs["qa"] / f"{repaired_layout.stem}.template-critic.qa.yaml"
        dump_config(repaired_critique, qa_path)
        run_manifest["qa"].append(str(qa_path))
        run_manifest.setdefault("template_critiques", []).append(str(qa_path))
        if run_manifest.get("template_micro_repairs"):
            last = run_manifest["template_micro_repairs"][-1]
            if isinstance(last, dict):
                last["repaired_layout"] = str(repaired_layout)
                if edit_input.resolve() != current_layout.resolve():
                    last["edit_input"] = str(edit_input)
                last["qa"] = str(qa_path)
        if _template_critic_accepts(repaired_critique, config):
            return repaired_layout
        if _template_critique_can_defer_to_downstream(repaired_critique):
            run_manifest.setdefault("template_critic_deferred", []).append(
                {
                    "image": str(repaired_layout),
                    "qa": str(qa_path),
                    "reason": "repairable template issues deferred to deterministic placeholder QA/final layout micro-repair",
                }
            )
            return repaired_layout
        if not _template_critique_is_micro_repairable(repaired_critique):
            return None
        current_layout = repaired_layout
        current_critique = repaired_critique
    return None


def _template_critique_is_micro_repairable(critique: Mapping[str, Any]) -> bool:
    issues = [item for item in critique.get("issues") or [] if isinstance(item, Mapping)]
    if not issues:
        return True
    repairable_markers = (
        "text",
        "typo",
        "glyph",
        "notation",
        "gamma",
        "placeholder_contract",
        "aspect",
        "ratio",
        "geometry",
        "spelling",
        "scientific_accuracy",
        "result-copy",
        "copy",
    )
    blocking_markers = (
        "fake_science",
        "fake scientific",
        "fake data",
        "real plot",
        "fabricated plot",
        "missing placeholder",
        "missing [fig",
        "no placeholder",
        "internal",
    )
    critical = [item for item in issues if str(item.get("severity") or "").lower() == "critical"]
    candidates = critical or issues
    for issue in candidates:
        blob = " ".join(
            str(issue.get(key) or "").lower()
            for key in ("category", "message", "suggested_prompt_repair", "location")
        )
        if any(marker in blob for marker in blocking_markers) and not any(marker in blob for marker in repairable_markers):
            return False
        if not any(marker in blob for marker in repairable_markers):
            return False
    return True


def _template_critique_can_defer_to_downstream(critique: Mapping[str, Any]) -> bool:
    if not _template_critique_is_micro_repairable(critique):
        return False
    checks = critique.get("checks") if isinstance(critique.get("checks"), Mapping) else {}
    if checks:
        if checks.get("no_internal_text") is False or checks.get("no_fake_science") is False:
            return False
        if checks.get("information_plan_visible") is False or checks.get("art_direction_strong") is False:
            return False
        if checks.get("placeholder_contract_clean") is False:
            return False
    # Do not defer placeholder geometry/aspect failures.  The regression that
    # made real figures look worse came from accepting a visually wrong
    # placeholder, then letting hidden replacement planning expand the figure
    # frame upward into public text.  Geometry problems must be fixed by
    # template image-edit/regeneration before real figures are pasted.
    for issue in critique.get("issues") or []:
        if not isinstance(issue, Mapping):
            continue
        blob = " ".join(
            str(issue.get(key) or "").lower()
            for key in ("category", "message", "suggested_prompt_repair", "location")
        )
        if any(marker in blob for marker in ("placeholder", "aspect", "ratio", "geometry", "figure card")):
            return False
    # Template critic is intentionally broad.  Only defer local public-text/glyph
    # issues that downstream strict QA can safely re-check.
    return True


def _build_template_image_edit_micro_repair_prompt(
    *,
    critique: Mapping[str, Any],
    spec: Mapping[str, Any],
    extra_instructions: str = "",
) -> str:
    project = spec.get("project") if isinstance(spec.get("project"), Mapping) else {}
    title = str(project.get("title") or spec.get("title") or "scientific poster")
    issues = [item for item in critique.get("issues") or [] if isinstance(item, Mapping)]
    repairs = _template_critic_repairs(critique)
    issue_lines: list[str] = []
    for idx, issue in enumerate(issues[:8], start=1):
        severity = str(issue.get("severity") or "issue")
        category = str(issue.get("category") or "")
        message = str(issue.get("message") or "")
        location = str(issue.get("location") or "")
        suggested = str(issue.get("suggested_prompt_repair") or "")
        line = f"{idx}. [{severity}/{category}] {message}"
        if location:
            line += f" Location: {location}."
        if suggested:
            line += f" Repair target: {suggested}"
        issue_lines.append(line)
    if not issue_lines:
        issue_lines.append("1. Repair only the small local template issues identified by the critic.")
    repair_lines = [f"- {item}" for item in repairs[:10] if item.strip()]
    qa_blob = " ".join([str(critique.get("summary") or ""), *issue_lines, *repair_lines]).lower()
    parts = [
        "Perform a MICRO-REPAIR image edit on the provided scientific poster placeholder-layout template.",
        f"Poster title/topic: {title}",
        "",
        "The input image is a generated template BEFORE real scientific figure replacement. It intentionally contains blank dashed [FIG NN] placeholders.",
        "Do NOT redraw the poster from scratch. Preserve the overall composition, colors, background art, card hierarchy, typography style, and all placeholder IDs.",
        "Do NOT insert real or fake scientific figures. Keep every figure area as a blank dashed placeholder with its [FIG NN] label and aspect-ratio text.",
        "",
        "Allowed edit scope:",
        "- Correct local public-text typos, corrupted glyphs, and notation errors.",
        "- If the critic says a placeholder aspect ratio is wrong, locally resize that dashed placeholder rectangle/card so it visibly matches the printed aspect ratio, while keeping the same [FIG NN] ID and blank-placeholder contract.",
        "- If the critic says a result value is unsupported, replace only that local text with the supplied/critic-approved value.",
        "",
        "Critic findings to repair:",
        *issue_lines,
    ]
    if repair_lines:
        parts.extend(["", "Apply these repair targets:", *repair_lines])
    if any(marker in qa_blob for marker in ("gamma", "h→z", "h->z", "zγ", "mzy", "latin y")):
        parts.append(
            "Notation hint: where the decay means H to Z gamma, render it as H→Zγ in large headings, or use the words 'Z gamma' in small labels if γ risks looking like Latin y."
        )
    if any(marker in qa_blob for marker in ("aspect", "ratio", "placeholder")):
        parts.append(
            "Placeholder hint: prioritize visible rectangle proportions over decorative symmetry; a 2.7:1 placeholder must be a shallow wide strip, while 1.22:1 and 1.32:1 placeholders should be modest landscape rectangles, not panoramic or portrait boxes."
        )
    parts.append("Return the complete edited placeholder-layout poster image, not a crop.")
    if extra_instructions.strip():
        parts.extend(["", "Additional strict user instructions:", extra_instructions.strip()])
    return "\n".join(parts)


def _template_critic_accepts(critique: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    scores = dict(critique.get("scores") or {})
    if bool(cfg_get(dict(config), "autoposter.template_critic.require_pass", True)) and not bool(critique.get("passes")):
        return False
    thresholds = {
        "overall": float(cfg_get(dict(config), "autoposter.template_critic.min_overall_score", 0.72) or 0.0),
        "artistry": float(cfg_get(dict(config), "autoposter.template_critic.min_artistry_score", 0.65) or 0.0),
        "information_density": float(cfg_get(dict(config), "autoposter.template_critic.min_information_density_score", 0.65) or 0.0),
        "placeholder_contract": float(cfg_get(dict(config), "autoposter.template_critic.min_placeholder_contract_score", 0.75) or 0.0),
    }
    for key, threshold in thresholds.items():
        if float(scores.get(key) or 0.0) < threshold:
            return False
    return True


def _template_critic_repairs(critique: Mapping[str, Any]) -> list[str]:
    repairs = [
        str(item).strip()
        for item in critique.get("prompt_repairs") or []
        if str(item).strip() and not _repair_contradicts_placeholder_contract(str(item))
    ]
    for issue in critique.get("issues") or []:
        if not isinstance(issue, Mapping):
            continue
        repair = str(issue.get("suggested_prompt_repair") or "").strip()
        if repair and repair not in repairs and not _repair_contradicts_placeholder_contract(repair):
            repairs.append(repair)
    if not repairs:
        repairs = [
            "Regenerate the whole poster with stronger editorial HEP artistry, richer compact public facts, cleaner placeholder boxes, and more legible typography."
        ]
    return repairs[:10]


def _repair_contradicts_placeholder_contract(text: str) -> bool:
    low = str(text or "").lower()
    contradiction_markers = [
        "only the centered [fig",
        "only the [fig",
        "only [fig",
        "show only the placeholder id",
        "inside each figure area show only",
        "do not render aspect",
        "do not print aspect",
        "remove aspect",
        "without aspect",
        "move all labels",
        "move labels outside",
        "public caption/headline outside them",
    ]
    return any(marker in low for marker in contradiction_markers)


def _prompt_with_template_critic_repairs(base_prompt: str, *, repairs: list[str], round_index: int) -> str:
    unique_repairs: list[str] = []
    for repair in repairs:
        clean = sanitize_public_text(str(repair)).strip()
        if clean and clean not in unique_repairs and not _repair_contradicts_placeholder_contract(clean):
            unique_repairs.append(clean)
        if len(unique_repairs) >= 10:
            break
    if not unique_repairs:
        unique_repairs = [
            "Increase information density with compact public facts and badges while keeping all scientific figures as blank placeholders.",
            "Make the poster feel more like a premium HEP conference poster and less like a slide deck.",
        ]
    repair_lines = "\n".join(f"- {item}" for item in unique_repairs)
    return (
        base_prompt.rstrip()
        + "\n\n"
        + f"REGENERATION CRITIQUE ROUND {round_index} (apply these changes to the whole poster; do not mention this critique in the poster):\n"
        + repair_lines
        + "\n- Regenerate a complete fresh poster, not a patched or cropped collage.\n"
    )


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
    if backend in {"openai_responses", "openai_compatible", "openai_compatible_responses"}:
        responses_cfg = dict(cfg_get(cfg, "llm.openai_responses", {}) or {})
        endpoint = (
            responses_cfg.get("endpoint")
            or cfg_get(cfg, "llm.endpoint", None)
            or responses_cfg.get("base_url")
            or cfg_get(cfg, "llm.base_url", None)
        )
        if endpoint and str(endpoint).rstrip("/").endswith("/v1"):
            endpoint = str(endpoint).rstrip("/") + "/responses"
        if not endpoint:
            raise RuntimeError("llm.openai_responses.endpoint or llm.endpoint is required for openai_responses backend")
        return OpenAICompatibleResponsesProvider(
            endpoint=str(endpoint),
            model=str(llm_model),
            api_key_env=str(responses_cfg.get("api_key_env") or cfg_get(cfg, "llm.api_key_env", "OPENAI_API_KEY")),
            response_format=str(responses_cfg.get("response_format") or cfg_get(cfg, "llm.response_format", "json_schema")),
            reasoning_effort=str(responses_cfg.get("reasoning_effort") or cfg_get(cfg, "llm.reasoning_effort", "") or "") or None,
            timeout=timeout,
            proxy=str(responses_cfg.get("proxy") or cfg_get(cfg, "llm.proxy", "") or "") or None,
        )

    raise RuntimeError(f"unsupported llm.backend={backend!r}. Supported: chatgpt_account")


def _opt(value: Any, config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    return value if value is not None else cfg_get(dict(config), dotted, default)


def _opt_bool(value: Any, config: Mapping[str, Any], dotted: str, default: bool) -> bool:
    return bool(value if value is not None else cfg_get(dict(config), dotted, default))


def _image_size_from_config(value: Any) -> tuple[int, int]:
    text = str(value or "").strip().lower()
    match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Codex image_generation portrait default.
    return 1024, 1536


def _native_sibling_for_promoted_layout(path: Path) -> Path | None:
    native = path.with_name(f"{path.stem}-native{path.suffix}")
    return native if native.exists() else None


def _effective_generated_scale_for_layout(
    image_path: Path,
    generated_scale: float,
    config: Mapping[str, Any],
) -> float:
    """Return the true scale of the layout image being processed.

    The harness may keep a 4x review copy for users while routing VLM/replacement
    work through the native 1024×1536 image to avoid large-image transport
    failures.  Export scaling must use the true work-layout scale, otherwise a
    native export can be mislabeled as a 4x export.
    """
    try:
        with Image.open(image_path) as im:
            width, height = im.size
    except Exception:
        return generated_scale
    native_w, native_h = _image_size_from_config(cfg_get(dict(config), "image_generation.size", "1024x1536"))
    if width <= native_w * 1.5 and height <= native_h * 1.5:
        return 1.0
    return generated_scale


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
                "body_layout": "five major modules on a pale scientific background with one dominant warm-white key-result figure card that preserves source aspect",
                "color_grammar": "primary result = CMS blue; secondary interpretation = magenta/purple; limits/results = restrained gold and black accents; every plot/diagram surface remains warm-white or very pale neutral",
                "typography": "CMS/CERN editorial sans-serif: bold condensed title, clear numbered module headers, dark readable body text on light cards, compact bullets.",
                "color_palette": "Deep navy/indigo outer atmosphere with CMS cobalt blue, violet/magenta accents, restrained amber/gold highlights, and warm-white/pearl figure cards.",
                "figure_surface": "All scientific figure areas must use warm-white or pearl card interiors; dark colors may frame figures but must not fill plot-containing blocks.",
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
            "summary": "premium domain-adaptive scientific conference poster, modern editorial design, artistic but readable, not a collage",
            "aspect": "A0 vertical / 2:3 ratio",
            "top_band": "strong title banner with concise identity text and abstract scientific artwork",
            "body_layout": "4-6 large numbered modules with one dominant result region, generous gutters, varied card shapes, and light paper-like figure cards",
            "color_grammar": "one primary accent color for headline results and one secondary accent color for contrasts; all figure surfaces remain warm-white or very pale neutral",
            "typography": "Modern editorial sans-serif with bold title, crisp section headers, compact readable bullets, and a disciplined type scale.",
            "color_palette": "Deep indigo or graphite atmosphere, one disciplined primary accent, one secondary contrast, restrained warm highlights, and warm-white content/figure cards.",
            "figure_surface": "Every scientific figure placeholder must live on a warm-white, pearl, or very pale neutral card/mat; never on a dark content block.",
        },
        {
            "decorative_art_constraints": [
                "Decorative artwork must remain abstract.",
                "Do not draw fake plots, fake tables, or unlabeled scientific diagrams outside placeholders.",
            ],
            "forbidden_phrases": ["internal workflow", "production workflow", "production-process", "replacement", "placeholder explanation"],
        },
    )


def _resolve_domain_profile(
    *,
    text: str,
    assets_manifest: Mapping[str, Any],
    provider: ChatGPTAccountResponsesProvider,
    config: Mapping[str, Any],
    dirs: Mapping[str, Path],
    args: argparse.Namespace,
    resolution: Mapping[str, Any] | None,
    arxiv_bundle: Any,
) -> dict[str, Any] | None:
    requested = str(getattr(args, "domain_profile", None) or cfg_get(dict(config), "autoposter.domain_profile", "auto") or "auto")
    normalized = _normalize_domain_profile_name(requested)
    if normalized in {"none", "off", "disabled", "false"}:
        return None
    profile_configs = cfg_get(dict(config), "domain_profiles", {}) or {}
    available = [str(key) for key in profile_configs.keys()] or ["generic"]
    if normalized != "auto":
        if normalized not in profile_configs:
            raise RuntimeError(f"unknown domain_profile={requested!r}; available: auto, " + ", ".join(sorted(available)))
        return _explicit_domain_profile(normalized, profile_configs.get(normalized) or {})

    classifier_cfg = cfg_get(dict(config), "autoposter.domain_classifier", {}) or {}
    if isinstance(classifier_cfg, Mapping) and not bool(classifier_cfg.get("enabled", True)):
        return _explicit_domain_profile("generic", profile_configs.get("generic") or {})

    arxiv_metadata: dict[str, Any] = {}
    if resolution:
        arxiv_metadata.update(dict(resolution))
    metadata_path = getattr(arxiv_bundle, "metadata_path", None) if arxiv_bundle else None
    if metadata_path:
        try:
            arxiv_metadata["downloaded_metadata"] = load_config(metadata_path)
        except Exception:
            arxiv_metadata["downloaded_metadata_path"] = str(metadata_path)

    envelope = _call_llm_stage_with_retries(
        "paper_domain_profile_from_text",
        config,
        paper_domain_profile_from_text,
        text,
        assets_manifest=assets_manifest,
        provider=provider,
        arxiv_metadata=arxiv_metadata,
        available_profiles=available,
        max_text_chars=int(cfg_get(dict(config), "autoposter.domain_classifier.max_text_chars", 12000) or 12000),
        extra_instructions=str(cfg_get(dict(config), "autoposter.domain_classifier.extra_instructions", "") or ""),
    )
    detected = dict(envelope["result"])
    detected.setdefault("mode", "auto")
    detected.setdefault("available_profiles", available)
    return detected


def _normalize_domain_profile_name(name: str) -> str:
    normalized = (name or "generic").strip().lower().replace("-", "_").replace(".", "_")
    aliases = {
        "default": "auto",
        "adaptive": "auto",
        "field_auto": "auto",
        "high_energy_physics": "hep",
        "particle_physics": "hep",
        "hep_ex": "hep",
        "hep_ph": "hep",
        "hep_th": "hep",
        "machine_learning": "cs_ml",
        "computer_science": "cs_ml",
        "cs": "cs_ml",
        "ml": "cs_ml",
        "biomed": "bio",
        "biomedical": "bio",
        "astrophysics": "astro",
        "astronomy": "astro",
        "materials": "chemistry",
        "materials_science": "chemistry",
    }
    return aliases.get(normalized, normalized)


def _explicit_domain_profile(name: str, profile_cfg: Mapping[str, Any]) -> dict[str, Any]:
    style = dict(profile_cfg.get("style") or {})
    return {
        "domain_label": str(style.get("domain_label") or name),
        "domain_profile": name,
        "confidence": 1.0,
        "arxiv_categories": [],
        "rationale": "Explicit domain profile selected by configuration or CLI.",
        "visual_grammar": _as_string_list(style.get("domain_poster_grammar") or style.get("visual_grammar") or [], limit=8),
        "figure_types": _as_string_list(style.get("figure_types") or [], limit=12),
        "layout_priorities": _as_string_list(style.get("layout_priorities") or [], limit=8),
        "text_priorities": _as_string_list([style.get("domain_text_guidance")] if style.get("domain_text_guidance") else [], limit=8),
        "cautionary_rules": _as_string_list((profile_cfg.get("extras") or {}).get("decorative_art_constraints") or [], limit=8),
        "suggested_style": "",
        "mode": "explicit",
    }


def _domain_profile_preset(
    name: str,
    config: Mapping[str, Any] | None = None,
    *,
    detected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    normalized = _normalize_domain_profile_name(name)
    profiles = cfg_get(dict(config or {}), "domain_profiles", {}) or {}
    profile_cfg = profiles.get(normalized) or profiles.get(name) or profiles.get("generic") or {}
    project = dict(profile_cfg.get("project") or {})
    style = dict(profile_cfg.get("style") or {})
    extras = dict(profile_cfg.get("extras") or {})
    if detected:
        style["domain_profile"] = str(detected.get("domain_profile") or normalized)
        style["domain_detection_summary"] = str(detected.get("rationale") or "")
        if detected.get("visual_grammar"):
            style["detected_visual_grammar"] = _as_string_list(detected.get("visual_grammar"), limit=8)
        if detected.get("layout_priorities"):
            style["detected_layout_priorities"] = _as_string_list(detected.get("layout_priorities"), limit=8)
        if detected.get("text_priorities"):
            style["detected_text_priorities"] = _as_string_list(detected.get("text_priorities"), limit=8)
        if detected.get("cautionary_rules"):
            extras = _merge_spec_extras(
                extras,
                {"decorative_art_constraints": _as_string_list(detected.get("cautionary_rules"), limit=8)},
            )
    return project, style, extras


def _as_string_list(value: Any, *, limit: int = 12) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else ([value] if value else [])
    out: list[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _merge_spec_extras(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(left or {}))
    for key, value in (right or {}).items():
        if key in {"decorative_art_constraints", "forbidden_phrases"}:
            existing = [str(item) for item in merged.get(key) or []]
            low = {item.lower() for item in existing}
            for item in value or []:
                text = str(item)
                if text.lower() not in low:
                    existing.append(text)
                    low.add(text.lower())
            merged[key] = existing
        else:
            merged[key] = deep_merge(merged.get(key), value)
    return merged


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

    p = sub.add_parser("micro-repair", help="apply local deterministic typo/text repairs to a rendered poster")
    p.add_argument("--input", required=True)
    p.add_argument("--plan", required=True, help="YAML/JSON file with repairs: [...]")
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.set_defaults(func=cmd_micro_repair)

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

    p = sub.add_parser("llm-content-outline", help="draft a P2P-style high-density content outline from source text")
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--assets-manifest")
    p.add_argument("--project-json")
    p.add_argument("--max-sections", type=int)
    p.add_argument("--max-facts", type=int)
    p.add_argument("--max-formulas", type=int)
    p.add_argument("--extra-instructions")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config")
    p.set_defaults(func=cmd_llm_content_outline)

    p = sub.add_parser("llm-domain-profile", help="classify paper domain and produce domain-adaptive prompt guidance")
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--assets-manifest")
    p.add_argument("--arxiv-metadata")
    p.add_argument("--max-text-chars", type=int)
    p.add_argument("--extra-instructions")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config")
    p.set_defaults(func=cmd_llm_domain_profile)

    p = sub.add_parser("llm-storyboard", help="draft a structured storyboard from source text and assets")
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--assets-manifest")
    p.add_argument("--spec")
    p.add_argument("--extra-instructions")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config")
    p.set_defaults(func=cmd_llm_storyboard)

    p = sub.add_parser("llm-select-figures", help="select and map figure assets onto poster placeholders")
    p.add_argument("--text", required=True)
    p.add_argument("--assets-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--spec")
    p.add_argument("--storyboard")
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

    p = sub.add_parser("llm-template-critic", help="critique a generated placeholder poster before replacement")
    p.add_argument("--spec", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt")
    p.add_argument("--extra", help="extra template critic instructions")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config")
    p.set_defaults(func=cmd_llm_template_critic)

    p = sub.add_parser("llm-rewrite-flowcharts", help="rewrite spec flowchart nodes into concrete paper-grounded analysis schematics")
    p.add_argument("--spec", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-out", help="optional path for the raw LLM rewrite envelope")
    p.add_argument("--text-char-limit", type=int, default=24000)
    p.add_argument("--extra", help="extra flowchart rewrite instructions")
    p.add_argument("--allow-unevidenced", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config")
    p.set_defaults(func=cmd_llm_rewrite_flowcharts)

    p = sub.add_parser("autoposter", help="one-command paper/assets → spec → prompt → optional generation/replacement pipeline")
    p.add_argument("--config", help="YAML/JSON harness config; defaults to POSTER_HARNESS_CONFIG or built-in strict config")
    p.add_argument("--query", help="natural-language paper query; LLM uses web_search to resolve arXiv")
    p.add_argument("--arxiv-id", help="known arXiv id; skips search but still downloads arXiv paper/source")
    p.add_argument("--paper", help="local PDF/PPTX/TXT/MD/TEX source")
    p.add_argument("--text-source", help="explicit rich text source; use this instead of implicit source switching")
    p.add_argument("--out", help="run directory; defaults to config paths.runs_dir plus paper/arXiv id")
    p.add_argument("--assets-dir", action="append", default=[], help="extra directory of real figure assets; can be repeated")
    p.add_argument("--style", help="style preset from config, e.g. cms-hep or generic")
    p.add_argument("--domain-profile", help="domain profile: auto (default), generic, hep, cs_ml, bio, astro, math, chemistry, or none")
    p.add_argument(
        "--content-mode",
        help=(
            "content planning mode: standard (default/main-compatible) or hep_dense "
            "(optional regen2/P2P-style higher information density; aliases: dense, regen2)"
        ),
    )
    p.add_argument("--variants", type=int, help="number of image-generation variants")
    p.add_argument("--poster-sets", type=int, help="number of strict-QA poster sets to collect; default 2")
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
    p.add_argument("--template-critic", dest="template_critic", action="store_true", default=None, help="enable VLM template critic before replacement")
    p.add_argument("--no-template-critic", dest="template_critic", action="store_false")
    p.add_argument("--template-regens", type=int, help="maximum full-poster regeneration rounds after template critic rejection")
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
