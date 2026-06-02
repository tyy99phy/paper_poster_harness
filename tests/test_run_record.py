from __future__ import annotations

import json
from pathlib import Path

import yaml

from poster_harness.cli import main
from poster_harness.run_record import build_record, render_markdown, write_record


def _dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "fake_run"
    for sub in ["assets", "exports", "generated", "input", "prompts", "qa", "scratch", "specs"]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    (run / "assets" / "Figure_001.png").write_bytes(b"not-a-real-image-but-a-real-asset-file")
    (run / "exports" / "poster.png").write_bytes(b"poster")
    (run / "generated" / "paper-placeholder-native.png").write_bytes(b"template")
    (run / "input" / "extracted_text.txt").write_text("source paper text", encoding="utf-8")
    (run / "prompts" / "poster_prompt.txt").write_text("prompt", encoding="utf-8")

    out = "runs/fake_run"
    stem = "paper-placeholder-native"
    _dump(run / "specs" / "domain_profile.yaml", {"domain_profile": "hep"})
    _dump(
        run / "specs" / "figure_selection.yaml",
        {
            "selected_figures": [
                {
                    "placeholder_id": "FIG 01",
                    "asset": "Figure_001.png",
                    "source_path": f"{out}/assets/Figure_001.png",
                    "label": "Limit plot | with pipe",
                    "aspect": "1:1 square",
                    "role": "hero_result",
                }
            ]
        },
    )
    _dump(
        run / "scratch" / f"{stem}.detections.yaml",
        {
            "image_size": {"width": 100, "height": 100},
            "placeholders": [{"id": "FIG 01", "bbox": [0, 0, 100, 100], "confidence": 0.97}],
            "placements": {"FIG 01": [0, 0, 100, 100]},
        },
    )
    _dump(
        run / "qa" / f"{stem}.template-critic.qa.yaml",
        {"passes": True, "summary": "template OK", "scores": {"overall": 0.91}, "issues": []},
    )
    _dump(run / "qa" / f"{stem}.placeholder.qa.yaml", {"passes": True, "summary": "placeholders OK"})
    _dump(
        run / "qa" / f"{stem}.final.qa.yaml",
        {
            "passes": True,
            "summary": "final OK",
            "score": 0.88,
            "checks": {"public_text_clean": True, "placeholders_accounted_for": True, "section_count": 4},
        },
    )
    _dump(
        run / "run_manifest.yaml",
        {
            "paper": f"{out}/input/paper.pdf",
            "out": out,
            "content_mode": "standard",
            "domain_profile": f"{out}/specs/domain_profile.yaml",
            "domain_profile_name": "hep",
            "text_source": f"{out}/input/extracted_text.txt",
            "figure_selection": f"{out}/specs/figure_selection.yaml",
            "copy_deck": f"{out}/specs/copy_deck.yaml",
            "storyboard": f"{out}/specs/storyboard.yaml",
            "physics_quiz": f"{out}/specs/physics_quiz.yaml",
            "poster_spec": f"{out}/specs/poster_spec.yaml",
            "prompt": f"{out}/prompts/poster_prompt.txt",
            "required_successes": 1,
            "max_candidate_batches": 1,
            "generated_all": [f"{out}/generated/{stem}.png"],
            "generated_candidates": [f"{out}/generated/{stem}.png"],
            "template_critiques": [f"{out}/qa/{stem}.template-critic.qa.yaml"],
            "qa": [
                f"{out}/qa/{stem}.template-critic.qa.yaml",
                f"{out}/qa/{stem}.placeholder.qa.yaml",
                f"{out}/qa/{stem}.final.qa.yaml",
            ],
            "poster_sets": [
                {
                    "index": 1,
                    "template": f"{out}/generated/{stem}.png",
                    "exports": [f"{out}/exports/poster.png"],
                }
            ],
            "exports": [f"{out}/exports/poster.png"],
        },
    )
    return run


def test_run_record_builds_json_and_markdown(tmp_path: Path) -> None:
    run = _make_run(tmp_path)

    record = build_record(run)
    markdown = render_markdown(record)

    assert record["schema_version"] == "1.2"
    assert record["objective_scoreboard"]["provenance"] == "1/1"
    assert record["objective_scoreboard"]["aspect_contract"] == "1/1"
    assert record["objective_scoreboard"]["public_text_clean"] is True
    assert len(record["qa_reports"]) == 3
    assert "Limit plot \\| with pipe" in markdown
    assert "## QA / failure accounting" in markdown
    assert "## Artifact index" in markdown
    assert "## Complete workflow trace" in markdown
    assert "prompt" in {item["role"] for item in record["workflow_artifacts"]}

    json_path, md_path = write_record(run)
    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_id"] == "fake_run"


def test_record_cli_writes_outputs(tmp_path: Path, capsys) -> None:
    run = _make_run(tmp_path)

    assert main(["record", str(run)]) == 0

    out = capsys.readouterr().out
    assert "run_record.json" in out
    assert (run / "run_record.json").exists()
    assert (run / "run_record.md").exists()
