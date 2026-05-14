from __future__ import annotations

from typing import Any, Mapping, Sequence

from poster_harness.flowchart_rewrite import apply_flowchart_rewrites, rewrite_flowcharts_from_paper
from poster_harness.llm_stages import FLOWCHART_NODE_RULES, draft_spec_from_text, paper_content_outline_from_text


class FakeProvider:
    def __init__(self, responses: Mapping[str, Mapping[str, Any]]):
        self._responses = dict(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        stage_name: str,
        prompt: str,
        schema: Mapping[str, Any],
        system_prompt: str | None = None,
        image_paths: Sequence[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.calls.append({"stage": stage_name, "prompt": prompt, "schema": schema, "system_prompt": system_prompt})
        if stage_name not in self._responses:
            raise AssertionError(f"unexpected stage call: {stage_name}")
        return {"result": dict(self._responses[stage_name]), "raw": ""}


def test_flowchart_node_rules_demand_concrete_content() -> None:
    joined = "\n".join(FLOWCHART_NODE_RULES).lower()
    assert "information capsule" in joined
    assert "preselection" in joined
    assert "pt>30" in joined
    assert "skip the flowchart" in joined


def test_draft_spec_prompt_contains_flowchart_node_rules() -> None:
    captured: dict[str, str] = {}

    class Provider(FakeProvider):
        def generate_json(self, **kw: Any) -> dict[str, Any]:
            captured["prompt"] = kw["prompt"]
            return super().generate_json(**kw)

    provider = Provider(
        {
            "draft_spec_from_text": {
                "project": {"title": "t"},
                "style": {},
                "sections": [],
                "placeholders": [],
                "conclusion": [],
            }
        }
    )
    draft_spec_from_text("paper text", provider=provider)
    prompt = captured["prompt"].lower()
    assert "information capsule" in prompt
    assert "branch" in prompt
    assert "delta phi" in prompt


def test_content_outline_stage_normalizes_dynamic_sections_and_facts() -> None:
    provider = FakeProvider(
        {
            "paper_content_outline_from_text": {
                "paper_type": "HEP result paper",
                "dynamic_sections": [
                    {
                        "title": "Signal and control regions",
                        "purpose": "Show SR/CR strategy",
                        "must_include_facts": ["SR binned in pTmiss"],
                        "specialist_details": ["WZ CR constrains normalization"],
                        "figure_links": ["limit.png", "missing.png"],
                    }
                ],
                "high_density_facts": [
                    {
                        "fact": "Profile-likelihood CLs limit",
                        "section_hint": "fit",
                        "priority": "must",
                        "evidence": "paper says CLs",
                        "render_as": "fit_chip",
                    }
                ],
                "essential_formulas": [
                    {"formula": "|V_muN|^2", "meaning": "mixing", "priority": "should", "evidence": "paper formula"}
                ],
                "figure_text_guidance": [
                    {
                        "asset": "limit.png",
                        "communicates": "limit curve",
                        "nearby_text": "Observed 95% CL limit",
                        "priority": "must",
                    }
                ],
                "coverage_priorities": ["fit/result coverage"],
            }
        }
    )
    env = paper_content_outline_from_text(
        "paper text",
        assets_manifest={"assets": [{"asset": "limit.png", "label": "Limit"}]},
        provider=provider,
    )
    result = env["result"]
    assert provider.calls[0]["stage"] == "paper_content_outline_from_text"
    assert result["paper_type"] == "hep_result_paper"
    assert result["dynamic_sections"][0]["figure_links"] == ["limit.png"]
    assert result["high_density_facts"][0]["render_as"] == "fit_chip"
    assert result["figure_text_guidance"][0]["nearby_text"] == "Observed 95% CL limit"


def test_rewrite_flowcharts_skips_when_no_section_has_flowchart() -> None:
    provider = FakeProvider({})
    env = rewrite_flowcharts_from_paper({"sections": [{"id": 1, "title": "t"}]}, "paper text", provider=provider)
    assert env.get("_skipped") is True
    assert provider.calls == []


def test_rewrite_flowcharts_replaces_generic_nodes_with_evidenced_concrete_nodes() -> None:
    spec = {
        "sections": [
            {"id": 2, "title": "Method", "flowchart": ["pp collisions", "selection", "fit"], "text": []},
        ]
    }
    provider = FakeProvider(
        {
            "flowchart_rewrite": {
                "sections": [
                    {
                        "section_id": 2,
                        "nodes": [
                            {"label": "Run 2 138 fb-1 pp at 13 TeV", "evidence": "138 fb-1 at 13 TeV"},
                            {"label": ">=2 same-sign muons: pT>30 GeV, |eta|<2.4", "evidence": "pT>30 GeV"},
                            {"label": "HMN SR Delta phi(ll) | Weinberg SR pTmiss", "evidence": "signal regions"},
                        ],
                    }
                ]
            }
        }
    )
    env = rewrite_flowcharts_from_paper(spec, "paper mentions 138 fb-1 and pT>30 GeV", provider=provider)
    assert provider.calls[0]["stage"] == "flowchart_rewrite"
    assert "Good:" in provider.calls[0]["prompt"]
    new_spec = apply_flowchart_rewrites(spec, env)
    assert new_spec["sections"][0]["flowchart"] == [
        "Run 2 138 fb-1 pp at 13 TeV",
        ">=2 same-sign muons: pT>30 GeV, |eta|<2.4",
        "HMN SR Delta phi(ll) | Weinberg SR pTmiss",
    ]
    assert spec["sections"][0]["flowchart"] == ["pp collisions", "selection", "fit"]


def test_apply_flowchart_rewrites_drops_unevidenced_and_unknown_nodes() -> None:
    spec = {"sections": [{"id": 2, "title": "x", "flowchart": ["old"], "text": []}]}
    env = {
        "result": {
            "sections": [
                {
                    "section_id": 2,
                    "nodes": [
                        {"label": "hallucinated", "evidence": ""},
                        {"label": "real pT>30 GeV", "evidence": "paper quote"},
                    ],
                },
                {"section_id": 99, "nodes": [{"label": "ghost", "evidence": "x"}]},
            ]
        }
    }
    new_spec = apply_flowchart_rewrites(spec, env)
    assert new_spec["sections"][0]["flowchart"] == ["real pT>30 GeV"]
    assert new_spec["_flowchart_rewrite"]["rewritten_sections"][0]["section_id"] == 2


def test_apply_flowchart_rewrites_clears_section_when_all_nodes_unevidenced() -> None:
    spec = {"sections": [{"id": 2, "title": "x", "flowchart": ["selection", "fit"], "text": []}]}
    env = {
        "result": {
            "sections": [
                {
                    "section_id": 2,
                    "nodes": [{"label": "generic replacement", "evidence": ""}],
                }
            ]
        }
    }
    new_spec = apply_flowchart_rewrites(spec, env)
    assert new_spec["sections"][0]["flowchart"] == []
    assert new_spec["_flowchart_rewrite"]["rewritten_sections"][0]["cleared"] is True
