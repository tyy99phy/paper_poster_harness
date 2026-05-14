from poster_harness.config import dump_config, load_autoposter_config, normalize_content_mode


def test_standard_content_mode_keeps_main_defaults(monkeypatch):
    monkeypatch.delenv("POSTER_HARNESS_CONFIG", raising=False)

    config = load_autoposter_config()

    assert config["autoposter"]["content_mode"] == "standard"
    assert config["autoposter"]["content_outline"]["enabled"] is False
    assert config["autoposter"]["copy_deck"]["max_units"] == 34
    assert "text_density" not in config["styles"]["cms-hep"]["style"]
    assert "12-20 short bullets/fact chips" in config["styles"]["cms-hep"]["style"]["information_density"]


def test_hep_dense_content_mode_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("POSTER_HARNESS_CONFIG", raising=False)
    path = tmp_path / "config.yaml"
    dump_config({"autoposter": {"content_mode": "hep_dense"}}, path)

    config = load_autoposter_config(path)

    assert config["autoposter"]["content_mode"] == "hep_dense"
    assert config["autoposter"]["content_outline"]["enabled"] is True
    assert config["autoposter"]["copy_deck"]["max_units"] == 42
    assert "text_density" in config["styles"]["cms-hep"]["style"]
    assert "18-28 short bullets/fact chips" in config["styles"]["cms-hep"]["style"]["information_density"]


def test_config_overrides_win_after_selected_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("POSTER_HARNESS_CONFIG", raising=False)
    path = tmp_path / "config.yaml"
    dump_config(
        {
            "autoposter": {
                "content_mode": "hep_dense",
                "copy_deck": {"max_units": 50},
            }
        },
        path,
    )

    config = load_autoposter_config(path)

    assert config["autoposter"]["content_mode"] == "hep_dense"
    assert config["autoposter"]["content_outline"]["enabled"] is True
    assert config["autoposter"]["copy_deck"]["max_units"] == 50


def test_cli_content_mode_override_selects_dense_overlay(monkeypatch, tmp_path):
    monkeypatch.delenv("POSTER_HARNESS_CONFIG", raising=False)
    path = tmp_path / "config.yaml"
    dump_config({"autoposter": {"content_mode": "standard"}}, path)

    config = load_autoposter_config(path, content_mode="regen2")

    assert config["autoposter"]["content_mode"] == "hep_dense"
    assert config["autoposter"]["content_outline"]["enabled"] is True
    assert config["autoposter"]["copy_deck"]["max_units"] == 42


def test_content_mode_aliases():
    assert normalize_content_mode("main") == "standard"
    assert normalize_content_mode("p2p-content-density") == "hep_dense"
    assert normalize_content_mode("regen2") == "hep_dense"
