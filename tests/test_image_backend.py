from pathlib import Path

from PIL import Image

from poster_harness.image_backend import build_image_request_body, encode_image_as_data_url


def test_image_edit_request_body_includes_input_image(tmp_path: Path):
    image_path = tmp_path / "poster.png"
    Image.new("RGB", (4, 6), "white").save(image_path)

    body = build_image_request_body(
        prompt="fix typo only",
        model="gpt-5.5",
        size="auto",
        quality="high",
        input_images=[image_path],
        instructions="edit only",
    )

    assert body["instructions"] == "edit only"
    assert body["tools"] == [{"type": "image_generation", "quality": "high"}]
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "fix typo only"}
    assert content[1]["type"] == "input_image"
    assert content[1]["detail"] == "high"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_encode_image_as_data_url_uses_mime_type(tmp_path: Path):
    image_path = tmp_path / "poster.jpg"
    Image.new("RGB", (2, 2), "white").save(image_path)

    assert encode_image_as_data_url(image_path).startswith("data:image/jpeg;base64,")

from poster_harness.cli import _materialize_image_edit_layout, _prepare_micro_repair_edit_input


def test_micro_repair_edit_input_is_downsampled_to_config_size(tmp_path: Path):
    source = tmp_path / "poster-4x.png"
    Image.new("RGB", (4096, 6144), "white").save(source)

    edit_input = _prepare_micro_repair_edit_input(
        source=source,
        scratch_dir=tmp_path,
        stem="poster",
        round_index=1,
        config={"image_generation": {"size": "1024x1536"}},
    )

    assert edit_input != source
    assert Image.open(edit_input).size == (1024, 1536)


def test_materialize_image_edit_layout_restores_target_size(tmp_path: Path):
    previous = tmp_path / "poster-4x.png"
    edited_native = tmp_path / "poster-imageedit1-native.png"
    Image.new("RGB", (4096, 6144), "white").save(previous)
    Image.new("RGB", (1024, 1536), "white").save(edited_native)

    repaired_layout = _materialize_image_edit_layout(
        edited_source=edited_native,
        previous_layout=previous,
        out_path=tmp_path / "poster-microrepair1-layout.png",
    )

    assert repaired_layout.name == "poster-microrepair1-layout.png"
    assert Image.open(repaired_layout).size == (4096, 6144)
