from pathlib import Path

from PIL import Image, ImageDraw

from poster_harness.micro_repair import apply_micro_repairs


def test_micro_repair_box_text_patch_is_local(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (260, 120), (245, 245, 240))
    draw = ImageDraw.Draw(im)
    draw.text((40, 40), "bad text", fill=(10, 10, 10))
    untouched = im.getpixel((10, 10))
    im.save(base)

    out = tmp_path / "out.png"
    apply_micro_repairs(
        image_path=base,
        out_path=out,
        repairs=[
            {
                "type": "text_patch",
                "box": [36, 34, 210, 78],
                "erase": "box",
                "fill": [245, 245, 240],
                "text": "good text",
                "font_size": 18,
                "color": [0, 0, 0],
                "padding": [4, 4],
            }
        ],
    )
    result = Image.open(out).convert("RGB")
    assert result.getpixel((10, 10)) == untouched
    assert result.getpixel((45, 45)) != im.getpixel((45, 45))


def test_micro_repair_glyph_mask_does_not_paint_whole_box(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (180, 90), (10, 20, 40))
    draw = ImageDraw.Draw(im)
    draw.text((40, 30), "VBP", fill=(245, 245, 245))
    untouched = im.getpixel((20, 20))
    im.save(base)

    out = tmp_path / "out.png"
    apply_micro_repairs(
        image_path=base,
        out_path=out,
        repairs=[
            {
                "type": "glyph_patch",
                "box": [88, 26, 128, 60],
                "erase": "text_mask",
                "text_threshold": "light",
                "fill": [10, 20, 40],
                "text": "F",
                "font_size": 18,
                "color": [245, 245, 245],
                "padding": [0, 0],
            }
        ],
    )
    result = Image.open(out).convert("RGB")
    assert result.getpixel((20, 20)) == untouched
    # The patch should introduce light pixels in the local glyph box while
    # preserving the dark background outside it.
    crop = result.crop((88, 26, 128, 60))
    assert any(sum(pixel) / 3 > 180 for pixel in crop.getdata())
