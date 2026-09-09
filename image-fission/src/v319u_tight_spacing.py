"""
v319u: ARMED text-only fission — FIXED layout (consistent design, tight spacing, sharp text)

User feedback (2026-09-09):
  - 小字糊在一起 (small text blurred/mushed)
  - 中间大字为什么要隔那么开 (big text over-spaced, 204px gaps in v319t)
  - 字体设计一样, 风格参考就行 (font design should be SAME as original, just reference style)
  - 排版正常排版 (normal layout)
  - 别糊字乱字字体乱排 (no blur/garble/messed layout)
  - 没有的内容去git里学模型跟技能 (go to git for skills/fonts)

Key insights:
  1. Original uses ONE font design across all 3 bands. v319t used Arial Narrow
     for small band (different design) — violated "字体设计一样".
     Fix: use Black Ops One (military display) for ALL bands.
  2. Small text "WE HONOR THE" in Black Ops One at capH=51 has nat_w=611,
     band=517. Scaling to fit compresses letters → overlap. Fix: use shorter
     phrase that fits naturally with the font's own kerning.
  3. PIL draw.text y-coordinate is FONT CELL TOP, not cap top. Black Ops One
     bbox_A[1]=21 (at fsize=78), so cap top = y + 21. For cap top at y0_band,
     y = y0_band - 21.
  4. v319t used per-char draw+paste → anti-alias fringe artifacts on small
     text. Fix: whole-word draw.text() + 3x supersample + LANCZOS downscale.
  5. STEEL (5 letters) in Black Ops One at capH=99 has nat_w=573, band=1322.
     204px gaps looked "乱排". Fix: pick 8-10 letter words that fill band
     with tight spacing (STRONGHOLD, WARRIORS).

Font: Black Ops One (Google Fonts, OFL) — best free military display font
      (from AnyText en.json catalog at ComfyUI_Anytext/.../multi_fonts/).
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v319s_perband_lama_bigmargin import (
    SRC, OUT_DIR as _OUT_DIR, BANDS as _BANDS_LIST,
)
SOURCE = str(SRC)
OUT_LAMA = str(_OUT_DIR / "armed_lama_cleaned.png")
OUT_DIR = str(_OUT_DIR)
BANDS = {b["name"]: b for b in _BANDS_LIST}

FONTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
FONT_PATH = os.path.join(FONTDIR, "BlackOpsOne-Regular.ttf")

# All bands use Black Ops One for consistent design
# Words chosen to fit bands with tight natural spacing (no huge gaps, no overlap)
BAND_TEXT = {
    "small": ("NEVER QUIT",   78),   # nat_w=488, band=517, gap=4px  — fits naturally
    "big1":  ("STRONGHOLD",  152),  # nat_w=1071, band=1322, gap=28px — fits naturally
    "big2":  ("WARRIORS",    152),  # nat_w=870, band=917, gap=9.4px  — fits naturally
}

SUPERSAMPLE = 3


def render_band_text(word, fsize, target_w, target_cap_h, color, ss=SUPERSAMPLE):
    """Render word at given fsize, optionally with small horizontal scale to fill target_w.

    Returns (layer_rgba, final_cap_h) ready to paste, or None on failure.
    The cap top of the returned layer is at the top edge of the image.
    """
    font = ImageFont.truetype(FONT_PATH, fsize)
    nat_w = int(round(font.getlength(word)))
    bbox_A = font.getbbox("A")
    cap_h = bbox_A[3] - bbox_A[1]
    cap_top_offset = bbox_A[1]  # offset from cell top to cap top

    # Compute inter-letter gap to fill target_w (tight spacing)
    n = len(word)
    gap = max(0, int(round((target_w - nat_w) / max(1, n - 1)))) if n > 1 else 0

    # Render at supersampled resolution for sharp edges
    fsize_ss = fsize * ss
    font_ss = ImageFont.truetype(FONT_PATH, fsize_ss)
    bbox_A_ss = font_ss.getbbox("A")
    cap_h_ss = bbox_A_ss[3] - bbox_A_ss[1]
    cap_top_offset_ss = bbox_A_ss[1]

    # Pad to ensure cap top has room
    pad = int(abs(cap_top_offset_ss)) + 20 * ss
    layer_w = int(target_w * ss) + pad * 2
    layer_h = fsize_ss + pad * 2
    layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # Place word so cap top is at y=pad
    d.text((pad, pad - cap_top_offset_ss), word, font=font_ss, fill=color, spacing=gap * ss)

    # Crop to tight bbox
    bbox = layer.getbbox()
    if bbox is None:
        return None
    layer = layer.crop(bbox)

    # Downscale to 1x
    layer_1x = layer.resize(
        (layer.width // ss, layer.height // ss), Image.LANCZOS
    )

    # The cap top is at the top of the layer (we placed it at y=pad and cropped)
    return layer_1x, layer_1x.height, nat_w, gap


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bg = Image.open(OUT_LAMA).convert("RGB")
    out = bg.copy()

    print(f"v319u: consistent design (Black Ops One all bands) + tight spacing + 3x SS")
    print(f"  BG: {OUT_LAMA} {bg.size}")
    print()

    for band_key, (word, fsize) in BAND_TEXT.items():
        band = BANDS[band_key]
        y0, y1 = band["y0"], band["y1"]
        band_h = y1 - y0
        orig_cx = band["orig_cx"]
        target_w = band["orig_w"]

        result = render_band_text(word, fsize, target_w, band_h, (20, 20, 20), ss=SUPERSAMPLE)
        if result is None:
            print(f"  {band_key}: FAILED")
            continue
        layer, layer_h, nat_w, gap = result
        lw, lh = layer.size

        # Position: center on orig_cx, cap top at y0 (band top)
        paste_x = orig_cx - lw // 2
        paste_y = y0  # cap top at band top

        out.paste(layer, (paste_x, paste_y), layer)
        pct = 100.0 * lw / target_w
        print(f"  {band_key}: '{word}' fsize={fsize} nat_w={nat_w} gap={gap} "
              f"-> w={lw}/{target_w} ({pct:.1f}%) h={lh} x={paste_x} y={paste_y}")

    out_path = os.path.join(OUT_DIR, "armed_text_only_v319u.jpg")
    out.save(out_path, quality=92)
    print(f"\n  Saved: {out_path}")

    # Band comparison
    src = Image.open(SOURCE).convert("RGB")
    cmp_w = 1300
    panels = []
    for label, img in [("ORIGINAL", src), ("CLEANED v319s (LaMa)", bg), ("FINAL v319u", out)]:
        crop = img.crop((130, 460, 1430, 770))
        crop = crop.resize((cmp_w, int(crop.height * cmp_w / crop.width)))
        panel = Image.new("RGB", (cmp_w, crop.height + 30), "white")
        d = ImageDraw.Draw(panel)
        d.text((10, 5), label, fill="red")
        panel.paste(crop, (0, 30))
        panels.append(panel)
    total_h = sum(p.height for p in panels)
    compare = Image.new("RGB", (cmp_w, total_h), "white")
    y = 0
    for p in panels:
        compare.paste(p, (0, y))
        y += p.height
    cmp_path = os.path.join(OUT_DIR, "armed_text_band_compare_v319u.jpg")
    compare.save(cmp_path, quality=88)
    print(f"  Saved: {cmp_path}")


if __name__ == "__main__":
    main()

