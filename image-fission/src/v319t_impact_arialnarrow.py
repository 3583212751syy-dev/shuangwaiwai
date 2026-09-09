"""v319t (v2): ARMED text-only fission — condensed font + width-matched words.

KEY FIX from v319t v1: d.text() anchors at TOP of font cell, not baseline.
  - Draw at y=0 in a tall layer, crop to bbox
  - Crop top = cap top, crop height = cap_h
  - paste_y = y1_band - cap_h (puts baseline at band y1)

Font sizes calibrated to match original capH exactly:
  - small: Arial Narrow Bold @ fsize=71 -> capH=51 (matches band h=51)
  - big:   Impact @ fsize=125 -> capH=99 (matches band h=99)

Words chosen to match original letter counts for width fill via spacing.
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = r"E:\Desktop\双接口\image-fission"
CLEANED_PNG = os.path.join(ROOT, "jobs", "v319s", "armed_lama_cleaned_FINAL.png")
SOURCE_JPG  = os.path.join(ROOT, "jobs", "v315", "source_armed.jpg")
JOB_NEW = os.path.join(ROOT, "jobs", "v319t")
os.makedirs(JOB_NEW, exist_ok=True)

# Band geometry (from v319s)
BANDS = {
    "small": {"y0": 472, "y1": 523, "h": 51,  "cx": 829, "orig_w": 517,
              "word": "WE HONOR THE", "fpath": r"C:\Windows\Fonts\ARIALNB.TTF", "fsize": 71},
    "big1":  {"y0": 541, "y1": 640, "h": 99,  "cx": 829, "orig_w": 1322,
              "word": "STEEL",        "fpath": r"C:\Windows\Fonts\impact.ttf", "fsize": 125},
    "big2":  {"y0": 656, "y1": 755, "h": 99,  "cx": 863, "orig_w": 917,
              "word": "SOLDIER",      "fpath": r"C:\Windows\Fonts\impact.ttf", "fsize": 125},
}

img = Image.open(CLEANED_PNG).convert("RGB")
print(f"Loaded cleaned: {CLEANED_PNG}  size={img.size}")

for band_key, b in BANDS.items():
    y0, y1 = b["y0"], b["y1"]
    cx = b["cx"]
    target_w = b["orig_w"]
    word = b["word"]
    fpath = b["fpath"]
    fsize = b["fsize"]

    font = ImageFont.truetype(fpath, fsize)
    ascent, descent = font.getmetrics()
    cap_h = font.getbbox("A")[3] - font.getbbox("A")[1]
    n_gaps = max(1, len(word) - 1)
    # Use getlength (advance width) for spacing — this is what PIL uses to advance
    char_advances = [font.getlength(ch) for ch in word]
    nat_w = sum(char_advances)
    spacing = max(0, int(round((target_w - nat_w) / n_gaps)))

    # Render in a tall transparent layer, draw at y=0 (cell top)
    layer_h = fsize + 60  # enough for ascent+descent
    total_w = int(nat_w + spacing * n_gaps) + 40
    layer = Image.new("RGBA", (total_w, layer_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x = 20
    for i, (ch, adv) in enumerate(zip(word, char_advances)):
        d.text((x, 0), ch, font=font, fill="black")
        x += adv
        if i < len(word) - 1:
            x += spacing

    # Crop to tight bbox (cap top to baseline, no descenders in our words)
    bbox = layer.getbbox()
    if bbox is None:
        print(f"  {band_key}: SKIP (empty bbox)")
        continue
    layer_cropped = layer.crop(bbox)
    rw, rh = layer_cropped.size
    pct = 100.0 * rw / target_w
    # Position: crop top = cap top. Baseline in crop = cap_h (from top of crop).
    # We want baseline at band y1, so paste_y = y1 - cap_h = y0
    paste_x = cx - rw // 2
    paste_y = y0
    img.paste(layer_cropped, (paste_x, paste_y), layer_cropped)
    print(f"  {band_key}: '{word}' fsize={fsize} capH={cap_h} ascent={ascent} "
          f"nat_w={nat_w} spacing={spacing} -> w={rw}/{target_w} ({pct:.2f}%) "
          f"paste=({paste_x},{paste_y})")

# Save
out_full = os.path.join(JOB_NEW, "armed_text_only_v319t.jpg")
img.save(out_full, quality=92)
print(f"Saved: {out_full}")

# 3-band compare
src = Image.open(SOURCE_JPG).convert("RGB")
band_h_total = 210
cmp_w = src.size[0]
cmp_h = band_h_total * 3
cmp_img = Image.new("RGB", (cmp_w, cmp_h), "white")
d = ImageDraw.Draw(cmp_img)
lbl_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 30)
labels = ["ORIGINAL", "CLEANED v319s (LaMa)", "FINAL v319t (Impact/A.Narrow)"]
imgs = [src, Image.open(CLEANED_PNG).convert("RGB"), img]
y = 0
for lbl, im in zip(labels, imgs):
    crop = im.crop((0, 440, im.size[0], 780))
    cmp_img.paste(crop, (0, y))
    d.text((20, y + 6), lbl, font=lbl_font, fill="red")
    y += band_h_total
out_cmp = os.path.join(JOB_NEW, "armed_text_band_compare_v319t.jpg")
cmp_img.save(out_cmp, quality=90)
print(f"Saved compare: {out_cmp}")

# Side-by-side
side = Image.new("RGB", (src.size[0]*2 + 20, src.size[1]), (80, 80, 80))
side.paste(src, (0, 0))
side.paste(img, (src.size[0] + 20, 0))
out_side = os.path.join(JOB_NEW, "armed_compare_v319t.jpg")
side.save(out_side, quality=85)
print(f"Saved side: {out_side}")
