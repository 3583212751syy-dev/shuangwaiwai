"""Debug: render each band's text layer separately to see what's actually produced."""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = r"E:\Desktop\双接口\image-fission"
JOB = os.path.join(ROOT, "jobs", "v319t")
os.makedirs(JOB, exist_ok=True)

FONT_BIG   = r"C:\Windows\Fonts\impact.ttf"
FONT_SMALL = r"C:\Windows\Fonts\ARIALNB.TTF"

BANDS = {
    "small": {"y0": 472, "y1": 523, "h": 51,  "cx": 829, "orig_w": 517,
              "word": "WE HONOR THE", "fpath": FONT_SMALL, "fsize": 71},
    "big1":  {"y0": 541, "y1": 640, "h": 99,  "cx": 829, "orig_w": 1322,
              "word": "STEEL", "fpath": FONT_BIG, "fsize": 138},
    "big2":  {"y0": 656, "y1": 755, "h": 99,  "cx": 863, "orig_w": 917,
              "word": "SOLDIER", "fpath": FONT_BIG, "fsize": 138},
}

for bk, b in BANDS.items():
    word = b["word"]
    fpath = b["fpath"]
    fsize = b["fsize"]
    target_w = b["orig_w"]
    cx = b["cx"]
    y1 = b["y1"]

    font = ImageFont.truetype(fpath, fsize)
    ascent, descent = font.getmetrics()
    cap_h = font.getbbox("A")[3] - font.getbbox("A")[1]
    n_gaps = max(1, len(word) - 1)
    char_widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in word]
    nat_w = sum(char_widths)
    spacing = max(0, int(round((target_w - nat_w) / n_gaps)))

    # Render on white bg for debug visibility
    total_w = nat_w + spacing * n_gaps + 40
    local_h = fsize + 60
    layer = Image.new("RGBA", (total_w, local_h), (255, 255, 255, 255))
    d = ImageDraw.Draw(layer)
    x = 20
    for i, (ch, cw) in enumerate(zip(word, char_widths)):
        d.text((x, 30 + ascent), ch, font=font, fill="black")
        x += cw
        if i < len(word) - 1:
            x += spacing

    dbg_path = os.path.join(JOB, f"debug_layer_{bk}.png")
    layer.save(dbg_path)
    print(f"{bk}: word='{word}' fsize={fsize} nat_w={nat_w} spacing={spacing} "
          f"total_w={total_w} layer_h={local_h} ascent={ascent} cap_h={cap_h} "
          f"-> saved {dbg_path}")
