# -*- coding: utf-8 -*-
"""Build 4-way comparison: orig / v174 / v181 / v183 (camouflage)"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

paths = {
    "ORIG":       ROOT / "web_gallery/img/orig_camo_4.jpg",
    "v174 (配色最准+椰子最清晰)": ROOT / "outputs/v174/v174_camo_4.jpg",
    "v181 (颜色漂+乱码——已废)": ROOT / "jobs/smoke_v181/v181_camo_4.jpg",
    "v183 (配色对+底部队列数量裂变)": ROOT / "jobs/smoke_v183/v183_camo_4.jpg",
}

H_target = 720
gap = 14
ims = []
labels = []
for label, p in paths.items():
    im = Image.open(p).convert("RGB")
    w, h = im.size
    nw = int(w * H_target / h)
    ims.append(im.resize((nw, H_target), Image.LANCZOS))
    labels.append(label)

W = sum(im.width for im in ims) + gap * (len(ims) - 1)
canvas = Image.new("RGB", (W, H_target + 60), (14, 15, 18))
x = 0
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font = ImageFont.load_default()
for im, label in zip(ims, labels):
    canvas.paste(im, (x, 50))
    color = (255, 106, 61) if "v183" in label else (200, 100, 50) if "v174" in label else (255, 80, 80) if "v181" in label else (230, 234, 240)
    draw.text((x + 6, 12), label, fill=color, font=font)
    x += im.width + gap

out = ROOT / "jobs/smoke_v183/compare_4way_orig_v174_v181_v183.jpg"
canvas.save(out, quality=92)
print(f"[4way] saved {out} {out.stat().st_size/1024/1024:.2f}MB")