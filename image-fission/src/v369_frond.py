#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v369 —— 原图单叶（frond）4x 放大，锁定锯齿几何"""
import sys
from pathlib import Path

from PIL import Image
from scipy import ndimage as ndi
import numpy as np

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp          # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
OUT = ROOT / "jobs" / "v366_measure"
OUT.mkdir(parents=True, exist_ok=True)

src = Image.open(SRC).convert("RGB")
a = np.asarray(src).astype(np.float32)
ink = cpp.tree_ink_mask(src, 55.0, 14.0, 25, False)
dt = ndi.distance_transform_edt(ink)

# 3 处单叶特写（原图坐标）
boxes = [(250, 560, 470, 720), (600, 640, 800, 790), (860, 60, 1080, 230)]
tiles = []
for b in boxes:
    c = src.crop(b)
    z = 460 / c.height
    tiles.append(c.resize((int(c.width * z), 460), Image.LANCZOS))
    sub = ink[b[1]:b[3], b[0]:b[2]]
    d = dt[b[1]:b[3], b[0]:b[2]]
    v = 2.0 * d[sub]
    print(f"{b}  笔宽 中位={np.median(v):.1f} p90={np.percentile(v,90):.1f}"
          f"  墨={100*sub.mean():.1f}%")

gap = 14
W = sum(t.width for t in tiles) + gap * (len(tiles) + 1)
sh = Image.new("RGB", (W, 460 + 2 * gap), (235, 235, 235))
x = gap
for t in tiles:
    sh.paste(t, (x, gap))
    x += t.width + gap
sh.save(OUT / "frond_zoom.jpg", quality=95)
print("saved", sh.size)
