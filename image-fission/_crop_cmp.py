"""_crop_cmp.py — 原图 vs 成品 1:1 局部对照（p4 树干/树冠、p6 鹰头）。"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
P4O = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
P4N = ROOT / "jobs" / "router_out_v329" / "Pinterest (4)_variant.jpg"
P6O = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
P6N = ROOT / "jobs" / "router_out_v329" / "pinterest6_variant.jpg"
OUT = ROOT / "jobs" / "_cmpcrop"
OUT.mkdir(parents=True, exist_ok=True)


def sheet(o, n, boxes, scale, dst, titles=("ORIG", "NEW")):
    ow, oh = o.size
    nw, nh = n.size
    cells = []
    for (x0, y0, x1, y1) in boxes:
        co = o.crop((x0, y0, x1, y1)).resize(((x1 - x0) * scale, (y1 - y0) * scale),
                                             Image.NEAREST)
        bx = (int(x0 * nw / ow), int(y0 * nh / oh), int(x1 * nw / ow), int(y1 * nh / oh))
        cn = n.crop(bx).resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.NEAREST)
        cells.append((co, cn))
    cw, ch = cells[0][0].size
    W = cw * 2 + 24
    H = (ch + 30) * len(cells) + 10
    sh = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(sh)
    d.text((6, 6), titles[0], fill=(0, 0, 0))
    d.text((cw + 30, 6), titles[1], fill=(200, 0, 0))
    y = 26
    for a, b in cells:
        sh.paste(a, (6, y)); sh.paste(b, (cw + 24, y))
        y += ch + 30
    sh.save(str(dst), quality=95)
    print("saved", dst, sh.size)


o4 = Image.open(P4O).convert("RGB"); n4 = Image.open(P4N).convert("RGB")
W4, H4 = o4.size
print("p4 size", W4, H4, "->", n4.size)
# 树干区 / 树冠区 各两块
b4 = [(int(0.06*W4), int(0.30*H4), int(0.06*W4)+230, int(0.30*H4)+230),
      (int(0.40*W4), int(0.55*H4), int(0.40*W4)+230, int(0.55*H4)+230),
      (int(0.62*W4), int(0.16*H4), int(0.62*W4)+230, int(0.16*H4)+230)]
sheet(o4, n4, b4, 2, OUT / "p4_crop.jpg")

o6 = Image.open(P6O).convert("RGB"); n6 = Image.open(P6N).convert("RGB")
W6, H6 = o6.size
print("p6 size", W6, H6, "->", n6.size)
b6 = [(int(0.20*W6), int(0.20*H6), int(0.20*W6)+240, int(0.20*H6)+240),
      (int(0.30*W6), int(0.46*H6), int(0.30*W6)+240, int(0.46*H6)+240)]
sheet(o6, n6, b6, 2, OUT / "p6_crop.jpg")
