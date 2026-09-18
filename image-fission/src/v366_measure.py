#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v366 —— p4 原图笔触量化：线宽 / 元素高度 / 锯齿参数测量"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp          # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
OUT = ROOT / "jobs" / "v366_measure"
OUT.mkdir(parents=True, exist_ok=True)


def stroke_width(ink):
    dt = ndi.distance_transform_edt(ink)
    v = 2.0 * dt[ink]
    return v


def main():
    orig = Image.open(SRC).convert("RGB")
    W, H = orig.size
    ink = cpp.tree_ink_mask(orig, 55.0, 14.0, 25, False)

    sw = stroke_width(ink)
    print(f"图 {W}x{H}  墨={100*ink.mean():.2f}%")
    print("墨区笔触宽度 2*EDT 分位:",
          {p: round(float(np.percentile(sw, p)), 1)
           for p in (5, 10, 25, 50, 75, 90, 95)})
    h = np.histogram(sw, bins=np.arange(0, 61, 2))
    top = np.argsort(-h[0])[:8]
    print("笔触宽度直方图峰值(bin左值:占比):",
          [(int(h[1][i]), round(float(h[0][i]) / sw.size * 100, 1)) for i in top])

    # 按连通块看：大块的「骨架宽度」
    lbl, n = ndi.label(ink)
    area = np.bincount(lbl.ravel())[1:]
    order = np.argsort(-area)[:24]
    objs = ndi.find_objects(lbl)
    print("\n面积前24块:")
    for k in order:
        sl = objs[k]
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        sub = (lbl[sl] == k + 1)
        d = ndi.distance_transform_edt(sub)
        mw = 2.0 * d.max()
        print(f"  area={area[k]:7d} bbox={bw:4d}x{bh:4d} 最大内切宽={mw:5.1f}"
              f" 宽/高={mw/max(1,bh):.4f}")

    # 单独棕榈裁切（1:1 原分辨率）用于目测
    boxes = [(120, 540, 560, 1080, "scribble_palm"),
             (940, 380, 1242, 900, "solid_palm"),
             (330, 1480, 700, 1754, "tuft"),
             (600, 130, 980, 560, "scribble2"),
             (30, 1760, 0, 0, "x")]
    for x0, y0, x1, y1, nm in boxes[:4]:
        x1 = min(W, x1); y1 = min(H, y1)
        c = orig.crop((x0, y0, x1, y1))
        s = 2 if c.width < 500 else 1
        c.resize((c.width * s, c.height * s), Image.LANCZOS).save(
            OUT / f"{nm}.jpg", quality=95)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
