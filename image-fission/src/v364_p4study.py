#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v364 —— p4 迷彩棕榈 材质/结构研究 + 干净底方案对比

目的：
  A. 量化原图元素结构（墨占比 / 元素块尺寸分布 / 元素高度分布）
  B. 对比三种「去棕榈留迷彩底」方案：
     L  = LaMa（上一版，会带灰蓝模糊 blob）
     E1 = EDT 最近非墨色扩展（Voronoi，锐利平涂）
     E2 = E1 + 标签中值平滑（边界更自然）
  C. 切 1:1 元素特写，建立风格参考（实心瘦叶棕榈 / 涂鸦线棕榈 / 涂鸦草丛）
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp          # noqa: E402
import v268_lama_clean as lama_mod                   # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
OUT = ROOT / "jobs" / "v364_study"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    orig = Image.open(SRC).convert("RGB")
    W, H = orig.size
    a = np.asarray(orig).astype(np.float32)
    print(f"原图 {W}x{H}")

    ink = cpp.tree_ink_mask(orig, 55.0, 14.0, 25, False)
    ink_all = cpp.tree_ink_mask(orig, 55.0, 14.0, 1, False) | ink
    print(f"墨(大块)={100*ink.mean():.2f}%  墨(含碎屑)={100*ink_all.mean():.2f}%")

    # ---- A. 元素块尺寸分布
    lbl, n = ndi.label(ink)
    sizes = np.bincount(lbl.ravel())[1:]
    sizes = np.sort(sizes)[::-1]
    print(f"连通块 n={n}")
    for thr in (20000, 8000, 2000, 500, 100):
        print(f"  >= {thr:6d}px : {(sizes >= thr).sum():4d}")
    objs = ndi.find_objects(lbl)
    hh = []
    area = np.bincount(lbl.ravel())[1:]
    order = np.argsort(-area)
    for k in order[:200]:
        sl = objs[k]
        if sl is None:
            continue
        hh.append(sl[0].stop - sl[0].start)
    hh = np.array(hh)
    print("元素块高度分位 (前200块):",
          {p: int(np.percentile(hh, p)) for p in (10, 25, 50, 75, 90, 100)})

    # ---- B. 三种底
    m = ndi.binary_dilation(ink_all, iterations=8)
    mimg = Image.fromarray((m * 255).astype(np.uint8))
    bg_lama = lama_mod.lama_inpaint(orig, mimg, removal_strength=235,
                                    edge_smoothness=4)
    bg_lama.save(OUT / "bg_L_lama.jpg", quality=95)

    idx = ndi.distance_transform_edt(m, return_distances=False,
                                     return_indices=True)
    fill = a[idx[0], idx[1]]
    bg_e1 = Image.fromarray(fill.clip(0, 255).astype(np.uint8))
    bg_e1.save(OUT / "bg_E1_edt.jpg", quality=95)

    # E2: 对「最近源坐标」做中值平滑 → 边界更有机
    lab = (idx[0].astype(np.int32) * (W + 1) + idx[1].astype(np.int32))
    lab_s = ndi.median_filter(lab, size=9)
    sy, sx = np.divmod(lab_s, W + 1)
    fill2 = a[np.clip(sy, 0, H - 1), np.clip(sx, 0, W - 1)]
    bg_e2 = Image.fromarray(fill2.clip(0, 255).astype(np.uint8))
    bg_e2.save(OUT / "bg_E2_edtmed.jpg", quality=95)

    # 迷彩色板（众数色，供对照）
    arr = a.reshape(-1, 3)[~ink.reshape(-1)]
    q = (arr // 8).astype(np.int64)
    key = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    _, first, cnt = np.unique(key, return_index=True, return_counts=True)
    top = []
    for o in np.argsort(-cnt)[:40]:
        c = arr[first[o]]
        if 45 < c.mean() < 215:
            top.append((int(cnt[o]), [int(v) for v in c]))
        if len(top) >= 10:
            break
    print("迷彩众数色(占比前10):")
    for cn, c in top:
        print(f"   {cn:8d}px  {c}")

    # 三底拼接
    z = 620 / H
    tw, th = int(W * z), 620
    sheet = Image.new("RGB", (tw * 3 + 40, th + 20), (205, 205, 205))
    for i, im in enumerate([bg_lama, bg_e1, bg_e2]):
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (10 + i * (tw + 10), 10))
    sheet.save(OUT / "bg_cmp3.jpg", quality=95)

    # ---- C. 元素 1:1 特写（供风格参考）
    crops = [(60, 520, 460, 1010), (1080, 380, 1480, 880),
             (300, 1500, 700, 1980), (700, 120, 1100, 620),
             (1180, 1120, 1580, 1620), (40, 60, 440, 560),
             (620, 1300, 1020, 1800), (900, 780, 1300, 1280)]
    per = 4
    cw = 400
    for gi in range(2):
        g = Image.new("RGB", (cw * 2 + 30, cw * 2 + 30), (205, 205, 205))
        for k in range(per):
            b = crops[gi * per + k]
            b = (max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3]))
            c = orig.crop(b)
            g.paste(c, ((k % 2) * (cw + 10) + 10,
                        (k // 2) * (cw + 10) + 10))
        g.save(OUT / f"elems_{gi}.jpg", quality=95)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
