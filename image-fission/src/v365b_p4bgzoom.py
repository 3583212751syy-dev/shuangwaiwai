#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v365b —— p4 底方案 1:1 放大对比：原图 / EDT / EDT+中值 / 高斯后验"""
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
OUT = ROOT / "jobs" / "v365_p4bg"
OUT.mkdir(parents=True, exist_ok=True)


def pal_from_regions(a, free):
    """从大色区取色：对自由区做连通块，取面积前 N 块的中位色。"""
    q = (a // 6).astype(np.int64)
    key = q[..., 0] * 1_000_000 + q[..., 1] * 1000 + q[..., 2]
    key = np.where(free, key, -1)
    vals, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    cols = []
    for o in np.argsort(-cnt):
        c = vals[o]
        if c < 0:
            continue
        sel = (inv == o).reshape(free.shape)
        m = a[sel].mean(0)
        if any(np.abs(m - x).max() < 30 for x in cols):
            continue
        cols.append(m)
        if len(cols) >= 5:
            break
    return np.stack(cols)


def main():
    orig = Image.open(SRC).convert("RGB")
    W, H = orig.size
    a = np.asarray(orig).astype(np.float32)
    ink = cpp.tree_ink_mask(orig, 55.0, 14.0, 25, False)
    ink_all = cpp.tree_ink_mask(orig, 55.0, 14.0, 1, False) | ink
    masked = ndi.binary_dilation(ink_all, iterations=8)

    pal = pal_from_regions(a, ~masked)
    print("区域众数色板:", [[int(v) for v in c] for c in pal])

    # EDT 精确色填充
    idx = ndi.distance_transform_edt(masked, return_distances=False,
                                     return_indices=True)
    edt = a.copy()
    edt[masked] = a[idx[0], idx[1]][masked]
    Image.fromarray(edt.clip(0, 255).astype(np.uint8)).save(OUT / "bg_edt.jpg", quality=95)

    # EDT 后对填充区做「索引中值」平滑（在色板索引域上，保证不产生混合色）
    def near_i(x):
        d = ((x[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
        return d.argmin(-1).astype(np.int16)
    ii = near_i(edt)
    ii_s = ndi.median_filter(ii, size=11)
    e2 = a.copy()
    e2[masked] = pal[ii_s][masked]
    Image.fromarray(e2.clip(0, 255).astype(np.uint8)).save(OUT / "bg_edt_med.jpg", quality=95)

    # 高斯后验
    K = len(pal)
    onehot = np.zeros(ii.shape + (K,), np.float32)
    np.put_along_axis(onehot, ii[..., None], 1.0, -1)
    w = (~masked).astype(np.float32)
    num = np.stack([ndi.gaussian_filter(onehot[..., k] * w, 30.0)
                    for k in range(K)], -1)
    post = num / (ndi.gaussian_filter(w, 30.0)[..., None] + 1e-6)
    e3 = a.copy()
    e3[masked] = pal[post.argmax(-1)][masked]
    Image.fromarray(e3.clip(0, 255).astype(np.uint8)).save(OUT / "bg_post.jpg", quality=95)

    for gi, box in enumerate([(60, 500, 610, 1050), (640, 200, 1190, 750),
                              (150, 1200, 700, 1750)]):
        bw, bh = box[2] - box[0], box[3] - box[1]
        row = [orig, Image.open(OUT / "bg_edt.jpg"),
               Image.open(OUT / "bg_edt_med.jpg"), Image.open(OUT / "bg_post.jpg")]
        sheet = Image.new("RGB", (bw * 4 + 50, bh + 20), (205, 205, 205))
        for i, im in enumerate(row):
            sheet.paste(im.crop(box), (10 + i * (bw + 10), 10))
        sheet.save(OUT / f"zoom_{gi}.jpg", quality=95)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
