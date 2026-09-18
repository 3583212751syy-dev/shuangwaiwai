#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v365 —— p4 干净迷彩底（平滑边界版）

思路：迷彩 = 大块平涂色区。抹掉棕榈后，用「色区后验平滑 + 重新量化」重填：
  ① 众数色板 P（只取真正的大色区，过滤抗锯齿/过渡色）
  ② 全图每个非墨像素 → 最近色板索引 idx
  ③ 墨区像素：对 one-hot(idx) 做高斯模糊（σ 大）→ 得到平滑后验场
     取 argmax → 边界是平滑曲线（等值线），不是 Voronoi 直线
  ④ 墨区外用原图像素 → 原迷彩零改动
对比 LaMa / EDT-Voronoi / EDT+median / 本方案。
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

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
OUT = ROOT / "jobs" / "v365_p4bg"
OUT.mkdir(parents=True, exist_ok=True)


def palette_mode(a, ink, ncol=5, dmin=34):
    arr = a.reshape(-1, 3)[~ink.reshape(-1)]
    q = (arr // 8).astype(np.int64)
    key = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    _, first, cnt = np.unique(key, return_index=True, return_counts=True)
    cols = []
    for o in np.argsort(-cnt):
        c = arr[first[o]].astype(np.float32)
        if not (40 < c.mean() < 220):
            continue
        if any(np.abs(c - x).max() < dmin for x in cols):
            continue
        cols.append(c)
        if len(cols) >= ncol:
            break
    return np.stack(cols)


def nearest_idx(a, pal):
    d = ((a[:, :, None, :].astype(np.float32) - pal[None, None, :, :]) ** 2).sum(-1)
    return d.argmin(-1).astype(np.int16)


def smooth_fill(a, pal, masked, sigma=26.0):
    idx = nearest_idx(a, pal)
    K = len(pal)
    onehot = np.zeros(idx.shape + (K,), np.float32)
    np.put_along_axis(onehot, idx[..., None], 1.0, axis=-1)
    onehot[masked] = 0.0
    # 只对已知区做平滑（masked 处权重 0 → 自然由邻域填补）
    w = (~masked).astype(np.float32)
    num = np.stack([ndi.gaussian_filter(onehot[..., k] * w, sigma)
                    for k in range(K)], -1)
    den = ndi.gaussian_filter(w, sigma)[..., None] + 1e-6
    post = num / den
    out = pal[post.argmax(-1)].astype(np.uint8)
    return Image.fromarray(out)


def main():
    orig = Image.open(SRC).convert("RGB")
    W, H = orig.size
    a = np.asarray(orig).astype(np.float32)

    ink = cpp.tree_ink_mask(orig, 55.0, 14.0, 25, False)
    ink_all = cpp.tree_ink_mask(orig, 55.0, 14.0, 1, False) | ink
    masked = ndi.binary_dilation(ink_all, iterations=8)
    print(f"{W}x{H} 墨={100*ink.mean():.2f}%")

    pal = palette_mode(a, ink, 5)
    print("色板:", [[int(v) for v in c] for c in pal])

    for sg in (14.0, 26.0, 42.0):
        f = smooth_fill(a, pal, masked, sigma=sg)
        fa = np.asarray(f)
        bg = a.copy()
        bg[masked] = fa[masked]
        Image.fromarray(bg.clip(0, 255).astype(np.uint8)).save(
            OUT / f"bg_S{int(sg)}.jpg", quality=95)

    # 拼图：原图 / S14 / S26 / S42
    z = 560 / H
    tw, th = int(W * z), 560
    ims = [orig] + [Image.open(OUT / f"bg_S{s}.jpg") for s in (14, 26, 42)]
    sheet = Image.new("RGB", (tw * 4 + 50, th + 20), (205, 205, 205))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((tw, th), Image.LANCZOS),
                    (10 + i * (tw + 10), 10))
    sheet.save(OUT / "bg_cmp4.jpg", quality=95)

    # 1:1 局部：看填充边界质量
    for gi, box in enumerate([(60, 520, 560, 1020), (700, 120, 1200, 620)]):
        cr = Image.new("RGB", (500 * 2 + 30, 500 + 20), (205, 205, 205))
        cr.paste(orig.crop(box), (10, 10))
        cr.paste(Image.open(OUT / "bg_S26.jpg").crop(box), (520, 10))
        cr.save(OUT / f"bg_zoom_{gi}.jpg", quality=95)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
