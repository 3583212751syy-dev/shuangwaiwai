#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v361 —— p4 迷彩棕榈「程序化风格重绘」裂变

为什么放弃 SDXL：
  p4 是平涂矢量剪影（纯黑棕榈 + 迷彩底）。SDXL inpaint 实测把剪影打成
  1648 个毛丝碎块（原图仅 91），树干环纹糊掉 → 用户判"质量低、剪影是碎的"。
  这是模型能力硬边界，换分辨率/阈值/CN 都救不了。

本方案：
  ① LaMa 抹掉全部原棕榈墨迹 → 干净迷彩底
     （已排除：最近邻色扩展=沿形状留幽灵影；错位复制=碎片化拼贴；
       闭运算掩膜=无效 → LaMa 对平滑色块背景最优）
  ② 色板量化（原图迷彩众数色 6 色 + 索引中值滤波）→ 平涂锐利
     ⚠️ 只在墨迹区内使用；墨迹区外一律保留原图 → 保住原迷彩原貌
  ③ styles/palm_draw 程序化绘制同风格全新棕榈（平涂纯黑、锐利边缘、
     实心波浪干+横缝、饱满弯刀叶）
  ④ 按原图局部墨密度布点（固定间距 NMS + 二分对齐墨量到原图 22.63%）
     → 同构（疏密/尺度同）+ 异内容（每株形态全新）
"""
import sys
import time
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp          # noqa: E402
from styles.palm_draw import make_palm               # noqa: E402
import v268_lama_clean as lama_mod                   # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
OUT = ROOT / "jobs" / "v361_p4palms"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_INK = 0.2263          # 原图墨占比
CELL = 130
COV_MIN = 0.055
NMS_D = 208.0                # 株间固定最小距离
H_LO, H_HI = 170.0, 600.0


def camo_palette(orig, ink, ncol=6):
    """从迷彩区**众数色**取色板（避免取到抗锯齿白边与近黑边）。"""
    arr = np.asarray(orig).reshape(-1, 3)
    arr = arr[~ink.reshape(-1)]
    q = (arr // 8).astype(np.int64)
    key = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    _, first, cnt = np.unique(key, return_index=True, return_counts=True)
    cols = []
    for o in np.argsort(-cnt):
        c = arr[first[o]].astype(np.float32)
        if c.mean() < 45 or c.mean() > 215:
            continue
        if any(np.abs(c - np.asarray(x)).max() < 30 for x in cols):
            continue
        cols.append(c.tolist())
        if len(cols) >= ncol:
            break
    if len(cols) < 3:
        cols = [[163, 145, 115], [118, 108, 91], [86, 69, 48],
                [70, 82, 55], [55, 57, 40]]
    return np.array(cols, dtype=np.float32)


def flat_palette(bg, pal, med=7):
    arr = np.asarray(bg).astype(np.float32)
    d = ((arr[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
    idx = ndi.median_filter(d.argmin(-1), size=med)
    return Image.fromarray(pal[idx].astype(np.uint8))


def density_candidates(ink, cell=CELL):
    H, W = ink.shape
    out = []
    for j in range(H // cell):
        for i in range(W // cell):
            sub = ink[j * cell:(j + 1) * cell, i * cell:(i + 1) * cell]
            out.append((float(sub.mean()), (i + 0.5) * cell, (j + 0.5) * cell))
    out.sort(key=lambda t: -t[0])
    return out


def place_palms(cen, rng, size_gain=1.0, nms=NMS_D):
    placed = []
    for cov, cx, cy in cen:
        if cov < COV_MIN:
            continue
        h = (H_LO + (H_HI - H_LO) * min(1.0, cov / 0.40) ** 0.80) * size_gain \
            * rng.uniform(0.86, 1.16)
        if any((cx - px) ** 2 + (cy - py) ** 2 < nms ** 2 for px, py, ph in placed):
            continue
        placed.append((cx, cy, h))
    return placed


def render(placed, W, H, rng, seed0=7000):
    canvas = Image.new("L", (W, H), 0)
    for k, (cx, cy, h) in enumerate(placed):
        cs = rng.uniform(0.95, 1.32)
        spr = make_palm(int(h), seed=seed0 + k, crown_scale=cs)
        al = spr.point(lambda v: 255 - v)
        tilt = rng.uniform(-12, 12)
        if abs(tilt) > 2.5:
            al = al.rotate(tilt, resample=Image.BICUBIC, expand=True, fillcolor=0)
        canvas.paste(al, (int(cx - al.width / 2), int(cy + h * 0.50 - al.height)), al)
    return np.asarray(canvas).astype(np.float32) / 255.0


def main():
    t0 = time.time()
    orig = Image.open(SRC).convert("RGB")
    W, H = orig.size
    a = np.asarray(orig).astype(np.float32)

    ink = cpp.tree_ink_mask(orig, 55.0, 14.0, 25, False)
    # 擦除用掩膜：min_px=1，连小墨点/碎屑一起抹掉，否则会作为"残留墨点"留在背景
    ink_all = cpp.tree_ink_mask(orig, 55.0, 14.0, 1, False) | ink
    ink_col = a[ink].mean(0)
    print(f"原图 {W}x{H}  墨={100*ink.mean():.2f}%  墨色={ink_col.round(1)}")

    pal = camo_palette(orig, ink, 6)
    print("迷彩色板:", [[round(float(v)) for v in c] for c in pal])

    m = ndi.binary_dilation(ink_all, iterations=8)
    mimg = Image.fromarray((m * 255).astype(np.uint8))
    bg_raw = lama_mod.lama_inpaint(orig, mimg, removal_strength=235, edge_smoothness=4)
    bg_flat = flat_palette(bg_raw, pal, med=7)
    inside = ndi.binary_dilation(ink_all, iterations=6)
    arr = np.asarray(bg_flat).copy()
    arr[~inside] = a[~inside].astype(np.uint8)
    bg = Image.fromarray(arr)
    bg.save(OUT / "bg_flat.jpg", quality=95)
    print(f"[1] LaMa 清底 + 色板量化 {time.time()-t0:.0f}s")

    cen = density_candidates(ink)
    lo, hi, best = 0.55, 2.60, None
    for it in range(7):
        g = 0.5 * (lo + hi)
        rng = random.Random(4242)
        pl = place_palms(cen, rng, size_gain=g)
        al = render(pl, W, H, rng)
        got = float(al.mean())
        if best is None or abs(got - TARGET_INK) < best[0]:
            best = (abs(got - TARGET_INK), g, pl, al, got)
        print(f"    iter{it} gain={g:.2f} 株={len(pl):3d} 墨={100*got:5.2f}%")
        if got < TARGET_INK:
            lo = g
        else:
            hi = g
        if abs(got - TARGET_INK) < 0.005:
            break
    _, gain, placed, alpha, got = best
    print(f"[2] 定稿 gain={gain:.2f} 株={len(placed)} 墨={100*got:.2f}%"
          f"（目标 {100*TARGET_INK:.2f}%）")

    out = np.asarray(bg).astype(np.float32) * (1.0 - alpha[..., None]) \
        + ink_col[None, None, :] * alpha[..., None]
    res = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    res.save(OUT / "p4_palms.jpg", quality=95)

    oink = cpp.tree_ink_mask(res, 55.0, 14.0, 25, False)
    lbl, _ = ndi.label(oink)
    nblk = len(ndi.find_objects(lbl))
    covd = (oink & ndi.binary_dilation(ink, iterations=14)).sum() / max(1, ink.sum())
    print(f"[3] 成品 墨={100*oink.mean():.2f}%  连通块={nblk}  "
          f"覆盖原墨足迹={100*covd:.0f}%  总耗时 {time.time()-t0:.0f}s")

    z = 900 / H
    th = (int(W * z), 900)
    sheet = Image.new("RGB", (th[0] * 2 + 30, 940), (205, 205, 205))
    sheet.paste(orig.resize(th, Image.LANCZOS), (10, 20))
    sheet.paste(res.resize(th, Image.LANCZOS), (th[0] + 20, 20))
    sheet.save(OUT / "S_full_cmp.jpg", quality=95)

    for i, box in enumerate([(60, 150, 470, 760), (620, 380, 1030, 990),
                             (60, 1150, 470, 1754)]):
        cr = Image.new("RGB", ((box[2] - box[0]) * 2 + 30, (box[3] - box[1]) + 20),
                       (205, 205, 205))
        cr.paste(orig.crop(box), (10, 10))
        cr.paste(res.crop(box), (box[2] - box[0] + 20, 10))
        cr.save(OUT / f"S_1to1_{i}.jpg", quality=95)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
