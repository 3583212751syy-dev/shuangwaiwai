#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v368 —— p4 迷彩棕榈「风格重绘」v2（替代 v361）

修正 v361 的两个缺陷：
  ✗ 底：LaMa 把抹掉区糊成模糊块 → 判定「质量低」
  ✓ 底：EDT 最近源色扩展 → 平涂锐利、只用原图真实色（零混合灰）
  ✗ 株：单一「胖实心三角」型 → 与原作者笔触不符
  ✓ 株：三类元素（solid 细叶实心 / scribble 涂鸦线+羽刺 / tuft 草丛）
        全部按 v366 实测笔宽（线宽≈0.010H、羽刺、锯齿波长）绘制
  ✗ 布点：全局墨量对齐但局部大块空白
  ✓ 布点：以原图局部墨密度的**概率场**采样 + 残差二次补小元素
"""
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp          # noqa: E402
from styles.palm_draw import make_palm               # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
OUT = ROOT / "jobs" / "v368_p4v2"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_INK = 0.2263
NMS_F = 0.46
# v372：原图绝大多数是「实心树干 + 宽叶片」冠；涂鸦/人字齿只占少数。
# 旧配比 scribble 0.42 让整幅变细、读成线框 → 用户批「质量低」。现以 solid 为主。
KIND_P = [('solid', 0.56), ('chevron', 0.20), ('scribble', 0.08), ('tuft', 0.16)]


def edt_fill(a, masked):
    idx = ndi.distance_transform_edt(masked, return_distances=False,
                                     return_indices=True)
    out = a.copy()
    out[masked] = a[idx[0], idx[1]][masked]
    return out


def dens_map(ink, cell, sigma):
    H, W = ink.shape
    ch, cw = H // cell, W // cell
    d = ink[:ch * cell, :cw * cell].reshape(ch, cell, cw, cell).mean((1, 3))
    d = ndi.gaussian_filter(d, sigma / cell)
    return d, cell


def pick_kind(rng):
    u = rng.random()
    acc = 0.0
    for k, p in KIND_P:
        acc += p
        if u <= acc:
            return k
    return KIND_P[0][0]


def build(ink, W, H, rng, size_scale, dens, cell, cov_min=0.012, nms_f=0.34):
    ch, cw = dens.shape
    cands = []
    for j in range(ch):
        for i in range(cw):
            cands.append((float(dens[j, i]),
                          (i + 0.5) * cell + rng.uniform(-cell, cell) * 0.5,
                          (j + 0.5) * cell + rng.uniform(-cell, cell) * 0.5))
    cands.sort(key=lambda t: -t[0])
    placed = []
    for cov, x, y in cands:
        if cov < cov_min:
            continue
        # v372：原图是一幅「大小混排马赛克」（实测株高 100~520px，小株很多）。
        # 旧式 150+320*cov 让最小株=150 且偏大 → 只有 28 株大株、缺小株填空。
        h = (108 + 355 * min(1.0, cov / 0.40) ** 0.85) * size_scale \
            * rng.uniform(0.76, 1.26)
        h = float(np.clip(h, 96, 620))
        if x < -h * 0.2 or x > W + h * 0.2 or y < 0 or y > H + h * 0.05:
            continue
        r = nms_f * h
        if any((x - px) ** 2 + (y - py) ** 2
               < max(r, nms_f * ph) ** 2 for px, py, ph in placed):
            continue
        placed.append((x, y, h))
    return placed


def render(placed, W, H, rng, seed0=9100):
    canvas = Image.new("L", (W, H), 0)
    for k, (cx, cy, h) in enumerate(placed):
        kind = pick_kind(rng)
        spr = make_palm(int(round(h)), seed=seed0 + k, kind=kind,
                        crown_scale=rng.uniform(0.9, 1.2))
        al = spr.point(lambda v: 255 - v)
        tilt = rng.uniform(-14, 14)
        if abs(tilt) > 2.5:
            al = al.rotate(tilt, resample=Image.BICUBIC, expand=True,
                           fillcolor=0)
        canvas.paste(al, (int(cx - al.width / 2),
                          int(cy + h * 0.50 - al.height)), al)
    return np.asarray(canvas).astype(np.float32) / 255.0


def main():
    t0 = time.time()
    orig = Image.open(SRC).convert("RGB")
    W, H = orig.size
    a = np.asarray(orig).astype(np.float32)

    ink = cpp.tree_ink_mask(orig, 55.0, 14.0, 25, False)
    ink_all = cpp.tree_ink_mask(orig, 55.0, 14.0, 1, False) | ink
    ink_col = a[ink].mean(0)
    print(f"原图 {W}x{H}  墨={100*ink.mean():.2f}%  墨色={ink_col.round(1)}")

    # ① EDT 清底（零模糊、只用真实色）
    masked = ndi.binary_dilation(ink_all, iterations=8)
    # 原图 x=0 是 1px 纯白 JPEG 边 → 若不封掉，EDT 会把白沿整条左边界渗进抹除区
    masked[:1, :] = True
    masked[-1:, :] = True
    masked[:, :1] = True
    masked[:, -1:] = True
    bg = Image.fromarray(edt_fill(a, masked).clip(0, 255).astype(np.uint8))
    bg.save(OUT / "bg_edt.jpg", quality=95)
    print(f"[1] EDT 清底 {time.time()-t0:.0f}s")

    # ② 密度场
    dens, cell = dens_map(ink, 40, 52.0)

    # ③ 二分 size_scale 对齐墨量
    lo, hi, best = 0.14, 0.62, None
    for it in range(9):
        f = 0.5 * (lo + hi)
        rng = random.Random(20260917)
        pl = build(ink, W, H, rng, 1.0, dens, cell, nms_f=f)
        al = render(pl, W, H, random.Random(771))
        got = float(al.mean())
        if best is None or abs(got - TARGET_INK) < best[0]:
            best = (abs(got - TARGET_INK), f, pl, al, got)
        print(f"    it{it} nms={f:.3f} 株={len(pl):3d} 墨={100*got:5.2f}%")
        if got < TARGET_INK:
            hi = f
        else:
            lo = f
        if abs(got - TARGET_INK) < 0.004:
            break
    _, g, placed, alpha, got = best
    hs = np.array([p[2] for p in placed])
    print(f"[2] 定稿 nms={g:.3f} 株={len(placed)} 墨={100*got:.2f}%"
          f"（目标 {100*TARGET_INK:.2f}%）  高:中位={np.median(hs):.0f}"
          f" p10={np.percentile(hs,10):.0f} p90={np.percentile(hs,90):.0f}")

    out = np.asarray(bg).astype(np.float32) * (1.0 - alpha[..., None]) \
        + ink_col[None, None, :] * alpha[..., None]
    res = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    res.save(OUT / "p4_v2.jpg", quality=95)

    # ④ 指标
    oink = cpp.tree_ink_mask(res, 55.0, 14.0, 25, False)
    lbl, _ = ndi.label(oink)
    nblk = len(ndi.find_objects(lbl))
    asz = np.bincount(lbl.ravel())[1:]
    covd = (oink & ndi.binary_dilation(ink, iterations=16)).sum() / max(1, ink.sum())
    print(f"[3] 成品 墨={100*oink.mean():.2f}%  连通块={nblk}"
          f" (>=100px:{int((asz>=100).sum())} >=500px:{int((asz>=500).sum())})"
          f"  覆盖原足迹={100*covd:.0f}%  {time.time()-t0:.0f}s")
    lbl0, _ = ndi.label(ink)
    a0 = np.bincount(lbl0.ravel())[1:]
    print(f"    原图对照 连通块={len(a0)} (>=100px:{int((a0>=100).sum())}"
          f" >=500px:{int((a0>=500).sum())})")

    z = 900 / H
    th = (int(W * z), 900)
    sheet = Image.new("RGB", (th[0] * 2 + 30, 940), (205, 205, 205))
    sheet.paste(orig.resize(th, Image.LANCZOS), (10, 20))
    sheet.paste(res.resize(th, Image.LANCZOS), (th[0] + 20, 20))
    sheet.save(OUT / "S_full_cmp.jpg", quality=95)
    for i, box in enumerate([(60, 150, 560, 800), (620, 380, 1120, 1030),
                             (60, 1150, 560, 1800), (700, 1050, 1200, 1700)]):
        bw, bh = box[2] - box[0], box[3] - box[1]
        cr = Image.new("RGB", (bw * 2 + 30, bh + 20), (205, 205, 205))
        cr.paste(orig.crop(box), (10, 10))
        cr.paste(res.crop(box), (bw + 20, 10))
        cr.save(OUT / f"S_1to1_{i}.jpg", quality=95)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
