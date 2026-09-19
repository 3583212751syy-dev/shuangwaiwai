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
import argparse
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

# v403：可调参数（出多版候选用）。默认值 = v368/v372 的定稿值。
ARGS = None
P_TILT = 14.0          # 树干左右倾角范围（±度）
P_CROWN = (0.9, 1.2)   # 冠幅相对缩放范围
P_DILATE = 8           # EDT 清底时对墨迹的膨胀（越小 → 背景迷彩改动越少）
P_SEED = 20260917      # 布点随机种子
P_RSEED = 771          # 绘制随机种子


def parse_args():
    global ARGS, KIND_P, P_TILT, P_CROWN, P_DILATE, P_SEED, P_RSEED, OUT
    ap = argparse.ArgumentParser(description="p4 迷彩棕榈程序化重绘（可出多版候选）")
    ap.add_argument("--tag", default="v368", help="输出后缀（jobs/v403_p4cands/<tag>.jpg）")
    ap.add_argument("--out-dir", default="", help="输出目录（默认 jobs/v368_p4v2）")
    ap.add_argument("--dilate", type=int, default=8, help="EDT 清底膨胀半径")
    ap.add_argument("--tilt", type=float, default=14.0, help="树干倾角范围 ±度")
    ap.add_argument("--crown-lo", type=float, default=0.9)
    ap.add_argument("--crown-hi", type=float, default=1.2)
    ap.add_argument("--size-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--rseed", type=int, default=771)
    ap.add_argument("--solid", type=float, default=0.56)
    ap.add_argument("--chevron", type=float, default=0.20)
    ap.add_argument("--scribble", type=float, default=0.08)
    ap.add_argument("--tuft", type=float, default=0.16)
    a = ap.parse_args()
    ARGS = a
    KIND_P = [('solid', a.solid), ('chevron', a.chevron),
              ('scribble', a.scribble), ('tuft', a.tuft)]
    P_TILT, P_CROWN, P_DILATE = a.tilt, (a.crown_lo, a.crown_hi), a.dilate
    P_SEED, P_RSEED = a.seed, a.rseed
    if a.out_dir:
        OUT = Path(a.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    return a


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
                        crown_scale=rng.uniform(*P_CROWN))
        al = spr.point(lambda v: 255 - v)
        tilt = rng.uniform(-P_TILT, P_TILT)
        if abs(tilt) > 2.5:
            al = al.rotate(tilt, resample=Image.BICUBIC, expand=True,
                           fillcolor=0)
        canvas.paste(al, (int(cx - al.width / 2),
                          int(cy + h * 0.50 - al.height)), al)
    return np.asarray(canvas).astype(np.float32) / 255.0


def main():
    ag = parse_args()
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
    bg.save(OUT / f"{ag.tag}_bg.jpg", quality=92)
    print(f"[1] EDT 清底 {time.time()-t0:.0f}s")

    # ② 密度场
    dens, cell = dens_map(ink, 40, 52.0)

    # ③ 二分 size_scale 对齐墨量
    lo, hi, best = 0.14, 0.62, None
    for it in range(9):
        f = 0.5 * (lo + hi)
        rng = random.Random(P_SEED)
        pl = build(ink, W, H, rng, ag.size_scale, dens, cell, nms_f=f)
        al = render(pl, W, H, random.Random(P_RSEED))
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
    res.save(OUT / f"{ag.tag}.jpg", quality=95)

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
    sheet.save(OUT / f"{ag.tag}_cmp.jpg", quality=95)
    for i, box in enumerate([(60, 150, 560, 800), (620, 380, 1120, 1030),
                             (60, 1150, 560, 1800), (700, 1050, 1200, 1700)]):
        bw, bh = box[2] - box[0], box[3] - box[1]
        cr = Image.new("RGB", (bw * 2 + 30, bh + 20), (205, 205, 205))
        cr.paste(orig.crop(box), (10, 10))
        cr.paste(res.crop(box), (bw + 20, 10))
        cr.save(OUT / f"{ag.tag}_1to1_{i}.jpg", quality=95)
    print("saved ->", OUT, ag.tag)


if __name__ == "__main__":
    main()
