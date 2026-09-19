# -*- coding: utf-8 -*-
"""v407_compose.py —— 把 v407 候选字标按**原图排版位置**贴回 p6，出对比（第 27 轮）

排版几何（本轮新口径，对齐原图实测）
------------------------------------
原图 MRCHOSR：墨迹 x[12,3524] 宽 3513；**密集字身带 y[359,1286]**。
旧版按「墨迹顶 = y88」对齐 → 只保证最上面那根尖刺的位置，字身实际落哪不管；
本轮改为按**密集字身带顶 = y359**对齐 → 字身位置与原图重合（用户要的"排版位置"）。

合成只用**白色墨迹 alpha 叠加**，不做任何色块遮盖（原图内容靠 LaMa 底板保持）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
BASE = ROOT / "jobs" / "v403_wordbank" / "_base_notitle.jpg"
GLY = ROOT / "jobs" / "v407_title"
OUT = ROOT / "jobs" / "v407_title"

TARGET_W = 3513.0
DENSE_TOP = 359.0          # 原图密集字身带顶
TEXT_RGB = np.array([238, 240, 248], np.float32)


def glyph_alpha(path: Path, W: int, H: int, dense_top: float = DENSE_TOP,
                thr_lo: float = 0.42, thr_hi: float = 0.30):
    """字标 → alpha 层。极性自动判定 + 阈值切背景 + **按密集字身带顶对齐**落位。"""
    g = Image.open(path).convert("L")
    ga = np.asarray(g, np.float32) / 255.0
    bw = max(2, int(round(min(g.size) * 0.04)))
    border = np.concatenate([ga[:bw].ravel(), ga[-bw:].ravel(),
                             ga[:, :bw].ravel(), ga[:, -bw:].ravel()])
    bg_med = float(np.median(border))
    inverted = bg_med > 0.5
    if inverted:
        ga = 1.0 - ga
    thr, span = (0.55, 0.35) if inverted else (thr_lo, thr_hi)
    ga = np.clip((ga - thr) / span, 0.0, 1.0)
    inkm = ga > 0.35
    ys, xs = np.where(inkm)
    if len(ys) == 0:
        raise SystemExit(f"字标无墨迹：{path}")
    by0, by1, bx0, bx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    rws = inkm.sum(1)
    dense = np.where(rws > rws.max() * 0.25)[0]
    y0 = max(by0, int(dense.min()) - 55)
    y1 = min(by1, int(dense.max()) + 55)
    sub = ga[y0:y1, bx0:bx1]
    sh, sw = sub.shape
    sc = TARGET_W / float(sw)
    tw, th = int(round(sw * sc)), int(round(sh * sc))
    gly = Image.fromarray((sub * 255).astype(np.uint8), "L").resize((tw, th), Image.LANCZOS)
    aa = np.asarray(gly, np.float32) / 255.0
    # 密集带顶在 sub 内的行号（相对 sub 顶）
    d_rel = max(0, int(dense.min()) - y0)
    cy = int(round(dense_top - d_rel * sc))
    cy = max(-th, min(cy, H))
    a = np.zeros((H, W), np.float32)
    cx0 = (W - tw) // 2
    x1 = min(W, cx0 + tw)
    sy0, sy1 = max(0, cy), min(H, cy + th)
    if sy1 > sy0:
        a[sy0:sy1, cx0:x1] = aa[sy0 - cy:sy1 - cy, :x1 - cx0]
    return a, dict(bg_med=bg_med, inverted=inverted, slab=(tw, th), pos=(cx0, cy),
                   dense_rel=d_rel)


def compose(base: Image.Image, alpha: np.ndarray) -> Image.Image:
    b = np.asarray(base, np.float32)
    o = b * (1 - alpha[..., None]) + TEXT_RGB[None, None, :] * alpha[..., None]
    return Image.fromarray(np.clip(o, 0, 255).astype(np.uint8), "RGB")


def band(im: Image.Image, w: int = 1500) -> Image.Image:
    c = im.crop((0, 0, im.width, 1560))
    return c.resize((w, int(1560 * w / im.width)), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glyphs", default="", help="逗号分隔的字标路径；空=用 GLY/w_*.png")
    ap.add_argument("--sheet", default="S_贴图对比.jpg")
    ap.add_argument("--dense-top", type=float, default=DENSE_TOP)
    a = ap.parse_args()

    base = Image.open(BASE).convert("RGB")
    W, H = base.size
    print(f"[base] {W}x{H}  {BASE}")

    if a.glyphs:
        paths = [Path(p) for p in a.glyphs.split(",") if p.strip()]
    else:
        paths = sorted(GLY.glob("w_*.png"))

    tiles = [("ORIG MRCHOSR", Image.open(ORIG).convert("RGB"))]
    for p in paths:
        try:
            al, geo = glyph_alpha(p, W, H, a.dense_top)
        except SystemExit as e:
            print("  !", e)
            continue
        out = compose(base, al)
        dst = OUT / f"comp_{p.stem}.jpg"
        out.save(dst, quality=93)
        tiles.append((f"{p.stem}  bg={geo['bg_med']:.2f} slab={geo['slab'][0]}x{geo['slab'][1]}",
                      out))
        print(f"  ok {p.name}  bg_med={geo['bg_med']:.2f} inv={geo['inverted']} "
              f"slab={geo['slab']} pos={geo['pos']} -> {dst.name}")

    # 对比拼版：原标题带 + 各候选标题带
    s = [(l, band(im)) for l, im in tiles]
    th = max(x[1].height for x in s)
    gap, cap = 10, 24
    cols = 1
    sh = Image.new("RGB", (s[0][1].width + 2 * gap, (th + cap) * len(s) + gap), (22, 22, 22))
    d = ImageDraw.Draw(sh)
    for i, (l, im) in enumerate(s):
        y = gap + i * (th + cap)
        sh.paste(im, (gap, y + cap))
        d.text((gap + 4, y + 5), l, fill=(255, 214, 110))
    sh.save(OUT / a.sheet, quality=93)
    print("sheet ->", OUT / a.sheet, sh.size)


if __name__ == "__main__":
    main()
