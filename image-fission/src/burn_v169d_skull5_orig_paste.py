#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v169d: 从原图直接提取文字像素（红字+黑边），贴到 v164 裂变底图
- 主体（骷髅/蛇/玫瑰/血）用 v164 裂变版（v147 禁词干净底图）
- 文字 TRUE/NEVER/DIES 100% 原图字体原样保留（像素级 1:1，不重烧）
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs")
ORIG = ROOT / "ComfyUI" / "input" / "pinterest_skull_5.jpg"
BASE = ROOT / "jobs" / "smoke_v164" / "v164_skull_5.jpg"
OUT = DESK / "image-fission-v169d-skull_5-orig-text-pasted.jpg"
OUT_COMPARE = DESK / "image-fission-v169d-skull_5-compare.jpg"


def main():
    orig = Image.open(ORIG).convert("RGB")
    base = Image.open(BASE).convert("RGB")
    W, H = base.size
    print(f"[base] {W}x{H}")
    print(f"[orig] {orig.size}")

    # 把原图 resize 到 base 大小（stretch，保持像素级对齐）
    orig_r = orig.resize((W, H), Image.LANCZOS)
    arr = np.array(orig_r)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 红字掩码：红色 + 红明显大于绿蓝
    red_mask = (R > 90) & (R - G > 40) & (R - B > 40)
    # 黑边掩码：黑色像素
    black_mask = (R < 60) & (G < 60) & (B < 60)

    # 扩展 red_mask 包含邻接黑边（PIL MaxFilter 形态学膨胀）
    red_mask_img = Image.fromarray((red_mask * 255).astype("uint8"))
    expanded_img = red_mask_img.filter(ImageFilter.MaxFilter(7))  # 扩展 7 像素
    expanded = np.array(expanded_img) > 128

    # 文字像素 = 扩展区 ∩ (红字 OR 黑边)
    text_mask = expanded & (red_mask | black_mask)
    print(f"[text pixels] {text_mask.sum()} ({text_mask.sum()/text_mask.size*100:.2f}%)")

    # 转 RGBA 图层（仅文字像素不透明，其他透明）
    orig_rgba = orig_r.convert("RGBA")
    alpha = (text_mask * 255).astype("uint8")
    orig_rgba.putalpha(Image.fromarray(alpha))

    # 粘贴到 v164 底图
    base_rgba = base.convert("RGBA")
    base_rgba.paste(orig_rgba, (0, 0), orig_rgba)
    out_im = base_rgba.convert("RGB")

    out_im.save(OUT, "JPEG", quality=92, optimize=True)
    print(f"[ok] {OUT.name}")

    # ---------- 四联对照：原图 | v164干净 | v169B PIL烧字 | v169d 原图文字粘贴 ----------
    v169b = Image.open(DESK / "image-fission-v169b-skull_5-original-text.jpg").convert("RGB")
    H_show = 1450
    gap = 20

    def fit(im):
        w = int(im.width * H_show / im.height)
        return im.resize((w, H_show), Image.LANCZOS)

    panels = [
        (fit(orig), "原图 TRUE/NEVER/DIES (字体参考)"),
        (fit(Image.open(BASE)), "v164 裂变底图 (v147 禁词干净)"),
        (fit(v169b), "v169B PIL 烧字 (PirataOne 9/10)"),
        (fit(out_im), "v169d 原图文字像素直贴 (10/10 字体 1:1)"),
    ]
    total_w = sum(p.width for p, _ in panels) + gap * (len(panels) - 1)
    header_h = 70
    canvas = Image.new("RGB", (total_w, H_show + header_h), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    x = 0
    for p, lb in panels:
        canvas.paste(p, (x, header_h))
        d.text((x + 14, 22), lb, fill=(255, 235, 200))
        x += p.width + gap
    canvas.save(OUT_COMPARE, "JPEG", quality=86, optimize=True)
    print(f"[compare] {OUT_COMPARE.name}")


if __name__ == "__main__":
    main()