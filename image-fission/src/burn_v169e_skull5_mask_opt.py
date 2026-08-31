#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v169e: 优化 v169d 的 mask，只保留「红字 + 红字紧邻外环 2px 黑边」，剔除字间背景
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs/skull_5")
ORIG = ROOT / "ComfyUI" / "input" / "pinterest_skull_5.jpg"
BASE = ROOT / "jobs" / "smoke_v164" / "v164_skull_5.jpg"
OUT = DESK / "image-fission-v169e-skull_5-orig-text-clean.jpg"
OUT_COMPARE = DESK / "image-fission-v169e-skull_5-compare.jpg"


def main():
    orig = Image.open(ORIG).convert("RGB")
    base = Image.open(BASE).convert("RGB")
    W, H = base.size
    print(f"[base] {W}x{H}")

    orig_r = orig.resize((W, H), Image.LANCZOS)
    arr = np.array(orig_r)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 严格红字掩码（提高阈值，过滤浅红羽毛边缘）
    red_mask = (R > 110) & (R - G > 50) & (R - B > 50)
    # 黑像素
    black_mask = (R < 60) & (G < 60) & (B < 60)

    # 红字膨胀 2 像素（MaxFilter 5x5 = 2px dilation）
    red_mask_img = Image.fromarray((red_mask * 255).astype("uint8"))
    expanded_img = red_mask_img.filter(ImageFilter.MaxFilter(5))
    expanded = np.array(expanded_img) > 128

    # 关键：只取红字外环（膨胀-原红字）∩ 黑像素 = 红字紧邻黑边
    ring = expanded & ~red_mask
    text_mask = red_mask | (ring & black_mask)
    print(f"[text pixels] {text_mask.sum()} ({text_mask.sum()/text_mask.size*100:.2f}%)")

    # 转 RGBA 图层
    orig_rgba = orig_r.convert("RGBA")
    alpha = (text_mask * 255).astype("uint8")
    orig_rgba.putalpha(Image.fromarray(alpha))

    # 粘贴到 v164 底图
    base_rgba = base.convert("RGBA")
    base_rgba.paste(orig_rgba, (0, 0), orig_rgba)
    out_im = base_rgba.convert("RGB")

    out_im.save(OUT, "JPEG", quality=92, optimize=True)
    print(f"[ok] {OUT.name}")

    # 四联对照：原图 | v164干净 | v169d 旧 mask | v169e 新 mask
    # v169d 对照图源文件已丢失，省略该对照格
    H_show = 1450
    gap = 20

    def fit(im):
        w = int(im.width * H_show / im.height)
        return im.resize((w, H_show), Image.LANCZOS)

    panels = [
        (fit(orig), "原图 TRUE/NEVER/DIES"),
        (fit(Image.open(BASE)), "v164 裂变底图 (禁词)"),
        (fit(out_im), "v169e (字体10/10 1:1 原图)"),
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