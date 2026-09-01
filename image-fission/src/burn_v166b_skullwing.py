#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v166b metal_6 logo 改进版
- 用 MetalMania 字体（更接近 death metal logo 风格）+ 字母逐字旋转 + 加形态各异的尖刺钩
- AI 自主选词保持 SKULLWING（SKULL+WING，对应图元素）
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs")

BASE = ROOT / "jobs" / "smoke_v164" / "v164_metal_6.jpg"
out = DESK / "image-fission-v166b-metal_6-SKULLWING-logo.jpg"
out_compare = DESK / "image-fission-v166b-metal_6-LOGO-compare.jpg"

WORD = "SKULLWING"

print(f"[load] {BASE}")
im = Image.open(BASE).convert("RGB")
W, H = im.size

# 顶部 logo 字矩形
mask_y0 = int(H * 0.005)
mask_y1 = int(H * 0.20)
mask_x0 = int(W * 0.04)
mask_x1 = int(W * 0.96)

out_im = im.copy()
overlay = Image.new("RGB", (mask_x1 - mask_x0, mask_y1 - mask_y0), (0, 0, 0))
out_im.paste(overlay, (mask_x0, mask_y0))
# 红橙分隔线
d_pre = ImageDraw.Draw(out_im)
sep_y = mask_y1 + 8
if sep_y + 4 < H:
    d_pre.rectangle((0, sep_y, W, sep_y + 3), fill=(180, 60, 20))

# ---------- 渲染 logo 层 ----------
FONT_PATHS = [
    "E:/Desktop/双接口/image-fission/fonts/MetalMania-Regular.ttf",
    "E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf",
]
font_path = None
for fp in FONT_PATHS:
    if Path(fp).exists():
        font_path = fp
        break

target_h = int((mask_y1 - mask_y0) * 0.62)
logo_w = int(W * 0.90)
logo_h = int(target_h * 3.0)

logo_canvas = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
ldraw = ImageDraw.Draw(logo_canvas)

core_size = int(target_h * 0.50)
font_core = ImageFont.truetype(font_path, core_size)

# ---------- 逐字母渲染，每字错位旋转 ----------
random.seed(7)
x_cursor = int(logo_w * 0.04)
y_base = (logo_h - core_size) // 2

letters = []
for i, ch in enumerate(WORD):
    # 每个字母大小随机微抖
    sz = int(core_size * random.uniform(0.92, 1.08))
    f = ImageFont.truetype(font_path, sz)
    bbox = ldraw.textbbox((0, 0), ch, font=f, anchor="lt")
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # 字母位置
    y = y_base + random.randint(-10, 10)
    # 每个字母单独画到独立小层 → 旋转 → 贴回 canvas
    letter_im = Image.new("RGBA", (w + 100, h + 200), (0, 0, 0, 0))
    ld = ImageDraw.Draw(letter_im)
    # 黑边描红
    shadow_sz = int(sz * 1.04)
    fs = ImageFont.truetype(font_path, shadow_sz)
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            if dx*dx + dy*dy <= 25 and (dx != 0 or dy != 0):
                ld.text((50+dx, 100+dy), ch, font=fs, fill=(0, 0, 0, 255))
    # 主字：白
    ld.text((50, 100), ch, font=f, fill=(255, 255, 255, 255))
    # 旋转角度随机
    angle = random.uniform(-8, 8)
    rotated = letter_im.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
    # 贴回
    r_w, r_h = rotated.size
    paste_x = x_cursor - 50
    paste_y = y - 100
    # 边界
    if paste_x < 0: paste_x = 0
    if paste_x + r_w > logo_w: r_w = logo_w - paste_x
    if paste_y < 0: paste_y = 0
    if paste_y + r_h > logo_h: r_h = logo_h - paste_y
    if r_w > 0 and r_h > 0:
        logo_canvas.paste(rotated, (paste_x, paste_y), rotated)
    letters.append((x_cursor, y, w, h))
    x_cursor += int(w * 0.85) + random.randint(-5, 5)

# 在字母上下再覆盖一层密集尖刺
# 计算整体字矩形
if letters:
    min_x = min(l[0] for l in letters) - 10
    max_x = max(l[0] + l[2] for l in letters) + 10
    min_y = min(l[1] for l in letters) - 5
    max_y = max(l[1] + l[3] for l in letters) + 5

    n_spikes = 80
    span = max_x - min_x
    for i in range(n_spikes):
        x_pos = min_x + (i / n_spikes) * span
        # 上尖刺
        spike_h = core_size * random.uniform(0.4, 0.95)
        tri = [(x_pos - 12, min_y), (x_pos + 12, min_y), (x_pos, min_y - spike_h)]
        ldraw.polygon(tri, fill=(0, 0, 0, 255))
        # 内白
        inner = [
            (x_pos - 6, min_y),
            (x_pos + 6, min_y),
            (x_pos, min_y - spike_h * 0.7)
        ]
        ldraw.polygon(inner, fill=(255, 255, 255, 255))
        # 下尖刺
        tri2 = [(x_pos - 12, max_y), (x_pos + 12, max_y), (x_pos, max_y + spike_h)]
        ldraw.polygon(tri2, fill=(0, 0, 0, 255))
        inner2 = [
            (x_pos - 6, max_y),
            (x_pos + 6, max_y),
            (x_pos, max_y + spike_h * 0.7)
        ]
        ldraw.polygon(inner2, fill=(255, 255, 255, 255))

print(f"[logo] {len(WORD)} chars, spikes={n_spikes*2}")

# ---------- 柔化 + 合成 ----------
logo_layer = logo_canvas.filter(ImageFilter.SMOOTH)

paste_x = mask_x0 + (mask_x1 - mask_x0 - logo_w) // 2
paste_y = mask_y0 + ((mask_y1 - mask_y0) - logo_h) // 2
if paste_x < mask_x0: paste_x = mask_x0
if paste_y < mask_y0 - 50: paste_y = mask_y0 - 50

print(f"[paste] ({paste_x},{paste_y})")
out_im.paste(logo_layer, (paste_x, paste_y), logo_layer)

# USM 锐化
final = out_im.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=3))
final.save(out, "JPEG", quality=92, optimize=True)
print(f"[ok] {out.name} {out.stat().st_size/1024/1024:.2f}MB")

# 拼图对照
orig = Image.open(BASE).convert("RGB")
right = final.copy()
H_show = 1800
def fit(im, H_target):
    w = int(im.width * H_target / im.height)
    return im.resize((w, H_target), Image.LANCZOS)

r1 = fit(orig, H_show)
r2 = fit(right, H_show)
gap = 30
canvas = Image.new("RGB", (r1.width + r2.width + gap, H_show + 80), (24, 24, 24))
canvas.paste(r1, (0, 80))
canvas.paste(r2, (r1.width + gap, 80))
d = ImageDraw.Draw(canvas)
d.text((20, 22), "v164 metal_6 (illegal logo)", fill=(255, 255, 200))
d.text((r1.width + gap + 20, 22), "v166b + SKULLWING (MetalMania + rotation)", fill=(255, 255, 200))
canvas.save(out_compare, "JPEG", quality=88, optimize=True)
print(f"[compare] {out_compare.name} {out_compare.stat().st_size/1024/1024:.2f}MB")
