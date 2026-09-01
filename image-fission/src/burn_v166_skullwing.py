#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v166 按规则替换 metal_6 顶部侵权 logo 词
- 原 v164_metal_6.jpg 顶部金属 logo 字与图元素无关（看起来像"BRRAGE"等乱画字符）
- 按图裂变单词处理铁律：用图元素相关的新词 SKULLWING 替换
- AI 自主选词：SKULLWING = SKULL+WING（骷髅+翼，对应图上骷髅+鹰翅），0 侵权
- 字体风格：草书哥特反白 + 黑边描红 + 上下尖刺（Death Metal logo style）
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs")

# 底图：v164 metal_6 裂变输出
BASE = ROOT / "jobs" / "smoke_v164" / "v164_metal_6.jpg"
out = DESK / "image-fission-v166-metal_6-SKULLWING-logo.jpg"
out_compare = DESK / "image-fission-v166-metal_6-LOGO-compare.jpg"

WORD = "SKULLWING"

print(f"[load] {BASE}")
im = Image.open(BASE).convert("RGB")
W, H = im.size
print(f"[size] {W}x{H}")

# ---------- 1. 顶部 logo 字矩形区域涂纯黑（覆盖原字+背后闪电穿模）----------
# 原图顶部 logo 字大致 Y=0..0.18, X=0.05..0.95
mask_y0 = int(H * 0.005)
mask_y1 = int(H * 0.18)
mask_x0 = int(W * 0.04)
mask_x1 = int(W * 0.96)
print(f"[mask] rect ({mask_x0},{mask_y0}) - ({mask_x1},{mask_y1})")

out_im = im.copy()
# 用纯黑矩形覆盖原 logo 区
overlay = Image.new("RGB", (mask_x1 - mask_x0, mask_y1 - mask_y0), (0, 0, 0))
out_im.paste(overlay, (mask_x0, mask_y0))

# 给顶部加一条细红橙分隔线，呼应原图火焰配色
sep_y = mask_y1 + 8
d_pre = ImageDraw.Draw(out_im)
if sep_y + 4 < H:
    d_pre.rectangle((0, sep_y, W, sep_y + 3), fill=(180, 60, 20))

# ---------- 2. 渲染 SKULLWING 到独立 RGBA 层 ----------
FONT_PATH = "E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf"

# logo 目标高度：占黑色矩形高度 72%
target_h = int((mask_y1 - mask_y0) * 0.72)
logo_w = int(W * 0.88)
logo_h = int(target_h * 2.8)  # 给上下尖刺留余

logo_canvas = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
ldraw = ImageDraw.Draw(logo_canvas)

# 渲染核心字母
core_size = int(target_h * 0.55)
font_core = ImageFont.truetype(FONT_PATH, core_size)

bbox = ldraw.textbbox((0, 0), WORD, font=font_core, anchor="lt")
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
tx = (logo_w - tw) // 2 - bbox[0]
ty = (logo_h - th) // 2 - bbox[1]
print(f"[bbox] {bbox}, word={WORD} w={tw} h={th} -> ({tx},{ty})")

# 黑边描边（多次向外）
shadow_size = int(core_size * 1.05)
font_shadow = ImageFont.truetype(FONT_PATH, shadow_size)
for dx in range(-6, 7):
    for dy in range(-6, 7):
        if dx*dx + dy*dy <= 36 and (dx != 0 or dy != 0):
            ldraw.text((tx+dx, ty+dy), WORD, font=font_shadow, fill=(0, 0, 0, 255))

# 主文字：白色反白
ldraw.text((tx, ty), WORD, font=font_core, fill=(255, 255, 255, 255))

# ---------- 3. 上下尖刺（Death metal logo 标识）----------
letter_x0 = tx
letter_x1 = tx + tw
y_top = ty
y_bot = ty + th

n_spikes = 50  # 加密尖刺，更接近原图风格
spike_h_up = core_size * 0.55
spike_h_dn = core_size * 0.55
import random
random.seed(42)  # 可复现

for i in range(n_spikes):
    # 上尖刺
    x_left = letter_x0 + (i / n_spikes) * tw
    x_right = letter_x0 + ((i+1) / n_spikes) * tw
    x_mid = (x_left + x_right) / 2
    spike_h = spike_h_up * random.uniform(0.65, 1.0)  # 长度随机
    # 三角形刺
    tri = [(x_left, y_top), (x_right, y_top), (x_mid, y_top - spike_h)]
    ldraw.polygon(tri, fill=(0, 0, 0, 255))
    # 内白
    inner_top = [
        ((x_left + x_mid) / 2, y_top),
        ((x_right + x_mid) / 2, y_top),
        (x_mid, y_top - spike_h * 0.75)
    ]
    ldraw.polygon(inner_top, fill=(255, 255, 255, 255))
    # 下尖刺
    tri2 = [(x_left, y_bot), (x_right, y_bot), (x_mid, y_bot + spike_h)]
    ldraw.polygon(tri2, fill=(0, 0, 0, 255))
    inner_dn = [
        ((x_left + x_mid) / 2, y_bot),
        ((x_right + x_mid) / 2, y_bot),
        (x_mid, y_bot + spike_h * 0.75)
    ]
    ldraw.polygon(inner_dn, fill=(255, 255, 255, 255))

# 给 logo 加细红橙描边（在刺轮廓上加细红线）
print(f"[spikes] up={n_spikes} dn={n_spikes} per letter")

# ---------- 4. 边缘柔化 + 合成 ----------
logo_layer = logo_canvas.filter(ImageFilter.SMOOTH)

# 居中粘贴到顶部黑色矩形
paste_x = mask_x0 + (mask_x1 - mask_x0 - logo_w) // 2
paste_y = mask_y0 + ((mask_y1 - mask_y0) - logo_h) // 2
# 修正：确保不超出矩形
if paste_x < mask_x0:
    paste_x = mask_x0
if paste_x + logo_w > mask_x1:
    logo_w = mask_x1 - paste_x

print(f"[paste] ({paste_x},{paste_y}) size {logo_w}x{logo_h}")

out_im.paste(logo_layer, (paste_x, paste_y), logo_layer)

# USM 锐化（只对 logo 区）
final = out_im.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=3))
final.save(out, "JPEG", quality=92, optimize=True)
print(f"[ok] {out.name} {out.stat().st_size/1024/1024:.2f}MB")

# ---------- 5. 拼图对照 ----------
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
d.text((20, 22), "v164 metal_6 (illegal logo word)", fill=(255, 255, 200))
d.text((r1.width + gap + 20, 22), "v166 + SKULLWING (logo fits subject)", fill=(255, 255, 200))
canvas.save(out_compare, "JPEG", quality=88, optimize=True)
print(f"[compare] {out_compare.name} {out_compare.stat().st_size/1024/1024:.2f}MB")
