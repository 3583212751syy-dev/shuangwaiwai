#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v165 burn EAGLSAGE — death metal logo 风格 PIL 烧字（修正版）
- 取 stageA_4x.jpg 为底图（干净无字）
- 烧在正下方独立黑色 banner（不在原 5 字母槽）
- PirataOne 哥特 + 黑边描红 + 上下尖刺
- USM 锐化
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs")

# 取 v163 center_fixed.png 为底图（v163 真身：大秃鹰为主体）
BASE = ROOT / "jobs" / "smoke_v163" / "center_fixed.png"
BASE_4X = ROOT / "jobs" / "smoke_v163" / "stageA_4x.jpg"
out = DESK / "image-fission-v165-eagle_2-EAGLSAGE-logo.jpg"
out_compare = DESK / "image-fission-v165-eagle_2-LOGO-compare.jpg"

print(f"[load] {BASE}")
im = Image.open(BASE).convert("RGB")
# center_fixed 只有 912x1216，先 4x 上采样
if im.size[0] < 2000:
    print(f"[upscale] {im.size} -> 4x")
    im = im.resize((im.size[0]*4, im.size[1]*4), Image.LANCZOS)
W, H = im.size
print(f"[size] {W}x{H}")

# ---------- 1. 在底图下方加黑色 banner ----------
banner_h = int(H * 0.11)
canvas = Image.new("RGB", (W, H + banner_h), (0, 0, 0))
canvas.paste(im, (0, 0))
# banner 上半加一条红橙分隔线（呼应原图火焰）
d_pre = ImageDraw.Draw(canvas)
sep_y = H + 5
d_pre.rectangle((0, sep_y, W, sep_y + 6), fill=(180, 60, 20))

out_base = canvas.copy()

# ---------- 2. 渲染 EAGLSAGE 文字到 LOGO 层 ----------
WORD = "EAGLSAGE"
FONT_PATH = "E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf"

# 字体大小：占 banner 高度 75%
target_h = int(banner_h * 0.72)
logo_w = int(W * 0.85)
logo_h = int(target_h * 2.6)  # 给上下尖刺留余

logo_canvas = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
ldraw = ImageDraw.Draw(logo_canvas)

core_size = int(target_h * 0.55)
font_core = ImageFont.truetype(FONT_PATH, core_size)

bbox = ldraw.textbbox((0, 0), WORD, font=font_core, anchor="lt")
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
tx = (logo_w - tw) // 2 - bbox[0]
ty = (logo_h - th) // 2 - bbox[1]
print(f"[bbox] {bbox}, word={WORD} w={tw} h={th} -> ({tx},{ty})")

# 黑色厚外描边（多次）
shadow_size = int(core_size * 1.03)
font_shadow = ImageFont.truetype(FONT_PATH, shadow_size)
for dx in range(-5, 6):
    for dy in range(-5, 6):
        if dx*dx + dy*dy <= 25 and (dx != 0 or dy != 0):
            ldraw.text((tx+dx, ty+dy), WORD, font=font_shadow, fill=(0, 0, 0, 255))

# 主文字：白色反白
ldraw.text((tx, ty), WORD, font=font_core, fill=(255, 255, 255, 255))

# ---------- 3. 上下尖刺（death metal logo 特征）----------
letter_x0 = tx
letter_x1 = tx + tw
y_top = ty
y_bot = ty + th

n_spikes = 40
spike_h_up = core_size * 0.50
spike_h_dn = core_size * 0.50

for i in range(n_spikes):
    x_left = letter_x0 + (i / n_spikes) * tw
    x_right = letter_x0 + ((i+1) / n_spikes) * tw
    x_mid = (x_left + x_right) / 2
    # 上尖刺
    tri = [(x_left, y_top), (x_right, y_top), (x_mid, y_top - spike_h_up)]
    ldraw.polygon(tri, fill=(0, 0, 0, 255))
    # 内白
    inner_top = [
        ((x_left + x_mid) / 2, y_top),
        ((x_right + x_mid) / 2, y_top),
        (x_mid, y_top - spike_h_up * 0.75)
    ]
    ldraw.polygon(inner_top, fill=(255, 255, 255, 255))
    # 下尖刺
    tri2 = [(x_left, y_bot), (x_right, y_bot), (x_mid, y_bot + spike_h_dn)]
    ldraw.polygon(tri2, fill=(0, 0, 0, 255))
    inner_dn = [
        ((x_left + x_mid) / 2, y_bot),
        ((x_right + x_mid) / 2, y_bot),
        (x_mid, y_bot + spike_h_dn * 0.75)
    ]
    ldraw.polygon(inner_dn, fill=(255, 255, 255, 255))

print(f"[spikes] up={n_spikes} dn={n_spikes}")

# ---------- 4. 边缘柔化 + 合成 ----------
logo_layer = logo_canvas.filter(ImageFilter.SMOOTH)

# 居中粘贴到 banner 区
paste_x = (W - logo_w) // 2
paste_y = H + (banner_h - logo_h) // 2
print(f"[paste] ({paste_x},{paste_y}) size {logo_w}x{logo_h}")

out_im = out_base.copy()
out_im.paste(logo_layer, (paste_x, paste_y), logo_layer)

# USM 锐化
out_im = out_im.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))

out_im.save(out, "JPEG", quality=92, optimize=True)
print(f"[ok] {out.name} {out.stat().st_size/1024/1024:.2f}MB")

# ---------- 5. 拼图对照：原图（center_fixed 4x） | LOGO 版本 ----------
orig = Image.open(BASE).convert("RGB")
if orig.size[0] < 2000:
    orig = orig.resize((orig.size[0]*4, orig.size[1]*4), Image.LANCZOS)
right = out_im.copy()

H_show = 1600
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
d.text((20, 22), "v163 eagle_2 stageA_4x (clean, no logo)", fill=(255, 255, 200))
d.text((r1.width + gap + 20, 22), "v165 + EAGLSAGE (death metal logo style)", fill=(255, 255, 200))
canvas.save(out_compare, "JPEG", quality=88, optimize=True)
print(f"[compare] {out_compare.name} {out_compare.stat().st_size/1024/1024:.2f}MB")
