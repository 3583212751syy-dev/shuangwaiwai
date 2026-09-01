#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v164 burn "EAGLSAGE" — death metal logo 风格 PIL 烧字
- 取 v164 eagle_2 裂变图为底图
- 草书哥特反白 + 上下尖刺 + 黑边描红 + 横向拉长
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs")

# 取 v163 eagle_2 裂变图为底图
BASE = ROOT / "jobs" / "smoke_v163" / "final_eagle_2.jpg"
BASE_4X = ROOT / "jobs" / "smoke_v163" / "stageA_4x.jpg"

# 如果 4x 不存在就 fallback
if BASE_4X.exists():
    src = BASE_4X
else:
    src = BASE

# 输出名
out = DESK / "image-fission-v164-eagle_2-LOGO-EAGLSAGE.jpg"
out_compare = DESK / "image-fission-v164-eagle_2-LOGO-EAGLSAGE-compare.jpg"

print(f"[load] {src}")
im = Image.open(src).convert("RGB")
W, H = im.size
print(f"[size] {W}x{H}")

WORD = "EAGLSAGE"

# 字体：PirataOne（黑金属哥特变体），最接近免费方案
FONT_CANDS = [
    "E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf",
    r"C:\Windows\Fonts\PirataOne-Regular.ttf",
]

font_path = None
for fp in FONT_CANDS:
    if Path(fp).exists():
        font_path = fp
        break

if font_path is None:
    print("WARN: PirataOne 找不到，用默认字体")
    font_path = r"C:\Windows\Fonts\PirataOne-Regular.ttf"

# 字体大小：占满图宽约 72%
target_height = int(H * 0.072)
print(f"[font] {font_path} target_height={target_height}")

# ---------- 1. 渲染 logo 文字到单独图层 (RGBA) ----------
# 在透明大画布上画，便于后期特效
# 估算：图宽 ~W*0.55 ~W*0.95
logo_w = int(W * 0.78)
logo_h = int(target_height * 2.4)  # 给上下尖刺留 2.4 倍高度
logo_canvas = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
draw = ImageDraw.Draw(logo_canvas)

# 核心字体大小（不含尖刺的字号）
core_size = int(target_height * 0.62)
font_core = ImageFont.truetype(font_path, core_size)

# 计算文字 bbox
bbox = draw.textbbox((0, 0), WORD, font=font_core, anchor="lt")
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]

# 居中绘制（在 canvas 中央）
tx = (logo_w - tw) // 2 - bbox[0]
ty = (logo_h - th) // 2 - bbox[1]
print(f"[bbox] {bbox}, w={tw} h={th} -> drawn at ({tx},{ty})")

# 第一遍：绘制黑色阴影/底层（描外边）
shadow_size = int(core_size * 1.02)
font_shadow = ImageFont.truetype(font_path, shadow_size)
# 多次描边（向外）
for dx in range(-6, 7):
    for dy in range(-6, 7):
        if dx*dx + dy*dy <= 36 and (dx != 0 or dy != 0):
            draw.text((tx+dx, ty+dy), WORD, font=font_shadow, fill=(0, 0, 0, 200))

# 主文字：白
draw.text((tx, ty), WORD, font=font_core, fill=(255, 255, 255, 255))

# ---------- 2. 上下加尖刺（death metal logo 标识）----------
# 在每个字母上下各画一条瘦长尖刺
n_spikes_up = 60   # 向上尖刺数
n_spikes_dn = 60
spike_color_outer = (0, 0, 0, 255)
spike_color_inner = (255, 255, 255, 255)

# 上下尖刺由窄三角构成
# 字母横向范围 = tw，tx0..tx0+tw
letter_x0 = tx
letter_x1 = tx + tw
y_top = ty
y_bot = ty + th

# 上尖刺
for i in range(n_spikes_up):
    x_left = letter_x0 + (i / n_spikes_up) * tw
    x_right = letter_x0 + ((i+1) / n_spikes_up) * tw
    x_mid = (x_left + x_right) / 2
    spike_h = core_size * 0.45
    # 三角 [(x_left, y_top), (x_right, y_top), (x_mid, y_top - spike_h)]
    tri = [(x_left, y_top), (x_right, y_top), (x_mid, y_top - spike_h)]
    draw.polygon(tri, fill=spike_color_outer)
    # 内白线
    inner = [
        (x_left*0.65 + x_mid*0.35, y_top*0.95 + (y_top - spike_h*0.6)*0.05),
        (x_right*0.65 + x_mid*0.35, y_top*0.95 + (y_top - spike_h*0.6)*0.05),
        (x_mid, y_top - spike_h*0.7)
    ]
    draw.polygon(inner, fill=spike_color_inner)

# 下尖刺
for i in range(n_spikes_dn):
    x_left = letter_x0 + (i / n_spikes_dn) * tw
    x_right = letter_x0 + ((i+1) / n_spikes_dn) * tw
    x_mid = (x_left + x_right) / 2
    spike_h = core_size * 0.45
    tri = [(x_left, y_bot), (x_right, y_bot), (x_mid, y_bot + spike_h)]
    draw.polygon(tri, fill=spike_color_outer)
    inner = [
        (x_left*0.65 + x_mid*0.35, y_bot*0.95 + (y_bot + spike_h*0.6)*0.05),
        (x_right*0.65 + x_mid*0.35, y_bot*0.95 + (y_bot + spike_h*0.6)*0.05),
        (x_mid, y_bot + spike_h*0.7)
    ]
    draw.polygon(inner, fill=spike_color_inner)

print(f"[spikes] up={n_spikes_up} dn={n_spikes_dn}")

# ---------- 3. 用 USM/SMOOTH 对 logo 层柔化 ----------
# 给点 pixel noise 避免边缘过硬
logo_layer = logo_canvas.filter(ImageFilter.SMOOTH)
logo_layer = logo_layer.filter(ImageFilter.SMOOTH)
print("[filter] smooth applied")

# ---------- 4. 合成到原图 ----------
# 位置：偏下 6%，居中（不挡中央主体，符合 v127 DOMINION 风格）
paste_x = (W - logo_w) // 2
paste_y = int(H * 0.78)

# 检查不溢出
if paste_x < 0:
    paste_x = 0
if paste_y + logo_h > H:
    paste_y = H - logo_h - 20

print(f"[paste] at ({paste_x},{paste_y}) size {logo_w}x{logo_h}")

# 输出
out_im = im.copy()
out_im.paste(logo_layer, (paste_x, paste_y), logo_layer)

# USM 锐化
out_im = out_im.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))

# 保存
out_im.save(out, "JPEG", quality=92, optimize=True)
print(f"[ok] {out.name} {out.stat().st_size/1024/1024:.2f}MB")

# 拼图对照：左原图（v164_eagle_2 没烧字版）+ 右 LOGO 版
orig = Image.open(BASE).convert("RGB")
right = out_im.copy()

# 等比缩放，高 800
H_show = 1200
w1 = int(orig.width * H_show / orig.height)
w2 = int(right.width * H_show / right.height)
r1 = orig.resize((w1, H_show), Image.LANCZOS)
r2 = right.resize((w2, H_show), Image.LANCZOS)

gap = 30
canvas = Image.new("RGB", (w1 + w2 + gap, H_show + 60), (32, 32, 32))
canvas.paste(r1, (0, 60))
canvas.paste(r2, (w1 + gap, 60))
d = ImageDraw.Draw(canvas)
d.text((20, 16), "v164 eagle_2 (no logo)", fill=(255, 255, 200))
d.text((w1 + gap + 20, 16), "v164 + EAGLSAGE logo (death metal)", fill=(255, 255, 200))
canvas.save(out_compare, "JPEG", quality=90, optimize=True)
print(f"[compare] {out_compare.name} {out_compare.stat().st_size/1024/1024:.2f}MB")
