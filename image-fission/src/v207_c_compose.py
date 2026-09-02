"""
v207_c — v206 升级：字体 Lora-VF Bold（替代 timesbd.ttf）+ 1px 白描边 + USM 锐化

对比 v206_c 的改动（每个字用途已明确）：
  - **F_TIMES_BD → F_LORA_VF + set_variation_by_axes([700])** = Didone 罗马衬线同族替代
    用途：解决 v206 字体 Laplacian 193（仅原图 31%）的核心瓶颈
  - **每字额外加 1px 白色描边**（在黑边 mask 上叠白边）
    用途：让烧字在紫底上对比度从纯黑边升级为 "白底+黑边"，接近原图 BACARDÍ 高对比
  - **ImageFilter.UnsharpMask(radius=2, percent=80)** 整体锐化
    用途：补偿 PIL 烧字 + LANCZOS 缩放的边缘软化

输入（沿用 v206 三件套，不重跑 SDXL）：
  - jobs/smoke_v206/_v206_a_clean.png  — 全紫底干净画布
  - jobs/smoke_v206/v206_b_newbat.png   — SDXL 已生成的新蝙蝠
  - jobs/smoke_v206/_v206_a_bat_info.json — bat_bbox / bat_center / BAT_R

输出：
  - jobs/smoke_v206/v207_bat_logo_lora_final.jpg  — 主产物
  - jobs/smoke_v206/_compare_v207.jpg             — 对照图
"""
import sys
import os
import math
import json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT = Path("E:/Desktop/双接口/image-fission")
JOB = PROJECT / "jobs" / "smoke_v206"  # 复用 v206 的中间产物
CLEAN_IN = JOB / "_v206_a_clean.png"
NEWBAT_IN = JOB / "v206_b_newbat.png"
INFO_IN = JOB / "_v206_a_bat_info.json"
OUT_FINAL = JOB / "v207_bat_logo_lora_final.jpg"
OUT_COMPARE = JOB / "_compare_v207.jpg"
print(f"[job] {JOB}")

info = json.loads(INFO_IN.read_text())
W = info["W"]; H = info["H"]
bx, by, bw, bh = info["bat_bbox"]
bcx, bcy = info["bat_center"]
BAT_R = info["bat_R"]
BG_PURPLE = tuple(info["BG_PURPLE"])
RING_DEEP = tuple(info["RING_DEEP"])

# ---- 配色 ----
WHITE = (255, 255, 255)
BAT_BLACK = (23, 4, 23)
DEEP_PURPLE = (45, 10, 60)

# ---- 字体：Lora-VF Bold 700（替代 timesbd.ttf，Didone 罗马衬线同族）----
F_LORA = r"E:/Desktop/双接口/image-fission/fonts/Lora-VF.ttf"


def lora(size: int, weight: int = 700) -> ImageFont.FreeTypeFont:
    """Load Lora-VF font with specified weight axis (400-700)."""
    f = ImageFont.truetype(F_LORA, size)
    try:
        f.set_variation_by_axes([weight])
    except Exception as e:
        print(f"[warn] set_variation_by_axes({weight}) failed: {e}")
    return f


def draw_text_double_outline(draw: ImageDraw.ImageDraw, xy: tuple, text: str,
                             font: ImageFont.FreeTypeFont, fg=(45, 10, 60),
                             outline_inner=(0, 0, 0), outline_outer_white=(255, 255, 255),
                             r_inner: int = 5, r_outer: int = 1) -> None:
    """Triple-layer text: 5x5 black mask (v206 style) + 1px white outer ring (NEW v207).

    用法：先 5x5 黑边 mask 保证字体边缘清晰，再叠 1px 白色外描边，
    模拟原图 BACARDÍ 高对比的字底效果。
    """
    x, y = xy
    # 1) 5x5 black mask (v206 风格 — 保留)
    for dx in range(-r_inner, r_inner + 1):
        for dy in range(-r_inner, r_inner + 1):
            if dx * dx + dy * dy <= r_inner * r_inner:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_inner)
    # 2) 1px white outer outline (NEW)
    for dx in range(-r_outer - r_inner, r_outer + r_inner + 1):
        for dy in range(-r_outer - r_inner, r_outer + r_inner + 1):
            # ring-shaped: in inner 5x5 black zone OR in the outer ring just outside
            d2 = dx * dx + dy * dy
            if r_inner * r_inner < d2 <= (r_inner + r_outer) * (r_inner + r_outer):
                draw.text((x + dx, y + dy), text, font=font, fill=outline_outer_white)
    # 3) foreground text
    draw.text((x, y), text, font=font, fill=fg)


# ---- [1] 读 v206_a 干净紫底 ----
clean_pil = Image.open(CLEAN_IN).convert("RGBA")
print(f"[1] clean: {clean_pil.size}")

# ---- [2] 读 SDXL 新蝙蝠 + 紫调重映射（同 v206）----
newbat_pil = Image.open(NEWBAT_IN).convert("RGB")
print(f"[2] newbat: {newbat_pil.size}")
_nb = np.array(newbat_pil, np.float32)
_lum = _nb.mean(axis=2)
_lum_n = (_lum - _lum.min()) / (max(_lum.max() - _lum.min(), 1e-3))
_deep = np.array(DEEP_PURPLE, np.float32)
_bg = np.array(BG_PURPLE, np.float32)
_mapped = _deep[None, None, :] + _lum_n[:, :, None] * (_bg - _deep)[None, None, :]
dark = (_lum_n < 0.45).astype(np.uint8) * 255
num, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
bat_mask = lab == largest
_mapped[~bat_mask] = _bg
newbat_pil = Image.fromarray(_mapped.astype(np.uint8)).convert("RGBA")
print(f"[2b] 蝙蝠剪影提取完成(最大连通域={largest})")

# ---- [3] resize + paste ----
target_size = BAT_R * 2
newbat_resized = newbat_pil.resize((target_size, target_size), Image.LANCZOS)
paste_x = bcx - BAT_R
paste_y = bcy - BAT_R
canvas = clean_pil.copy()
canvas.paste(newbat_resized, (paste_x, paste_y), newbat_resized)
print(f"[3] pasted ({paste_x},{paste_y}) size={target_size}")

draw = ImageDraw.Draw(canvas)

# ---- [4] 顶弧 MOONCREST (Lora Bold) ----
arc_text = "MOONCREST"
arc_radius = BAT_R + 70
arc_size = int(BAT_R * 0.16)
f_arc = lora(arc_size, weight=700)
total_angle = 110
n_chars = len(arc_text)
for i, ch in enumerate(arc_text):
    t = (i + 0.5) / n_chars
    deg_pil = 270 - t * total_angle
    rad_pil = math.radians(deg_pil)
    x = bcx + arc_radius * math.cos(rad_pil)
    y = bcy + arc_radius * math.sin(rad_pil)
    ch_img = Image.new("RGBA", (arc_size * 3, arc_size * 3), (0, 0, 0, 0))
    d_ch = ImageDraw.Draw(ch_img)
    # 单字小 — 不加白描边（会糊掉弧字）
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx * dx + dy * dy <= 4:
                d_ch.text((arc_size + dx, arc_size + dy), ch, font=f_arc, fill=BAT_BLACK)
    d_ch.text((arc_size, arc_size), ch, font=f_arc, fill=DEEP_PURPLE)
    rot_angle = -math.degrees(rad_pil) - 90
    ch_rot = ch_img.rotate(rot_angle, resample=Image.BICUBIC, center=(arc_size, arc_size))
    canvas.paste(ch_rot, (int(x - arc_size), int(y - arc_size)), ch_rot)
print(f"[4] MOONCREST 顶弧 radius={arc_radius} (Lora Bold)")

# ---- [5] Est. 左侧 (Lora Bold + 白描边) ----
est_text = "EST."
est_size = int(BAT_R * 0.20)
f_est = lora(est_size, weight=700)
est_y = bcy + 30
est_x = bx - 100
draw_text_double_outline(draw, (est_x, est_y), est_text, f_est,
                         fg=DEEP_PURPLE, outline_inner=BAT_BLACK,
                         outline_outer_white=WHITE, r_inner=2, r_outer=1)
print(f"[5] EST. ({est_x},{est_y}) (Lora Bold + 白描边)")

# ---- [6] MMXXVI 右侧 (Lora Bold + 白描边) ----
year_text = "MMXXVI"
f_year = lora(est_size, weight=700)
year_y = bcy + 30
year_x = bx + bw + 50
draw_text_double_outline(draw, (year_x, year_y), year_text, f_year,
                         fg=DEEP_PURPLE, outline_inner=BAT_BLACK,
                         outline_outer_white=WHITE, r_inner=2, r_outer=1)
print(f"[6] MMXXVI ({year_x},{year_y})")

# ---- [7] 大字 MOONCREST (Lora Bold + 白描边) ----
big_text = "MOONCREST"
big_size = 200
f_big = lora(big_size, weight=700)
bb = draw.textbbox((0, 0), big_text, font=f_big)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
big_x = (W - tw) // 2
big_y = 1050 - bb[1]
draw_text_double_outline(draw, (big_x, big_y), big_text, f_big,
                         fg=DEEP_PURPLE, outline_inner=BAT_BLACK,
                         outline_outer_white=WHITE, r_inner=4, r_outer=1)
print(f"[7] MOONCREST big (Lora Bold 700 + 白描边)")

# ---- [8] 小字 CURSE (Lora Bold + 白描边) ----
small_text = "CURSE"
small_size = 160
f_small = lora(small_size, weight=700)
bb = draw.textbbox((0, 0), small_text, font=f_small)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
small_x = (W - tw) // 2
small_y = 1240 - bb[1]
draw_text_double_outline(draw, (small_x, small_y), small_text, f_small,
                         fg=DEEP_PURPLE, outline_inner=BAT_BLACK,
                         outline_outer_white=WHITE, r_inner=3, r_outer=1)
print(f"[8] CURSE (Lora Bold 700 + 白描边)")

# ---- [9] 倒三角 ----
tri_cx = bcx
tri_top_y = 1370
tri_w = 30; tri_h = 30
draw.polygon([
    (tri_cx, tri_top_y + tri_h),
    (tri_cx - tri_w, tri_top_y),
    (tri_cx + tri_w, tri_top_y),
], fill=DEEP_PURPLE)
print(f"[9] ▼ 倒三角")

# ---- [10] USM 锐化 (NEW v207) ----
# 烧字后整图轻锐化，补偿 LANCZOS 缩放和 PIL 抗锯齿的边缘软化
canvas = canvas.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
print(f"[10] USM UnsharpMask(radius=2, percent=80) applied")

# ---- [11] 保存 ----
canvas = canvas.convert("RGB")
canvas.save(OUT_FINAL, quality=95)
print(f"[save] {OUT_FINAL}")

# ---- [12] 对照图 ----
SRC_ORIG = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
orig = Image.open(SRC_ORIG).convert("RGB")
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 40), (235, 235, 238))
f_lbl = ImageFont.truetype(F_LORA, 28)  # 仅用于标签
f_lbl.set_variation_by_axes([600])
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 80, 4), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 200, 4), "v207  LORA VF Bold + WHITE OUTLINE + USM",
        font=f_lbl, fill=(20, 20, 20))
comp.paste(orig, (0, 40))
comp.paste(canvas, (W + gap, 40))
comp.save(OUT_COMPARE, quality=92)
print(f"[compare] {OUT_COMPARE}")
print("\n[done] v207 完成 — 请跑 qc_split_image.py <orig> <v207_bat_logo_lora_final.jpg> 验证")
