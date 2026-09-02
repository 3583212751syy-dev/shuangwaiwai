"""
v206_c — 完整保留 BACARDÍ 结构 + PIL 烧字（v206_b 元素 + v203 mask 几何）

设计原则（用户 11:17 明确）：
  - **构图保留**：BACARDÍ 5 段垂直布局 1:1
  - **结构保留**：portrait + 中央徽章 + 上方弧字 + 中部大字 + 下方副字 + 倒三角
  - **颜色保留**：紫色配色锁死
  - **元素裂变**：
    * 弧字 LA CASA → MOONCREST
    * 大字 BACARDÍ → MOONCREST
    * 小字 MYHEART → CURSE
    * 侧字 Est. → EST.
    * 侧字 1862 → MMXXVI
    * 中央 bat → SDXL v206_b 新蝙蝠

5 段结构：
  ┌────────────────────────────────────┐
  │        (顶弧 MOONCREST)             │ <- y=300-720
  │     ┌─ 中央徽章 bat ─┐               │
  │     │   (SDXL v206)   │               │ <- bat bbox
  │     │   Est.    1862   │ <- y=820-900 │ <- 侧字
  │     └─────────────┘                  │
  │        MOONCREST 大字                │ <- y=980-1170
  │         CURSE 小字                    │ <- y=1185-1335
  │            ▼                          │ <- y=1370-1440
  └────────────────────────────────────┘
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
JOB = PROJECT / "jobs" / "smoke_v206"
CLEAN_IN = JOB / "_v206_a_clean.png"
NEWBAT_IN = JOB / "v206_b_newbat.png"
INFO_IN = JOB / "_v206_a_bat_info.json"
OUT_FINAL = JOB / "v206_bat_logo_final.jpg"
OUT_COMPARE = JOB / "_compare_v206.jpg"
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

# ---- 字体（系统现有）----
F_TIMES_BD = r"C:/Windows/Fonts/timesbd.ttf"
F_GEORGIA_BD = r"C:/Windows/Fonts/georgiab.ttf"

# ---- [1] 读 v206_a 干净紫底 ----
clean_pil = Image.open(CLEAN_IN).convert("RGBA")
print(f"[1] clean: {clean_pil.size}")

# ---- [2] 读 SDXL 新蝙蝠 + 紫调重映射 + 最大连通域提取蝙蝠剪影 ----
newbat_pil = Image.open(NEWBAT_IN).convert("RGB")
print(f"[2] newbat: {newbat_pil.size}")
_nb = np.array(newbat_pil, np.float32)
_lum = _nb.mean(axis=2)
_lum_n = (_lum - _lum.min()) / (max(_lum.max() - _lum.min(), 1e-3))  # 0..1 暗→亮
_deep = np.array(DEEP_PURPLE, np.float32)      # 暗部=蝙蝠深紫
_bg = np.array(BG_PURPLE, np.float32)          # 画布紫(无缝)
_mapped = _deep[None, None, :] + _lum_n[:, :, None] * (_bg - _deep)[None, None, :]
# 提取最大深色连通域 = 蝙蝠剪影；其余(含紫渐变背景)压平为画布紫→消除深紫圆盘
dark = (_lum_n < 0.45).astype(np.uint8) * 255
num, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
bat_mask = lab == largest
_mapped[~bat_mask] = _bg
newbat_pil = Image.fromarray(_mapped.astype(np.uint8)).convert("RGBA")
print(f"[2b] 蝙蝠剪影提取完成(最大连通域={largest}) 背景压平为画布紫")

# ---- [3] resize 到 bat_safe 直径 + paste 居中 ----
target_size = BAT_R * 2  # ~570
newbat_resized = newbat_pil.resize((target_size, target_size), Image.LANCZOS)
paste_x = bcx - BAT_R
paste_y = bcy - BAT_R
canvas = clean_pil.copy()
canvas.paste(newbat_resized, (paste_x, paste_y), newbat_resized)
print(f"[3] pasted ({paste_x},{paste_y}) size={target_size}")

draw = ImageDraw.Draw(canvas)

# ---- [4] 顶弧 MOONCREST ----
arc_text = "MOONCREST"
arc_radius = BAT_R + 70  # 必须落在蝙蝠椭圆(r=BAT_R)之外，贴外圈像原图 LA CASA
arc_size = int(BAT_R * 0.16)
f_arc = ImageFont.truetype(F_TIMES_BD, arc_size)
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
    d_ch.text((arc_size, arc_size), ch, font=f_arc, fill=DEEP_PURPLE + (255,))
    rot_angle = -math.degrees(rad_pil) - 90
    ch_rot = ch_img.rotate(rot_angle, resample=Image.BICUBIC, center=(arc_size, arc_size))
    canvas.paste(ch_rot, (int(x - arc_size), int(y - arc_size)), ch_rot)
print(f"[4] MOONCREST 顶弧 半径={arc_radius}")

# ---- [5] 侧字 Est. (左侧 y=830) ----
est_text = "EST."
est_size = int(BAT_R * 0.20)
f_est = ImageFont.truetype(F_GEORGIA_BD, est_size)
est_y = bcy + 30
est_x = bx - 100
for dx in range(-2, 3):
    for dy in range(-2, 3):
        if dx * dx + dy * dy <= 4:
            draw.text((est_x + dx, est_y + dy), est_text, font=f_est, fill=BAT_BLACK)
draw.text((est_x, est_y), est_text, font=f_est, fill=DEEP_PURPLE)
print(f"[5] EST. 左侧 ({est_x},{est_y})")

# ---- [6] 侧字 MMXXVI (右侧 y=830) ----
year_text = "MMXXVI"
year_size = est_size
f_year = ImageFont.truetype(F_GEORGIA_BD, year_size)
year_y = bcy + 30
year_x = bx + bw + 50
for dx in range(-2, 3):
    for dy in range(-2, 3):
        if dx * dx + dy * dy <= 4:
            draw.text((year_x + dx, year_y + dy), year_text, font=f_year, fill=BAT_BLACK)
draw.text((year_x, year_y), year_text, font=f_year, fill=DEEP_PURPLE)
print(f"[6] MMXXVI 右侧 ({year_x},{year_y})")

# ---- [7] 大字 MOONCREST (替代 BACARDÍ y=980-1158) ----
big_text = "MOONCREST"
big_size = 200
f_big = ImageFont.truetype(F_TIMES_BD, big_size)
bb = draw.textbbox((0, 0), big_text, font=f_big)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
big_x = (W - tw) // 2
big_y = 1050 - bb[1]
for dx in range(-4, 5):
    for dy in range(-4, 5):
        if dx * dx + dy * dy <= 16:
            draw.text((big_x + dx, big_y + dy), big_text, font=f_big, fill=BAT_BLACK)
draw.text((big_x, big_y), big_text, font=f_big, fill=DEEP_PURPLE)
print(f"[7] MOONCREST 大字 字号={big_size}")

# ---- [8] 小字 CURSE (替代 MYHEART y=1185-1335) ----
small_text = "CURSE"
small_size = 160
f_small = ImageFont.truetype(F_TIMES_BD, small_size)
bb = draw.textbbox((0, 0), small_text, font=f_small)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
small_x = (W - tw) // 2
small_y = 1240 - bb[1]
for dx in range(-3, 4):
    for dy in range(-3, 4):
        if dx * dx + dy * dy <= 9:
            draw.text((small_x + dx, small_y + dy), small_text, font=f_small, fill=BAT_BLACK)
draw.text((small_x, small_y), small_text, font=f_small, fill=DEEP_PURPLE)
print(f"[8] CURSE 小字 字号={small_size}")

# ---- [9] 倒三角 ----
tri_cx = bcx
tri_top_y = 1370
tri_w = 30; tri_h = 30
draw.polygon([
    (tri_cx, tri_top_y + tri_h),
    (tri_cx - tri_w, tri_top_y),
    (tri_cx + tri_w, tri_top_y),
], fill=DEEP_PURPLE)
print(f"[9] ▼ 倒三角 ({tri_cx},{tri_top_y})")

# ---- [10] 保存 ----（不做 USM：避免蝙蝠硬边过冲产生白晕）
canvas = canvas.convert("RGB")
canvas.save(OUT_FINAL, quality=95)
print(f"[save] {OUT_FINAL}")

# ---- [11] 对照图 ----
SRC_ORIG = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
orig = Image.open(SRC_ORIG).convert("RGB")
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 40), (235, 235, 238))
f_lbl = ImageFont.truetype(F_TIMES_BD, 28)
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 80, 4), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 200, 4), "v206  STRUCTURE 1:1 + ELEMENT SWAP", font=f_lbl, fill=(20, 20, 20))
comp.paste(orig, (0, 40))
comp.paste(canvas, (W + gap, 40))
comp.save(OUT_COMPARE, quality=92)
print(f"[compare] {OUT_COMPARE}")
print("\n[done] v206 完成")
