"""
v205_c v3 — 终极版（横幅大字 + 放大蝙蝠 + 不切前景）

v205_c v2 自检 7/10：
  - 顶弧 MOONCREST 被圆形 mask 切了半截（"TS ERCNOOM"）
  - 蝙蝠徽章偏小（占 1/4 画面）

v205_c v3 修复：
  1. **SDXL 输出背景已紫色**，无需 mask 切白底，直接 paste
  2. **MOONCREST 改成顶部横幅**（白底反白+深紫字，BACARDÍ 横幅风）
  3. **蝙蝠徽章放大**：resize 目标 = BAT_R*2 = 760px
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
JOB = PROJECT / "jobs" / "smoke_v205"
CLEAN_IN = JOB / "_v205_a_clean.png"
NEWBAT_IN = JOB / "v205_b_newbat.png"
INFO_IN = JOB / "_v205_a_bat_info.json"
OUT_FINAL = JOB / "v205_bat_logo_final.jpg"
OUT_COMPARE = JOB / "_compare_v205.jpg"
print(f"[job] {JOB}")

info = json.loads(INFO_IN.read_text())
W = info["W"]; H = info["H"]
bx, by, bw, bh = info["bat_bbox"]
bcx, bcy = info["bat_center"]
BAT_R = info["bat_place_radius"]
BG_PURPLE = tuple(info["BG_PURPLE"])
RING_DEEP = tuple(info["RING_DEEP"])
print(f"[info] {W}x{H} bat=({bcx},{bcy}) R={BAT_R}")

# [1] 读 v205_a clean
clean_pil = Image.open(CLEAN_IN).convert("RGBA")

# [2] 读 SDXL 新蝙蝠（背景已紫，直接 paste）
newbat_pil = Image.open(NEWBAT_IN).convert("RGBA")

# [3] resize 放大：蝙蝠徽章 = 70% H（紧凑瓶身风）
target_size = int(H * 0.62)  # 1240px（接近 BACARDÍ 徽章占比）
newbat_resized = newbat_pil.resize((target_size, target_size), Image.LANCZOS)
print(f"[3] newbat {target_size}x{target_size}")

# [4] paste 居中上移（蝙蝠徽章贴顶，下方留给大字+副字+三角）
paste_x = bcx - target_size // 2
paste_y = int(H * 0.08)  # 顶端往下 8%（让出顶部小紫底边距）
canvas = clean_pil.copy()
canvas.paste(newbat_resized, (paste_x, paste_y), newbat_resized)
print(f"[4] pasted ({paste_x},{paste_y})  size={target_size}")

draw = ImageDraw.Draw(canvas)

# ---- 配色 ----
WHITE = (255, 255, 255)
BAT_BLACK = (23, 4, 23)
DEEP_PURPLE = (45, 10, 60)
GOLD = (215, 165, 50)

F_TIMES_BD = r"C:/Windows/Fonts/timesbd.ttf"
F_GEORGIA_BD = r"C:/Windows/Fonts/georgiab.ttf"

# ---- 5a 顶部横幅 MOONCREST（白底反白+深紫字，模仿 BACARDÍ "HOUSE OF XXX" 横幅）----
# 横幅位置：贴图顶，紧凑
banner_w = int(W * 0.78)
banner_h = int(target_size * 0.14)
banner_x1 = (W - banner_w) // 2
banner_y1 = 80  # 顶部 80px 紫底留白
banner_x2 = banner_x1 + banner_w
banner_y2 = banner_y1 + banner_h
# 横幅底色 = 深紫色环带
draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2], fill=RING_DEEP)
draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2],
               outline=BAT_BLACK, width=4)
# 横幅字 MOONCREST（罗马衬线粗体白色）
banner_text = "MOONCREST"
moon_size = int(banner_h * 0.60)
f_moon = ImageFont.truetype(F_TIMES_BD, moon_size)
bb = draw.textbbox((0, 0), banner_text, font=f_moon)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
moon_x = banner_x1 + (banner_w - tw) // 2
moon_y = banner_y1 + (banner_h - th) // 2 - bb[1]
# 黑描边
for dx in range(-3, 4):
    for dy in range(-3, 4):
        if dx * dx + dy * dy <= 9:
            draw.text((moon_x + dx, moon_y + dy), banner_text,
                      font=f_moon, fill=BAT_BLACK)
draw.text((moon_x, moon_y), banner_text, font=f_moon, fill=WHITE)
print(f"[5a] MOONCREST banner 横幅 {banner_w}x{banner_h} 字={moon_size}")

# ---- 5b 中部大字 CURSE（罗马衬线粗体，深紫+黑描边，像 BACARDÍ 大字 65% 宽）----
center_text = "CURSE"
cur_size = int(BAT_R * 0.85)  # 字号放大（R=285 → 字号 240）
f_cur = ImageFont.truetype(F_TIMES_BD, cur_size)
bb = draw.textbbox((0, 0), center_text, font=f_cur)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
# CURSE 位置：蝙蝠徽章下沿 + 60
cur_y = paste_y + target_size + 60
cur_x = (W - tw) // 2
# 黑描边
for dx in range(-6, 7):
    for dy in range(-6, 7):
        if dx * dx + dy * dy <= 30:
            draw.text((cur_x + dx, cur_y + dy - bb[1]), center_text,
                      font=f_cur, fill=BAT_BLACK)
draw.text((cur_x, cur_y - bb[1]), center_text, font=f_cur, fill=DEEP_PURPLE)
print(f"[5b] CURSE 大字 字号={cur_size} y={cur_y}")

# ---- 5c 底部 EST. MMXXVI（金色）----
bot_text = "EST. MMXXVI"
bot_size = int(BAT_R * 0.22)
f_bot = ImageFont.truetype(F_GEORGIA_BD, bot_size)
bb = draw.textbbox((0, 0), bot_text, font=f_bot)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
bot_y = cur_y + cur_size + 50
bot_x = (W - tw) // 2
for dx in range(-3, 4):
    for dy in range(-3, 4):
        if dx * dx + dy * dy <= 9:
            draw.text((bot_x + dx, bot_y + dy - bb[1]), bot_text,
                      font=f_bot, fill=BAT_BLACK)
draw.text((bot_x, bot_y - bb[1]), bot_text, font=f_bot, fill=GOLD)
print(f"[5c] EST.MMXXVI 字号={bot_size} y={bot_y}")

# ---- 5d 倒三角 ▼ ----
tri_cx = bcx
tri_top_y = bot_y + bot_size + 35
tri_w = 38; tri_h = 38
draw.polygon([
    (tri_cx, tri_top_y + tri_h),
    (tri_cx - tri_w, tri_top_y),
    (tri_cx + tri_w, tri_top_y),
], fill=DEEP_PURPLE)
print(f"[5d] ▼ ({tri_cx},{tri_top_y})")

# ---- [6] USM 锐化 + 保存 ----
canvas = canvas.convert("RGB")
canvas = canvas.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
canvas.save(OUT_FINAL, quality=95)
print(f"[save] {OUT_FINAL}")

# ---- [7] 对照图 ----
SRC_ORIG = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
orig = Image.open(SRC_ORIG).convert("RGB")
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 40), (235, 235, 238))
f_lbl = ImageFont.truetype(F_TIMES_BD, 28)
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 80, 4), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 200, 4), "v205  SDXL NEW BAT + BANNER TYPO", font=f_lbl, fill=(20, 20, 20))
comp.paste(orig, (0, 40))
comp.paste(canvas, (W + gap, 40))
comp.save(OUT_COMPARE, quality=92)
print(f"[compare] {OUT_COMPARE}")

# 重新输出 bat_info 给 v205_v4 调整
info_out = dict(info)
info_out["bat_paste_offset"] = [paste_x, paste_y]
info_out["bat_paste_size"] = target_size
with open(INFO_IN, "w") as f:
    json.dump(info_out, f, indent=2)
print(f"[info-updated] bat_paste_size={target_size}")

print("\n[done] v205_c v3 完成")
