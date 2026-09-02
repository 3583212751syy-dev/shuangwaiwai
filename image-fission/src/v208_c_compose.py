"""
v208_c — v207 升级：按原图实测 bbox 反算字号（"原图多大字体排版就怎么做"）

读 measure_orig_text.py 的实测结果，按原图各文字真实像素宽度反算 Lora-VF 字号，
让 v208 输出的文字 bbox 1:1 对齐原图 BACARDÍ 徽章排版比例。
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
OUT_FINAL = JOB / "v208_bat_logo_lora_realistic_final.jpg"
OUT_COMPARE = JOB / "_compare_v208.jpg"

F_LORA = r"E:/Desktop/双接口/image-fission/fonts/Lora-VF.ttf"
SRC_ORIG = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"

# ---- 原图实测（来自 measure_orig_text.py 2026-09-02 11:59 实跑）----
# 每个元素对应：v208 渲染文字 / 目标宽度 / 目标 bbox y center
# 这些是"原图多大字体排版就怎么做"的真值
ORIG_TARGETS = {
    "MOONCREST_big":   {"text": "MOONCREST",  "target_w": 613, "y_center": 1074, "weight": 700},  # BACARDÍ 39.5%
    "CURSE_small":     {"text": "CURSE",      "target_w": 387, "y_center": 1261, "weight": 700},  # MYHEART 24.9%
    "MOONCREST_arc":   {"text": "MOONCREST",  "target_w": 514, "y_center":   0,  "weight": 700},  # LA CASA 33.1%
    "EST_left":        {"text": "EST.",       "target_w": 52,  "y_center":  862, "weight": 700},  # Est. 3.4%
    "MMXXVI_right":    {"text": "MMXXVI",     "target_w": 52,  "y_center":  862, "weight": 700},  # 1982 同等
}

print(f"[job] {JOB}")

info = json.loads(INFO_IN.read_text())
W = info["W"]; H = info["H"]
bx, by, bw, bh = info["bat_bbox"]
bcx, bcy = info["bat_center"]
BAT_R = info["bat_R"]
BG_PURPLE = tuple(info["BG_PURPLE"])

WHITE = (255, 255, 255)
BAT_BLACK = (23, 4, 23)
DEEP_PURPLE = (45, 10, 60)


def lora(size: int, weight: int = 700) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(F_LORA, size)
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    return f


def calibrate_size(text: str, target_w: int, weight: int = 700, hint: int = 200) -> int:
    """二分查找字号使 text 渲染宽度匹配 target_w 像素。
    PIL truetype 的 textbbox 宽度对 size 近似线性，15 次迭代足够。
    """
    lo, hi = max(8, hint // 4), hint * 3
    for _ in range(18):
        mid = (lo + hi) // 2
        f = lora(mid, weight)
        bb = f.getbbox(text)
        if bb is None:
            return hint
        w = bb[2] - bb[0]
        if w < target_w:
            lo = mid
        else:
            hi = mid
    size = (lo + hi) // 2
    f = lora(size, weight)
    bw = f.getbbox(text)[2] - f.getbbox(text)[0]
    print(f"  calibrate '{text}': size={size}  ->  bbox_w={bw} (target {target_w})")
    return size


# ---- [pre-calibrate] ----
print("[pre-calibrate 按原图实测宽度反算字号]")
sizes = {}
for key, t in ORIG_TARGETS.items():
    sizes[key] = calibrate_size(t["text"], t["target_w"], t["weight"])


def text_xy_to_baseline_xy(s: str, font, x_left: int, y_baseline: int) -> tuple:
    """PIL textbbox 是 ink-box；我们希望 baseline 落在 y_baseline。
    算 baseline offset = -bb[1] (top offset 是从 baseline 往上的负偏移)
    """
    bb = font.getbbox(s)
    return (x_left - bb[0], y_baseline - bb[1])


def center_x_for(s: str, font, W_canvas: int) -> int:
    bb = font.getbbox(s)
    w = bb[2] - bb[0]
    return (W_canvas - w) // 2 - bb[0]


def draw_text_triple(draw, xy, text, font, fg=DEEP_PURPLE,
                     outline_inner=BAT_BLACK, outline_outer=WHITE,
                     r_inner=5, r_outer=1):
    x, y = xy
    for dx in range(-r_outer - r_inner, r_outer + r_inner + 1):
        for dy in range(-r_outer - r_inner, r_outer + r_inner + 1):
            d2 = dx * dx + dy * dy
            if d2 <= r_inner * r_inner:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_inner)
            elif d2 <= (r_inner + r_outer) * (r_inner + r_outer):
                draw.text((x + dx, y + dy), text, font=font, fill=outline_outer)
    draw.text((x, y), text, font=font, fill=fg)


# ---- [1] 读 v206 干净底 + 新蝙蝠 ----
clean_pil = Image.open(CLEAN_IN).convert("RGBA")
newbat_pil = Image.open(NEWBAT_IN).convert("RGB")
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

target_size = BAT_R * 2
newbat_resized = newbat_pil.resize((target_size, target_size), Image.LANCZOS)
canvas = clean_pil.copy()
canvas.paste(newbat_resized, (bcx - BAT_R, bcy - BAT_R), newbat_resized)

draw = ImageDraw.Draw(canvas)

# ---- [2] 大字 MOONCREST (calibrate 后字号) ----
f_big = lora(sizes["MOONCREST_big"], 700)
big_text = ORIG_TARGETS["MOONCREST_big"]["text"]
big_y_baseline = ORIG_TARGETS["MOONCREST_big"]["y_center"]
big_x = center_x_for(big_text, f_big, W)
big_xy = text_xy_to_baseline_xy(big_text, f_big, big_x, big_y_baseline)
draw_text_triple(draw, big_xy, big_text, f_big,
                 fg=DEEP_PURPLE, outline_inner=BAT_BLACK,
                 outline_outer=WHITE, r_inner=6, r_outer=1)
print(f"[2] 大字 MOONCREST (size={sizes['MOONCREST_big']}, y_baseline={big_y_baseline})")

# ---- [3] 小字 CURSE ----
f_small = lora(sizes["CURSE_small"], 700)
small_text = ORIG_TARGETS["CURSE_small"]["text"]
small_y_baseline = ORIG_TARGETS["CURSE_small"]["y_center"]
small_x = center_x_for(small_text, f_small, W)
small_xy = text_xy_to_baseline_xy(small_text, f_small, small_x, small_y_baseline)
draw_text_triple(draw, small_xy, small_text, f_small,
                 fg=DEEP_PURPLE, outline_inner=BAT_BLACK,
                 outline_outer=WHITE, r_inner=5, r_outer=1)
print(f"[3] 小字 CURSE (size={sizes['CURSE_small']}, y_baseline={small_y_baseline})")

# ---- [4] 顶弧 MOONCREST ----
arc_size = sizes["MOONCREST_arc"]
f_arc = lora(arc_size, 700)
arc_text = ORIG_TARGETS["MOONCREST_arc"]["text"]
arc_radius = BAT_R + 70
n_chars = len(arc_text)
for i, ch in enumerate(arc_text):
    t = (i + 0.5) / n_chars
    deg_pil = 270 - t * 110
    rad_pil = math.radians(deg_pil)
    x = bcx + arc_radius * math.cos(rad_pil)
    y = bcy + arc_radius * math.sin(rad_pil)
    ch_img = Image.new("RGBA", (arc_size * 3, arc_size * 3), (0, 0, 0, 0))
    d_ch = ImageDraw.Draw(ch_img)
    # 单字小——只用黑色描边（白描边在小字会糊）
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx * dx + dy * dy <= 4:
                d_ch.text((arc_size + dx, arc_size + dy), ch, font=f_arc, fill=BAT_BLACK)
    d_ch.text((arc_size, arc_size), ch, font=f_arc, fill=DEEP_PURPLE)
    rot_angle = -math.degrees(rad_pil) - 90
    ch_rot = ch_img.rotate(rot_angle, resample=Image.BICUBIC, center=(arc_size, arc_size))
    canvas.paste(ch_rot, (int(x - arc_size), int(y - arc_size)), ch_rot)
print(f"[4] 顶弧 (size={arc_size}, radius={arc_radius})")

# ---- [5] Est. 左边字 ----
f_est = lora(sizes["EST_left"], 700)
est_text = ORIG_TARGETS["EST_left"]["text"]
est_y_center = ORIG_TARGETS["EST_left"]["y_center"]
# EST 在 bat 中心水平两侧。原图 EST x=420-472 大致，水平相对 bcx 偏左 ~355 px
est_x = bcx - 355
est_xy = text_xy_to_baseline_xy(est_text, f_est, est_x, est_y_center)
draw_text_triple(draw, est_xy, est_text, f_est, r_inner=3, r_outer=1)
print(f"[5] EST. (size={sizes['EST_left']}, baseline=({est_x},{est_y_center}))")

# ---- [6] MMXXVI 右边字 ----
f_year = lora(sizes["MMXXVI_right"], 700)
year_text = ORIG_TARGETS["MMXXVI_right"]["text"]
year_x = bcx + 355
year_xy = text_xy_to_baseline_xy(year_text, f_year, year_x, est_y_center)
draw_text_triple(draw, year_xy, year_text, f_year, r_inner=3, r_outer=1)
print(f"[6] MMXXVI (size={sizes['MMXXVI_right']}, baseline=({year_x},{est_y_center}))")

# ---- [7] 倒三角 ----
tri_cx = bcx
tri_top_y = est_y_center + 90
draw.polygon([
    (tri_cx, tri_top_y + 30),
    (tri_cx - 30, tri_top_y),
    (tri_cx + 30, tri_top_y),
], fill=DEEP_PURPLE)
print(f"[7] 倒三角 ({tri_cx},{tri_top_y})")

# ---- [8] USM 锐化 (percent=120 略强于 v207) ----
canvas = canvas.filter(ImageFilter.UnsharpMask(radius=2.2, percent=120, threshold=2))
print(f"[8] USM(radius=2.2, percent=120, threshold=2)")

# ---- [9] 保存 ----
canvas = canvas.convert("RGB")
canvas.save(OUT_FINAL, quality=95)
print(f"[save] {OUT_FINAL}")

# ---- [10] 对照 ----
orig = Image.open(SRC_ORIG).convert("RGB")
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 40), (235, 235, 238))
f_lbl = ImageFont.truetype(F_LORA, 28)
f_lbl.set_variation_by_axes([600])
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 80, 4), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 240, 4), "v208  CALIBRATED sizes match orig bbox",
        font=f_lbl, fill=(20, 20, 20))
comp.paste(orig, (0, 40))
comp.paste(canvas, (W + gap, 40))
comp.save(OUT_COMPARE, quality=92)
print(f"[compare] {OUT_COMPARE}")
print("\n[done] v208 完成 — 大字/小字/边字 字号均由原图实测 bbox 反算")
