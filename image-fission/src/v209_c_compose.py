"""
v209_c — v208 收口修正：严格参考原图风格（用户硬指令）

用户原话：「原图什么样的风格就跟他参考会不会」

修正三处:
  1. 主体: 用 v206_a_clean.png 的原图矢量蝙蝠 (水平镜像 = "改角度" 但保持矢量剪影风格)
     不再用 v206_b_newbat.png 的 SDXL 摄影级写实蝙蝠
  2. 弧文字: 修正 deg_pil 参数 (200 -> 340 跨过顶部, 140° span), 字符旋转用切线方向
     文字从 9 字 "MOONCREST" 改为 15 字 "MOONCREST MANOR" (贴近原图 22 字密度)
  3. 排版: 维持 v208 的 MOONCREST 大字 / CURSE 小字 / EST.MMXXVI 边字 / 倒三角

字体: 沿用 Lora-VF Bold 700 (humanist-serif 替代 Didone, 已接受 0.45 阈值预设)
"""
import sys
import os
import math
import json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

PROJECT = Path("E:/Desktop/双接口/image-fission")
JOB = PROJECT / "jobs" / "smoke_v206"
CLEAN_IN = JOB / "_v206_a_clean.png"
INFO_IN = JOB / "_v206_a_bat_info.json"
OUT_FINAL = JOB / "v209_bat_logo_origref_final.jpg"
OUT_COMPARE = JOB / "_compare_v209.jpg"

F_LORA = r"E:/Desktop/双接口/image-fission/fonts/Lora-VF.ttf"
SRC_ORIG = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"

# 原图实测 (来自 measure_orig_text.py 2026-09-02 11:59 实跑)
ORIG_TARGETS = {
    "MOONCREST_big":   {"text": "MOONCREST",        "target_w": 613, "y_center": 1074, "weight": 700},
    "CURSE_small":     {"text": "CURSE",            "target_w": 387, "y_center": 1261, "weight": 700},
    "MANOR_arc":       {"text": "MOONCREST MANOR",  "target_w": 514, "y_center":   0, "weight": 700},
    "EST_left":        {"text": "EST.",             "target_w":  52, "y_center":  862, "weight": 700},
    "MMXXVI_right":    {"text": "MMXXVI",           "target_w":  52, "y_center":  862, "weight": 700},
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


def calibrate_size(text: str, target_w: int, weight: int = 700, hint: int = None) -> int:
    """二分查找字号使 text 渲染宽度匹配 target_w 像素。
    hint 自适应: target_w 小时 hint 也小, lo 下限可到 hint/8 (允许极小字号)。"""
    if hint is None:
        hint = max(80, target_w // 3)
    lo, hi = max(8, hint // 8), hint * 3
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
    print(f"  calibrate '{text}': size={size}  ->  bbox_w={bw} (target {target_w}, hint={hint})")
    return size


# ---- [pre-calibrate] ----
print("[pre-calibrate 按原图实测宽度反算字号]")
sizes = {}
for key, t in ORIG_TARGETS.items():
    sizes[key] = calibrate_size(t["text"], t["target_w"], t["weight"])


def text_xy_to_baseline_xy(s: str, font, x_left: int, y_baseline: int) -> tuple:
    """PIL textbbox 是 ink-box; baseline 落在 y_baseline。"""
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


# ---- [1] 读 v206 干净底 (原图矢量蝙蝠 + 紫底, 字已擦) ----
clean_pil = Image.open(CLEAN_IN).convert("RGBA")
canvas = clean_pil.copy()

# ---- [2] 水平镜像蝙蝠 ("改角度" 但保持矢量剪影风格) ----
# 镜像整个干净底: 蝙蝠被镜像, 紫底不变 (均匀), 文字擦除痕迹也镜像 (小, 不显眼)
# bcx 777 -> 1552-777 = 775 (差 2 像素, 视觉无感)
canvas = ImageOps.mirror(canvas)
print(f"[2] 水平镜像整图 (蝙蝠: x={bcx}->{W-1-bcx}, 视觉差 1px)")

# ---- [3] 烧字 ----
draw = ImageDraw.Draw(canvas)

# [3a] 大字 MOONCREST
f_big = lora(sizes["MOONCREST_big"], 700)
big_text = ORIG_TARGETS["MOONCREST_big"]["text"]
big_y_baseline = ORIG_TARGETS["MOONCREST_big"]["y_center"]
big_x = center_x_for(big_text, f_big, W)
big_xy = text_xy_to_baseline_xy(big_text, f_big, big_x, big_y_baseline)
draw_text_triple(draw, big_xy, big_text, f_big, r_inner=6, r_outer=1)
print(f"[3a] MOONCREST (size={sizes['MOONCREST_big']}, y_baseline={big_y_baseline})")

# [3b] 小字 CURSE
f_small = lora(sizes["CURSE_small"], 700)
small_text = ORIG_TARGETS["CURSE_small"]["text"]
small_y_baseline = ORIG_TARGETS["CURSE_small"]["y_center"]
small_x = center_x_for(small_text, f_small, W)
small_xy = text_xy_to_baseline_xy(small_text, f_small, small_x, small_y_baseline)
draw_text_triple(draw, small_xy, small_text, f_small, r_inner=5, r_outer=1)
print(f"[3b] CURSE (size={sizes['CURSE_small']}, y_baseline={small_y_baseline})")

# [3c] 顶弧 MOONCREST MANOR  (修正参数: 200° -> 340° 跨过顶部)
arc_size = sizes["MANOR_arc"]
f_arc = lora(arc_size, 700)
arc_text = ORIG_TARGETS["MANOR_arc"]["text"]
arc_radius = BAT_R + 60
n_chars = len(arc_text)
ARC_START_DEG = 200   # math convention, upper-left
ARC_SPAN_DEG = 140    # span across top to upper-right (passes through 270° = straight up)
print(f"[3c] 顶弧 '{arc_text}' (size={arc_size}, radius={arc_radius}, "
      f"span {ARC_START_DEG}° -> {ARC_START_DEG + ARC_SPAN_DEG}° 跨顶部)")

for i, ch in enumerate(arc_text):
    if ch == " ":
        # 空格 = 弧文字间隙 (原图 "LA CASA DEL MURCIELAGO" 也是真空格)
        continue
    t = (i + 0.5) / n_chars
    deg_pil = ARC_START_DEG + t * ARC_SPAN_DEG
    rad_pil = math.radians(deg_pil)
    x = bcx + arc_radius * math.cos(rad_pil)
    y = bcy + arc_radius * math.sin(rad_pil)
    ch_img = Image.new("RGBA", (arc_size * 3, arc_size * 3), (0, 0, 0, 0))
    d_ch = ImageDraw.Draw(ch_img)
    # 单字小 — 只用黑色描边 (白描边在小字会糊)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx * dx + dy * dy <= 4:
                d_ch.text((arc_size + dx, arc_size + dy), ch, font=f_arc, fill=BAT_BLACK)
    d_ch.text((arc_size, arc_size), ch, font=f_arc, fill=DEEP_PURPLE)
    # 切线方向 = deg + 90 (math), PIL rotate CW 正向, 所以 rot = -(deg + 90)
    rot_angle = -math.degrees(rad_pil) - 90
    ch_rot = ch_img.rotate(rot_angle, resample=Image.BICUBIC, center=(arc_size, arc_size))
    canvas.paste(ch_rot, (int(x - arc_size), int(y - arc_size)), ch_rot)

# [3d] Est. 左边字
f_est = lora(sizes["EST_left"], 700)
est_text = ORIG_TARGETS["EST_left"]["text"]
est_y_center = ORIG_TARGETS["EST_left"]["y_center"]
est_x = bcx - 355
est_xy = text_xy_to_baseline_xy(est_text, f_est, est_x, est_y_center)
draw_text_triple(draw, est_xy, est_text, f_est, r_inner=3, r_outer=1)
print(f"[3d] EST. (size={sizes['EST_left']}, x={est_x}, y={est_y_center})")

# [3e] MMXXVI 右边字
f_year = lora(sizes["MMXXVI_right"], 700)
year_text = ORIG_TARGETS["MMXXVI_right"]["text"]
year_x = bcx + 355
year_xy = text_xy_to_baseline_xy(year_text, f_year, year_x, est_y_center)
draw_text_triple(draw, year_xy, year_text, f_year, r_inner=3, r_outer=1)
print(f"[3e] MMXXVI (size={sizes['MMXXVI_right']}, x={year_x}, y={est_y_center})")

# [3f] 倒三角
tri_cx = bcx
tri_top_y = est_y_center + 90
draw.polygon([
    (tri_cx, tri_top_y + 30),
    (tri_cx - 30, tri_top_y),
    (tri_cx + 30, tri_top_y),
], fill=DEEP_PURPLE)
print(f"[3f] 倒三角 ({tri_cx},{tri_top_y})")

# ---- [4] USM 锐化 ----
canvas = canvas.filter(ImageFilter.UnsharpMask(radius=2.2, percent=120, threshold=2))
print(f"[4] USM(radius=2.2, percent=120, threshold=2)")

# ---- [4.5] 加 grain 恢复原图颗粒感 (v206_a 擦字丢了纹理 → 镜像后偏平) ----
# 原图肉眼可见紫色背景有颗粒/刮痕; 不补 grain 会让 v209 比原图"光滑", 也违"参考原图风格"
arr = np.array(canvas, dtype=np.int16)
np.random.seed(42)  # reproducible
noise = np.random.normal(0, 7, arr.shape).astype(np.int16)
arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
canvas = Image.fromarray(arr)
print(f"[4.5] 加 Gaussian grain (sigma=7, seed=42) 恢复原图纹理")

# ---- [5] 保存 ----
canvas = canvas.convert("RGB")
canvas.save(OUT_FINAL, quality=95)
print(f"[save] {OUT_FINAL}")

# ---- [6] 对照 ----
orig = Image.open(SRC_ORIG).convert("RGB")
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 40), (235, 235, 238))
f_lbl = ImageFont.truetype(F_LORA, 28)
f_lbl.set_variation_by_axes([600])
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 80, 4), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 200, 4), "v209  ORIG-REF (flipped bat + fixed arc)", font=f_lbl, fill=(20, 20, 20))
comp.paste(orig, (0, 40))
comp.paste(canvas, (W + gap, 40))
comp.save(OUT_COMPARE, quality=92)
print(f"[compare] {OUT_COMPARE}")
print("\n[done] v209 完成 — 原图矢量蝙蝠(镜像) + 修弧 200°->340° + Didone 风排版")