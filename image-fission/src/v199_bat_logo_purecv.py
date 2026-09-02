"""
v199 bat_logo 纯 PIL/CV 裂变（无 SDXL、无模型下载、~5 秒）
================================================================
用户 2026-09-02 两次纠正后的正确方案：
  1) 迷彩 ref + 提示词脑补蝙蝠 = 混图瞎做（v197 错）
  2) 原图是紫色，背景就该是紫色（配色铁律#7）
正确做法：把 BACARDÍ 紫底原图当唯一源，主体（蝙蝠+环）原样保留，旋转 15°
（铁律#1 允许改角度），原侵权字先在原图上实色填充擦掉（mask 和文字像素 0
错位），再整图旋转，最后 PIL 烧新字。配色严格锁原图 4 色（浅紫/深紫/黑/白）。

步骤：
  [1] 读 BACARDÍ 紫底源
  [2] 测出徽章圆环中心 (cx,cy) + 外半径 R（minEnclosingCircle，测量非猜）
  [3] 前景蒙版 + 原字覆盖区（顶弧环带 / bat下方 / 底部 / 左右侧）
  [4] 【关键】在原图上实色填充擦字（顶弧→环深紫，其余→浅紫底）→ 得到 clean 图
  [5] 旋转 clean 图 15° CCW（borderValue=浅紫，角落自动填底色）
  [6] PIL 烧新字 MOONCREST(顶横幅) / CURSE(中央) / EST.MMXXVI(底)
  [7] USM 锐化 → 出图 + 原图|结果 对照
"""
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import sys

# ============== 路径 ==============
PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
JOB = PROJECT / "jobs" / "smoke_v199"
JOB.mkdir(parents=True, exist_ok=True)
FONTS = PROJECT / "fonts"
OUT = JOB / "v199_bat_logo_rotated.jpg"
COMPARE = JOB / "_compare_v199.jpg"

# ============== 配色（严格锁原图 4 色）==============
PURPLE_BG_RGB = (185, 150, 170)   # 浅紫底（取自原图角落实测）
RING_FILL = (75, 45, 90)          # 环带深紫（inpaint 后的兜底色）
DARK_PURPLE = (38, 20, 50)        # 蝙蝠/环的原色（深紫）
BLACK = (15, 10, 20)
WHITE = (245, 235, 240)
F_PIRATA = str(FONTS / "PirataOne-Regular.ttf")
F_RYE = str(FONTS / "Rye-Regular.ttf")

# ============== [1] 读源（cv2 非 ASCII 路径 workaround）==============
def _read_img_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit(f"can't decode: {path}")
    return im
img_bgr = _read_img_unicode(SRC)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"[load] {SRC.name}  {W}x{H}")

# ============== [2] 测徽章圆环中心 + 外半径 ==============
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, dark = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
best = None
for c in contours:
    if len(c) < 50:
        continue
    (ccx, ccy), rr = cv2.minEnclosingCircle(c)
    if rr < 120:
        continue
    if best is None or rr > best[2]:
        best = (int(ccx), int(ccy), int(rr))
if best is None:
    best = (W // 2, int(H * 0.52), int(W * 0.46))
cx, cy, R = best
R_in = int(R * 0.74)
print(f"[ring] center=({cx},{cy})  R_outer={R}  R_in={R_in}")

# ============== [3] 原字覆盖区（在原图坐标系，旋转前用，0 错位）==============
# 3a. 顶弧：环带上半 (R_in ~ R)，y < cy + 0.05R 盖住 "LA CASA..." 尾字母
annulus = np.zeros((H, W), np.uint8)
cv2.circle(annulus, (cx, cy), int(R * 0.99), 255, -1)
cv2.circle(annulus, (cx, cy), R_in, 0, -1)
upper_half = np.zeros((H, W), np.uint8)
upper_half[: int(cy + R * 0.05), :] = 255
mask_arc = cv2.bitwise_and(annulus, upper_half)  # → 填 RING_FILL

# 3b. 内部字区（bat 下方 + 底部 + 左右侧）→ 填 PURPLE_BG
mask_inner = np.zeros((H, W), np.uint8)
# bat 正下方（BACARDÍ）
cv2.rectangle(mask_inner,
              (int(cx - R * 0.66), int(cy + R * 0.26)),
              (int(cx + R * 0.66), int(cy + R * 0.60)), 255, -1)
# 底部字（Y/EHEART 等）
cv2.rectangle(mask_inner,
              (int(cx - R * 0.52), int(cy + R * 0.60)),
              (int(cx + R * 0.52), int(cy + R * 0.92)), 255, -1)
# 左右侧（Est. / 1862）
cv2.rectangle(mask_inner,
              (int(cx - R * 0.70), int(cy - R * 0.28)),
              (int(cx - R * 0.22), int(cy + R * 0.28)), 255, -1)
cv2.rectangle(mask_inner,
              (int(cx + R * 0.22), int(cy - R * 0.28)),
              (int(cx + R * 0.70), int(cy + R * 0.28)), 255, -1)

# 3c. bat_safe 椭圆（绝不能擦到蝙蝠主体）
bat_safe = np.zeros((H, W), np.uint8)
cv2.ellipse(bat_safe, (cx, int(cy - R * 0.05)),
            (int(R * 0.30), int(R * 0.27)), 0, 0, 360, 255, -1)
mask_inner = cv2.bitwise_and(mask_inner, cv2.bitwise_not(bat_safe))

# 3d. 大核膨胀 15px 确保笔画全吞（含笔画外溢 ~5px + 旋转前的安全量）
k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask_arc_d = cv2.dilate(mask_arc, k_d, iterations=1)
mask_inner_d = cv2.dilate(mask_inner, k_d, iterations=1)
# bat_safe 也轻微膨胀作为"双保险"（覆盖 mask 边缘万一多吃了的 bat 区域）
bat_safe_d = cv2.dilate(bat_safe, k_d, iterations=1)
mask_inner_d = cv2.bitwise_and(mask_inner_d, cv2.bitwise_not(bat_safe_d))
print(f"[masks] arc={int((mask_arc_d>0).sum())}px  inner={int((mask_inner_d>0).sum())}px")

# ============== [4] 在原图上实色填充擦字（mask 和文字像素 0 错位）==============
clean = img_rgb.copy()
clean[mask_arc_d > 0] = RING_FILL
clean[mask_inner_d > 0] = PURPLE_BG_RGB
print("[erase] old text filled in ORIGINAL (zero misalignment)")

# ============== [5] 旋转 clean 图 15° CCW（角落自动填浅紫底）==============
ANGLE = 15.0
M = cv2.getRotationMatrix2D((cx, cy), ANGLE, 1.0)
result_rgb = cv2.warpAffine(clean, M, (W, H),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=PURPLE_BG_RGB)
print(f"[rotate] {ANGLE}° CCW around ({cx},{cy})")

# ============== [6] PIL 烧新字（全部直字，避开 arc_text 抗锯齿坑）==============
result_pil = Image.fromarray(result_rgb).convert("RGBA")
draw = ImageDraw.Draw(result_pil)

# ---- MOONCREST 顶部横幅（深紫底+白字）----
banner_w = int(W * 0.50)
banner_h = int(R * 0.24)
banner_x1 = (W - banner_w) // 2
banner_y1 = int(cy - R * 1.08)
banner_x2 = banner_x1 + banner_w
banner_y2 = banner_y1 + banner_h
draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2], fill=DARK_PURPLE + (255,))
draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2], outline=(0, 0, 0, 255), width=3)
moon_size = int(banner_h * 0.58)
f_moon = ImageFont.truetype(F_PIRATA, moon_size)
moon_txt = "MOONCREST"
bb = draw.textbbox((0, 0), moon_txt, font=f_moon)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
draw.text((banner_x1 + (banner_w - tw) // 2,
           banner_y1 + (banner_h - th) // 2 - bb[1]),
          moon_txt, font=f_moon, fill=(255, 255, 255, 255))

# ---- CURSE 中央大字（bat 正下方，PirataOne）----
curse_y = int(cy + R * 0.40)
curse_size = int(R * 0.22)
f_curse = ImageFont.truetype(F_PIRATA, curse_size)
curse_txt = "CURSE"
bb = draw.textbbox((0, 0), curse_txt, font=f_curse)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
curse_tx = (W - tw) // 2
curse_ty = curse_y - bb[1]
# 黑色描边
for dx, dy in [(-2, -1), (2, -1), (-2, 1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2)]:
    draw.text((curse_tx + dx, curse_ty + dy), curse_txt,
              font=f_curse, fill=(0, 0, 0, 255))
draw.text((curse_tx, curse_ty), curse_txt,
          font=f_curse, fill=DARK_PURPLE + (255,))

# ---- EST. MMXXVI 底部小字 ----
est_y = int(cy + R * 0.66)
est_size = int(R * 0.075)
f_est = ImageFont.truetype(F_RYE, est_size)
est_txt = "EST. MMXXVI"
bb = draw.textbbox((0, 0), est_txt, font=f_est)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
est_tx = (W - tw) // 2
est_ty = est_y - bb[1]
for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    draw.text((est_tx + dx, est_ty + dy), est_txt,
              font=f_est, fill=(0, 0, 0, 255))
draw.text((est_tx, est_ty), est_txt, font=f_est, fill=DARK_PURPLE + (255,))
print("[text] MOONCREST banner + CURSE center + EST.MMXXVI bottom")

# ============== [7] USM 锐化 + 保存 + 对照 ==============
result_pil = result_pil.convert("RGB")
result_pil = result_pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
result_pil.save(OUT, quality=95)
print(f"[save] {OUT}")

# 对照图
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 40), (230, 230, 234))
f_lbl = ImageFont.truetype(F_PIRATA, 28)
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 60, 4), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 110, 4), "v199  ROT15+", font=f_lbl, fill=(20, 20, 20))
comp.paste(Image.fromarray(img_rgb), (0, 40))
comp.paste(result_pil, (W + gap, 40))
comp.save(COMPARE, quality=92)
print(f"[compare] {COMPARE}")
print(f"[done] job dir: {JOB}")
