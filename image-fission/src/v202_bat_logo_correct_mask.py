"""
v202 bat_logo 纯 PIL/CV 裂变 v3 — 正确 mask 几何（无 SDXL、~5 秒）
================================================================
v200/v201 失败的根因：连通域分析时把 BACARDÍ 的 "B" 字母误判为 "Est."
  - comp170 bbox=(291,1002,133x146) center=(356,1074) 实际是 BACARDÍ 的 "B"
  - 真实 Est. 在 y=800-900（bat 中心水平线两侧），不在 y=1002-1148

v202 正确 mask 几何（从 bat_mid_crop + y-line scan 验证）:
  - LA CASA 弧字：y=300-720, x=200-1350（完整覆盖弧形）
  - bat 中部两侧 Est./1862：y=780-940, x=200-500 + x=1100-1350
  - bat 下方 BACARDÍ/MYHEART/底三角：y=980-1465, x=200-1350
  - 全程减去 bat_safe 椭圆（绝不能擦到蝙蝠主体）

步骤：
  [1] 读 BACARDÍ 紫底源
  [2] 连通域分析 → 锁定蝙蝠+环的 bbox
  [3] 三段 mask（顶弧 / 侧字 / 底字）减去 bat_safe
  [4] 在原图上实色填充擦字
  [5] 整图旋转 15° CW（borderValue=真实背景色）
  [6] PIL 烧新字
  [7] USM 锐化 → 出图 + 对照
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

# ============== 路径 ==============
PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
JOB = PROJECT / "jobs" / "smoke_v202"
JOB.mkdir(parents=True, exist_ok=True)
FONTS = PROJECT / "fonts"
OUT = JOB / "v202_bat_logo_rotated.jpg"
COMPARE = JOB / "_compare_v202.jpg"

# ============== 配色（实测采样）==============
BG_PURPLE = (183, 127, 171)
RING_DARK = (90, 20, 120)
BAT_BLACK = (23, 4, 23)
WHITE = (255, 255, 255)
F_PIRATA = str(FONTS / "PirataOne-Regular.ttf")
F_RYE = str(FONTS / "Rye-Regular.ttf")

# ============== [1] 读源 ==============
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

# ============== [2] 连通域分析 → bat bbox ==============
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cands = []
for c in contours:
    a = cv2.contourArea(c)
    if a < 30000:
        continue
    x, y, w, h = cv2.boundingRect(c)
    cands.append((a, x, y, w, h))
cands.sort(reverse=True)
bat_a, bx, by, bw, bh = cands[0]
bcx, bcy = bx + bw // 2, by + bh // 2
R_bat = max(bw, bh) // 2
print(f"[bat] bbox=({bx},{by},{bw},{bh})  center=({bcx},{bcy})  R={R_bat}")

# ============== [3] 三段 mask ==============
# bat_safe 椭圆（绝不能擦到蝙蝠主体）— 比实际 bat 大 60px 留足安全量
bat_safe = np.zeros((H, W), np.uint8)
safe_w = int(bw / 2 + 60)
safe_h = int(bh / 2 + 60)
cv2.ellipse(bat_safe, (bcx, bcy), (safe_w, safe_h), 0, 0, 360, 255, -1)

mask_all = np.zeros((H, W), np.uint8)
# 3a. 顶弧 LA CASA — y=300-720 全宽
cv2.rectangle(mask_all, (200, 300), (1350, 730), 255, -1)
# 3b. bat 中部两侧 Est./1862 — y=770-960
cv2.rectangle(mask_all, (200, 770), (530, 960), 255, -1)    # 左侧（含 BACARDÍ 的 B 字母左半）
cv2.rectangle(mask_all, (1000, 770), (1350, 960), 255, -1)  # 右侧（1862 真位 x=1024 起）
# 3c. 下方 BACARDÍ/MYHEART/底三角 — y=980-1465
cv2.rectangle(mask_all, (200, 980), (1350, 1465), 255, -1)
# 减去 bat_safe
mask_all = cv2.bitwise_and(mask_all, cv2.bitwise_not(bat_safe))

# 膨胀 14px（笔画外溢），不破坏 bat_safe
k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask_all_d = cv2.dilate(mask_all, k_d, iterations=1)
bat_safe_d = cv2.dilate(bat_safe, k_d, iterations=1)
mask_all_d = cv2.bitwise_and(mask_all_d, cv2.bitwise_not(bat_safe_d))
print(f"[masks] all_d={int((mask_all_d>0).sum())}px")

# ============== [4] 在原图上实色填充擦字 ==============
clean = img_rgb.copy()
clean[mask_all_d > 0] = BG_PURPLE
print("[erase] old text filled with REAL bg color")

# ============== [5] 旋转 clean 图 15° CW ==============
ANGLE = -15.0
M = cv2.getRotationMatrix2D((W // 2, H // 2), ANGLE, 1.0)
result_rgb = cv2.warpAffine(clean, M, (W, H),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=BG_PURPLE)
print(f"[rotate] {-ANGLE}° CW around ({W//2},{H//2})")

# ============== [6] PIL 烧新字 ==============
result_pil = Image.fromarray(result_rgb).convert("RGBA")
draw = ImageDraw.Draw(result_pil)

# ---- MOONCREST 顶部横幅 ----
banner_w = int(W * 0.46)
banner_h = int(R_bat * 0.50)
banner_x1 = (W - banner_w) // 2
banner_y1 = max(40, by - 50 - banner_h)
banner_x2 = banner_x1 + banner_w
banner_y2 = banner_y1 + banner_h
draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2],
               fill=RING_DARK + (255,))
draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2],
               outline=BAT_BLACK + (255,), width=4)
moon_size = int(banner_h * 0.55)
f_moon = ImageFont.truetype(F_PIRATA, moon_size)
moon_txt = "MOONCREST"
bb = draw.textbbox((0, 0), moon_txt, font=f_moon)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
draw.text((banner_x1 + (banner_w - tw) // 2,
           banner_y1 + (banner_h - th) // 2 - bb[1]),
          moon_txt, font=f_moon, fill=WHITE + (255,))

# ---- CURSE 中央大字 ----
curse_y = by + bh + 120
curse_size = int(R_bat * 0.55)
f_curse = ImageFont.truetype(F_PIRATA, curse_size)
curse_txt = "CURSE"
bb = draw.textbbox((0, 0), curse_txt, font=f_curse)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
curse_tx = (W - tw) // 2
curse_ty = curse_y - bb[1]
for dx, dy in [(-3, -1), (3, -1), (-3, 1), (3, 1),
               (-1, -3), (1, -3), (-1, 3), (1, 3),
               (-2, -2), (2, -2), (-2, 2), (2, 2)]:
    draw.text((curse_tx + dx, curse_ty + dy), curse_txt,
              font=f_curse, fill=BAT_BLACK + (255,))
draw.text((curse_tx, curse_ty), curse_txt,
          font=f_curse, fill=RING_DARK + (255,))

# ---- EST. MMXXVI 底部小字 ----
est_y = curse_y + int(R_bat * 0.95)
est_size = int(R_bat * 0.22)
f_est = ImageFont.truetype(F_RYE, est_size)
est_txt = "EST. MMXXVI"
bb = draw.textbbox((0, 0), est_txt, font=f_est)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
est_tx = (W - tw) // 2
est_ty = est_y - bb[1]
for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    draw.text((est_tx + dx, est_ty + dy), est_txt,
              font=f_est, fill=BAT_BLACK + (255,))
draw.text((est_tx, est_ty), est_txt, font=f_est, fill=RING_DARK + (255,))
print("[text] MOONCREST banner + CURSE center + EST.MMXXVI bottom")

# ============== [7] USM 锐化 + 保存 + 对照 ==============
result_pil = result_pil.convert("RGB")
result_pil = result_pil.filter(ImageFilter.UnsharpMask(
    radius=1.5, percent=120, threshold=2))
result_pil.save(OUT, quality=95)
print(f"[save] {OUT}")

# 对照图
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 40), (235, 235, 238))
f_lbl = ImageFont.truetype(F_PIRATA, 28)
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 60, 4), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 130, 4), "v202  ROT15-", font=f_lbl, fill=(20, 20, 20))
comp.paste(Image.fromarray(img_rgb), (0, 40))
comp.paste(result_pil, (W + gap, 40))
comp.save(COMPARE, quality=92)
print(f"[compare] {COMPARE}")
print(f"[done] job dir: {JOB}")
