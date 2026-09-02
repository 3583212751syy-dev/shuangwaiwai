"""
v200 bat_logo 纯 PIL/CV 裂变（无 SDXL、无模型下载、~5 秒）
================================================================
v199 失败的根因 + v200 修复：
  ❌ v199 用 minEnclosingCircle 找"环中心"，得到 (774, 682) — 但这是
     外弧的对称中心，不是蝙蝠/环的几何中心
  ❌ v199 mask_inner 用 cy+0.26R~0.92R 覆盖 BACARDÍ — 但实际 BACARDÍ 在
     cy+0.78R~1.15R（y=999-1149），所以 mask 跟 BACARDÍ 差了 200+px
     → BACARDÍ / MYHEART / Est. / 1862 全部漏擦
  ❌ v199 PURPLE_BG_RGB=(185,150,170) — 跟真实背景 (183,127,171) 差了 25 蓝
     → 填充块颜色跟原图明显不同，看起来像"白盒子贴上去"
  ✅ v200 修复：直接连通域分析取蝙蝠+环的 bbox，用真实像素坐标定 mask，
     用真实采样色 (183,127,171) 填底

步骤：
  [1] 读 BACARDÍ 紫底源
  [2] 连通域分析 → 锁定蝙蝠+环的 bbox（中心 cx,cy_bat，尺寸 w_bat,h_bat）
  [3] 在原图坐标系生成 4 个 mask（bat 椭圆安全网 / LA CASA 顶弧 / 内部字区
      [BACARDÍ+MYHEART+Est./1862+底三角] / 整体外扩）
  [4] 在原图上实色填充擦字（bat_safe 严格排除）
  [5] 整图旋转 15° CCW（borderValue=真实背景色，角落自动填底色）
  [6] PIL 烧新字（顶弧 MOONCREST / 中央 CURSE / 底 EST.MMXXVI）
  [7] USM 锐化 → 出图 + 原图|结果 对照
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

# ============== 路径 ==============
PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
JOB = PROJECT / "jobs" / "smoke_v200"
JOB.mkdir(parents=True, exist_ok=True)
FONTS = PROJECT / "fonts"
OUT = JOB / "v200_bat_logo_rotated.jpg"
COMPARE = JOB / "_compare_v200.jpg"

# ============== 配色（实测采样：4 个角 + 4 条边中心都是 (183,127,171)）==============
BG_PURPLE = (183, 127, 171)      # 实测：原图背景浅紫
RING_DARK = (90, 20, 120)        # 实测：原图内环深紫
BAT_BLACK = (23, 4, 23)          # 实测：原图蝙蝠近黑
WHITE = (255, 255, 255)
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

# ============== [2] 连通域分析 → 找蝙蝠+环的 bbox ==============
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cands = []
for c in contours:
    a = cv2.contourArea(c)
    if a < 30000:                  # 排除小字块（Est. / 1862 / BACARDÍ 单字符）
        continue
    x, y, w, h = cv2.boundingRect(c)
    cands.append((a, x, y, w, h, x + w // 2, y + h // 2))
cands.sort(reverse=True)
# 第一个 = 蝙蝠+环（实测 area=167674 bbox=(522,498,510,477)）
bat_a, bx, by, bw, bh, bcx, bcy = cands[0]
print(f"[bat] bbox=({bx},{by},{bw},{bh})  center=({bcx},{bcy})  area={int(bat_a)}")
R_bat = max(bw, bh) // 2  # 240

# ============== [3] mask 几何（全部基于 bat 中心 + R_bat 相对坐标）==============
# 3a. bat_safe 椭圆（绝不能擦到蝙蝠主体）— 略大于实际 bat，留 30px 安全量
bat_safe = np.zeros((H, W), np.uint8)
safe_w = int(bw / 2 + 35)
safe_h = int(bh / 2 + 35)
cv2.ellipse(bat_safe, (bcx, bcy), (safe_w, safe_h), 0, 0, 360, 255, -1)

# 3b. 顶弧 mask_arc：蝙蝠+环上方的一段弧形带（覆盖 LA CASA 顶弧字）
#     上边 y = 280（字上沿）  下边 = bat 顶 - 10 安全间距
#     左右 边 = bat 左右 ± 20px
mask_arc = np.zeros((H, W), np.uint8)
arc_top = 280
arc_bot = by - 10
arc_left = bx - 20
arc_right = bx + bw + 20
cv2.rectangle(mask_arc, (arc_left, arc_top), (arc_right, arc_bot), 255, -1)
# 顶弧与 bat_safe 重叠的部分剔除（虽然 bat_top 已经低于 arc_bot，但保险）
mask_arc = cv2.bitwise_and(mask_arc, cv2.bitwise_not(bat_safe))

# 3c. 内部字区 mask_text：Est./1862 + BACARDÍ + MYHEART + 底三角
mask_text = np.zeros((H, W), np.uint8)
# BACARDÍ + Est. + 1862（同一 y 行）
cv2.rectangle(mask_text,
              (280, 985), (1240, 1158), 255, -1)
# MYHEART
cv2.rectangle(mask_text,
              (570, 1185), (1110, 1335), 255, -1)
# 底三角
cv2.rectangle(mask_text,
              (650, 1365), (880, 1440), 255, -1)
# 严格排除 bat_safe
mask_text = cv2.bitwise_and(mask_text, cv2.bitwise_not(bat_safe))

# 3d. 大核膨胀 18px（笔画外溢 + 旋转后边缘细节保留）
k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
mask_arc_d = cv2.dilate(mask_arc, k_d, iterations=1)
mask_text_d = cv2.dilate(mask_text, k_d, iterations=1)
# bat_safe 也膨胀作为"双保险"
bat_safe_d = cv2.dilate(bat_safe, k_d, iterations=1)
mask_arc_d = cv2.bitwise_and(mask_arc_d, cv2.bitwise_not(bat_safe_d))
mask_text_d = cv2.bitwise_and(mask_text_d, cv2.bitwise_not(bat_safe_d))
print(f"[masks] arc={int((mask_arc_d>0).sum())}px  text={int((mask_text_d>0).sum())}px")

# ============== [4] 在原图上实色填充擦字（mask 和文字像素 0 错位）==============
clean = img_rgb.copy()
clean[mask_arc_d > 0] = BG_PURPLE
clean[mask_text_d > 0] = BG_PURPLE
print("[erase] old text filled in ORIGINAL with REAL bg color")

# ============== [5] 旋转 clean 图 15° CCW（角落自动填真实背景色）==============
ANGLE = 15.0
M = cv2.getRotationMatrix2D((W // 2, H // 2), ANGLE, 1.0)  # 整图中心旋转
result_rgb = cv2.warpAffine(clean, M, (W, H),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=BG_PURPLE)
print(f"[rotate] {ANGLE}° CCW around ({W//2},{H//2}) — full image center")

# ============== [6] PIL 烧新字（全部直字）==============
result_pil = Image.fromarray(result_rgb).convert("RGBA")
draw = ImageDraw.Draw(result_pil)

# ---- MOONCREST 顶部横幅（深紫底+白字）----
banner_w = int(W * 0.46)
banner_h = int(R_bat * 0.55)
banner_x1 = (W - banner_w) // 2
# 放在 bat 顶上方约 80px 处
banner_y1 = max(40, by - 60 - banner_h)
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

# ---- CURSE 中央大字（bat 正下方，PirataOne）----
curse_y = by + bh + 150  # bat 底下方 150px
curse_size = int(R_bat * 0.55)
f_curse = ImageFont.truetype(F_PIRATA, curse_size)
curse_txt = "CURSE"
bb = draw.textbbox((0, 0), curse_txt, font=f_curse)
tw = bb[2] - bb[0]; th = bb[3] - bb[1]
curse_tx = (W - tw) // 2
curse_ty = curse_y - bb[1]
# 黑色描边（8 方向）
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

# 对照图（原图旋转 15° 作为对照右图）
orig_rot = Image.fromarray(img_rgb).rotate(-15, resample=Image.BICUBIC, fillcolor=BG_PURPLE)
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 40), (235, 235, 238))
f_lbl = ImageFont.truetype(F_PIRATA, 28)
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 60, 4), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 130, 4), "v200  ROT15+", font=f_lbl, fill=(20, 20, 20))
comp.paste(Image.fromarray(img_rgb), (0, 40))
comp.paste(result_pil, (W + gap, 40))
comp.save(COMPARE, quality=92)
print(f"[compare] {COMPARE}")
print(f"[done] job dir: {JOB}")
