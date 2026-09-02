"""
v212 — v211 精修 (徽章结构已保住, 修字鬼影/位置/三角)

= v211 暴露的 4 个问题 =
1. text mask 只抓灰度<=60 黑像素, 抗锯齿边缘没抓 → BACARDÍ 鬼影
2. EST./1862 字被 mask 擦后, 烧字位置偏出 360px 范围 → 显示不全
3. arc MOONCREST MANOR 半径=260 切到 bat, 字符压 bat
4. 三角 mask 太宽, 把三角局部擦掉了

= v212 修复 =
1. text mask 用 dilate 7px 抓抗锯齿边缘
2. EST./1862 烧字 x 改为原图精确位置 (cx-R-30 / cx+R-60)
3. arc 半径 260 -> 310 让字符写在 bat 上方 (留 50px 余量)
4. 三角 mask 用实测三角 bbox (716-836, 1410-1480), 三角内三角外都保留
5. BG_PURPLE 用原图背景采样精确值(取原图右上 5x5 像素平均)
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
import json
from pathlib import Path

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
JOB = PROJECT / "jobs" / "smoke_v206"
JOB.mkdir(parents=True, exist_ok=True)
INFO = JOB / "_analyze_orig_info.json"
F_LORA = r"E:/Desktop/双接口/image-fission/fonts/Lora-VF.ttf"

OUT = JOB / "v212_bacardi_mooncrest_final.jpg"
CMP = JOB / "_compare_v212.jpg"


def lora(size, weight=700):
    f = ImageFont.truetype(F_LORA, size)
    try: f.set_variation_by_axes([weight])
    except Exception: pass
    return f


def calibrate(text, target_w, weight=700, hint=None):
    if hint is None: hint = max(80, target_w // 3)
    lo, hi = max(8, hint // 8), hint * 3
    for _ in range(18):
        mid = (lo + hi) // 2
        f = lora(mid, weight)
        bb = f.getbbox(text)
        if bb is None: return hint
        w = bb[2] - bb[0]
        if w < target_w: lo = mid
        else: hi = mid
    return (lo + hi) // 2


def baseline_xy(text, font, x_left, y_baseline):
    bb = font.getbbox(text)
    return (x_left - bb[0], y_baseline - bb[1])


def center_x(text, font, W_canvas):
    bb = font.getbbox(text)
    w = bb[2] - bb[0]
    return (W_canvas - w) // 2 - bb[0]


def draw_text_d(draw, xy, text, font, fg, outline, outline_outer=None, r_inner=5, r_outer=1):
    x, y = xy
    if outline_outer is not None:
        for dx in range(-r_outer - r_inner, r_outer + r_inner + 1):
            for dy in range(-r_outer - r_inner, r_outer + r_inner + 1):
                d2 = dx * dx + dy * dy
                if r_inner * r_inner < d2 <= (r_inner + r_outer) * (r_inner + r_outer):
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_outer)
    for dx in range(-r_inner, r_inner + 1):
        for dy in range(-r_inner, r_inner + 1):
            d2 = dx * dx + dy * dy
            if d2 <= r_inner * r_inner:
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fg)


# ============================================================
# [1] load
# ============================================================
print("[1] load")
data = np.fromfile(str(SRC), dtype=np.uint8)
img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"  size {W}x{H}")

# 背景色精确采样 (取右上 10x10 像素平均, 远离徽章)
bg_sample = img_rgb[10:30, W-30:W-10, :].mean(axis=(0, 1)).astype(np.uint8)
BG_PURPLE = tuple(int(v) for v in bg_sample)
print(f"  BG_PURPLE 采样: {BG_PURPLE}")

# 已擦区域外圈 (用于检测鬼影): 在大字母 BACARDÍ/MYHEART 上边缘采样"已擦"的颜色
# (= BG_PURPLE 应该完全等于此)

info = json.loads(INFO.read_text())
ring = info["ring"]
text_info = info["text"]


# ============================================================
# [2] text mask (精确 + 抗锯齿 dilate 7px)
# ============================================================
print("\n[2] text mask + dilate 7px (抗锯齿)")
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# 阈值放宽到 90 抓更多抗锯齿像素
_, dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
bat_only = np.zeros_like(dark)
if n > 1:
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    bat_only = np.where(labels == idx, 255, 0).astype(np.uint8)
text_only = cv2.subtract(dark, bat_only)

# 抗锯齿 dilate: 用大椭圆核 15x15 一次
k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
text_only = cv2.dilate(text_only, k_d, iterations=1)

# 排除三角 (实测三角: cx=776, 三角宽 60, y=1420-1465, 顶 (776,1420), 左右角 (746, 1465), (806, 1465))
tri_keep = np.zeros((H, W), np.uint8)
cv2.rectangle(tri_keep, (716, 1410), (836, 1480), 255, -1)
text_only = cv2.bitwise_and(text_only, cv2.bitwise_not(tri_keep))

print(f"  text_only 像素数: {int((text_only > 0).sum())}")


# ============================================================
# [3] 纯色填充
# ============================================================
print("\n[3] 填 BG_PURPLE")
img_clean = img_rgb.copy()
img_clean[text_only > 0] = BG_PURPLE
Image.fromarray(img_clean).save(JOB / "_v212_cleaned.png", quality=95)


# ============================================================
# [4] 烧新字
# ============================================================
print("\n[4] 烧新字")
canvas = Image.fromarray(img_clean).convert("RGBA")
draw = ImageDraw.Draw(canvas)
PURPLE_FG = (35, 8, 50)
WHITE_OUT = (250, 240, 250)

# 4a: 大字 MOONCREST
big_text = "MOONCREST"
big_size = calibrate(big_text, 660, hint=200)
f_big = lora(big_size, 700)
big_x = center_x(big_text, f_big, W)
big_y = text_info["big_y_baseline"]
big_xy = baseline_xy(big_text, f_big, big_x, big_y)
draw_text_d(draw, big_xy, big_text, f_big, PURPLE_FG, (0, 0, 0),
            outline_outer=WHITE_OUT, r_inner=5, r_outer=1)
print(f"  big '{big_text}' size={big_size} x={big_x} y={big_y}")

# 4b: 中字 CURSE
small_text = "CURSE"
small_size = calibrate(small_text, 420, hint=170)
f_small = lora(small_size, 700)
small_x = center_x(small_text, f_small, W)
small_y = text_info["small_y_baseline"] + 5
small_xy = baseline_xy(small_text, f_small, small_x, small_y)
draw_text_d(draw, small_xy, small_text, f_small, PURPLE_FG, (0, 0, 0),
            outline_outer=WHITE_OUT, r_inner=5, r_outer=1)
print(f"  small '{small_text}' size={small_size} x={small_x} y={small_y}")

# 4c: 顶弧 MOONCREST MANOR (半径 310 让字符写在 bat 上方)
arc_text = "MOONCREST MANOR"
arc_r = text_info["arc_text_r"] + 55   # 260+55 = 315
arc_size = calibrate(arc_text, 540, hint=80)
f_arc = lora(arc_size, 700)
ARC_START_DEG = 200
ARC_SPAN_DEG = 140
print(f"  arc '{arc_text}' size={arc_size} r={arc_r}")
n_chars = len(arc_text)
for i, ch in enumerate(arc_text):
    if ch == " ": continue
    t = (i + 0.5) / n_chars
    deg = ARC_START_DEG + t * ARC_SPAN_DEG
    rad = math.radians(deg)
    x = ring["cx"] + arc_r * math.cos(rad)
    y = ring["cy"] + arc_r * math.sin(rad)
    ch_img = Image.new("RGBA", (arc_size * 3, arc_size * 3), (0, 0, 0, 0))
    d_ch = ImageDraw.Draw(ch_img)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx*dx + dy*dy <= 4:
                d_ch.text((arc_size + dx, arc_size + dy), ch, font=f_arc, fill=(0, 0, 0))
    d_ch.text((arc_size, arc_size), ch, font=f_arc, fill=PURPLE_FG)
    rot = -math.degrees(rad) - 90
    ch_rot = ch_img.rotate(rot, resample=Image.BICUBIC, center=(arc_size, arc_size))
    canvas.paste(ch_rot, (int(x - arc_size), int(y - arc_size)), ch_rot)
print("  arc 字写完")

# 4d: Est. (实测: 在 bat 左翼下方 y=1030, x=410 附近)
est_text = "EST."
est_size = calibrate(est_text, 70, hint=80)
f_est = lora(est_size, 700)
est_x = 380
est_y = text_info["big_y_baseline"] - 165
est_xy = baseline_xy(est_text, f_est, est_x, est_y)
draw_text_d(draw, est_xy, est_text, f_est, PURPLE_FG, (0, 0, 0),
            outline_outer=WHITE_OUT, r_inner=3, r_outer=1)
print(f"  EST. size={est_size} x={est_x} y={est_y}")

# 4e: MMXXVI
yr_text = "MMXXVI"
yr_size = calibrate(yr_text, 70, hint=80)
f_yr = lora(yr_size, 700)
yr_x = 1020
yr_y = est_y
yr_xy = baseline_xy(yr_text, f_yr, yr_x, yr_y)
draw_text_d(draw, yr_xy, yr_text, f_yr, PURPLE_FG, (0, 0, 0),
            outline_outer=WHITE_OUT, r_inner=3, r_outer=1)
print(f"  MMXXVI size={yr_size} x={yr_x} y={yr_y}")


# ============================================================
# [5] USM + 保存
# ============================================================
print("\n[5] USM")
final = canvas.convert("RGB")
final = final.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))
final.save(OUT, quality=95)
print(f"  [save] {OUT}")


# ============================================================
# [6] 对照图
# ============================================================
print("\n[6] 对照")
orig = Image.open(SRC).convert("RGB")
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 50), (235, 235, 238))
f_lbl = ImageFont.truetype(F_LORA, 28)
f_lbl.set_variation_by_axes([600])
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 60, 8), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 240, 8), "v212 BACARDI->MOONCREST (cleaned+burn)", font=f_lbl, fill=(20, 20, 20))
comp.paste(orig, (0, 50))
comp.paste(final, (W + gap, 50))
comp.save(CMP, quality=92)
print(f"  [compare] {CMP}")
print("\n=== v212 DONE ===")
