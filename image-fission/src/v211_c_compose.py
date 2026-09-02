"""
v211 — BACARDÍ→MOONCREST 徽章裂变 (v210 根因修复)

= v210 失败根因 =
cv2.inpaint (TELEA radius=5) 会把文字擦除扩散到周边像素,
  把徽章外框 / 底三角 / bat 下半 涂抹成怪异羽毛阴影.
  即使限制 mask 不覆盖 bat, inpaint 也会从 mask 边缘扩散污染徽章元素.

= v211 修复 =
1. 完全放弃 cv2.inpaint
2. 用 analyze_orig 的 text_only (精确笔画像素 mask) **直接填 BG_PURPLE 纯色**
   - mask 是精确到笔画像素, 不包含 bat/外框/三角相邻区域
   - 填色无扩散, 不会涂抹任何徽章元素
3. 排除倒三角区域 — 保留原三角
4. 大字 MOONCREST / 中字 CURSE 写位置准确(实测 baseline_y)
5. arc MOONCREST MANOR 跨顶部

= 字体 =
Lora-VF Bold 700 (humanist-serif 替代 Didone, --top-thr 0.45 预设接受)
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

OUT = JOB / "v211_bacardi_mooncrest_final.jpg"
CMP = JOB / "_compare_v211.jpg"


def lora(size, weight=700):
    f = ImageFont.truetype(F_LORA, size)
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    return f


def calibrate(text, target_w, weight=700, hint=None):
    if hint is None:
        hint = max(80, target_w // 3)
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
    """烧字: 描边 + 可选白外圈 + 主色."""
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
# [1] load 原图
# ============================================================
print("[1] load")
data = np.fromfile(str(SRC), dtype=np.uint8)
img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"  size {W}x{H}")

info = json.loads(INFO.read_text())
ring = info["ring"]
text_info = info["text"]

# ============================================================
# [2] text_only 精确 mask (笔画像素, 不含 bat/外框/三角)
# ============================================================
print("\n[2] text_only mask (精确笔画像素)")
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
# 找 bat (最大连通域)
n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
bat_only = np.zeros_like(dark)
if n > 1:
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    bat_only = np.where(labels == idx, 255, 0).astype(np.uint8)
# text_only = 黑像素 - bat
text_only = cv2.subtract(dark, bat_only)
# 排除底三角区域(保留原三角)
# 三角实测: cx=776, y=1420-1465, 顶底点 (776, 1420), 左右角 (746, 1465) (806, 1465)
# 三角宽度 60, 高度 ~45
tri_keep = np.zeros((H, W), np.uint8)
cv2.rectangle(tri_keep, (716, 1410), (836, 1480), 255, -1)  # 给三角多一些边界保险
text_only = cv2.bitwise_and(text_only, cv2.bitwise_not(tri_keep))

# 排除圆形徽章外框外侧(防止擦字侵蚀边缘)
# 外框外圆 r=421, cx=776, cy=745
# 在圆内的文字外扩 30px 仍属徽章
# text_only 在徽章外就是 BACARDÍ 大字 MYHEART 三角(三角已排)
# 安全: 不动 bat_only, 不动徽章外圆环, text_only 仅覆盖纯文字笔画

print(f"  text_only 像素数: {int((text_only > 0).sum())}")

# ============================================================
# [3] 直接填 BG_PURPLE (纯色, 不扩散)
# ============================================================
print("\n[3] 纯色填 BG_PURPLE (不扩散)")
BG_PURPLE = np.array(info.get("BG_PURPLE", [183, 127, 171]), dtype=np.uint8)  # 取实测
# 原图浅紫粉(背景) RGB (183, 127, 171) — 但 text_only 区域原本是黑色字
# 填什么色? 原图字背景在 BACARDÍ 字母后面也是浅紫粉(在徽章外)
# 但 BACARDÍ 字下面就是徽章底色 — 不对, BACARDÍ 在徽章**外** (徽章是上半的圆, BACARDÍ 在 y=950+)
# 让 BACARDÍ 后面填原图背景色即可
img_clean = img_rgb.copy()
img_clean[text_only > 0] = BG_PURPLE
print(f"  填充完成")

# 保存中间产物
Image.fromarray(img_clean).save(JOB / "_v211_cleaned.png", quality=95)
print(f"  [save] _v211_cleaned.png")

# ============================================================
# [4] 烧新字
# ============================================================
print("\n[4] 烧新字 (PIL on top of cleaned image)")
canvas = Image.fromarray(img_clean).convert("RGBA")
draw = ImageDraw.Draw(canvas)

PURPLE_FG = (35, 8, 50)
WHITE_OUT = (250, 240, 250)

# 4a: 大字 MOONCREST
big_text = "MOONCREST"
big_size = calibrate(big_text, 613, hint=200)
f_big = lora(big_size, 700)
big_x = center_x(big_text, f_big, W)
big_y = text_info["big_y_baseline"]
big_xy = baseline_xy(big_text, f_big, big_x, big_y)
draw_text_d(draw, big_xy, big_text, f_big, PURPLE_FG, (0, 0, 0),
            outline_outer=WHITE_OUT, r_inner=5, r_outer=1)
print(f"  big '{big_text}' size={big_size} x={big_x} y={big_y}")

# 4b: 中字 CURSE
small_text = "CURSE"
small_size = calibrate(small_text, 387, hint=170)
f_small = lora(small_size, 700)
small_x = center_x(small_text, f_small, W)
small_y = text_info["small_y_baseline"] + 8
small_xy = baseline_xy(small_text, f_small, small_x, small_y)
draw_text_d(draw, small_xy, small_text, f_small, PURPLE_FG, (0, 0, 0),
            outline_outer=WHITE_OUT, r_inner=5, r_outer=1)
print(f"  small '{small_text}' size={small_size} x={small_x} y={small_y}")

# 4c: 顶弧 MOONCREST MANOR
arc_text = "MOONCREST MANOR"
arc_r = text_info["arc_text_r"]
arc_size = calibrate(arc_text, 514, hint=80)
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

# 4d: Est. (保持与原图对应位置, 紧贴 bat 左下)
est_text = "EST."
est_size = calibrate(est_text, 52, hint=80)
f_est = lora(est_size, 700)
est_x = 410
est_y = text_info["big_y_baseline"] - 145
est_xy = baseline_xy(est_text, f_est, est_x, est_y)
draw_text_d(draw, est_xy, est_text, f_est, PURPLE_FG, (0, 0, 0),
            outline_outer=WHITE_OUT, r_inner=3, r_outer=1)
print(f"  EST. size={est_size} x={est_x} y={est_y}")

# 4e: MMXXVI
yr_text = "MMXXVI"
yr_size = calibrate(yr_text, 52, hint=80)
f_yr = lora(yr_size, 700)
yr_x = 1010
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
dl.text((W + gap + W // 2 - 220, 8), "v211 BACARDI->MOONCREST (fill+burn)", font=f_lbl, fill=(20, 20, 20))
comp.paste(orig, (0, 50))
comp.paste(final, (W + gap, 50))
comp.save(CMP, quality=92)
print(f"  [compare] {CMP}")
print("\n=== v211 DONE ===")
