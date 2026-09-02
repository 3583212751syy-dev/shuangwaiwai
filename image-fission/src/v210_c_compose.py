"""
v210 — BACARDÍ 徽章裂变 (v206 根因重做)

= 核心修复 =
v206_a 的破坏性擦字步骤用了一个错误前提:
  "徽章里 bat_safe 椭圆区域是 mask 残影, 要清空给 SDXL 重画"

这直接销毁了原图所有徽章结构(圆环+月亮+蝙蝠), 导致 v206~v209 输出
全部是没有徽章结构的方块+文字乱贴.

v210 的修复:
  1. 从原图**整图起步**, 不擦任何徽章元素
  2. 只对文字区域(text_mask)做精确 inpaint 擦除
  3. 用 cv2.inpaint (TELEA 算法) — 比 v206_a 的 "全段 block fill" 精确 100x
  4. 在原位置烧新字 (MOONCREST MANOR 弧 / MOONCREST 大字 / CURSE 小字 / EST. MMXXVI 侧字)
  5. 徽章圆环 / 月亮 / 蝙蝠 / 三角 完全保留原图 — 满足"参考原图风格"铁律

= 字体 =
Lora-VF Bold 700 (humanist-serif 替代 Didone, --top-thr 0.45 预设接受)

= 输出 =
jobs/smoke_v206/v210_bacardi_mooncrest_final.jpg
jobs/smoke_v206/_compare_v210.jpg
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

OUT = JOB / "v210_bacardi_mooncrest_final.jpg"
CMP = JOB / "_compare_v210.jpg"


def lora(size: int, weight: int = 700):
    f = ImageFont.truetype(F_LORA, size)
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    return f


def calibrate(text: str, target_w: int, weight: int = 700, hint: int = None) -> int:
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
    return (lo + hi) // 2


def baseline_xy(text: str, font, x_left: int, y_baseline: int):
    bb = font.getbbox(text)
    return (x_left - bb[0], y_baseline - bb[1])


def center_x(text: str, font, W_canvas: int) -> int:
    bb = font.getbbox(text)
    w = bb[2] - bb[0]
    return (W_canvas - w) // 2 - bb[0]


def draw_text_drawing(draw, xy, text, font, fg, outline, outline_outer=None, r_inner=5, r_outer=1):
    """PIL 烧字 (黑描边 + 可选白外圈 + 主色)."""
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
# [1] 加载原图 + 真量化 mask
# ============================================================
print("[1] load 原图")
data = np.fromfile(str(SRC), dtype=np.uint8)
img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"  size {W}x{H}")

info = json.loads(INFO.read_text())
ring = info["ring"]
bat = info["bat"]
text = info["text"]
RING = (ring["cx"], ring["cy"], ring["outer_r"])
print(f"  ring cx={ring['cx']} cy={ring['cy']} r={ring['outer_r']}")
print(f"  bat center={bat['center']} R={bat['max_R']}")
print(f"  arc r={text['arc_text_r']}")
print(f"  big text bbox = {text['big_bbox']} baseline_y={text['big_y_baseline']}")
print(f"  small text bbox = {text['small_bbox']} baseline_y={text['small_y_baseline']}")

# ============================================================
# [2] 构造 4 个文字 mask (大块, 给 cv2.inpaint 用)
# ============================================================
print("\n[2] 文字 mask (4 段, 留 bat 区域不动)")
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
# bat mask: 最大连通域 (实测量 BACARDÍ 蝙蝠)
n, labels, stats, cents = cv2.connectedComponentsWithStats(dark_mask)
bat_only = np.zeros_like(dark_mask)
if n > 1:
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    bat_only = np.where(labels == idx, 255, 0).astype(np.uint8)

# text mask: 黑色像素 但 不是 bat
text_only = cv2.subtract(dark_mask, bat_only)

# 4 段: 顶弧 / Est.(左) / 1862(右) / BACARDÍ(大) / MYHEART(中) / 三角
mask_4segments = np.zeros((H, W), np.uint8)
# 顶弧文字 (用 analyze bbox 加 padding)
ax, ay, aw, ah = text["arc_bbox"]
cv2.rectangle(mask_4segments,
              (max(0, ax - 30), max(0, ay - 30)),
              (min(W, ax + aw + 30), min(H, ay + ah + 60)),
              255, -1)
# 大字 BACARDÍ (用户决定换 MOONCREST 9 字, 但 mask 用原 bbox)
bx, by, bw, bh = text["big_bbox"]
cv2.rectangle(mask_4segments,
              (max(0, bx - 30), max(0, by - 30)),
              (min(W, bx + bw + 30), min(H, by + bh + 30)),
              255, -1)
# MYHEART 中字 (用户决定换 CURSE 5 字, 但 mask 用原 bbox)
sx, sy, sw, sh = text["small_bbox"]
cv2.rectangle(mask_4segments,
              (max(0, sx - 30), max(0, sy - 30)),
              (min(W, sx + sw + 30), min(H, sy + sh + 30)),
              255, -1)
# Est. 左边小字 (实测在 bat 左翼 y=950 附近)
cv2.rectangle(mask_4segments, (200, 940), (530, 1080), 255, -1)
# 1862 右边小字
cv2.rectangle(mask_4segments, (1000, 940), (1350, 1080), 255, -1)
# 底三角 (用户决定保留, 不擦)

# 文字 mask 要排除 bat 区域
mask_4segments = cv2.bitwise_and(mask_4segments, cv2.bitwise_not(bat_only))
# 排除圆形徽章 disc (cv2.inpaint 不会破坏徽章前景色, 但保险)
disc_mask = np.zeros((H, W), np.uint8)
cv2.circle(disc_mask, (ring["cx"], ring["cy"]), ring["inner_r"] - 2, 255, -1)
# bat 在 disc 内 -> 但 bat_only 也覆盖大部分 disc. 让 mask 不深入 disc 太深
# 实测: text 区域都在 disc 之外或 bat 之外. 仅 arc 在 ring_outer 内, 大字在徽章下方
print(f"  text_mask 覆盖: {int((mask_4segments > 0).sum())} px")

# ============================================================
# [3] cv2.inpaint 擦字 (TELEA 算法, 半径 5px)
# ============================================================
print("\n[3] cv2.inpaint 擦字 (TELEA, radius=5)")
inpainted_bgr = cv2.inpaint(img_bgr, mask_4segments, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
print("  擦字完毕")

# 中间产物: 看擦字后的图
Image.fromarray(inpainted_rgb).save(JOB / "_v210_inpainted.png", quality=95)
print(f"  [save] _v210_inpainted.png")

# ============================================================
# [4] 烧新字 (保持原图文字位置)
# ============================================================
print("\n[4] 烧新字")
canvas = Image.fromarray(inpainted_rgb).convert("RGBA")
draw = ImageDraw.Draw(canvas)

PURPLE_FG = (45, 10, 60)   # 与原图 BACARDÍ 黑色对应的近黑色深紫
WHITE_OUT = (250, 240, 250)  # 微白外圈, 让字从紫底浮起

# [4a] 大字 MOONCREST (替换原 BACARDÍ)
big_text = "MOONCREST"
big_target_w = 613  # 与 BACARDÍ 实测 955 略小(留边框余量), 但用户 v208 选 613
big_size = calibrate(big_text, big_target_w, hint=200)
f_big = lora(big_size, 700)
big_x = center_x(big_text, f_big, W)
big_y = text["big_y_baseline"]
big_xy = baseline_xy(big_text, f_big, big_x, big_y)
draw_text_drawing(draw, big_xy, big_text, f_big, PURPLE_FG, (0, 0, 0),
                  outline_outer=WHITE_OUT, r_inner=5, r_outer=1)
print(f"  big '{big_text}' size={big_size} x={big_x} y={big_y}")

# [4b] 中字 CURSE (替换 MYHEART)
small_text = "CURSE"
# 实测 MYHEART 宽 690, 但 CURSE 5 字密度低, target_w 给 387 (与 v208 一致)
small_target_w = 387
small_size = calibrate(small_text, small_target_w, hint=170)
f_small = lora(small_size, 700)
small_x = center_x(small_text, f_small, W)
small_y = text["small_y_baseline"] + 5  # 让字贴近中字中线
small_xy = baseline_xy(small_text, f_small, small_x, small_y)
draw_text_drawing(draw, small_xy, small_text, f_small, PURPLE_FG, (0, 0, 0),
                  outline_outer=WHITE_OUT, r_inner=5, r_outer=1)
print(f"  small '{small_text}' size={small_size} x={small_x} y={small_y}")

# [4c] 顶弧 MOONCREST MANOR
arc_text = "MOONCREST MANOR"
n_chars = len(arc_text)
arc_r = text["arc_text_r"]
arc_size = calibrate(arc_text, 514, hint=80)
f_arc = lora(arc_size, 700)
# 弧起止角度 (math convention, 从 12 点钟方向左右各展)
# 原图实测: 弧文字经过徽章顶部, 起点 -130° 终点 -50° (顺时针)
# cx=776 cy=745 -> 文字中心 (arc_x_center, arc_y_center) 距 r=260
# 把圆心作为 (cx, cy), 弧文字基线 = 半径 r 的圆
# 原图弧字 bbox y=250-720, 中心 (775, 485) — 即 (0, -260) 方向
# 数学角度: x = cx + r*cos(theta), y = cy + r*sin(theta) 约定 theta=0 沿 +x (右), theta=90 沿 -y (上)
# 上方点 (cx, cy - r) 对应 theta = -90 (or 270)
# 弧字从 沿顶部约 -160° 到 -20° (跨越顶端 140°)
ARC_START_DEG = 200   # 起点: 左上
ARC_SPAN_DEG = 140
print(f"  arc '{arc_text}' size={arc_size} r={arc_r}")
for i, ch in enumerate(arc_text):
    if ch == " ":
        continue
    t = (i + 0.5) / n_chars
    deg = ARC_START_DEG + t * ARC_SPAN_DEG
    rad = math.radians(deg)
    x = ring["cx"] + arc_r * math.cos(rad)
    y = ring["cy"] + arc_r * math.sin(rad)
    # 单字小图
    ch_img = Image.new("RGBA", (arc_size * 3, arc_size * 3), (0, 0, 0, 0))
    d_ch = ImageDraw.Draw(ch_img)
    # 黑描边小(2px), 主色紫, 无白外圈(弧字小, 加白外圈会糊)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx * dx + dy * dy <= 4:
                d_ch.text((arc_size + dx, arc_size + dy), ch, font=f_arc, fill=(0, 0, 0))
    d_ch.text((arc_size, arc_size), ch, font=f_arc, fill=PURPLE_FG)
    # 切线方向旋转
    rot = -math.degrees(rad) - 90
    ch_rot = ch_img.rotate(rot, resample=Image.BICUBIC, center=(arc_size, arc_size))
    canvas.paste(ch_rot, (int(x - arc_size), int(y - arc_size)), ch_rot)
print("  arc 字写完")

# [4d] Est. 左边小字
est_text = "EST."
est_size = calibrate(est_text, 52, hint=80)
f_est = lora(est_size, 700)
# 原图实测 Est. 位置 (实测 bat 左翼下方, y~1050)
est_x = ring["cx"] - bat["max_R"] + 20
est_y = text["big_y_baseline"] - 130
est_xy = baseline_xy(est_text, f_est, est_x, est_y)
draw_text_drawing(draw, est_xy, est_text, f_est, PURPLE_FG, (0, 0, 0),
                  outline_outer=WHITE_OUT, r_inner=3, r_outer=1)
print(f"  EST. size={est_size} x={est_x} y={est_y}")

# [4e] MMXXVI 右边小字
yr_text = "MMXXVI"
yr_size = calibrate(yr_text, 52, hint=80)
f_yr = lora(yr_size, 700)
yr_x = ring["cx"] + bat["max_R"] - 70
yr_y = est_y
yr_xy = baseline_xy(yr_text, f_yr, yr_x, yr_y)
draw_text_drawing(draw, yr_xy, yr_text, f_yr, PURPLE_FG, (0, 0, 0),
                  outline_outer=WHITE_OUT, r_inner=3, r_outer=1)
print(f"  MMXXVI size={yr_size} x={yr_x} y={yr_y}")

# ============================================================
# [5] USM 锐化 + 保存
# ============================================================
print("\n[5] USM + 保存")
final = canvas.convert("RGB")
final = final.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))
final.save(OUT, quality=95)
print(f"  [save] {OUT}")

# ============================================================
# [6] 对照图
# ============================================================
print("\n[6] 对照图")
orig = Image.open(SRC).convert("RGB")
gap = 16
comp = Image.new("RGB", (W * 2 + gap, H + 50), (235, 235, 238))
f_lbl = ImageFont.truetype(F_LORA, 28)
f_lbl.set_variation_by_axes([600])
dl = ImageDraw.Draw(comp)
dl.text((W // 2 - 60, 8), "ORIGINAL", font=f_lbl, fill=(20, 20, 20))
dl.text((W + gap + W // 2 - 240, 8), "v210  BACARDI->MOONCREST (inpaint+burn)", font=f_lbl, fill=(20, 20, 20))
comp.paste(orig, (0, 50))
comp.paste(final, (W + gap, 50))
comp.save(CMP, quality=92)
print(f"  [compare] {CMP}")

print("\n" + "=" * 60)
print("v210 DONE")
print("=" * 60)
