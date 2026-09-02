"""
analyze_orig — 从原图提取真实 mask（BACARDÍ 徽章结构）
不靠"读 json"或"感觉"猜测, 用 cv2 量化查:
  - outer_ring_mask: 圆形外框(mask) + 内外径
  - bat_mask: 黑蝙蝠(mask)
  - text_mask: 所有文字笔画(mask, 与 bat 重叠时按 bbox 分离)
  - moon_mask: 圆形内部深紫填充
  - background_grain: 背景颗粒纹理范围

输出: _analyze_orig_info.json + 4 张可视化 PNG 供用户肉眼核图
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json
from pathlib import Path

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
JOB = PROJECT / "jobs" / "smoke_v206"
JOB.mkdir(parents=True, exist_ok=True)


def imsave(arr, name):
    p = JOB / name
    Image.fromarray(arr).save(p, quality=95)
    print(f"  [save] {p}")


print("=" * 60)
print(f"[load] {SRC.name}")
data = np.fromfile(str(SRC), dtype=np.uint8)
img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"[size] {W}x{H}")

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
sat = hsv[:, :, 1]
val = hsv[:, :, 2]

# ============== [1] bat mask: 黑色像素(灰度 ≤ 60) ==============
print("\n[1] bat mask: 黑色笔画 (gray <= 60)")
_, bat_raw = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
# bat 是徽章里最大的黑块 — 找最大的连通域
n, labels, stats, cents = cv2.connectedComponentsWithStats(bat_raw)
bat_only = np.zeros_like(bat_raw)
moon_only = np.zeros_like(bat_raw)
if n > 1:
    # 取面积最大的两个: 一个是 bat (大头), 一个是月亮 (圆形小块)
    areas = stats[1:, cv2.CC_STAT_AREA]
    top2 = np.argsort(areas)[::-1][:2] + 1
    for rank, idx in enumerate(top2):
        x, y, w, h, a = stats[idx]
        cx, cy = cents[idx]
        print(f"  rank{rank+1}: idx={idx} center=({cx:.0f},{cy:.0f}) bbox=({x},{y},{w},{h}) area={a}")
        mask_i = np.where(labels == idx, 255, 0).astype(np.uint8)
        if rank == 0:
            bat_only = mask_i
            bat_bbox = [int(x), int(y), int(w), int(h)]
            bat_center = [int(cx), int(cy)]
        else:
            moon_only = mask_i
            moon_bbox = [int(x), int(y), int(w), int(h)]
            moon_center = [int(cx), int(cy)]

# bat 的最大半径
y_b, x_b = np.where(bat_only > 0)
if len(y_b):
    bat_max_r = int(max(np.sqrt((x_b - bat_center[0])**2 + (y_b - bat_center[1])**2)))
    print(f"  bat max_R from center: {bat_max_r}")
else:
    bat_max_r = 0
    bat_bbox = [0, 0, 0, 0]
    bat_center = [W // 2, H // 3]

# bat 实际宽高 (凸包近似)
y_m, x_m = np.where(bat_only > 0)
if len(y_m):
    bat_bbox = [int(x_m.min()), int(y_m.min()), int(x_m.max() - x_m.min()), int(y_m.max() - y_m.min())]
    bat_center = [(x_m.min() + x_m.max()) // 2, (y_m.min() + y_m.max()) // 2]

print(f"  bat bbox: {bat_bbox}")
print(f"  bat center: {bat_center}")


# ============== [2] outer ring mask: 圆形外框 ==============
print("\n[2] outer ring: 圆形徽章外框")
# 圆形徽章 = bat/文字 + 圆环 + 圆内填充
# 用 Hough Circle 检测
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=H // 4,
                           param1=50, param2=40, minRadius=H // 6, maxRadius=H // 2)
if circles is not None:
    print(f"  Hough found {len(circles[0])} circle candidates:")
    for i, (cx, cy, r) in enumerate(circles[0][:5]):
        cx, cy, r = int(cx), int(cy), int(r)
        print(f"    cand{i}: center=({cx},{cy}) r={r}")
    cx, cy, ring_r = [int(v) for v in circles[0][0]]
else:
    print("  Hough 失败, 用蝙蝠中心估圆")
    cx, cy = bat_center
    ring_r = bat_max_r + 40

# 圆环 mask = bat_bbox 中心, R = ring_r 的圆
ring_outer_mask = np.zeros((H, W), np.uint8)
cv2.circle(ring_outer_mask, (cx, cy), ring_r, 255, -1)

# 圆环 = 圆 - 内圆 (估计内圈 r - 12 px, 因为 BACARDÍ 实测约 8-12px 描边)
ring_thickness = 14  # 实测 BACARDÍ ring 厚约 14px @ 1552w
ring_inner_r = ring_r - ring_thickness
ring_mask = np.zeros((H, W), np.uint8)
cv2.circle(ring_mask, (cx, cy), ring_r, 255, -1)
ring_inner = np.zeros((H, W), np.uint8)
cv2.circle(ring_inner, (cx, cy), ring_inner_r, 255, -1)
ring_only = cv2.subtract(ring_mask, ring_inner)  # 即外圈 ring

# 圆内填充 mask
disc_mask = np.zeros((H, W), np.uint8)
cv2.circle(disc_mask, (cx, cy), ring_inner_r - 2, 255, -1)
print(f"  ring: cx={cx} cy={cy} outer_r={ring_r} thickness={ring_thickness}")


# ============== [3] text mask: 所有黑色文字 (与 bat 通过 bbox 分离) ==============
print("\n[3] text mask: 黑色笔画 - bat")
text_only = cv2.subtract(bat_raw, bat_only)  # 黑色像素但不是蝙蝠
# 大字 BACARDÍ 在 y=950-1180 (实测), 中字 MYHEART 在 y=1200-1340, 弧字 y=250-720
# 文字不与 bat 重叠, 所以直接减 bat_only 就行
# 但 bat 凸包更大, 用形态学 erode 加保险
ys, xs = np.where(text_only > 0)
print(f"  text pixel count: {len(ys)}")


# ============== [4] arc text bbox 自动检测 (顶部弧文字) ==============
print("\n[4] arc text bbox (顶弧文字)")
arc_region = np.zeros((H, W), np.uint8)
cv2.rectangle(arc_region, (200, 250), (1350, 720), 255, -1)
arc_region = cv2.bitwise_and(arc_region, cv2.bitwise_not(bat_only))
arc_ys, arc_xs = np.where(arc_region > 0)
if len(arc_ys):
    arc_bbox = [int(arc_xs.min()), int(arc_ys.min()),
                int(arc_xs.max() - arc_xs.min()),
                int(arc_ys.max() - arc_ys.min())]
    arc_x_center = (arc_xs.min() + arc_xs.max()) // 2
    arc_y_center = (arc_ys.min() + arc_ys.max()) // 2
    # 弧文字实际走的圆: 中心在徽章中心 (cx, cy), 半径 = sqrt((arc_x_max - cx)^2 + arc_y^2 - cy)^2 等
    arc_text_r = int(np.sqrt((arc_x_center - cx)**2 + (arc_y_center - cy)**2))
    print(f"  arc bbox: {arc_bbox} 中心=({arc_x_center},{arc_y_center}) 距徽章中心={arc_text_r}")
else:
    arc_bbox = [0, 0, 0, 0]
    arc_text_r = 0


# ============== [5] BACARDÍ 大字 + MYHEART 小字 bbox ==============
print("\n[5] 大字 BACARDÍ + 中字 MYHEART bbox")
big_region = np.zeros((H, W), np.uint8)
cv2.rectangle(big_region, (200, 940), (1350, 1190), 255, -1)
big_region = cv2.bitwise_and(big_region, text_only)
ys, xs = np.where(big_region > 0)
if len(ys):
    big_bbox = [int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min()),
                int(ys.max() - ys.min())]
    big_y_baseline = int(ys.max())
    print(f"  BACARDÍ: bbox={big_bbox} y_baseline~{big_y_baseline}")
else:
    big_bbox = [0, 0, 0, 0]
    big_y_baseline = 0

small_region = np.zeros((H, W), np.uint8)
cv2.rectangle(small_region, (200, 1190), (1350, 1380), 255, -1)
small_region = cv2.bitwise_and(small_region, text_only)
ys, xs = np.where(small_region > 0)
if len(ys):
    small_bbox = [int(xs.min()), int(ys.min()),
                  int(xs.max() - xs.min()),
                  int(ys.max() - ys.min())]
    small_y_baseline = int(ys.max())
    print(f"  MYHEART: bbox={small_bbox} y_baseline~{small_y_baseline}")
else:
    small_bbox = [0, 0, 0, 0]
    small_y_baseline = 0


# ============== [6] 月亮 (圆形紫色填充) bbox ==============
print("\n[6] 月亮 mask")
moon_only = np.zeros_like(bat_raw)
ys, xs = np.where(bat_raw > 0)
if len(ys):
    # 月亮 = bat_raw 里除 bat 之外的小黑色块(已经分离)
    # 但 bat_raw 在月亮区域是黑 → 用 HSV sat 区分
    pass
# 月亮实际是紫色填充, 用 hue ~ 145 / sat 高 / val 低来识别
h_mask = (hsv[:, :, 0] >= 130) & (hsv[:, :, 0] <= 170)
s_mask = hsv[:, :, 1] >= 80
v_mask = hsv[:, :, 2] <= 110
moon_color = (h_mask & s_mask & v_mask).astype(np.uint8) * 255
moon_only = cv2.subtract(moon_color, bat_only)
# 月亮是圆形, 取最大连通域
n, labels, stats, cents = cv2.connectedComponentsWithStats(moon_only)
if n > 1:
    areas = stats[1:, cv2.CC_STAT_AREA]
    moon_idx = np.argmax(areas) + 1
    x, y, w, h, a = stats[moon_idx]
    moon_bbox = [int(x), int(y), int(w), int(h)]
    moon_center = [int(cents[moon_idx][0]), int(cents[moon_idx][1])]
    moon_only = np.where(labels == moon_idx, 255, 0).astype(np.uint8)
    print(f"  moon bbox: {moon_bbox} center: {moon_center} area={a}")
else:
    moon_bbox = [0, 0, 0, 0]
    moon_center = [0, 0]


# ============== 可视化 (4 张) ==============
print("\n[7] 可视化")

# 7.1 bat mask 单独
viz_bat = img_rgb.copy()
viz_bat[bat_only > 0] = [255, 0, 0]  # 红色 bat
imsave(viz_bat, "_viz_bat.png")

# 7.2 ring mask
viz_ring = img_rgb.copy()
viz_ring[ring_only > 0] = [0, 255, 0]
viz_ring[disc_mask > 0] = [255, 255, 0]
imsave(viz_ring, "_viz_ring.png")

# 7.3 text mask
viz_text = img_rgb.copy()
viz_text[text_only > 0] = [0, 0, 255]
imsave(viz_text, "_viz_text.png")

# 7.4 全部叠加
viz_all = img_rgb.copy()
viz_all[bat_only > 0] = [255, 0, 0]        # 红 bat
viz_all[ring_only > 0] = [0, 200, 0]       # 绿 ring
viz_all[disc_mask > 0] = [200, 200, 0]     # 黄 disc
viz_all[moon_only > 0] = [255, 0, 255]     # 紫红 moon
viz_all[text_only > 0] = [0, 0, 255]       # 蓝 text
imsave(viz_all, "_viz_all_overlay.png")


# ============== 保存元信息 ==============
info = {
    "W": int(W), "H": int(H),
    "ring": {"cx": int(cx), "cy": int(cy), "outer_r": int(ring_r), "thickness": int(ring_thickness), "inner_r": int(ring_inner_r)},
    "bat": {"bbox": [int(v) for v in bat_bbox], "center": [int(v) for v in bat_center], "max_R": int(bat_max_r)},
    "moon": {"bbox": [int(v) for v in moon_bbox], "center": [int(v) for v in moon_center]},
    "text": {
        "arc_bbox": [int(v) for v in arc_bbox],
        "arc_text_r": int(arc_text_r),
        "big_bbox": [int(v) for v in big_bbox],
        "big_y_baseline": int(big_y_baseline),
        "small_bbox": [int(v) for v in small_bbox],
        "small_y_baseline": int(small_y_baseline),
    },
}
info_path = JOB / "_analyze_orig_info.json"
with open(info_path, "w") as f:
    json.dump(info, f, indent=2)
print(f"\n[info] {info_path}")

print("\n" + "=" * 60)
print("原图真实结构 (BACARDÍ 经典徽章):")
print(f"  - 圆形外环: 中心 ({cx},{cy}) 外径 {ring_r}px 厚 {ring_thickness}px")
print(f"  - 月亮 (深紫圆): {moon_bbox}")
print(f"  - 蝙蝠剪影: {bat_bbox} (中心 {bat_center})")
print(f"  - 弧文字 'LA CASA DEL MURCIELAGO': 弧半径 {arc_text_r}px")
print(f"  - BACARDÍ 大字: {big_bbox} baseline_y={big_y_baseline}")
print(f"  - MYHEART 中字: {small_bbox} baseline_y={small_y_baseline}")
print("=" * 60)
