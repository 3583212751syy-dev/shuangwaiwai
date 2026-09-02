"""
v206_a — mask 擦字工艺（v203 验证的几何，**不旋转**，bat_safe 内清空）

设计原则（用户 11:17 明确）：
  - **构图保留**：BACARDÍ 原始 logo 结构
  - **颜色保留**：紫色配色锁死
  - **结构保留**：1552x2000 portrait + 5 段垂直布局
  - **只元素裂变**：bat / 字 / 装饰元素 可换

mask 几何（v203 验证 100% 干净）：
  - 顶弧 y=300-720
  - bat 中部两侧 y=770-960 (左 x=200-510, 右 x=1000-1350)
  - 下方 y=980-1465 (含 BACARDÍ/MYHEART/底三角)
  - **bat_safe 椭圆中心区域 y=498-975, x=492-1062 也要清空**（v206_b 要 paste 新蝙蝠）

输出：1552x2000 全紫底图，bat bbox 矩形保留原 bat 像素但要被擦空到紫底。
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
JOB = PROJECT / "jobs" / "smoke_v206"
JOB.mkdir(parents=True, exist_ok=True)
OUT = JOB / "_v206_a_clean.png"
print(f"[job] {JOB}")

# 配色（v203 实测）
BG_PURPLE = (183, 127, 171)
RING_DEEP = (90, 20, 120)

# [1] 读原图
data = np.fromfile(str(SRC), dtype=np.uint8)
img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"[load] {W}x{H}")

# [2] bat bbox 定位
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cands = []
for c in contours:
    a = cv2.contourArea(c)
    if a < 30000: continue
    x, y, w, h = cv2.boundingRect(c)
    cands.append((a, x, y, w, h))
cands.sort(reverse=True)
bat_a, bx, by, bw, bh = cands[0]
bcx, bcy = bx + bw // 2, by + bh // 2
print(f"[bat] bbox=({bx},{by},{bw}x{bh}) center=({bcx},{bcy})")

# bat_safe 椭圆 = bat bbox 半宽 +30
BAT_R = max(bw, bh) // 2 + 30  # ~285
print(f"[bat-safe] R={BAT_R}")

# [3] bat_safe 椭圆（bat bbox 矩形范围内）
bat_safe = np.zeros((H, W), np.uint8)
safe_w = int(bw / 2 + 30); safe_h = int(bh / 2 + 30)
cv2.ellipse(bat_safe, (bcx, bcy), (safe_w, safe_h), 0, 0, 360, 255, -1)
bat_bbox_mask = np.zeros((H, W), np.uint8)
cv2.rectangle(bat_bbox_mask, (bx, by), (bx + bw, by + bh), 255, -1)
bat_safe = cv2.bitwise_and(bat_safe, bat_bbox_mask)

# [4] mask 三段 (v203 几何 + bat_safe 内清空 = 4 段)
mask_all = np.zeros((H, W), np.uint8)
# 顶弧 LA CASA
cv2.rectangle(mask_all, (200, 300), (1350, 730), 255, -1)
# bat 中部两侧 Est./1862
cv2.rectangle(mask_all, (200, 770), (530, 960), 255, -1)
cv2.rectangle(mask_all, (1000, 770), (1350, 960), 255, -1)
# 下方 BACARDÍ/MYHEART/底三角
cv2.rectangle(mask_all, (200, 980), (1350, 1465), 255, -1)
# bat_safe 内清空给 SDXL 新蝙蝠
mask_all = cv2.bitwise_or(mask_all, bat_safe)

# 膨胀（笔画外溢吸收）
k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
mask_all_d = cv2.dilate(mask_all, k_d, iterations=1)
print(f"[masks] all_d={int((mask_all_d>0).sum())}px")

# [5] 填充紫底
clean = img_rgb.copy()
clean[mask_all_d > 0] = BG_PURPLE

# [6] 中心 bat_safe 区域填 RING_DEEP（蝙蝠区域标记，等 SDXL 新蝙蝠覆盖）
clean[bat_safe > 0] = RING_DEEP

Image.fromarray(clean).save(OUT, quality=95)
print(f"[save] {OUT}")

# [7] 保存元信息供 v206_c 用
import json
info = {
    "W": W, "H": H,
    "bat_bbox": [bx, by, bw, bh],
    "bat_center": [bcx, bcy],
    "bat_R": BAT_R,
    # 文字坐标（v203 验证的精确位置）
    "arc_outer_radius": BAT_R,
    "arc_inner_radius": BAT_R - 40,
    # BACARDÍ 大字位置（v203 验证）
    "big_text_top_y": 1000,
    "big_text_size": 200,
    "small_text_top_y": 1185,
    "small_text_size": 170,
    "est_year_top_y": 1050,  # Est./1862 在 bat 中心水平
    "est_year_size": 60,
    "tri_top_y": 1370,
    "BG_PURPLE": list(BG_PURPLE),
    "RING_DEEP": list(RING_DEEP),
}
info_path = JOB / "_v206_a_bat_info.json"
with open(info_path, "w") as f:
    json.dump(info, f, indent=2)
print(f"[info] {info_path}")
