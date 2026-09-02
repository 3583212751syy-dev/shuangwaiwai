"""
v205_a — 「全紫底」阶段（终极简化版）

v205_a_v1 三段 mask 暴露问题：边界可见 + 笔画擦不净（BACARDÍ/LA CASA 残影）

v205_a_v2 终极方案：
  1. **整张图全部填紫底**（不再精细 mask 工艺）
  2. **bat bbox 中心位置用一个深紫色块标记**（v205_c 会用新蝙蝠覆盖此区域）
  3. 保留原图配色紫色 (183,127,171)，环带深紫色 (90,20,120) 作为 bat 区域占位

输出：一张「全紫底 + 中心 bat 占位块」的干净图。等 v205_c paste 新蝙蝠上去。
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
JOB = PROJECT / "jobs" / "smoke_v205"
JOB.mkdir(parents=True, exist_ok=True)
OUT = JOB / "_v205_a_clean.png"

BG_PURPLE = (183, 127, 171)
RING_DEEP = (90, 20, 120)
print(f"[job] {JOB}")

# [1] 读原图
data = np.fromfile(str(SRC), dtype=np.uint8)
img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"[load] {W}x{H}")

# [2] bat bbox 定位（v203 验证的连通域方法）
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

# [3] 全图填紫底
clean = np.full_like(img_rgb, BG_PURPLE, dtype=np.uint8)

# [4] bat bbox 中心画一个深紫占位圆（v205_c paste 新蝙蝠的位置锚点）
# 用比 v205_c 新蝙蝠稍大的圆，方便透明边缘过渡
BAT_PLACE_R = max(bw, bh) // 2 + 30  # 比 bat bbox 半宽大 30px
cv2.circle(clean, (bcx, bcy), BAT_PLACE_R, RING_DEEP, -1)
print(f"[bat-place] center=({bcx},{bcy}) R={BAT_PLACE_R}  深紫占位圆")

# [5] 输出
Image.fromarray(clean).save(OUT, quality=95)
print(f"[save] {OUT}")

# 同时保存 bat bbox 信息供 v205_c 用
info_path = JOB / "_v205_a_bat_info.json"
import json
with open(info_path, "w") as f:
    json.dump({
        "W": W, "H": H,
        "bat_bbox": [bx, by, bw, bh],
        "bat_center": [bcx, bcy],
        "bat_place_radius": BAT_PLACE_R,
        "BG_PURPLE": list(BG_PURPLE),
        "RING_DEEP": list(RING_DEEP),
    }, f, indent=2)
print(f"[info] {info_path}")
print(f"\n[done] v205_a 全紫底完成。下一步：v205_b SDXL 新蝙蝠")
