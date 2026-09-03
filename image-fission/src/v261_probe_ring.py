"""v261_probe_ring — 诊断用: 把圆环内所有暗色连通域列出来, 看清真实结构
(主体到底是哪一块 / 文字各是哪几块), 为正式 detect 的分拣规则提供依据。
"""
import numpy as np
import cv2
from pathlib import Path

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

CX, CY, R_IN, R_OUT = 776, 744, 413, 427   # 上一步实测

bgr = cv2.imdecode(np.fromfile(str(COMFY_INPUT / IMG), dtype=np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

yy, xx = np.mgrid[:h, :w]
dist = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)

for th in (45, 60, 90):
    dark = ((gray < th) & (dist < R_IN - 6)).astype(np.uint8) * 255
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    print(f"\n===== dark_th={th} | 圆内连通域 {n-1} 个 =====")
    for rank, i in enumerate(order[:12]):
        idx = i + 1
        x, y, bw, bh, area = [int(v) for v in stats[idx]]
        cx_, cy_ = cent[idx]
        d_center = float(np.hypot(cx_ - CX, cy_ - CY))
        fill = area / float(bw * bh) if bw * bh else 0
        print(f"  #{rank:2d} area={area:7d} bbox=({x:4d},{y:4d},{bw:4d},{bh:4d}) "
              f"centroid=({cx_:6.1f},{cy_:6.1f}) dist_from_center={d_center:6.1f} fill={fill:.2f}")
