"""
measure_orig_text.py — 自动测量原图 BACARDÍ 徽章 5 段文字的实际 bbox

按"读原图 → 反算字号"工作流：原图大字 BACARDÍ 实际多大 px、宽多少，v208 就还原成多大。

输出：每个区域最大文字连通域 bbox (x, y, w, h) 像素 + 笔画宽度估计。
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT = Path(r"E:\Desktop\双接口\image-fission")
ORIG = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"

orig_rgb = np.array(Image.open(ORIG).convert("RGB"))
H, W = orig_rgb.shape[:2]
print(f"原图尺寸 {W} x {H}")

# 紫色背景参考色（实测主色）
BG = np.array([181, 124, 169], dtype=np.float32)
diff = np.abs(orig_rgb.astype(np.float32) - BG[None, None, :]).mean(axis=2)
text_mask = (diff > 30).astype(np.uint8) * 255  # 非紫色像素 = 文字/图

# 5 段区域（来自 v206_c_compose.py 注释）
regions = {
    "BACARDI大字":           (200,  980, 1350, 1170),  # BACARDÍ 占此区
    "MYHEART小字":           (200, 1180, 1350, 1340),  # MYHEART/CURSE 占此区
    "LA_CASA顶弧":           (380,  300, 1170,  720),  # 弧字
    "EST左边字":             (250,  780,  500,  950),  # Est.
    "1982右边字":            (1050, 780, 1300,  950),  # 1982
}

print()
print("=" * 64)
print(f"{'区域':18s} {'bbox(x,y,w,h)':>32s} {'占H宽%':>8s}")
print("=" * 64)

for name, (x0, y0, x1, y1) in regions.items():
    crop = text_mask[y0:y1, x0:x1]
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    # 找宽度 >=30 的"字形"连通域（防噪）
    big = [(i, stats[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] >= 30]
    if big:
        i_big = max(big, key=lambda x: x[1][cv2.CC_STAT_AREA])[0]
        x, y, w, h, a = stats[i_big]
        abs_bbox = (x + x0, y + y0, w, h)
        bbox_str = f"({abs_bbox[0]:4d},{abs_bbox[1]:4d},{abs_bbox[2]:4d},{abs_bbox[3]:3d})"
        width_pct = 100 * w / W
        print(f"  {name:14s} {bbox_str:>32s} {width_pct:6.1f}%")
        # 笔画粗度估计：用 mask 像素总数 / bbox 面积 × 经验系数
        if h > 0:
            stroke_pct = 100 * a / (w * h)
            print(f"  {'  笔画覆盖%':14s} {'='*32} {stroke_pct:6.1f}%")
    else:
        print(f"  {name:14s} {'(no text found)':>32s}")

print()
print("=" * 64)
print("v208 字号目标:")
print(f"  MOONCREST 大字 = {W*0.62:.0f} 像素宽 (目标 62% 原图宽) = ~{int(W*0.62/0.65*1.0)} pt Lora Bold")
print(f"  CURSE 小字   = {W*0.62:.0f} 像素宽 = ~{int(W*0.62/0.6*0.85)} pt Lora Bold")
