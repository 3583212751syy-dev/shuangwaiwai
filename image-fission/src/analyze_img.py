"""粗略判断图片是否「平面印花」而非「实体照片」：
   - unique_colors: 颜色数量少 → 平面色块；多 → 照片渐变
   - mean_sat: 饱和度
   - edge_density: 边缘密度（线稿/轮廓多 → 图案感）
"""
import sys
from PIL import Image
import numpy as np


def analyze(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    # 降采样统计颜色
    small = img.resize((64, 64))
    colors = small.getcolors(64 * 64)
    colors.sort(reverse=True)
    top = sum(c for c, _ in colors[:8])
    total = 64 * 64
    top8_ratio = top / total  # 前8主色占比，越高越平面
    uniq = len(colors)
    # 饱和度
    r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn)
    mean_sat = sat.mean() / 255.0
    # 边缘密度（简单灰度梯度）
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    edge = (gx + gy) / 2 / 255.0
    return {
        "size": f"{w}x{h}",
        "uniq_colors@64": uniq,
        "top8_color_ratio": round(top8_ratio, 2),
        "mean_sat": round(mean_sat, 2),
        "edge_density": round(edge, 3),
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(p, analyze(p))
        except Exception as e:
            print(p, "ERR", repr(e))
