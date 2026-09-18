"""_diag_p4.py — 量化 p4「一块块」的真因：笔画粗细 / 连通块数 / 漂浮碎块。

指标（全部在 lum<30 定标的墨迹掩膜上，见 MEMORY 🔴28d）：
  · ink%            墨量占比
  · 平均笔宽        2*EDT(骨架) 的均值（像素）
  · 连通块数 N      <50px 的碎点数（碎化最直观的指标）
  · 大块数          >5000px 的连通块数（一棵树往往是一个大块）
  · 最大块占比      连通块面积分布
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
NEW = ROOT / "jobs" / "router_out_v329" / "Pinterest (4)_variant.jpg"
ALPHA = ROOT / "jobs" / "v346_p4aff" / "_swap_alpha.png"


def stroke_width(m):
    """2*EDT(骨架) 均值 ≈ 平均笔宽。"""
    e = ndi.distance_transform_edt(m)
    from skimage.morphology import skeletonize
    sk = skeletonize(m)
    if not sk.any():
        return 0.0
    return float(2.0 * e[sk].mean())


def rep(tag, m):
    lab, n = ndi.label(m, np.ones((3, 3), bool))
    sz = np.bincount(lab.ravel())[1:] if n else np.array([0])
    big = int((sz > 5000).sum())
    small = int((sz < 50).sum())
    tiny = int((sz < 12).sum())
    top = np.sort(sz)[::-1][:6]
    print(f"[{tag}] ink={100*m.mean():.2f}%  N={n}  <50px={small}  <12px={tiny}  "
          f">5000px={big}  笔宽={stroke_width(m):.2f}px  top6area={top.tolist()}")


print("=== p4 墨迹形态对比 (lum<30) ===")
for tag, p in (("ORIG", SRC), ("NEW ", NEW)):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    rep(tag, lum < 30.0)

if ALPHA.exists():
    al = np.asarray(Image.open(ALPHA).convert("L"), np.float32)
    print()
    rep("ALPH", al > 127)
    print(f"[ALPH] alpha 中间调(20~235)占比 = {100*((al>20)&(al<235)).mean():.2f}%"
          f"   (亚像素/灰度态越多，重采样把细线糊成块的风险越大)")
