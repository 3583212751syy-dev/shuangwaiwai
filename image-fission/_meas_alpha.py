"""_meas_alpha.py — 只量 jobs/v346_p4aff/_swap_alpha.png 的形态指标（供调参扫描）。"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

p = Path(sys.argv[1] if len(sys.argv) > 1
         else "jobs/v346_p4aff/_swap_alpha.png")
al = np.asarray(Image.open(p).convert("L"), np.float32)
m = al > 127
e = ndi.distance_transform_edt(m)
sk = skeletonize(m)
wid = float(2.0 * e[sk].mean()) if sk.any() else 0.0
lab, n = ndi.label(m, np.ones((3, 3), bool))
sz = np.bincount(lab.ravel())[1:] if n else np.array([0])
print(f"ALPHA ink={100*m.mean():.2f}%  N={n}  <12px={int((sz<12).sum())}  "
      f"笔宽={wid:.2f}px")
