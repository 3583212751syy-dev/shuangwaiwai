# -*- coding: utf-8 -*-
"""v306 recon: zoom source anatomy (ring / wings / membrane / legs / head) before pipeline design."""
import cv2
import numpy as np
import os
from PIL import Image

SRC = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
OUT = r"E:\Desktop\双接口\image-fission\jobs\v306\_recon"
os.makedirs(OUT, exist_ok=True)

# cv2.imread/imwrite 在中文路径下失效（v304 教训），统一 PIL 读写
img = np.array(Image.open(SRC).convert("RGB"))[:, :, ::-1].copy()  # RGB->BGR
H, W = img.shape[:2]
print("image size:", W, "x", H)
maxc = img.max(axis=2).astype(np.int32)

# ---- locate badge ring (outermost dark structure in upper 2/3) ----
dark = ((maxc < 60)).astype(np.uint8)
dark_upper = dark.copy()
dark_upper[int(H * 0.60):, :] = 0  # exclude big text zone
ys, xs = np.nonzero(dark_upper)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
bcx = (int(x0) + int(x1)) // 2
bcy = int(y0) + 380  # ring top + approx R
print("dark-upper bbox x[%d,%d] y[%d,%d] -> est badge center (%d,%d)" % (x0, x1, y0, y1, bcx, bcy))

col = dark_upper[:, bcx]
rows = np.nonzero(col)[0]
if len(rows):
    print("center column dark rows: %d .. %d (ring top/bottom)" % (rows.min(), rows.max()))

def crop(name, x0c, y0c, x1c, y1c, zoom):
    x0c = max(0, x0c); y0c = max(0, y0c)
    x1c = min(W, x1c); y1c = min(H, y1c)
    c = img[y0c:y1c, x0c:x1c]
    c = cv2.resize(c, (int(c.shape[1] * zoom), int(c.shape[0] * zoom)), interpolation=cv2.INTER_CUBIC)
    p = os.path.join(OUT, name + ".png")
    Image.fromarray(c[:, :, ::-1]).save(p)  # BGR->RGB
    print("saved", name, c.shape[1], "x", c.shape[0])

crop("badge_full", bcx - 480, bcy - 480, bcx + 480, bcy + 480, 1.4)
crop("wing_L", bcx - 460, bcy - 160, bcx - 20, bcy + 340, 2.0)
crop("wing_R", bcx + 20, bcy - 160, bcx + 460, bcy + 340, 2.0)
crop("head", bcx - 150, bcy - 200, bcx + 150, bcy + 40, 3.0)
crop("bottom_legs", bcx - 240, bcy + 140, bcx + 240, bcy + 480, 3.0)
crop("ring_top", bcx - 260, bcy - 430, bcx + 260, bcy - 290, 2.6)
crop("ring_left", bcx - 430, bcy - 80, bcx - 280, bcy + 80, 4.0)
crop("ring_bottom", bcx - 220, bcy + 300, bcx + 220, bcy + 430, 2.6)

def stat(name, y, x, r=6):
    patch = img[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3)
    m = patch.mean(axis=0)
    print("%-16s BGR mean=(%5.0f,%5.0f,%5.0f) maxc~%3.0f" % (name, m[0], m[1], m[2], m.max()))

stat("bg_below_badge", bcy + 540, bcx)
stat("bg_left_of_badge", bcy, bcx - 560)
stat("ring_left_edge", bcy, bcx - 360, 3)
stat("ring_right_edge", bcy, bcx + 360, 3)

wx0, wy0, wx1, wy1 = bcx - 440, bcy - 140, bcx - 60, bcy + 300
wm = maxc[wy0:wy1, wx0:wx1]
hist_lo = int(((wm >= 90) & (wm < 160)).sum())
hist_mid = int(((wm >= 160) & (wm < 220)).sum())
print("left-wing box maxc: 90-160 px=%d, 160-220 px=%d, <90 px=%d" % (hist_lo, hist_mid, int((wm < 90).sum())))
mem_mask = (wm >= 90) & (wm < 160)
if mem_mask.any():
    mv = img[wy0:wy1, wx0:wx1][mem_mask].mean(axis=0)
    print("membrane(90-160) BGR mean=(%.0f,%.0f,%.0f)" % (mv[0], mv[1], mv[2]))

bb = (maxc < 60)
print("black body px count:", int(bb.sum()), " BGR mean=", img[bb].mean(axis=0).round(0))

bz = dark[bcy + 150:bcy + 470, bcx - 240:bcx + 240]
nz = int(bz.sum())
print("bottom-zone dark px:", nz)
colsum = bz.sum(axis=0)
peaks = np.where(colsum > 3)[0]
if len(peaks):
    print("bottom dark x-ranges (rel):", peaks.min(), "..", peaks.max())
    gaps = np.where(np.diff(peaks) > 5)[0]
    segs = []
    start = peaks[0]
    for g in gaps:
        segs.append((int(start), int(peaks[g])))
        start = peaks[g + 1]
    segs.append((int(start), int(peaks[-1])))
    print("clusters:", segs)
print("done")
