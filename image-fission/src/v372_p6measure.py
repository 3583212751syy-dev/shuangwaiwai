# -*- coding: utf-8 -*-
"""p6  landmark measurement on the ORIGINAL Pinterest (6).jpg
Locate: title band bbox, eagle bbox, skull bbox, horn roots, wing envelope.
"""
import os
import numpy as np
from PIL import Image

SRC = r'E:/Desktop/图裂变测试图/Pinterest (6).jpg'
OUT = 'jobs/v372_p6measure'
os.makedirs(OUT, exist_ok=True)

im = Image.open(SRC).convert('RGB')
W, H = im.size
a = np.asarray(im).astype(np.int16)
print('size', W, H)

r, g, b = a[..., 0], a[..., 1], a[..., 2]
lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
mx = a.max(2); mn = a.min(2)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1) * 255.0, 0.0)

# ink (any non-black)
ink = lum > 38
print('ink frac %.4f' % ink.mean())

# ---- row profile: find the black gap between title and subject ----
rowf = ink.mean(1)
# print coarse profile
for y in range(0, H, 60):
    seg = rowf[y:y + 60].mean()
    bar = '#' * int(seg * 200)
    print('y%5d  %.3f %s' % (y, seg, bar))

# ---- colour classes ----
white = (lum > 150) & (sat < 60)
brown = (lum > 55) & (sat > 70) & (r > g + 12) & (g > b + 6)
print('white frac %.4f brown frac %.4f' % (white.mean(), brown.mean()))

def bbox(m, thr=0.002):
    ys = np.where(m.mean(1) > thr)[0]
    xs = np.where(m.mean(0) > thr)[0]
    if len(ys) == 0 or len(xs) == 0:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])

print('ink bbox', bbox(ink))
print('white bbox', bbox(white))
print('brown bbox', bbox(brown))

# ---- brown (eagle + horns) column split ----
bc = brown.sum(0)
mid = W // 2
print('brown col profile every 150px:')
for x in range(0, W, 150):
    print('  x%5d %6d' % (x, bc[x:x + 150].sum()))

# ---- brown rows ----
br = brown.sum(1)
print('brown row profile every 100px:')
for y in range(0, H, 100):
    v = br[y:y + 100].sum()
    if v > 3000:
        print('  y%5d %8d %s' % (y, v, '#' * min(int(v / 120000 * 120), 120)))

# ---- white skull: rows in lower half ----
print('white row profile (lower half) every 100px:')
wr = white.sum(1)
for y in range(H // 2, H, 100):
    v = wr[y:y + 100].sum()
    if v > 3000:
        print('  y%5d %8d %s' % (y, v, '#' * min(int(v / 250000 * 120), 120)))

# ---- save masks for eyeball check ----
def save(m, name):
    Image.fromarray((m * 255).astype(np.uint8)).resize((W // 4, H // 4)).save(
        os.path.join(OUT, name))
save(ink, 'm_ink.png')
save(white, 'm_white.png')
save(brown, 'm_brown.png')
print('saved to', OUT)
