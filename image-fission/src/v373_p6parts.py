# -*- coding: utf-8 -*-
"""p6 part segmentation + background statistics."""
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SRC = r'E:/Desktop/图裂变测试图/Pinterest (6).jpg'
OUT = 'jobs/v372_p6measure'
os.makedirs(OUT, exist_ok=True)

im = Image.open(SRC).convert('RGB')
W, H = im.size
a = np.asarray(im).astype(np.float32)
lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
mx = a.max(2); mn = a.min(2)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1) * 255.0, 0.0)

print('--- background stats (corner 60px boxes) ---')
for name, sl in [('TL', (slice(0, 60), slice(0, 60))), ('TR', (slice(0, 60), slice(-60, None))),
                 ('BL', (slice(-60, None), slice(0, 60))), ('BR', (slice(-60, None), slice(-60, None))),
                 ('mid', (slice(3400, 3460), slice(300, 360)))]:
    v = lum[sl]
    print('  %s lum mean %.1f max %.1f  pxmax %.1f' % (name, v.mean(), v.max(), a[sl].max()))

print('--- global histogram of lum ---')
hist, edges = np.histogram(lum, bins=16, range=(0, 256))
for i, h in enumerate(hist):
    print('  %3d-%3d %8d %.4f' % (edges[i], edges[i + 1], h, h / lum.size))

ink = lum > 40
print('ink(lum>40) frac %.4f' % ink.mean())
ink2 = lum > 26
print('ink(lum>26) frac %.4f' % ink2.mean())

# --- connected components of the subject (lum>40) ---
lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
order = np.argsort(sizes)[::-1]
print('--- top 24 components of ink ---')
objs = ndi.find_objects(lab)
rows = []
for k in order[:24]:
    lid = k + 1
    sl = objs[lid - 1]
    y0, y1 = sl[0].start, sl[0].stop
    x0, x1 = sl[1].start, sl[1].stop
    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
    rows.append((int(sizes[k]), x0, y0, x1, y1, cx, cy))
    print('  id%5d size%9d bbox(%4d,%4d,%4d,%4d) %4dx%4d  c=(%.0f,%.0f)' % (
        lid, sizes[k], x0, y0, x1, y1, x1 - x0, y1 - y0, cx, cy))

np.save(os.path.join(OUT, 'lum.npy'), lum.astype(np.float32))
print('ok')
