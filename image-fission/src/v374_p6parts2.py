# -*- coding: utf-8 -*-
"""p6: isolate brown (wings/horns) and white (skull/head) parts."""
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SRC = r'E:/Desktop/图裂变测试图/Pinterest (6).jpg'
OUT = 'jobs/v372_p6measure'
im = Image.open(SRC).convert('RGB')
W, H = im.size
a = np.asarray(im).astype(np.float32)
r, g, b = a[..., 0], a[..., 1], a[..., 2]
lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
mx = a.max(2); mn = a.min(2)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1) * 255.0, 0.0)

brown = (lum > 50) & (sat > 65) & (r > g + 10) & (g > b + 4)
white = (lum > 145) & (sat < 62)

print('brown frac %.4f  white frac %.4f' % (brown.mean(), white.mean()))


def comps(m, name, topn=18, min_size=4000):
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    objs = ndi.find_objects(lab)
    print('=== %s : %d comps' % (name, n))
    out = []
    for k in order[:topn]:
        if sizes[k] < min_size:
            break
        sl = objs[k]
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        cy, cx = ndi.center_of_mass(m, lab, k + 1)
        out.append((int(sizes[k]), x0, y0, x1, y1, cx, cy))
        print('  size%9d bbox(%5d,%5d,%5d,%5d) %5dx%5d  com=(%.0f,%.0f)' % (
            sizes[k], x0, y0, x1, y1, x1 - x0, y1 - y0, cx, cy))
    return out


bc = comps(brown, 'brown')
wc = comps(white, 'white')

# ---- restrict to lower band to find horns specifically ----
horn_zone = brown.copy()
horn_zone[:2000] = False
comps(horn_zone, 'brown y>2000 (horns/body/talons)', 14)

# ---- skull: white comps in y 2400..3500 ----
sk = white.copy()
sk[:2400] = False
sk[3600:] = False
comps(sk, 'white 2400<y<3600 (skull)', 10)
