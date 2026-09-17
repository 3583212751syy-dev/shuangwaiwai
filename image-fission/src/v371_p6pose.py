# -*- coding: utf-8 -*-
"""v371  p6 主体结构裂变  (三层法 + 单位分解仿射场)

分层（严格照原图 z 序）
    base(黑) -> ①射线层(刚性绕颅心旋转) -> ②主体层(单位分解多部件仿射) -> ③标题层(原样)
关键点
  * 原图背景 = 纯黑 => 位移可作用于整幅画，腾空区自动变黑；无需 inpaint、无接缝。
  * **细白闪电射线单独成层刚体旋转**：它们细长、跨多部件权重区，若混进主体层会被
    单位分解边界"剪断/撕碎"（上一版伪影的根源）。独立成层后绝对平整。
  * 主体层权重 = 单位分解(Σw 归一化) => 相邻部件不再互抢像素，根除 pinch。
  * y<1120 冻结（标题带保护），1120~1560 平滑放行。
"""
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def smoothstep(x, lo=0.0, hi=1.0):
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


SRC = r'E:/Desktop/图裂变测试图/Pinterest (6).jpg'
OUT = 'jobs/v371_p6pose'
os.makedirs(OUT, exist_ok=True)

im = Image.open(SRC).convert('RGB')
W, H = im.size
a = np.asarray(im).astype(np.float32)
r, g, b = a[..., 0], a[..., 1], a[..., 2]
lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
mx = a.max(2); mn = a.min(2)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1) * 255.0, 0.0)
print('size', W, H)
gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
STRUCT8 = ndi.generate_binary_structure(2, 2)

# ================= 1. 白色连通域分类 =================
white = (lum > 142) & (sat < 66)
lab_w, n_w = ndi.label(white, STRUCT8)
sz = ndi.sum(np.ones_like(lab_w), lab_w, range(1, n_w + 1))
objs = ndi.find_objects(lab_w)
EAGLE_BOX = (900, 560, 3130, 2340)     # x0,y0,x1,y1

title_ids, skull_ids, bg_ids = [], [], []
for i, sl in enumerate(objs):
    if sl is None:
        continue
    lid = i + 1
    s = sz[i]
    y0, y1 = sl[0].start, sl[0].stop
    x0, x1 = sl[1].start, sl[1].stop
    if y0 < 600 and (y1 - y0) > 120:
        title_ids.append(lid); continue
    if s > 300000 and y0 > 2000:
        skull_ids.append(lid); continue
    in_eagle = (x0 >= EAGLE_BOX[0] and y0 >= EAGLE_BOX[1]
                and x1 <= EAGLE_BOX[2] and y1 <= EAGLE_BOX[3])
    if s >= 2000 and not in_eagle:
        bg_ids.append(lid)

title_m = ndi.binary_dilation(np.isin(lab_w, title_ids), STRUCT8, 5)
skull_m = np.isin(lab_w, skull_ids)
ray_m = ndi.binary_dilation(np.isin(lab_w, bg_ids), STRUCT8, 3)
ray_m &= ~title_m
print('title %.4f  skull %.4f  ray %.4f  (title %d, skull %d, ray %d comps)'
      % (title_m.mean(), skull_m.mean(), ray_m.mean(),
         len(title_ids), len(skull_ids), len(bg_ids)))

# ================= 2. 主体层 =================
subj = (lum > 34) & ~title_m & ~ray_m
subj = ndi.binary_opening(subj, np.ones((2, 2), bool))
lab_s, n_s = ndi.label(subj, STRUCT8)
szs = ndi.sum(np.ones_like(lab_s), lab_s, range(1, n_s + 1))
subj = np.isin(lab_s, np.where(szs >= 120)[0] + 1)
print('subject %.4f' % subj.mean())


# ================= 3. 位移场 =================
def sample(arr, Dy, Dx):
    coords = np.stack([gy - Dy, gx - Dx])
    if arr.ndim == 2:
        return ndi.map_coordinates(arr, coords, order=1, mode='nearest')
    return np.stack([ndi.map_coordinates(arr[..., c], coords, order=1, mode='nearest')
                     for c in range(arr.shape[2])], -1)


def plate(cx, cy, rx, ry, fade=0.55):
    q = np.sqrt(((gx - cx) / rx) ** 2 + ((gy - cy) / ry) ** 2)
    return 1.0 - smoothstep(q, 1.0 - fade, 1.0)


#      name   plate                                     pivot            theta     scale
PARTS = [
    ('wl', plate(800, 1400, 950, 1230, 0.55), (1330, 1800), -0.130, 1.000),
    ('wr', plate(2640, 1400, 950, 1230, 0.55), (2215, 1800), +0.105, 1.000),
    ('bd', plate(1771, 1890, 440, 520, 0.60), (1771, 2060), -0.030, 1.000),
    ('hd', plate(1800, 1740, 320, 280, 0.50), (1800, 1790), +0.175, 1.015),
    ('hl', plate(730, 2520, 700, 700, 0.55), (1250, 2860), +0.100, 1.170),
    ('hr', plate(2590, 2520, 700, 700, 0.55), (2290, 2860), -0.100, 1.170),
    ('sk', plate(1830, 3150, 620, 760, 0.55), (1830, 3200), +0.048, 1.045),
]
ws = [p[1] for p in PARTS]
tot = np.maximum(np.sum(ws, 0), 1.0e-6)
ws = [w / np.maximum(tot, 1.0) for w in ws]

Dy = np.zeros((H, W), np.float32); Dx = np.zeros((H, W), np.float32)
for (name, _w, (px, py), th, sc), w in zip(PARTS, ws):
    dx = gx - px; dy = gy - py
    c = np.cos(th); s = np.sin(th)
    nx = c * dx - s * dy
    ny = s * dx + c * dy
    Dy += w * (ny * sc - dy)
    Dx += w * (nx * sc - dx)
gm = smoothstep(gy, 1120.0, 1560.0).astype(np.float32)
Dy *= gm; Dx *= gm
print('subject disp max %.1f  p99 %.1f  mean %.1f'
      % (np.hypot(Dy, Dx).max(), np.percentile(np.hypot(Dy, Dx), 99), np.hypot(Dy, Dx).mean()))

# 射线层：绕颅心刚体旋转 + 微放
RY = 0.125
RS = 1.055
SC = (1830.0, 2990.0)
rc = np.cos(RY); rs = np.sin(RY)
rx0 = (gx - SC[0]) * RS
ry0 = (gy - SC[1]) * RS
ryx = SC[0] + (rc * rx0 - rs * ry0)
ryy = SC[1] + (rs * rx0 + rc * ry0)
DrY = (ryy - gy).astype(np.float32); DrX = (ryx - gx).astype(np.float32)
print('ray disp max %.1f' % np.hypot(DrY, DrX).max())

# ================= 4. 合成 =================
base = a.copy()
base[subj] = 0.0
base[ray_m] = 0.0

ray_w = sample(a, DrY, DrX)
ray_al = np.clip(sample(ray_m.astype(np.float32), DrY, DrX), 0, 1)
out = ray_w * ray_al[..., None] + base * (1.0 - ray_al[..., None])

sub_w = sample(a, Dy, Dx)
sub_al = np.clip(sample(subj.astype(np.float32), Dy, Dx), 0, 1)
out = sub_w * sub_al[..., None] + out * (1.0 - sub_al[..., None])

tal = ndi.gaussian_filter(title_m.astype(np.float32), 1.0)
out = a * tal[..., None] + out * (1.0 - tal[..., None])

out = np.clip(out, 0, 255).astype(np.uint8)
Image.fromarray(out).save(os.path.join(OUT, 'p6_pose.jpg'), quality=96)
print('saved', os.path.join(OUT, 'p6_pose.jpg'))
