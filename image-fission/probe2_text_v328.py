"""probe2_text_v328.py — 形态学黑帽/白帽精确定位文字行（对任意笔画粗细都实心）。

黑帽 = grey_closing(lum, S) - lum : 检测「比周围暗且比 S 细」的实心目标（暗字）
白帽 = lum - grey_opening(lum, S) : 检测「比周围亮且比 S 细」的实心目标（亮字）
S 取 ~1.6x 最大笔画粗细，保证字母内部也被填实（旧中值法对粗笔画会空心）。
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path('.')
OUT = ROOT / 'jobs' / '_probe'; OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / 'regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}

REGIONS = {
    'b78e60':     (340, 400, 1260, 800),
    '6978':       (60, 120, 1500, 1500),
    'pinterest3': (0, 30, 736, 500),
    'pinterest6': (60, 40, 3480, 1000),
}


def caps(img_arr, size=161, contrast=55):
    lum = 0.299 * img_arr[..., 0] + 0.587 * img_arr[..., 1] + 0.114 * img_arr[..., 2]
    dark = ndi.grey_closing(lum, size=size) - lum
    light = lum - ndi.grey_opening(lum, size=size)
    return dark > contrast, light > contrast


def solid(m, min_area=260, fill=True):
    if not m.any():
        return m
    if fill:
        m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:] = sizes >= min_area
        m = keep[lab]
    return m


def bands(mask, min_px=4, gap=14):
    rows = mask.sum(axis=1)
    out, cur = [], None
    for y, v in enumerate(rows):
        if v >= min_px:
            cur = [y, y] if cur is None else [cur[0], y]
        elif cur is not None and y - cur[1] > gap:
            out.append(cur); cur = None
    if cur is not None:
        out.append(cur)
    res = []
    for y1, y2 in out:
        sub = mask[y1:y2 + 1]
        cols = np.where(sub.any(axis=0))[0]
        if len(cols):
            res.append({'y1': int(y1), 'y2': int(y2), 'x1': int(cols.min()), 'x2': int(cols.max()),
                        'px': int(sub.sum())})
    return res


for iid, (rx1, ry1, rx2, ry2) in REGIONS.items():
    img = Image.open(BY[iid]['path']).convert('RGB')
    W, H = img.size
    arr = np.asarray(img.crop((rx1, ry1, rx2, ry2)), np.float32)
    d, l = caps(arr)
    d, l = solid(d), solid(l)
    print(f"\n===== {iid} {W}x{H} region=({rx1},{ry1},{rx2},{ry2})")
    for tag, m, col in (('DARK', d, [255, 0, 0]), ('LIGHT', l, [0, 150, 255])):
        bs = bands(m)
        print(f"  [{tag}] bands={len(bs)}")
        for b in sorted(bs, key=lambda z: -z['px'])[:10]:
            print(f"     y {b['y1']+ry1:5d}..{b['y2']+ry1:5d} (h={b['y2']-b['y1']+1:4d})  "
                  f"x {b['x1']+rx1:5d}..{b['x2']+rx1:5d} (w={b['x2']-b['x1']+1:4d})  px={b['px']}")
    vis = arr.copy()
    vis[d] = [255, 0, 0]; vis[l] = [0, 150, 255]
    Image.fromarray(vis.astype(np.uint8)).save(OUT / f'{iid}_probe2.png')
print('\n[probe2] done')
