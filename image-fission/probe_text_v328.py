"""probe_text_v328.py — 探测 5 张原图里文字的真实位置/行高/极值，用于重建 text_plan。

核心：**局部对比度笔画检测**（不是固定亮度阈值）。
  - 文字笔画 = 显著暗于(或亮于)自身周围中位亮度的像素
  - 迷彩大色块内部亮度均匀 -> 局部对比度≈0 -> 不会被误当成文字
这解决了旧管线「bbox 矩形掩膜 --> LaMa 填平成遮挡块」和「bbox 坐标本身就错位」两个根因。
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path('.')
OUT = ROOT / 'jobs' / '_probe'; OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / 'regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}

# 每图给一个宽松搜索区（原图像素坐标），只用来限定范围，不当作文字 bbox
REGIONS = {
    'b78e60':     (300, 380, 1300, 780),
    '6978':       (60, 120, 1520, 1520),
    'pinterest3': (0, 30, 736, 420),
    'pinterest6': (60, 40, 3480, 900),
    'pinterest4': (0, 0, 1242, 1754),
}


def glyph_strokes(arr, window=61, contrast=25, min_thick=4, min_area=250):
    """局部对比度笔画检测 + 粗细过滤，返回 (暗笔画mask, 亮笔画mask)。

    粗细过滤：腐蚀 min_thick 像素，1~3px 的迷彩色块分界线会消失，只有真正的粗笔画能留下。
    """
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    med = ndi.median_filter(lum, size=window)
    st = np.ones((min_thick, min_thick), bool)

    def clean(m):
        core = ndi.binary_erosion(m, structure=st)
        if not core.any():
            return np.zeros_like(m)
        m = ndi.binary_dilation(core, structure=st)
        lab, n = ndi.label(m)
        if n:
            sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sizes >= min_area
            m = keep[lab]
        return m

    dark = clean((med - lum) > contrast)
    light = clean((lum - med) > contrast)
    return dark, light


def bands(mask, min_row_px=6, gap=6):
    rows = mask.sum(axis=1)
    out, cur = [], None
    for y, v in enumerate(rows):
        if v >= min_row_px:
            if cur is None:
                cur = [y, y]
            else:
                cur[1] = y
        else:
            if cur is not None:
                if y - cur[1] > gap:
                    out.append(cur); cur = None
    if cur is not None:
        out.append(cur)
    res = []
    for y1, y2 in out:
        sub = mask[y1:y2 + 1]
        cols = np.where(sub.any(axis=0))[0]
        if len(cols) == 0:
            continue
        res.append({'y1': int(y1), 'y2': int(y2), 'x1': int(cols.min()), 'x2': int(cols.max()),
                    'h': int(y2 - y1 + 1), 'w': int(cols.max() - cols.min() + 1),
                    'px': int(sub.sum())})
    return res


for iid, reg in REGIONS.items():
    ic = BY[iid]
    img = Image.open(ic['path']).convert('RGB')
    W, H = img.size
    x1, y1, x2, y2 = reg
    crop = np.asarray(img.crop((x1, y1, x2, y2)), np.float32)
    dark, light = glyph_strokes(crop)
    print(f"\n===== {iid}  {W}x{H}  region={reg}")
    for name, m in (('DARK-strokes', dark), ('LIGHT-strokes', light)):
        bs = bands(m)
        tot = m.sum()
        print(f"  [{name}] px={int(tot)} ({tot/m.size*100:.1f}%)  bands={len(bs)}")
        for b in sorted(bs, key=lambda z: -z['px'])[:8]:
            print(f"     y {b['y1']+y1:5d}..{b['y2']+y1:5d} (h={b['h']:4d})  "
                  f"x {b['x1']+x1:5d}..{b['x2']+x1:5d} (w={b['w']:4d})  px={b['px']}")
    # 可视化
    vis = crop.copy()
    vis[dark] = [255, 0, 0]
    vis[light] = [0, 160, 255]
    Image.fromarray(vis.astype(np.uint8)).save(OUT / f'{iid}_probe.png')
print('\n[probe] done ->', OUT)
