"""验证笔画级掩膜一次性擦除 + 统一重绘（替代逐条矩形掩膜）。
只擦字笔画（不碰绶带/装饰），LaMa 上下文充足 -> 预期背景干净、无残影、无色块。"""
import json, sys
from pathlib import Path
from PIL import Image
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from scipy import ndimage
from styles import base
import v268_lama_clean as lc

OUT = Path('jobs/router_out_v326/_exp'); OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open('regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}


def stroke_mask(img_arr, plan, dilate=10, extra_bbox_pad=0):
    H, W = img_arr.shape[:2]
    gm = np.zeros((H, W), bool)
    for it in plan:
        m = base._detect_text_mask(img_arr, it['bbox']) > 127
        gm |= m
        if extra_bbox_pad:
            x1, y1, x2, y2 = [int(v) for v in it['bbox']]
            gm[max(0, y1 - extra_bbox_pad):min(H, y2 + extra_bbox_pad),
               max(0, x1 - extra_bbox_pad):min(W, x2 + extra_bbox_pad)] = True
    gm = ndimage.binary_dilation(gm, iterations=dilate)
    return gm


def draw_plan(img, plan):
    from PIL import ImageDraw
    for it in plan:
        bbox = it['bbox']; word = it['word']; fk = it.get('font', 'blackopsone')
        color = tuple(it.get('color', [0, 0, 0]))
        if it.get('arc') == 'up':
            a = base._render_arc_word_alpha(img.size, word, bbox, fk, arc='up',
                                            arc_center=it.get('arc_center'),
                                            arc_radius=it.get('arc_radius'),
                                            start_angle=float(it.get('arc_start', 225)),
                                            end_angle=float(it.get('arc_end', 315)))
            arr = np.asarray(img, dtype=np.float32)
            a3 = a[..., None]
            arr = arr * (1 - a3) + np.array(color, np.float32) * a3
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGB')
        else:
            img = base._render_word(img, word, bbox, fk, color)
    return img


def run(iid, dilate=10, pad=0, tag=''):
    ic = BY[iid]; plan = ic.get('text_plan') or []
    img = Image.open(ic['path']).convert('RGB'); W, H = img.size
    arr = np.asarray(img, dtype=np.uint8)
    gm = stroke_mask(arr, plan, dilate=dilate, extra_bbox_pad=pad)
    print(f"[{iid}] size={W}x{H} mask_frac={gm.mean():.3f}")
    mp = Image.fromarray((gm * 255).astype(np.uint8), 'L')
    mp.save(OUT.parent / f'_stroke_{iid}_mask.png')
    cleaned = lc.lama_inpaint(img, mp, removal_strength=240, edge_smoothness=4)
    cleaned.save(OUT / f'{iid}{tag}_cleaned.jpg', quality=95)
    final = draw_plan(cleaned, plan)
    final.save(OUT / f'{iid}{tag}_final.jpg', quality=92)
    print("   saved", OUT / f'{iid}{tag}_final.jpg')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('iid'); ap.add_argument('--dilate', type=int, default=10)
    ap.add_argument('--pad', type=int, default=0); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    run(a.iid, dilate=a.dilate, pad=a.pad, tag=a.tag)
