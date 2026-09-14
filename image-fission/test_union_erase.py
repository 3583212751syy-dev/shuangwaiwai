"""一次性全局掩膜擦除 + 统一重绘。
- 普通文字行：bbox + 小膨胀（矩形，LaMa 填充更均匀）
- 弧形文字：以 arc_center/arc_radius 构造圆环带掩膜（只擦弧带，保留绶带本体）
- 全图单次 LaMa -> 再统一画新词。"""
import json, sys, math
from pathlib import Path
from PIL import Image
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from scipy import ndimage
from styles import base
import v268_lama_clean as lc
import arc_text

OUT = Path('jobs/router_out_v326/_exp'); OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open('regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}


def build_mask(W, H, plan, dilate=8, arc_half=95, arc_a0=25, arc_a1=155, lum=None, dark_thr=110):
    m = np.zeros((H, W), bool)
    for it in plan:
        x1, y1, x2, y2 = [int(v) for v in it['bbox']]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
        if it.get('mode') == 'dark' and lum is not None:
            # 亮底深色笔画检测：只取 bbox 内明显比背景暗的像素（弧字/年份字在浅色绶带上）
            roi = lum[y1:y2, x1:x2]
            m[y1:y2, x1:x2] |= roi < dark_thr
        elif it.get('arc') == 'up' and it.get('arc_center'):
            roi = lum[y1:y2, x1:x2]
            thr = min(dark_thr, np.percentile(roi, 22))
            m[y1:y2, x1:x2] |= roi < thr
        else:
            m[max(0, y1 - dilate):min(H, y2 + dilate),
              max(0, x1 - dilate):min(W, x2 + dilate)] = True
    return m


def draw_plan(img, plan):
    for it in plan:
        bbox = it['bbox']; word = it['word']; fk = it.get('font', 'blackopsone')
        color = tuple(it.get('color', [0, 0, 0]))
        if it.get('arc') == 'up':
            W, H = img.size
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fp = base.FONTS.get(fk, base.FONTS['blackopsone'])
            cx, cy = float(it['arc_center'][0]), float(it['arc_center'][1])
            r = float(it.get('arc_radius', 520))
            bh = y2 - y1
            fs = int(bh * 0.32)
            avail = r * math.radians((float(it.get('arc_end', 315)) - float(it.get('arc_start', 225))) % 360)
            while fs >= 14:
                if arc_text.fit_arc_text_width(word, str(fp), fs, r) <= avail * 0.9:
                    break
                fs = int(fs * 0.93)
            tmp = Image.new('RGB', (W, H), (0, 0, 0))
            arc_text.draw_arc_text(tmp, word, str(fp), fs, (255, 255, 255),
                                   (cx, cy), r, float(it.get('arc_start', 225)),
                                   float(it.get('arc_end', 315)), char_spacing_px=3)
            a = np.asarray(tmp.convert('L'), np.float32) / 255.0
            arr = np.asarray(img, np.float32)
            a3 = a[..., None]
            img = Image.fromarray(np.clip(arr * (1 - a3) + np.array(color, np.float32) * a3, 0, 255).astype(np.uint8), 'RGB')
        else:
            img = base._render_word(img, word, bbox, fk, color)
    return img


def run(iid, dilate=8, arc_half=95, tag=''):
    ic = BY[iid]; plan = ic.get('text_plan') or []
    img = Image.open(ic['path']).convert('RGB'); W, H = img.size
    lum = np.asarray(img.convert('L'), np.float32)
    m = build_mask(W, H, plan, dilate=dilate, arc_half=arc_half, lum=lum)
    print(f"[{iid}] {W}x{H} mask_frac={m.mean():.3f}")
    Image.fromarray((m * 255).astype(np.uint8), 'L').save(OUT.parent / f'_union_{iid}_mask.png')
    cleaned = lc.lama_inpaint(img, Image.fromarray((m * 255).astype(np.uint8), 'L'),
                              removal_strength=240, edge_smoothness=5)
    cleaned.save(OUT / f'{iid}{tag}_clean.jpg', quality=95)
    draw_plan(cleaned, plan).save(OUT / f'{iid}{tag}_final2.jpg', quality=92)
    print("   saved", OUT / f'{iid}{tag}_final2.jpg')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('iid'); ap.add_argument('--dilate', type=int, default=8)
    ap.add_argument('--arc_half', type=int, default=95); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    run(a.iid, dilate=a.dilate, arc_half=a.arc_half, tag=a.tag)
