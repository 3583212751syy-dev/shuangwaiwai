"""6978 专用：弧字整块矩形掩膜 + 保护圆形徽章本体，单次 LaMa，统一重绘。"""
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
ic = [i for i in cfg['images'] if i['id'] == '6978'][0]
img = Image.open(ic['path']).convert('RGB'); W, H = img.size
arr = np.asarray(img, np.float32)

# 1) 徽章本体：中心带内的「暗且高饱和紫」像素（背景粉底亮、饱和度低，排除掉）
lum = np.asarray(img.convert('L'), np.float32)
r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
band = np.zeros((H, W), bool); band[250:800, 480:1080] = True
emblem = (b > g + 30) & (lum < 165) & band
ys, xs = np.where(emblem)
if len(xs) > 500:
    px1, px2 = np.percentile(xs, 2), np.percentile(xs, 98)
    py1, py2 = np.percentile(ys, 2), np.percentile(ys, 98)
    ecx, ecy = (px1 + px2) / 2, (py1 + py2) / 2
    er = max(px2 - px1, py2 - py1) / 2
else:
    ecx, ecy, er = 780, 500, 210
print(f"emblem center=({ecx:.0f},{ecy:.0f}) r={er:.0f} px={emblem.sum()}")

# 2) 掩膜 = 亮底上的深色笔画（全图文字均为黑色），排除徽章保护圆
yy, xx = np.mgrid[0:H, 0:W]
prot_circle = ((xx - ecx) ** 2 + (yy - ecy) ** 2) <= (220.0) ** 2
region = np.zeros((H, W), bool); region[240:1470, 40:1510] = True
m = (lum < 118) & region & ~prot_circle
m = ndimage.binary_dilation(m, iterations=4)
print("mask frac", round(m.mean(), 3))
Image.fromarray((m * 255).astype(np.uint8), 'L').save(OUT.parent / '_f6978_mask.png')

cleaned = lc.lama_inpaint(img, Image.fromarray((m * 255).astype(np.uint8), 'L'),
                          removal_strength=240, edge_smoothness=5)
cleaned.save(OUT / '6978_rect_clean.jpg', quality=95)

# 3) 重绘新词
def draw(img, it):
    bbox = it['bbox']; word = it['word']; fk = it.get('font', 'blackopsone')
    color = tuple(it.get('color', [0, 0, 0]))
    if it.get('arc') == 'up':
        W, H = img.size; x1, y1, x2, y2 = [int(v) for v in bbox]
        fp = base.FONTS.get(fk, base.FONTS['blackopsone'])
        cx, cy = float(it['arc_center'][0]), float(it['arc_center'][1]); rr = float(it.get('arc_radius', 520))
        fs = int((y2 - y1) * 0.30)
        avail = rr * math.radians((float(it.get('arc_end', 315)) - float(it.get('arc_start', 225))) % 360)
        while fs >= 14:
            if arc_text.fit_arc_text_width(word, str(fp), fs, rr) <= avail * 0.9:
                break
            fs = int(fs * 0.93)
        tmp = Image.new('RGB', (W, H), (0, 0, 0))
        arc_text.draw_arc_text(tmp, word, str(fp), fs, (255, 255, 255), (cx, cy), rr,
                               float(it.get('arc_start', 225)), float(it.get('arc_end', 315)), char_spacing_px=3)
        a = np.asarray(tmp.convert('L'), np.float32) / 255.0
        a3 = a[..., None]
        return Image.fromarray(np.clip(np.asarray(img, np.float32) * (1 - a3) + np.array(color, np.float32) * a3, 0, 255).astype(np.uint8), 'RGB')
    return base._render_word(img, word, bbox, fk, color)

for it in (ic.get('text_plan') or []):
    cleaned = draw(cleaned, it)
cleaned.save(OUT / '6978_rect_final.jpg', quality=92)
print('saved 6978_rect_clean.jpg / 6978_rect_final.jpg')
