"""test_v328_b78.py — 自动测行 + 笔画擦除 + 原位同高重画（b78e60）。"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
from styles import textfix as tf

cfg = json.load(open('regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}
OUT = Path('jobs/_probe'); OUT.mkdir(parents=True, exist_ok=True)

SEARCH = (360, 430, 1200, 790)   # 宽松搜索区（含三行字），不含狗牌
WORDS = ['WE DEFEND THE', 'STEEL', 'HAWKS']

img = Image.open(BY['b78e60']['path']).convert('RGB')
W, H = img.size

lines = tf.detect_lines(img, SEARCH, pad=40)
print(f'detected {len(lines)} lines')
for (b, m) in lines:
    print(f'   box={b}  w={b[2]-b[0]} h={b[3]-b[1]}  mask={int(m.sum())}')

mask = np.zeros((H, W), bool)
for (b, m) in lines:
    mask |= m
vis = np.asarray(img).copy(); vis[mask] = [255, 0, 0]
Image.fromarray(vis).save(OUT / 'b78_mask2_vis.jpg', quality=92)

erased = tf.erase(img, mask, edge_smoothness=4)
erased.save(OUT / 'b78_erased2.jpg', quality=95)

out = erased
for i, (b, m) in enumerate(lines):
    if i >= len(WORDS):
        break
    col = tf.text_color(img, m)
    print(f'   draw {WORDS[i]!r} in {b} color={col}')
    out = tf.draw_line(out, WORDS[i], b, 'blackopsone', col)
out.save(OUT / 'b78_redrawn2.jpg', quality=95)
print('[done]')
