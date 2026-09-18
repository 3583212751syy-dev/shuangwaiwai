"""grid_overlay.py — 在原图上打 100px 坐标网格，用于人工精确标定文字位置。"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path('.')
OUT = ROOT / 'jobs' / '_probe'; OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / 'regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}
FONT = str(ROOT / 'fonts' / 'BlackOpsOne-Regular.ttf')

for iid in sys.argv[1:]:
    img = Image.open(BY[iid]['path']).convert('RGB')
    W, H = img.size
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, max(26, W // 42))
    step = 100 if W < 2000 else 200
    for x in range(0, W, step):
        major = (x % (step * 5) == 0)
        d.line([(x, 0), (x, H)], fill=(255, 0, 0) if major else (0, 200, 0), width=3 if major else 1)
        if major:
            d.text((x + 4, 4), str(x), font=f, fill=(255, 255, 0))
    for y in range(0, H, step):
        major = (y % (step * 5) == 0)
        d.line([(0, y), (W, y)], fill=(255, 0, 0) if major else (0, 200, 0), width=3 if major else 1)
        if major:
            d.text((4, y + 4), str(y), font=f, fill=(255, 255, 0))
    p = OUT / f'{iid}_grid.png'
    img.save(p)
    print('saved', p, img.size, 'step', step)
