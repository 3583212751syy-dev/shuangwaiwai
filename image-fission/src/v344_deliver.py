"""v344 交付图：四图**完整成品**对比（左原图 / 右裂变完整结果）。

用户第 12 轮要求：「给我反的结果图一定是完完整整的结果图展示，包含主体元素，小元素，
文本裂变」→ 本脚本出的是**整图**（含文字裂变、小元素裂变、主体裂变），不再是主体裁片。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SRC = Path('E:/Desktop/图裂变测试图')
OUT = Path('jobs/router_out_v329')
DST = Path('E:/Desktop/v344_四图完整成品_第13轮交付')
DST.mkdir(parents=True, exist_ok=True)

ITEMS = [
    ('1_pinterest3_小蝴蝶+牛仔字', 'Pinterest (3).jpg', 'pinterest3_variant.jpg'),
    ('2_pinterest4_迷彩棕榈', 'Pinterest (4).jpg', 'Pinterest (4)_variant.jpg'),
    ('3_pinterest6_鹰骷髅+尖刺标题', 'Pinterest (6).jpg', 'pinterest6_variant.jpg'),
    ('4_6978_蝙蝠+弧字', '6978fabda2cc99629fa9e81f802762d3.jpg', '6978_variant.jpg'),
]

TW = 430
try:
    F = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    F2 = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
except Exception:
    F = F2 = ImageFont.load_default()

rows = []
for name, a, b in ITEMS:
    ia = Image.open(SRC / a).convert('RGB')
    ib = Image.open(OUT / b).convert('RGB')
    if ia.size != ib.size:
        ib = ib.resize(ia.size, Image.LANCZOS)
    d = np.abs(np.asarray(ia, np.int16) - np.asarray(ib, np.int16)).mean(2)
    sc = TW / ia.width
    ta = ia.resize((TW, int(ia.height * sc)), Image.LANCZOS)
    tb = ib.resize((TW, int(ib.height * sc)), Image.LANCZOS)
    rows.append((name, ta, tb, float(d.mean()), float(100 * (d > 40).mean())))
    # 单图另存一份全分辨率对比
    z = min(1.0, 1500 / ia.width)
    ca = ia.resize((int(ia.width * z), int(ia.height * z)), Image.LANCZOS)
    cb = ib.resize(ca.size, Image.LANCZOS)
    single = Image.new('RGB', (ca.width * 2 + 16, ca.height), (25, 25, 28))
    single.paste(ca, (0, 0)); single.paste(cb, (ca.width + 16, 0))
    single.save(DST / f'{name}.jpg', quality=92)

LBL, GAP, BAR = 26, 14, 56
W = TW * 2 + GAP * 3
H = BAR + sum(r[1].height + LBL + GAP for r in rows)
sheet = Image.new('RGB', (W, H), (24, 24, 27))
dr = ImageDraw.Draw(sheet)
dr.text((12, 10), 'v344 完整裂变成品（左:原图 | 右:完整结果 = 主体重生 + 小元素 + 文本裂变）',
        fill=(240, 240, 240), font=F)
y = BAR
for name, ta, tb, dm, pc in rows:
    dr.text((GAP, y + 3), f'{name}    meanAbsDiff={dm:.1f}  changed>40={pc:.1f}%',
            fill=(226, 226, 226), font=F2)
    y += LBL
    sheet.paste(ta, (GAP, y)); sheet.paste(tb, (GAP * 2 + TW, y))
    y += ta.height + GAP
p = DST / '00_四图完整成品总表.jpg'
sheet.save(p, quality=90)
print('sheet ->', p, sheet.size)
for name, ta, tb, dm, pc in rows:
    print(f'  {name}: mean={dm:.1f} changed>40={pc:.1f}%')
