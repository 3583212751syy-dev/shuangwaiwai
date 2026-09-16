"""make_v329_gallery.py — 生成 v329 五图对照画廊（原图 vs 变体）与 PNG 拼图。"""
import json
from pathlib import Path
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from PIL import Image, ImageDraw

ROOT = Path('.')
OUT = ROOT / 'jobs' / 'router_out_v329'
cfg = json.load(open(ROOT / 'regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}

PAIRS = [
    ('b78e60',    'b78e60_variant.jpg',           '军标迷彩文字：ARMED FORCES → STEEL HAWKS'),
    ('pinterest3', 'pinterest3_variant.jpg',      '牛仔贴布：UPCY → DENIM，主体蝴蝶换新设计'),
    ('pinterest4', 'Pinterest (4)_variant.jpg',   '迷彩棕榈：色块重塑 + 逐棵不同角度重画'),
    ('pinterest6', 'pinterest6_variant.jpg',      '黑金属尖刺标题：MRCHOSR → VORGRAVEN'),
    ('6978',      '6978_variant.jpg',             '蝙蝠徽章：BACARDÍ MOHEART → NOCTAVEN MOONHEART'),
]

rows_html = []
tiles = []
ORIG = OUT / '_orig'
ORIG.mkdir(exist_ok=True)
for iid, fn, note in PAIRS:
    op = BY[iid]['path']
    vp = OUT / fn
    if not Path(vp).exists():
        continue
    o, v = Image.open(op).convert('RGB'), Image.open(vp).convert('RGB')
    oname = f'{iid}_orig.jpg'
    if not (ORIG / oname).exists():
        o.save(ORIG / oname, quality=88)
    rows_html.append(f"""
    <section>
      <h2>{iid} — {note}</h2>
      <div class="pair">
        <figure><img src="_orig/{oname}"><figcaption>原图</figcaption></figure>
        <figure><img src="{fn}"><figcaption>变体 v329</figcaption></figure>
      </div>
    </section>""")
    for im, tag in ((o, 'ORIG'), (v, 'V329')):
        tiles.append((tag, iid, im))

TW = 380
scaled = [(t, i, im.resize((TW, int(im.height * TW / im.width)), Image.LANCZOS)) for t, i, im in tiles]
Hmax = max(s.height for _, _, s in scaled)
sheet = Image.new('RGB', (TW * len(scaled), Hmax + 30), (18, 18, 20))
d = ImageDraw.Draw(sheet)
for k, (tag, iid, s) in enumerate(scaled):
    sheet.paste(s, (k * TW, 30))
    d.text((k * TW + 6, 10), f'{iid} {tag}', fill=(255, 210, 120) if tag == 'ORIG' else (130, 255, 180))
sheet.save(OUT / 'v329_contact_sheet.png')

html = """<!doctype html><meta charset="utf-8"><title>v329 五图裂变对照</title>
<style>
 body{background:#111;color:#eee;font:14px/1.6 system-ui,'Microsoft YaHei',sans-serif;margin:0;padding:24px}
 h1{font-size:20px;margin:0 0 6px}
 p.sub{color:#9aa;margin:0 0 24px}
 section{background:#1a1a1e;border:1px solid #2b2b33;border-radius:10px;padding:14px 16px;margin-bottom:20px}
 h2{font-size:15px;margin:0 0 12px;color:#ffd27a;font-weight:600}
 .pair{display:flex;gap:14px;flex-wrap:wrap}
 figure{margin:0;flex:1 1 320px;max-width:520px}
 img{width:100%;border-radius:6px;display:block}
 figcaption{color:#8f9;font-size:12px;margin-top:6px}
</style>
<h1>v329 五图裂变对照</h1>
<p class="sub">2026-09-15 按用户反馈重做：文字一律原位重画（禁遮挡覆盖）· pinterest4 色块重塑+逐棵不同树 · p3/6978 主体与文字双裂变</p>
""" + "\n".join(rows_html) + f"""
<section><h2>接触印相（原图/变体）</h2>
<img src="v329_contact_sheet.png"></section>
"""
(OUT / 'gallery_v329.html').write_text(html, encoding='utf-8')
print('[gallery]', OUT / 'gallery_v329.html', 'tiles', len(scaled))
