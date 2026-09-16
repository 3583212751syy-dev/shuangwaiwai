"""v329 五图对照 gallery + contact sheet 生成器（可重复运行）。

用法: python build_gallery_v329.py
产出: jobs/router_out_v329/gallery_v329.html
      jobs/router_out_v329/v329_contact_sheet.png
数据源: jobs/router_out_v329/*_variant.jpg + _orig/*_orig.jpg
"""
import os
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'jobs', 'router_out_v329')

# (变体文件名, 原图文件名, 标题, 本轮改了什么)
ITEMS = [
    ('b78e60_variant.jpg', 'b78e60_orig.jpg',
     'b78e60 — 军标迷彩：FORGED IN THE DARK STORM',
     '文字原位重画（禁遮挡）· 迷彩湖泊色块几何重塑 · dog tag 角度裂变'),
    ('pinterest3_variant.jpg', 'pinterest3_orig.jpg',
     'pinterest3 — 牛仔贴布：UPCY → DENIM，小元素蝴蝶裂变加大',
     '主体刺绣蝶保持（用户已认可）· 两只小元素蝶各自 ±20% 体型/±15° 翅角/±8° 斜掠'
     '（幅度明显但仍是蝴蝶）· 虚线轨迹原位重画且圆点复刻原色原径'),
    ('Pinterest (4)_variant.jpg', 'pinterest4_orig.jpg',
     'pinterest4 — 迷彩棕榈：湖泊色块圆润衔接',
     '量化 → 连续指示场双线性扭曲 → 标签曲率流圆润化 → 边界抗锯齿（消除 45° 阶梯）'
     '· 叶冠按元素级聚合（91 笔画 → 11 元素）刚体倾斜'),
    ('pinterest6_variant.jpg', 'pinterest6_orig.jpg',
     'pinterest6 — 金属尖刺鹰：主体裂变（用户点赞样例）+ 补文字',
     '主体裂变风格保留不动 · 文字按金属尖刺字体原位替换'),
    ('6978_variant.jpg', '6978_orig.jpg',
     '6978 — BACARDÍ → NOCTAVEN：蝙蝠主体裂变加大',
     '仿射扩张（横向 1.08 / 纵向 1.20，绕蝠内枢轴）——零腾空、零软边 · 右翼额外 1.06 '
     '径向扩张（不对称斜掠）· 抬头 +12px/放大 1.08 · 尾侧摆成弧 · '
     '掩膜净化（剔除缎带残片/文字碎片）· 硬剪影化断掉半透明暗晕'),
]


def build_gallery():
    rows = []
    for var, orig, title, note in ITEMS:
        rows.append(f"""
    <section>
      <h2>{title}</h2>
      <p class="note">{note}</p>
      <div class="pair">
        <figure><img src="_orig/{orig}"><figcaption>原图</figcaption></figure>
        <figure><img src="{var}"><figcaption>变体</figcaption></figure>
      </div>
    </section>""")
    html = f"""<!doctype html><meta charset="utf-8"><title>v338 五图裂变对照</title>
<style>
 body{{background:#111;color:#eee;font:14px/1.6 system-ui,'Microsoft YaHei',sans-serif;margin:0;padding:24px}}
 h1{{font-size:20px;margin:0 0 6px}}
 p.sub{{color:#9aa;margin:0 0 24px}}
 section{{background:#1a1a1e;border:1px solid #2b2b33;border-radius:10px;padding:14px 16px;margin-bottom:20px}}
 h2{{font-size:15px;margin:0 0 6px;color:#ffd27a;font-weight:600}}
 p.note{{color:#9aa;font-size:12px;margin:0 0 12px;line-height:1.7}}
 .pair{{display:flex;gap:14px;flex-wrap:wrap}}
 figure{{margin:0;flex:1 1 320px;max-width:560px}}
 img{{width:100%;border-radius:6px;display:block}}
 figcaption{{color:#8f9;font-size:12px;margin-top:6px}}
</style>
<h1>v338 五图裂变对照</h1>
<p class="sub">第 9 轮反馈：湖泊色块圆润衔接 · 小元素可裂变多一点 · 蝙蝠裂变加大 · 文本不动。
文字一律原位重画（禁遮挡）· 主体物种/位置不变，只改形变维度。</p>
{''.join(rows)}
"""
    p = os.path.join(OUT, 'gallery_v329.html')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(html)
    return p


def build_contact_sheet(cell_w=430, gap=10, label_h=18):
    ims = []
    for var, orig, title, _ in ITEMS:
        for tag, fn in (('原图', orig), ('变体', var)):
            path = os.path.join(OUT, '_orig' if tag == '原图' else '', fn)
            if not os.path.exists(path):
                path = os.path.join(OUT, fn)
            im = Image.open(path).convert('RGB')
            sc = cell_w / im.width
            im = im.resize((cell_w, max(1, int(im.height * sc))), Image.LANCZOS)
            ims.append((title.split(' —')[0] + ' ' + tag, im))
    cols = 2
    rows = (len(ims) + cols - 1) // cols
    ch = max(im.height for _, im in ims)
    sheet = Image.new('RGB', (cols * cell_w + (cols + 1) * gap,
                              rows * (ch + label_h) + (rows + 1) * gap), (250, 250, 252))
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(ims):
        cx = gap + (i % cols) * (cell_w + gap)
        cy = gap + (i // cols) * (ch + label_h + gap)
        d.text((cx + 2, cy + 2), name, fill=(20, 20, 20))
        sheet.paste(im, (cx, cy + label_h))
    p = os.path.join(OUT, 'v329_contact_sheet.png')
    sheet.save(p)
    return p


if __name__ == '__main__':
    print('[gallery]', build_gallery())
    print('[sheet]', build_contact_sheet())
