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
     'pinterest3 — 牛仔贴布：UPCY → DENIM，小元素蝶放大 + 迷你蝶队列',
     '第 10 轮：小元素蝶改「刚体旋转 + 等比放大（×1.22 / ×1.14）」，刺绣针脚零扭曲；'
     '另按主体蝶形态新增 6 只「近大远小」飞行队列迷你蝶（避开文字带与主体紧框）'),
    ('Pinterest (4)_variant.jpg', 'pinterest4_orig.jpg',
     'pinterest4 — 迷彩棕榈：前景树按阵风弯曲（湖泊已认可，未动）',
     '第 10 轮：新增全局平滑风场 tree_wind（位移只依赖坐标 → 相邻树弯向不同、干冠同步），'
     '实测 |dx|max=161px · 湖泊色块保持上一轮圆润结果不动'),
    ('pinterest6_variant.jpg', 'pinterest6_orig.jpg',
     'pinterest6 — 金属尖刺鹰：MRCHOSR → VOIDFALLEN + 主体裂变',
     '第 10 轮：① 新文本按**原图标题排版框**定标（原图白墨 x[12,3524] y[88,1800]，'
     '旧版只有全宽 70% 且顶到 y=60）→ 改「横向满幅 3513 + ink 顶对齐 88」，两 bbox 重合；'
     '② 主体裂变加大：双翼绕肩上扬 0.30/0.22rad（不等=斜掠）+ 双角外掀 0.20rad'
     '（旧版双角完全没动）+ 骷髅倾斜 0.13rad + 下颌下沉 85px；'
     '③ 每部件加径向斜坡权重，位移在部件与躯干交界处 → 0，杜绝交界处拉伸黑斑'),
    ('6978_variant.jpg', '6978_orig.jpg',
     '6978 — BACARDÍ → NOCTAVEN：蝙蝠主体裂变加大',
     '仿射扩张（横向 1.08 / 纵向 1.20，绕蝠内枢轴）——零腾空、零软边 · '
     '双翼绕肩换姿（不等角 L0.20/R0.32 = 斜掠）· 抬头 +12px/放大 1.08 · '
     '尾侧摆成弧 · 文字原位改写'),
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
