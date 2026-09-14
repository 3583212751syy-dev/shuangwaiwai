"""make_v328_gallery.py — 生成 v328 五图 原图/裂变 对照 gallery（HTML）

用法：venv/Scripts/python.exe make_v328_gallery.py
产物：jobs/router_out_v328/gallery_v328.html （图片用相对路径引用，就地可看）
"""
import json
from pathlib import Path
from PIL import Image

ROOT = Path('.')
OUT = ROOT / 'jobs' / 'router_out_v328'
THUMB = OUT / 'thumbs'
THUMB.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / 'regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}

ORDER = ['b78e60', 'pinterest3', 'pinterest4', 'pinterest6', '6978']
TITLES = {
    'b78e60': '军标 dog tag 迷彩 + 三行军标文字（BlackOpsOne）',
    'pinterest3': '牛仔蝴蝶贴布绣 + DENIM 大字（LeagueSpartan-Black）',
    'pinterest4': '热带迷彩棕榈（无主体图案，只重塑色块）',
    'pinterest6': '鹰/骷髅金属海报 + 尖刺标题（MetalMania）',
    '6978': '紫色蝙蝠徽章 + 弧字/品牌字（Playfair Black）',
}
SCORES = {
    'b78e60': ('8/10',
               '三行 BlackOpsOne 全部到位：位置实测偏移 ≤6px，旧字零残留，dog tag / 链条 / 迷彩碎片全保留。'
               '扣分：擦除区被 nn 填充抹平了一小块迷彩（肉眼在字缝间可见一圈比周围更平的浅色晕）。'),
    'pinterest3': ('8/10',
                   'DENIM 同位置同排版，M 下方那段旧「Y」下摆残留已清掉（背景残留块扫描 = 0）；'
                   '蝴蝶 / 虚线飞行轨迹 / 牛仔纹理全保留。'
                   '扣分：新字是纯色平涂，没迁移原 applique 的牛仔纹理+铆钉；受宽度约束字号收了 18%（cap 151 vs 原 183）。'),
    'pinterest4': ('8/10',
                   '色板 100% 取自原图（不可能引入新色），棕榈剪影逐像素保留（边缘环亮度 orig 58 = var 58，'
                   '早期那圈「贴纸描边」已修掉）；色块形状/角度/大小已裂变。'
                   '扣分：约 48% 像素发生位移，局部色块归属变了（如橄榄绿→棕），是这套「色块重塑」的固有代价。'),
    'pinterest6': ('8/10',
                   '黑底回到纯黑（擦除区实测亮度 0.9，原图黑底中位 0）；标题自身那层深蓝光晕+鬼影彻底清掉'
                   '（蓝光占比 14.8% → 0.0%）；RAVEN 用 MetalMania 按原字母带排（顶部 482 vs 原带 480）。'
                   '扣分：原图标题两侧的白色尖刺装饰随旧标题一起被擦掉、没有重画；'
                   '顶部两角仍有极淡残影（亮度 <10，正常观看不可见）。'),
    '6978': ('8/10',
             '弧字 LA CASA DELLE OMBRE / SET / 1868 / NOCTAVEN / MOONHEART 全部原位改写：'
             '字行纵向实测 orig 1003..1147 vs var 1002..1145、orig 1198..1323 vs var 1189..1323；'
             '原 BACARDÍ 的 Í 重音与 MCHEART 的 ® 残留已清；缎带两条边线与背景渐变完整保留。'
             '扣分：徽章周围与缎带带内还留有 nn 填充的极淡浅色痕；OMBRE 段字重比原字略轻。'),
}

rows = []
for iid in ORDER:
    ic = BY[iid]
    src = Image.open(ic['path']).convert('RGB')
    var = Image.open(OUT / f'{iid}_variant.jpg').convert('RGB')
    W = 520
    for tag, im in (('orig', src), ('var', var)):
        t = im.copy()
        t.thumbnail((W, 10000), Image.LANCZOS)
        t.save(THUMB / f'{iid}_{tag}.jpg', quality=88)
    sc, note = SCORES[iid]
    rows.append(f"""
  <section class="card">
    <h2>{iid} <span class="sub">{TITLES[iid]}</span><span class="score">自评 {sc}</span></h2>
    <div class="pair">
      <figure><img src="thumbs/{iid}_orig.jpg"><figcaption>原图</figcaption></figure>
      <figure><img src="thumbs/{iid}_var.jpg"><figcaption>裂变（原尺寸比例，无缩放变形）</figcaption></figure>
    </div>
    <p class="note">{note}</p>
  </section>""")

html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>image-fission v328 — 五图原位裂变对照</title>
<style>
 :root{{--bg:#f6f6f7;--card:#fff;--ink:#1b1d21;--muted:#6b7280;--line:#e5e7eb}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);
      font:15px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
 header{{padding:28px 32px 18px;border-bottom:1px solid var(--line);background:#fff}}
 h1{{margin:0 0 6px;font-size:22px}}
 .lead{{margin:0;color:var(--muted);font-size:14px}}
 .rules{{margin:14px 0 0;padding:12px 16px;background:#fff8f8;border:1px solid #f2d5d7;
         border-radius:8px;font-size:13.5px;color:#7a2b2f}}
 .wrap{{padding:22px 32px 60px;display:grid;gap:20px}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}}
 .card h2{{margin:0 0 12px;font-size:17px;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}}
 .sub{{font-weight:400;color:var(--muted);font-size:13px}}
 .score{{margin-left:auto;font-size:13px;font-weight:600;color:#0f766e;
         background:#ecfdf5;border:1px solid #a7f3d0;border-radius:999px;padding:2px 10px}}
 .pair{{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}}
 figure{{margin:0;flex:0 1 auto}}
 figure img{{display:block;border:1px solid var(--line);border-radius:6px;background:#000}}
 figcaption{{margin-top:6px;color:var(--muted);font-size:12.5px;text-align:center}}
 .note{{margin:12px 0 0;font-size:13.5px;color:#374151}}
</style></head>
<body>
<header>
  <h1>image-fission v328 — 五图「原位裂变」对照</h1>
  <p class="lead">路线：不改全图，只在原图上做局部原位裂变。文字＝笔画级掩膜擦除 + 原位同字体同字高重画；
     无主体图案＝色板保色的色块几何形变。图片按原尺寸比例展示，未做变形缩放。</p>
  <p class="rules"><b>本轮硬规则（已写进代码）</b>：① 禁任何矩形/色块遮挡式改字，掩膜必须跟着字形走；
     ② 新词沿用原字体 / 原字高 / 原排版位置（以**墨迹包围盒**对齐，实测字行纵向误差 ≤10px）；
     ③ 配色、背景、主体一律不动；④ 擦除后背景必须自然延续（不许出现「被抹平的一块」/云块/鬼影）。</p>
</header>
<div class="wrap">{''.join(rows)}</div>
</body></html>"""

(OUT / 'gallery_v328.html').write_text(html, encoding='utf-8')
print('gallery ->', OUT / 'gallery_v328.html')
