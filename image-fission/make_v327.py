"""v327 汇总：对 5 张图产出可用裂变图 + 对比 gallery。
策略（本轮验证过的可靠做法）：
  - 一次性全局掩膜 + 单次 LaMa 擦除 + 统一重绘（替代旧「逐条矩形掩膜+回贴」）
  - 6978：只擦徽章下方品牌文字块（弧字本轮保留，见报告）
  - b78e60：三行文字并集矩形掩膜（dilate 6）
  - pinterest3/4/6：沿用已验证产物
"""
import json, sys, shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from scipy import ndimage
from styles import base
import v268_lama_clean as lc

ROOT = Path('.')
OUT = ROOT / 'jobs' / 'router_out_v327'; OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / 'regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}
KEYS = ['6978', 'b78e60', 'pinterest3', 'pinterest4', 'pinterest6']
SRC_OLD = ROOT / 'jobs' / 'router_out_v326'


def lama(img, mask_bool, removal_strength=235, edge_smoothness=5, feather=6.0):
    """LaMa 擦除 + 羽化融合：掩膜内部取 LaMa 结果，边界平滑过渡，消除硬边涂抹。"""
    mp = Image.fromarray((mask_bool * 255).astype(np.uint8), 'L')
    cleaned = lc.lama_inpaint(img, mp, removal_strength=removal_strength, edge_smoothness=edge_smoothness)
    if feather and feather > 0:
        a = ndimage.gaussian_filter(mask_bool.astype(np.float32), sigma=feather)[..., None]
        a = np.clip(a, 0, 1)
        outa = np.asarray(img, np.float32) * (1 - a) + np.asarray(cleaned, np.float32) * a
        return Image.fromarray(np.clip(outa, 0, 255).astype(np.uint8), 'RGB')
    return cleaned


def draw_plan(img, plan, skip_arc=True):
    for it in plan:
        if skip_arc and it.get('arc'):
            continue
        img = base._render_word(img, it['word'], it['bbox'], it.get('font', 'blackopsone'),
                                tuple(it.get('color', [0, 0, 0])))
    return img


def do_6978():
    ic = BY['6978']; img = Image.open(ic['path']).convert('RGB'); W, H = img.size
    lum = np.asarray(img.convert('L'), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    # 只擦徽章下方品牌文字块（弧字保留）：y 780..1400 的深色笔画
    # 上段(y780..900)避开绶带下垂端(x<400 / x>1170)，避免擦到绶带造成晕染
    region = np.zeros((H, W), bool)
    region[900:1400, 120:1440] = True
    region[780:900, 400:1170] = True
    m = (lum < 118) & region
    m = ndimage.binary_dilation(m, iterations=5)
    print(f"[6978] mask_frac={m.mean():.3f}")
    cleaned = lama(img, m, removal_strength=240, edge_smoothness=5, feather=0.0)
    cleaned.save(OUT / '_6978_clean.jpg', quality=95)
    return draw_plan(cleaned, ic['text_plan'], skip_arc=True)


def do_b78e60():
    ic = BY['b78e60']; img = Image.open(ic['path']).convert('RGB'); W, H = img.size
    lum = np.asarray(img.convert('L'), np.float32)
    # set.json 的 bbox 偏低 ~50px，改按「暗底上的深色笔画」自动定位三行文字
    region = np.zeros((H, W), bool); region[420:762, 380:1180] = True
    m = (lum < 105) & region
    m = ndimage.binary_dilation(m, iterations=5)
    print(f"[b78e60] mask_frac={m.mean():.3f} (auto dark-stroke)")
    cleaned = lama(img, m)
    cleaned.save(OUT / '_b78e60_clean.jpg', quality=95)
    return draw_plan(cleaned, ic['text_plan'], skip_arc=False)


def main():
    res = {}
    for iid in KEYS:
        if iid == '6978':
            res[iid] = do_6978()
        elif iid == 'b78e60':
            res[iid] = do_b78e60()
        else:
            p = SRC_OLD / iid / '01_custom_1.jpg'
            res[iid] = Image.open(p).convert('RGB')
            print(f"[{iid}] reuse {p}")
        res[iid].save(OUT / f'{iid}_variant.jpg', quality=92)

    # gallery
    cards = []
    for iid in KEYS:
        ic = BY[iid]
        cards.append(f"""<div class="card"><h2>{iid} <span class="tag">{ic['style_key']}</span></h2>
        <div class="pair">
          <div><div class="cap">原图</div><img src="file:///{ic['path']}"></div>
          <div><div class="cap">裂变 v327</div><img src="file:///{(OUT / (iid + '_variant.jpg')).resolve()}"></div>
        </div></div>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>image-fission v327 对比</title>
<style>body{{background:#f5f5f7;font-family:system-ui,'Microsoft YaHei';margin:0;padding:24px;color:#1d1d1f}}
h1{{font-size:20px}} .card{{background:#fff;border-radius:14px;padding:16px;margin:16px 0;box-shadow:0 2px 10px rgba(0,0,0,.08)}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .pair img{{width:100%;border-radius:8px;display:block}}
.cap{{font-size:13px;color:#666;margin-bottom:6px}} .tag{{font-size:12px;background:#eef;color:#446;padding:2px 8px;border-radius:8px}}</style>
</head><body><h1>image-fission v327 — 原图 vs 裂变（5 图）</h1>
<p style="color:#666;font-size:14px">6978/b78e60 使用「一次性全局掩膜 + 单次 LaMa 擦除 + 统一重绘」；pinterest3/4/6 沿用已验证产物。</p>
{''.join(cards)}</body></html>"""
    (OUT / 'gallery_v327.html').write_text(html, encoding='utf-8')
    print('[DONE]', OUT / 'gallery_v327.html')


if __name__ == '__main__':
    main()
