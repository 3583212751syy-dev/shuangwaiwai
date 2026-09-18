# -*- coding: utf-8 -*-
"""run_five_v326.py — 对 5 张演示图各跑 1 张裂变（v326 升级后），生成对比 gallery。

演示集（用户 2026-09-14 指定）：
  6978(BACARDÍ 蝙蝠)        -> subject_badge_text
  b78e60(军标狗牌迷彩)       -> camo_pattern
  pinterest3(牛仔蝴蝶 UPCY)  -> subject_badge_text
  pinterest4(迷彩棕榈)       -> camo_pattern
  pinterest6(鹰骷髅金属)      -> subject_badge_textless

路由：fission_router.route_one() 按 set.json 的 style_key 派发到对应独立模块。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys_path = str(ROOT)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from fission_router import route_one

CFG_PATH = ROOT / "regression_set" / "set.json"
OUT_ROOT = ROOT / "jobs" / "router_out_v326"
SHOW_IDS = ["6978", "b78e60", "pinterest3", "pinterest4", "pinterest6"]


def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    by_id = {img["id"]: img for img in cfg["images"]}
    results = []
    for iid in SHOW_IDS:
        img = by_id.get(iid)
        if not img:
            print(f"[skip] {iid} 不在 set.json")
            continue
        img_out = OUT_ROOT / iid
        img_out.mkdir(parents=True, exist_ok=True)
        print(f"\n==== {iid} ({img['style_key']}) ====")
        try:
            res = route_one(img["path"], img_out, seed=None)
        except Exception as e:
            print(f"[ERROR] {iid}: {repr(e)}")
            res = {"image": img["path"], "style_key": img["style_key"], "variants": [], "selfcheck": ""}
        results.append((iid, res))

    build_gallery(results, cfg)
    print("\n[DONE] 5 图裂变完成，gallery 见", OUT_ROOT / "gallery_v326.html")


def build_gallery(results, cfg):
    by_id = {img["id"]: img for img in cfg["images"]}
    cards = []
    for iid, res in results:
        orig = res["image"]
        variants = res.get("variants") or []
        var_html = '<img src="file:///{}">'.format(variants[0]) if variants else '<div class="warn">无输出</div>'
        notes = res.get("selfcheck", "")
        strategy = by_id.get(iid, {}).get("strategy", "")
        cards.append(f"""<div class="card">
  <h2>{iid or '?'} <span class="tag">{res['style_key']}</span></h2>
  <div class="pair">
    <div><div class="cap">原图</div><img src="file:///{orig}"></div>
    <div><div class="cap">裂变(v326)</div>{var_html}</div>
  </div>
  <div class="meta">{strategy}</div>
  <div class="notes">自检：{notes}</div>
</div>""")
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>image-fission 5图演示 · v326 camo-blob-morph</title>
<style>
body{{font-family:Segoe UI,system-ui,sans-serif;background:#f5f5f5;color:#222;padding:20px;margin:0;}}
h1{{font-size:22px;margin:0 0 4px;}}
.sub{{color:#666;margin-bottom:18px;font-size:13px;}}
.wrap{{display:grid;grid-template-columns:1fr 1fr;gap:14px;max-width:1500px;margin:auto;}}
.card{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,.08);}}
h2{{font-size:16px;margin:6px 0;}}
.tag{{display:inline-block;background:#eef;color:#315;padding:2px 9px;border-radius:12px;font-size:12px;font-weight:600;}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px;}}
.pair img{{width:100%;height:auto;border:1px solid #e0e0e0;border-radius:6px;}}
.cap{{font-size:12px;color:#555;margin:4px 0 2px;font-weight:600;}}
.meta{{font-size:12px;color:#444;margin-top:8px;}}
.notes{{font-size:11px;color:#777;margin-top:4px;}}
.warn{{color:#c00;padding:20px;text-align:center;}}
</style></head><body>
<h1>image-fission 5图演示 · v326 camo-blob-morph</h1>
<div class="sub">每图按识别风格调用独立模块（per-style isolated）。before=原图，after=v326 升级后裂变。{len(cards)} 张。</div>
<div class="wrap">{''.join(cards)}</div>
</body></html>"""
    out = OUT_ROOT / "gallery_v326.html"
    out.write_text(html, encoding="utf-8")
    print(f"[gallery] -> {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
