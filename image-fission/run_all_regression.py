# -*- coding: utf-8 -*-
"""run_all_regression.py — 用升级后的 4 风格独立模块，对 6 图回归集各跑一张裂变并生成对比 gallery。

每风格对应独立代码模块（styles/<style_key>.py），互不影响；本脚本只负责：
 1. 读 regression_set/set.json
 2. 按每图 style_key 调对应模块 .fission()
 3. 拼 before/after 对比 gallery
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from styles import pattern_no_subject, camo_pattern, subject_badge_text, subject_badge_textless

MODS = {
    "pattern_no_subject": pattern_no_subject,
    "camo_pattern": camo_pattern,
    "subject_badge_text": subject_badge_text,
    "subject_badge_textless": subject_badge_textless,
}

CFG_PATH = ROOT / "regression_set" / "set.json"
OUT_ROOT = ROOT / "jobs" / "regression_v325_upgrade"


def find_variant(out_dir: Path):
    cands = sorted({p for p in out_dir.rglob("*.jpg") if p.is_file()})
    # 优先取顶层 01_* 文件
    top = [c for c in cands if c.parent == out_dir]
    if top:
        return top[0]
    return cands[0] if cands else None


def run_all():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    stageA = OUT_ROOT / "stageA"
    stageA.mkdir(parents=True, exist_ok=True)
    for img in cfg["images"]:
        mod = MODS.get(img["style_key"])
        if mod is None:
            print(f"[skip] {img['id']}: no module for {img['style_key']}")
            continue
        out = stageA / img["id"]
        print(f"\n==== {img['id']}  ->  {img['style_key']} ====")
        try:
            mod.fission(img["path"], out, cfg)
        except Exception as e:
            print(f"[ERROR] {img['id']}: {repr(e)}")
    build_gallery(cfg)


def build_gallery(cfg):
    cards = []
    for img in cfg["images"]:
        out = OUT_ROOT / "stageA" / img["id"]
        var = find_variant(out)
        mod = MODS.get(img["style_key"])
        notes = mod.selfcheck_notes() if mod else ""
        orig_rel = str(Path(img["path"]).relative_to(Path("E:/Desktop")))
        if var:
            var_rel = str(var.relative_to(OUT_ROOT))  # 相对 gallery 所在目录
            var_html = f'<img src="{var_rel}">'
        else:
            var_html = '<div class="warn">无输出</div>'
        cards.append(f"""<div class="card">
  <h2>{img['id']} <span class="tag">{img['style_key']}</span></h2>
  <div class="pair">
    <div><div class="cap">原图</div><img src="file:///E:/Desktop/{orig_rel}"></div>
    <div><div class="cap">裂变</div>{var_html}</div>
  </div>
  <div class="meta">{img.get('strategy','')}</div>
  <div class="notes">自检：{notes}</div>
</div>""")
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>image-fission 6图回归 v325 升级对比</title>
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
<h1>image-fission 6图回归 · v325 升级对比</h1>
<div class="sub">每风格独立代码模块（per-style isolated）；before=原图，after=升级后裂变。{len(cards)} 张。</div>
<div class="wrap">{''.join(cards)}</div>
</body></html>"""
    out_html = OUT_ROOT / "gallery_v325.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\n[DONE] gallery -> {out_html}  ({out_html.stat().st_size} bytes)")


if __name__ == "__main__":
    run_all()
