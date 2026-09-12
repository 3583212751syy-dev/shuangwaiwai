# -*- coding: utf-8 -*-
"""文字材质专用 runner：以原图为底，不做 ComfyUI 裂变，只看文字替换效果。"""
import json, base64
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "regression_set" / "set.json"
OUT_ROOT = ROOT / "jobs" / "regression_v325_upgrade" / "text_only"
GALLERY = ROOT / "jobs" / "regression_v325_upgrade" / "gallery_text_only_v325c.html"

sys_path_set = False

def ensure_imports():
    global sys_path_set
    if not sys_path_set:
        import sys
        sys.path.insert(0, str(ROOT))
        sys_path_set = True

def img_to_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()

def run():
    ensure_imports()
    from styles import base
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    rows = []
    for img in cfg["images"]:
        plan = img.get("text_plan", [])
        if not plan:
            continue
        out_dir = OUT_ROOT / img["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n==== text-only: {img['id']} ====")
        orig = Image.open(img["path"]).convert("RGB")
        result = base.replace_text_plan_v2(orig, plan, dilate=8, use_material=True)
        out_path = out_dir / "01_text_only.jpg"
        result.save(out_path, "JPEG", quality=92)
        print(f"saved -> {out_path}")
        rows.append((img["id"], img["style_key"], Path(img["path"]), out_path))

    # build gallery
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>v325c 文字材质对比（原图 vs 文字替换）</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#f5f5f5;padding:20px;color:#333}
h1{font-size:22px;margin-bottom:8px}
.note{font-size:14px;color:#666;margin-bottom:24px}
.card{background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:28px;padding:16px}
.id{font-size:18px;font-weight:700;margin-bottom:6px}
.style{font-size:13px;color:#888;margin-bottom:12px}
.pair{display:flex;gap:16px;flex-wrap:wrap}
.pair>div{flex:1;min-width:280px;text-align:center}
.pair img{max-width:100%;max-height:520px;border:1px solid #ddd;border-radius:6px}
.label{font-size:12px;color:#666;margin-top:6px}
</style></head><body>
<h1>v325c 文字材质对比：原图 vs 仅文字替换（背景 100% 不动）</h1>
<div class="note">本轮只验证文字替换：笔画级擦除 + 弧形文字 + 原字材质迁移。ComfyUI 元素裂变未运行。</div>
"""
    for id_, style, orig_p, out_p in rows:
        html += f"""<div class="card">
<div class="id">{id_}</div>
<div class="style">style: {style}</div>
<div class="pair">
<div><img src="data:image/jpeg;base64,{img_to_b64(orig_p)}"><div class="label">原图</div></div>
<div><img src="data:image/jpeg;base64,{img_to_b64(out_p)}"><div class="label">文字替换后</div></div>
</div></div>\n"""
    html += "</body></html>"
    GALLERY.write_text(html, encoding="utf-8")
    print(f"\nGALLERY -> {GALLERY}")

if __name__ == "__main__":
    run()
