# -*- coding: utf-8 -*-
"""
build_v322d_gallery.py — 为 v322d 产出生成"原图 ↔ 变体"对照 Gallery
用法
====
    ./venv/Scripts/python.exe src/build_v322d_gallery.py
产物
====
    jobs/fission_v322d/gallery.html  (内嵌 base64 图, 单文件, 直接 present_files)
"""
import os, base64, io, json, sys
from pathlib import Path
from PIL import Image

SRC = Path(r"E:/Desktop/图裂变测试图")
V322D = Path(r"E:/Desktop/双接口/image-fission/jobs/fission_v322d")


def _img_to_data_uri(img, max_w=320):
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_uri(p):
    if not p:
        return ""
    try:
        if Path(p).exists():
            return _img_to_data_uri(p)
    except Exception:
        pass
    return ""


def main():
    # 收集所有 v322d 产出，按 base 名归组
    grouped = {}
    for vf in sorted(V322D.glob("*.jpg")):
        stem = vf.stem
        if "__" not in stem:
            continue
        base, vname = stem.rsplit("__", 1)
        # 找原图（按 base 名匹配扩展名）
        orig = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            cand = SRC / (base + ext)
            if cand.exists():
                orig = cand
                break
        if orig is None:
            # 文件名里可能带空格 (Pinterest 系列)，模糊找
            for p in SRC.iterdir():
                if p.stem == base:
                    orig = p
                    break
        prof_path = V322D / f"{base}.profile.json"
        profile = {}
        if prof_path.exists():
            try:
                with open(prof_path, encoding="utf-8") as f:
                    profile = json.load(f)
            except Exception:
                pass
        g = grouped.setdefault(base, {"orig": orig, "vars": [], "profile": profile})
        g["vars"].append((vname, vf))

    # 渲染
    rows = []
    sorted_bases = sorted(grouped.keys(), key=lambda x: (grouped[x]["profile"].get("image_type", "?") != "text_poster", x))
    for base in sorted_bases:
        g = grouped[base]
        prof = g["profile"]
        n_vars = len(g["vars"])
        prof_str = (
            f"type=<b>{prof.get('image_type', '?')}</b> · has_text={prof.get('has_text', '?')} · "
            f"edges={prof.get('edge_density', 0):.3f} · bg_uni={prof.get('bg_uniformity', 0):.2f} · "
            f"colors={prof.get('n_unique_colors_q', '?')} · bands={len(prof.get('text_bands', []))} · "
            f"suggested: dh={prof.get('suggested_color_strength', '?')}, "
            f"cn={prof.get('suggested_controlnet_strength', '?')}, "
            f"denoise={prof.get('suggested_denoise', '?')}"
        )
        type_hint = ""
        t = prof.get("image_type", "?")
        if t == "text_poster":
            type_hint = " → 走 v321 文字带擦除 + HSV 微调"
        elif t == "graphic_logo":
            type_hint = " → 走 v321 文字带 + HSV + 微仿射"
        elif t == "pattern":
            type_hint = " → 走 v321 pattern 分支（仅 HSV — 这种图正是用户口中的'没有大主体元素的图案'）"
        elif t == "unknown":
            type_hint = " → 默认 HSV（v321 分类器没识别出来——> v323 路线草案重点扩展）"
        else:
            type_hint = f" → v321 {t} 分支"
        src_uri = _safe_uri(g["orig"]) if g["orig"] else ""
        var_cards = ""
        for n, vp in sorted(g["vars"]):
            uri = _safe_uri(vp)
            var_cards += (
                f'<div class="v"><img src="{uri}" loading="lazy">'
                f'<div class="cap">{n}</div></div>'
                if uri else
                f'<div class="v err">missing {vp.name}</div>'
            )
        rows.append(
            f'<section>'
            f'<div class="head">'
            f'<div class="src"><img src="{src_uri}" loading="lazy"></div>'
            f'<div class="meta">'
            f'<h3>{base}</h3>'
            f'<div class="prof">{prof_str}{type_hint}</div>'
            f'</div></div>'
            f'<div class="variants">{var_cards}</div>'
            f'</section>'
        )
    n_orig = len(grouped)
    n_var = sum(len(g["vars"]) for g in grouped.values())
    by_type = {}
    for g in grouped.values():
        t = g["profile"].get("image_type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    type_summary = " | ".join(f"{k}:{v}" for k, v in sorted(by_type.items(), key=lambda x: -x[1]))
    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>图裂变 v322d — 原图 vs 变体 Gallery</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#fafafa;padding:24px;color:#222;margin:0;}}
h1{{margin:0 0 4px;font-size:22px;}}.sub{{color:#666;margin-bottom:18px;font-size:13px;}}
section{{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:14px;margin-bottom:14px;}}
.head{{display:flex;gap:14px;align-items:flex-start;margin-bottom:10px;}}
.src img{{max-width:160px;max-height:160px;border:1px solid #ddd;}}
.meta{{flex:1;}}.meta h3{{margin:0;font-size:14px;font-family:ui-monospace,Consolas,monospace;}}
.prof{{color:#444;font-size:12px;font-family:ui-monospace,Consolas,monospace;margin-top:6px;line-height:1.55;}}
.variants{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;}}
.v{{background:#f4f4f4;border-radius:6px;overflow:hidden;}}
.v img{{width:100%;display:block;}}
.cap{{padding:5px;font-size:11px;text-align:center;color:#333;font-family:ui-monospace,Consolas,monospace;}}
.v.err{{padding:14px;font-size:12px;color:#a00;}}
.legend{{background:#fffbe6;border:1px solid #ffe58f;padding:10px 14px;border-radius:6px;margin-bottom:14px;font-size:12px;line-height:1.6;}}
.legend b{{color:#c33;}}
.type-stat{{background:#eef;border:1px solid #cce;padding:8px 12px;border-radius:6px;margin-bottom:12px;font-size:12px;font-family:ui-monospace,Consolas,monospace;}}
</style></head><body>
<h1>图裂变 v322d — Gallery（原图 ↔ 3 变体）</h1>
<div class="sub">{n_orig} 张原图 × 3 变体 = {n_var} 张裂变图 · type 分布: {type_summary}</div>
<div class="legend">
<b>⚠ 关键事实:</b>
<ol style="margin:6px 0 0 18px;padding:0;line-height:1.55;">
<li>本 gallery 列出每张原图对应的 v322d 3 变体。你截图质问"做的什么东西给我、跟原图什么关联"——关联就在这一页。</li>
<li>所有变体严守 v307 红线（色相±15°/饱和度±20%/明度±10%/仿射±5%±3°）。</li>
<li>13c8b7（红底黑佩斯利）等 pattern 型图：v322b 故意只做温和 hue 漂移避免拆碎花纹 → <b>肉眼可见度低</b>——这是 v323 要解决的核心痛点。</li>
<li>b78e60（军标迷彩）含 dog_tag+chain：v322d 把这两个小元素单独做了 scale/rot/hue 差异化，已可见。</li>
<li>text_poster 类（13c8b7/6978fab/b78e60）：文字带已擦除并替换成原创虚构词（绝不输出 BACARDÍ 等真实品牌）。</li>
<li>未知分类图占 {by_type.get('unknown', 0)}/{n_orig} 张 = {by_type.get('unknown', 0)*100//max(n_orig,1)}%——v321 分类器太苛刻，v323 路线会扩展 camo/denim/lace/photo/paisley/military_heroposter 等子型。</li>
</ol>
</div>
{''.join(rows)}
</body></html>"""
    out = V322D / "gallery.html"
    out.write_text(html, encoding="utf-8")
    print(f"✓ gallery -> {out}")
    print(f"  原图配对 {n_orig} 张, 3 变体合计 {n_var} 张")
    print(f"  type 分布: {type_summary}")


if __name__ == "__main__":
    main()
