# -*- coding: utf-8 -*-
"""spec → FLUX pos/neg prompt 生成器。
把 VLM 语义拆解（ensemble_spec.json）+ 程序像素实测（*_pxmeasure.json）合成生成层 prompt。
LOCK：色板精确 RGB/占比、构图分区/占比、元素种类数量大小、纹理肌理工艺——全部钉死。
FISSION：只留一句「主体元素换成新形态」，不改色不改构图。

用法:
  python _zai_spec2prompt.py --spec <ensemble_spec.json> --px <pxmeasure.json> --out <prompt.json> [--name 04]
  python _zai_spec2prompt.py --batch <spec_dir> --out-dir <prompt_dir>
"""
import argparse, json, os, glob, re

# 中文色名 → 英文生成用词补充
COLOR_EN = {
    "深棕黑": "deep brown-black", "砖红": "brick red", "古铜金": "antique brass gold",
    "金色": "gold", "暖橙奶油": "warm orange cream", "近黑": "near-black",
    "灰紫": "muted grey-violet", "蓝": "blue", "白": "white", "红": "red",
    "黑": "black", "黄": "yellow", "橙": "orange", "奶油": "cream",
}

def rgb_str(rgb):
    try:
        return "RGB(%d,%d,%d)" % tuple(int(v) for v in rgb[:3])
    except Exception:
        return ""

def pct(v, default=None):
    m = re.search(r"(\d+(?:\.\d+)?)", str(v) if v is not None else "")
    return float(m.group(1)) if m else default

def unwrap(spec):
    """兼容 {meta,spec} 包裹格式与纯 spec 格式。"""
    if isinstance(spec, dict) and "spec" in spec and isinstance(spec["spec"], dict):
        return spec["spec"]
    return spec

def build(spec, px, name="", fission_hint=""):
    spec = unwrap(spec)
    if spec is None:
        raise ValueError("spec 为空")
    canvas = spec.get("canvas", {}) or {}
    bg = spec.get("background", {}) or {}
    comp = spec.get("composition", {}) or {}
    craft = spec.get("texture_craft", {}) or {}
    pal = spec.get("palette", []) or []
    elems = spec.get("elements", []) or []
    zon = spec.get("text_zones", []) or []
    kw = spec.get("style_keywords", []) or []
    warn = spec.get("fidelity_warning", []) or []

    # ---- 背景色：VLM 语义 + 像素实测交叉，优先实测精确 RGB ----
    px_bg = px.get("background_rgb_measured") if px else None
    px_pal = (px.get("palette_measured") or []) if px else []
    bg_rgb = px_bg or (bg.get("base_color_rgb") or [])
    bg_share = px_pal[0]["share_pct"] if px_pal else None
    # 暗色语义锚：FLUX 只吃 RGB 数字会偏亮，必须加 very dark / near-black 文字锚
    lum = sum(bg_rgb[:3]) / 3 if bg_rgb else 128
    dark_anchor = ("VERY DARK near-black " if lum < 60 else
                   ("dark muted " if lum < 110 else ""))
    bg_line = (f"The ENTIRE background is one flat field of {dark_anchor}EXACT {rgb_str(bg_rgb)} "
               f"({bg.get('color_name','')}), covering about {bg_share} percent of the canvas. "
               f"{'It must read as almost black, NOT medium brown, NOT grey-brown. ' if lum < 60 else ''}"
               "NO other background colour, NO gradient wash, NO colour-block panels.")
    if bg.get("texture_detail"):
        bg_line += f" Subtle background texture: {bg.get('texture_detail')}."
    margin = (px.get("margin") or {}) if px else {}
    if margin.get("top_px", -1) >= 0:
        cw, ch = margin.get("canvas", [896, 1152])
        bg_line += (f" Wide empty margins: top {round(margin['top_px']/ch*100)}%, "
                    f"bottom {round(margin['bottom_px']/ch*100)}%, "
                    f"left {round(margin['left_px']/cw*100)}%, right {round(margin['right_px']/cw*100)}% "
                    "of canvas — nothing touches the edges.")

    # ---- 色板：逐色精确 RGB + 占比 ----
    pal_lines, used = [], set()
    for c in pal[:8]:
        rgb = c.get("rgb") or []
        if not rgb:
            continue
        key = tuple(int(v)//12 for v in rgb[:3])
        if key in used:
            continue
        used.add(key)
        # 用实测色板校正最近的 VLM 色 + 占比（像素交程序实测）
        share = c.get('approx_share_pct', '?')
        matched = False
        if px_pal:
            best = min(px_pal, key=lambda p: sum((p["rgb"][i]-rgb[i])**2 for i in range(3)))
            dist = sum((best["rgb"][i]-rgb[i])**2 for i in range(3))
            if dist < 2500:
                rgb = best["rgb"]
                share = best["share_pct"]  # 占比以实测为准，避免与背景行数字打架
                matched = True
        # VLM 色若在实测中不存在（色值差距大），其占比是脑补值 → 不写百分比，只写语义
        share_txt = f"about {share}%" if matched or not px_pal else "accent amount"
        pal_lines.append(f"- {c.get('role','accent')} {c.get('name','')} EXACT {rgb_str(rgb)}"
                         f" {share_txt} — {c.get('where','')}")
    palette_block = "STRICT PALETTE — reproduce these exact colours, no colour-shift:\n" + "\n".join(pal_lines)

    # ---- 构图：★不采信 VLM 的 layout 文字描述（04 实测它把盾牌误读成"中央圆形放射状"）
    # 只用程序实测的几何客观量：留白%、亮条 run 位置、主体宽度反推 ----
    main_w = None
    if margin.get("left_px", -1) >= 0:
        cw0 = margin.get("canvas", [896, 1152])[0]
        main_w = round((1 - (margin["left_px"] + margin["right_px"]) / cw0) * 100)
    comp_lines = ["COMPOSITION — measured geometry from the reference image, follow exactly:"]
    if margin.get("top_px", -1) >= 0:
        cw0, ch0 = margin["canvas"]
        comp_lines.append(f"  Subject cluster occupies the CENTRE {int(main_w)}% of the width: "
                          f"left margin {round(margin['left_px']/cw0*100)}%, "
                          f"right margin {round(margin['right_px']/cw0*100)}%, "
                          f"top margin {round(margin['top_px']/ch0*100)}%, "
                          f"bottom margin {round(margin['bottom_px']/ch0*100)}%. "
                          "All four margins are EMPTY background — keep them completely clear.")
    # 实测亮条 run → 转成"这些高度有横向亮色带"的客观提示（不猜内容）
    bright = (px.get("bright_row_runs") or []) if px else []
    bands = [r for r in bright if r[1] - r[0] >= 8]
    if bands:
        comp_lines.append("  Measured horizontal bright bands at y≈" +
                          ", ".join(f"{r[0]}-{r[1]}" for r in bands[:6]) +
                          " px — these are metallic plates/banners, reproduce as blank raised plates.")
    comp_lines.append("  The arrangement is VERTICAL, top to bottom; do NOT scatter elements across the canvas.")
    comp_block = "\n".join(comp_lines)

    # ---- 元素：只给「种类 + 数量」参考，★不写 VLM 脑补的 center/size 坐标 ----
    el_lines = []
    seen_el = set()
    for e in elems[:20]:
        t = str(e.get('type', 'element'))
        if t in seen_el:
            continue
        seen_el.add(t)
        el_lines.append(f"- {t}: quantity about {e.get('count',1)} "
                        f"({e.get('detail','')[:40]}).")
    elem_block = ("ELEMENT KINDS & COUNTS — same kinds, same counts as reference "
                  "(their exact shapes will be re-imagined):\n" + "\n".join(el_lines))

    # ---- 文本区：★只声明「这里是空白区」，绝不复述字体/颜色/内容，否则 FLUX 必写伪字 ----
    tz_pos = [str(t.get('position', '')) for t in zon[:10] if t.get('position')]
    text_block = ("ALL lettering plates, banners and badge centres are COMPLETELY BLANK — "
                  "the generator must draw NO letters, NO numbers, NO words anywhere. "
                  "The later layout stage will add correct lettering into these blank plates: " +
                  "; ".join(tz_pos[:6]) + ". Leave those surfaces empty and clean.") if tz_pos else (
                  "NO text, NO letters, NO numbers anywhere — all plates and banners stay completely blank.")

    # ---- 工艺肌理 ----
    tex = (f"CRAFT / SURFACE: {craft.get('craft','')} — {craft.get('surface','')}. "
           f"Edges: {craft.get('edge_quality','')}. Aging: {craft.get('aging','')}. "
           f"Signature technique: {craft.get('signature_technique','')}.")
    if px and px.get("texture"):
        tex += f" [measured surface roughness: {px['texture']['roughness']}, grad {px['texture']['grad_mean']}]."
    kw_line = "STYLE KEYWORDS: " + ", ".join(str(k) for k in kw[:20]) + "."

    # ---- FISSION：唯一变量 ----
    fis = (fission_hint or
           "The subject elements themselves must be RE-IMAGINED as new forms of the same species "
           "(different pose/shape/design), but their colours, sizes, counts, positions, "
           "the palette, the composition and all textures stay locked to the reference. "
           "NEVER copy the reference pixels; NEVER change a colour.")

    pos = "\n".join([f"A poster artwork faithful to the reference in colour, composition, texture and layout. [{name}]",
                     bg_line, palette_block, comp_block, elem_block, text_block, tex, kw_line, fis,
                     "Crisp clean hard edges, sharp focus, extremely detailed, no blurring, no melted shapes."])

    neg_parts = [
        "text, letters, words, numbers, percentages, typography, captions, watermark, logo, "
        "gibberish, fake lettering, pseudo text, Random letters, unreadable words, engraved letters, "
        "circular lettering, text ring, arc inscription, letters around the badge, numbers on badges",
        "wrong subject type, wrong shape, round badge instead of shield, circular medallion instead of shield, "
        "ribbon banner, ribbon scroll, three badges at bottom, extra round badges",
        "colour shift, colour change, different hue, hue drift, recoloured, tinted, colour cast",
        "oversized subject, subject filling the canvas, small margins, cropped edges, wrong proportions",
        "missing elements, fewer elements, extra elements, wrong element count, wrong element size",
        "flat colour blocks, smooth gradient, plastic, clean vector clipart, low detail, cheap cartoon",
        "blurry, smudged, soft edges, airbrush haze, melted shapes, deformed anatomy, missing limbs",
        "paper border, paper edge, frame, vignette ring, circular backdrop, disc behind subject",
        "changed background colour, gradient background, colour-block panels, patch boundaries",
    ]
    for w in warn:
        if w:
            neg_parts.append("AVOID: " + str(w))
    neg = "\n".join(neg_parts)
    return {"name": name, "pos": pos, "neg": neg,
            "_src": {"has_spec": True, "has_px": bool(px),
                     "n_elements": len(elems), "n_colors": len(pal), "n_textzones": len(zon)}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="")
    ap.add_argument("--px", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--fission-hint", default="")
    ap.add_argument("--batch", default="", help="spec 目录，找 */ensemble_spec.json + *_pxmeasure.json")
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()

    if a.batch:
        os.makedirs(a.out_dir or ".", exist_ok=True)
        for sp in sorted(glob.glob(os.path.join(a.batch, "**", "ensemble_spec.json"), recursive=True)):
            d = os.path.dirname(sp)
            nm = os.path.basename(d)
            pxs = glob.glob(os.path.join(d, "*_pxmeasure.json"))
            px = json.load(open(pxs[0], encoding="utf-8")) if pxs else None
            spec = json.load(open(sp, encoding="utf-8"))
            out = build(spec, px, nm)
            op = os.path.join(a.out_dir or ".", f"{nm}_prompt.json")
            with open(op, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"[ok] {nm}: {out['_src']} -> {op}")
        return

    spec = json.load(open(a.spec, encoding="utf-8"))
    px = json.load(open(a.px, encoding="utf-8")) if a.px and os.path.exists(a.px) else None
    out = build(spec, px, a.name, a.fission_hint)
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)

if __name__ == "__main__":
    main()
