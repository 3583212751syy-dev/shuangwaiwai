# -*- coding: utf-8 -*-
"""汇总拆解 spec 成 Markdown 报告 + 拼版对比图。
用法: python _zai_spec_report.py --root outputs/_zai_spec --out <报告.md> --collage <拼版.png>
"""
import argparse, glob, json, os, math
from PIL import Image, ImageDraw, ImageFont

def fmt_rgb(rgb):
    try:
        return f"({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
    except Exception:
        return "?"

def num_desc(n):
    return f"~{n}种"

def build_md(root, nums):
    lines = ["# 原图深度拆解报告（多模型 VLM + 程序像素实测）", "",
             "> 语义（元素/文字/工艺）来自 qwen2.5vl+gemma3 集合；色值/占比/留白/亮条位置来自程序实测（KMeans+扫描）。",
             "> 标注图：青框=元素、品红框=文字区、绿线=实测留白、左上角色块=实测主色板。", ""]
    for n in nums:
        d = os.path.join(root, n)
        es = os.path.join(d, "ensemble_spec.json")
        ps = glob.glob(os.path.join(d, "*_pxmeasure.json"))
        if not os.path.exists(es) or not ps:
            continue
        spec = json.load(open(es, encoding="utf-8"))
        px = json.load(open(ps[0], encoding="utf-8"))
        lines += [f"## {n} 号原图", ""]
        # 实测色板
        pal = (px.get("palette_measured") or [])[:6]
        lines.append("**实测主色板**（RGB / 占比）：" +
                     " · ".join(f"{fmt_rgb(c['rgb'])} {c['share_pct']}%" for c in pal))
        m = px.get("margin") or {}
        if m.get("top_px", -1) >= 0:
            cw, ch = m["canvas"]
            lines.append(f"**实测留白**：上 {round(m['top_px']/ch*100)}% · 下 {round(m['bottom_px']/ch*100)}% · "
                         f"左 {round(m['left_px']/cw*100)}% · 右 {round(m['right_px']/cw*100)}%")
        lines.append("")
        elems = spec.get("elements") or []
        if elems:
            lines.append("**元素清单**（种类/数量，VLM 集合）：")
            for e in elems[:12]:
                lines.append(f"- {e.get('type','?')} ×{e.get('count','?')} — {str(e.get('detail',''))[:40]}"
                             + (f" [共识{e['_voted_by']}]" if e.get("_voted_by") else ""))
            lines.append("")
        tz = spec.get("text_zones") or []
        if tz:
            bad = ("unreadable", "读到的文字", "内容", "content", "?", "N/A")
            contents = [str(t.get("content", "")).strip() for t in tz
                        if t.get("content") and str(t.get("content", "")).strip() not in bad
                        and len(str(t.get("content", ""))) < 30]
            arrs = sorted(set(str(t.get("arrangement", "")) for t in tz if t.get("arrangement")))
            lines.append(f"**文字区**：{len(tz)} 处；可读内容：{', '.join(contents[:12]) or '无（装饰伪字）'}"
                         f"；排列：{', '.join(arrs)}")
            lines.append("")
        craft = spec.get("texture_craft") or {}
        if craft:
            lines.append(f"**工艺肌理**：{craft.get('craft','')} · {craft.get('surface','')} · "
                         f"做旧={craft.get('aging','')}")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)

def make_collage(root, nums, out, cols=2):
    ims = []
    for n in nums:
        cands = glob.glob(os.path.join(root, n, "*annotated*.png"))
        if cands:
            im = Image.open(cands[0]).convert("RGB")
            im.thumbnail((400, 520))
            ims.append((n, im.copy()))
    if not ims:
        return
    cellw, cellh = 420, 560
    rows = math.ceil(len(ims) / cols)
    canvas = Image.new("RGB", (cellw * cols, cellh * rows), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    except Exception:
        f = ImageFont.load_default()
    for i, (n, im) in enumerate(ims):
        x, y = (i % cols) * cellw, (i // cols) * cellh
        canvas.paste(im, (x + (cellw - im.width)//2, y + 30))
        d.text((x + 10, y + 5), n + " 号原图拆解标注", fill=(20, 20, 20), font=f)
    canvas.save(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/_zai_spec")
    ap.add_argument("--out", default="outputs/_zai_spec/拆解报告.md")
    ap.add_argument("--collage", default="outputs/_zai_spec/拆解标注拼版.png")
    a = ap.parse_args()
    nums = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.root, "[0-1][0-9]")) if os.path.isdir(p))
    md = build_md(a.root, nums)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[report] {a.out} ({len(md)} chars, {len(nums)} images)")
    make_collage(a.root, nums, a.collage)
    print(f"[collage] {a.collage}")

if __name__ == "__main__":
    main()
