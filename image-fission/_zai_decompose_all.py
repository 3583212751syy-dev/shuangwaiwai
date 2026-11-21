# -*- coding: utf-8 -*-
"""批量拆解 + 可视化标注 + 四模型对比表。
流程：对每张原图跑 VLM 四模型拆解（或指定模型）→ 程序像素实测 → 融合 → 生成 prompt → 标注图。
用法:
  python _zai_decompose_all.py --imgs <目录> --out-root <目录> --models all [--limit N] [--only 04,05]
  python _zai_decompose_all.py --annotate-only --out-root <目录>   # 只重画标注图
"""
import argparse, json, os, sys, glob, time, subprocess, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["gemma3:12b", "qwen3-vl:8b", "qwen2.5vl:7b", "minicpm-v:8b"]

def run(cmd):
    print("[RUN]", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("[ERR]", (r.stderr or "")[-500:], file=sys.stderr)
    return r

def pct_to_xy(s, W, H):
    """解析 'x%,y%' 或 'x,y' → 像素坐标。"""
    if not s:
        return None
    import re
    nums = re.findall(r"(\d+(?:\.\d+)?)", str(s))
    if len(nums) < 2:
        return None
    x, y = float(nums[0]), float(nums[1])
    if x > 1 or y > 1:
        x, y = x / 100.0, y / 100.0
    return (int(x * W), int(y * H))

def wh_from(s, W, H):
    import re
    nums = re.findall(r"(\d+(?:\.\d+)?)", str(s))
    if len(nums) < 2:
        return (W * 0.2, H * 0.06)
    w, h = float(nums[0]), float(nums[1])
    if w <= 1 and h <= 1:
        return (int(w * W), int(h * H))
    return (int(w / 100 * W), int(h / 100 * H))

def annotate(img_path, spec, px, out_path):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        fs = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    except Exception:
        f = fs = ImageFont.load_default()

    # 元素框（青色）
    for e in (spec.get("elements") or [])[:25]:
        c = pct_to_xy(e.get("center"), W, H)
        if not c:
            continue
        w, h = wh_from(e.get("size_pct"), W, H)
        x0, y0 = max(0, c[0] - w // 2), max(0, c[1] - h // 2)
        x1, y1 = min(W, c[0] + w // 2), min(H, c[1] + h // 2)
        d.rectangle([x0, y0, x1, y1], outline=(0, 255, 255, 210), width=3)
        d.text((x0 + 3, max(0, y0 - 24)), str(e.get("type", ""))[:14], fill=(0, 255, 255, 255), font=fs)

    # 文本区（品红）
    for t in (spec.get("text_zones") or [])[:15]:
        c = pct_to_xy(t.get("center_pct"), W, H)
        if not c:
            continue
        w, h = wh_from(t.get("box_wh_pct"), W, H)
        x0, y0 = max(0, c[0] - w // 2), max(0, c[1] - h // 2)
        d.rectangle([x0, y0, min(W, c[0] + w // 2), min(H, c[1] + h // 2)],
                    outline=(255, 0, 255, 230), width=3)
        d.text((x0 + 3, max(0, y0 - 22)), str(t.get("content", ""))[:16],
               fill=(255, 120, 255, 255), font=fs)

    # 实测色板色块（左上角）
    pal = (px or {}).get("palette_measured") or []
    x = 8
    for c in pal[:8]:
        rgb = tuple(c["rgb"])
        d.rectangle([x, 8, x + 36, 36], fill=rgb + (255,), outline=(255, 255, 255, 255))
        d.text((x, 38), str(c.get("share_pct", "")), fill=(255, 255, 0, 255), font=fs)
        x += 42

    # 实测留白边（绿色虚线感：画四条线）
    m = (px or {}).get("margin") or {}
    if m.get("top_px", -1) >= 0:
        cw, ch = m["canvas"]
        d.line([(m["left_px"], 0), (m["left_px"], ch)], fill=(0, 255, 0, 150), width=2)
        d.line([(cw - m["right_px"], 0), (cw - m["right_px"], ch)], fill=(0, 255, 0, 150), width=2)
        d.line([(0, m["top_px"]), (cw, m["top_px"])], fill=(0, 255, 0, 150), width=2)
        d.line([(0, ch - m["bottom_px"]), (cw, ch - m["bottom_px"])], fill=(0, 255, 0, 150), width=2)

    out = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    out.save(out_path)
    return out_path

def model_compare(specs, out_path):
    lines = ["| 字段 | " + " | ".join(specs.keys()) + " |",
             "|---|---|" + "---|" * len(specs)]
    fields = ["palette", "elements", "text_zones", "composition", "texture_craft", "style_keywords", "fidelity_warning"]
    for f in fields:
        row = []
        for m, s in specs.items():
            v = (s or {}).get(f) or []
            if isinstance(v, dict):
                n = len([x for x in v.values() if x])
            else:
                n = len(v)
            row.append(str(n))
        lines.append(f"| {f} | " + " | ".join(row) + " |")
    # 色值对比（主色）
    row = []
    for m, s in specs.items():
        p = (s or {}).get("palette") or []
        row.append(str(p[0].get("rgb")) if p and isinstance(p[0], dict) else "-")
    lines.append("| 主色RGB(语义) | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("说明：像素实测见各目录 *_pxmeasure.json（色值/占比/坐标以此为准）。")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", default="")
    ap.add_argument("--out-root", default="outputs/_zai_spec")
    ap.add_argument("--models", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--annotate-only", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_root, exist_ok=True)
    if a.annotate_only:
        imgs = glob.glob(os.path.join(a.out_root, "**", "*_spec.json"), recursive=True)
        for sp in imgs:
            d = os.path.dirname(sp)
            px_f = glob.glob(os.path.join(d, "*_pxmeasure.json"))
            # 找原图
            num = os.path.basename(d)[:2]
            orig = glob.glob(f"D:/Users/Administrator/Desktop/双接口/outputs/masks_1_9/_fz_in_A_{num}_*.jpg")
            if not orig or not px_f:
                continue
            spec = json.load(open(sp, encoding="utf-8")).get("spec") or json.load(open(sp, encoding="utf-8"))
            px = json.load(open(px_f[0], encoding="utf-8"))
            annotate(orig[0], spec, px, os.path.join(d, "annotated_" + os.path.basename(sp).replace("_spec.json", ".png")))
        return

    imgs = sorted(glob.glob(os.path.join(a.imgs, "_fz_in_A_*.jpg")))
    def getnum(p):
        parts = os.path.basename(p).split("_")
        return parts[4] if len(parts) > 4 else "??"
    if a.only:
        want = set(a.only.split(","))
        imgs = [p for p in imgs if getnum(p) in want]
    if a.limit:
        imgs = imgs[:a.limit]
    models = MODELS if a.models == "all" else a.models.split(",")

    for img in imgs:
        num = getnum(img)
        d = os.path.join(a.out_root, num)
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        print(f"\n===== {num} {os.path.basename(img)} =====", file=sys.stderr)
        # 1) 像素实测（快）
        run([PY, os.path.join(HERE, "_zai_pxmeasure.py"), "--img", img,
             "--out", os.path.join(d, f"{num}_pxmeasure.json"), "--dump-strips", d])
        # 2) VLM 多模型拆解
        specs = {}
        for m in models:
            outp = os.path.join(d, f"{m.replace(':','_')}_spec.json")
            if os.path.exists(outp):
                j = json.load(open(outp, encoding="utf-8"))
                if j.get("meta", {}).get("ok"):
                    specs[m] = j.get("spec")
                    print(f"  {m}: cached OK", file=sys.stderr)
                    continue
            run([PY, os.path.join(HERE, "_zai_decompose.py"), "--img", img,
                 "--model", m, "--out", outp])
            if os.path.exists(outp):
                j = json.load(open(outp, encoding="utf-8"))
                if j.get("meta", {}).get("ok"):
                    specs[m] = j.get("spec")
        # 3) 融合
        if specs:
            merged = os.path.join(d, "ensemble_spec.json")
            run([PY, os.path.join(HERE, "_zai_decompose.py"), "--ensemble", d, "--out", merged])
            spec = json.load(open(merged, encoding="utf-8")) if os.path.exists(merged) else specs[list(specs)[0]]
        else:
            spec = {}
        # 4) 生成 prompt
        px_f = os.path.join(d, f"{num}_pxmeasure.json")
        if spec:
            run([PY, os.path.join(HERE, "_zai_spec2prompt.py"),
                 "--spec", os.path.join(d, "ensemble_spec.json") if specs else "",
                 "--px", px_f, "--name", num, "--out", os.path.join(d, f"{num}_prompt.json")])
        # 5) 标注图
        px = json.load(open(px_f, encoding="utf-8")) if os.path.exists(px_f) else None
        if spec and px:
            try:
                annotate(img, spec, px, os.path.join(d, f"{num}_annotated.png"))
            except Exception as e:
                print("[annotate err]", e, file=sys.stderr)
        # 6) 模型对比表
        model_compare(specs, os.path.join(d, f"{num}_model_compare.md"))
        print(f"===== {num} DONE in {round(time.time()-t0)}s =====", file=sys.stderr)

if __name__ == "__main__":
    main()
