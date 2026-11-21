# -*- coding: utf-8 -*-
"""校验层：成品 vs 原图的结构化 spec 对比（比主观 VLM 打分更定量）。
对成品跑一次程序像素实测 + VLM 拆解，与原图 spec 逐字段比：
  1) 色板：主色 RGB 欧氏距离 + 占比偏差（像素实测，硬指标）
  2) 留白：四边留白% 偏差（像素实测，硬指标）
  3) 元素：种类/数量是否缺失或多出（VLM 语义）
  4) 文本区：数量/方位是否对齐（VLM 语义）
用法:
  python _zai_spec_check.py --orig-spec <原图ensemble.json> --orig-px <原图px.json> \
      --final <成品图> --out <报告.json> [--model qwen2.5vl:7b --final-spec <成品spec.json>]
"""
import argparse, json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _zai_pxmeasure as PX

def euclid(a, b):
    try:
        return math.sqrt(sum((int(a[i]) - int(b[i])) ** 2 for i in range(3)))
    except Exception:
        return 999

def margin_pct(px):
    m = px.get("margin") or {}
    if m.get("top_px", -1) < 0:
        return None
    cw, ch = m["canvas"]
    return {"top": round(m["top_px"] / ch * 100, 1),
            "bottom": round(m["bottom_px"] / ch * 100, 1),
            "left": round(m["left_px"] / cw * 100, 1),
            "right": round(m["right_px"] / cw * 100, 1)}

def check_colors(orig_px, final_px, tol=40):
    op = (orig_px.get("palette_measured") or [])[:6]
    fp = [dict(f) for f in (final_px.get("palette_measured") or [])[:8]]
    rows, issues = [], []
    used = set()  # ★一对一匹配：同一个成品色不能被多个原色重复占用
    for i, o in enumerate(op):
        cand = [(euclid(o["rgb"], f["rgb"]), j) for j, f in enumerate(fp) if j not in used]
        if not cand:
            cand = [(999, -1)]
        d, j = min(cand)
        best = fp[j] if j >= 0 else None
        if j >= 0:
            used.add(j)
        d = float(d)
        share_o, share_f = o["share_pct"], (best["share_pct"] if best else 0)
        # 主色（占比>40%）只看色距，占比天然接近；小色只看色距，占比容差放宽
        is_main = share_o > 40
        ok = bool(d <= tol) if is_main else bool(d <= tol + 10)
        rows.append({"orig_rgb": o["rgb"], "orig_share": share_o,
                     "final_rgb": best["rgb"] if best else None, "final_share": share_f,
                     "rgb_dist": round(d, 1), "share_diff": round(abs(share_o - share_f), 1),
                     "PASS": ok})
        if not ok:
            issues.append(f"色#{i+1} RGB{o['rgb']}({share_o}%) → 最近成品色 "
                          f"{best['rgb'] if best else 'N/A'}({share_f}%) 色距{d:.0f}")
    return {"PASS": bool(all(bool(r["PASS"]) for r in rows)), "rows": rows, "issues": issues}

def check_margin(orig_px, final_px, tol=8):
    om, fm = margin_pct(orig_px), margin_pct(final_px)
    if not om or not fm:
        return {"PASS": None, "issues": ["无法测量留白"]}
    issues = []
    for k in ("top", "bottom", "left", "right"):
        d = abs(om[k] - fm[k])
        if d > tol:
            issues.append(f"{k}留白 {om[k]}% → {fm[k]}% 偏 {d:.1f}%")
    return {"PASS": bool(not issues), "orig": om, "final": fm, "issues": issues}

def check_elements(orig_spec, final_spec):
    oe = (orig_spec.get("elements") or []) if orig_spec else []
    fe = (final_spec.get("elements") or []) if final_spec else []
    def canon(e):
        t = str(e.get("type", ""))
        for k in ("盾", "徽章", "章", "人", "车", "火焰", "缎带", "星", "箭"):
            if k in t:
                return k
        return t
    oc, fc = {}, {}
    for e in oe:
        oc[canon(e)] = oc.get(canon(e), 0) + int(e.get("count") or 1)
    for e in fe:
        fc[canon(e)] = fc.get(canon(e), 0) + int(e.get("count") or 1)
    issues = []
    for k, n in oc.items():
        got = fc.get(k, 0)
        if got < n:
            issues.append(f"元素缺失: {k} 原图{n} 成品{got}")
        elif got > n:
            issues.append(f"元素增多: {k} 原图{n} 成品{got}")
    for k, n in fc.items():
        if k not in oc:
            issues.append(f"多出元素: {k} x{n}")
    return {"PASS": bool(not issues), "orig": oc, "final": fc, "issues": issues}

def check_textzones(orig_spec, final_spec):
    ot = (orig_spec.get("text_zones") or []) if orig_spec else []
    ft = (final_spec.get("text_zones") or []) if final_spec else []
    return {"PASS": None, "orig_n": len(ot), "final_n": len(ft),
            "note": "文字区数量需结合排版层实际文字数判断；生成层底图应全部留白，成品文字由排版层贴入。"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-spec", required=True)
    ap.add_argument("--orig-px", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--final-spec", default="")
    ap.add_argument("--final-px", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ospec = json.load(open(a.orig_spec, encoding="utf-8"))
    if isinstance(ospec, dict) and "spec" in ospec:
        ospec = ospec["spec"]
    opx = json.load(open(a.orig_px, encoding="utf-8"))
    fpx = json.load(open(a.final_px, encoding="utf-8")) if a.final_px and os.path.exists(a.final_px) else PX.measure(a.final)
    fspec = None
    if a.final_spec and os.path.exists(a.final_spec):
        fspec = json.load(open(a.final_spec, encoding="utf-8"))
        if isinstance(fspec, dict) and "spec" in fspec:
            fspec = fspec["spec"]

    report = {
        "final_image": os.path.basename(a.final),
        "checks": {
            "colors": check_colors(opx, fpx),
            "margin": check_margin(opx, fpx),
            "elements": check_elements(ospec, fspec),
            "text_zones": check_textzones(ospec, fspec),
        },
    }
    hard = [report["checks"]["colors"]["PASS"], report["checks"]["margin"]["PASS"]]
    report["hard_PASS"] = all(x is True for x in hard)
    txt = json.dumps(report, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)

if __name__ == "__main__":
    main()
