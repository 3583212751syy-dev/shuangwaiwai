"""v261_detect_layout — 元素原位裂变 · Stage0 元素普查 (v2, 正确分拣顺序)

修复 v1 的错误: v1 先排文字带再找主体, 把蝙蝠身(y518~974)切成文字带,
导致主体只捕捉到 112px 高的碎片。正确顺序是:
  1) 用径向剖面找徽章圆环 (center/r_inner/r_outer)
  2) 圆内最大暗色连通域 = 主体 (蝙蝠), 与 v253 BAT_BBOX 交叉验证
  3) 在「排除圆环带 + 排除主体 bbox」的前提下, 行投影找文字带
  4) 主体 bbox 之外、圆环带之外的次级连通域 = 装饰小元素

输出: jobs/v261/layout.json (全管线唯一坐标真值源) + 可选 _debug_layout.png
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
JOB = PROJECT / "jobs" / "v261"
JOB.mkdir(parents=True, exist_ok=True)
FONT_PATH = PROJECT / "fonts" / "PirataOne-Regular.ttf"

DEFAULT_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"


def radial_profile_circle(rgb):
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.5,
                               minDist=int(min(h, w) * 0.4), param1=100, param2=60,
                               minRadius=int(min(h, w) * 0.15),
                               maxRadius=int(min(h, w) * 0.45))
    if circles is None:
        return None
    c = np.array(sorted(circles[0], key=lambda t: -t[2]))
    for x, y, r in c:
        if abs(x - w / 2) < w * 0.12:
            cx, cy, r0 = int(x), int(y), int(r)
            break
    else:
        x, y, r = c[0]; cx, cy, r0 = int(x), int(y), int(r)
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rs, vs = [], []
    for r in range(max(5, r0 - 90), min(int(dist.max()), r0 + 90)):
        band = np.abs(dist - r) < 1.0
        rs.append(r); vs.append(float(gray[band].mean()) if band.sum() > 50 else 0.0)
    grad = np.abs(np.gradient(vs))
    edges = sorted([int(rs[i]) for i in np.argsort(-grad)[:2]])
    r_inner, r_outer = (edges[0], edges[1]) if len(edges) == 2 else (r0 - 14, r0)
    if r_outer - r_inner < 4:
        r_inner, r_outer = r_outer - 14, r_outer
    return cx, cy, r_outer, r_inner


def extract_palette(rgb, k=8):
    small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    z = small.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(z, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    out = []
    for i in order:
        r, g, b = [int(v) for v in centers[i]]
        out.append({"rgb": [r, g, b], "hex": "#%02X%02X%02X" % (r, g, b),
                    "pct": round(float(counts[i] / counts.sum() * 100), 2)})
    return out


def ring_mask(shape, cx, cy, ri, ro, pad=10):
    h, w = shape[:2]
    yy, xx = np.mgrid[:h, :w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (d >= ri - pad) & (d <= ro + pad)


def find_subject(gray, dist, r_in, dark_th=60):
    """圆内最大暗色连通域 = 主体 (蝙蝠)。限 dist<r_in-6 排除圆环带。"""
    inside = dist < (r_in - 6)
    dark = ((gray < dark_th) & inside).astype(np.uint8) * 255
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = [int(v) for v in stats[idx]]
    return {"bbox": [x, y, bw, bh], "center": [x + bw // 2, y + bh // 2],
            "area_px": area, "fill_ratio": round(area / float(bw * bh), 4) if bw * bh else 0}


def find_text_bands(gray, exclude, dark_th=120, min_gap=22, min_run=8):
    h, w = gray.shape
    dark = (gray < dark_th) & (~exclude)
    rows = dark.sum(axis=1).astype(float)
    rows = np.convolve(rows, np.ones(5, np.float32) / 5, mode="same")
    thr = max(rows.max() * 0.12, 6.0)
    active = rows > thr
    bands, y = [], 0
    while y < h:
        if active[y]:
            y0 = y
            while y < h and active[y]:
                y += 1
            if bands and (y0 - bands[-1][1]) < min_gap:
                bands[-1][1] = y
            else:
                bands.append([y0, y])
        else:
            y += 1
    out = []
    for y0, y1 in bands:
        if (y1 - y0) < min_run:
            continue
        seg = dark[y0:y1]
        cols = np.where(seg.sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        out.append({"y0": int(y0), "y1": int(y1), "x0": int(cols.min()),
                    "x1": int(cols.max()) + 1, "height_px": int(y1 - y0),
                    "width_px": int(cols.max() - cols.min() + 1),
                    "center_y": int((y0 + y1) / 2)})
    return out


def find_decor(gray, exclude, dark_th=140, min_area=120, max_area=55000):
    dark = (gray < dark_th).astype(np.uint8) * 255
    dark[exclude] = 0
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    items = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < min_area or area > max_area or bw < 6 or bh < 6:
            continue
        items.append({"bbox": [x, y, bw, bh], "center": [int(cent[i][0]), int(cent[i][1])],
                      "area_px": area})
    items.sort(key=lambda d: -d["area_px"])
    return items


def dump_debug(rgb, layout, out_png):
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    em = layout.get("emblem")
    if em:
        cx, cy = em["center"]
        d.ellipse([cx - em["r_outer"], cy - em["r_outer"], cx + em["r_outer"], cy + em["r_outer"]],
                  outline=(0, 255, 0, 255), width=4)
        d.ellipse([cx - em["r_inner"], cy - em["r_inner"], cx + em["r_inner"], cy + em["r_inner"]],
                  outline=(0, 200, 120, 255), width=3)
    for b in layout["text_bands"]:
        d.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=(255, 60, 60, 255), width=3)
    if layout.get("subject"):
        x, y, bw, bh = layout["subject"]["bbox"]
        d.rectangle([x, y, x + bw, y + bh], outline=(60, 140, 255, 255), width=4)
    for it in layout["decor"]:
        x, y, bw, bh = it["bbox"]
        d.rectangle([x, y, x + bw, y + bh], outline=(255, 200, 0, 255), width=3)
    img.save(str(out_png), quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=DEFAULT_IMG)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    src = COMFY_INPUT / args.img
    if not src.exists():
        print(f"[FATAL] {src}"); sys.exit(1)

    bgr = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(f"[img] {src.name} {w}x{h}")

    layout = {"source_image": str(src), "canvas": {"w": w, "h": h}}
    layout["palette"] = extract_palette(rgb)
    print("[palette]", [(p["hex"], f"{p['pct']}%") for p in layout["palette"][:5]])

    em = radial_profile_circle(rgb)
    cx, cy, ro, ri = em
    layout["emblem"] = {"center": [cx, cy], "r_outer": ro, "r_inner": ri}
    print(f"[emblem] center=({cx},{cy}) r_out={ro} r_in={ri}")

    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = ring_mask(rgb.shape, cx, cy, ri, ro, pad=10)

    subj = find_subject(gray, dist, ri)
    layout["subject"] = subj
    if subj:
        print(f"[subject] bbox={subj['bbox']} center={subj['center']} area={subj['area_px']} fill={subj['fill_ratio']}")

    excl = ring.copy()
    if subj:
        x, y, bw, bh = subj["bbox"]
        excl[y - 12:y + bh + 12, x - 12:x + bw + 12] = True
    bands = find_text_bands(gray, excl)
    layout["text_bands"] = bands
    print(f"[text] {len(bands)} 条:")
    for i, b in enumerate(bands):
        print(f"   #{i} y={b['y0']}~{b['y1']} cy={b['center_y']} x={b['x0']}~{b['x1']} "
              f"h={b['height_px']} w={b['width_px']}")

    dec = find_decor(gray, excl)
    layout["decor"] = dec
    print(f"[decor] {len(dec)} 个 (top5): {[it['bbox'] for it in dec[:5]]}")

    hist = {"v253_BAT_BBOX": [522, 518, 508, 456], "v253_BIG": [776, 1070],
            "v259_main": [776, 800], "v260_main": [776, 829], "v253_SUB": [776, 1285],
            "v259_sub": [776, 1261], "v260_sub": [776, 1261]}
    layout["_history_crosscheck"] = hist
    if subj:
        sy, sh = subj["bbox"][1], subj["bbox"][3]
        print("[crosscheck] 主体实测 y=", sy, "~", sy + sh)
        for k in ("v259_main", "v260_main", "v253_BIG"):
            inside = sy <= hist[k][1] <= sy + sh
            print(f"   {k} Y={hist[k][1]} -> {'落在主体身上(错)' if inside else '主体外(OK)'}")

    (JOB / "layout.json").write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] layout.json -> {JOB / 'layout.json'}")
    if args.debug:
        dump_debug(rgb, layout, JOB / "_debug_layout.png")
        print(f"[OK] debug -> {JOB / '_debug_layout.png'}")


if __name__ == "__main__":
    main()
