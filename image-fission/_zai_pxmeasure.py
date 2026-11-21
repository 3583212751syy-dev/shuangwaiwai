# -*- coding: utf-8 -*-
"""程序化像素实测：给 VLM 拆解 spec 补精确数值（色板/占比/亮条坐标/文字连通块/纹理粗糙度）。
语义交给 VLM，像素交给这里。OpenCV 读不了中文路径，统一用 PIL 读。

用法:
  python _zai_pxmeasure.py --img <原图> --out <json>
  python _zai_pxmeasure.py --img <原图> --out <json> --dump-strips <可视化目录>
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw

def load(path):
    im = Image.open(path).convert("RGB")
    return np.array(im)

def quantize_palette(a, k=8, sample=4):
    """KMeans 风格主色聚类（用 cv2 若可用，否则 numpy 简易 k-means）。返回 [(rgb, share_pct)]。"""
    small = a[::sample, ::sample].reshape(-1, 3).astype(np.float32)
    try:
        import cv2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0)
        _, label, center = cv2.kmeans(small, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
        cnt = np.bincount(label.ravel(), minlength=k)
    except Exception:
        # 纯 numpy k-means
        rng = np.random.default_rng(0)
        center = small[rng.choice(len(small), k, replace=False)].copy()
        for _ in range(30):
            d = ((small[:, None, :] - center[None, :, :]) ** 2).sum(2)
            label = d.argmin(1)
            center = np.array([small[label == i].mean(0) if (label == i).any() else center[i]
                               for i in range(k)])
        cnt = np.bincount(label, minlength=k)
    order = cnt.argsort()[::-1]
    total = cnt.sum()
    return [([int(round(v)) for v in center[i]], round(float(cnt[i]) / total * 100, 2))
            for i in order]

def strip_runs(a, axis=1, bright_thr=170, min_len=3, dark=False):
    """扫描每行/列的高亮（或暗）像素计数，找亮条/横带/边框连续 run。"""
    if axis == 1:  # 按行扫 → 返回 y 区间
        line = (a.max(2) >= bright_thr).sum(1) if not dark else (a.max(2) <= bright_thr).sum(1)
    else:
        line = (a.max(2) >= bright_thr).sum(0) if not dark else (a.max(2) <= bright_thr).sum(0)
    thr = line.max() * 0.35
    mask = line >= thr
    runs, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_len:
                runs.append([int(s), int(i - 1), round(float(line[s:i].max()), 1)])
            s = None
    if s is not None and len(mask) - s >= min_len:
        runs.append([int(s), int(len(mask) - 1), round(float(line[s:].max()), 1)])
    return runs

def text_blobs(a, bright=True, thr=200, min_area=40, max_area=20000):
    """文字/亮元素连通块检测，返回 bbox（供排版层定位参考）。"""
    try:
        import cv2
        g = a.max(2) if bright else (255 - a.min(2))
        bw = (g > thr).astype(np.uint8) * 255
        n, lab, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if min_area <= area <= max_area and w <= a.shape[1] * 0.5 and h <= a.shape[0] * 0.35:
                out.append({"bbox": [int(x), int(y), int(x+w), int(y+h)],
                            "area": int(area), "center": [round(float(cents[i][0]),1), round(float(cents[i][1]),1)]})
        out.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return out[:200]
    except Exception as e:
        return [{"error": str(e)}]

def texture_rough(a):
    """纹理粗糙度：全图/中心区梯度能量与局部标准差，判断平滑 vs 颗粒。"""
    g = a.astype(np.float32).mean(2)
    gy, gx = np.gradient(g)
    grad = np.sqrt(gx*gx + gy*gy)
    h, w = g.shape
    cen = g[h//4:3*h//4, w//4:3*w//4]
    return {"grad_mean": round(float(grad.mean()), 2),
            "grad_p90": round(float(np.percentile(grad, 90)), 2),
            "center_std": round(float(cen.std()), 2),
            "roughness": "high" if grad.mean() > 12 else ("medium" if grad.mean() > 5 else "low")}

def border_margin(a, bg_rgb, tol=30):
    """估算主体四周留白：从四边向内扫，找连续偏离底色的距离。"""
    bg = np.array(bg_rgb, dtype=np.int16)
    d = np.abs(a.astype(np.int16) - bg).max(2)
    fg = d > tol
    h, w = fg.shape
    def first_true(line):
        idx = np.where(line)[0]
        return int(idx[0]) if len(idx) else -1
    top = next((y for y in range(h) if fg[y].sum() > w*0.02), -1)
    bot = next((y for y in range(h-1, -1, -1) if fg[y].sum() > w*0.02), -1)
    left = next((x for x in range(w) if fg[:, x].sum() > h*0.02), -1)
    right = next((x for x in range(w-1, -1, -1) if fg[:, x].sum() > h*0.02), -1)
    return {"top_px": top, "bottom_px": h-1-bot, "left_px": left, "right_px": w-1-right,
            "canvas": [w, h]}

def dump_strips(a, runs, outpath):
    im = Image.fromarray(a).copy()
    d = ImageDraw.Draw(im)
    for y0, y1, _ in runs:
        d.rectangle([0, y0, a.shape[1], y1], outline=(0, 255, 0), width=3)
    im.save(outpath)

def measure(img, dump_dir=""):
    a = load(img)
    pal = quantize_palette(a)
    bg = pal[0][0]
    res = {
        "image": os.path.basename(img),
        "canvas": {"w": a.shape[1], "h": a.shape[0]},
        "palette_measured": [{"rgb": rgb, "share_pct": sh,
                              "hex": "#%02X%02X%02X" % tuple(rgb)} for rgb, sh in pal],
        "background_rgb_measured": bg,
        "bright_row_runs": strip_runs(a, axis=1, bright_thr=170),
        "bright_col_runs": strip_runs(a, axis=0, bright_thr=170),
        "dark_row_runs": strip_runs(a, axis=1, dark=True),
        "text_blobs": text_blobs(a),
        "texture": texture_rough(a),
        "margin": border_margin(a, bg),
    }
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        dump_strips(a, res["bright_row_runs"], os.path.join(dump_dir, "bright_rows.png"))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump-strips", default="")
    a = ap.parse_args()
    res = measure(a.img, a.dump_strips)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)

if __name__ == "__main__":
    main()
