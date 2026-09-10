# -*- coding: utf-8 -*-
"""
detect_text_lines.py -- 用 maxc<70 (v299 黑字判定) + 小 kernel 水平闭运算 + 按 cy 聚类
返回每条独立文字带的精确 bbox, 不合并不同行
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

def detect_lines(img_path, dark_thr=70, max_h_frac=0.20, gap_split=20):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    # v299 经验: max(axis=2) < thr -> 黑字
    dark = (arr.max(axis=2) < dark_thr).astype(np.uint8) * 255
    # 用小水平 kernel 让同字水平连接, 不连接不同行
    k_h = max(5, int(w * 0.005))
    k_v = max(2, int(h * 0.003))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, k_v))
    bw = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if hh < 15 or hh > h * max_h_frac or ww < 30:
            continue
        if area / max(1, ww * hh) < 0.18:
            continue
        cands.append((x, y, x + ww, y + hh, area))
    if not cands:
        return [], h, w
    # 按 cy 排序, 用 gap_split 拆行 (不合并)
    cands.sort(key=lambda b: (b[1] + b[3]) // 2)
    bands = []
    cur = list(cands[0])
    for b in cands[1:]:
        cur_cy = (cur[1] + cur[3]) // 2
        b_cy = (b[1] + b[3]) // 2
        cur_h = cur[3] - cur[1]
        b_h = b[3] - b[1]
        # 同一行: cy 差 < max(cur_h,b_h) // 2
        if abs(b_cy - cur_cy) <= max(cur_h, b_h) * 0.6:
            cur[0] = min(cur[0], b[0]); cur[1] = min(cur[1], b[1])
            cur[2] = max(cur[2], b[2]); cur[3] = max(cur[3], b[3])
        else:
            bands.append(tuple(cur)); cur = list(b)
    bands.append(tuple(cur))
    out = []
    for (x1, y1, x2, y2, _) in bands:
        x1 = max(0, x1 - 8); y1 = max(0, y1 - 6)
        x2 = min(w, x2 + 8); y2 = min(h, y2 + 6)
        out.append((x1, y1, x2, y2))
    return out, h, w


if __name__ == "__main__":
    targets = [
        r"E:/Desktop/图裂变测试图/b78e60de8dfdf44acda99395326a7298.jpg",
        r"E:/迁移/Documents/My Pictures/Saved Pictures/歪歪.library/images/MTI8C3579DAMO.info/6978fabda2cc99629fa9e81f802762d3.jpg",
    ]
    for p in targets:
        bands, h, w = detect_lines(p)
        print(f"\n{p}  ({w}x{h})  -- {len(bands)} bands:")
        for i, b in enumerate(bands):
            print(f"  band{i}: x1={b[0]} y1={b[1]} x2={b[2]} y2={b[3]}  "
                  f"w={b[2]-b[0]}  h={b[3]-b[1]}  cx={(b[0]+b[2])//2}  cy={(b[1]+b[3])//2}")
        # 可视化
        img = Image.open(p).convert("RGB").copy()
        d = ImageDraw.Draw(img)
        for i, b in enumerate(bands):
            d.rectangle(b, outline=(0, 255, 0), width=4)
            d.text((b[0]+4, b[1]+4), f"band{i}", fill=(0, 255, 0))
        out_p = str(Path(p).with_suffix(".lines.png"))
        img.save(out_p)
        print(f"  -> {out_p}")
