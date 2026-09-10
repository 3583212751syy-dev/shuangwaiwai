# -*- coding: utf-8 -*-
"""
detect_text_bands.py -- 自动检测原图中的文字带位置 (返回 bbox 列表)
  策略: 灰度化 -> Otsu 二值化 -> 形态学闭运算连接同字符 -> 找连通域 ->
        按 cy 聚类成行 (相邻 cy 合并) -> 按面积/宽高比过滤掉非文字区域
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

def detect_bands(img_path, dark=True, gap_merge_px=30, min_band_h=18, max_band_h_frac=0.18):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Otsu 找主体阈值; dark=True 文字偏暗, dark=False 文字偏亮
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if dark:
        bw = 255 - bw   # 暗文字 -> 白mask
    # 形态学闭运算: 同字符水平连接
    k_h = max(3, int(w * 0.012))   # 字符间水平间距
    k_v = max(3, int(h * 0.008))   # 同一行内高度容差
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, k_v))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # 找连通域
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # 过滤: 高度在合理范围, 宽高比符合文字行
    cands = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if hh < min_band_h or hh > h * max_band_h_frac:
            continue
        if ww < 20:
            continue
        if area / max(1, ww * hh) < 0.20:
            continue
        cands.append((x, y, x + ww, y + hh, area))
    if not cands:
        return [], h, w
    # 按 cy 聚类 (gap_merge_px 内合并)
    cands.sort(key=lambda b: (b[1] + b[3]) // 2)
    bands = []
    cur = list(cands[0])
    for b in cands[1:]:
        cur_cy = (cur[1] + cur[3]) // 2
        b_cy = (b[1] + b[3]) // 2
        if abs(b_cy - cur_cy) <= max(cur[3] - cur[1], b[3] - b[1]) + gap_merge_px:
            cur[0] = min(cur[0], b[0]); cur[1] = min(cur[1], b[1])
            cur[2] = max(cur[2], b[2]); cur[3] = max(cur[3], b[3])
        else:
            bands.append(tuple(cur)); cur = list(b)
    bands.append(tuple(cur))
    # bbox 转为 (x1, y1, x2, y2) 且扩展 10px 边距
    out = []
    for (x1, y1, x2, y2, _) in bands:
        x1 = max(0, x1 - 8); y1 = max(0, y1 - 6)
        x2 = min(w, x2 + 8); y2 = min(h, y2 + 6)
        out.append((x1, y1, x2, y2))
    return out, h, w


if __name__ == "__main__":
    targets = [
        r"E:/Desktop/图裂变测试图/b78e60de8dfdf44acda99395326a7298.jpg",
        r"E:/Desktop/图裂变测试图/13c8b7bf8dae757e6c2d4b3d6a860f9d.jpg",
        r"E:/迁移/Documents/My Pictures/Saved Pictures/歪歪.library/images/MTI8C3579DAMO.info/6978fabda2cc99629fa9e81f802762d3.jpg",
    ]
    for p in targets:
        bands, h, w = detect_bands(p, dark=True)
        print(f"\n{p}  ({w}x{h})")
        print(f"  检测到 {len(bands)} 段文字带:")
        for i, b in enumerate(bands):
            print(f"    band{i}: bbox={b}  w={b[2]-b[1]}  h={b[3]-b[1]}  area={(b[2]-b[0])*(b[3]-b[1])}")
        # 画可视化
        from PIL import ImageDraw
        img = Image.open(p).convert("RGB").copy()
        d = ImageDraw.Draw(img)
        for i, b in enumerate(bands):
            d.rectangle(b, outline=(0, 255, 0), width=4)
            d.text((b[0]+4, b[1]+4), f"band{i}", fill=(0, 255, 0))
        out_p = str(Path(p).with_suffix(".bands.png"))
        img.save(out_p)
        print(f"  可视化 -> {out_p}")