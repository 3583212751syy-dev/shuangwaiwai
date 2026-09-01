#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v169f: 原图文字「内容裂变」——保留原图字体美术形态（1:1 像素级），只换字母拼写。
       TRUE/NEVER/DIES -> RUST/RUINS/DUST（全由原图字母集 {T,R,U,E,N,V,D,I,S} 组成）。
       步骤：OCR 定位三行 -> 垂直投影切分单字母 -> 用原图字母像素逐位替换 -> 放大贴回 v164 底图。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import easyocr

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs")
ORIG = ROOT / "ComfyUI" / "input" / "pinterest_skull_5.jpg"
BASE = ROOT / "jobs" / "smoke_v164" / "v164_skull_5.jpg"
OUT = DESK / "image-fission-v169f-skull_5-RUST-RUINS-DUST.jpg"
OUT_COMPARE = DESK / "image-fission-v169f-skull_5-compare.jpg"

TARGET = {"TRUE": "RUST", "NEVER": "RUINS", "DIES": "DUST"}


def split_letters(col_sum, W, min_gap=8):
    """垂直投影切分字母，合并 <min_gap 的字母内间隙。"""
    raw = []
    i = 0
    while i < W:
        if col_sum[i] > 0:
            j = i
            while j < W and col_sum[j] > 0:
                j += 1
            raw.append((i, j))
            i = j
        else:
            i += 1
    if not raw:
        return raw
    merged = [raw[0]]
    for b in raw[1:]:
        prev = merged[-1]
        if b[0] - prev[1] < min_gap:
            merged[-1] = (prev[0], b[1])
        else:
            merged.append(b)
    return merged


def main():
    base = Image.open(BASE).convert("RGB")
    W, H = base.size
    print(f"[base] {W}x{H}")

    orig_pil = Image.open(ORIG).convert("RGB")
    W0, H0 = orig_pil.size
    print(f"[orig] {W0}x{H0}")
    arr = np.array(orig_pil)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    red = (R > 110) & (R - G > 50) & (R - B > 50)

    # OCR 定位三行
    reader = easyocr.Reader(["en"], gpu=True)
    res = reader.readtext(arr)
    words = []
    for bbox, text, conf in res:
        t = text.upper().strip()
        if t in TARGET:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            words.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), t))
    words.sort(key=lambda w: (w[1] + w[3]) / 2)  # 按 y 中心排序
    print(f"[ocr words] {[w[4] for w in words]}")

    # 收集原图字母模板（均匀切段 + 收窄到段中心 50% 避免邻字干扰）
    templates = {}
    for (x0, y0, x1, y1, text) in words:
        chars = list(text)
        n = len(chars)
        bw = (x1 - x0) / n
        for i, ch in enumerate(chars):
            cx0 = int(x0 + i * bw + bw * 0.25)
            cx1 = int(x0 + (i + 1) * bw - bw * 0.25)
            seg = orig_pil.crop((cx0, y0, cx1, y1))
            if ch not in templates:
                templates[ch] = seg
    print(f"[templates] {sorted(templates.keys())}")

    # 逐行拼目标词（用原图对应字母模板，居中贴且缩 85% 避免重叠）
    text_layer = Image.new("RGBA", (W0, H0), (0, 0, 0, 0))
    for (x0, y0, x1, y1, text) in words:
        target = TARGET[text]
        n = len(target)
        bw = (x1 - x0) / n
        for i, ch in enumerate(target):
            seg = templates[ch]
            gx0 = int(x0 + i * bw)
            gx1 = int(x0 + (i + 1) * bw)
            slot_w = gx1 - gx0
            slot_h = y1 - y0
            # 缩 85% 留间隙，居中
            paste_w = int(slot_w * 0.85)
            paste_h = int(slot_h * 0.92)
            seg_r = seg.resize((paste_w, paste_h), Image.LANCZOS)
            px = gx0 + (slot_w - paste_w) // 2
            py = y0 + (slot_h - paste_h) // 2
            ba = np.array(seg_r)
            bR, bG, bB = ba[:, :, 0], ba[:, :, 1], ba[:, :, 2]
            bred = (bR > 110) & (bR - bG > 50) & (bR - bB > 50)
            bred_img = Image.fromarray((bred * 255).astype("uint8"))
            bexp = np.array(bred_img.filter(ImageFilter.MaxFilter(5))) > 128
            bblack = (bR < 60) & (bG < 60) & (bB < 60)
            bmask = bred | (bexp & ~bred & bblack)
            seg_rgba = seg_r.convert("RGBA")
            seg_rgba.putalpha(Image.fromarray((bmask * 255).astype("uint8")))
            text_layer.paste(seg_rgba, (px, py), seg_rgba)
        print(f"  [{text}] -> {target}")

    # 放大到 base 尺寸并贴到 v164 底图
    text_big = text_layer.resize((W, H), Image.LANCZOS)
    base_rgba = base.convert("RGBA")
    base_rgba.paste(text_big, (0, 0), text_big)
    out_im = base_rgba.convert("RGB")
    out_im.save(OUT, "JPEG", quality=92, optimize=True)
    print(f"[ok] {OUT.name}")

    # 四联对照：原图 | v164底图 | v169e(原词) | v169f(裂变词)
    v169e = Image.open(DESK / "image-fission-v169e-skull_5-orig-text-clean.jpg").convert("RGB")
    H_show = 1450
    gap = 20

    def fit(im):
        w = int(im.width * H_show / im.height)
        return im.resize((w, H_show), Image.LANCZOS)

    panels = [
        (fit(orig_pil), "原图 TRUE/NEVER/DIES"),
        (fit(base), "v164 裂变底图 (禁词)"),
        (fit(v169e), "v169e 原词保留 TRUE/NEVER/DIES"),
        (fit(out_im), "v169f 文字裂变 RUST/RUINS/DUST"),
    ]
    total_w = sum(p.width for p, _ in panels) + gap * (len(panels) - 1)
    header_h = 70
    canvas = Image.new("RGB", (total_w, H_show + header_h), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    x = 0
    for p, lb in panels:
        canvas.paste(p, (x, header_h))
        d.text((x + 14, 22), lb, fill=(255, 235, 200))
        x += p.width + gap
    canvas.save(OUT_COMPARE, "JPEG", quality=86, optimize=True)
    print(f"[compare] {OUT_COMPARE.name}")


if __name__ == "__main__":
    main()
