#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v170: v169f 失败根因修复。
- 不二次 mask 提取（避免 MaxFilter 切粗描边丢失 R 右腿/I 封口/D 弧底/S 装饰点）。
- 不缩放（LANCZOS 会让笔画变细）。
- 每行字符按列投影精确切分（非均匀切分），保留真实字宽。
- 每行源字符 → 同行优先；跨行借字母（S 仅在 DIES）。
- 字符 RGBA：bbox 内 red+black_mask，黑色用 red_dilated 限制在紧邻描边范围内，
  避免字内黑边蔓延到主体（蛇/血/玫瑰）区域。
"""
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy import ndimage
import easyocr

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs")
ORIG = ROOT / "ComfyUI" / "input" / "pinterest_skull_5.jpg"
BASE = ROOT / "jobs" / "smoke_v164" / "v164_skull_5.jpg"
OUT = DESK / "image-fission-v170-skull_5-RUST-RUINS-DUST.jpg"
OUT_COMPARE = DESK / "image-fission-v170-skull_5-compare.jpg"

TARGET = {"TRUE": "RUST", "NEVER": "RUINS", "DIES": "DUST"}


def find_char_xranges(red_row_mask, min_gap=6):
    """垂直投影：在行红字 mask 上找字符水平边界（精确切分）。"""
    col_sum = red_row_mask.sum(axis=0)
    W = len(col_sum)
    blocks = []
    in_block = False
    start = 0
    for i, v in enumerate(col_sum):
        if v > 0 and not in_block:
            start = i
            in_block = True
        elif v == 0 and in_block:
            blocks.append((start, i))
            in_block = False
    if in_block:
        blocks.append((start, W))
    if not blocks:
        return []
    merged = [blocks[0]]
    for b in blocks[1:]:
        if b[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    # 若切分数量少于预期，fallback 用均匀切分（在外部处理）
    return merged


def char_rgba(orig_arr, bbox):
    """对单字符 bbox 取红字+紧邻黑边 RGBA。"""
    x0, y0, x1, y1 = bbox
    seg = orig_arr[y0:y1, x0:x1]
    R, G, B = seg[:, :, 0], seg[:, :, 1], seg[:, :, 2]
    red = (R > 100) & (R - G > 30) & (R - B > 30)
    # 把红色 mask 扩张到包含全部黑边（外环黑色描边）
    red_dil = ndimage.binary_dilation(red, iterations=6)
    # 黑边 = 黑色像素 且 在 red_dil 附近 12px 内（不蔓延到主体）
    black = (R < 50) & (G < 50) & (B < 50)
    black_close = ndimage.binary_dilation(black, iterations=2)  # 标记黑边附近
    # 限定：只在 red_dil 扩张区内接受 black
    accept_zone = ndimage.binary_dilation(red, iterations=14)
    edge_mask = (red | (black & accept_zone)).astype("uint8") * 255
    # 二次精修：去掉 ~1px 散点
    edge_mask = ndimage.binary_closing(edge_mask > 128, iterations=1).astype("uint8") * 255

    rgba = np.zeros((*seg.shape[:2], 4), dtype="uint8")
    rgba[edge_mask > 0, :3] = seg[edge_mask > 0]
    rgba[edge_mask > 0, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def main():
    base = Image.open(BASE).convert("RGB")
    W, H = base.size
    print(f"[base] {W}x{H}")

    orig_pil = Image.open(ORIG).convert("RGB")
    W0, H0 = orig_pil.size
    print(f"[orig] {W0}x{H0}")
    arr = np.array(orig_pil)

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    red = (R > 100) & (R - G > 30) & (R - B > 30)

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    res = reader.readtext(arr)
    rows = []
    for bbox, text, conf in res:
        t = text.upper().strip()
        if t in TARGET:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            rows.append({
                "text": t,
                "bbox": (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                "conf": conf,
            })
    rows.sort(key=lambda r: r["bbox"][1])
    print("[rows]", [(r["text"], r["bbox"]) for r in rows])

    # === 1. 精确切分每行字符 bbox（按列投影） ===
    char_map = {}  # ch -> list of (x0,y0,x1,y1)
    for r in rows:
        rx0, ry0, rx1, ry1 = r["bbox"]
        row_red = red[ry0:ry1, rx0:rx1]
        xranges = find_char_xranges(row_red, min_gap=8)
        n = len(r["text"])
        if len(xranges) < n:
            # fallback 均匀切分
            bw = (rx1 - rx0) / n
            xranges = [(int(rx0 + i * bw), int(rx0 + (i + 1) * bw)) for i in range(n)]
        # 取前 n 个块（投影过细则合并前面的）
        if len(xranges) > n:
            merged = []
            i = 0
            for k in range(n):
                if k == n - 1:
                    merged.append((xranges[i][0], xranges[-1][1]))
                else:
                    target_count = (len(xranges) - i) - (n - 1 - k)
                    if target_count > 1:
                        end = xranges[i + target_count - 1][1]
                        merged.append((xranges[i][0], end))
                        i += target_count
                    else:
                        merged.append(xranges[i])
                        i += 1
            xranges = merged
        for i, ch in enumerate(r["text"]):
            x0, x1 = xranges[i]
            char_map.setdefault(ch, []).append((x0, ry0, x1, ry1))
        print(f"  [{r['text']}] xranges={xranges}")

    print(f"[char_map keys] {sorted(char_map.keys())}")

    # === 2. 预生成每个字符的 RGBA（在不同行可能多份） ===
    char_rgba_pool = {}
    for ch, bboxes in char_map.items():
        rgba_list = []
        for bb in bboxes:
            rgba = char_rgba(arr, bb)
            rgba_list.append((bb, rgba))
        char_rgba_pool[ch] = rgba_list

    # === 3. 拼装三行 target 文字（不缩放、不二次 mask） ===
    text_layer = Image.new("RGBA", (W0, H0), (0, 0, 0, 0))

    for r in rows:
        src_text = r["text"]
        tgt_text = TARGET[src_text]
        rx0, ry0, rx1, ry1 = r["bbox"]
        row_w = rx1 - rx0
        row_h = ry1 - ry0
        n_tgt = len(tgt_text)
        slot_w = row_w / n_tgt

        for i, ch in enumerate(tgt_text):
            if ch not in char_rgba_pool:
                print(f"  [MISS] {ch} in {tgt_text}")
                continue
            # 优先用同行字符（用 src_text 在 [i] 的字符 if exists），否则取 char_map[ch] 第一份
            prefer = None
            src_chars = list(src_text)
            same_pos = src_chars[i] if i < len(src_chars) else None
            if same_pos == ch:
                # 用 src_text[i] 的 bbox（同行）
                idx = src_chars.index(ch, i, i + 1) if ch in src_chars[i:i + 1] else i
                if 0 <= idx < len(src_chars):
                    prefer = src_chars.index(ch, i, i + 1)
                    # 直接用 src row r 的 xranges[i] 字符
                    row_red = red[ry0:ry1, rx0:rx1]
                    xranges = find_char_xranges(row_red, min_gap=8)
                    if len(xranges) >= i + 1:
                        x0, x1 = xranges[i]
                        same_row_bbox = (x0, ry0, x1, ry1)
                        rgba = char_rgba(arr, same_row_bbox)
                        ch_h = ry1 - ry0
                        ch_w = x1 - x0
                        paste_x = int(rx0 + i * slot_w + (slot_w - ch_w) / 2)
                        paste_y = ry0 + (row_h - ch_h) // 2
                        text_layer.paste(rgba, (paste_x, paste_y), rgba)
                        print(f"  [{tgt_text}[{i}]={ch}] same-row paste @x={paste_x} w={ch_w}")
                        continue
            # 跨行 / 同行但位置不匹配：从 char_rgba_pool 取第一份
            bb, rgba = char_rgba_pool[ch][0]
            cw, ch_h = rgba.size
            paste_x = int(rx0 + i * slot_w + (slot_w - cw) / 2)
            paste_y = ry0 + (row_h - ch_h) // 2
            text_layer.paste(rgba, (paste_x, paste_y), rgba)
            print(f"  [{tgt_text}[{i}]={ch}] cross-row paste @x={paste_x} w={cw} h={ch_h}")

    # === 4. 整体 resize 到 base 尺寸 + 贴回 v164 底图 ===
    text_big = text_layer.resize((W, H), Image.NEAREST if False else Image.LANCZOS)
    base_rgba = base.convert("RGBA")
    # 锐化以补偿 LANCZOS 模糊（v169e 没锐化，导致糊；这里加 USM）
    text_big = text_big.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))
    base_rgba.paste(text_big, (0, 0), text_big)
    out_im = base_rgba.convert("RGB")
    out_im.save(OUT, "JPEG", quality=92, optimize=True)
    print(f"[ok] {OUT.name}")

    # === 五联对照：原图 | v164底图 | v169e | v169f | v170 ===
    v169e = Image.open(DESK / "image-fission-v169e-skull_5-orig-text-clean.jpg").convert("RGB")
    v169f = Image.open(DESK / "image-fission-v169f-skull_5-RUST-RUINS-DUST.jpg").convert("RGB")
    H_show = 1450
    gap = 18

    def fit(im):
        w = int(im.width * H_show / im.height)
        return im.resize((w, H_show), Image.LANCZOS)

    panels = [
        (fit(orig_pil), "原图 TRUE/NEVER/DIES"),
        (fit(base), "v164 裂变底图 (无字)"),
        (fit(v169e), "v169e 原词保留"),
        (fit(v169f), "v169f 裂变 4/10 (糊字缺字)"),
        (fit(out_im), "v170 严格约束裂变 (待评)"),
    ]
    total_w = sum(p.width for p, _ in panels) + gap * (len(panels) - 1)
    header_h = 70
    canvas = Image.new("RGB", (total_w, H_show + header_h), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    x = 0
    for p, lb in panels:
        canvas.paste(p, (x, header_h))
        d.text((x + 10, 22), lb, fill=(255, 235, 200))
        x += p.width + gap
    canvas.save(OUT_COMPARE, "JPEG", quality=86, optimize=True)
    print(f"[compare] {OUT_COMPARE.name}")


if __name__ == "__main__":
    from PIL import ImageFilter
    main()
