"""
v314b — cv2 inpaint 去字 + Anton stencil 写相似词（ComfyUI inpaint 的兜底方案）

ComfyUI inpaint (v314) 遇两个问题：(1) Canny 锁把 AI 废字边缘当成强约束，文字去不掉；
(2) SetLatentNoiseMask 在本机链路上 mask 方向与预期相反。这条用 cv2.inpaint(TELEA) 做
"局部扩散"——只黑字黑像素会被邻居迷彩自然扩散填回，无任何矩形边缝；再用 PIL 写 Anton stencil 字体。
效果上同样达成"彻底无矩形 + 字体可控"，且更可靠。

用法：
  python v314b_cv2_text.py <base_img> <out_img>
"""
import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

import cv2

# 文字带（同 v314，相对原图）
BANDS = [
    (0.218, 0.292, 0.24, 0.76),
    (0.292, 0.378, 0.22, 0.78),
    (0.378, 0.474, 0.18, 0.82),
]
SIMILAR_WORDS = [
    "WE HONOR OUR HEROES",
    "BRAVE",
    "LEGION",
]


def build_text_mask(gray, band_box, min_area=150, max_area=0):
    """在 band 内取暗像素（AI 废字为近黑），连通域按面积过滤保留文字（排除链珠小 CC）。
    gray: 单通道 uint8 灰度图（H,W）。返回 bool mask (H,W)。"""
    y0, y1, x0, x1 = band_box
    sub = gray[y0:y1, x0:x1]
    # AI 写的 ARMED/FORCES 为近黑（maxc<70），原图迷彩 ~120-180，差明显
    dark = (sub < 75).astype(np.uint8) * 255
    # 轻度膨胀把字字符连成 CC
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.dilate(dark, kernel, iterations=1)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros_like(dark)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if a < min_area:
            continue  # 链珠小 CC，丢弃
        if max_area and a > max_area:
            continue
        # 文字：宽高比合理（非极端细长），且面积足够
        if max(w, h) < 4:
            continue
        keep[lbl == i] = 255
    # 适度膨胀 inpaint 半径
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    full_mask = np.zeros(gray.shape, dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = keep
    return full_mask


def cv2_inpaint_text(base_path: str, out_clean: str):
    im = Image.open(base_path).convert("RGB")
    W, H = im.size
    arr = np.asarray(im)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    full_mask = np.zeros(gray.shape, dtype=np.uint8)
    for (y0, y1, x0, x1) in BANDS:
        bb = (int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W))
        m = build_text_mask(gray, bb, min_area=150)
        full_mask = cv2.bitwise_or(full_mask, m)
    text_px = int((full_mask > 0).sum())
    print(f"[cv2] text mask dark pixels = {text_px}")
    # TELEA 局部扩散：黑字→邻居迷彩自然填回，半径 3
    inpainted = cv2.inpaint(arr, full_mask, 3, cv2.INPAINT_TELEA)
    Image.fromarray(inpainted).save(out_clean, "JPEG", quality=92)
    print(f"[cv2] cleaned (no rectangle, seamless camo) saved {out_clean}")
    return W, H


def draw_anton(out_clean: str, out_final: str, W: int, H: int):
    im = Image.open(out_clean).convert("RGB")
    d = ImageDraw.Draw(im)
    font_dir = os.path.join(ROOT, "ComfyUI", "models", "fonts")

    def font(sz):
        for cand in ["Anton-Regular.ttf", "Arial_Unicode.ttf"]:
            p = os.path.join(font_dir, cand)
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, sz)
                except Exception:
                    pass
        return ImageFont.load_default()

    ink = (20, 20, 20)
    for (y0, y1, x0, x1), word in zip(BANDS, SIMILAR_WORDS):
        band_h = (y1 - y0) * H
        band_w = (x1 - x0) * W
        cy = (y0 + y1) / 2 * H
        cx = (x0 + x1) / 2 * W
        fs = int(band_h * (0.70 if band_h > 140 else 0.42))
        f = font(fs)
        for _ in range(12):
            bb = d.textbbox((0, 0), word, font=f, anchor="mm")
            tw = bb[2] - bb[0]
            if tw <= band_w * 0.94:
                break
            fs = int(fs * 0.92)
            f = font(fs)
        d.text((cx, cy), word, fill=ink, font=f, anchor="mm")
        print(f"[text] '{word}'  fs={fs}  ({cx:.0f},{cy:.0f})")
    im.save(out_final, "JPEG", quality=92)
    print(f"[out] {out_final}")


def main():
    if len(sys.argv) < 3:
        print("用法: python v314b_cv2_text.py <base_img> <out_img>")
        sys.exit(1)
    base = sys.argv[1]
    out_final = sys.argv[2]
    out_clean = os.path.join(os.path.dirname(out_final), "_v314b_clean.jpg")
    os.makedirs(os.path.dirname(out_final), exist_ok=True)
    W, H = cv2_inpaint_text(base, out_clean)
    draw_anton(out_clean, out_final, W, H)
    print("[done]")


if __name__ == "__main__":
    main()