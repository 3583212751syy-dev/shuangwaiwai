#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建真实正面+背面袋子素材（v3 — 简化奶油色方案）
"""
import numpy as np
from PIL import Image, ImageFilter
import cv2

OUT = r"E:/Desktop/茶叶/成品_第十七轮_Gemini即梦高质量/_assets"
JIMENG_FRONT_RGB = r"E:/Desktop/茶叶/成品_样稿_confirm/_jimeng_front.png"
REAL_LABEL = r"E:/Desktop/茶叶/成品_样稿_confirm/_label.png"
BACK_PHOTO = r"E:/Desktop/茶叶/成品_样稿_confirm/产品背面图片.png"

def grabcut_bag(img_rgb_path, rect=None):
    img = Image.open(img_rgb_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    if rect is None:
        rect = (int(w*0.08), int(h*0.02), int(w*0.88), int(h*0.98))
    mask = np.zeros((h,w), np.uint8)
    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    cv2.grabCut(arr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = ((mask==1) | (mask==3)).astype(np.uint8) * 255
    white = (arr[:,:,0]>235)&(arr[:,:,1]>235)&(arr[:,:,2]>235)
    fg[white] = 0
    fg_bin = (fg>128).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg_bin, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = 1 + int(np.argmax(areas))
        keep = (lbl == biggest)
        fg[~keep] = 0
    fg_im = Image.fromarray(fg, "L").filter(ImageFilter.GaussianBlur(1.8))
    fg = np.array(fg_im)
    rgba = np.zeros((h,w,4), np.uint8)
    rgba[:,:,:3] = arr
    rgba[:,:,3] = fg
    return Image.fromarray(rgba, "RGBA")

def crop_to_alpha(im_rgba, min_alpha=10):
    a = np.array(im_rgba.split()[3])
    ys, xs = np.where(a > min_alpha)
    if len(xs)==0: return im_rgba
    return im_rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)))

# ---------- 正面：grabCut即梦袋 + 真标签 ----------
def build_front():
    bag = crop_to_alpha(grabcut_bag(JIMENG_FRONT_RGB, rect=(140, 30, 900, 1000)))
    bw, bh = bag.size

    lbl = Image.open(REAL_LABEL).convert("RGBA")
    a2 = np.array(lbl.split()[3]); ys2, xs2 = np.where(a2>10)
    lb = (int(xs2.min()), int(ys2.min()), int(xs2.max()+1), int(ys2.max()+1))
    lbl_c = lbl.crop(lb)
    lw, lh = lbl_c.size

    target_w = int(bw * 0.54)
    target_h = int(lh * (target_w / lw))
    lbl_s = lbl_c.resize((target_w, target_h), Image.LANCZOS)
    px = (bw - target_w)//2
    py = (bh - target_h)//2 + int(bh*0.03)

    # 奶油色矩形：覆盖 AI 标签区域（包含底部 NET WT 残留）
    bag_arr = np.array(bag).copy()
    cream = (250, 244, 232, 255)
    fill_w = int(bw * 0.58)
    fill_h = int(bh * 0.70)  # 足够覆盖到 AI 标签底部（含 NET WT）
    fx = (bw - fill_w)//2
    fy = (bh - fill_h)//2 + int(bh * 0.02)
    for y in range(fy, min(bh, fy+fill_h)):
        for x in range(fx, min(bw, fx+fill_w)):
            if bag_arr[y,x,3] > 10:
                bag_arr[y,x] = cream
    bag2 = Image.fromarray(bag_arr, "RGBA")

    # 贴真标签（paste + mask 保险）
    bag2.paste(lbl_s, (px, py), mask=lbl_s.split()[3])

    out = Image.new("RGBA", bag2.size, (0,0,0,0))
    out.alpha_composite(bag2, (0,0))
    p = OUT + "/product_front_real.png"
    out.save(p, "PNG")
    print(f"✓ front {p} {out.size}")
    return p

# ---------- 背面：grabCut ----------
def build_back():
    b = crop_to_alpha(grabcut_bag(BACK_PHOTO, rect=(100, 30, 950, 1000)))
    p = OUT + "/product_back_real.png"
    b.save(p, "PNG")
    print(f"✓ back  {p} {b.size}")
    return p

if __name__ == "__main__":
    build_front()
    build_back()