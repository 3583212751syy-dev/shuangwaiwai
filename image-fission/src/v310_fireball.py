#!/usr/bin/env python
# v310 fixup for FIREBALL: 5 区 paste
import sys
from PIL import Image
import numpy as np

def force_gray_preserving_luma(arr):
    """对 FIREBALL 不去饱和（保留金+红），但仍是 RGB 形式。FIREBALL 本就是彩色"""
    return arr.copy()  # 直接保留彩色

def inpaint_box(arr, mask, iters=10):
    m = mask.astype(bool); out = arr.copy()
    for _ in range(iters):
        avg = np.zeros_like(out); w = np.zeros(out.shape[:2], np.float32)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            avg += np.roll(out, (dy, dx), axis=(0,1)); w += 1
        avg /= w[..., None]
        out[m] = avg[m]
    return out

def main():
    if len(sys.argv) < 4:
        print("usage: v310_fireball.py <src> <ai> <out>")
        return
    src_path, ai_path, out_path = sys.argv[1:4]
    src = Image.open(src_path).convert("RGB")
    ai = Image.open(ai_path).convert("RGB")
    work_w, work_h = 768, 1024
    sw = src.resize((work_w, work_h), Image.LANCZOS)
    aw = ai.resize((work_w, work_h), Image.LANCZOS)
    s = np.asarray(sw, np.float32); a = np.asarray(aw, np.float32)
    h, w = s.shape[:2]

    SX, SY = work_w/1556, work_h/2000
    # FIREBALL 五区（基于 src 1556x2000 实际坐标）
    # 顶红字 "FIREBALL WHISKY": y 50-180, x 800-1450
    top_red = (int(50*SY), int(180*SY), int(800*SX), int(1450*SX))
    # 骷髅头主体: y 130-1450, x 50-1080（中央偏左）
    skull = (int(130*SY), int(1450*SY), int(50*SX), int(1080*SX))
    # 右侧竖排金字 "FIREBALL": y 250-1750, x 1100-1500
    right_gold = (int(250*SY), int(1750*SY), int(1100*SX), int(1500*SX))
    # 顶部小徽章 (红魔鬼+FIREBALL): y 660-960, x 1100-1450
    top_badge = (int(660*SY), int(960*SY), int(1100*SX), int(1450*SX))

    combined = np.zeros((h,w), bool)
    for y0,y1,x0,x1 in [top_red, skull, right_gold, top_badge]:
        combined[y0:y1, x0:x1] = True
    # 不 inpaint（颜色版），AI 内容略缩后直接 paste
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    src_img = Image.fromarray(np.clip(s, 0, 255).astype(np.uint8))
    for box in [top_red, skull, right_gold, top_badge]:
        y0,y1,x0,x1 = box
        region = src_img.crop((x0, y0, x1, y1))
        out.paste(region, (x0, y0))
    out = out.resize((src.size[0], src.size[1]), Image.LANCZOS)
    out.save(out_path, "JPEG", quality=92)
    print("[OK] saved:", out_path)

if __name__ == "__main__":
    main()
