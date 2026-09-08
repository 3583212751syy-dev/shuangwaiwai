#!/usr/bin/env python
# v310 fixup v3: 直接 crop 三块矩形 paste（接受 bbox 内背景被替换）
# - AI 出图强制灰度（去颜色偏暖）
# - src 真字 + 真狗牌直接矩形贴回 → 100% 不混入 AI 乱字
# - bbox 重叠 OK（PIL paste 时"后盖前"）
import sys
from PIL import Image
import numpy as np

def force_gray(arr):
    mx = arr.max(2); mn = arr.min(2); g = (mx + mn) / 2
    return np.stack([g, g, g], axis=2).astype(np.float32)

def inpaint_box(arr, mask, iters=10):
    m = mask.astype(bool)
    out = arr.copy()
    for _ in range(iters):
        avg = np.zeros_like(out); w = np.zeros(out.shape[:2], np.float32)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            avg += np.roll(out, (dy, dx), axis=(0,1)); w += 1
        avg /= w[..., None]
        out[m] = avg[m]
    return out

def main():
    if len(sys.argv) < 4:
        print("usage: v310_fixup.py <src> <ai> <out>")
        return
    src_path, ai_path, out_path = sys.argv[1:4]

    src = Image.open(src_path).convert("RGB")
    ai = Image.open(ai_path).convert("RGB")
    work_w, work_h = 768, 1024
    SX, SY = work_w / 1556, work_h / 2000
    src_w = src.resize((work_w, work_h), Image.LANCZOS)
    ai_w = ai.resize((work_w, work_h), Image.LANCZOS)
    s = np.asarray(src_w, np.float32)
    a = np.asarray(ai_w, np.float32)
    h, w = s.shape[:2]

    # AI 强制灰度
    a_gray = force_gray(a)

    # bbox（基于 work 768x1024） — 已按 src 实际 y 分段校正
    big_text = (232, 410, int(330*SX), int(1240*SX))
    dogtag = (400, 920, int(660*SX), int(900*SX))

    # 在 AI 上抹除两个区
    combined = np.zeros((h,w), bool)
    for y0,y1,x0,x1 in [big_text, dogtag]:
        combined[y0:y1, x0:x1] = True
    a_clean = inpaint_box(a_gray, combined, iters=12)
    print(f"[inpaint] AI region removed px: {combined.sum()}")

    # 硬贴：直接 crop 矩形 paste（无 mask，最暴力最稳）
    out = Image.fromarray(np.clip(a_clean, 0, 255).astype(np.uint8))
    src_img = Image.fromarray(np.clip(s, 0, 255).astype(np.uint8))
    for box in [big_text, dogtag]:
        y0,y1,x0,x1 = box
        region = src_img.crop((x0, y0, x1, y1))
        out.paste(region, (x0, y0))

    # 输出（原尺寸）
    out = out.resize((src.size[0], src.size[1]), Image.LANCZOS)
    out.save(out_path, "JPEG", quality=92)
    print("[OK] saved:", out_path)

if __name__ == "__main__":
    main()
