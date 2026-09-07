# -*- coding: utf-8 -*-
"""v306 diagnosis: locate plate fill errors and residue clusters."""
import os
import numpy as np
import cv2
from PIL import Image

JOBS = r"E:\Desktop\双接口\image-fission\jobs\v306"
SRC = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"

arr = np.array(Image.open(SRC).convert("RGB"))[:, :, ::-1].copy()  # BGR
H, W = arr.shape[:2]
print("src size:", W, "x", H)

plate = np.array(Image.open(os.path.join(JOBS, "_chk", "plate.png")).convert("RGB"))[:, :, ::-1].copy()

# 1) plate vs arr：修改区里的"过亮/过暗"修复错误
pb = plate.astype(np.int16).sum(axis=2)
ab = arr.astype(np.int16).sum(axis=2)
err_bright = ((pb - ab) > 70).astype(np.uint8) * 255   # 修复后比原图亮很多
err_dark = ((ab - pb) > 70).astype(np.uint8) * 255

def top_cc(mask, tag, n=8):
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    order = np.argsort(-st[1:, 4])[:n]
    print(f"  [{tag}] total={int((mask>0).sum())}")
    for oi in order:
        i = 1 + int(oi)
        x, y, w_, h_, a = st[i]
        if a < 60:
            break
        print(f"    bbox=({x},{y} {w_}x{h_}) area={a}")

print("plate too-bright patches:")
top_cc(err_bright, "bright")
print("plate too-dark patches:")
top_cc(err_dark, "dark")

# 2) spread final vs plate 残差（coverage 外的意外改动）
fin = np.array(Image.open(os.path.join(JOBS, "v306_spread_body.png")).convert("RGB"))[:, :, ::-1].copy()
if fin.shape[:2] != (H, W):
    print("spread final size differs:", fin.shape[:2], "skip resid")
else:
    diff = np.abs(fin.astype(np.int16) - plate.astype(np.int16)).max(axis=2)
    resid = (diff > 30).astype(np.uint8) * 255
    print("spread final vs plate residue:")
    top_cc(resid, "resid", n=12)
