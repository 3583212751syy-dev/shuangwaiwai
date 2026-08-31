#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR 识别 skull_5 原图文字：PIL 读图 + numpy 传参，规避 OpenCV 中文路径 bug"""
import numpy as np
from PIL import Image
import easyocr

IMG = "E:/Desktop/双接口/image-fission/ComfyUI/input/pinterest_skull_5.jpg"
reader = easyocr.Reader(['en'], gpu=True)


def ocr_array(arr, label):
    print(f"=== {label} ===")
    for (bbox, text, conf) in reader.readtext(arr, detail=1):
        print(f"  {conf:.3f}  {repr(text)}")


# 原图
im = Image.open(IMG).convert("RGB")
ocr_array(np.array(im), "1) 直接 OCR 原图")

# 预处理：黑底红字 -> 白字黑底高对比
arr = np.array(im)
mask = (arr[:, :, 0] > 90) & (arr[:, :, 0] - arr[:, :, 1] > 40) & (arr[:, :, 0] - arr[:, :, 2] > 40)
binary = np.where(mask, 255, 0).astype("uint8")
bin_img = Image.fromarray(binary).convert("RGB")
ocr_array(np.array(bin_img), "2) 红字掩码预处理 OCR")

# 放大 2x
big = bin_img.resize((bin_img.width * 2, bin_img.height * 2), Image.LANCZOS)
ocr_array(np.array(big), "3) 预处理+2x 放大 OCR")

# 反色版（白底红字 -> 直接红字变亮）也试一次
inv = Image.fromarray(255 - arr).convert("RGB")
ocr_array(np.array(inv), "4) 反色 OCR")
