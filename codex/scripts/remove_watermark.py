#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_watermark.py — 自动去除图片水印（LaMa 深度学习修复）

工作原理：
  1. 水印检测：自动定位「半透明白/灰色覆盖水印」（最常见的电商图水印类型）；
     也可手动指定矩形区域(--region)或提供黑白蒙版(--mask)。
  2. 图像修复：用 LaMa 模型把水印区域智能修补填回，尽量还原底层背景。

依赖安装：
  pip install simple-lama-inpainting opencv-python numpy pillow
  （simple-lama-inpainting 会自动带上 torch / torchvision）

首次运行会自动下载 LaMa 模型权重(~200MB)到用户缓存目录。

示例：
  # 自动检测 + 修复（最常用）
  python remove_watermark.py -i input.jpg -o output.png

  # 同时导出检测到的蒙版，便于检查/微调
  python remove_watermark.py -i input.jpg -o output.png --mask-out mask.png

  # 手动指定水印所在的矩形区域 x,y,w,h
  python remove_watermark.py -i input.jpg -o output.png --region 100,50,300,80

  # 用自定义蒙版（白色=需要去除修复的区域）
  python remove_watermark.py -i input.jpg -o output.png --mask my_mask.png

  # 批量处理整个目录
  python remove_watermark.py -i ./images -o ./out --batch

  # 调整自动检测灵敏度(0.0~1.0，越大越激进)
  python remove_watermark.py -i input.jpg -o output.png --sensitivity 0.7
"""

import argparse
import os
import sys
import glob

import numpy as np
import cv2
from PIL import Image


# --------------------------------------------------------------------------- #
# 水印蒙版检测
# --------------------------------------------------------------------------- #
def detect_watermark_mask(img_bgr, sensitivity=0.5, min_area=30, dilate=3):
    """
    自动检测半透明亮色覆盖水印，返回二值蒙版(255=水印区域)。

    思路：水印通常是叠在画面上的半透明白/灰文字或 logo，相对局部背景偏亮、
    且饱和度低。用「比局部背景亮 + 低饱和度」两个条件初筛，再做形态学清理。
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 局部背景估计：大半径高斯模糊，得到不含细小水印的背景趋势
    sigma = max(h, w) * 0.02 + 1.0
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    # 比局部背景亮多少
    diff = cv2.subtract(gray, bg)

    # 饱和度：白/灰水印饱和度低
    sat = hsv[:, :, 1]

    # 阈值：sensitivity 越大 -> 门槛越低 -> 检出越多
    t = int(10 + (1.0 - sensitivity) * 45)
    bright = diff > t
    low_sat = sat < 70
    mask = (bright & low_sat).astype(np.uint8) * 255

    # 形态学：闭运算连接笔画，开运算去噪点
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # 去掉过小的连通块
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    # 膨胀一点，覆盖水印抗锯齿边缘
    if dilate > 0:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        clean = cv2.dilate(clean, k_dil, iterations=1)

    return clean


def region_to_mask(shape, region):
    """把 x,y,w,h 矩形转成蒙版。"""
    x, y, w, h = region
    mask = np.zeros(shape[:2], dtype=np.uint8)
    H, W = shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    mask[y0:y1, x0:x1] = 255
    return mask


def load_mask(path, shape):
    """读取外部蒙版图，缩放对齐到目标尺寸，二值化。"""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"读取蒙版失败：{path}")
    H, W = shape[:2]
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


# --------------------------------------------------------------------------- #
# LaMa 修复
# --------------------------------------------------------------------------- #
_LAMA = None


def get_lama():
    """惰性加载 LaMa 模型（只在真正需要修复时加载，避免无谓下载）。"""
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        print("正在加载 LaMa 模型（首次会下载权重，请稍候）…", flush=True)
        _LAMA = SimpleLama()
    return _LAMA


def inpaint_with_lama(img_bgr, mask_u8):
    """
    用 LaMa 修复。返回 BGR 结果。
    img_bgr: HxWx3 BGR uint8
    mask_u8: HxW uint8, 255=待修复
    """
    lama = get_lama()
    # LaMa 接受 RGB PIL 图 + 单通道 mask PIL 图
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    mask_pil = Image.fromarray(mask_u8).convert("L")

    # 若蒙版为空，直接返回原图
    if mask_pil.getbbox() is None:
        print("提示：未检测到水印区域，跳过修复。")
        return img_bgr

    result_pil = lama(img_pil, mask_pil)
    result_rgb = np.array(result_pil)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def process_one(in_path, out_path, args):
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[跳过] 无法读取图片：{in_path}")
        return False

    # 1) 取蒙版
    if args.mask:
        mask = load_mask(args.mask, img.shape)
    elif args.region:
        mask = region_to_mask(img.shape, args.region)
    else:
        mask = detect_watermark_mask(
            img, sensitivity=args.sensitivity,
            min_area=args.min_area, dilate=args.dilate
        )

    # 导出蒙版供检查
    if args.mask_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.mask_out)), exist_ok=True)
        cv2.imwrite(args.mask_out, mask)
        print(f"蒙版已保存：{args.mask_out}")

    # 蒙版覆盖率提示
    ratio = float(np.count_nonzero(mask)) / (mask.shape[0] * mask.shape[1]) * 100
    print(f"  检测到水印占比：{ratio:.1f}%")

    # 2) LaMa 修复
    result = inpaint_with_lama(img, mask)

    # 3) 保存
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, result)
    print(f"  完成 → {out_path}")
    return True


def parse_region(s):
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--region 需要格式 x,y,w,h")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(
        description="自动去除图片水印（LaMa 深度学习修复）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python remove_watermark.py -i in.jpg -o out.png",
    )
    ap.add_argument("-i", "--input", required=True, help="输入图片或目录（--batch 时）")
    ap.add_argument("-o", "--output", required=True, help="输出图片或目录（--batch 时）")
    ap.add_argument("--batch", action="store_true", help="批量处理整个目录")
    ap.add_argument("--region", type=parse_region, help="手动指定水印矩形 x,y,w,h")
    ap.add_argument("--mask", help="自定义黑白蒙版（白=待去除区域）")
    ap.add_argument("--mask-out", help="把自动检测到的蒙版导出到此路径")
    ap.add_argument("--sensitivity", type=float, default=0.5,
                    help="自动检测灵敏度 0.0~1.0，越大越激进（默认0.5）")
    ap.add_argument("--min-area", type=int, default=30, help="蒙版最小连通块像素（默认30）")
    ap.add_argument("--dilate", type=int, default=3, help="蒙版向外膨胀像素（默认3）")
    args = ap.parse_args()

    if args.batch:
        files = [f for f in glob.glob(os.path.join(args.input, "*")) if f.lower().endswith(IMG_EXTS)]
        if not files:
            print(f"目录内没有图片：{args.input}")
            sys.exit(1)
        os.makedirs(args.output, exist_ok=True)
        ok = 0
        for f in files:
            name = os.path.basename(f)
            out_f = os.path.join(args.output, os.path.splitext(name)[0] + ".png")
            print(f"[{name}]")
            if process_one(f, out_f, args):
                ok += 1
        print(f"\n批量完成：{ok}/{len(files)} 张")
    else:
        if not os.path.isfile(args.input):
            print(f"输入文件不存在：{args.input}")
            sys.exit(1)
        process_one(args.input, args.output, args)


if __name__ == "__main__":
    main()
