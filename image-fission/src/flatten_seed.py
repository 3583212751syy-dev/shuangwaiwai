"""把真实照片/实物参考图预处理成「平涂色块」风格，减少材质纹理对 IPAdapter 的干扰。
用于 denim 等原图本身是面料/实物照片的情况，让裂变结果更接近平面印花图案。
可选：模糊/抹去顶部文字区域，避免生成图复制原图中的可读文字/商标。
"""
from PIL import Image, ImageFilter, ImageEnhance
import sys


def flatten(path, out_path, size=1024, colors=8, contrast=1.4, median=5, mask_top=0.0):
    img = Image.open(path).convert("RGB")
    # 统一尺寸，避免过大参考图拖慢
    img = img.resize((size, size), Image.LANCZOS)
    # 中值滤波抹掉面料纹理/噪点
    if median:
        img = img.filter(ImageFilter.MedianFilter(size=median))
    # 增强对比，让色块更分明
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    # 量化到有限颜色，得到平涂感，再转回 RGB 以便保存 JPEG
    img = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")

    # 可选：用背景色覆盖顶部文字区域，防止 ControlNet/Canny 把原文字复制过去
    if mask_top and 0 < mask_top < 1:
        w, h = img.size
        # 取底部背景平均色（通常最干净）
        bg = img.crop((0, h - 20, w, h)).resize((1, 1), Image.LANCZOS).getpixel((0, 0))
        overlay = Image.new("RGB", (w, int(h * mask_top)), bg)
        img.paste(overlay, (0, 0))

    img.save(out_path, quality=95)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: flatten_seed.py <input.jpg> <output.jpg> [colors] [contrast] [mask_top_ratio]")
        sys.exit(1)
    flatten(
        sys.argv[1],
        sys.argv[2],
        colors=int(sys.argv[3]) if len(sys.argv) > 3 else 8,
        contrast=float(sys.argv[4]) if len(sys.argv) > 4 else 1.4,
        mask_top=float(sys.argv[5]) if len(sys.argv) > 5 else 0.0,
    )
    print(f"flattened -> {sys.argv[2]}")
