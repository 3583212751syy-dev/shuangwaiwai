"""
fireball 烧字 v2 — 修正版：
- 单行版式（与原 FIREBALL 高度 134px → 裂变 343px 一致）
- 字体：MetalMania-Regular（金属摇滚风，最贴原图斜体火焰动感字）
- 词：SKULLFIRE（9字母匹配 FIREBALL 长度，与骷髅+火焰主题契合，不侵权）
- 位置：裂变图 (2169, 1306, 2872, 1649)，仅原 FIREBALL 文字区，不再覆盖下方骷髅
- 颜色：原图黑字 → 仍用黑字 #1c1a16（接近 OCR 检测到的文字色）
"""

import os, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont

JOB = r"E:/Desktop/双接口/image-fission/jobs/smoke_v185"
GEN = f"{JOB}/v185_fireball_skull_BAD.jpg"  # 用原始未烧字版（v186 会再跑新的 fireball）
OUT = f"{JOB}/v185_fireball_skull_burned_v2.jpg"
FONT = r"E:/Desktop/双接口/image-fission/fonts/MetalMania-Regular.ttf"
WORD = "SKULLFIRE"
COLOR = (28, 26, 22, 255)  # 近黑色

# 裂变图 3840x5120，原 FIREBALL 文字区（单行版式高度 343）
BOX = (2169, 1306, 2872, 1649)


def fit_font(text, font_path, max_w, max_h):
    lo, hi = 50, 600
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        lw = f.getlength(text)
        lh = f.size * 1.05
        if lw <= max_w and lh <= max_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ImageFont.truetype(font_path, best)


def main():
    src = np.array(Image.open(GEN).convert("RGB"))
    img = Image.fromarray(src).convert("RGBA")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = BOX
    max_w, max_h = x2 - x1, y2 - y1
    f = fit_font(WORD, FONT, max_w * 0.96, max_h * 0.92)
    lw = f.getlength(WORD)
    lh = f.size * 1.0
    lx = x1 + (max_w - lw) / 2
    ly = y1 + (max_h - lh) / 2 + max_h * 0.05  # 微调垂直居中
    draw.text((lx, ly), WORD, font=f, fill=COLOR)
    out = Image.alpha_composite(img, Image.new("RGBA", img.size, (0, 0, 0, 0))).convert("RGB")
    out.save(OUT, quality=95)
    sz = os.path.getsize(OUT)
    print(f"烧字完成 -> {OUT}  ({sz//1024} KB)")
    print(f"  word={WORD} font=MetalMania box={BOX} size={max_w}x{max_h} font_pt={f.size} actual_w={lw:.0f} actual_h={lh:.0f}")


if __name__ == "__main__":
    main()