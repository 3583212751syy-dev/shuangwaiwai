"""v124 字体融合烧字 v2：用原图真词 + 严格匹配原图颜色
用户反馈（2026-08-27）：「原图里也没有这个黄色字体」「字体跟原图有什么关系」
v124 修复：用原图真词
- eagle_2: JACKIE DIANNIES（PirataOne 哥特, 火焰灰白/橘）
- skull_5: TRUE NEVER DIES（PirataOne 哥特, 红, 分两行）
- metal_6: MRCHGSR（MetalMania 死亡金属, 白色不是金黄!）
- denim_3: UPCY（Rye 牛仔, 靛蓝）
- illust_1 / camo_4 无字
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import os

FONTS = {
    "metal_mania": r"E:/Desktop/双接口/image-fission/fonts/MetalMania-Regular.ttf",
    "pirata_one":  r"E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf",
    "rye":         r"E:/Desktop/双接口/image-fission/fonts/Rye-Regular.ttf",
}

JOBS = "E:/Desktop/双接口/image-fission/jobs/"
SRC = JOBS + sorted([d for d in os.listdir(JOBS) if d.startswith("smoke_v124")])[-1]
OUT = "C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v124"
Path(OUT).mkdir(parents=True, exist_ok=True)
CROP = 8

# 字体烧字 + 融合参数（用原图真词）
BURNS = [
    {"src": f"{SRC}/eagle_2.jpg", "name": "eagle_2",
     "word": "JACKE DIANNIES", "font": "pirata_one", "size_ratio": 0.075,
     "fill": (245, 230, 215, 255), "stroke": (20, 12, 8, 255),
     "stroke_w": 4, "y_ratio": 0.13, "spacing": 5,
     "shadow_off": 5, "shadow_alpha": 130, "fade_ratio": 0.18},
    {"src": f"{SRC}/denim_3.jpg", "name": "denim_3",
     "word": "UPCY", "font": "rye", "size_ratio": 0.085,
     "fill": (50, 60, 90, 255), "stroke": (255, 255, 255, 255),
     "stroke_w": 5, "y_ratio": 0.10, "spacing": 8,
     "shadow_off": 6, "shadow_alpha": 140, "fade_ratio": 0.15},
    {"src": f"{SRC}/skull_5.jpg", "name": "skull_5",
     "word": "TRUE\nNEVER DIES", "font": "pirata_one", "size_ratio": 0.065,
     "fill": (215, 30, 40, 255), "stroke": (8, 8, 12, 255),
     "stroke_w": 4, "y_ratio": 0.05, "spacing": 5,
     "shadow_off": 5, "shadow_alpha": 130, "fade_ratio": 0.18,
     "multi_line": True},
    {"src": f"{SRC}/metal_6.jpg", "name": "metal_6",
     "word": "MRCHGSR", "font": "metal_mania", "size_ratio": 0.09,
     "fill": (250, 250, 250, 255), "stroke": (10, 10, 10, 255),
     "stroke_w": 4, "y_ratio": 0.05, "spacing": 4,
     "shadow_off": 5, "shadow_alpha": 140, "fade_ratio": 0.18},
]


def burn_fused(cfg):
    im = Image.open(cfg["src"]).convert("RGBA")
    w, h = im.size
    font_size = int(h * cfg["size_ratio"])
    font = ImageFont.truetype(FONTS[cfg["font"]], font_size)
    spacing = cfg.get("spacing", 6)
    multi_line = cfg.get("multi_line", False)

    text = cfg["word"]
    if multi_line:
        lines = text.split("\n")
    else:
        lines = [text]

    # 计算总高度（多行间距 1.3 倍行高）
    line_height = int(font_size * 1.3)
    total_h = line_height * len(lines) + (len(lines) - 1) * int(font_size * 0.2)

    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    draw = ImageDraw.Draw(layer)

    sh_off = cfg.get("shadow_off", 6)
    sh_alpha = cfg.get("shadow_alpha", 140)

    start_y = int(h * cfg["y_ratio"])

    for i, line in enumerate(lines):
        widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in line]
        total_w = sum(widths) + spacing * (len(line) - 1)
        x = (w - total_w) // 2
        y = start_y + i * line_height

        cur_x = x
        for ch in line:
            cw = font.getbbox(ch)[2] - font.getbbox(ch)[0]
            # 阴影
            sd.text((cur_x + sh_off, y + sh_off), ch, font=font,
                    fill=(0, 0, 0, sh_alpha),
                    stroke_width=max(1, cfg["stroke_w"] // 2),
                    stroke_fill=(0, 0, 0, min(sh_alpha + 40, 255)))
            # 主字
            draw.text((cur_x, y), ch, font=font,
                      fill=cfg["fill"], stroke_width=cfg["stroke_w"],
                      stroke_fill=cfg["stroke"])
            cur_x += cw + spacing

    # 主字层末端 mask 渐变
    arr = np.array(layer)
    fade_start = int(h * (1 - cfg.get("fade_ratio", 0.15)))
    if fade_start < arr.shape[0]:
        for i in range(fade_start, arr.shape[0]):
            a = max(0, int(255 * (1 - (i - fade_start) / max(1, arr.shape[0] - fade_start))))
            arr[i, :, 3] = np.minimum(arr[i, :, 3], a)
    layer = Image.fromarray(arr)

    out = Image.alpha_composite(im, shadow)
    out = Image.alpha_composite(out, layer).convert("RGB")
    out = out.crop((CROP, CROP, w - CROP, h - CROP))
    out.save(f"{OUT}/{cfg['name']}_final.jpg", "JPEG", quality=92)
    print(f"  OK {cfg['name']} word={text!r} font={cfg['font']} size={out.size}")


def no_text(name):
    im = Image.open(f"{SRC}/{name}.jpg").convert("RGB")
    w, h = im.size
    im = im.crop((CROP, CROP, w - CROP, h - CROP))
    im.save(f"{OUT}/{name}_final.jpg", "JPEG", quality=92)
    print(f"  OK {name} (无字) size={im.size}")


def main():
    no_text("illust_1")
    no_text("camo_4")
    for cfg in BURNS:
        burn_fused(cfg)


if __name__ == "__main__":
    main()