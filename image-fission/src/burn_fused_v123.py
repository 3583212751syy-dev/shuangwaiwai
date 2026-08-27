"""v123 字体融合烧字：阴影 + 末端 mask 渐变 + 颜色贴主体
让 PIL 烧字看起来是图的一部分，不是后期硬贴：
1. 阴影层（向下右偏移 + 半透明黑）→ 字有立体感
2. 末端 mask 渐变（字底部 15% 像素 alpha 渐隐）→ 字末端融入主体
3. 颜色 / 描边匹配画面主色（米白/红/金黄等）
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

# v123 跑完目录
JOBS = "E:/Desktop/双接口/image-fission/jobs/"
SRC = JOBS + sorted([d for d in os.listdir(JOBS) if d.startswith("smoke_v123")])[-1]
OUT = "C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v123"
Path(OUT).mkdir(parents=True, exist_ok=True)
CROP = 8

# 字体烧字 + 融合参数
BURNS = [
    {"src": f"{SRC}/eagle_2.jpg", "name": "eagle_2",
     "word": "FERAL", "font": "pirata_one", "size_ratio": 0.085,
     "fill": (235, 225, 200, 255), "stroke": (15, 15, 25, 255),
     "stroke_w": 4, "y_ratio": 0.10, "spacing": 6,
     "shadow_off": 6, "shadow_alpha": 140, "fade_ratio": 0.15},
    {"src": f"{SRC}/denim_3.jpg", "name": "denim_3",
     "word": "UPCY", "font": "rye", "size_ratio": 0.085,
     "fill": (50, 60, 90, 255), "stroke": (255, 255, 255, 255),
     "stroke_w": 5, "y_ratio": 0.10, "spacing": 8,
     "shadow_off": 6, "shadow_alpha": 140, "fade_ratio": 0.15},
    {"src": f"{SRC}/skull_5.jpg", "name": "skull_5",
     "word": "VENOM", "font": "pirata_one", "size_ratio": 0.085,
     "fill": (220, 35, 45, 255), "stroke": (8, 8, 12, 255),
     "stroke_w": 5, "y_ratio": 0.10, "spacing": 6,
     "shadow_off": 6, "shadow_alpha": 140, "fade_ratio": 0.15},
    {"src": f"{SRC}/metal_6.jpg", "name": "metal_6",
     "word": "THRASH", "font": "metal_mania", "size_ratio": 0.09,
     "fill": (242, 205, 105, 255), "stroke": (12, 8, 4, 255),
     "stroke_w": 5, "y_ratio": 0.10, "spacing": 4,
     "shadow_off": 6, "shadow_alpha": 140, "fade_ratio": 0.15},
]


def burn_fused(cfg):
    im = Image.open(cfg["src"]).convert("RGBA")
    w, h = im.size
    font_size = int(h * cfg["size_ratio"])
    font = ImageFont.truetype(FONTS[cfg["font"]], font_size)
    text = cfg["word"]
    spacing = cfg.get("spacing", 6)

    widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in text]
    total_w = sum(widths) + spacing * (len(text) - 1)
    x = (w - total_w) // 2
    y = int(h * cfg["y_ratio"])

    # 1. 阴影层（向下右偏移 + 半透明）
    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sh_off = cfg.get("shadow_off", 6)
    sh_alpha = cfg.get("shadow_alpha", 140)
    sx = x + sh_off; sy = y + sh_off
    cur_x = sx
    for ch in text:
        cw = font.getbbox(ch)[2] - font.getbbox(ch)[0]
        sd.text((cur_x, sy), ch, font=font,
                fill=(0, 0, 0, sh_alpha), stroke_width=cfg["stroke_w"]//2 + 1,
                stroke_fill=(0, 0, 0, min(sh_alpha + 40, 255)))
        cur_x += cw + spacing

    # 2. 主字层
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cur_x = x
    for ch in text:
        cw = font.getbbox(ch)[2] - font.getbbox(ch)[0]
        draw.text((cur_x, y), ch, font=font,
                  fill=cfg["fill"], stroke_width=cfg["stroke_w"],
                  stroke_fill=cfg["stroke"])
        cur_x += cw + spacing

    # 3. 主字层末端 mask 渐变（让字底部融进主体）
    arr = np.array(layer)
    fade_start = int(h * (1 - cfg.get("fade_ratio", 0.15)))
    if fade_start < arr.shape[0]:
        for i in range(fade_start, arr.shape[0]):
            a = max(0, int(255 * (1 - (i - fade_start) / max(1, arr.shape[0] - fade_start))))
            arr[i, :, 3] = np.minimum(arr[i, :, 3], a)
    layer = Image.fromarray(arr)

    # 4. 合成（阴影在下 + 原图 + 字在上）
    out = Image.alpha_composite(im, shadow)
    out = Image.alpha_composite(out, layer).convert("RGB")
    out = out.crop((CROP, CROP, w - CROP, h - CROP))
    out.save(f"{OUT}/{cfg['name']}_final.jpg", "JPEG", quality=92)
    print(f"  OK {cfg['name']} word={text} font={cfg['font']} size={out.size}")


def no_text(name, src_name):
    im = Image.open(f"{SRC}/{src_name}").convert("RGB")
    w, h = im.size
    im = im.crop((CROP, CROP, w - CROP, h - CROP))
    im.save(f"{OUT}/{name}_final.jpg", "JPEG", quality=92)
    print(f"  OK {name} (无字) size={im.size}")


def main():
    no_text("illust_1", "illust_1.jpg")
    no_text("camo_4", "camo_4.jpg")
    for cfg in BURNS:
        burn_fused(cfg)


if __name__ == "__main__":
    main()