"""v122 后期处理：裁边去杂 + 字体融合烧字 v2
用户反馈（2026-08-27）：
- 一律不要边框 → 四边各裁 8px 去边缘杂色
- 字体按参考图+效果图画风融合（不突兀）→ 新下载画风字体：
  MetalMania（死亡金属）/ PirataOne（哥特）/ Rye（西部牛仔复古）
  全部 Google Fonts OFL 免费商用授权

用法：python src/post_v122.py
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

FONTS = {
    "metal_mania": r"E:/Desktop/双接口/image-fission/fonts/MetalMania-Regular.ttf",
    "pirata_one":  r"E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf",
    "rye":         r"E:/Desktop/双接口/image-fission/fonts/Rye-Regular.ttf",
}

SRC = "E:/Desktop/双接口/image-fission/jobs/" + \
      sorted([d for d in __import__('os').listdir("E:/Desktop/双接口/image-fission/jobs") if d.startswith("smoke_v122")])[-1]
OUT = "C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v122"
Path(OUT).mkdir(parents=True, exist_ok=True)

CROP = 8  # 每边裁 8px 去杂边

# 烧字配置：按画风融合选字体
BURNS = [
    {   # 西部牛仔复古风 → Rye（跟 denim 的 vintage 感融合）
        "in": f"{SRC}/denim_3_w055.jpg", "out": f"{OUT}/denim_3_final.jpg",
        "text": "UPCY", "font": "rye", "size_ratio": 0.085,
        "fill": (50, 60, 90, 255), "stroke": (255, 255, 255, 255),
        "stroke_w": 5, "y_ratio": 0.10, "spacing": 8,
    },
    {   # 哥特徽章 → PirataOne（跟 eagle 哥特风融合）
        "in": f"{SRC}/eagle_2_w070.jpg", "out": f"{OUT}/eagle_2_final.jpg",
        "text": "FERAL", "font": "pirata_one", "size_ratio": 0.085,
        "fill": (235, 225, 200, 255), "stroke": (15, 15, 25, 255),
        "stroke_w": 4, "y_ratio": 0.10, "spacing": 6,
    },
    {   # 哥特骷髅 → PirataOne
        "in": f"{SRC}/skull_5_w070.jpg", "out": f"{OUT}/skull_5_final.jpg",
        "text": "VENOM", "font": "pirata_one", "size_ratio": 0.085,
        "fill": (220, 35, 45, 255), "stroke": (8, 8, 12, 255),
        "stroke_w": 5, "y_ratio": 0.10, "spacing": 6,
    },
    {   # 死亡金属 logo → MetalMania（跟 metal 风融合）
        "in": f"{SRC}/metal_6_w070.jpg", "out": f"{OUT}/metal_6_final.jpg",
        "text": "THRASH", "font": "metal_mania", "size_ratio": 0.09,
        "fill": (242, 205, 105, 255), "stroke": (12, 8, 4, 255),
        "stroke_w": 5, "y_ratio": 0.10, "spacing": 4,
    },
]


def burn(cfg):
    im = Image.open(cfg["in"]).convert("RGBA")
    w, h = im.size
    font_size = int(h * cfg["size_ratio"])
    font = ImageFont.truetype(FONTS[cfg["font"]], font_size)
    text = cfg["text"]
    spacing = cfg.get("spacing", 6)

    widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in text]
    total_w = sum(widths) + spacing * (len(text) - 1)

    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x = (w - total_w) // 2
    y = int(h * cfg["y_ratio"])
    for ch in text:
        cw = font.getbbox(ch)[2] - font.getbbox(ch)[0]
        draw.text((x, y), ch, font=font, fill=cfg["fill"],
                  stroke_width=cfg["stroke_w"], stroke_fill=cfg["stroke"])
        x += cw + spacing

    out = Image.alpha_composite(im, layer).convert("RGB")
    # 裁边去杂
    out = out.crop((CROP, CROP, w - CROP, h - CROP))
    out.save(cfg["out"], "JPEG", quality=92)
    print(f"  OK {cfg['out'].split('/')[-1]}  text={text} font={cfg['font']} size={out.size}")


def no_text(in_name, out_name):
    im = Image.open(f"{SRC}/{in_name}").convert("RGB")
    w, h = im.size
    im = im.crop((CROP, CROP, w - CROP, h - CROP))
    im.save(f"{OUT}/{out_name}", "JPEG", quality=92)
    print(f"  OK {out_name} (无字, 裁边) size={im.size}")


def main():
    no_text("illust_1_w055.jpg", "illust_1_final.jpg")
    no_text("camo_4_w075.jpg", "camo_4_final.jpg")
    for cfg in BURNS:
        burn(cfg)


if __name__ == "__main__":
    main()