"""PIL 后期烧字 v2：v121 终版 6 张上烧字（AI 生成时已禁字，防乱字）
规则（用户 2026-08-27 确立）：原图字体没侵权就按原图文本烧；侵权换有意义短词。
v121 终版文字：denim_3=UPCY / eagle_2=FERAL / skull_5=VENOM / metal_6=THRASH（有意义）

用法：python src/burn_text_v121.py
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

FONTS = {
    "denim_impact": r"C:/Windows/Fonts/impact.ttf",
    "antique_bold": r"C:/Windows/Fonts/ANTQUAB.TTF",
    "agency_bold":  r"C:/Windows/Fonts/AGENCYB.TTF",
}

SRC = r"E:/Desktop/双接口/image-fission/jobs/smoke_v121_1787796560"
OUT = r"C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v121"

BURNS = [
    {
        "in": f"{SRC}/denim_3_w045.jpg",
        "out": f"{OUT}/denim_3_final.jpg",
        "text": "UPCY", "font": "denim_impact", "size_ratio": 0.08,
        "fill": (50, 60, 90, 255), "stroke": (255, 255, 255, 255),
        "stroke_w": 6, "y_ratio": 0.10, "spacing": 12,
    },
    {
        "in": f"{SRC}/eagle_2_w070.jpg",
        "out": f"{OUT}/eagle_2_final.jpg",
        "text": "FERAL", "font": "antique_bold", "size_ratio": 0.07,
        "fill": (240, 230, 210, 255), "stroke": (20, 20, 30, 255),
        "stroke_w": 5, "y_ratio": 0.10, "spacing": 10,
    },
    {
        "in": f"{SRC}/skull_5_w070.jpg",
        "out": f"{OUT}/skull_5_final.jpg",
        "text": "VENOM", "font": "antique_bold", "size_ratio": 0.07,
        "fill": (220, 30, 40, 255), "stroke": (10, 10, 15, 255),
        "stroke_w": 6, "y_ratio": 0.10, "spacing": 10,
    },
    {
        "in": f"{SRC}/metal_6_w070.jpg",
        "out": f"{OUT}/metal_6_final.jpg",
        "text": "THRASH", "font": "agency_bold", "size_ratio": 0.075,
        "fill": (240, 200, 100, 255), "stroke": (15, 10, 5, 255),
        "stroke_w": 7, "y_ratio": 0.10, "spacing": 14,
    },
]


def burn(cfg):
    im = Image.open(cfg["in"]).convert("RGBA")
    w, h = im.size
    font_size = int(h * cfg["size_ratio"])
    font = ImageFont.truetype(FONTS[cfg["font"]], font_size)
    text = cfg["text"]
    spacing = cfg.get("spacing", 8)

    char_widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in text]
    total_w = sum(char_widths) + spacing * (len(text) - 1)

    txt_layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)

    x_start = (w - total_w) // 2
    y = int(h * cfg["y_ratio"])
    cur_x = x_start
    for ch in text:
        ch_w = font.getbbox(ch)[2] - font.getbbox(ch)[0]
        draw.text((cur_x, y), ch, font=font,
                  fill=cfg["fill"], stroke_width=cfg["stroke_w"], stroke_fill=cfg["stroke"])
        cur_x += ch_w + spacing

    out = Image.alpha_composite(im, txt_layer).convert("RGB")
    out.save(cfg["out"], "JPEG", quality=92)
    print(f"  OK {cfg['out'].split('/')[-1]}  text={text}  font={cfg['font']}  size={im.size}")


def main():
    for cfg in BURNS:
        burn(cfg)


if __name__ == "__main__":
    main()