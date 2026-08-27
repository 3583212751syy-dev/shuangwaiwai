"""PIL 后期烧字：AI 拼字不稳，改用 PIL 后期叠字（匹配画面字体风格）。
规则（用户 2026-08-27 确立）：原图字体没侵权就按原图文本烧；侵权换有意义短词。

用法：python src/burn_text.py
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

# 字体映射（Windows C:/Windows/Fonts/）
FONTS = {
    "denim_impact":   r"C:/Windows/Fonts/impact.ttf",          # 牛仔粗挤压
    "antique_bold":   r"C:/Windows/Fonts/ANTQUAB.TTF",          # 古体粗衬线 / 哥特
    "agency_bold":    r"C:/Windows/Fonts/AGENCYB.TTF",          # 金属感 / 死亡金属
    "alger":          r"C:/Windows/Fonts/ALGER.TTF",            # 装饰感
    "stencil":        r"C:/Windows/Fonts/STENCIL.TTF",          # 军风模板
}

# 烧字配置：(输入图, 输出名, 文本, 字体key, 字号, 颜色, 描边色, 位置(y_ratio), 描边宽)
# y_ratio: 0=顶/0.5=中/0.85=底
BURNS = [
    # denim_3: 原图 UPCY (牛仔粗体)
    {
        "in":  r"E:/Desktop/双接口/image-fission/jobs/smoke_v118_1787738617/denim_3.jpg",
        "out": r"C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v118/denim_3_text.jpg",
        "text": "UPCY", "font": "denim_impact", "size_ratio": 0.08,
        "fill": (50, 60, 90, 255),       # 靛蓝深色
        "stroke": (255, 255, 255, 255),  # 白色描边
        "stroke_w": 6, "y_ratio": 0.10, "spacing": 12, "letter_spacing": 1.15,
    },
    # eagle_2: 哥特占位 FERAL（等用户告知原图真词后改）
    {
        "in":  r"E:/Desktop/双接口/image-fission/jobs/smoke_v118_1787738617/eagle_2.jpg",
        "out": r"C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v118/eagle_2_text.jpg",
        "text": "FERAL", "font": "antique_bold", "size_ratio": 0.07,
        "fill": (240, 230, 210, 255),    # 米白
        "stroke": (20, 20, 30, 255),
        "stroke_w": 5, "y_ratio": 0.10, "spacing": 10, "letter_spacing": 1.20,
    },
    # skull_5: 哥特衬线占位 VENOM（等用户告知原图真词后改）
    {
        "in":  r"E:/Desktop/双接口/image-fission/jobs/smoke_v118_1787738617/skull_5.jpg",
        "out": r"C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v118/skull_5_text.jpg",
        "text": "VENOM", "font": "antique_bold", "size_ratio": 0.07,
        "fill": (220, 30, 40, 255),      # 红
        "stroke": (10, 10, 15, 255),
        "stroke_w": 6, "y_ratio": 0.10, "spacing": 10, "letter_spacing": 1.20,
    },
    # metal_6: 死亡金属 logo MRCHGSR（原图）
    {
        "in":  r"E:/Desktop/双接口/image-fission/jobs/smoke_v118_1787738617/metal_6.jpg",
        "out": r"C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v118/metal_6_text.jpg",
        "text": "MRCHGSR", "font": "agency_bold", "size_ratio": 0.075,
        "fill": (240, 200, 100, 255),    # 金黄
        "stroke": (15, 10, 5, 255),
        "stroke_w": 7, "y_ratio": 0.10, "spacing": 14, "letter_spacing": 1.18,
    },
]


def burn(cfg):
    im = Image.open(cfg["in"]).convert("RGBA")
    w, h = im.size
    font_size = int(h * cfg["size_ratio"])
    font = ImageFont.truetype(FONTS[cfg["font"]], font_size)
    text = cfg["text"]
    spacing = cfg.get("spacing", 8)

    # 测量每个字符宽度（letter_spacing 控制字符间距）
    char_widths = []
    for ch in text:
        bbox = font.getbbox(ch)
        char_widths.append(bbox[2] - bbox[0])
    total_w = sum(char_widths) + spacing * (len(text) - 1)
    spacing_extra = int(spacing * cfg.get("letter_spacing", 1.0))
    total_w += spacing_extra * (len(text) - 1)
    # 简化重算：每字符间距 = spacing
    total_w = sum(char_widths) + spacing * (len(text) - 1)

    # 创建文字层（透明）
    txt_layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)

    # 居中 x
    x_start = (w - total_w) // 2
    y = int(h * cfg["y_ratio"])

    cur_x = x_start
    for i, ch in enumerate(text):
        bbox = font.getbbox(ch)
        ch_w = bbox[2] - bbox[0]
        # 描边
        draw.text((cur_x, y), ch, font=font,
                  fill=cfg["fill"], stroke_width=cfg["stroke_w"], stroke_fill=cfg["stroke"])
        cur_x += ch_w + spacing

    # 合成到原图
    out = Image.alpha_composite(im, txt_layer).convert("RGB")
    out.save(cfg["out"], "JPEG", quality=92)
    print(f"  OK {cfg['out'].split('/')[-1]}  text={text}  font={cfg['font']}  size={im.size}")
    return out


def main():
    for cfg in BURNS:
        burn(cfg)


if __name__ == "__main__":
    main()