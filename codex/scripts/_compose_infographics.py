from PIL import Image, ImageDraw, ImageFont
import os

BASE_DIR = "E:/Desktop/茶叶/成品_第七轮_站立图文修正"
PRODUCT_PATH = os.path.join(BASE_DIR, "_assets", "product_standing.png")

THEMES = {
    "nature": {
        "name": "自然有机风",
        "title_color": (90, 55, 35),
        "text_color": (75, 60, 50),
        "accent": (120, 150, 80),
        "accent2": (180, 120, 60),
        "bg_panel": (255, 252, 247, 200),
        "table_header": (120, 150, 80),
        "table_row1": (255, 252, 247, 180),
        "table_row2": (245, 240, 230, 180),
        "gold": (180, 140, 70),
    },
    "premium": {
        "name": "高端礼品风",
        "title_color": (240, 220, 180),
        "text_color": (235, 235, 235),
        "accent": (200, 170, 110),
        "accent2": (180, 120, 60),
        "bg_panel": (30, 25, 22, 200),
        "table_header": (160, 130, 80),
        "table_row1": (45, 40, 35, 180),
        "table_row2": (60, 53, 48, 180),
        "gold": (220, 190, 130),
    }
}

W, H = 1024, 1536

FONT_TITLE = "C:/Windows/Fonts/georgiab.ttf"
FONT_BODY = "C:/Windows/Fonts/arial.ttf"
FONT_BODY_BOLD = "C:/Windows/Fonts/arialbd.ttf"

def get_fonts():
    return {
        "h1": ImageFont.truetype(FONT_TITLE, 64),
        "h2": ImageFont.truetype(FONT_TITLE, 48),
        "h3": ImageFont.truetype(FONT_BODY_BOLD, 34),
        "body": ImageFont.truetype(FONT_BODY, 28),
        "body_bold": ImageFont.truetype(FONT_BODY_BOLD, 28),
        "table_header": ImageFont.truetype(FONT_BODY_BOLD, 26),
        "small": ImageFont.truetype(FONT_BODY, 24),
        "badge": ImageFont.truetype(FONT_BODY_BOLD, 26),
    }

FONTS = get_fonts()

def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def draw_rounded_rect(draw, xy, fill, radius=20):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill)

def draw_circle_icon(draw, cx, cy, r, fill, text, font, fg=(255,255,255)):
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill)
    tw, th = text_size(draw, text, font)
    draw.text((cx-tw//2, cy-th//2-2), text, font=font, fill=fg)

def draw_badge(draw, x, y, w, h, text, bg, fg, font):
    draw_rounded_rect(draw, (x, y, x+w, y+h), fill=bg, radius=16)
    tw, th = text_size(draw, text, font)
    draw.text((x+(w-tw)//2, y+(h-th)//2-2), text, font=font, fill=fg)

def paste_product(img, product, x, y, target_height):
    pw, ph = product.size
    scale = target_height / ph
    new_w = int(pw * scale)
    new_h = target_height
    prod = product.resize((new_w, new_h), Image.LANCZOS)
    img.paste(prod, (x, y), prod)

# ---------- 08 selling points ----------
def compose_selling(style, bg_path, out_path):
    theme = THEMES[style]
    img = Image.open(bg_path).convert("RGBA").resize((W, H), Image.LANCZOS)
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # Left panel for readability (wider)
    draw_rounded_rect(draw, (40, 60, 740, 1430), fill=theme["bg_panel"], radius=30)

    cx = 390
    y = 110
    draw.text((cx, y), "WHY CHOOSE", font=FONTS["h3"], fill=theme["accent"], anchor="mm")
    y += 55
    draw.text((cx, y), "TEVOYATEA Rooibos Tea", font=FONTS["h2"], fill=theme["title_color"], anchor="mm")
    y += 70
    draw.text((cx, y), "100% South African · Caffeine-Free · Calming", font=FONTS["body"], fill=theme["accent"], anchor="mm")

    points = [
        ("Naturally Caffeine-Free", "Enjoy any time of day."),
        ("Rich in Antioxidants", "With aspalathin for wellness."),
        ("100% South African Origin", "Sourced from Cederberg."),
        ("Naturally Sweet, No Sugar", "A hint of honey-like flavor."),
        ("Calming & Sleep-Friendly", "Your evening wind-down ritual."),
    ]

    y = 340
    for i, (title, desc) in enumerate(points):
        cx_icon, cy = 110, y + 32
        draw_circle_icon(draw, cx_icon, cy, 30, theme["accent"], str(i+1), FONTS["body_bold"])
        draw.text((165, y), title, font=FONTS["h3"], fill=theme["title_color"])
        draw.text((165, y+40), desc, font=FONTS["body"], fill=theme["text_color"])
        y += 120

    # Badges
    badge_y = 1310
    draw_badge(draw, 100, badge_y, 220, 58, "100% NATURAL", theme["accent"], (255,255,255), FONTS["badge"])
    draw_badge(draw, 360, badge_y, 220, 58, "SOUTH AFRICAN", theme["accent2"], (255,255,255), FONTS["badge"])

    img = Image.alpha_composite(img, overlay)
    # Product already in the background image; no extra paste to avoid white-fringe
    img.convert("RGB").save(out_path, quality=95)
    print("Saved", out_path)

# ---------- 09 brewing guide ----------
def compose_brewing(style, bg_path, out_path):
    theme = THEMES[style]
    img = Image.open(bg_path).convert("RGBA").resize((W, H), Image.LANCZOS)
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # Top title panel
    draw_rounded_rect(draw, (60, 70, 964, 260), fill=theme["bg_panel"], radius=30)
    draw.text((512, 130), "HOW TO BREW", font=FONTS["h2"], fill=theme["accent"], anchor="mm")
    draw.text((512, 200), "The Perfect Cup of Rooibos", font=FONTS["h1"], fill=theme["title_color"], anchor="mm")

    steps = [
        ("01", "Boil Water", "Heat fresh water to 200-212°F / 90-100°C."),
        ("02", "Add Tea", "Use 1-2 tsp loose leaf or 1 tea bag per cup."),
        ("03", "Steep", "Let it steep for 5-7 minutes."),
        ("04", "Enjoy", "Sip hot, or chill over ice with honey & lemon."),
    ]

    y = 330
    for num, title, desc in steps:
        draw_rounded_rect(draw, (90, y, 934, y+170), fill=theme["bg_panel"], radius=24)
        draw_circle_icon(draw, 165, y+85, 46, theme["accent"], num, FONTS["h3"])
        draw.text((240, y+35), title, font=FONTS["h3"], fill=theme["title_color"])
        draw.text((240, y+85), desc, font=FONTS["body"], fill=theme["text_color"])
        y += 205

    # Pro tip
    tip_y = 1280
    draw_rounded_rect(draw, (90, tip_y, 934, tip_y+110), fill=theme["accent"], radius=24)
    draw.text((512, tip_y+35), "PRO TIP", font=FONTS["h3"], fill=(255,255,255), anchor="mm")
    draw.text((512, tip_y+78), "Rooibos won't turn bitter if you steep a little longer.", font=FONTS["body"], fill=(255,255,255), anchor="mm")

    img = Image.alpha_composite(img, overlay)
    img.convert("RGB").save(out_path, quality=95)
    print("Saved", out_path)

# ---------- 10 comparison ----------
def compose_comparison(style, bg_path, out_path):
    theme = THEMES[style]
    img = Image.open(bg_path).convert("RGBA").resize((W, H), Image.LANCZOS)
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # Title
    draw.text((512, 100), "Rooibos vs Green & Black Tea", font=FONTS["h1"], fill=theme["title_color"], anchor="mm")
    draw.text((512, 175), "Why TEVOYATEA Rooibos is the smarter daily choice", font=FONTS["body"], fill=theme["accent"], anchor="mm")

    rows = [
        ("Feature", "Rooibos", "Green/Black"),
        ("Caffeine", "0 mg", "25–50 mg"),
        ("Tannins", "Very Low", "High"),
        ("Best Time", "Any time", "Morning"),
        ("Sleep Friendly", "Yes", "Limited"),
        ("Naturally Sweet", "Yes", "No"),
    ]

    x0, y0 = 50, 250
    col_w = [280, 320, 320]
    row_h = 82
    header_h = 86

    # Header
    draw.rectangle([x0, y0, x0+sum(col_w), y0+header_h], fill=theme["table_header"])
    x = x0
    for j, cell in enumerate(rows[0]):
        tw, th = text_size(draw, cell, FONTS["table_header"])
        draw.text((x+col_w[j]//2, y0+header_h//2-th//2), cell, font=FONTS["table_header"], fill=(255,255,255), anchor="lm")
        x += col_w[j]

    y = y0 + header_h
    for i, row in enumerate(rows[1:]):
        fill = theme["table_row1"] if i % 2 == 0 else theme["table_row2"]
        draw.rectangle([x0, y, x0+sum(col_w), y+row_h], fill=fill)
        x = x0
        for j, cell in enumerate(row):
            font = FONTS["body_bold"] if j == 0 else FONTS["body"]
            tw, th = text_size(draw, cell, font)
            color = theme["title_color"] if style=="nature" else theme["text_color"]
            draw.text((x+col_w[j]//2, y+row_h//2-th//2), cell, font=font, fill=color, anchor="lm")
            x += col_w[j]
        y += row_h

    # Conclusion
    con_y = y + 80
    draw_rounded_rect(draw, (90, con_y, 934, con_y+140), fill=theme["accent"], radius=28)
    draw.text((512, con_y+40), "THE VERDICT", font=FONTS["h3"], fill=(255,255,255), anchor="mm")
    draw.text((512, con_y+95), "Naturally caffeine-free, low in tannins, and calm in every cup.", font=FONTS["body"], fill=(255,255,255), anchor="mm")

    img = Image.alpha_composite(img, overlay)
    img.convert("RGB").save(out_path, quality=95)
    print("Saved", out_path)

# ---------- run ----------
if __name__ == "__main__":
    nature_bg = {
        "selling": os.path.join(BASE_DIR, "自然有机风", "Clean_vertical_product_backgro_2026-07-21T07-26-33.png"),
        "brewing": os.path.join(BASE_DIR, "自然有机风", "Clean_vertical_background_for__2026-07-21T07-26-30.png"),
        "comparison": os.path.join(BASE_DIR, "自然有机风", "Clean_minimal_vertical_backgro_2026-07-21T07-26-31.png"),
    }
    premium_bg = {
        "selling": os.path.join(BASE_DIR, "高端礼品风", "Clean_vertical_product_backgro_2026-07-21T07-27-34.png"),
        "brewing": os.path.join(BASE_DIR, "高端礼品风", "Clean_vertical_background_for__2026-07-21T07-27-32.png"),
        "comparison": os.path.join(BASE_DIR, "高端礼品风", "Clean_minimal_vertical_backgro_2026-07-21T07-27-32.png"),
    }

    compose_selling("nature", nature_bg["selling"], os.path.join(BASE_DIR, "自然有机风", "08_selling_points.png"))
    compose_brewing("nature", nature_bg["brewing"], os.path.join(BASE_DIR, "自然有机风", "09_brewing_guide.png"))
    compose_comparison("nature", nature_bg["comparison"], os.path.join(BASE_DIR, "自然有机风", "10_comparison.png"))

    compose_selling("premium", premium_bg["selling"], os.path.join(BASE_DIR, "高端礼品风", "08_selling_points.png"))
    compose_brewing("premium", premium_bg["brewing"], os.path.join(BASE_DIR, "高端礼品风", "09_brewing_guide.png"))
    compose_comparison("premium", premium_bg["comparison"], os.path.join(BASE_DIR, "高端礼品风", "10_comparison.png"))
