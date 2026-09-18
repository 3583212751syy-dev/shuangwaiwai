"""Compose front + back pouch scene with product intro text overlay.

Layout follows two-pouch reference (front hero left, back smaller right-back)
plus typography in the style of 主图介绍 references (torn-paper card headline
+ 4 pill badges bottom).
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

BASE = "E:/Desktop/茶叶/成品_第十五轮_两包场景加文字排版"
ASSETS = os.path.join(BASE, "_assets")
OUT = os.path.join(BASE, "01_two_pouches_intro.png")
SIZE = 1024

# ---------- helpers ----------
def load_font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\georgiab.ttf" if bold else r"C:\Windows\Fonts\georgia.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def text_w(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]

def text_h(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[3] - bbox[1]

def draw_pill(draw, x, y, w, h, fill, outline=None, radius=None):
    if radius is None:
        radius = h // 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=outline)

# ---------- 1. background ----------
bg_path = os.path.join(ASSETS, "Premium_e_commerce_product_pho_2026-07-22T05-18-49.png")
bg = Image.open(bg_path).convert("RGBA")
bg = bg.resize((SIZE, SIZE), Image.LANCZOS)
# Crop the AI watermark (bottom-right "图片由AI生成")
bg = bg.crop((0, 0, SIZE, SIZE - 40))
bg = bg.resize((SIZE, SIZE), Image.LANCZOS)

# ---------- 2. pouch composites ----------
front = Image.open(os.path.join(ASSETS, "..", "..", "成品_第十一轮_正背组合重制", "_assets", "product_front_label_fixed.png")).convert("RGBA")
back = Image.open(os.path.join(ASSETS, "..", "..", "成品_第九轮_真实站立包装", "_assets", "product_back_clean.png")).convert("RGBA")

# Drop shadow helper using PIL alpha
def make_shadow(mask_img, offset_y=14, blur=18, opacity=120):
    shadow = Image.new("RGBA", mask_img.size, (0, 0, 0, 0))
    alpha = mask_img.split()[3]
    shadow.putalpha(alpha)
    # Darken to grey-black
    black = Image.new("RGBA", mask_img.size, (30, 22, 14, 255))
    shadow = Image.composite(black, Image.new("RGBA", mask_img.size, (0, 0, 0, 0)), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    # Reduce alpha
    a = shadow.split()[3].point(lambda p: int(p * opacity / 255))
    shadow.putalpha(a)
    return shadow

# --- Back pouch (smaller, right-back) ---
back_w = 280
back_ratio = back.height / back.width
back_h = int(back_w * back_ratio)
back_resized = back.resize((back_w, back_h), Image.LANCZOS)
back_x, back_y = 720, 470
# shadow under back
back_shadow = make_shadow(back_resized, blur=14, opacity=110)
bg.paste(back_shadow, (back_x + 4, back_y + 18), back_shadow)
bg.paste(back_resized, (back_x, back_y), back_resized)

# --- Front pouch (hero, center-left) ---
front_w = 470
front_ratio = front.height / front.width
front_h = int(front_w * front_ratio)
front_resized = front.resize((front_w, front_h), Image.LANCZOS)
front_x, front_y = 130, 440
# shadow under front
front_shadow = make_shadow(front_resized, blur=22, opacity=140)
bg.paste(front_shadow, (front_x + 6, front_y + 24), front_shadow)
bg.paste(front_resized, (front_x, front_y), front_resized)

# ---------- 3. text overlay ----------
overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# --- Top-left torn-paper card with headline ---
# Card geometry
card_x, card_y, card_w, card_h = 60, 70, 540, 200
# Card shadow
shadow_layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
sd = ImageDraw.Draw(shadow_layer)
sd.rounded_rectangle([card_x + 8, card_y + 14, card_x + card_w + 8, card_y + card_h + 14],
                     radius=8, fill=(20, 14, 8, 90))
shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(12))

# Card body (cream paper)
card_layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
cd = ImageDraw.Draw(card_layer)
cd.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + card_h],
                     radius=6, fill=(248, 240, 224, 252),
                     outline=(178, 142, 90, 255), width=2)
# Composite card + shadow onto overlay
overlay = Image.alpha_composite(overlay, shadow_layer)
overlay = Image.alpha_composite(overlay, card_layer)
draw = ImageDraw.Draw(overlay)

# Headline
f_eyebrow = load_font(22, bold=False)
f_title1 = load_font(50, bold=True)
f_title2 = load_font(58, bold=True)
f_sub = load_font(22, bold=False)

eyebrow = "AUTHENTIC SOUTH AFRICAN"
t1 = "RED ROOIBOS"
t2 = "TEA"
sub = "Caffeine-Free Herbal Tea  ·  Cederberg Mountains"

ex = card_x + 30
ey = card_y + 26
draw.text((ex, ey), eyebrow, font=f_eyebrow, fill=(120, 80, 30, 255))
ty1 = ey + 32
draw.text((ex, ty1), t1, font=f_title1, fill=(78, 38, 18, 255))
# Title 2 to the right of title 1 if fits
tw1 = text_w(draw, t1, f_title1)
draw.text((ex + tw1 + 18, ty1 - 6), t2, font=f_title2, fill=(160, 60, 30, 255))
# Subtitle
sy = card_y + card_h - 42
draw.text((ex, sy), sub, font=f_sub, fill=(80, 55, 30, 255))

# --- Bottom 4 pill badges ---
badges = [
    ("100% NATURAL",        (76, 110, 60, 255),  (245, 248, 240, 255)),
    ("CAFFEINE FREE",       (170, 50, 40, 255),  (252, 244, 240, 255)),
    ("ANTIOXIDANT RICH",    (130, 80, 30, 255),  (250, 244, 230, 255)),
    ("GREAT HOT OR ICED",   (50, 95, 110, 255),  (240, 248, 250, 255)),
]
f_pill = load_font(22, bold=True)
pill_w, pill_h = 220, 56
pill_gap = 16
total_w = pill_w * 4 + pill_gap * 3
pill_x0 = (SIZE - total_w) // 2
pill_y = 820
# Pill background band (subtle warm)
band_layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
bd = ImageDraw.Draw(band_layer)
bd.rounded_rectangle([pill_x0 - 22, pill_y - 22, pill_x0 + total_w + 22, pill_y + pill_h + 22],
                     radius=36, fill=(252, 246, 232, 200),
                     outline=(180, 140, 90, 255), width=2)
overlay = Image.alpha_composite(overlay, band_layer)
draw = ImageDraw.Draw(overlay)

for i, (text, fg, bg_color) in enumerate(badges):
    px = pill_x0 + i * (pill_w + pill_gap)
    # Pill shadow
    sh = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    sd2 = ImageDraw.Draw(sh)
    sd2.rounded_rectangle([px + 3, pill_y + 5, px + pill_w + 3, pill_y + pill_h + 5],
                          radius=pill_h // 2, fill=(40, 25, 12, 80))
    sh = sh.filter(ImageFilter.GaussianBlur(6))
    overlay = Image.alpha_composite(overlay, sh)
    draw = ImageDraw.Draw(overlay)
    # Pill body
    draw.rounded_rectangle([px, pill_y, px + pill_w, pill_y + pill_h],
                           radius=pill_h // 2, fill=bg_color,
                           outline=fg, width=2)
    # Centered text
    tw = text_w(draw, text, f_pill)
    th = text_h(draw, text, f_pill)
    draw.text((px + (pill_w - tw) // 2, pill_y + (pill_h - th) // 2 - 2),
              text, font=f_pill, fill=fg)

# ---------- 4. composite final ----------
final = Image.alpha_composite(bg, overlay)
final.convert("RGB").save(OUT, "PNG", quality=95)
print(f"saved -> {OUT}")
