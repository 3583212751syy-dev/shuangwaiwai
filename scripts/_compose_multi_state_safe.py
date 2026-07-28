from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import random

random.seed(7)

bg_path = "E:/Desktop/茶叶/成品_第十二轮_参考图风格重制/_assets/Commercial_product_photography_2026-07-22T01-33-02.png"
front_path = "E:/Desktop/茶叶/成品_第十一轮_正背组合重制/_assets/product_front_label_fixed.png"
out_path = "E:/Desktop/茶叶/成品_第十三轮_多角度状态场景/02_safe_standing_lying.png"

bg = Image.open(bg_path).convert("RGBA")
W, H = 980, 960
bg = bg.crop((0, 0, W, H))
pouch = Image.open(front_path).convert("RGBA")


def make_shadow(alpha, blur_radius, opacity_scale, offset_x, offset_y):
    a_data = alpha.copy().point(lambda p: int(p * opacity_scale))
    shadow_rgba = Image.merge("RGBA", [
        Image.new("L", alpha.size, 35),
        Image.new("L", alpha.size, 28),
        Image.new("L", alpha.size, 22),
        a_data
    ])
    shadow = shadow_rgba.filter(ImageFilter.GaussianBlur(blur_radius))
    canvas = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    canvas.paste(shadow, (offset_x, offset_y), shadow)
    return canvas


def render_instance(src, scale, angle, brightness, blur=0):
    w = int(src.width * scale)
    h = int(src.height * scale)
    resized = src.resize((w, h), Image.LANCZOS)
    if angle != 0:
        rotated = resized.rotate(angle, expand=True, resample=Image.BICUBIC)
    else:
        rotated = resized.copy()
    if brightness != 1.0:
        rotated = ImageEnhance.Brightness(rotated).enhance(brightness)
    if blur > 0:
        rotated = rotated.filter(ImageFilter.GaussianBlur(blur))
    return rotated


def add_loose_tea_pile(img, cx, cy):
    draw = ImageDraw.Draw(img)
    for _ in range(120):
        ox = random.randint(-35, 35)
        oy = random.randint(-18, 18)
        size = random.randint(2, 6)
        shade = random.choice([(140, 65, 35), (155, 72, 42), (125, 58, 32), (170, 80, 48)])
        alpha = random.randint(160, 230)
        draw.ellipse([cx + ox - size, cy + oy - size,
                      cx + ox + size, cy + oy + size], fill=(*shade, alpha))


# Hero standing left-center
inst_hero = render_instance(pouch, 0.60, 0, brightness=1.0)
alpha_hero = inst_hero.getchannel("A")
x_hero, y_hero = 180, 235

# Lying sealed pouch foreground right, angled naturally
inst_lying = render_instance(pouch, 0.34, 85, brightness=0.97, blur=0.4)
alpha_lying = inst_lying.getchannel("A")
x_lying, y_lying = 640, 665

# Loose tea pile near lying pouch
pile_overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
add_loose_tea_pile(pile_overlay, 780, 800)

# Composite
canvas = bg.copy()

# Back: none extra
# Lying pouch (mid/foreground)
shadow_lying = make_shadow(alpha_lying, blur_radius=10, opacity_scale=0.45, offset_x=12, offset_y=12)
s_lying = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
s_lying.paste(shadow_lying, (x_lying, y_lying), shadow_lying)
canvas = Image.alpha_composite(canvas, s_lying)
canvas.paste(inst_lying, (x_lying, y_lying), inst_lying)

# Tea pile
canvas = Image.alpha_composite(canvas, pile_overlay)

# Hero standing (front)
shadow_hero = make_shadow(alpha_hero, blur_radius=14, opacity_scale=0.40, offset_x=20, offset_y=24)
s_hero = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
s_hero.paste(shadow_hero, (x_hero, y_hero), shadow_hero)
canvas = Image.alpha_composite(canvas, s_hero)
canvas.paste(inst_hero, (x_hero, y_hero), inst_hero)

canvas.save(out_path, "PNG")
print(f"Saved {out_path}")
