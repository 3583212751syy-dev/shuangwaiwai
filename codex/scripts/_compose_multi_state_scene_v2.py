from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import random

random.seed(42)

# Paths
bg_path = "E:/Desktop/茶叶/成品_第十二轮_参考图风格重制/_assets/Commercial_product_photography_2026-07-22T01-33-02.png"
front_path = "E:/Desktop/茶叶/成品_第十一轮_正背组合重制/_assets/product_front_label_fixed.png"
out_path = "E:/Desktop/茶叶/成品_第十三轮_多角度状态场景/01_multi_state_scene_v2.png"

bg = Image.open(bg_path).convert("RGBA")
# Crop watermark aggressively and keep safe composition area
W, H = 980, 960
bg = bg.crop((0, 0, W, H))

pouch = Image.open(front_path).convert("RGBA")


def make_shadow(alpha, blur_radius, opacity_scale, offset_x, offset_y):
    """Create a soft contact shadow from an alpha mask."""
    a_data = alpha.copy().point(lambda p: int(p * opacity_scale))
    shadow_rgba = Image.merge("RGBA", [
        Image.new("L", alpha.size, 35),
        Image.new("L", alpha.size, 28),
        Image.new("L", alpha.size, 22),
        a_data
    ])
    shadow = shadow_rgba.filter(ImageFilter.GaussianBlur(blur_radius))
    offset_canvas = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    offset_canvas.paste(shadow, (offset_x, offset_y), shadow)
    return offset_canvas


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


def add_tea_spill(img, top_x, top_y, length, direction=1):
    """Draw organic-looking loose rooibos spilling from an open top."""
    draw = ImageDraw.Draw(img)
    # Dark opening
    draw.ellipse([top_x - 10, top_y - 7, top_x + 10, top_y + 7], fill=(55, 30, 15, 220))

    # Stream of grains with random jitter
    for i in range(45):
        t = i / 44.0
        x = top_x + int(t * 35 * direction) + random.randint(-3, 3)
        y = top_y + int(t * length) + random.randint(-2, 4)
        size = random.randint(1, 4)
        shade = random.choice([(130, 60, 35), (150, 70, 40), (110, 50, 30), (160, 75, 45)])
        alpha = random.randint(140, 220)
        draw.ellipse([x - size, y - size, x + size, y + size], fill=(*shade, alpha))

    # Pile at bottom: dense cluster of overlapping grains
    pile_x = top_x + int(35 * direction) + random.randint(-5, 5)
    pile_y = top_y + length + 12
    for _ in range(80):
        ox = random.randint(-22, 22)
        oy = random.randint(-10, 10)
        size = random.randint(2, 5)
        shade = random.choice([(140, 65, 35), (155, 72, 42), (125, 58, 32), (170, 80, 48)])
        alpha = random.randint(160, 230)
        draw.ellipse([pile_x + ox - size, pile_y + oy - size,
                      pile_x + ox + size, pile_y + oy + size], fill=(*shade, alpha))


# ===== Instance 1: Standing hero (front-left) =====
inst_hero = render_instance(pouch, 0.60, 0, brightness=1.0)
alpha_hero = inst_hero.getchannel("A")
x_hero, y_hero = 170, 230

# ===== Instance 2: Sealed lying pouch (back-right, smaller) =====
inst_lying = render_instance(pouch, 0.30, 95, brightness=0.93, blur=0.8)
alpha_lying = inst_lying.getchannel("A")
x_lying, y_lying = 720, 250

# ===== Instance 3: Open/tipped pouch with tea spilling (foreground-right) =====
inst_open = render_instance(pouch, 0.42, -20, brightness=0.98, blur=0.3)
alpha_open = inst_open.getchannel("A")
# Find top-left area (the opening after slight rotation)
w3, h3 = inst_open.size
# Scan top edge around center-left for highest alpha point
top_x, top_y = w3 // 2 - 30, 0
for y in range(h3):
    found = False
    for x in range(max(0, w3//2 - 50), min(w3, w3//2)):
        if alpha_open.getpixel((x, y)) > 120:
            top_x, top_y = x, y
            found = True
            break
    if found:
        break

add_tea_spill(inst_open, top_x, top_y, 110, direction=1)
alpha_open = inst_open.getchannel("A")  # re-get after spill
x_open, y_open = 560, 600


# Composite back to front: sealed lying (back) -> open tipped (mid) -> hero (front)
canvas = bg.copy()

# 1) Back sealed lying pouch
shadow_lying = make_shadow(alpha_lying, blur_radius=10, opacity_scale=0.30, offset_x=10, offset_y=12)
s_lying = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
s_lying.paste(shadow_lying, (x_lying, y_lying), shadow_lying)
canvas = Image.alpha_composite(canvas, s_lying)
canvas.paste(inst_lying, (x_lying, y_lying), inst_lying)

# 2) Open tipped pouch with tea spill
shadow_open = make_shadow(alpha_open, blur_radius=12, opacity_scale=0.42, offset_x=16, offset_y=18)
s_open = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
s_open.paste(shadow_open, (x_open, y_open), shadow_open)
canvas = Image.alpha_composite(canvas, s_open)
canvas.paste(inst_open, (x_open, y_open), inst_open)

# 3) Hero standing pouch
shadow_hero = make_shadow(alpha_hero, blur_radius=14, opacity_scale=0.40, offset_x=20, offset_y=24)
s_hero = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
s_hero.paste(shadow_hero, (x_hero, y_hero), shadow_hero)
canvas = Image.alpha_composite(canvas, s_hero)
canvas.paste(inst_hero, (x_hero, y_hero), inst_hero)

canvas.save(out_path, "PNG")
print(f"Saved {out_path}")
