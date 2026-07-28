from PIL import Image, ImageFilter, ImageEnhance

bg_path = "E:/Desktop/茶叶/成品_第十三轮_多角度状态场景/_assets/Commercial_product_photography_2026-07-22T02-08-25.png"
front_path = "E:/Desktop/茶叶/成品_第十一轮_正背组合重制/_assets/product_front_label_fixed.png"
back_path = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"
out_path = "E:/Desktop/茶叶/成品_第十三轮_多角度状态场景/03_reference_garden_front_back.png"

bg = Image.open(bg_path).convert("RGBA")
# Crop watermark
W, H = 1000, 980
bg = bg.crop((0, 0, W, H))

front = Image.open(front_path).convert("RGBA")
back = Image.open(back_path).convert("RGBA")


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


# Back pouch: behind and to the right, smaller, slightly dimmed/blurred for depth
inst_back = render_instance(back, 0.52, 0, brightness=0.88, blur=0.9)
alpha_back = inst_back.getchannel("A")
x_back, y_back = 590, 260

# Front pouch: hero, center-left
inst_front = render_instance(front, 0.66, 0, brightness=1.0)
alpha_front = inst_front.getchannel("A")
x_front, y_front = 220, 210

# Compose back to front
canvas = bg.copy()

# Back pouch shadow (sun from upper-left -> shadow lower-right)
shadow_back = make_shadow(alpha_back, blur_radius=14, opacity_scale=0.32, offset_x=22, offset_y=26)
s_back = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
s_back.paste(shadow_back, (x_back, y_back), shadow_back)
canvas = Image.alpha_composite(canvas, s_back)
canvas.paste(inst_back, (x_back, y_back), inst_back)

# Front pouch shadow
shadow_front = make_shadow(alpha_front, blur_radius=16, opacity_scale=0.38, offset_x=26, offset_y=30)
s_front = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
s_front.paste(shadow_front, (x_front, y_front), shadow_front)
canvas = Image.alpha_composite(canvas, s_front)
canvas.paste(inst_front, (x_front, y_front), inst_front)

# Slight warm color grade to match golden hour
r, g, b, a = canvas.split()
r = r.point(lambda i: int(min(255, i * 1.04)))
g = g.point(lambda i: int(min(255, i * 1.01)))
canvas = Image.merge("RGBA", (r, g, b, a))

canvas.save(out_path, "PNG")
print(f"Saved {out_path}")
