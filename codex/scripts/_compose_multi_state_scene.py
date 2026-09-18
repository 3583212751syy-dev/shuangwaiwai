from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import os

# Paths
bg_path = "E:/Desktop/茶叶/成品_第十二轮_参考图风格重制/_assets/Commercial_product_photography_2026-07-22T01-33-02.png"
front_path = "E:/Desktop/茶叶/成品_第十一轮_正背组合重制/_assets/product_front_label_fixed.png"
out_path = "E:/Desktop/茶叶/成品_第十三轮_多角度状态场景/01_multi_state_scene.png"

bg = Image.open(bg_path).convert("RGBA")
# Crop watermark at bottom-right: keep top-left 1000x980
bg = bg.crop((0, 0, 1000, 980))

pouch = Image.open(front_path).convert("RGBA")


def make_shadow(alpha, blur_radius, opacity_scale, offset_x, offset_y):
    """Create a soft contact shadow from an alpha mask."""
    # Dark shadow layer
    shadow = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    # Fill non-transparent with dark color scaled by opacity
    a = alpha.copy()
    # Multiply alpha by opacity scale
    a_data = a.getchannel("A")
    a_data = a_data.point(lambda p: int(p * opacity_scale))
    shadow_rgba = Image.merge("RGBA", [
        Image.new("L", alpha.size, 30),
        Image.new("L", alpha.size, 25),
        Image.new("L", alpha.size, 20),
        a_data
    ])
    shadow = shadow_rgba.filter(ImageFilter.GaussianBlur(blur_radius))
    # Offset
    offset_canvas = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    offset_canvas.paste(shadow, (offset_x, offset_y), shadow)
    return offset_canvas


def render_instance(src, scale, angle, brightness):
    """Rotate/scale pouch, return (image, alpha)."""
    w = int(src.width * scale)
    h = int(src.height * scale)
    resized = src.resize((w, h), Image.LANCZOS)
    if angle != 0:
        rotated = resized.rotate(angle, expand=True, resample=Image.BICUBIC)
    else:
        rotated = resized.copy()
    # Adjust brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(rotated)
        rotated = enhancer.enhance(brightness)
    # Depth blur if needed
    return rotated


def add_tea_spill(img, top_x, top_y, length):
    """Draw loose rooibos tea spilling from an open top."""
    draw = ImageDraw.Draw(img)
    tea_color = (140, 65, 35, 210)
    # Opening dark ellipse
    draw.ellipse([top_x - 12, top_y - 8, top_x + 12, top_y + 8], fill=(60, 35, 20, 200))
    # Stream of grains
    for i in range(18):
        t = i / 17.0
        x = top_x + int(t * 25)
        y = top_y + int(t * length) + (i * i) // 8
        size = 3 + i // 5
        alpha = 210 - i * 6
        draw.ellipse([x - size, y - size, x + size, y + size], fill=(140, 65, 35, alpha))
    # Pile at bottom
    pile_x = top_x + 28
    pile_y = top_y + length + 20
    for i in range(8):
        left = pile_x - 14 + i * 3
        right = pile_x + 10 - i
        if right <= left:
            right = left + 2
        draw.ellipse([left, pile_y - 6 + i, right, pile_y + 8 + i], fill=(150, 70, 40, max(0, 180 - i * 15)))


# ===== Instance 1: Open/tipped pouch (back) =====
scale1 = 0.40
angle1 = -75  # counter-clockwise: top tilts down-left
inst1 = render_instance(pouch, scale1, angle1, brightness=0.93)
# Apply slight blur for depth
inst1 = inst1.filter(ImageFilter.GaussianBlur(0.8))
alpha1 = inst1.getchannel("A")
# Find topmost non-transparent point near horizontal center
a1_data = list(alpha1.getdata())
w1, h1 = inst1.size
top_x, top_y = w1 // 2, 0
for y in range(h1):
    for x in range(max(0, w1//2 - 30), min(w1, w1//2 + 30)):
        if a1_data[y * w1 + x] > 120:
            top_x, top_y = x, y
            break
    else:
        continue
    break
# Add tea spill overlay to a copy
inst1_with_tea = inst1.copy()
add_tea_spill(inst1_with_tea, top_x, top_y, 90)
# Position: back right
x1, y1 = 690, 260

# ===== Instance 2: Lying pouch (mid-ground) =====
scale2 = 0.34
angle2 = 92  # lying nearly horizontal
inst2 = render_instance(pouch, scale2, angle2, brightness=0.96)
inst2 = inst2.filter(ImageFilter.GaussianBlur(0.6))
alpha2 = inst2.getchannel("A")
# Position: right side, on table, in front of glass
x2, y2 = 680, 640

# ===== Instance 3: Standing hero pouch (front) =====
scale3 = 0.58
angle3 = 0
inst3 = render_instance(pouch, scale3, angle3, brightness=1.0)
alpha3 = inst3.getchannel("A")
# Position: center-left
x3, y3 = 180, 240


# Composite back to front
canvas = bg.copy()

# 1) Open tipped pouch + shadow
shadow1 = make_shadow(inst1_with_tea, blur_radius=14, opacity_scale=0.35, offset_x=18, offset_y=22)
shadow1_canvas = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
shadow1_canvas.paste(shadow1, (x1, y1), shadow1)
canvas = Image.alpha_composite(canvas, shadow1_canvas)

canvas.paste(inst1_with_tea, (x1, y1), inst1_with_tea)

# 2) Lying pouch + shadow
shadow2 = make_shadow(inst2, blur_radius=10, opacity_scale=0.45, offset_x=12, offset_y=10)
shadow2_canvas = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
shadow2_canvas.paste(shadow2, (x2, y2), shadow2)
canvas = Image.alpha_composite(canvas, shadow2_canvas)

canvas.paste(inst2, (x2, y2), inst2)

# 3) Standing hero + shadow
shadow3 = make_shadow(inst3, blur_radius=16, opacity_scale=0.40, offset_x=22, offset_y=28)
shadow3_canvas = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
shadow3_canvas.paste(shadow3, (x3, y3), shadow3)
canvas = Image.alpha_composite(canvas, shadow3_canvas)

canvas.paste(inst3, (x3, y3), inst3)

# Final crop to clean square (remove any residual watermark)
canvas = canvas.crop((0, 0, 1000, 980))
canvas.save(out_path, "PNG")
print(f"Saved {out_path}")
