from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import os

# Paths
bg_path = "E:/Desktop/茶叶/成品_第十二轮_参考图风格重制/_assets/Commercial_product_photography_2026-07-22T01-33-02.png"
front_path = "E:/Desktop/茶叶/成品_第十一轮_正背组合重制/_assets/product_front_label_fixed.png"
out_path = "E:/Desktop/茶叶/成品_第十二轮_参考图风格重制/01_iced_refresh_reference.png"

# Load images
bg = Image.open(bg_path).convert("RGBA")
front = Image.open(front_path).convert("RGBA")

# Resize product to be the hero
front_h = 540
ratio = front_h / front.height
front_w = int(front.width * ratio)
front = front.resize((front_w, front_h), Image.LANCZOS)

# Position product
x = 270
y = 385

# ---- Create realistic contact shadow from product shape ----
# Use bottom 15% of product alpha, stretch down and blur
front_array = front.load()
shadow_mask = Image.new("L", front.size, 0)
shadow_pixels = shadow_mask.load()

bottom_region = int(front_h * 0.18)
for j in range(front_h - bottom_region, front_h):
    for i in range(front_w):
        alpha = front_array[i, j][3]
        if alpha > 30:
            # Fade out toward bottom
            fade = (j - (front_h - bottom_region)) / bottom_region
            shadow_pixels[i, j] = int(alpha * (1 - fade * 0.7))

# Blur and stretch shadow
shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(8))
# Stretch vertically to simulate shadow on table
shadow_mask = shadow_mask.resize((front_w, int(bottom_region * 2.2)), Image.LANCZOS)

# Create shadow layer under product
shadow_layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
shadow_rgba = Image.new("RGBA", shadow_mask.size, (30, 25, 15, 0))
shadow_rgba.putalpha(ImageEnhance.Brightness(shadow_mask).enhance(0.55))

# Place shadow slightly below product bottom
sx = x
sy = y + front_h - int(bottom_region * 1.2)
shadow_layer.paste(shadow_rgba, (sx, sy), shadow_rgba)

# Composite shadow
bg = Image.alpha_composite(bg, shadow_layer)

# Adjust product to match scene
front_enh = ImageEnhance.Brightness(front).enhance(1.02)
front_enh = ImageEnhance.Contrast(front_enh).enhance(1.01)

# Paste product
bg.paste(front_enh, (x, y), front_enh)

# Subtle ambient occlusion on right edge (light from left)
ao = Image.new("RGBA", bg.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(ao)
for i in range(25):
    alpha = int(20 * (1 - i / 25))
    draw.line([(x + front_w - i, y), (x + front_w - i, y + front_h)], fill=(0, 0, 0, alpha), width=1)
bg = Image.alpha_composite(bg, ao)

# Save
final = bg.convert("RGB")
final.save(out_path, quality=95)
print(f"Saved: {out_path}")
