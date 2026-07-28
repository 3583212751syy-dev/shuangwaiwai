from PIL import Image, ImageFilter, ImageEnhance
import os

# Paths
bg_path = "E:/Desktop/茶叶/成品_第十二轮_参考图风格重制/_assets/Commercial_product_photography_2026-07-22T01-33-02.png"
front_path = "E:/Desktop/茶叶/成品_第十一轮_正背组合重制/_assets/product_front_label_fixed.png"
out_path = "E:/Desktop/茶叶/成品_第十二轮_参考图风格重制/01_iced_refresh_reference.png"

# Load images
bg = Image.open(bg_path).convert("RGBA")
front = Image.open(front_path).convert("RGBA")

# Resize product to be the hero (larger than the glass in background)
# Glass is roughly 220px tall in 1024 image; make pouch ~520px tall
front_h = 540
ratio = front_h / front.height
front_w = int(front.width * ratio)
front = front.resize((front_w, front_h), Image.LANCZOS)

# Position: center-left hero, placed naturally on table surface
x = 270
y = 385

# Create shadow layer
shadow = Image.new("RGBA", bg.size, (0, 0, 0, 0))
from PIL import ImageDraw

# Contact shadow: wide soft ellipse directly under the pouch bottom
shadow_w = front_w - 20
shadow_h = 32
sx = x + 10
sy = y + front_h - 24
for i in range(shadow_h, 0, -1):
    alpha = int(75 * (1 - i / shadow_h))
    if alpha <= 2:
        continue
    offset = int((shadow_h - i) * 2.0)
    draw = ImageDraw.Draw(shadow)
    draw.ellipse([sx + offset, sy + i, sx + shadow_w - offset, sy + i + 5], fill=(40, 30, 20, alpha))

# Darker core shadow right at the contact edge
core_w = front_w - 80
core_h = 12
cx = x + 40
cy = y + front_h - 18
for i in range(core_h, 0, -1):
    alpha = int(45 * (1 - i / core_h))
    if alpha <= 2:
        continue
    offset = int((core_h - i) * 1.5)
    draw = ImageDraw.Draw(shadow)
    draw.ellipse([cx + offset, cy + i, cx + core_w - offset, cy + i + 3], fill=(30, 25, 15, alpha))

# Composite shadow onto background
bg = Image.alpha_composite(bg, shadow)

# Adjust product brightness/contrast to match scene lighting (soft window light)
front_enh = ImageEnhance.Brightness(front).enhance(1.0)
front_enh = ImageEnhance.Contrast(front_enh).enhance(1.02)

# Paste product
bg.paste(front_enh, (x, y), front_enh)

# Slight ambient occlusion on right side of pouch (light from left)
ao = Image.new("RGBA", bg.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(ao)
# Soft gradient on right edge of pouch
for i in range(30):
    alpha = int(25 * (1 - i / 30))
    draw.line([(x + front_w - i, y), (x + front_w - i, y + front_h)], fill=(0, 0, 0, alpha), width=1)
bg = Image.alpha_composite(bg, ao)

# Convert to RGB and save
final = bg.convert("RGB")
final.save(out_path, quality=95)
print(f"Saved: {out_path}")
