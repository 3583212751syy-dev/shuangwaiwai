from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from pathlib import Path

out = Path(r"E:\Desktop\茶叶\成品_第二十轮_参考复刻")
bg_path = out / "Create_a_premium_e_commerce_de_2026-07-28T03-35-55.png"
prod_path = Path(r"E:\Desktop\茶叶\成品_第十九轮_热冰对比C\_assets\product_front_real.png")
img = Image.open(bg_path).convert("RGBA")
W,H = img.size

# Remove generated watermark with a clean editorial panel, then keep the composition uncluttered.
d = ImageDraw.Draw(img)
d.rectangle((700, 946, 1024, 1024), fill=(255, 243, 181, 245))

# Prepare the real package asset: remove only the near-white studio background.
prod = Image.open(prod_path).convert("RGBA")
a = np.array(prod)
rgb = a[:,:,:3]
near_white = (rgb[:,:,0] > 238) & (rgb[:,:,1] > 238) & (rgb[:,:,2] > 238)
a[near_white, 3] = 0
prod = Image.fromarray(a)
# Tight crop around non-transparent pixels.
bbox = prod.getbbox()
prod = prod.crop(bbox)
# Place as the hero product, preserving the real label artwork.
max_h = 640
scale = max_h / prod.height
prod = prod.resize((int(prod.width*scale), max_h), Image.Resampling.LANCZOS)
px = 585
py = 310
shadow = Image.new("RGBA", img.size, (0,0,0,0))
sd = ImageDraw.Draw(shadow)
sd.ellipse((px-18, py+max_h-30, px+prod.width+35, py+max_h+28), fill=(93,55,25,80))
shadow = shadow.filter(ImageFilter.GaussianBlur(20))
img = Image.alpha_composite(img, shadow)
img.alpha_composite(prod, (px,py))

d = ImageDraw.Draw(img)
# Typography: crisp English text added after image generation.
font_dir = Path(r"C:\Windows\Fonts")
def font(name, size):
    return ImageFont.truetype(str(font_dir/name), size)
red = (174, 37, 28, 255)
dark = (76, 47, 30, 255)
cream = (255, 247, 211, 255)
# Centered title in the two generated red bars.
for text, y, size in [("NATURAL ROOIBOS", 52, 58), ("GOODNESS", 118, 58)]:
    f = font("arialbd.ttf", size)
    box = d.textbbox((0,0), text, font=f)
    d.text(((W-(box[2]-box[0]))/2, y), text, font=f, fill=red)
# Subtitle
f = font("georgia.ttf", 27)
sub = "A smooth herbal tea for daily relaxation"
b = d.textbbox((0,0), sub, font=f)
d.text(((W-(b[2]-b[0]))/2, 188), sub, font=f, fill=dark)

# Replace abstract pill bars with concise benefit copy.
pills = [
    ("100% NATURAL", 320),
    ("CAFFEINE-FREE", 405),
    ("GENTLE HERBAL BLEND", 490),
    ("HOT OR ICED", 575),
]
for text, y in pills:
    f = font("arialbd.ttf", 24)
    d.ellipse((62, y, 94, y+32), fill=cream)
    d.text((108, y+3), text, font=f, fill=cream)
# Small product descriptor on the right of the pouch.
f = font("georgia.ttf", 24)
d.text((700, 820), "Rooibos Tea • 100g", font=f, fill=dark)
f2 = font("arial.ttf", 18)
d.text((700, 855), "Natural • Smooth • Caffeine-Free", font=f2, fill=dark)
# tiny footer
f3 = font("arial.ttf", 16)
d.text((54, 972), "BREW YOUR MOMENT", font=f3, fill=(117,73,39,255))

final = img.convert("RGB")
final.save(out / "01_reference_style_demo.png", quality=95)
print(out / "01_reference_style_demo.png")
