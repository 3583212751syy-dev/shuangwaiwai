from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import os

PRODUCT = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"
OUT_DIR = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿"

# (bg_path, out_name, size, cx, cy_ground, bright, warm, shadow_alpha, shadow_offset, shadow_blur)
SCENES = [
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Clean_commercial_product_photo_2026-07-21T08-31-11.png",
     "01_iced_refresh.png", 440, 380, 690, 1.05, 1.0, 55, (10, 20), 16),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Clean_outdoor_lifestyle_produc_2026-07-21T08-31-12.png",
     "02_lifestyle_garden.png", 500, 540, 650, 1.06, 1.04, 60, (14, 24), 17),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Premium_dark_ingredients_produ_2026-07-21T08-31-12.png",
     "03_ingredients_dark.png", 470, 510, 660, 0.96, 0.98, 100, (10, 18), 18),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Cinematic_origin_product_photo_2026-07-21T08-31-11.png",
     "04_origin_farm.png", 490, 510, 700, 1.0, 1.02, 65, (12, 22), 17),
]

def add_shadow(bg, cx, cy, w, h, alpha, offset, blur):
    shadow = Image.new("RGBA", bg.size, (0,0,0,0))
    draw = ImageDraw.Draw(shadow)
    sx = cx + offset[0]
    sy = cy + offset[1]
    draw.ellipse([sx - w//2, sy - h//6, sx + w//2, sy + h//6], fill=(0,0,0,alpha))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    # Add contact shadow at base
    contact = Image.new("RGBA", bg.size, (0,0,0,0))
    cdraw = ImageDraw.Draw(contact)
    cdraw.ellipse([cx - w//2.2, cy - h//12, cx + w//2.2, cy + h//12], fill=(0,0,0,int(alpha*0.6)))
    contact = contact.filter(ImageFilter.GaussianBlur(4))
    return Image.alpha_composite(Image.alpha_composite(bg, shadow), contact)

def color_warm(img, factor):
    r, g, b, a = img.split()
    if factor >= 1:
        r = r.point(lambda i: min(255, int(i * factor)))
        b = b.point(lambda i: int(i / (1 + (factor-1)*0.5)))
    else:
        r = r.point(lambda i: int(i * factor))
        b = b.point(lambda i: min(255, int(i * (1.05 - factor*0.05))))
    return Image.merge("RGBA", (r, g, b, a))

for bg_path, out_name, size, cx, cy_ground, bright, warm, shadow_alpha, shadow_offset, shadow_blur in SCENES:
    bg = Image.open(bg_path).convert("RGBA")
    prod = Image.open(PRODUCT).convert("RGBA")
    
    ratio = size / prod.height
    new_w = int(prod.width * ratio)
    prod = prod.resize((new_w, size), Image.LANCZOS)
    
    enhancer = ImageEnhance.Brightness(prod)
    prod = enhancer.enhance(bright)
    prod = color_warm(prod, warm)
    
    px = cx - prod.width // 2
    py = cy_ground - prod.height
    
    bg = add_shadow(bg, cx, cy_ground, int(prod.width * 0.85), int(size * 0.22), shadow_alpha, shadow_offset, shadow_blur)
    bg.paste(prod, (px, py), prod)
    bg = ImageEnhance.Contrast(bg).enhance(1.04)
    
    out_path = os.path.join(OUT_DIR, out_name)
    bg.convert("RGB").save(out_path, quality=95)
    print("Saved", out_path)
