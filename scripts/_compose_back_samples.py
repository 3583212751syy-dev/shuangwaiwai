from PIL import Image, ImageDraw, ImageFilter
import os

OUT_DIR = "E:/Desktop/茶叶/成品_第九轮_真实站立包装"
PRODUCT = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"

SCENES = [
    ("E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/Frontal_commercial_product_pho_2026-07-21T08-50-21.png",
     "01_iced_refresh.png", {"size": 420, "x": 512, "y": 540, "shadow_alpha": 60, "bright": 1.05}),
    ("E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/Frontal_outdoor_lifestyle_prod_2026-07-21T08-50-22.png",
     "02_lifestyle_garden.png", {"size": 430, "x": 512, "y": 550, "shadow_alpha": 70, "bright": 1.05}),
    ("E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/Frontal_premium_dark_ingredien_2026-07-21T08-50-21.png",
     "03_ingredients_dark.png", {"size": 420, "x": 512, "y": 540, "shadow_alpha": 100, "bright": 0.95}),
    ("E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/Frontal_cinematic_origin_produ_2026-07-21T08-50-21.png",
     "04_origin_farm.png", {"size": 430, "x": 512, "y": 550, "shadow_alpha": 70, "bright": 1.0}),
]

def add_contact_shadow(bg_draw, cx, cy, w, h, alpha):
    """Draw soft oval shadow at base of product"""
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.ellipse([0, 0, w, h], fill=(0, 0, 0, alpha))
    shadow = shadow.filter(ImageFilter.GaussianBlur(15))
    bg_draw._image.paste(shadow, (cx - w//2, cy - h//3), shadow)

def adjust_brightness(img, factor):
    if factor == 1.0:
        return img
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

product = Image.open(PRODUCT).convert("RGBA")

for bg_path, out_name, cfg in SCENES:
    bg = Image.open(bg_path).convert("RGBA")
    W, H = bg.size
    
    # Resize product preserving aspect ratio
    pw, ph = product.size
    new_h = cfg["size"]
    new_w = int(pw * new_h / ph)
    prod = product.resize((new_w, new_h), Image.LANCZOS)
    prod = adjust_brightness(prod, cfg["bright"])
    
    cx, cy = cfg["x"], cfg["y"]
    px = cx - new_w // 2
    py = cy - new_h // 2
    
    # Contact shadow
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sw, sh = new_w + 40, 30
    sx, sy = cx - sw//2, cy + new_h//2 - 15
    sdraw.ellipse([sx, sy, sx+sw, sy+sh], fill=(0, 0, 0, cfg["shadow_alpha"]))
    shadow = shadow.filter(ImageFilter.GaussianBlur(20))
    bg = Image.alpha_composite(bg, shadow)
    
    # Paste product
    bg.paste(prod, (px, py), prod)
    
    out_path = os.path.join(OUT_DIR, out_name)
    bg.convert("RGB").save(out_path, quality=95)
    print("Saved", out_path)
