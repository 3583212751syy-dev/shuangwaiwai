from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import os

PRODUCT = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"
OUT_DIR = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿"

SCENES = [
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Professional_commercial_produc_2026-07-21T07-55-17.png",
     "01_iced_refresh.png", {"size": 520, "x": 500, "y": 520, "bright": 1.05, "warm": 1.0, "shadow_alpha": 55, "shadow_offset": (10, 25)}),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Warm_lifestyle_product_photogr_2026-07-21T07-55-17.png",
     "02_lifestyle_garden.png", {"size": 540, "x": 500, "y": 560, "bright": 1.08, "warm": 1.05, "shadow_alpha": 65, "shadow_offset": (12, 28)}),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Premium_ingredients_focused_pr_2026-07-21T07-55-18.png",
     "03_ingredients_dark.png", {"size": 520, "x": 500, "y": 520, "bright": 0.95, "warm": 0.98, "shadow_alpha": 100, "shadow_offset": (8, 20)}),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Elegant_origin_product_photogr_2026-07-21T07-55-17.png",
     "04_origin_farm.png", {"size": 540, "x": 500, "y": 560, "bright": 1.02, "warm": 1.02, "shadow_alpha": 70, "shadow_offset": (12, 28)}),
]

def add_shadow(bg, cx, cy, w, h, alpha, offset):
    shadow = Image.new("RGBA", bg.size, (0,0,0,0))
    draw = ImageDraw.Draw(shadow)
    sx = cx + offset[0]
    sy = cy + offset[1]
    # Elliptical shadow
    draw.ellipse([sx - w//2, sy - h//4, sx + w//2, sy + h//4], fill=(0,0,0,alpha))
    shadow = shadow.filter(ImageFilter.GaussianBlur(15))
    return Image.alpha_composite(bg, shadow)

def color_warm(img, factor):
    r, g, b, a = img.split()
    r = r.point(lambda i: min(255, int(i * factor)))
    if factor >= 1:
        # Warm: boost red more than blue
        b = b.point(lambda i: int(i / (1 + (factor-1)*0.5)))
    else:
        # Cool: reduce red
        r = r.point(lambda i: int(i * factor))
    return Image.merge("RGBA", (r, g, b, a))

for bg_path, out_name, cfg in SCENES:
    bg = Image.open(bg_path).convert("RGBA")
    prod = Image.open(PRODUCT).convert("RGBA")
    
    # Resize product
    ratio = cfg["size"] / prod.height
    new_w = int(prod.width * ratio)
    prod = prod.resize((new_w, cfg["size"]), Image.LANCZOS)
    
    # Brightness
    enhancer = ImageEnhance.Brightness(prod)
    prod = enhancer.enhance(cfg["bright"])
    
    # Warm/cool adjustment
    prod = color_warm(prod, cfg["warm"])
    
    # Center bottom of product
    cx = cfg["x"]
    cy = cfg["y"]
    px = cx - prod.width // 2
    py = cy - prod.height
    
    # Add shadow first
    bg = add_shadow(bg, cx, cy, prod.width, int(cfg["size"]*0.25), cfg["shadow_alpha"], cfg["shadow_offset"])
    
    # Paste product
    bg.paste(prod, (px, py), prod)
    
    # Slight overall contrast
    bg = ImageEnhance.Contrast(bg).enhance(1.05)
    
    out_path = os.path.join(OUT_DIR, out_name)
    bg.convert("RGB").save(out_path, quality=95)
    print("Saved", out_path)
