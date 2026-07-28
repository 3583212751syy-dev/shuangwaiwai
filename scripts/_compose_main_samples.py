import os
import sys
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

PRODUCT_PATH = "E:/Desktop/茶叶/成品_第七轮_站立图文修正/_assets/product_standing.png"
OUT_DIR = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿"

SCENES = [
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Professional_commercial_produc_2026-07-21T07-55-17.png",
     "01_iced_refresh.png", {"size": 520, "x": 510, "y": 520, "bright": 1.05, "warm": 1.0, "shadow_alpha": 60}),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Warm_lifestyle_product_photogr_2026-07-21T07-55-17.png",
     "02_lifestyle_garden.png", {"size": 560, "x": 500, "y": 580, "bright": 1.08, "warm": 1.05, "shadow_alpha": 70}),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Premium_ingredients_focused_pr_2026-07-21T07-55-18.png",
     "03_ingredients_dark.png", {"size": 540, "x": 500, "y": 540, "bright": 0.88, "warm": 0.95, "shadow_alpha": 90}),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Elegant_origin_product_photogr_2026-07-21T07-55-17.png",
     "04_origin_farm.png", {"size": 560, "x": 500, "y": 560, "bright": 1.0, "warm": 1.02, "shadow_alpha": 75}),
]

def clean_product(path):
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    # Crop to central product area to remove black edge artifacts
    left = int(w * 0.05)
    top = int(h * 0.02)
    right = int(w * 0.95)
    bottom = int(h * 0.98)
    img = img.crop((left, top, right, bottom))
    # Clean alpha: remove near-black pixels at edges
    pixels = img.load()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = pixels[x, y]
            if a > 0 and r < 30 and g < 30 and b < 30:
                pixels[x, y] = (0, 0, 0, 0)
    return img

def make_shadow(size, alpha, blur=8):
    shadow = Image.new("RGBA", (size[0], size[1]//3), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.ellipse([0, 0, size[0], size[1]//3], fill=(0, 0, 0, alpha))
    return shadow.filter(ImageFilter.GaussianBlur(blur))

def compose(scene_path, out_name, cfg):
    bg = Image.open(scene_path).convert("RGBA")
    prod = clean_product(PRODUCT_PATH)
    # Resize preserving aspect ratio
    ratio = cfg["size"] / prod.height
    new_w = int(prod.width * ratio)
    new_h = cfg["size"]
    prod = prod.resize((new_w, new_h), Image.LANCZOS)

    # Brightness / warmth adjustments
    if cfg["bright"] != 1.0:
        enh = ImageEnhance.Brightness(prod)
        prod = enh.enhance(cfg["bright"])
    if cfg["warm"] != 1.0:
        # Simple color balance via color enhance
        enh = ImageEnhance.Color(prod)
        prod = enh.enhance(cfg["warm"])

    # Position
    x = cfg["x"] - new_w // 2
    y = cfg["y"] - new_h // 2

    # Shadow
    shadow = make_shadow((new_w, int(new_h * 0.25)), cfg["shadow_alpha"])
    sx = cfg["x"] - shadow.width // 2
    sy = cfg["y"] + new_h // 2 - shadow.height // 3
    bg.alpha_composite(shadow, (sx, sy))

    # Product
    bg.alpha_composite(prod, (x, y))

    # Save
    out_path = os.path.join(OUT_DIR, out_name)
    bg.convert("RGB").save(out_path, quality=95)
    print("Saved", out_path)

if __name__ == "__main__":
    for scene_path, out_name, cfg in SCENES:
        compose(scene_path, out_name, cfg)
