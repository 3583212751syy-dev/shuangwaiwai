import os
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

PRODUCT_PATH = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"
OUT_DIR = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿"

SCENES = [
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Professional_commercial_produc_2026-07-21T07-55-17.png",
     "01_iced_refresh.png", {"size": 460, "x": 300, "y": 520, "bright": 1.08, "warm": 1.0, "shadow_alpha": 55}),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Warm_lifestyle_product_photogr_2026-07-21T07-55-17.png",
     "02_lifestyle_garden.png", {"size": 500, "x": 400, "y": 600, "bright": 1.12, "warm": 1.08, "shadow_alpha": 65}),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Premium_ingredients_focused_pr_2026-07-21T07-55-18.png",
     "03_ingredients_dark.png", {"size": 480, "x": 480, "y": 500, "bright": 0.85, "warm": 0.95, "shadow_alpha": 100}),
    ("E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/Elegant_origin_product_photogr_2026-07-21T07-55-17.png",
     "04_origin_farm.png", {"size": 500, "x": 500, "y": 560, "bright": 1.0, "warm": 1.03, "shadow_alpha": 70}),
]

def load_product(path):
    return Image.open(path).convert("RGBA")

def make_shadow(w, h, alpha, blur=10):
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.ellipse([0, 0, w, h], fill=(0, 0, 0, alpha))
    return shadow.filter(ImageFilter.GaussianBlur(blur))

def compose(scene_path, out_name, cfg):
    bg = Image.open(scene_path).convert("RGBA")
    prod = load_product(PRODUCT_PATH)
    ratio = cfg["size"] / prod.height
    new_w = int(prod.width * ratio)
    new_h = cfg["size"]
    prod = prod.resize((new_w, new_h), Image.LANCZOS)

    # Adjust brightness/color
    if cfg["bright"] != 1.0:
        prod = ImageEnhance.Brightness(prod).enhance(cfg["bright"])
    if cfg["warm"] != 1.0:
        prod = ImageEnhance.Color(prod).enhance(cfg["warm"])

    x = cfg["x"] - new_w // 2
    y = cfg["y"] - new_h // 2

    # Contact shadow
    sw, sh = int(new_w * 1.1), int(new_h * 0.12)
    shadow = make_shadow(sw, sh, cfg["shadow_alpha"], blur=12)
    sx = cfg["x"] - sw // 2
    sy = cfg["y"] + new_h // 2 - sh // 3
    bg.alpha_composite(shadow, (sx, sy))

    # Product
    bg.alpha_composite(prod, (x, y))

    out_path = os.path.join(OUT_DIR, out_name)
    bg.convert("RGB").save(out_path, quality=95)
    print("Saved", out_path)

if __name__ == "__main__":
    for scene_path, out_name, cfg in SCENES:
        compose(scene_path, out_name, cfg)
