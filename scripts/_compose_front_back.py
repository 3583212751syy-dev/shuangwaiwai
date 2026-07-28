from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os

OUT_DIR = "E:/Desktop/茶叶/成品_第十轮_正背组合图"
FRONT = "E:/Desktop/茶叶/成品_第十轮_正背组合图/_assets/product_front_clean.png"
BACK = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"

# (bg_path, out_name, front_cfg, back_cfg)
# front_cfg/back_cfg: size=height, x, y (anchor center), bright
SCENES = [
    ("E:/Desktop/茶叶/成品_第十轮_正背组合图/_assets/Commercial_product_photography_2026-07-21T09-24-18.png",
     "01_iced_refresh.png",
     {"size": 520, "x": 430, "y": 580, "bright": 1.05},
     {"size": 340, "x": 620, "y": 500, "bright": 1.05, "alpha": 0.85}),
    ("E:/Desktop/茶叶/成品_第十轮_正背组合图/_assets/Commercial_product_photography_2026-07-21T09-24-50.png",
     "02_lifestyle_garden.png",
     {"size": 520, "x": 520, "y": 580, "bright": 1.05},
     {"size": 340, "x": 690, "y": 500, "bright": 1.05, "alpha": 0.85}),
    ("E:/Desktop/茶叶/成品_第十轮_正背组合图/_assets/Commercial_product_photography_2026-07-21T09-26-08.png",
     "03_ingredients_dark.png",
     {"size": 500, "x": 460, "y": 570, "bright": 0.95},
     {"size": 330, "x": 630, "y": 490, "bright": 0.95, "alpha": 0.80}),
    ("E:/Desktop/茶叶/成品_第十轮_正背组合图/_assets/Commercial_product_photography_2026-07-21T09-24-49.png",
     "04_origin_farm.png",
     {"size": 520, "x": 500, "y": 580, "bright": 1.0},
     {"size": 340, "x": 660, "y": 500, "bright": 1.0, "alpha": 0.85}),
]

def load_and_resize(path, height):
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    new_w = int(w * height / h)
    return img.resize((new_w, height), Image.LANCZOS)

def adjust_brightness(img, factor):
    if factor == 1.0:
        return img
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def add_shadow_layer(bg_size, cx, cy, w, h, alpha, blur=20):
    shadow = Image.new("RGBA", bg_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    sw, sh = w + 40, 30
    sx, sy = cx - sw // 2, cy + h // 2 - 12
    draw.ellipse([sx, sy, sx + sw, sy + sh], fill=(0, 0, 0, alpha))
    return shadow.filter(ImageFilter.GaussianBlur(blur))

front_img = Image.open(FRONT).convert("RGBA")
back_img = Image.open(BACK).convert("RGBA")

for bg_path, out_name, fcfg, bcfg in SCENES:
    bg = Image.open(bg_path).convert("RGBA")
    W, H = bg.size

    # --- Place back product first (behind) ---
    back = load_and_resize(BACK, bcfg["size"])
    back = adjust_brightness(back, bcfg["bright"])
    bw, bh = back.size
    # Apply slight transparency to push it back visually
    if bcfg.get("alpha", 1.0) < 1.0:
        r, g, b, a = back.split()
        a = a.point(lambda i: int(i * bcfg["alpha"]))
        back = Image.merge("RGBA", (r, g, b, a))
    bx = bcfg["x"] - bw // 2
    by = bcfg["y"] - bh // 2

    # Back shadow
    back_shadow = add_shadow_layer((W, H), bcfg["x"], bcfg["y"], bw, bh, 50, blur=18)
    bg = Image.alpha_composite(bg, back_shadow)
    bg.paste(back, (bx, by), back)

    # --- Place front product (main) ---
    front = load_and_resize(FRONT, fcfg["size"])
    front = adjust_brightness(front, fcfg["bright"])
    fw, fh = front.size
    fx = fcfg["x"] - fw // 2
    fy = fcfg["y"] - fh // 2

    # Front shadow (stronger, sharper)
    front_shadow = add_shadow_layer((W, H), fcfg["x"], fcfg["y"], fw, fh, 70, blur=16)
    bg = Image.alpha_composite(bg, front_shadow)
    bg.paste(front, (fx, fy), front)

    out_path = os.path.join(OUT_DIR, out_name)
    bg.convert("RGB").save(out_path, quality=95)
    print("Saved", out_path)
