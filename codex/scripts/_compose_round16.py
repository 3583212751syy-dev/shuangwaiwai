"""Round 16: 5 designed two-pouch hero variants.

Universal structure: title band (top) + two real pouches centered (front hero
+ back behind) + badge row (bottom). Five distinct backgrounds + five distinct
typographic treatments. Pouches are the REAL assets (no AI redraw) -> label &
bag shape stay 100% original.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os

BASE = "E:/Desktop/茶叶/成品_第十六轮_多张排版变体"
BG_DIR = os.path.join(BASE, "_bg")
FRONT = "E:/Desktop/茶叶/成品_第十一轮_正背组合重制/_assets/product_front_label_fixed.png"
BACK = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"
SIZE = 1024

# ---------- font helpers ----------
def load_font(size, bold=True, serif=False):
    if serif:
        cands = [r"C:\Windows\Fonts\georgiab.ttf", r"C:\Windows\Fonts\georgia.ttf",
                 r"C:\Windows\Fonts\timesbd.ttf"]
    else:
        cands = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
                 r"C:\Windows\Fonts\calibrib.ttf"]
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def txt_w(draw, t, f):
    b = draw.textbbox((0, 0), t, font=f); return b[2] - b[0]
def txt_h(draw, t, f):
    b = draw.textbbox((0, 0), t, font=f); return b[3] - b[1]

# ---------- asset prep (crop to content bbox, scale by height) ----------
def crop_content(path):
    im = Image.open(path).convert("RGBA")
    a = np.array(im.split()[3])
    ys, xs = np.where(a > 10)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return im.crop(bbox)

def scale_h(im, h):
    w = max(1, int(h * im.width / im.height))
    return im.resize((w, h), Image.LANCZOS)

front_full = crop_content(FRONT)
back_full = crop_content(BACK)

def make_shadow(mask, blur=18, opacity=130):
    sh = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    alpha = mask.split()[3]
    black = Image.new("RGBA", mask.size, (28, 20, 12, 255))
    sh = Image.composite(black, Image.new("RGBA", mask.size, (0, 0, 0, 0)), alpha)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    a = sh.split()[3].point(lambda p: int(p * opacity / 255))
    sh.putalpha(a)
    return sh

def remove_watermark(bg):
    bg = bg.resize((SIZE, SIZE), Image.LANCZOS)
    bg = bg.crop((0, 0, SIZE, SIZE - 44))          # drop bottom AI watermark strip
    last = bg.crop((0, bg.height - 1, SIZE, bg.height))
    canvas = Image.new("RGBA", (SIZE, SIZE))
    canvas.paste(bg, (0, 0))
    canvas.paste(last, (0, SIZE - 44))              # repeat last row to fill 44px
    return canvas

# ---------- generic layered draw helpers ----------
def add_layer(base, layer):
    return Image.alpha_composite(base, layer)

def soft_shadow_rect(cx0, cy0, cx1, cy1, radius, color, blur=12, off=(8, 14)):
    L = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    d.rounded_rectangle([cx0 + off[0], cy0 + off[1], cx1 + off[0], cy1 + off[1]],
                        radius=radius, fill=color)
    L = L.filter(ImageFilter.GaussianBlur(blur))
    return L

# ============================================================
# THEME DEFINITIONS
# ============================================================
THEMES = {
    "01_garden": dict(
        bg="Professional_commercial_produc_2026-07-22T05-30-13.png",
        serif=False,
        accent=(78, 38, 18), sub=(80, 55, 30),
        eyebrow="AUTHENTIC SOUTH AFRICAN",
        t1="RED ROOIBOS", t2="TEA",
        sub_="Caffeine-Free Herbal Tea  ·  Cederberg Mountains",
        card="cream",
        badges=("pill4", [(76, 110, 60), (170, 50, 40), (130, 80, 30), (50, 95, 110)],
                ["100% NATURAL", "CAFFEINE FREE", "ANTIOXIDANT RICH", "GREAT HOT OR ICED"]),
    ),
    "02_dark": dict(
        bg="Moody_dark_luxury_commercial_p_2026-07-22T05-30-13.png",
        serif=True,
        accent=(214, 172, 96), sub=(196, 158, 96),
        eyebrow=None,
        t1="PREMIUM", t2="ROOIBOS TEA",
        sub_="Single-Origin  ·  Cederberg, South Africa",
        card=None,
        badges=("pill3gold", (214, 172, 96), ["100% NATURAL", "CAFFEINE FREE", "ANTIOXIDANT RICH"]),
    ),
    "03_iced": dict(
        bg="Clean_bright_studio_product_ba_2026-07-22T05-30-13.png",
        serif=False,
        accent=(36, 104, 120), sub=(60, 90, 100),
        eyebrow=None,
        t1="ENJOY IT", t2="HOT OR ICED",
        sub_="Naturally caffeine-free red rooibos",
        card=None,
        badges=("circle4", (36, 104, 120), [["100%", "NATURAL"], ["CAFFEINE", "FREE"],
                                            ["ANTIOXIDANT", "RICH"], ["SUGAR", "FREE"]]),
    ),
    "04_cozy": dict(
        bg="Cozy_warm_lifestyle_product_ph_2026-07-22T05-30-12.png",
        serif=True,
        accent=(92, 52, 26), sub=(110, 70, 40),
        eyebrow=None,
        t1="UNWIND WITH", t2="EVERY CUP",
        sub_="A calm, caffeine-free evening ritual",
        card=None,
        badges=("circle3cap", (92, 52, 26), [("CALM", "Caffeine-Free"),
                                             ("PURE", "100% Natural"),
                                             ("FIT", "Antioxidant Rich")]),
    ),
    "05_marble": dict(
        bg="Elegant_luxury_flat_lay_produc_2026-07-22T05-30-13.png",
        serif=True,
        accent=(40, 40, 44), sub=(70, 70, 74),
        eyebrow=None,
        t1="100% NATURAL", t2="ROOIBOS TEA",
        sub_="South African Red Bush  ·  Cederberg",
        card=None, center=True,
        badges=("pill3slim", (40, 40, 44), ["CAFFEINE FREE", "ANTIOXIDANT RICH", "GREAT HOT OR ICED"]),
    ),
}

# ---------- placement (universal) ----------
def placements():
    # front hero centered; back behind-right
    fh, fy = 648, 214
    front = scale_h(front_full, fh)
    fx = (SIZE - front.width) // 2
    bh, by = 470, 300
    back = scale_h(back_full, bh)
    bx = fx + front.width // 2 - 40   # behind, peeking right of hero
    return front, fx, fy, back, bx, by

# ============================================================
def draw_title(overlay, th):
    draw = ImageDraw.Draw(overlay)
    x0 = 60
    accent = th["accent"]; sub_text = th["sub_"]; sub_col = th["sub"]
    if th.get("card") == "cream":
        # cream torn-paper card behind title
        cx0, cy0, cx1, cy1 = 40, 48, 600, 218
        overlay = add_layer(overlay, soft_shadow_rect(cx0, cy0, cx1, cy1, 8, (20, 14, 8, 90)))
        L = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        d = ImageDraw.Draw(L)
        d.rounded_rectangle([cx0, cy0, cx1, cy1], radius=6,
                            fill=(248, 240, 224, 250), outline=(178, 142, 90, 255), width=2)
        overlay = add_layer(overlay, L)
        draw = ImageDraw.Draw(overlay)
    y = 64
    if th.get("eyebrow"):
        fe = load_font(22, bold=False, serif=False)
        draw.text((x0, y), th["eyebrow"], font=fe, fill=(120, 80, 30, 255))
        y += 30
    f1 = load_font(54, bold=True, serif=th["serif"])
    f2 = load_font(58, bold=True, serif=th["serif"])
    draw.text((x0, y), th["t1"], font=f1, fill=accent)
    w1 = txt_w(draw, th["t1"], f1)
    draw.text((x0 + w1 + 20, y - 4), th["t2"], font=f2, fill=accent)
    y += txt_h(draw, th["t1"], f1) + 16
    fs = load_font(22, bold=False, serif=False)
    draw.text((x0, y), sub_text, font=fs, fill=sub_col)
    # accent rule
    if th["card"] != "cream":
        rl = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        rd = ImageDraw.Draw(rl)
        if th.get("center"):
            tw = txt_w(draw, th["t1"], f1)
            rd.line([(SIZE//2 - tw//2, y - 6), (SIZE//2 + tw//2, y - 6)], fill=accent + (255,), width=2)
        else:
            rd.line([(x0, y - 6), (x0 + 360, y - 6)], fill=accent + (255,), width=3)
        overlay = add_layer(overlay, rl)
    return overlay

def draw_badges(overlay, th):
    kind = th["badges"][0]
    draw = ImageDraw.Draw(overlay)
    if kind == "pill4":
        _, colors, labels = th["badges"]
        _, fg_texts, _ = th["badges"], None, None
        pw, ph, gap = 218, 56, 16
        total = pw * 4 + gap * 3
        x0 = (SIZE - total) // 2
        y = 884
        band = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        bd = ImageDraw.Draw(band)
        bd.rounded_rectangle([x0 - 22, y - 22, x0 + total + 22, y + ph + 22],
                             radius=36, fill=(252, 246, 232, 205), outline=(180, 140, 90, 255), width=2)
        overlay = add_layer(overlay, band)
        draw = ImageDraw.Draw(overlay)
        fp = load_font(22, bold=True)
        for i, (lab, col) in enumerate(zip(labels, colors)):
            px = x0 + i * (pw + gap)
            overlay = add_layer(overlay, soft_shadow_rect(px, y, px + pw, y + ph, ph // 2, (40, 25, 12, 80), blur=6, off=(3, 5)))
            draw = ImageDraw.Draw(overlay)
            draw.rounded_rectangle([px, y, px + pw, y + ph], radius=ph // 2,
                                   fill=(252, 246, 238, 255), outline=col, width=2)
            tw = txt_w(draw, lab, fp); thh = txt_h(draw, lab, fp)
            draw.text((px + (pw - tw)//2, y + (ph - thh)//2 - 2), lab, font=fp, fill=col)
    elif kind == "pill3gold":
        _, col, labels = th["badges"]
        pw, ph, gap = 244, 52, 20
        total = pw * 3 + gap * 2
        x0 = (SIZE - total) // 2
        y = 892
        fp = load_font(23, bold=True)
        for i, lab in enumerate(labels):
            px = x0 + i * (pw + gap)
            draw.rounded_rectangle([px, y, px + pw, y + ph], radius=ph // 2,
                                   fill=(30, 24, 16, 160), outline=col, width=2)
            tw = txt_w(draw, lab, fp); thh = txt_h(draw, lab, fp)
            draw.text((px + (pw - tw)//2, y + (ph - thh)//2 - 2), lab, font=fp, fill=col)
    elif kind == "pill3slim":
        _, col, labels = th["badges"]
        pw, ph, gap = 300, 46, 18
        total = pw * 3 + gap * 2
        x0 = (SIZE - total) // 2
        y = 898
        fp = load_font(22, bold=True)
        for i, lab in enumerate(labels):
            px = x0 + i * (pw + gap)
            draw.rounded_rectangle([px, y, px + pw, y + ph], radius=ph // 2,
                                   fill=(255, 255, 255, 220), outline=col, width=2)
            tw = txt_w(draw, lab, fp); thh = txt_h(draw, lab, fp)
            draw.text((px + (pw - tw)//2, y + (ph - thh)//2 - 2), lab, font=fp, fill=col)
    elif kind == "circle4":
        _, col, pairs = th["badges"]
        d = 120
        centers = [150, 400, 650, 900]
        y = 880
        fp = load_font(18, bold=True)
        for c, (l1, l2) in zip(centers, pairs):
            overlay = add_layer(overlay, soft_shadow_rect(c - d//2, y, c + d//2, y + d, d//2, (40, 25, 12, 70), blur=6, off=(3, 5)))
            draw = ImageDraw.Draw(overlay)
            draw.ellipse([c - d//2, y, c + d//2, y + d], fill=(240, 247, 248, 250), outline=col, width=3)
            # two lines centered
            b1 = draw.textbbox((0,0), l1, font=fp); w1 = b1[2]-b1[0]; h1 = b1[3]-b1[1]
            b2 = draw.textbbox((0,0), l2, font=fp); w2 = b2[2]-b2[0]; h2 = b2[3]-b2[1]
            draw.text((c - w1//2, y + d//2 - (h1+h2+4)//2), l1, font=fp, fill=col)
            draw.text((c - w2//2, y + d//2 - (h1+h2+4)//2 + h1 + 4), l2, font=fp, fill=col)
    elif kind == "circle3cap":
        _, col, items = th["badges"]
        d = 108
        centers = [250, 512, 774]
        y = 858
        fp = load_font(22, bold=True); fc = load_font(18, bold=False)
        for c, (word, cap) in zip(centers, items):
            overlay = add_layer(overlay, soft_shadow_rect(c - d//2, y, c + d//2, y + d, d//2, (40, 25, 12, 70), blur=6, off=(3, 5)))
            draw = ImageDraw.Draw(overlay)
            draw.ellipse([c - d//2, y, c + d//2, y + d], fill=(250, 244, 232, 250), outline=col, width=3)
            bw = draw.textbbox((0,0), word, font=fp); ww = bw[2]-bw[0]; hh = bw[3]-bw[1]
            draw.text((c - ww//2, y + d//2 - hh//2 - 2), word, font=fp, fill=col)
            bc = draw.textbbox((0,0), cap, font=fc); wc = bc[2]-bc[0]
            draw.text((c - wc//2, y + d + 12), cap, font=fc, fill=col)
    return overlay

# ============================================================
def compose(theme_key):
    th = THEMES[theme_key]
    bg = remove_watermark(Image.open(os.path.join(BG_DIR, th["bg"])).convert("RGBA"))
    front, fx, fy, back, bx, by = placements()

    # back first (behind)
    bsh = make_shadow(back, blur=16, opacity=110)
    bg.paste(bsh, (bx + 4, by + 18), bsh)
    bg.paste(back, (bx, by), back)
    # front hero
    fsh = make_shadow(front, blur=22, opacity=140)
    bg.paste(fsh, (fx + 6, fy + 24), fsh)
    bg.paste(front, (fx, fy), front)

    overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    overlay = draw_title(overlay, th)
    overlay = draw_badges(overlay, th)
    final = Image.alpha_composite(bg, overlay)
    out = os.path.join(BASE, theme_key + ".png")
    final.convert("RGB").save(out, "PNG", quality=95)
    print("saved ->", out)

for k in THEMES:
    compose(k)
print("DONE")
