"""Round 18 — Compose D v2 with FLAT WARM YELLOW bg (matches 即梦 4847 style).
- 1280x800 canvas
- Flat warm yellow gradient background
- Big red bold serif title (2 lines, white halo)
- Subtitle on cream pill
- Real TEVOYATEA pouch centered (smaller, h=460 to avoid subtitle overlap)
- 4 corner circular badges with line icons + labels
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# === paths ===
BG = r"E:/Desktop/茶叶/成品_第十八轮_品牌叙事D/_bg/Flat_premium_warm_yellow_and_c_2026-07-24T03-20-39.png"
BG_LIFESTYLE = r"E:/Desktop/茶叶/成品_第十八轮_品牌叙事D/_bg/Warm_cream_and_soft_butter_yel_2026-07-24T03-14-47.png"
BAG = r"E:/Desktop/茶叶/成品_第十八轮_品牌叙事D/_assets/product_front_real.png"
OUT_DIR = r"E:/Desktop/茶叶/成品_第十八轮_品牌叙事D"
os.makedirs(OUT_DIR, exist_ok=True)

FONT_SANS  = r"C:/Windows/Fonts/arialbd.ttf"
FONT_REG   = r"C:/Windows/Fonts/arial.ttf"
FONT_SERIF = r"C:/Windows/Fonts/georgiab.ttf"

def f(path, size): return ImageFont.truetype(path, size)
def tw(draw, text, font):
    bbox = draw.textbbox((0,0), text, font=font)
    return bbox[2]-bbox[0], bbox[3]-bbox[1]

def load_bg(path, W, H, crop="center", trim_bottom_pct=0.10):
    """Load bg image, scale to fill WxH (cover), trim bottom watermark area, crop to WxH."""
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    # crop bottom (skip watermark band)
    trim_y = int(ih * trim_bottom_pct)
    im = im.crop((0, 0, iw, ih - trim_y))
    iw, ih = im.size
    scale = max(W/iw, H/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    im = im.resize((nw,nh), Image.LANCZOS)
    if crop == "center":
        x0 = (nw - W)//2; y0 = (nh - H)//2
    elif crop == "top":
        x0 = (nw - W)//2; y0 = 0
    else:
        x0 = 0; y0 = 0
    return im.crop((x0,y0,x0+W,y0+H))

# ----------- icon drawing helpers -----------
def icon_no_bean(d, cx, cy, col, w=6, s=1.0):
    # two crossed beans (caffeine-free meaning)
    d.ellipse([cx-32*s, cy-14*s, cx-12*s, cy+14*s], outline=col, width=w)
    d.line([cx-30*s, cy-8*s, cx-14*s, cy+8*s], fill=col, width=w-2)
    d.line([cx-14*s, cy-8*s, cx-30*s, cy+8*s], fill=col, width=w-2)
    d.ellipse([cx+12*s, cy-14*s, cx+32*s, cy+14*s], outline=col, width=w)
    d.line([cx+14*s, cy-8*s, cx+30*s, cy+8*s], fill=col, width=w-2)
    d.line([cx+30*s, cy-8*s, cx+14*s, cy+8*s], fill=col, width=w-2)
def icon_smile(d, cx, cy, col, w=6):
    d.ellipse([cx-28, cy-26, cx+28, cy+30], outline=col, width=w)
    d.arc([cx-18, cy-12, cx-10, cy-2], 200, 340, fill=col, width=w-2)
    d.arc([cx+10, cy-12, cx+18, cy-2], 200, 340, fill=col, width=w-2)
    d.arc([cx-14, cy-2, cx+14, cy+18], 200, 340, fill=col, width=w)
def icon_antioxidant(d, cx, cy, col, w=6):
    import math
    r = 28
    pts = [(cx + r*math.cos(math.radians(60*i+30)), cy + r*math.sin(math.radians(60*i+30))) for i in range(6)]
    d.polygon(pts, outline=col, width=w)
    d.line([cx-12, cy+14, cx, cy-16], fill=col, width=w-2)
    d.line([cx, cy-16, cx+12, cy+14], fill=col, width=w-2)
    d.line([cx-6, cy+2, cx+6, cy+2], fill=col, width=w-2)
def icon_two_cups(d, cx, cy, col, w=6):
    d.arc([cx-32, cy-10, cx-10, cy+16], 200, 340, fill=col, width=w)
    d.line([cx-32, cy+16, cx-10, cy+16], fill=col, width=w)
    d.line([cx-26, cy-18, cx-26, cy-26], fill=col, width=w-2)
    d.line([cx-16, cy-18, cx-16, cy-26], fill=col, width=w-2)
    d.arc([cx+10, cy-10, cx+32, cy+16], 200, 340, fill=col, width=w)
    d.line([cx+10, cy+16, cx+32, cy+16], fill=col, width=w)
    d.line([cx+22, cy-18, cx+28, cy+22], fill=col, width=w-2)
    d.rectangle([cx+18, cy-2, cx+24, cy+4], outline=col, width=w-2)

def draw_badge(draw, cx, cy, r, icon_fn, label, color_main, color_bg, color_ring, fs):
    draw.ellipse([cx-r-3, cy-r-3, cx+r+3, cy+r+3], outline=color_ring, width=4)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color_bg, outline=color_main, width=2)
    icon_fn(draw, cx, cy-2, color_main)
    pad_x, pad_y = 10, 5
    lw, lh = tw(draw, label, fs)
    bx0 = cx - lw//2 - pad_x
    by0 = cy + r + 6
    bx1 = cx + lw//2 + pad_x
    by1 = by0 + lh + pad_y*2
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill=color_bg, outline=color_main, width=1)
    draw.text((cx - lw//2, by0 + pad_y - 1), label, font=fs, fill=color_main)

W, H = 1280, 800
TITLE1, TITLE2 = "NATURAL ROOIBOS", "GOODNESS"
SUB = "South African Red Bush Tea · Caffeine-Free Herbal Blend"
TITLE_RED = (193, 53, 41)
SUB_DARK  = (78, 50, 30)
C_RING = (193, 53, 41)
C_BG   = (255, 250, 240)

# ---------- bag asset prepare (crop to content bbox) ----------
bag = Image.open(BAG).convert("RGBA")
ab = np.array(bag); a3 = ab[:,:,3]
ys2, xs2 = np.where(a3>30)
bx0,by0 = int(xs2.min()), int(ys2.min())
bbw,bbh = int(xs2.max()-xs2.min()), int(ys2.max()-ys2.min())
bag_c = bag.crop((bx0,by0,bx0+bbw,by0+bbh))
print("bag content", bag_c.size, "aspect w/h", round(bbw/bbh,3))

# ============================================================
# Variant A — PRIMARY: flat warm yellow bg (matches 4847)
# ============================================================
bg = load_bg(BG, W, H, crop="center", trim_bottom_pct=0.10)
canvas = bg.convert("RGB")
draw = ImageDraw.Draw(canvas)

# title with white halo
fs_title = f(FONT_SERIF, 76)
halo = (-2,0),(2,0),(0,-2),(0,2)
for ox,oy in halo:
    draw.text(((W-tw(draw,TITLE1,fs_title)[0])//2 + ox, 56+oy),  TITLE1, font=fs_title, fill=(255,255,255))
    draw.text(((W-tw(draw,TITLE2,fs_title)[0])//2 + ox, 138+oy), TITLE2, font=fs_title, fill=(255,255,255))
draw.text(((W-tw(draw,TITLE1,fs_title)[0])//2, 56),  TITLE1, font=fs_title, fill=TITLE_RED)
draw.text(((W-tw(draw,TITLE2,fs_title)[0])//2, 138), TITLE2, font=fs_title, fill=TITLE_RED)

# subtitle (cream pill)
fs_sub = f(FONT_SANS, 22)
sw, sh = tw(draw, SUB, fs_sub)
draw.rounded_rectangle([(W-sw)//2 - 14, 218, (W+sw)//2 + 14, 218+sh+10], radius=10, fill=(255,250,240))
draw.text(((W-sw)//2, 222), SUB, font=fs_sub, fill=SUB_DARK)

# product (height 460, centered, bottom y=720)
target_h = 460
rw = int(bbw * (target_h/bbh))
bag_s = bag_c.resize((rw,target_h), Image.LANCZOS)
bx0 = (W - rw)//2
by0 = 720 - target_h
canvas_rgba = canvas.convert("RGBA")
canvas_rgba.alpha_composite(bag_s, (bx0, by0))
# contact shadow
shadow = Image.new("RGBA", (W,H), (0,0,0,0))
sd = ImageDraw.Draw(shadow)
sd.ellipse([bx0+rw*0.18, by0+target_h-22, bx0+rw*0.82, by0+target_h+10], fill=(0,0,0,90))
shadow = shadow.filter(ImageFilter.GaussianBlur(18))
canvas = Image.alpha_composite(canvas_rgba, shadow).convert("RGB")
draw = ImageDraw.Draw(canvas)

# 4 corner badges
fs_b = f(FONT_SANS, 18)
R = 58
badges = [
    (175, 380, icon_no_bean,    "CAFFEINE-FREE"),
    (1105, 380, icon_smile,     "DAILY WELLNESS"),
    (175, 640, icon_antioxidant,"ANTIOXIDANT-RICH"),
    (1105, 640, icon_two_cups,  "HOT OR ICED"),
]
for cx,cy, icn, lbl in badges:
    draw_badge(draw, cx, cy, R, icn, lbl, SUB_DARK, C_BG, C_RING, fs_b)

canvas.save(os.path.join(OUT_DIR, "03_D_flat_yellow.png"), "PNG")
print("✓ 03_D_flat_yellow.png  ", canvas.size)

# ============================================================
# Variant B — LIFESTYLE: bg1 (hands+cup blurred top), keeps brand-story warmth
# Smaller bag, halo title, same badges.
# ============================================================
bg2 = load_bg(BG_LIFESTYLE, W, H, crop="top", trim_bottom_pct=0.18)
canvas2 = bg2.convert("RGB")
d2 = ImageDraw.Draw(canvas2)

# lighten top so title pops above hands (the hands+cup area is busy)
# blend a translucent cream over the upper 220px
overlay = Image.new("RGB", (W, 220), (252, 240, 210))
mask_o = Image.new("L", (W, 220), 200)  # 78% cream blend
canvas2.paste(overlay, (0,0), mask_o)
d2 = ImageDraw.Draw(canvas2)

# title (same red, bigger halo because top still has some hands)
for ox,oy in [(-2,0),(2,0),(0,-2),(0,2),(-3,0),(3,0),(0,-3),(0,3)]:
    d2.text(((W-tw(d2,TITLE1,fs_title)[0])//2 + ox, 56+oy),  TITLE1, font=fs_title, fill=(255,255,255))
    d2.text(((W-tw(d2,TITLE2,fs_title)[0])//2 + ox, 138+oy), TITLE2, font=fs_title, fill=(255,255,255))
d2.text(((W-tw(d2,TITLE1,fs_title)[0])//2, 56),  TITLE1, font=fs_title, fill=TITLE_RED)
d2.text(((W-tw(d2,TITLE2,fs_title)[0])//2, 138), TITLE2, font=fs_title, fill=TITLE_RED)

# subtitle
d2.rounded_rectangle([(W-sw)//2 - 14, 218, (W+sw)//2 + 14, 218+sh+10], radius=10, fill=(255,250,240))
d2.text(((W-sw)//2, 222), SUB, font=fs_sub, fill=SUB_DARK)

# product
canvas2_rgba = canvas2.convert("RGBA")
canvas2_rgba.alpha_composite(bag_s, (bx0, by0))
canvas2 = Image.alpha_composite(canvas2_rgba, shadow).convert("RGB")
d2 = ImageDraw.Draw(canvas2)
for cx,cy, icn, lbl in badges:
    draw_badge(d2, cx, cy, R, icn, lbl, SUB_DARK, C_BG, C_RING, fs_b)

canvas2.save(os.path.join(OUT_DIR, "04_D_lifestyle.png"), "PNG")
print("✓ 04_D_lifestyle.png  ", canvas2.size)
