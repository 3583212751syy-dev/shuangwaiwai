#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round 17 — Gemini 高质量场景底图 + 真实 TEVOYATEA 包装合成
设计灵感：即梦 2750（左侧茶壶+中央站立袋+右侧4圆形徽章+大衬线标题）
硬约束：标签100%保留真实素材，绝不AI重绘；输出1024×1024；裁底部50px去水印
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WORK = r"C:/Users/lenovo/WorkBuddy/2026-07-21-10-57-36"
OUT  = r"E:/Desktop/茶叶/成品_第十七轮_Gemini即梦高质量"

FRONT_BAG = r"E:/Desktop/茶叶/成品_第十七轮_Gemini即梦高质量/_assets/product_front_real.png"
BACK_BAG  = r"E:/Desktop/茶叶/成品_第十七轮_Gemini即梦高质量/_assets/product_back_real.png"

FONT_BOLD = r"C:/Windows/Fonts/georgiab.ttf"      # 衬线粗体 — 标题
FONT_REG  = r"C:/Windows/Fonts/georgia.ttf"        # 衬线细 — 副标题/正文
FONT_SANS = r"C:/Windows/Fonts/arialbd.ttf"        # 无衬线粗 — 徽章/小字

# ---------- 工具 ----------
def load_font(path, size):
    try: return ImageFont.truetype(path, size)
    except Exception: return ImageFont.load_default()

def text_wh(draw, text, font):
    b = draw.textbbox((0,0), text, font=font)
    return b[2]-b[0], b[3]-b[1]

def shadow_layer(bag_rgba, opacity=110, blur=14):
    """返回袋子的接触阴影"""
    alpha = bag_rgba.split()[3]
    sh = Image.new("L", bag_rgba.size, 0)
    sh.paste(alpha, (0,0))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    # 下移 + 压暗
    out = Image.new("RGBA", bag_rgba.size, (0,0,0,0))
    mask = sh.point(lambda v: min(255, int(v*1.4)))
    out.putalpha(mask)
    # 整体降透明度
    a = out.split()[3].point(lambda v: int(v*opacity/255))
    out.putalpha(a)
    return out

def place_bag(canvas, bag_path, cx, cy, target_h, shadow=True):
    """把袋子缩放到 target_h 高，中心对齐 (cx,cy)"""
    bag = Image.open(bag_path).convert("RGBA")
    # 内容bbox裁切
    from numpy import asarray, where
    a = asarray(bag.split()[3])
    ys, xs = where(a > 10)
    if len(xs) == 0:
        return None
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
    bag = bag.crop(bbox)
    bw, bh = bag.size
    scale = target_h / bh
    new_w = max(1, int(bw*scale))
    bag = bag.resize((new_w, target_h), Image.LANCZOS)
    # 阴影
    if shadow:
        sh = shadow_layer(bag)
        canvas.alpha_composite(sh, (cx - new_w//2 + 6, cy - target_h//2 + 14))
    # 袋子
    canvas.alpha_composite(bag, (cx - new_w//2, cy - target_h//2))
    return bag

# ---------- 圆形徽章（用粗体字母代替手画icon） ----------
def draw_badge(draw, cx, cy, r, icon_kind, label, fs_label, color_main=(122, 70, 38), color_bg=(250, 244, 232)):
    # 圆背景（暖米色）
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color_bg, outline=(200,165,120), width=2)
    # 用粗体字母作为 icon
    icon_char = {"cup":"C","leaf":"L","drop":"D","cups":"H"}.get(icon_kind, "?")
    fs_icon = load_font(FONT_BOLD, int(r * 0.9))
    tw, th = text_wh(draw, icon_char, fs_icon)
    draw.text((cx - tw//2, cy - th//2 - 4), icon_char, font=fs_icon, fill=color_main)
    # 标签（圆外下方）
    lines = label.split("\n")
    y = cy + r + 6
    for ln in lines:
        tw2, th2 = text_wh(draw, ln, fs_label)
        draw.text((cx - tw2//2, y), ln, font=fs_label, fill=color_main)
        y += th2 + 3

# ---------- 主合成 ----------
def compose(variant):
    bg_path = variant["bg"]
    canvas = Image.open(bg_path).convert("RGBA")
    W, H = canvas.size
    overlay = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(canvas)  # 用于文字

    # === 袋子放置 ===
    if variant["layout"] == "hero_single":
        # 单袋hero：左中下
        place_bag(canvas, FRONT_BAG, cx=int(W*0.42), cy=int(H*0.66), target_h=int(H*0.72))
    elif variant["layout"] == "two_pouches":
        # 双袋：正面hero中左 + 背面右后缩小
        place_bag(canvas, FRONT_BAG, cx=int(W*0.40), cy=int(H*0.65), target_h=int(H*0.70))
        place_bag(canvas, BACK_BAG,  cx=int(W*0.74), cy=int(H*0.60), target_h=int(H*0.50), shadow=True)
    elif variant["layout"] == "lying_diag":
        # 斜躺：袋子倾斜放置在中
        bag = Image.open(FRONT_BAG).convert("RGBA")
        from numpy import asarray, where
        a = asarray(bag.split()[3]); ys, xs = where(a > 10)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
        bag = bag.crop(bbox)
        # 缩放并旋转
        target_h = int(H * 0.62)
        scale = target_h / bag.size[1]
        bag = bag.resize((max(1,int(bag.size[0]*scale)), target_h), Image.LANCZOS)
        bag_r = bag.rotate(-12, expand=True, resample=Image.BICUBIC)
        # 阴影
        sh = shadow_layer(bag_r)
        canvas.alpha_composite(sh, (int(W*0.50)-bag_r.size[0]//2+8, int(H*0.50)-bag_r.size[1]//2+18))
        canvas.alpha_composite(bag_r, (int(W*0.50)-bag_r.size[0]//2, int(H*0.50)-bag_r.size[1]//2))
    elif variant["layout"] == "hero_cozy":
        place_bag(canvas, FRONT_BAG, cx=int(W*0.45), cy=int(H*0.65), target_h=int(H*0.70))
    elif variant["layout"] == "hero_dark":
        place_bag(canvas, FRONT_BAG, cx=int(W*0.45), cy=int(H*0.65), target_h=int(H*0.74))

    # === 标题 + 副标题（顶部居中） ===
    fs_title = load_font(FONT_BOLD, 56)
    fs_sub   = load_font(FONT_REG, 26)
    title = variant["title"]
    sub   = variant["sub"]
    tc    = variant.get("title_color", (95, 40, 25))
    sc    = variant.get("sub_color", (105, 65, 45))
    # 标题两行
    lines = title.split("\n")
    y = 36
    for ln in lines:
        tw, th = text_wh(draw, ln, fs_title)
        # 柔和阴影
        for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
            draw.text(((W-tw)//2+ox, y+oy), ln, font=fs_title, fill=(0,0,0,55))
        draw.text(((W-tw)//2, y), ln, font=fs_title, fill=tc)
        y += th + 4
    # 副标题
    sw, sh = text_wh(draw, sub, fs_sub)
    draw.text(((W-sw)//2, y+4), sub, font=fs_sub, fill=sc)

    # === 右侧徽章列 ===
    if variant.get("badges_right"):
        bx = int(W * 0.88); first_y = int(H * 0.34); gap = int(H * 0.155)
        r = int(H * 0.072)
        fs_b = load_font(FONT_SANS, 17)
        for i, (kind, label) in enumerate(variant["badges_right"]):
            cy = first_y + i*gap
            draw_badge(draw, bx, cy, r, kind, label, fs_b,
                       color_main=(95, 50, 30) if not variant.get("dark") else (220, 195, 150),
                       color_bg=(250, 244, 232) if not variant.get("dark") else (45, 38, 30))

    # === 底部徽章行（P2双袋用）===
    if variant.get("badges_bottom"):
        fs_b = load_font(FONT_SANS, 18)
        n = len(variant["badges_bottom"])
        bw_each = 175; gap_x = 14; total_w = n*bw_each + (n-1)*gap_x
        x0 = (W - total_w)//2
        y0 = H - 120
        for i, (kind, label) in enumerate(variant["badges_bottom"]):
            x = x0 + i*(bw_each+gap_x)
            # 圆角胶囊
            d.rectangle([x, y0, x+bw_each, y0+44], fill=(250,244,232), outline=(200,165,120), width=2)
            tw, th = text_wh(draw, label, fs_b)
            draw.text((x + (bw_each-tw)//2, y0 + (44-th)//2 - 1), label, font=fs_b, fill=(95,50,30))
            # 小icon
            ic_x = x + 22
            ic_y = y0 + 22
            if kind == "cup":
                draw.arc([ic_x-7, ic_y-4, ic_x+7, ic_y+4], 200, 340, fill=(95,50,30), width=2)
                draw.line([ic_x-7, ic_y+4, ic_x+7, ic_y+4], fill=(95,50,30), width=2)
            elif kind == "leaf":
                draw.ellipse([ic_x-7, ic_y-5, ic_x+7, ic_y+5], outline=(95,50,30), width=2)
            elif kind == "drop":
                draw.polygon([(ic_x, ic_y-6),(ic_x-6, ic_y+4),(ic_x+6, ic_y+4)], outline=(95,50,30), width=2)

    # === 底部小标语 ===
    fs_cap = load_font(FONT_REG, 17)
    cap = variant.get("caption", "TEVOYATEA Rooibos Tea · South Africa · NET WT. 100g (3.5 oz)")
    cw, ch = text_wh(draw, cap, fs_cap)
    cap_color = (90, 60, 40) if not variant.get("dark") else (210, 185, 145)
    draw.text(((W-cw)//2, H-66), cap, font=fs_cap, fill=cap_color)

    # === 裁底95px水印 + 保存 ===
    out = canvas.crop((0, 0, W, H-95)).convert("RGB")
    out_path = os.path.join(OUT, variant["name"] + ".png")
    out.save(out_path, "PNG", optimize=True)
    print(f"✓ saved {out_path}  size={out.size}")
    return out_path


# ---------- 5 个变体 ----------
VARIANTS = [
    {
        "name": "01_master_hero",
        "bg": OUT + "/_bg/p1_master_hero.png",
        "layout": "hero_single",
        "title": "SMOOTH & RELAXING\nHERBAL TEA",
        "sub": "A Natural Cup for Everyday Calm",
        "title_color": (110, 45, 30),
        "sub_color": (110, 70, 50),
        "badges_right": [
            ("cup",  "Caffeine\nFree"),
            ("leaf", "Naturally\nSweet"),
            ("drop", "No Artificial\nFlavor"),
            ("cups", "Enjoy Hot\nor Iced"),
        ],
        "caption": "TEVOYATEA Rooibos Tea · 100% Natural · NET WT. 100g (3.5 oz)",
    },
    {
        "name": "02_two_pouches",
        "bg": OUT + "/_bg/p2_two_pouches.png",
        "layout": "two_pouches",
        "title": "AUTHENTIC SOUTH AFRICAN\nRED ROOIBOS TEA",
        "sub": "Hand-picked · Naturally Caffeine-Free · Rich in Antioxidants",
        "title_color": (105, 40, 25),
        "sub_color": (105, 65, 45),
        "badges_bottom": [
            ("leaf", "100% NATURAL"),
            ("cup",  "CAFFEINE FREE"),
            ("drop", "ANTIOXIDANT RICH"),
            ("cups", "HOT OR ICED"),
        ],
        "caption": "TEVOYATEA · Rooibos Tea · 100g · A Blend of Natural Herbs to Help Relieve Stress",
    },
    {
        "name": "03_flatlay_diag",
        "bg": OUT + "/_bg/p3_flatlay.png",
        "layout": "lying_diag",
        "title": "PURE\nROOIBOS",
        "sub": "A cup of South African calm",
        "title_color": (95, 45, 25),
        "sub_color": (100, 65, 40),
        "caption": "TEVOYATEA Rooibos Tea · NET WT. 100g (3.5 oz)",
    },
    {
        "name": "04_cozy_brewing",
        "bg": OUT + "/_bg/p4_cozy.png",
        "layout": "hero_cozy",
        "title": "EASY BREWING\nMETHOD",
        "sub": "Steep 5 min · Enjoy Hot or Over Ice",
        "title_color": (100, 45, 28),
        "sub_color": (105, 65, 45),
        "badges_right": [
            ("cup",  "5 min\nSteep"),
            ("leaf", "100%\nNatural"),
            ("cups", "Hot or\nIced"),
        ],
        "caption": "TEVOYATEA Rooibos Tea · A Blend of Natural Herbs · 100g",
    },
    {
        "name": "05_dark_premium",
        "bg": OUT + "/_bg/p5_dark.png",
        "layout": "hero_dark",
        "title": "PREMIUM ROOIBOS\nEXPERIENCE",
        "sub": "Single-Origin · South African Heritage",
        "title_color": (235, 210, 170),
        "sub_color": (210, 185, 145),
        "dark": True,
        "badges_right": [
            ("leaf", "Single\nOrigin"),
            ("drop", "Antioxidant\nRich"),
            ("cup",  "Caffeine\nFree"),
        ],
        "caption": "TEVOYATEA · Rooibos Tea · 100g · Premium Selection",
    },
]


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for v in VARIANTS:
        if only and only not in v["name"]:
            continue
        if not os.path.exists(v["bg"]):
            print(f"⚠ skip {v['name']}: bg missing {v['bg']}")
            continue
        try:
            compose(v)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"✗ {v['name']} failed: {e}")