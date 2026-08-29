#!/usr/bin/env python
"""
v166d SKULLWING 重画（彻底修复 v166c bug）：
- 每个字母主体 = 完整葫芦形（贯穿全高，无缝衔接）
- 字母底部不要孤立 V 字（删除或改为侧向小毛刺）
- 字符更狭长（高宽比 ≈2.5~3）
- 字符间连笔（不重叠但相邻接）
- 顶部 3 根扇形尖峰
- 上下密集刀刃带
- 字符整体轻微旋转 ±6°（扭曲错落）
"""
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pathlib import Path
import math

ROOT = Path("E:/Desktop/双接口/image-fission")
BASE = ROOT / "jobs" / "smoke_v164" / "v164_metal_6.jpg"
OUT_DIR = ROOT / "jobs" / "smoke_v166d"
OUT_DIR.mkdir(exist_ok=True, parents=True)
DESK = Path("E:/Desktop")

WORD = "SKULLWING"

print(f"[v166d] base={BASE}")
print(f"[v166d] word={WORD} ({len(WORD)} chars)")


def draw_death_letter(draw, glyph, x0, w, h, y_top, color, stroke, rot=0.0):
    """完整葫芦形主体 + 顶部尖峰（无底部 V）"""

    # === 主体：完整葫芦形（贯穿全高，无缝）===
    # K 特殊：3 横竖
    if glyph == "K":
        polys = []
        # 主竖
        polys.append([
            (x0 + w*0.28, y_top + h*0.05),
            (x0 + w*0.55, y_top + h*0.05),
            (x0 + w*0.55, y_top + h*0.95),
            (x0 + w*0.28, y_top + h*0.95),
        ])
        # 右上斜
        polys.append([
            (x0 + w*0.50, y_top + h*0.40),
            (x0 + w*0.70, y_top + h*0.40),
            (x0 + w*0.92, y_top + h*0.05),
            (x0 + w*0.78, y_top + h*0.05),
        ])
        # 右下斜
        polys.append([
            (x0 + w*0.50, y_top + h*0.55),
            (x0 + w*0.70, y_top + h*0.55),
            (x0 + w*0.95, y_top + h*0.95),
            (x0 + w*0.78, y_top + h*0.95),
        ])
    else:
        # 简洁菱形主体（4 边，绝不自交叉）
        polys = []
        main_diamond = [
            (x0 + w*0.50, y_top + h*0.05),  # 顶
            (x0 + w*0.78, y_top + h*0.50),  # 右
            (x0 + w*0.50, y_top + h*0.95),  # 底
            (x0 + w*0.22, y_top + h*0.50),  # 左
        ]
        polys.append(main_diamond)
        # 左突出（独立多边形）
        polys.append([
            (x0 + w*0.22, y_top + h*0.30),
            (x0 + w*0.08, y_top + h*0.42),
            (x0 + w*0.04, y_top + h*0.50),
            (x0 + w*0.08, y_top + h*0.58),
            (x0 + w*0.22, y_top + h*0.70),
        ])
        # 右突出（独立多边形）
        polys.append([
            (x0 + w*0.78, y_top + h*0.30),
            (x0 + w*0.92, y_top + h*0.42),
            (x0 + w*0.96, y_top + h*0.50),
            (x0 + w*0.92, y_top + h*0.58),
            (x0 + w*0.78, y_top + h*0.70),
        ])

    # === 顶部 3 根扇形尖峰 ===
    spike_h = h * 0.28
    for sx, sw in [(0.32, 0.20), (0.50, 0.14), (0.68, 0.20)]:
        # 三角尖
        polys.append([
            (x0 + w*sx - w*sw/2, y_top + h*0.05),
            (x0 + w*sx + w*sw/2, y_top + h*0.05),
            (x0 + w*sx + w*0.04, y_top - spike_h*0.88),
            (x0 + w*sx - w*0.04, y_top - spike_h*0.88),
        ])

    # === 应用旋转 ===
    if rot != 0:
        cx = x0 + w/2
        cy = y_top + h/2
        cos = math.cos(math.radians(rot))
        sin = math.sin(math.radians(rot))
        new_polys = []
        for poly in polys:
            new_polys.append([
                (cx + (p[0]-cx)*cos - (p[1]-cy)*sin,
                 cy + (p[0]-cx)*sin + (p[1]-cy)*cos)
                for p in poly
            ])
        polys = new_polys

    # === 绘制 ===
    for p in polys:
        draw.polygon(p, fill=color, outline=stroke)


def draw_logo(logo_h, logo_w, word=WORD, color=(248, 244, 230), stroke=(0, 0, 0)):
    """绘制整段 logo 到透明 RGBA 图层"""
    layer = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # 字符宽度（极狭长，高宽比 ~3.5），横贯全宽
    n = len(word)
    char_w_fixed = 130     # 固定字符宽（窄）
    char_h = int(logo_h * 0.55)   # 字符主体高，logo_h*0.55 = 473px
    char_y_top = (logo_h - char_h) // 2 + int(logo_h * 0.05)  # 偏下，给顶部尖峰+刀刃带更多空间

    # 横贯全宽：margin=40，gap 由总宽算
    margin = 60
    total_chars_w = char_w_fixed * n
    gap = (logo_w - 2*margin - total_chars_w) / (n - 1)
    char_x_start = margin

    # 旋转错落（±8° 伪随机）
    for i, ch in enumerate(word):
        x = char_x_start + i * (char_w_fixed + gap)
        rot = ((i * 7 + 3) % 17) - 8  # -8 ~ +8
        draw_death_letter(draw, ch, x, char_w_fixed, char_h, char_y_top, color, stroke, rot)

    # === 上下刀刃带 ===
    blade_color = (252, 250, 245)
    blade_stroke = (0, 0, 0)

    # 顶部刀刃带（位置在字符顶部尖峰之上）
    blade_y_top = char_y_top - char_h * 0.55
    n_blades = 24
    margin_b = 80
    for i in range(n_blades):
        bx = margin_b + (logo_w - 2*margin_b) * i / (n_blades - 1)
        bsz = (char_h * 0.10) if i % 2 == 0 else (char_h * 0.16)
        bw = char_w_fixed * 0.18
        poly = [
            (bx - bw, blade_y_top),
            (bx + bw, blade_y_top),
            (bx + bw*0.4, blade_y_top - bsz),
            (bx - bw*0.4, blade_y_top - bsz),
            (bx, blade_y_top - bsz*1.20),
        ]
        draw.polygon(poly, fill=blade_color, outline=blade_stroke)

    # 底部刀刃带
    blade_y_bot = char_y_top + char_h * 0.96
    for i in range(n_blades):
        bx = margin_b + (logo_w - 2*margin_b) * i / (n_blades - 1)
        bsz = (char_h * 0.10) if i % 2 == 0 else (char_h * 0.16)
        bw = char_w_fixed * 0.18
        poly = [
            (bx - bw, blade_y_bot),
            (bx + bw, blade_y_bot),
            (bx, blade_y_bot + bsz*1.20),
            (bx + bw*0.4, blade_y_bot + bsz),
            (bx - bw*0.4, blade_y_bot + bsz),
        ]
        draw.polygon(poly, fill=blade_color, outline=blade_stroke)

    return layer


# === 主流程 ===
print("[load] base image")
base = Image.open(BASE).convert("RGBA")
W, H = base.size
print(f"[size] {W}x{H}")

# 1. 顶部 logo 区域涂黑（覆盖原 v164 SDXL 复刻 logo + 字里穿透的闪电）
logo_h = int(H * 0.16)
black_strip = Image.new("RGBA", (W, logo_h), (0, 0, 0, 255))
mask = Image.new("L", (W, logo_h), 0)
mdraw = ImageDraw.Draw(mask)
mdraw.rectangle([(0, 25), (W, logo_h)], fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(radius=20))
base.paste(black_strip, (0, 0), mask)
print(f"[strip] logo region 0-{logo_h}px blackened, feathered")

# 2. 绘制死亡金属 logo
print("[logo] drawing SKULLWING death-metal-style logo...")
logo_layer = draw_logo(logo_h=logo_h, logo_w=W, word=WORD, color=(248, 244, 230), stroke=(0, 0, 0))

# DEBUG: 单独保存 logo_layer 看
LOGO_ONLY = OUT_DIR / "_debug_logo_layer.png"
logo_layer.save(LOGO_ONLY)
print(f"[DEBUG] logo_layer saved {LOGO_ONLY} | size={logo_layer.size}")

# 3. 合成到黑色 banner 上
banner = base.crop((0, 0, W, logo_h)).convert("RGBA")
# DEBUG: save banner pre-composite
banner.save(OUT_DIR / "_debug_banner_pre.png")
print(f"[DEBUG] banner pre-composite saved, size={banner.size}")
banner.alpha_composite(logo_layer)
# DEBUG: save banner post-composite
banner.save(OUT_DIR / "_debug_banner_post.png")
print(f"[DEBUG] banner post-composite saved")
base.paste(banner, (0, 0))
print("[composite] logo composited")

# 4. 红橙分隔线（呼应图火焰配色）
draw = ImageDraw.Draw(base)
sep_y = logo_h - 3
for offset in range(2):
    draw.rectangle([(0, sep_y + offset*2), (W, sep_y + offset*2 + 1)], fill=(220, 90, 30))

# 5. 转 RGB + USM 锐化
out = base.convert("RGB")
out = out.filter(ImageFilter.UnsharpMask(radius=3, percent=150, threshold=3))

# 6. 保存
LOCAL_OUT = OUT_DIR / "v166d_metal_6_logo.jpg"
DESK_OUT = DESK / "image-fission-v166d-metal_6-SKULLWING-logo.jpg"
out.save(LOCAL_OUT, "JPEG", quality=92, optimize=True)
out.save(DESK_OUT, "JPEG", quality=92, optimize=True)
print(f"[ok] local: {LOCAL_OUT} ({LOCAL_OUT.stat().st_size/1024/1024:.2f}MB)")
print(f"[ok] desk:  {DESK_OUT} ({DESK_OUT.stat().st_size/1024/1024:.2f}MB)")

# 7. 拼图对照
print("[compare] building 2-way comparison")
orig = Image.open(BASE).convert("RGB").copy()
new = Image.open(DESK_OUT).convert("RGB").copy()

H_cmp = 1000
def resize_h(im, H):
    w = int(im.width * H / im.height)
    return im.resize((w, H), Image.LANCZOS)

orig_r = resize_h(orig, H_cmp)
new_r = resize_h(new, H_cmp)
gap = 30
total_w = orig_r.width + new_r.width + gap
cmp = Image.new("RGB", (total_w, H_cmp + 80), (20, 20, 20))
cmp.paste(orig_r, (0, 60))
cmp.paste(new_r, (orig_r.width + gap, 60))

cmp_draw = ImageDraw.Draw(cmp)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 32)
except:
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
    except:
        font = ImageFont.load_default()
cmp_draw.text((20, 12), "v164 原图 (SDXL 复刻侵权字符 - logo 形乱)", fill=(255, 200, 100), font=font)
cmp_draw.text((orig_r.width + gap + 20, 12), "v166d SKULLWING (PIL polygon 程序化手绘死亡金属 logo 形态)", fill=(100, 220, 255), font=font)

CMP_LOCAL = OUT_DIR / "compare_v166d.jpg"
CMP_DESK = DESK / "image-fission-v166d-metal_6-SKULLWING-LOGO-compare.jpg"
cmp.save(CMP_LOCAL, "JPEG", quality=90)
cmp.save(CMP_DESK, "JPEG", quality=90)
print(f"[compare] local: {CMP_LOCAL}")
print(f"[compare] desk:  {CMP_DESK}")
print("[done]")
