#!/usr/bin/env python
"""
v166c SKULLWING 重画：用 PIL ImageDraw.polygon 程序化手绘死亡金属 logo
完全复制原图 logo 形态（刺冠 + 字符间互连 + 上下刀刃带 + 高度扭曲 + 白底反白 + 强烈狭长）

核心方法：每个字母 = 主体多边形 + 顶部尖峰 + 底部V字锯齿 + 字符扭曲旋转
整体 = 9个字母 + 连笔带 + 上下密集刀刃带
"""
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import math

ROOT = Path("E:/Desktop/双接口/image-fission")
BASE = ROOT / "jobs" / "smoke_v164" / "v164_metal_6.jpg"  # v164 干净底图，未加我的logo
OUT_DIR = ROOT / "jobs" / "smoke_v166c"
OUT_DIR.mkdir(exist_ok=True, parents=True)
DESK = Path("E:/Desktop")

WORD = "SKULLWING"  # 8 字, 骷髅+翼, 0 侵权
print(f"[v166c] base={BASE}")
print(f"[v166c] word={WORD} len={len(WORD)}")


def draw_death_letter(draw, glyph, x0, w, h, y_top, color, stroke, rot=0.0):
    """
    程序化手绘死亡金属 logo 单字母
    glyph: 字母字符
    x0: 字母左边缘 x 坐标
    w: 字母宽度
    h: 字母主体高度
    y_top: 字母主体顶端 y 坐标
    color: 字母填充色
    stroke: 描边色
    rot: 旋转角度（度）

    视觉特征：垂直狭长 + 顶部 3 根尖峰 + 底部 V 字锯齿 + 主体凹槽菱形
    """
    polys = []

    # === 主体（带凹槽菱形）===
    if glyph in "SILWNG":  # 直笔型
        main = [
            (x0 + w*0.25, y_top + h*0.05),
            (x0 + w*0.75, y_top + h*0.05),
            (x0 + w*0.85, y_top + h*0.45),
            (x0 + w*0.5,  y_top + h*0.55),  # 中央凹
            (x0 + w*0.15, y_top + h*0.45),
        ]
    elif glyph in "KUX":  # 主笔+斜笔
        # 主竖笔
        polys.append([
            (x0 + w*0.3, y_top + h*0.05),
            (x0 + w*0.55, y_top + h*0.05),
            (x0 + w*0.55, y_top + h*0.95),
            (x0 + w*0.3, y_top + h*0.95),
        ])
        # 右上斜笔
        polys.append([
            (x0 + w*0.45, y_top + h*0.4),
            (x0 + w*0.65, y_top + h*0.4),
            (x0 + w*0.85, y_top + h*0.05),
            (x0 + w*0.7,  y_top + h*0.05),
        ])
        polys.append([
            (x0 + w*0.45, y_top + h*0.55),
            (x0 + w*0.65, y_top + h*0.55),
            (x0 + w*0.9, y_top + h*0.95),
            (x0 + w*0.7,  y_top + h*0.95),
        ])
        main = None
    else:
        main = [
            (x0 + w*0.2, y_top + h*0.05),
            (x0 + w*0.8, y_top + h*0.05),
            (x0 + w*0.9, y_top + h*0.5),
            (x0 + w*0.5, y_top + h*0.65),
            (x0 + w*0.1, y_top + h*0.5),
        ]

    if main:
        polys.append(main)

    # === 顶部 3 根扇形尖峰 ===
    spike_h = h * 0.20
    for sx, sw in [(0.30, 0.22), (0.55, 0.16), (0.78, 0.20)]:
        polys.append([
            (x0 + w*sx - w*sw/2, y_top + h*0.02),
            (x0 + w*sx + w*sw/2, y_top + h*0.02),
            (x0 + w*sx + w*0.04, y_top - spike_h*0.85),
            (x0 + w*sx - w*0.04, y_top - spike_h*0.85),
        ])

    # === 底部 2 根 V 字锯齿 ===
    bot_h = h * 0.18
    polys.append([
        (x0 + w*0.15, y_top + h*0.9),
        (x0 + w*0.45, y_top + h*0.95 + bot_h*0.3),
        (x0 + w*0.5,  y_top + h*0.92),
        (x0 + w*0.55, y_top + h*0.95 + bot_h*0.7),
        (x0 + w*0.85, y_top + h*0.9),
        (x0 + w*0.7,  y_top + h*0.95),
        (x0 + w*0.5,  y_top + h*0.97),
        (x0 + w*0.3,  y_top + h*0.95),
    ])

    # === 画（应用旋转）===
    if rot != 0:
        cx = x0 + w/2
        cy = y_top + h/2
        cos = math.cos(math.radians(rot))
        sin = math.sin(math.radians(rot))
        rotated = []
        for poly in polys:
            rotated.append([
                (cx + (p[0]-cx)*cos - (p[1]-cy)*sin,
                 cy + (p[0]-cx)*sin + (p[1]-cy)*cos)
                for p in poly
            ])
        polys = rotated

    for p in polys:
        draw.polygon(p, fill=color, outline=stroke)


def draw_logo(logo_h, logo_w, x0, y0, word=WORD, color=(245, 240, 232), stroke=(0, 0, 0)):
    """
    在透明 RGBA 图层上画完整 logo
    返回图层 Image（RGBA）
    """
    layer = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    n = len(word)
    char_w = logo_w / (n * 1.15)  # 字母宽（含间距）
    char_h = logo_h * 0.55        # 字母主体高度

    # === 字符分布（横贯 logo）===
    total_chars_w = char_w * n * 0.95
    char_y_top = (logo_h - char_h) * 0.5

    # 字符绘制位置
    char_x_start = (logo_w - total_chars_w) / 2

    # 逐字旋转错落（旋转 ±7° 模仿原 logo 的歪斜扭曲感）
    rng_seed = 7
    for i, ch in enumerate(word):
        x = char_x_start + i * char_w * 1.05
        rot = (i * 17 + 3) % 13 - 6  # ±6° 伪随机错落
        draw_death_letter(draw, ch, x, char_w*0.9, char_h, char_y_top, color, stroke, rot)

    # === 上下刀刃带（密集三角刺横贯 logo）===
    blade_color = (245, 240, 232)
    blade_stroke = (0, 0, 0)

    # 顶部刀刃带
    blade_y_top = char_y_top - logo_h * 0.22
    blade_h = logo_h * 0.10
    n_blades = 24
    for i in range(n_blades):
        bx = (logo_w - 60) * i / (n_blades - 1) + 30
        # 大小交替
        bsz = (logo_h * 0.07) if i % 2 == 0 else (logo_h * 0.10)
        poly = [
            (bx - 18, blade_y_top + blade_h),
            (bx + 18, blade_y_top + blade_h),
            (bx, blade_y_top + blade_h - bsz),
            (bx - 4, blade_y_top + blade_h - bsz*0.3),
            (bx + 4, blade_y_top + blade_h - bsz*0.3),
        ]
        draw.polygon(poly, fill=blade_color, outline=blade_stroke)

    # 底部刀刃带
    blade_y_bot = char_y_top + char_h + logo_h * 0.10
    for i in range(n_blades):
        bx = (logo_w - 60) * i / (n_blades - 1) + 30
        bsz = (logo_h * 0.07) if i % 2 == 0 else (logo_h * 0.10)
        poly = [
            (bx - 18, blade_y_bot),
            (bx + 18, blade_y_bot),
            (bx, blade_y_bot + bsz),
            (bx - 4, blade_y_bot + bsz*0.3),
            (bx + 4, blade_y_bot + bsz*0.3),
        ]
        draw.polygon(poly, fill=blade_color, outline=blade_stroke)

    return layer


# === 主流程 ===
print("[load] base image")
base = Image.open(BASE).convert("RGBA")
W, H = base.size
print(f"[size] {W}x{H}")

# 1. 顶部 logo 区域涂纯黑（覆盖原 v164 SDXL 复刻的 logo 字 + 穿透字里的闪电）
logo_h = int(H * 0.16)  # logo 区高度 16% 顶
black_strip = Image.new("RGBA", (W, logo_h), (0, 0, 0, 255))
# 用原图底纹平滑过渡：上沿羽化 20px
mask = Image.new("L", (W, logo_h), 0)
mdraw = ImageDraw.Draw(mask)
mdraw.rectangle([(0, 20), (W, logo_h)], fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(radius=20))

base.paste(black_strip, (0, 0), mask)
print(f"[strip] logo region 0-{logo_h}px blackened with feathered top edge")

# 2. 手绘死亡金属 logo（白色字体 + 黑色描边 + 白底反白）
print("[logo] drawing SKULLWING death-metal-style logo...")
logo_layer = draw_logo(
    logo_h=logo_h,
    logo_w=W,
    x0=0, y0=0,
    color=(248, 244, 230),       # 米白
    stroke=(0, 0, 0),
)

# 3. 合成 logo 到黑色 banner
banner = base.crop((0, 0, W, logo_h)).convert("RGBA")
banner.alpha_composite(logo_layer)
base.paste(banner, (0, 0))
print(f"[composite] logo composited into top {logo_h}px banner")

# 4. 加一条红橙分隔线（呼应图火焰配色）
draw = ImageDraw.Draw(base)
sep_y = logo_h - 2
draw.rectangle([(0, sep_y), (W, sep_y + 1)], fill=(220, 90, 30))

# 5. 转回 RGB + USM 锐化
out = base.convert("RGB")
out = out.filter(ImageFilter.UnsharpMask(radius=3, percent=150, threshold=3))

# 6. 保存
LOCAL_OUT = OUT_DIR / "v166c_metal_6_logo_skullwing.jpg"
DESK_OUT = DESK / "image-fission-v166c-metal_6-SKULLWING-logo.jpg"

out.save(LOCAL_OUT, "JPEG", quality=92, optimize=True)
# 桌面副本
out.save(DESK_OUT, "JPEG", quality=92, optimize=True)
print(f"[ok] local: {LOCAL_OUT} ({LOCAL_OUT.stat().st_size/1024/1024:.2f}MB)")
print(f"[ok] desk:  {DESK_OUT} ({DESK_OUT.stat().st_size/1024/1024:.2f}MB)")

# 7. 拼图对照
print("[compare] building 3-way comparison...")
from PIL import ImageFont

orig = Image.open(BASE).convert("RGB").copy()
new = Image.open(DESK_OUT).convert("RGB").copy()
orig_w, orig_h = orig.size

# resize to height 800
H_cmp = 800
def resize_h(im, H):
    w = int(im.width * H / im.height)
    return im.resize((w, H), Image.LANCZOS)

orig_r = resize_h(orig, H_cmp)
new_r = resize_h(new, H_cmp)
gap = 20
total_w = orig_r.width + new_r.width + gap
cmp = Image.new("RGB", (total_w, H_cmp + 60), (20, 20, 20))
cmp.paste(orig_r, (0, 50))
cmp.paste(new_r, (orig_r.width + gap, 50))

cmp_draw = ImageDraw.Draw(cmp)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
except:
    font = ImageFont.load_default()
cmp_draw.text((20, 12), "v164 原图(SDXL复刻侵权字符)", fill=(255, 200, 100), font=font)
cmp_draw.text((orig_r.width + gap + 20, 12), "v166c SKULLWING (PIL手绘死亡金属logo)", fill=(100, 220, 255), font=font)

CMP_OUT = OUT_DIR / "compare_v166c.jpg"
CMP_DESK = DESK / "image-fission-v166c-metal_6-SKULLWING-LOGO-compare.jpg"
cmp.save(CMP_OUT, "JPEG", quality=90)
cmp.save(CMP_DESK, "JPEG", quality=90)
print(f"[compare] local: {CMP_OUT}")
print(f"[compare] desk:  {CMP_DESK}")
print("[done]")
