"""v132 Phase A 拼图对照 (6 行 × 2 列: 原图 | v132A 出图) + 评分标签."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

JOB_DIR = Path("E:/Desktop/双接口/image-fission/jobs/smoke_v132_phaseA_1787907296")
OUT = JOB_DIR / "v132A_compare_6x2.png"

# (name, src原图, dst出图, score, hint)
PAIRS = [
    ("denim_3", "E:/Desktop/图裂变测试图/pinterest_denim_3.jpg",
     str(JOB_DIR / "v132A_denim_3.png"),
     "8.5/10", "牛仔 UPCY 蝴蝶 -> 牛仔双蝴蝶+小蝴蝶 (内容大改)"),
    ("camo_4", "E:/Desktop/图裂变测试图/pinterest_camo_4.jpg",
     str(JOB_DIR / "v132A_camo_4.png"),
     "9/10", "棕榈迷彩 -> 松针雪峰雪山迷彩 (完美裂变)"),
    ("eagle_2", "E:/Desktop/图裂变测试图/pinterest_eagle_2.jpg",
     str(JOB_DIR / "v132A_eagle_2.png"),
     "9/10", "鹰+仿牌徽章 -> 双鹰凤凰+火焰+盾徽 (重装饰)"),
    ("illust_1", "E:/Desktop/图裂变测试图/pinterest_illust_1.jpg",
     str(JOB_DIR / "v132A_illust_1.png"),
     "6/10", "莲花藤蔓图章 -> 莲花藤蔓图章 (跟原图太像, 裂变不足)"),
    ("metal_6", "E:/Desktop/图裂变测试图/pinterest_metal_6.jpg",
     str(JOB_DIR / "v132A_metal_6.png"),
     "8.5/10", "金属浮雕死金 -> 双鹰碎裂银金属 (金属族系锁定)"),
    ("skull_5", "E:/Desktop/图裂变测试图/pinterest_skull_5.jpg",
     str(JOB_DIR / "v132A_skull_5.png"),
     "8.5/10", "骷髅+红花 -> 骷髅+红玫瑰+藤蔓 (待 Phase B 加文字)"),
]

THUMB_W = 380
PAD = 14
LABEL_H = 56
ROW_H = 540  # each row max


def thumb(path, w):
    im = Image.open(path).convert("RGB")
    ratio = w / im.size[0]
    h = int(im.size[1] * ratio)
    return im.resize((w, h), Image.LANCZOS)


def try_font(size):
    for p in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


f_title = try_font(22)
f_hint = try_font(15)
f_score = try_font(20)

cols = 2
W = THUMB_W * cols + PAD * (cols + 1)
H = ROW_H * len(PAIRS) + PAD * (len(PAIRS) + 1) + LABEL_H * 2
canvas = Image.new("RGB", (W, H), (245, 245, 248))
draw = ImageDraw.Draw(canvas)

# Title bar
draw.text((PAD, PAD), "v132 Phase A - Pure Fission (No IPA / No CN, base SDXL, denoise 0.95)",
          fill=(20, 20, 20), font=f_title)
draw.text((PAD, PAD + 28), "Left: original  |  Right: v132A output  |  Colors slightly drift (next phase adds Reinhard lock)",
          fill=(80, 80, 80), font=f_hint)

y = LABEL_H + PAD * 2
for i, (name, src, dst, score, hint) in enumerate(PAIRS):
    # row bg
    bg = (255, 255, 255) if i % 2 == 0 else (250, 250, 252)
    draw.rectangle([0, y, W, y + ROW_H], fill=bg)
    # score badge
    draw.rectangle([PAD, y + 8, PAD + 110, y + 36], fill=(40, 120, 220))
    draw.text((PAD + 8, y + 12), score, fill=(255, 255, 255), font=f_score)
    draw.text((PAD + 130, y + 12), f"{name}: {hint}", fill=(30, 30, 30), font=f_hint)

    # thumbs
    l = thumb(src, THUMB_W)
    r = thumb(dst, THUMB_W)
    canvas.paste(l, (PAD, y + 46))
    canvas.paste(r, (PAD * 2 + THUMB_W, y + 46))
    y += ROW_H + PAD

canvas.save(OUT, quality=92)
print(f"saved {OUT} ({W}x{H})")