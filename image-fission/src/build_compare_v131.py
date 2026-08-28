"""v131 拼图对照 (6 行 × 2 列: 原图 | v131 出图)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

JOB_DIR = Path("E:/Desktop/双接口/image-fission/jobs/smoke_v131_wide_1787906172")
OUT = JOB_DIR / "v131_compare_6x2.png"

PAIRS = [
    {"name": "denim_3", "src": "E:/Desktop/图裂变测试图/pinterest_denim_3.jpg",
     "dst": str(JOB_DIR / "v131_denim_3.png"),
     "hint": "UPCY 牛仔蝴蝶 → UPCY + 略动蝴蝶 (文字未换)"},
    {"name": "camo_4", "src": "E:/Desktop/图裂变测试图/pinterest_camo_4.jpg",
     "dst": str(JOB_DIR / "v131_camo_4.png"),
     "hint": "棕榈迷彩 → 几乎 100% 一致 (裂变失败)"},
    {"name": "eagle_2", "src": "E:/Desktop/图裂变测试图/pinterest_eagle_2.jpg",
     "dst": str(JOB_DIR / "v131_eagle_2.png"),
     "hint": "鹰+骷髅 → 鹰+骷髅 (文字出乱码)"},
    {"name": "illust_1", "src": "E:/Desktop/图裂变测试图/pinterest_illust_1.jpg",
     "dst": str(JOB_DIR / "v131_illust_1.png"),
     "hint": "黑白装饰 → 百合+藤蔓 (构图同)"},
    {"name": "metal_6", "src": "E:/Desktop/图裂变测试图/pinterest_metal_6.jpg",
     "dst": str(JOB_DIR / "v131_metal_6.png"),
     "hint": "METALLICA 风 → MEEALICA 风 (仿字)"},
    {"name": "skull_5", "src": "E:/Desktop/图裂变测试图/pinterest_skull_5.jpg",
     "dst": str(JOB_DIR / "v131_skull_5.png"),
     "hint": "TRUE/NEVER DIES → BREE/NEVER DIES (底部未换)"},
]

THUMB_W = 380
PAD = 12
LABEL_H = 36


def thumb(path, w):
    im = Image.open(path).convert("RGB")
    ratio = w / im.size[0]
    h = int(im.size[1] * ratio)
    return im.resize((w, h), Image.LANCZOS)


def text_image(draw, x, y, w, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x + (w - tw) // 2, y), text, fill="black", font=font)


try:
    font = ImageFont.truetype("arial.ttf", 18)
    font_b = ImageFont.truetype("arialbd.ttf", 18)
except Exception:
    font = ImageFont.load_default()
    font_b = font

# Pre-compute heights
row_imgs = []
for p in PAIRS:
    src = thumb(p["src"], THUMB_W)
    dst = thumb(p["dst"], THUMB_W)
    h = max(src.size[1], dst.size[1])
    row_imgs.append((p, src, dst, h))

W = THUMB_W * 2 + PAD * 3
H = sum(h + LABEL_H * 2 + PAD for _, _, _, h in row_imgs) + PAD

canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)
y = PAD
for p, src, dst, h in row_imgs:
    # row title (above pair)
    text_image(draw, PAD, y, W - PAD * 2, f"{p['name']}  |  {p['hint']}", font_b)
    y += LABEL_H
    # side labels
    text_image(draw, PAD, y + 8, THUMB_W, "ORIGINAL", font)
    text_image(draw, PAD * 2 + THUMB_W, y + 8, THUMB_W, "v131 (wide+colorlock)", font)
    y += LABEL_H
    canvas.paste(src, (PAD, y))
    canvas.paste(dst, (PAD * 2 + THUMB_W, y))
    y += h + PAD

canvas.save(OUT, "PNG", optimize=True)
print(f"saved {OUT}  ({canvas.size[0]}x{canvas.size[1]})")