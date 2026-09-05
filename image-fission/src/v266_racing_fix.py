#!/usr/bin/env python3
"""v266: RACING 图本地 PIL 精修：按实际文字位置 bbox 填充清字 + 精确重绘新词."""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

SRC = Path("E:/Desktop/图裂变测试图/184432b34a4787fbed628b3b986b37a2.jpg")
OUT_DIR = Path("E:/Desktop/双接口/image-fission/jobs/v266_racing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

img = Image.open(SRC).convert("RGB")
W, H = img.size
arr = np.array(img)

# 干净背景色（实测）
RED_BG = np.array([199, 0, 19], dtype=np.uint8)
SILVER_BG = np.array([213, 216, 228], dtype=np.uint8)

# 手工标定文字行：bbox + 背景色
# 坐标由网格图确定；top 下沿延到 y720 以覆盖 RACING  underline/shadow
TEXT_ROWS = [
    {"name": "top",      "bg": "red",    "bbox": (300, 400, 1200, 720),  "shear": -0.30, "scale": 0.92, "shadow": True},
    {"name": "motto",    "bg": "silver", "bbox": (340, 850, 1130, 910),  "shear": -0.12, "scale": 0.70},
    {"name": "city",     "bg": "silver", "bbox": (485, 1080, 995, 1150), "shear": 0.0,   "scale": 0.95},
    {"name": "sub",      "bg": "silver", "bbox": (470, 1190, 1000, 1255), "shear": 0.0,   "scale": 0.85},
    {"name": "bottom1",  "bg": "red",    "bbox": (475, 1400, 1020, 1510), "shear": -0.14, "scale": 0.85},
    {"name": "bottom2",  "bg": "red",    "bbox": (660, 1530, 845, 1665),  "shear": -0.10, "scale": 0.95},
]

# 需要保护的装饰元素：从原图贴回（避免文字 bbox 擦掉它们）
DECORATIONS = [
    {"name": "flag",  "bbox": (300, 440, 470, 510)},   # 左上角黑白格旗
    {"name": "speed", "bbox": (1050, 440, 1200, 510)}, # 右上角速度表
    {"name": "arrow", "bbox": (1000, 1055, 1085, 1125)}, # 右侧箭头
]

# ---- 字体加载 ----
def load_font(size, path="C:/Windows/Fonts/arialbi.ttf"):
    for p in (path, "C:/Windows/Fonts/ariblk.ttf", "C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def row_color(bg_name):
    return BLACK if bg_name == "silver" else WHITE

def draw_text_in_bbox(canvas, text, bbox, color, shear=0.0, scale=0.75, shadow=False):
    """在指定 bbox 内居中绘制文字，按宽度自适应字号."""
    draw = ImageDraw.Draw(canvas)
    x, y, x2, y2 = bbox
    bw, bh = x2 - x, y2 - y
    size = max(16, int(bh * scale))
    font = load_font(size)
    bbox_text = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
    if tw > bw * 0.95:
        size = int(size * (bw / tw) * 0.92)
        font = load_font(size)
        bbox_text = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
    cx = (x + x2) // 2
    cy = (y + y2) // 2
    tx = cx - tw // 2
    ty = cy - th // 2

    pad = 30
    tmp_w, tmp_h = tw + pad * 2, th + pad * 2
    if abs(shear) > 0.001:
        # 先画文字到临时层，再做剪切
        tmp = Image.new("RGBA", (tmp_w, tmp_h), (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(tmp)
        tdraw.text((pad, pad), text, font=font, fill=color)
        tmp = tmp.transform(tmp.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), resample=Image.BILINEAR)
        if shadow and color == WHITE:
            # 红底大字：加黑色阴影/描边，压住残影
            shadow_tmp = Image.new("RGBA", (tmp_w, tmp_h), (0, 0, 0, 0))
            sdraw = ImageDraw.Draw(shadow_tmp)
            sdraw.text((pad + 3, pad + 3), text, font=font, fill=BLACK)
            shadow_tmp = shadow_tmp.transform(shadow_tmp.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), resample=Image.BILINEAR)
            canvas.paste(shadow_tmp, (tx - pad, ty - pad), shadow_tmp)
        canvas.paste(tmp, (tx - pad, ty - pad), tmp)
    else:
        if shadow and color == WHITE:
            draw.text((tx + 3, ty + 3), text, font=font, fill=BLACK)
        draw.text((tx, ty), text, font=font, fill=color)
    return canvas

# ---- 清字底图：mask 检测文字 + inpaint 重建背景 ----
mask = np.zeros((H, W), dtype=np.uint8)
for r in TEXT_ROWS:
    x, y, x2, y2 = r["bbox"]
    roi = arr[y:y2, x:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    if r["bg"] == "red":
        # 红底：白字（亮）+ 黑边/阴影（暗）都要清
        m = ((gray > 160) | (gray < 60)).astype(np.uint8) * 255
    else:
        # 银底：黑字（暗）要清
        m = (gray < 100).astype(np.uint8) * 255
    # 扩张覆盖文字边缘与抗锯齿
    m = cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask[y:y2, x:x2] = cv2.bitwise_or(mask[y:y2, x:x2], m)

# 用 TELEA 重建背景（保留纹理）
cleaned = cv2.inpaint(arr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

# 恢复原图装饰元素（避免文字 bbox 误擦：旗/速度表/箭头）
for d in DECORATIONS:
    x, y, x2, y2 = d["bbox"]
    cleaned[y:y2, x:x2] = arr[y:y2, x:x2]

Image.fromarray(cleaned).save(OUT_DIR / "_cleaned_base.png")

# ---- 变体定义：每行新词 ----
VARIANTS = {
    "speed": ["SPEEDWAY", "FULL THROTTLE", "RAPID", "ROAD LEGEND", "FOURTY SIX", "88"],
    "motor": ["MOTORSPORT", "NEVER SLOW DOWN", "TITAN", "TRACK KING", "FOURTY SIX", "77"],
    "turbo": ["TURBO", "BOOST LIFE", "STREET", "RACE LEGEND", "FOURTY SIX", "66"],
}

# ---- 主体裂变：装饰微调 ----
def tweak_racing(canvas, tag):
    arr = np.array(canvas).astype(np.int16)
    # 速度表区域（右上角）轻微改色
    sx1, sy1, sx2, sy2 = 1050, 440, 1200, 510
    speed_roi = arr[sy1:sy2, sx1:sx2].astype(np.int16)
    if tag == "speed":
        shift = np.array([12, 0, -8])
    elif tag == "motor":
        shift = np.array([-8, 0, 5])
    else:
        shift = np.array([0, 5, 8])
    arr[sy1:sy2, sx1:sx2] = np.clip(speed_roi + shift, 0, 255).astype(np.uint8)
    # 箭头区域轻微改色
    ax1, ay1, ax2, ay2 = 1000, 1055, 1085, 1125
    arrow_roi = arr[ay1:ay2, ax1:ax2].astype(np.int16)
    if tag == "speed":
        shift2 = np.array([0, 0, 12])
    elif tag == "motor":
        shift2 = np.array([12, 0, 0])
    else:
        shift2 = np.array([0, 12, 0])
    arr[ay1:ay2, ax1:ax2] = np.clip(arrow_roi + shift2, 0, 255).astype(np.uint8)
    return Image.fromarray(arr.astype(np.uint8))

# ---- 生成变体 ----
paths = []
for tag, words in VARIANTS.items():
    out = Image.fromarray(cleaned).copy()
    for row, word in zip(TEXT_ROWS, words):
        color = row_color(row["bg"])
        draw_text_in_bbox(out, word, row["bbox"], color, shear=row.get("shear", 0.0), scale=row.get("scale", 0.75), shadow=row.get("shadow", False))
    out = tweak_racing(out, tag)
    p = OUT_DIR / f"racing_{tag}_final.png"
    out.save(p)
    paths.append(p)
    print("saved", p)

# ---- 对照图：原图 + 3 变体 ----
orig = Image.open(SRC).convert("RGB")
panels = [("ORIGINAL", orig)] + [(f"RACING {tag.upper()}", Image.open(p).convert("RGB")) for tag, p in zip(VARIANTS.keys(), paths)]
scale = 0.30
thumb = (int(W * scale), int(H * scale))
panels = [(t, im.resize(thumb, Image.LANCZOS)) for t, im in panels]
cell_w, cell_h = thumb
margin = 20
grid_w = cell_w * 4 + margin * 5
grid_h = cell_h + 80
canvas = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
draw = ImageDraw.Draw(canvas)
ft = load_font(28, "C:/Windows/Fonts/arialbd.ttf")
for i, (title, im) in enumerate(panels):
    x = i * (cell_w + margin) + margin
    y = margin
    canvas.paste(im, (x, y))
    bbox = draw.textbbox((0, 0), title, font=ft)
    tw = bbox[2] - bbox[0]
    draw.text((x + (cell_w - tw) // 2, y + cell_h + 12), title, fill=(20, 20, 20), font=ft)
grid_path = OUT_DIR / "_compare_racing.png"
canvas.save(grid_path)
print("grid", grid_path)
