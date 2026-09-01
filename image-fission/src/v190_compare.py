"""
v190_compare.py — 生成 v190 双图对照拼图

布局：每张 3 列（[原图 | v190 裂变 | v190 烧字]），2 行（bat_logo + camo_armed）
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SRC_BAT = "E:/Desktop/双接口/image-fission/ComfyUI/input/test_6978fabda2cc99629fa9e81f802762d3.jpg"
SRC_CAMO = "E:/Desktop/双接口/image-fission/ComfyUI/input/test_b78e60de8dfdf44acda99395326a7298.jpg"
V190_DIR = "E:/Desktop/双接口/image-fission/jobs/smoke_v190"
OUT = f"{V190_DIR}/_compare_v190.jpg"

def load_resized(p, target_w):
    img = Image.open(p).convert("RGB")
    ratio = target_w / img.width
    new_size = (target_w, int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)

# 每列宽 600
COL_W = 600
PADDING = 10
LABEL_H = 50

rows = [
    {
        "label": "BAT_LOGO (BACARDÍ HEART → 蝙蝠徽章侵权图)",
        "imgs": [
            ("原图", SRC_BAT),
            ("v190 裂变", f"{V190_DIR}/v190_bat_logo.jpg"),
            ("v190 烧字成品", f"{V190_DIR}/v190_bat_logo_burned.jpg"),
        ],
    },
    {
        "label": "CAMO_ARMED (WE SUPPORT THE ARMED FORCES → 军牌侵权图)",
        "imgs": [
            ("原图", SRC_CAMO),
            ("v190 裂变", f"{V190_DIR}/v190_camo_armed.jpg"),
            ("v190 烧字成品", f"{V190_DIR}/v190_camo_armed_burned.jpg"),
        ],
    },
]

# 计算总尺寸
col_heights = []
for row in rows:
    h = 0
    for lbl, p in row["imgs"]:
        img = load_resized(p, COL_W)
        h = max(h, img.height)
    col_heights.append(h + LABEL_H + PADDING * 2)

total_w = COL_W * 3 + PADDING * 4
total_h = sum(col_heights) + PADDING * (len(rows) + 1) + 100  # 100 是顶部标题区

canvas = Image.new("RGB", (total_w, total_h), (240, 240, 240))
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype("E:/Desktop/双接口/image-fission/fonts/Lora-VF.ttf", 24)
title_font = ImageFont.truetype("E:/Desktop/双接口/image-fission/fonts/Oswald-VF.ttf", 36)

draw.text((PADDING, 10), "v190 — bat_logo + camo_armed 双图对照（裂变 + 烧字）",
          font=title_font, fill=(20, 20, 20))

y_offset = 100
for ridx, row in enumerate(rows):
    # 行标签
    draw.text((PADDING, y_offset), row["label"], font=font, fill=(20, 20, 20))
    y_offset += LABEL_H
    # 列
    for cidx, (lbl, p) in enumerate(row["imgs"]):
        x = PADDING + cidx * (COL_W + PADDING)
        img = load_resized(p, COL_W)
        canvas.paste(img, (x, y_offset))
        # 列标签
        draw.text((x, y_offset + img.height + 5), lbl, font=font, fill=(40, 40, 40))
    y_offset += col_heights[ridx] + PADDING

canvas.save(OUT, quality=90)
print(f"saved → {OUT} ({Path(OUT).stat().st_size // 1024} KB)")