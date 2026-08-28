# -*- coding: utf-8 -*-
"""build_compare_v133.py -- 生成 v133 6 张拼图对照."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SRC = Path("E:/Desktop/图裂变测试图")
JOB_DIR = Path("E:/Desktop/双接口/image-fission/jobs/smoke_v133_layered_1787908933")
OUT = JOB_DIR / "v133_compare_6x2.png"

PAIRS = [
    {"name": "denim_3", "src": str(SRC/"pinterest_denim_3.jpg"),
     "dst": str(JOB_DIR/"v133_pinterest_denim_3.png"), "word": "WORN",
     "hint": "牛仔拼布蝴蝶 -> 牛仔布 WORN 大字特写"},
    {"name": "camo_4", "src": str(SRC/"pinterest_camo_4.jpg"),
     "dst": str(JOB_DIR/"v133_pinterest_camo_4.png"), "word": "(纯裂变)",
     "hint": "棕榈迷彩 -> 松林雪景迷彩"},
    {"name": "eagle_2", "src": str(SRC/"pinterest_eagle_2.jpg"),
     "dst": str(JOB_DIR/"v133_pinterest_eagle_2.png"), "word": "BRAVE",
     "hint": "鹰+JACKE DIANNIES -> 鹰+BRAVE"},
    {"name": "illust_1", "src": str(SRC/"pinterest_illust_1.jpg"),
     "dst": str(JOB_DIR/"v133_pinterest_illust_1.png"), "word": "(纯裂变)",
     "hint": "黑白装饰插画 -> 牡丹藤蔓(新)"},
    {"name": "metal_6", "src": str(SRC/"pinterest_metal_6.jpg"),
     "dst": str(JOB_DIR/"v133_pinterest_metal_6.png"), "word": "VENGEFUL",
     "hint": "METALLICA 风死金 -> 青铜盾徽+VENGEFUL"},
    {"name": "skull_5", "src": str(SRC/"pinterest_skull_5.jpg"),
     "dst": str(JOB_DIR/"v133_pinterest_skull_5.png"), "word": "BORN/ALWAYS",
     "hint": "骷髅+TRUE NEVER DIES -> 骷髅+BORN ALWAYS"},
]

THUMB_W = 500
GAP = 24
LABEL_H = 50
PAD = 20
TITLE_H = 60

# 加载所有图, 计算每张缩略图实际高度
items = []
for p in PAIRS:
    src = Image.open(p["src"]).convert("RGB")
    dst = Image.open(p["dst"]).convert("RGB")
    h = int(THUMB_W * src.height / src.width)
    items.append({**p, "src_img": src.resize((THUMB_W, h), Image.LANCZOS),
                  "dst_img": dst.resize((THUMB_W, h), Image.LANCZOS), "h": h})

max_h = max(it["h"] for it in items)
row_h = max_h + LABEL_H + PAD
canvas_w = THUMB_W * 2 + GAP + PAD * 2
canvas_h = TITLE_H + row_h * len(items) + PAD

canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
d = ImageDraw.Draw(canvas)
d.text((PAD, 18), "v133 Layered (Canny 0.55 + IPA 0.45 + denoise 0.72 + Reinhard + text override)",
       fill=(20, 20, 20))

font_label = ImageFont.load_default()
for i, it in enumerate(items):
    y0 = TITLE_H + i * row_h
    canvas.paste(it["src_img"], (PAD, y0))
    canvas.paste(it["dst_img"], (PAD + THUMB_W + GAP, y0))
    # 分隔线
    d.line([(PAD + THUMB_W + GAP//2, y0), (PAD + THUMB_W + GAP//2, y0 + max_h)],
           fill=(200, 200, 200), width=2)
    # 标签
    d.text((PAD, y0 + max_h + 5), f"ORIG {it['name']}", fill=(60, 60, 60), font=font_label)
    d.text((PAD + THUMB_W + GAP, y0 + max_h + 5),
           f"FISSION -> '{it['word']}' | {it['hint']}", fill=(200, 60, 60), font=font_label)
    d.text((PAD, y0 + max_h + 22), "AR ≈ original / 颜色保 / 文字换", fill=(60, 60, 60), font=font_label)

canvas.save(OUT, quality=85)
print(f"compare -> {OUT} ({canvas.size})")