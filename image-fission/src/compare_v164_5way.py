"""v164 5 张裂变对照拼图 + 复制桌面版。"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SRC_REF = Path(r"E:\Desktop\双接口\image-fission\ComfyUI\input")
SRC_OUT = Path(r"E:\Desktop\双接口\image-fission\jobs\smoke_v164")
DESK = Path(r"E:\Desktop")
COMPARE = SRC_OUT / "compare_v164_5way.jpg"

IDS = ["camo_4", "illust_1", "denim_3", "skull_5", "metal_6"]
LABELS = {
    "camo_4": "棕榈迷彩",
    "illust_1": "黑白卷草",
    "denim_3": "牛仔蝴蝶",
    "skull_5": "骷髅红翼蛇",
    "metal_6": "死亡金属鹰",
}

# --- 单图对照图 ---
for rid in IDS:
    ref_p = SRC_REF / f"pinterest_{rid}.jpg"
    out_p = SRC_OUT / f"v164_{rid}.jpg"
    if not (ref_p.exists() and out_p.exists()):
        print(f"[skip] {rid} 缺文件"); continue
    a = Image.open(ref_p).convert("RGB")
    b = Image.open(out_p).convert("RGB")
    H = 1200
    a2 = a.resize((int(a.width*H/a.height), H), Image.LANCZOS)
    b2 = b.resize((int(b.width*H/b.height), H), Image.LANCZOS)
    gap = 30
    W = a2.width + b2.width + gap*2 + 60
    canvas = Image.new("RGB", (W, H+60), (32,32,32))
    d = ImageDraw.Draw(canvas)
    d.text((gap+10, 12), f"原图 pinterest_{rid}", fill=(255,255,255))
    d.text((gap+a2.width+gap+10, 12), f"v164 裂变", fill=(255,255,255))
    canvas.paste(a2, (gap, 60))
    canvas.paste(b2, (gap+a2.width+gap, 60))
    cmp_p = SRC_OUT / f"compare_{rid}_v164.jpg"
    canvas.save(cmp_p, "JPEG", quality=85, optimize=True)
    print(f"  [compare] {cmp_p.name} {cmp_p.stat().st_size/1024/1024:.1f}MB")

# --- 5 路总览拼图 ---
H = 600
gap = 12
COLS = 2  # 原图 | v164
ROWS = len(IDS)
imgs = []
for rid in IDS:
    ref_p = SRC_REF / f"pinterest_{rid}.jpg"
    out_p = SRC_OUT / f"v164_{rid}.jpg"
    a = Image.open(ref_p).convert("RGB")
    b = Image.open(out_p).convert("RGB")
    a2 = a.resize((int(a.width*H/a.height), H), Image.LANCZOS)
    b2 = b.resize((int(b.width*H/b.height), H), Image.LANCZOS)
    imgs.append((LABELS[rid], a2, b2))

# 总宽 = 最宽图 × 列数 + gap
max_w = max(max(a.width, b.width) for _,a,b in imgs)
W = max_w * COLS + gap * (COLS+1) + 40  # 左右边距
TOT_H = H * ROWS + gap * (ROWS+1) + 60*ROWS  # 每行带一个标签条 60px
canvas = Image.new("RGB", (W, TOT_H+40), (24,24,24))
d = ImageDraw.Draw(canvas)
d.text((10, 10), "v147 模板 × 5 图 裂变对照（v164，主体保留 + 数量/小元素可变）", fill=(255,255,255))
y0 = 40
for i, (label, a, b) in enumerate(imgs):
    ya = y0 + i*(H + gap + 60)
    d.text((20, ya+8), f"{label}  |  原图 (左) | v164 (右)", fill=(220,220,220))
    canvas.paste(a, (gap+20, ya+60))
    canvas.paste(b, (W - b.width - gap - 20, ya+60))
canvas.save(COMPARE, "JPEG", quality=85, optimize=True)
print(f"\n[TOTAL] {COMPARE.name} {COMPARE.stat().st_size/1024/1024:.1f}MB")

# --- 复制桌面 v164 终版 ---
for rid in IDS:
    src = SRC_OUT / f"v164_{rid}.jpg"
    dst = DESK / f"image-fission-v164-{rid}.jpg"
    dst.write_bytes(src.read_bytes())
    print(f"  [DESK] {dst.name} {dst.stat().st_size/1024/1024:.1f}MB")
