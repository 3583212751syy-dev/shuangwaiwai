"""v130 烟测拼图: 6 张 (denim_3 + camo_4 + eagle_2 + illust_1 + metal_6 + skull_5)
                  左原图 | 右 v130 strong REDESIGN 裂变结果.

v130 vs v129:
  - denoise 0.55 -> 0.72 (balanced)
  - prompt 强化 REDESIGNED internal details
  - 假设全侵权 -> 文字强制换合法
  - 加摇杆参数 (mild/balanced/strong)
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os

JOB_DIR = Path("E:/Desktop/双接口/image-fission/jobs/smoke_v130_balanced_1787903181")
METAL6_DIR = Path("E:/Desktop/双接口/image-fission/jobs/smoke_v130_metal6_4mp")
OUT = JOB_DIR / "v130_compare_6x2.png"

PAIRS = [
    {"name": "denim_3", "src": "E:/Desktop/图裂变测试图/pinterest_denim_3.jpg",
     "dst": str(JOB_DIR / "v130_denim_3.png"), "hint": "UPCYCLE 牛仔蝴蝶 -> UPCY 牛仔蛾子"},
    {"name": "camo_4",  "src": "E:/Desktop/图裂变测试图/pinterest_camo_4.jpg",
     "dst": str(JOB_DIR / "v130_camo_4.png"), "hint": "棕榈迷彩 -> 松针雪景迷彩"},
    {"name": "eagle_2", "src": "E:/Desktop/图裂变测试图/pinterest_eagle_2.jpg",
     "dst": str(JOB_DIR / "v130_eagle_2.png"), "hint": "鹰 -> 凤凰 (重装饰, 轻微跑飞)"},
    {"name": "illust_1","src": "E:/Desktop/图裂变测试图/pinterest_illust_1.jpg",
     "dst": str(JOB_DIR / "v130_illust_1.png"), "hint": "黑白装饰插画 -> 玫瑰藤蔓 damask"},
    {"name": "metal_6", "src": "E:/Desktop/图裂变测试图/pinterest_metal_6.jpg",
     "dst": str(METAL6_DIR / "v130_metal_6.png"), "hint": "金属浮雕 -> 跑偏 (鹰+骷髅哥特风, 4MP保AR+resize回3543x4961)"},
    {"name": "skull_5", "src": "E:/Desktop/图裂变测试图/pinterest_skull_5.jpg",
     "dst": str(JOB_DIR / "v130_skull_5.png"), "hint": "骷髅 -> 骷髅+玫瑰+翅膀+荆棘 + 'TRUE NEVER DIES'"},
]

# 缩略尺寸: 每张缩小到 480 高, 拼图横向才能塞下 6 张
BASE_H = 480
PAD = 16
LABEL_H = 28
HEADER_H = 60
HINT_H = 22

try:
    font_big = ImageFont.truetype("arial.ttf", 28)
    font = ImageFont.truetype("arial.ttf", 16)
    font_small = ImageFont.truetype("arial.ttf", 13)
except Exception:
    font_big = ImageFont.load_default(); font = ImageFont.load_default(); font_small = ImageFont.load_default()

# 计算
loaded = []
for p in PAIRS:
    src = Image.open(p["src"])
    dst_path = Path(p["dst"])
    if dst_path.exists():
        dst = Image.open(p["dst"])
        ar_diff = (dst.size[0]/dst.size[1] - src.size[0]/src.size[1]) / (src.size[0]/src.size[1]) * 100
        a_r = src.resize((int(src.size[0] * BASE_H / src.size[1]), BASE_H), Image.LANCZOS)
        b_r = dst.resize((int(dst.size[0] * BASE_H / dst.size[1]), BASE_H), Image.LANCZOS)
        size_str = f"{dst.size[0]}x{dst.size[1]} ARdiff={ar_diff:+.1f}%"
    else:
        ar_diff = 0
        a_r = src.resize((int(src.size[0] * BASE_H / src.size[1]), BASE_H), Image.LANCZOS)
        b_r = a_r.copy()
        size_str = "MISSING"
    loaded.append({"p": p, "a": a_r, "b": b_r, "size": size_str,
                   "src_size": src.size, "dst_size": dst.size if dst_path.exists() else None})

max_w = max(r["a"].size[0] + r["b"].size[0] + PAD for r in loaded)
total_w = max_w + PAD * 2
total_h = HEADER_H + sum(BASE_H + LABEL_H + HINT_H + PAD + 10 for _ in loaded) + PAD

canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
draw = ImageDraw.Draw(canvas)

draw.text((PAD, 14),
          "v130 REDESIGN 6 张裂变对照  (balanced 摇杆: denoise=0.72 cn=0.5 ipa=0.5)",
          fill=(20, 20, 20), font=font_big)

y = HEADER_H
for r in loaded:
    pair_w = r["a"].size[0] + r["b"].size[0] + PAD
    x = (total_w - pair_w) // 2
    # 列标题
    draw.text((x, y), f"[ 原图 {r['p']['name']}  src={r['src_size'][0]}x{r['src_size'][1]} ]",
              fill=(60, 60, 60), font=font)
    draw.text((x + r["a"].size[0] + PAD, y),
              f"[ v130 裂变  {r['size']} ]",
              fill=(60, 60, 60), font=font)
    y += LABEL_H
    canvas.paste(r["a"], (x, y))
    canvas.paste(r["b"], (x + r["a"].size[0] + PAD, y))
    y += BASE_H
    # 内容提示
    color = (50, 110, 50) if r["size"] != "MISSING" else (200, 80, 80)
    draw.text((x, y), f"→ {r['p']['hint']}", fill=color, font=font_small)
    y += HINT_H + PAD + 4

canvas.save(OUT, optimize=True)
print(f"OK  {OUT}")
print(f"    size={canvas.size}  file={os.path.getsize(OUT)//1024//1024}MB")
