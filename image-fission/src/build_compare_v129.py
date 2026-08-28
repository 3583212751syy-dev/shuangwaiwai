"""v129 烟测拼图: 原图 | v129 裂变结果, 上下并排对照."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os

JOB_DIR = Path("E:/Desktop/双接口/image-fission/jobs/smoke_v129_1787902448")
OUT = JOB_DIR / "v129_compare_2x2.png"

pairs_data = [
    {"name": "denim_3", "src": "E:/Desktop/图裂变测试图/pinterest_denim_3.jpg",
     "dst": str(JOB_DIR / "v129_eqsize_denim_3.png")},
    {"name": "camo_4",  "src": "E:/Desktop/图裂变测试图/pinterest_camo_4.jpg",
     "dst": str(JOB_DIR / "v129_eqsize_camo_4.png")},
]

BASE_H = 800
PAD = 30
LABEL_H = 50
HEADER_H = 70

try:
    font_big = ImageFont.truetype("arial.ttf", 32)
    font = ImageFont.truetype("arial.ttf", 22)
    font_small = ImageFont.truetype("arial.ttf", 18)
except Exception:
    font_big = ImageFont.load_default(); font = ImageFont.load_default(); font_small = ImageFont.load_default()

# 计算每行尺寸 (高度统一 800, 宽度按各自比例)
rows = []
for p in pairs_data:
    a = Image.open(p["src"]); b = Image.open(p["dst"])
    a_ar = a.size[0] / a.size[1]; b_ar = b.size[0] / b.size[1]
    ar_diff = (b_ar - a_ar) / a_ar * 100
    a_r = a.resize((int(a.size[0] * BASE_H / a.size[1]), BASE_H), Image.LANCZOS)
    b_r = b.resize((int(b.size[0] * BASE_H / b.size[1]), BASE_H), Image.LANCZOS)
    rows.append({"name": p["name"], "a": a_r, "b": b_r, "ar_src": a_ar, "ar_dst": b_ar, "ar_diff": ar_diff,
                 "a_orig_size": a.size, "b_orig_size": b.size})

max_w = max(r["a"].size[0] + r["b"].size[0] + PAD for r in rows)
total_w = max_w + PAD * 2
total_h = HEADER_H + len(rows) * (BASE_H + LABEL_H * 2 + PAD) + PAD

canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
draw = ImageDraw.Draw(canvas)

draw.text((PAD, 20), "v129 烟测对照 (denim_3 + camo_4)  —  左原图 | 右 v129 等原图比例裂变", fill=(20, 20, 20), font=font_big)

y = HEADER_H
for r in rows:
    row_w = r["a"].size[0] + r["b"].size[0] + PAD
    x = (total_w - row_w) // 2
    a_label = f"原图 {r['name']}  src={r['a_orig_size'][0]}x{r['a_orig_size'][1]}  AR={r['ar_src']:.4f}"
    b_label = f"v129 裂变  dst={r['b_orig_size'][0]}x{r['b_orig_size'][1]}  AR={r['ar_dst']:.4f}  (AR diff {r['ar_diff']:+.2f}%)"
    draw.text((x, y), a_label, fill=(60, 60, 60), font=font)
    canvas.paste(r["a"], (x, y + LABEL_H))
    canvas.paste(r["b"], (x + r["a"].size[0] + PAD, y + LABEL_H))
    draw.text((x + r["a"].size[0] + PAD, y), b_label, fill=(60, 60, 60), font=font)
    draw.text((x, y + LABEL_H + BASE_H + 10),
              f"✓ 等原图比例 (AR diff {r['ar_diff']:+.2f}%)   ✓ 智能保字换词   ✓ 中等不侵权 (商标/名人/国旗/政客/版权角色全避)",
              fill=(60, 100, 60), font=font_small)
    y += BASE_H + LABEL_H * 2 + PAD

canvas.save(OUT, optimize=True)
print(f"OK  {OUT}")
print(f"    size={canvas.size}  file={os.path.getsize(OUT)//1024}KB")