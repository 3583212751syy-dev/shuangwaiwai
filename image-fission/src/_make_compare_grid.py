"""生成 2x2 原图 vs 裂变结果对照图."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
JOB = PROJECT / "jobs" / "v261"
SRC = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"

panels = [
    ("ORIGINAL", Image.open(SRC).convert("RGB")),
    ("RAVEN", Image.open(JOB / "v261_raven_final.png").convert("RGB")),
    ("OWL", Image.open(JOB / "v261_owl_final.png").convert("RGB")),
    ("FALCON", Image.open(JOB / "v261_falcon_final.png").convert("RGB")),
]

# 统一缩放到安全展示尺寸（避免过大）
w, h = 1552, 2000
scale = 0.6
thumb = (int(w * scale), int(h * scale))
panels = [(t, im.resize(thumb, Image.LANCZOS)) for t, im in panels]

# 2x2 拼图画布
cell_w, cell_h = thumb
margin = 20
grid_w = cell_w * 2 + margin * 3
grid_h = cell_h * 2 + margin * 3 + 60  # 底部留白
canvas = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
draw = ImageDraw.Draw(canvas)

# 尝试加载字体
try:
    font = ImageFont.truetype(str(PROJECT / "fonts" / "PirataOne-Regular.ttf"), 36)
except Exception:
    font = ImageFont.load_default()

positions = [
    (margin, margin),
    (cell_w + margin * 2, margin),
    (margin, cell_h + margin * 2),
    (cell_w + margin * 2, cell_h + margin * 2),
]
for (title, im), (x, y) in zip(panels, positions):
    canvas.paste(im, (x, y))
    # 标题
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x + (cell_w - tw) // 2, y + cell_h + 10), title, fill=(30, 30, 30), font=font)

out = JOB / "_grid_v261_compare.png"
canvas.save(out, "PNG")
print(out)
