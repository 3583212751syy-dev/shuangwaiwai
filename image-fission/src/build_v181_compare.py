"""v181 vs 原图 camo_4 并排对照图"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path("E:/Desktop/双接口/image-fission")
OUT = ROOT / "jobs/smoke_v181"
OUT.mkdir(exist_ok=True)

orig = Image.open(ROOT / "web_gallery/img/orig_camo_4.jpg").convert("RGB")
v181 = Image.open(ROOT / "jobs/smoke_v181/v181_camo_4.jpg").convert("RGB")

# 统一高度
H = 1200
def fit(img):
    w = int(img.width * H / img.height)
    return img.resize((w, H), Image.LANCZOS)

orig_r = fit(orig)
v181_r = fit(v181)

# 并排
gap = 30
W = orig_r.width + v181_r.width + gap
canvas = Image.new("RGB", (W, H + 60), "white")
canvas.paste(orig_r, (0, 60))
canvas.paste(v181_r, (orig_r.width + gap, 60))

# 标签
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 36)
except Exception:
    font = ImageFont.load_default()
draw.text((orig_r.width // 2 - 60, 10), "ORIG", fill="black", font=font)
draw.text((orig_r.width + gap + v181_r.width // 2 - 80, 10), "v181 FISS", fill="black", font=font)

out = OUT / "compare_v181_vs_orig.jpg"
canvas.save(out, "JPEG", quality=92)
print(f"OK -> {out} ({out.stat().st_size/1024:.0f}KB)")