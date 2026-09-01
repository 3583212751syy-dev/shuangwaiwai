"""v180e 最终对照：原图 vs v180e"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path("E:/Desktop/双接口/image-fission")
orig = ROOT / "web_gallery/img/orig_denim_3.jpg"
v180e = ROOT / "outputs/v180e/v180e_denim_3.jpg"
out = ROOT / "outputs/v180e/compare_final.jpg"

imgs = []
for p, label in [(orig, "ORIG (denim patchwork + UPGY)"),
                 (v180e, "v180e (OpenCV Telea inpaint: clean top + intact denim butterfly)")]:
    im = Image.open(p).convert("RGB")
    im.thumbnail((720, 720))
    imgs.append((im, label))

W, H = 720, 800
canvas = Image.new("RGB", (W*2 + 20, H), "white")
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()

for i, (im, label) in enumerate(imgs):
    canvas.paste(im, (i*(W+10), 0))
    bbox = draw.textbbox((0,0), label, font=font)
    tw = bbox[2]-bbox[0]
    draw.text((i*(W+10) + (W-tw)//2, 730), label, fill="black", font=font)

canvas.save(out, quality=92)
print(f"✓ {out} {out.stat().st_size/1024:.0f}KB")