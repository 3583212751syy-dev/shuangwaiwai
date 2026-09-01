"""v179 状态对照：原图 vs v178(干净矢量) vs v179b(真牛仔布料)"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path("E:/Desktop/双接口/image-fission")
orig = ROOT / "web_gallery/img/orig_denim_3.jpg"
v178 = ROOT / "outputs/v178/v178_denim_3.jpg"
v179b = ROOT / "outputs/v179b/v179b_denim_3.jpg"
out = ROOT / "outputs/v179b/compare_state_3way.jpg"

imgs = []
for p, label in [(orig, "原图 (牛仔拼贴 + UPGY字母)"),
                 (v178, "v178 (干净矢量蝴蝶, 无字母)"),
                 (v179b, "v179b (真牛仔布料, 但D字母仍在)")]:
    im = Image.open(p).convert("RGB")
    im.thumbnail((640, 640))
    imgs.append((im, label))

W, H = 640, 720  # 640 宽 + 底部 80 标签
canvas = Image.new("RGB", (W*3 + 20, H), "white")
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 20)
    fontb = ImageFont.truetype("arialbd.ttf", 22)
except Exception:
    font = ImageFont.load_default()
    fontb = font

for i, (im, label) in enumerate(imgs):
    canvas.paste(im, (i*(W+10), 0))
    bbox = draw.textbbox((0,0), label, font=font)
    tw = bbox[2]-bbox[0]
    draw.text((i*(W+10) + (W-tw)//2, 640+10), label, fill="black", font=font)

# 红色箭头批注：D 字母位置
draw.rectangle([10+(W+10)*2, 60, 10+(W+10)*2+W, 90], outline="red", width=4)
draw.text((10+(W+10)*2 + 10, 95), "<- 牛仔布字母 D 仍在", fill="red", font=fontb)

draw.rectangle([10+(W+10)*1, 110, 10+(W+10)*1+W, 140], outline="orange", width=3)
draw.text((10+(W+10)*1 + 10, 145), "<- v178 无字母 (矢量干净)", fill="orange", font=fontb)

canvas.save(out, quality=92)
print(f"✓ {out} {out.stat().st_size/1024:.0f}KB")