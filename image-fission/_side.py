"""_side.py — 通用左右对照（ORIG | NEW），可指定两图路径与输出名。"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

o = Image.open(sys.argv[1]).convert("RGB")
n = Image.open(sys.argv[2]).convert("RGB")
H = int(sys.argv[3]) if len(sys.argv) > 3 else 900
dst = Path(sys.argv[4] if len(sys.argv) > 4 else "jobs/_cmpcrop/side.jpg")


def fit(im):
    w = max(1, int(im.width * H / im.height))
    return im.resize((w, H), Image.LANCZOS)


a, b = fit(o), fit(n)
cv = Image.new("RGB", (a.width + b.width + 30, H + 26), (245, 245, 245))
d = ImageDraw.Draw(cv)
cv.paste(a, (10, 24)); cv.paste(b, (a.width + 20, 24))
d.text((10, 6), "ORIG", fill=(0, 0, 0))
d.text((a.width + 20, 6), "NEW", fill=(200, 0, 0))
dst.parent.mkdir(parents=True, exist_ok=True)
cv.save(str(dst), quality=94)
print("saved", dst, cv.size)
