import easyocr, numpy as np
from PIL import Image, ImageOps, ImageFilter

reader = easyocr.Reader(['en'], gpu=True)
p = r"E:/Desktop/图裂变测试图/pinterest_metal_6.jpg"
img = Image.open(p).convert("RGB")
# 放大 2x 提高小字识别
big = img.resize((img.width*2, img.height*2), Image.LANCZOS)
arr = np.array(big)
res = reader.readtext(arr, detail=1, paragraph=False, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
print(f"metal_6 size={img.size}, scaled 2x")
for (bbox, text, conf) in res:
    if conf < 0.05:
        continue
    xs=[pt[0]/2 for pt in bbox]; ys=[pt[1]/2 for pt in bbox]
    print(f"  {text!r} conf={round(float(conf),2)} box=({int(min(xs))},{int(min(ys))})-({int(max(xs))},{int(max(ys))})")
