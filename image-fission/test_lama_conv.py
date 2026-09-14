"""判定实验：验证 big-lama jit 的 mask 语义（1=去除区 还是 1=保留区）。
合成图：浅灰底 + 黑字块。mask=字块(白=去除)。
变体A：走 v268_lama_clean.lama_inpaint 的变换(invert->blur->threshold)
变体B：直接 mask/255（1=去除区）
看哪个能把黑字擦掉。"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import v268_lama_clean as lc

SRC = Image.new("RGB", (256, 256), (200, 200, 205))
d = ImageDraw.Draw(SRC)
d.rectangle([60, 100, 196, 156], fill=(20, 20, 20))       # 模拟文字块
# 加点背景纹理便于观察
for i in range(0, 256, 16):
    d.line([(i, 0), (i, 256)], fill=(185, 185, 190))

MASK = Image.new("L", (256, 256), 0)
ImageDraw.Draw(MASK).rectangle([55, 95, 201, 161], fill=255)   # 白=去除区

model, dev = lc.load_lama()
print("lama loaded on", dev)

img_t = lambda im: __import__("torch").from_numpy(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

def run_with_mask(p_mask):
    import torch
    it = img_t(SRC)
    mt = torch.from_numpy(np.array(p_mask, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.inference_mode():
        res = model(it, mt)
    return Image.fromarray((res[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))

# A: 现行变换
pm = ImageOps.invert(MASK)
pm = pm.filter(ImageFilter.GaussianBlur(radius=6))
grayA = pm.point(lambda x: 0 if x > 235 else 255)
run_with_mask(grayA).save("jobs/lama_test_A_current.png")

# B: 直接 1=去除区
run_with_mask(MASK).save("jobs/lama_test_B_direct.png")

# C: 反相 1=保留区
run_with_mask(ImageOps.invert(MASK)).save("jobs/lama_test_C_invert.png")
print("saved A/B/C")
