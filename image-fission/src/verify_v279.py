"""v279 客观质检 v2：用真实蝙蝠 mask 隔离主体，正确度量裂变 + 背景。"""
import numpy as np
from pathlib import Path
from PIL import Image
import easyocr

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
JOB = PROJECT / "jobs" / "v279"
ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"

orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
finals = {t: Image.open(JOB / f"v279_{t}_final.png").convert("RGB") for t in ("up", "spread", "fold")}
bat_mask = np.array(Image.open(COMFY_INPUT / "v279_bat_mask.png").convert("L")).astype(np.float32) / 255.0

# 背景区：整图减去蝙蝠 mask，再减去文字矩形/弧带（清字区必变，不算背景）
text_rects = [(1000, 150, 1180, 1400), (1180, 350, 1350, 1200),
              (720, 300, 880, 620), (720, 930, 880, 1250)]
H, W = np.array(orig).shape[:2]
ys, xs = np.ogrid[:H, :W]
ang = (np.degrees(np.arctan2(ys - 746, xs - 776)) - 190) % 360
arc = (ys - 746) ** 2 + (xs - 776) ** 2 >= (421 - 90) ** 2
arc &= (ys - 746) ** 2 + (xs - 776) ** 2 <= (421 + 70) ** 2
arc &= ang <= 160
bg = np.ones((H, W), np.float32) - bat_mask
for y0, x0, y1, x1 in text_rects:
    bg[y0:y1, x0:x1] = 0
bg[arc] = 0

o = np.array(orig).astype(np.float32)
ma = bat_mask[:, :, None]
bg3 = bg[:, :, None]


def reg_mae(a, b, m3):
    n = m3.sum()
    if n == 0:
        return 0.0
    return float((np.abs(a - b) * m3).sum() / n / 3.0)


print("== 1. 背景保留（原图 vs 终图，排除蝙蝠+文字）MAE(应≈0) 与 逐通道均值偏移 ==")
for t, f in finals.items():
    ff = np.array(f).astype(np.float32)
    print(f"  {t:6s}: MAE={reg_mae(o, ff, bg3):.3f}  dRGB="
          f"({ff[:,:,0][bg>0].mean()-o[:,:,0][bg>0].mean():+.2f},"
          f"{ff[:,:,1][bg>0].mean()-o[:,:,1][bg>0].mean():+.2f},"
          f"{ff[:,:,2][bg>0].mean()-o[:,:,2][bg>0].mean():+.2f})")

print("== 2. 蝙蝠裂变（终图之间，仅蝙蝠 mask 内）MAE(越高=姿态差越大) ==")
fa = {t: np.array(f).astype(np.float32) for t, f in finals.items()}
for a in ("up", "spread", "fold"):
    for b in ("up", "spread", "fold"):
        if a < b:
            print(f"  {a:6s} vs {b:6s}: {reg_mae(fa[a], fa[b], ma):.2f}")

print("== 3. 蝙蝠区 vs 原图（终图蝙蝠 vs 原图蝙蝠）MAE(衡量重绘幅度) ==")
om = np.array(Image.open(COMFY_INPUT / "v279_text_free_source.png").convert("RGB")).astype(np.float32)
for t in ("up", "spread", "fold"):
    print(f"  {t:6s}: {reg_mae(om, fa[t], ma):.2f}")

print("== 4. EasyOCR 各终图文字 ==")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
for t, f in finals.items():
    print(f"  {t:6s}: {[x[1] for x in reader.readtext(np.array(f))]}")
