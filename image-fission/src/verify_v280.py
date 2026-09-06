"""v280 客观质检：强制姿势引导下的裂变 + 背景保留 + OCR。"""
import numpy as np
from pathlib import Path
from PIL import Image
import easyocr

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
JOB = PROJECT / "jobs" / "v280"
ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"

cx, cy, r_badge = 776, 746, 300

text_rects = [(1000, 150, 1180, 1400), (1180, 350, 1350, 1200),
              (720, 300, 880, 620), (720, 930, 880, 1250)]

orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
finals = {t: Image.open(JOB / f"v280_{t}_final.png").convert("RGB") for t in ("up", "spread", "fold")}

H, W = np.array(orig).shape[:2]
ys, xs = np.ogrid[:H, :W]
badge = (xs - cx) ** 2 + (ys - cy) ** 2 <= r_badge ** 2
bg = np.ones((H, W), np.float32) - badge.astype(np.float32)
for y0, x0, y1, x1 in text_rects:
    bg[y0:y1, x0:x1] = 0

o = np.array(orig).astype(np.float32)
ba3 = badge[:, :, None].astype(np.float32)
bg3 = bg[:, :, None]


def reg_mae(a, b, m3):
    n = m3.sum()
    if n == 0:
        return 0.0
    return float((np.abs(a - b) * m3).sum() / n / 3.0)


print("== 1. 背景保留（原图 vs 终图，排除 inner badge+文字）MAE(应≈0) 与 逐通道均值偏移 ==")
for t, f in finals.items():
    ff = np.array(f).astype(np.float32)
    print(f"  {t:6s}: MAE={reg_mae(o, ff, bg3):.3f}  dRGB="
          f"({ff[:,:,0][bg>0].mean()-o[:,:,0][bg>0].mean():+.2f},"
          f"{ff[:,:,1][bg>0].mean()-o[:,:,1][bg>0].mean():+.2f},"
          f"{ff[:,:,2][bg>0].mean()-o[:,:,2][bg>0].mean():+.2f})")

print("== 2. Inner badge 内裂变（终图之间，仅 badge 圆内）MAE(越高=姿态差越大) ==")
fa = {t: np.array(f).astype(np.float32) for t, f in finals.items()}
for a in ("up", "spread", "fold"):
    for b in ("up", "spread", "fold"):
        if a < b:
            print(f"  {a:6s} vs {b:6s}: {reg_mae(fa[a], fa[b], ba3):.2f}")

print("== 3. Inner badge vs 原图（衡量重绘幅度）MAE ==")
for t in ("up", "spread", "fold"):
    print(f"  {t:6s}: {reg_mae(o, fa[t], ba3):.2f}")

print("== 4. EasyOCR 各终图文字 ==")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
for t, f in finals.items():
    print(f"  {t:6s}: {[x[1] for x in reader.readtext(np.array(f))]}")
