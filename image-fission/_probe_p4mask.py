"""_probe_p4mask.py — 看清"树干横档被焊死"到底发生在哪一步。

假设：cpp.tree_ink_mask 用 lum<55，而横档之间的"缝"其实是**棕褐色杆身**(lum 30~55,
sat≈28) → 在掩膜里档与缝都是墨 → 换位后整根杆被涂成近黑 = 实心条 = 用户说的"一块块"。
本脚本把几档掩膜渲染成图（黑墨/白底）+ 量化连通块数，肉眼 + 数字双证。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from styles import camo_palm_pattern as cpp  # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
OUT = ROOT / "jobs" / "_cmpcrop"
OUT.mkdir(parents=True, exist_ok=True)

img = Image.open(SRC).convert("RGB")
W, H = img.size
a = np.asarray(img, np.float32)
lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
sat = a.max(2) - a.min(2)

masks = {
    "A_inkmask(lum<55,sat<14)": cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0,
                                                   min_px=25, thin_only=False),
    "B_lum<30": lum < 30.0,
    "C_lum<45": lum < 45.0,
    "D_lum<55": lum < 55.0,
}
for k, m in masks.items():
    lab, n = ndi.label(m, np.ones((3, 3), bool))
    sz = np.bincount(lab.ravel())[1:] if n else np.array([0])
    print(f"[{k}] ink={100*m.mean():.2f}% N={n} <12px={int((sz<12).sum())} "
          f"<50px={int((sz<50).sum())}")

# 树干局部：把各掩膜裁同一块，2x，黑墨白底
BOX = (int(0.40 * W), int(0.55 * H), int(0.40 * W) + 230, int(0.55 * H) + 230)
TH = 460
cols = []
for k, m in masks.items():
    sub = (m[BOX[1]:BOX[3], BOX[0]:BOX[2]] * 255).astype(np.uint8)
    im = Image.fromarray(255 - sub, "L").resize((TH, TH), Image.NEAREST)
    cols.append((k, im.convert("RGB")))
orig = img.crop(BOX).resize((TH, TH), Image.NEAREST)
cols.insert(0, ("ORIG", orig))
cv = Image.new("RGB", (TH * len(cols) + 10 * (len(cols) + 1), TH + 24), (245, 245, 245))
d = ImageDraw.Draw(cv)
x = 10
for k, im in cols:
    cv.paste(im, (x, 20))
    d.text((x + 2, 6), k[:28], fill=(0, 0, 0))
    x += TH + 10
cv.save(str(OUT / "p4_mask_probe.jpg"), quality=95)
print("saved", OUT / "p4_mask_probe.jpg", cv.size)
print(f"缝像素统计（ORIG 树干局部）：lum30~55 & sat<14 占比 = "
      f"{100*((lum[BOX[1]:BOX[3], BOX[0]:BOX[2]]>30)&(lum[BOX[1]:BOX[3], BOX[0]:BOX[2]]<55)&(sat[BOX[1]:BOX[3], BOX[0]:BOX[2]]<14)).mean():.1f}%")
