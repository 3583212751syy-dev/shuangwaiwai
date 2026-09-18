#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v363 —— p3 牛仔蝶 SDXL 重生试验（照片级材质，SDXL 强项）。"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import v342_rebirth as R                              # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (3).jpg")
OUT = ROOT / "jobs" / "v363_p3rb"
OUT.mkdir(parents=True, exist_ok=True)

P_MAIN = ("denim patch butterfly applique, embroidered butterfly with frayed "
          "distressed denim fringe edges, blue denim fabric texture, navy "
          "thread wing veins, symmetrical winged insect, product photo, "
          "plain light gray background")
N_MAIN = ("asymmetric, extra wings, missing wing, text, letters, watermark, "
          "blurry, lowres, jpeg artifacts, deformed, cropped, outline, sketch")

VARIANTS = [
    ("A_dn72_cn45e60", dict(denoise=0.72, cn_strength=0.45, cn_end=0.60,
                            grow=10, wide=60, seed=911)),
    ("B_dn82_cn35e55", dict(denoise=0.82, cn_strength=0.35, cn_end=0.55,
                            grow=10, wide=90, seed=912)),
    ("C_dn62_cn60e75", dict(denoise=0.62, cn_strength=0.60, cn_end=0.75,
                            grow=8, wide=40, seed=913)),
]


def main():
    im = Image.open(SRC).convert("RGB")
    a = np.asarray(im)
    box = (40, 480, 700, 1110)
    x0, y0, x1, y1 = box
    sub = im.crop(box)
    # 底色
    q = (a.astype(np.int32) // 4)
    key = q[..., 0] * 4096 + q[..., 1] * 64 + q[..., 2]
    _, first, cnt = np.unique(key, return_index=True, return_counts=True)
    bgv = a.reshape(-1, 3)[first[np.argmax(cnt)]]
    fg = np.abs(np.asarray(sub).astype(np.float32) - bgv).max(2) > 16
    lbl, n = ndi.label(fg)
    sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    fg = lbl == (int(np.argmax(sizes)) + 1)
    mask = Image.fromarray((ndi.binary_dilation(fg, iterations=6) * 255)
                           .astype(np.uint8))
    print("fg=", fg.mean(), "box=", sub.size)

    t0 = time.time()
    for name, kw in VARIANTS:
        try:
            gen = R.rebirth_subject(sub, mask, P_MAIN, N_MAIN, tag=name, **kw)
            gen.save(OUT / f"{name}.jpg", quality=95)
            print(f"[{name}] ok {time.time()-t0:.0f}s")
        except Exception as e:                        # noqa: BLE001
            print(f"[{name}] FAIL {e}")
    # 对照
    ims = [(name, Image.open(OUT / f"{name}.jpg")) for name, _ in VARIANTS]
    W = sum(i.width for _, i in ims) + 20 * (len(ims) + 1)
    H = max(i.height for _, i in ims) + 40
    sh = Image.new("RGB", (W, H), (205, 205, 205))
    x = 20
    for _, i in ims:
        sh.paste(i, (x, 20)); x += i.width + 20
    sh.save(OUT / "S_cmp.jpg", quality=95)
    # 原图同区
    sub.save(OUT / "orig.jpg", quality=95)
    print("done")


if __name__ == "__main__":
    main()
