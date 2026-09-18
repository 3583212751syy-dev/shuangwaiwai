"""v376：p3 主蝶 / 小蝶 掩膜探测（为后续 SDXL 重生定位）。"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "jobs" / "v376_p3rb"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (3).jpg")

_W = np.array([0.299, 0.587, 0.114], np.float32)


def _lum(a):
    return a @ _W


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def fg_mask(img):
    a = np.asarray(img, np.float32)
    lum = _lum(a)
    sat = a.max(2) - a.min(2)
    return (lum < 210) | (sat > 26)


if __name__ == "__main__":
    img = Image.open(SRC).convert("RGB")
    a = np.asarray(img, np.float32)
    print("size", img.size)
    print("背景角点 lum", _lum(a[5:15, 5:15].reshape(-1, 3)).mean(),
          " 底色中位", np.median(_lum(a)))
    fg = fg_mask(img)
    fg[:360, :] = False                       # 字母带
    fgc = ndi.binary_closing(fg, structure=_disk(9))
    lab, n = ndi.label(fgc, structure=np.ones((3, 3), bool))
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    order = np.argsort(sz)[::-1][:8]
    for j in order:
        m = (lab == j + 1)
        ys, xs = np.where(m)
        print(f"  cc#{j+1} area={int(sz[j])} bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})"
              f" wh={xs.max()-xs.min()+1}x{ys.max()-ys.min()+1}")
    # 可视化
    vis = np.asarray(img).copy()
    vis[ndi.binary_dilation(fgc, structure=_disk(3))] = [255, 60, 60]
    Image.fromarray(vis).save(str(OUT / "mask_vis.jpg"), quality=92)
    print("saved", OUT / "mask_vis.jpg")
