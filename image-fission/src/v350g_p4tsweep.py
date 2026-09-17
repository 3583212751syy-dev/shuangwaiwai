"""v350g: p4 全局阈值纯扫描 —— 找"连通块数"与"笔宽"的平衡工作点。

前几轮已排除：局部密度阈值(v350)、厚度门控阈值(v350d)、门控闭运算(v350f)。
本脚本只动一个自由度：全局阈值 t。目标 = 在笔宽不显著超过原图(4.47)的前提下，
把连通块数从 197 压向原图的 91（块数 = "发毛"的直接来源）。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp            # noqa: E402

OUT = ROOT / "jobs" / "v350g_p4tsweep"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
REB = ROOT / "jobs" / "router_out_v329" / "_p4_rebirth_raw.jpg"
WARP = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
INK_RGB = np.array([12.0, 10.0, 9.0], np.float32)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def main():
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    base = np.asarray(Image.open(WARP).convert("RGB").resize((W, H), Image.LANCZOS),
                      np.float32)
    reb = Image.open(REB).convert("RGB")
    if reb.size != (W, H):
        reb = reb.resize((W, H), Image.LANCZOS)
    ink = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    sel = ndi.binary_dilation(ink, structure=disk(4))
    lg = np.asarray(reb, np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    eo = ndi.distance_transform_edt(ink)
    _, o_n = ndi.label(ink, structure=np.ones((3, 3), bool))
    print(f"原图 墨={100*ink.mean():.2f}% 笔宽={2*np.median(eo[ink]):.2f} 块数={o_n}")

    for t in (34, 38, 42, 46, 50, 54, 58, 62, 68, 76):
        ni = (lg < float(t)) & sel
        lab, n = ndi.label(ni, structure=np.ones((3, 3), bool))
        if n:
            sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sz >= 40
            ni = keep[lab]
        e = ndi.distance_transform_edt(ni)
        wd = 2.0 * float(np.median(e[ni])) if ni.any() else 0.0
        _, nn = ndi.label(ni, structure=np.ones((3, 3), bool))
        a3 = ni.astype(np.float32)[..., None]
        Image.fromarray(np.clip(base * (1 - a3) + INK_RGB[None, None, :] * a3, 0, 255)
                        .astype(np.uint8), "RGB").save(
            str(OUT / f"T{int(t):03d}.jpg"), quality=93)
        print(f"t={t:3d}  墨={100*ni.mean():6.2f}%  笔宽={wd:5.2f}  块数={nn:4d}")


if __name__ == "__main__":
    main()
