"""p6 主体提纯诊断：把「鹰/骷髅实心体」从白色闪电射线中分离出来。"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import v342_rebirth as R

J = ROOT / "jobs"
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")


def keep_big(m, min_px):
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return m
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    k = np.zeros(n + 1, bool)
    k[1:] = sz >= min_px
    return k[lab]


def main():
    img = Image.open(SRC).convert("RGB")
    a = np.asarray(img, np.float32)
    H, W = a.shape[:2]
    lum = R._lum(a)
    sat = a.max(2) - a.min(2)
    m = R.mask_p6(img)
    core = ndi.binary_closing(ndi.binary_opening(m, structure=R._disk(22)),
                              structure=R._disk(14))
    brown = (sat > 34) & (lum > 55) & (lum < 235) & ndi.binary_dilation(m, R._disk(4))
    subj = keep_big(core | brown, 4000)
    yy = np.mgrid[0:H, 0:W][0]
    YC, YB = 2660, 2390                 # 重叠带 2390~2660：鹰（含爪）压骷髅
    eagle = keep_big(subj & (yy < YC), 4000)
    skull = keep_big(subj & (yy >= YB), 4000)
    for nm, mk, col in (("eagle", eagle, (0, 255, 0)), ("skull", skull, (255, 0, 0))):
        ov = a.copy()
        ov[mk] = ov[mk] * 0.30 + np.array(col, np.float32) * 0.70
        Image.fromarray(ov.astype(np.uint8)).resize((W // 6, H // 6),
                                                     Image.LANCZOS).save(
            str(J / f"_p6_{nm}_ov.jpg"), quality=88)
        ys, xs = np.where(mk)
        print(f"[{nm}] area%={100*mk.mean():.1f} bbox x{xs.min()}-{xs.max()} "
              f"y{ys.min()}-{ys.max()}  crop={xs.max()-xs.min()}x{ys.max()-ys.min()}")
    both = a.copy()
    both[skull] = both[skull] * 0.3 + np.array([255, 0, 0], np.float32) * 0.7
    both[eagle] = both[eagle] * 0.3 + np.array([0, 255, 0], np.float32) * 0.7
    Image.fromarray(both.astype(np.uint8)).resize((W // 6, H // 6),
                                                  Image.LANCZOS).save(
        str(J / "_p6_parts_ov.jpg"), quality=88)
    print("saved", J / "_p6_parts_ov.jpg")


if __name__ == "__main__":
    main()
