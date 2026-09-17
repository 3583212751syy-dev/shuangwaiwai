"""p6 主体提纯诊断 v2：用「局部对比度」把实心插画主体从平坦白色射线里分离。

原理：闪电射线是**平坦纯白**（局部标准差≈0）；鹰/骷髅是**刻画插画**（内部大量黑线+
灰调 → 局部标准差大）。用 box 滤波算局部 std → 阈值 → 闭运算补实 → 取大连通域。
"""
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


def local_std(lum, k):
    m1 = ndi.uniform_filter(lum, k)
    m2 = ndi.uniform_filter(lum * lum, k)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def keep_big(m, min_px):
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return m
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    k = np.zeros(n + 1, bool)
    k[1:] = sz >= min_px
    return k[lab]


def bbox(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return "empty"
    return f"x{xs.min()}-{xs.max()} y{ys.min()}-{ys.max()} ({xs.max()-xs.min()}x{ys.max()-ys.min()})"


def build(img, k=31, sthr=9.0, close_r=26):
    a = np.asarray(img, np.float32)
    H, W = a.shape[:2]
    lum = R._lum(a)
    m = R.mask_p6(img)
    sd = local_std(lum, k)
    det = m & (sd > sthr)
    det = ndi.binary_closing(det, structure=R._disk(close_r))
    det = ndi.binary_dilation(det, structure=R._disk(10)) & \
        ndi.binary_dilation(m, structure=R._disk(6))
    det = keep_big(det, 20000)
    yy = np.mgrid[0:H, 0:W][0]
    YC, YB = 2660, 2400
    eagle = keep_big(det & (yy < YC), 20000)
    skull = keep_big(det & (yy >= YB), 20000)
    return a, m, det, eagle, skull


def main():
    img = Image.open(SRC).convert("RGB")
    a = np.asarray(img, np.float32)
    H, W = a.shape[:2]
    best = None
    for k, sthr, cr in ((31, 9.0, 26), (21, 7.0, 22), (41, 11.0, 30)):
        _, m, det, eg, sk = build(img, k, sthr, cr)
        print(f"[k={k} sthr={sthr} close={cr}] det%={100*det.mean():.1f} "
              f"eagle%={100*eg.mean():.1f} {bbox(eg)} | skull%={100*sk.mean():.1f} {bbox(sk)}")
        if best is None:
            best = (det, eg, sk)
    det, eg, sk = best
    for nm, mk, col in (("det", det, (255, 255, 0)), ("eagle", eg, (0, 255, 0)),
                        ("skull", sk, (255, 0, 0))):
        ov = a.copy()
        ov[mk] = ov[mk] * 0.28 + np.array(col, np.float32) * 0.72
        Image.fromarray(ov.astype(np.uint8)).resize((W // 6, H // 6),
                                                     Image.LANCZOS).save(
            str(J / f"_p6v2_{nm}_ov.jpg"), quality=88)
    ov = a.copy()
    ov[sk] = ov[sk] * 0.28 + np.array([255, 0, 0], np.float32) * 0.72
    ov[eg] = ov[eg] * 0.28 + np.array([0, 255, 0], np.float32) * 0.72
    Image.fromarray(ov.astype(np.uint8)).resize((W // 6, H // 6),
                                                Image.LANCZOS).save(
        str(J / "_p6v2_parts_ov.jpg"), quality=88)
    print("saved _p6v2_parts_ov.jpg")


if __name__ == "__main__":
    main()
