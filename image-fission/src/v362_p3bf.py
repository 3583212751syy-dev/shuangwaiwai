#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v362 —— p3 牛仔蝶「轮廓形状重铸」裂变

前两版失败的教训：
  · 只做比例拉伸 → 用户看不出"裂变"（还是同一只蝶被拉宽）
  · 硬做右半镜像覆盖左半 → 实物贴布左右细节不同，变成"半蝶复制"，反而失真

本版：**极坐标形状重铸**
  ① 取蝶层 → 身体轴 = 使左右镜像残差最小的 x；极心 = (轴, 质心y)
  ② 扫描原轮廓半径 R_old(φ)（φ 自正上方量起），重平滑(31/1440≈7.8°) +
     左右对称化 → 修掉"一边大一边小"
  ③ 新轮廓 R_new(φ) = R_old(φ)·f(φ)，f 只**外扩**（前翅更长/后翅更外展），
     保证 R_new ≥ R_old → 原图毛边流苏自然落在**新轮廓**上
  ④ 逐像素逆映射 ρ·R_old/R_new 采样原图 → 牛仔纹理/绣线/毛边 100% 保留，
     翅形却是全新的 → 真正的"异形同材质"
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path("E:/Desktop/双接口/image-fission")
OUT = ROOT / "jobs" / "v362_p3bf"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (3).jpg")

NB = 1440
TARGETS = [
    ("main", (40, 480, 700, 1110), dict(af=0.30, ah=0.38, scale=1.02)),
    ("up",   (395, 295, 575, 445), dict(af=0.30, ah=0.46, scale=1.04)),
    ("dn",   (200, 1065, 355, 1245), dict(af=0.32, ah=0.44, scale=1.04)),
]


def find_axis(mask, span=45):
    """使左右镜像残差最小的身体轴 x。"""
    h, w = mask.shape
    best, bx = None, w // 2
    for x0 in range(span + 2, w - span - 2):
        d = np.abs(mask[:, x0 - 1::-1][:, :span].astype(np.float32)
                   - mask[:, x0 + 1:x0 + 1 + span].astype(np.float32)).mean()
        if best is None or d < best:
            best, bx = d, x0
    return bx


def radial_profile(mask, cx, cy, nb=NB):
    """轮廓半径 R(φ)，φ 自正上方量起（顺时针为正），重平滑 + 左右对称化。"""
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(xx - cx, -(yy - cy))
    bi = ((phi + np.pi) / (2 * np.pi) * nb).astype(np.int32) % nb
    sel = mask & (r > 0.5)
    R = np.zeros(nb, np.float32)
    np.maximum.at(R, bi[sel], r[sel])
    bad = R <= 0
    if bad.any():
        good = np.nonzero(~bad)[0]
        if len(good) == 0:
            R[:] = 1.0
        else:
            R[bad] = np.interp(np.nonzero(bad)[0], good, R[good], period=nb)
    # 重平滑：原图毛边会让 R(φ) 毛糙；直接当新轮廓 → 边界锯齿/撕裂（实测小蝶被撕碎）
    R = ndi.uniform_filter1d(R, 31, mode='wrap')
    Rs = R.copy()
    for i in range(nb):
        Rs[i] = 0.5 * (R[i] + R[(-i) % nb])
    return Rs


def remap_shape(rgba, cx, cy, Rold, prm, nb=NB):
    H, W = rgba.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    r = np.hypot(dx, dy)
    phi = np.arctan2(dx, -dy)
    a = np.abs(phi)
    f = prm["scale"] + prm["af"] * np.exp(-((a - 1.05) / 0.55) ** 2) \
        + prm["ah"] * np.exp(-((a - 2.30) / 0.62) ** 2)
    bi = ((phi + np.pi) / (2 * np.pi) * nb).astype(np.int32) % nb
    Rn = Rold[bi] * f
    inside = r <= Rn
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(inside, Rold[bi] / np.maximum(Rn, 1e-3), 0.0)
    sr = r * ratios
    sx = cx + sr * np.sin(phi)
    sy = cy - sr * np.cos(phi)
    out = np.zeros_like(rgba)
    for c in range(4):
        out[..., c] = ndi.map_coordinates(rgba[..., c], [sy, sx], order=1,
                                          mode='constant', cval=0.0)
    out[..., 3] *= inside.astype(np.float32)
    return out


def main():
    t0 = time.time()
    im = Image.open(SRC).convert("RGB")
    a = np.asarray(im).astype(np.float32)
    H, W, _ = a.shape
    q = (a // 4).astype(np.int64)
    key = q[..., 0] * 4096 + q[..., 1] * 64 + q[..., 2]
    _, first, cnt = np.unique(key, return_index=True, return_counts=True)
    bgv = a.reshape(-1, 3)[first[np.argmax(cnt)]]
    print(f"原图 {W}x{H}  底色={bgv.round(0).tolist()}")

    res = a.copy()
    for name, box, prm in TARGETS:
        x0, y0, x1, y1 = box
        sub = a[y0:y1, x0:x1]
        fg = np.abs(sub - bgv[None, None, :]).max(2) > 16
        lbl, n = ndi.label(fg)
        if n == 0:
            print(f"[{name}] 无前景"); continue
        sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        fg = lbl == (int(np.argmax(sizes)) + 1)
        ys_, xs_ = np.nonzero(fg)
        cx = float(find_axis(fg))
        cy = float(ys_.mean())
        Rold = radial_profile(fg, cx, cy)
        print(f"[{name}] 前景={100*fg.mean():.1f}%  极心=({cx:.0f},{cy:.0f})  "
              f"R_old {Rold.min():.0f}..{Rold.max():.0f}")

        rgba = np.dstack([sub, (fg * 255).astype(np.float32)])
        o = remap_shape(rgba, cx, cy, Rold, prm)

        dst = res[y0:y1, x0:x1]
        dst[ndi.binary_dilation(fg, iterations=2)] = bgv
        al = (o[..., 3:4] / 255.0).clip(0, 1)
        res[y0:y1, x0:x1] = dst * (1 - al) + o[..., :3] * al
        Image.fromarray(res.clip(0, 255).astype(np.uint8)[y0:y1, x0:x1]).save(
            OUT / f"crop_{name}.jpg", quality=95)

    out_img = Image.fromarray(res.clip(0, 255).astype(np.uint8))
    out_img.save(OUT / "p3_bf.jpg", quality=95)
    print(f"完成 {time.time()-t0:.0f}s -> {OUT}")
    for name, box, _ in TARGETS:
        x0, y0, x1, y1 = box
        cw, ch = x1 - x0, y1 - y0
        s = Image.new("RGB", (cw * 2 + 30, ch + 20), (205, 205, 205))
        s.paste(im.crop(box), (10, 10))
        s.paste(out_img.crop(box), (cw + 20, 10))
        s.save(OUT / f"S_{name}.jpg", quality=95)


if __name__ == "__main__":
    main()
