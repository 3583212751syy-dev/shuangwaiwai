#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v370 —— p3 牛仔蝶「轮廓再设计」裂变（可自由增减，不再只是拉伸）

v362 失败原因：f(φ) 只是两个高斯"外扩"→ 读成"同一只蝶被拉宽"。

关键修正（映射公式）：
    输出半径  Rn(φ) = R_old(φ_src) · f(φ) · s
    归一化    τ     = r / Rn(φ)
    源采样    s_r   = τ · R_old(φ_src)      ← 边界 τ=1 恒映射到**原轮廓**
  ⇒ 无论 f 是 >1（长出去）还是 <1（收进来），新边界永远采到原图的
    **毛边流苏**，所以形状可自由增减而不丢毛边。这是"真裂变"的自由度来源。

f(φ) 由「多个窄高斯凹凸」构成，用 |φ| 保持左右镜像对称：
  前翅尖端外移 + 外缘加**第二道凸包** + 后翅尾突加长 + 两处凹口（前翅缺口/翅间隙）
  ⇒ 得到一只**结构不同**的蝶，而不是被拉伸的同一只。
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi

ROOT = Path("E:/Desktop/双接口/image-fission")
OUT = ROOT / "jobs" / "v370_p3design"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (3).jpg")

NB = 1440

# (φ_center rad, 幅度, σ)   —— φ 自正上方量起，顺时针为正；用 |φ| 保对称
# 尺寸框来自整图前景连通块检测（main 223105px；up 8553px@435-571,342-476；
# dn 5027px@236-333,1171-1268），外扩 20px 含毛边，且不含 UPCY 字母与点线。
DESIGN = {
    "main": dict(
        box=(0, 470, 736, 1150), scale=0.97, c2=0.18, c4=0.06,
        gauss=[(0.78, +0.20, 0.30), (1.42, +0.26, 0.26),
               (2.35, +0.18, 0.32), (2.92, +0.30, 0.30),
               (0.16, -0.16, 0.16), (1.95, -0.14, 0.22)],
        sym=0.85),
    "up": dict(
        mode="rigid", box=(413, 320, 594, 498),
        rot=+13.0, mirror=True, sx=1.08, sy=0.94),
    "dn": dict(
        mode="rigid", box=(214, 1149, 355, 1290),
        rot=-11.0, mirror=True, sx=1.06, sy=0.96),
}


def find_axis(mask, span=45, cx0=None, frac=0.30):
    """使左右镜像残差最小的身体轴 x（只在预期中心 ±frac·宽 内搜）。"""
    h, w = mask.shape
    if cx0 is None:
        cx0 = w // 2
    lo = max(span + 2, int(cx0 - frac * w))
    hi = min(w - span - 2, int(cx0 + frac * w))
    best, bx = None, cx0
    for x0 in range(lo, hi + 1):
        d = np.abs(mask[:, x0 - 1::-1][:, :span].astype(np.float32)
                   - mask[:, x0 + 1:x0 + 1 + span].astype(np.float32)).mean()
        if best is None or d < best:
            best, bx = d, x0
    return bx


def waist_cy(mask, cx, frac=0.07):
    """身体中心 y：取中轴窄带内掩膜质心的 y（比整块质心稳）。"""
    w = mask.shape[1]
    band = max(2, int(w * frac))
    x0 = max(0, cx - band)
    x1 = min(w, cx + band + 1)
    sub = mask[:, x0:x1]
    ys, _ = np.nonzero(sub)
    if len(ys) == 0:
        ys, _ = np.nonzero(mask)
    lo, hi = ys.min(), ys.max()
    core = ys[(ys > lo + (hi - lo) * 0.30) & (ys < lo + (hi - lo) * 0.75)]
    return float(core.mean() if len(core) else ys.mean())


def radial_profile(mask, cx, cy, nb=NB, sym=1.0, smooth=31):
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
        R[bad] = (np.interp(np.nonzero(bad)[0], good, R[good], period=nb)
                  if len(good) else 1.0)
    R = ndi.uniform_filter1d(R, smooth, mode='wrap')
    if sym > 0:
        Rm = 0.5 * (R + R[(-np.arange(nb)) % nb])
        R = (1 - sym) * R + sym * Rm
    return R


def design_f(phi, gauss, harm=()):
    a = np.abs(phi)
    f = np.ones_like(a)
    for pc, amp, sg in gauss:
        f += amp * np.exp(-((a - pc) / sg) ** 2)
    for k, amp, ph in harm:                     # cos(kφ) 关于 φ→-φ 为偶 ⇒ 保镜像对称
        f += amp * np.cos(k * phi + ph)
    return np.maximum(f, 0.35)


def remap_design(rgba, cx, cy, Rold, prm, nb=NB):
    H, W = rgba.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    phi = np.arctan2(dx, -dy)
    phi_src = phi + prm["c2"] * np.sin(2 * phi) + prm["c4"] * np.sin(4 * phi)
    bi_s = ((phi_src + np.pi) / (2 * np.pi) * nb).astype(np.int32) % nb
    f = design_f(phi, prm["gauss"], prm.get("harm", ())) * prm["scale"]
    # 自动适配：保证新轮廓不越出可用半宽/半高（越界会被 box 硬切出直角边）
    phi_g = np.linspace(-np.pi, np.pi, nb, endpoint=False)
    big = ((phi_g + np.pi) / (2 * np.pi) * nb).astype(np.int32) % nb
    fg_ = design_f(phi_g, prm["gauss"], prm.get("harm", ())) * prm["scale"]
    Rg = Rold[big] * fg_
    px = Rg * np.sin(phi_g)
    py = -Rg * np.cos(phi_g)
    avail_w = min(cx, W - cx) - 4
    avail_h = min(cy, H - cy) - 4
    fit = min(1.0, avail_w / max(1e-3, np.abs(px).max()),
              avail_h / max(1e-3, np.abs(py).max()))
    f = f * fit
    prm["_fit"] = fit
    Rn = np.maximum(Rold[bi_s] * f, 1.0)
    inside = r <= Rn
    sr = r * (Rold[bi_s] / Rn)
    sx = cx + sr * np.sin(phi_src)
    sy = cy - sr * np.cos(phi_src)
    out = np.zeros_like(rgba)
    for c in range(4):
        out[..., c] = ndi.map_coordinates(rgba[..., c], [sy, sx], order=1,
                                          mode='constant', cval=0.0)
    out[..., 3] *= inside.astype(np.float32)
    return out, inside


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
    for name, prm in DESIGN.items():
        x0, y0, x1, y1 = prm["box"]
        if prm.get("mode") == "rigid":
            # 小元素：极坐标会折叠（翅间凹口使形状非星形）→ 改用刚体变换
            # 镜像 + 旋转 + 轻微各向异性缩放 = 干净、对称、且元素确实换了内容
            M = 90
            rx0, ry0 = max(0, x0 - M), max(0, y0 - M)
            rx1, ry1 = min(W, x1 + M), min(H, y1 + M)
            reg = a[ry0:ry1, rx0:rx1]
            fgm = np.zeros(reg.shape[:2], bool)
            fg = np.abs(a[y0:y1, x0:x1] - bgv[None, None, :]).max(2) > 16
            lbl, n = ndi.label(fg)
            if n:
                sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
                fg = lbl == (int(np.argmax(sizes)) + 1)
            fgm[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = ndi.binary_fill_holes(fg)
            lay = np.dstack([reg, (fgm * 255).astype(np.float32)])
            layi = Image.fromarray(lay.astype(np.uint8), "RGBA")
            if prm["mirror"]:
                layi = ImageOps.mirror(layi)
            ecx = (x0 + x1) / 2 - rx0
            ecy = (y0 + y1) / 2 - ry0
            rotd = layi.rotate(prm["rot"], resample=Image.BICUBIC, center=(ecx, ecy))
            sw, sh_ = rotd.size
            sc = Image.new("RGBA", (int(sw * prm["sx"]), int(sh_ * prm["sy"])),
                           (0, 0, 0, 0))
            rs = rotd.resize(sc.size, Image.LANCZOS)
            ox = int(ecx - rs.width / 2)
            oy = int(ecy - rs.height / 2)
            dest = res[ry0:ry1, rx0:rx1].copy()
            # 只清除**该元素自身**的足迹（不能整块填底色 → 会误伤 UPCY 字母）
            dest[ndi.binary_dilation(fgm, iterations=6)] = bgv
            pa = np.asarray(rs).astype(np.float32)
            al = (pa[..., 3:4] / 255.0).clip(0, 1)
            hh = min(dest.shape[0], rs.height)
            ww = min(dest.shape[1], rs.width)
            oy0, ox0 = max(0, oy), max(0, ox)
            ay0, ax0 = max(0, -oy), max(0, -ox)
            hh = min(hh, dest.shape[0] - oy0, rs.height - ay0)
            ww = min(ww, dest.shape[1] - ox0, rs.width - ax0)
            sub_d = dest[oy0:oy0 + hh, ox0:ox0 + ww]
            sub_p = pa[ay0:ay0 + hh, ax0:ax0 + ww]
            sub_a = al[ay0:ay0 + hh, ax0:ax0 + ww]
            sub_d[:] = sub_d * (1 - sub_a) + sub_p[..., :3] * sub_a
            res[ry0:ry1, rx0:rx1] = dest
            Image.fromarray(np.clip(res[ry0:ry1, rx0:rx1], 0, 255)
                            .astype(np.uint8)).save(OUT / f"crop_{name}.jpg",
                                                    quality=95)
            print(f"[{name}] rigid rot={prm['rot']:+.0f} mirror={prm['mirror']}"
                  f" sx={prm['sx']} sy={prm['sy']}")
            continue
        sub = a[y0:y1, x0:x1]
        fg = np.abs(sub - bgv[None, None, :]).max(2) > 16
        lbl, n = ndi.label(fg)
        if n == 0:
            print(f"[{name}] 无前景")
            continue
        sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        fg = lbl == (int(np.argmax(sizes)) + 1)
        fg = ndi.binary_fill_holes(fg)
        ys_, xs_ = np.nonzero(fg)
        cx0 = 0.5 * (xs_.min() + xs_.max())
        cx = float(find_axis(fg, span=min(40, int(0.18 * fg.shape[1])),
                             cx0=int(cx0), frac=0.18))
        cy = waist_cy(fg, int(cx), frac=0.10)
        Rold = radial_profile(fg, cx, cy, sym=prm["sym"],
                              smooth=int(prm.get("smooth", 31)))
        print(f"[{name}] fg={100*fg.mean():.1f}% 极心=({cx:.0f},{cy:.0f}) "
              f"R {Rold.min():.0f}..{Rold.max():.0f}")

        rgba = np.dstack([sub, (fg * 255).astype(np.float32)])
        o, inside = remap_design(rgba, cx, cy, Rold, prm)
        print(f"[{name}] 自适应缩放 fit={prm.get('_fit', 1):.3f}")

        dst = res[y0:y1, x0:x1].copy()
        dst[ndi.binary_dilation(fg, iterations=2)] = bgv
        al = (o[..., 3:4] / 255.0).clip(0, 1)
        res[y0:y1, x0:x1] = dst * (1 - al) + o[..., :3] * al
        Image.fromarray(np.clip(res[y0:y1, x0:x1], 0, 255).astype(np.uint8)) \
            .save(OUT / f"crop_{name}.jpg", quality=95)

    out_img = Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))
    out_img.save(OUT / "p3_design.jpg", quality=95)
    for name, prm in DESIGN.items():
        x0, y0, x1, y1 = prm["box"]
        cw, ch = x1 - x0, y1 - y0
        s = Image.new("RGB", (cw * 2 + 30, ch + 20), (205, 205, 205))
        s.paste(im.crop((x0, y0, x1, y1)), (10, 10))
        s.paste(out_img.crop((x0, y0, x1, y1)), (cw + 20, 10))
        s.save(OUT / f"S_{name}.jpg", quality=95)
    print(f"完成 {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
