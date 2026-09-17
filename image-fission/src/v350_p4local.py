"""v350: p4 「块级局部自适应密度匹配」阈值 —— 治"叶片发毛/羽化"。

【病灶定位】（v349 交付件肉眼复核 + `camo_palm_pattern.fission` 第 595-605 行）
   rebirth 分支用的是**全图单一阈值**：
       _t = percentile(_lg[_sel], 100 * ink.sum() / sel.sum())
   SDXL 重画的棕榈叶是**大量深浅不一的细线**。单一阈值只保住"够黑"的那部分，
   浅一档的细叶被整片切断 → 短虚线 → 肉眼读成"绒毛毛边／线条比原图毛"（用户红线）。

【修法】块级局部密度匹配（local density matching）
   把图切成 B×B 块，每块（只统计 sel 内）各求**自己的**阈值，使该块落墨比例
   == 原图同一块的墨比例：
       d0_b = ink[block∩sel].sum() / sel[block∩sel].sum()
       T_b  = percentile(lg[block∩sel], 100*(1-d0_b))
   · 细叶块（原图墨密、SDXL 画得浅）→ 阈值自动抬高 → 细叶恢复连续；
   · 空白块 → T_b = 0（不落墨，靠 sel 限制在墨迹邻域内）；
   · 块缝：T 场线性上采样 + 高斯平滑；
   · 安全网：夹在 [lo, hi] × 全局阈值内，防个别块跑飞。

【QC 主指标】local-density MAE = mean( |块内落墨比例 − 原图块内墨比例| )
   全局阈值版这一指标高（浅叶块系统性掉墨），局部版应大幅下降。
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

OUT = ROOT / "jobs" / "v350_p4local"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
REB = ROOT / "jobs" / "router_out_v329" / "_p4_rebirth_raw.jpg"
WARP = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
INK_RGB = np.array([12.0, 10.0, 9.0], np.float32)

VARIANTS = [
    # name,  block, lo,   hi,   presmooth
    ("L1_b48_lo50_hi180_s0.0", 48, 0.50, 1.80, 0.0),
    ("L2_b48_lo50_hi180_s0.8", 48, 0.50, 1.80, 0.8),
    ("L3_b32_lo60_hi160_s0.6", 32, 0.60, 1.60, 0.6),
    ("L4_b64_lo50_hi200_s0.8", 64, 0.50, 2.00, 0.8),
    ("L5_b48_lo70_hi140_s0.8", 48, 0.70, 1.40, 0.8),
]


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def block_density_mae(mask, ref, B):
    """块级 |落墨比例 − 原墨比例| 均值（只在 ref 有墨的块上统计）。"""
    H, W = mask.shape
    ny, nx = (H + B - 1) // B, (W + B - 1) // B
    m = np.zeros((ny * B, nx * B), bool); m[:H, :W] = mask
    r = np.zeros((ny * B, nx * B), bool); r[:H, :W] = ref
    mb = m.reshape(ny, B, nx, B).mean((1, 3))
    rb = r.reshape(ny, B, nx, B).mean((1, 3))
    selb = rb > 0.01
    if not selb.any():
        return 0.0
    return float(np.abs(mb - rb)[selb].mean())


def main():
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    reb = Image.open(REB).convert("RGB")
    if reb.size != (W, H):
        reb = reb.resize((W, H), Image.LANCZOS)
    base = Image.open(WARP).convert("RGB")
    if base.size != (W, H):
        base = base.resize((W, H), Image.LANCZOS)
    _base = np.asarray(base, np.float32)

    ink = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    sel = ndi.binary_dilation(ink, structure=disk(4))
    lg = np.asarray(reb, np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)

    frac = float(ink.sum()) / max(1, int(sel.sum()))
    t_glob = float(np.clip(np.percentile(lg[sel], 100.0 * frac), 20.0, 200.0))
    g0 = (lg < t_glob) & sel
    print(f"[v350] size={W}x{H} 原墨={100*ink.mean():.2f}% sel={100*sel.mean():.1f}% "
          f"frac={frac:.4f} t_glob={t_glob:.1f} 全局版墨={100*g0.mean():.2f}%")

    rows = [("G0_global(旧交付)", g0, t_glob, block_density_mae(g0, ink, 48), 48, 0.0)]
    _a3g = g0.astype(np.float32)[..., None]
    Image.fromarray(np.clip(_base * (1.0 - _a3g) + INK_RGB[None, None, :] * _a3g,
                            0, 255).astype(np.uint8), "RGB").save(
        str(OUT / "G0_global(旧交付).jpg"), quality=95)
    W_ = np.asarray(img, np.float32)
    _blk = INK_RGB[None, None, :]

    for name, B, lo, hi, ps in VARIANTS:
        src = ndi.gaussian_filter(lg, ps) if ps > 0 else lg
        ny, nx = (H + B - 1) // B, (W + B - 1) // B
        Tg = np.zeros((ny, nx), np.float32)
        for iy in range(ny):
            y0, y1 = iy * B, min(H, (iy + 1) * B)
            for ix in range(nx):
                x0, x1 = ix * B, min(W, (ix + 1) * B)
                s = sel[y0:y1, x0:x1]
                ns = int(s.sum())
                if ns < 250:
                    Tg[iy, ix] = t_glob
                    continue
                d = float((ink[y0:y1, x0:x1] & s).sum()) / ns
                if d < 0.004:
                    Tg[iy, ix] = 0.0
                    continue
                Tg[iy, ix] = float(np.percentile(src[y0:y1, x0:x1][s], 100.0 * (1.0 - d)))
        w = (Tg > 0).astype(np.float32)
        z = lambda a: ndi.gaussian_filter(ndi.zoom(a, B, order=1)[:H, :W], B / 2.5)
        den = z(w)
        Tf = np.where(den > 0.15, z(Tg * w) / np.maximum(den, 1e-6), 0.0)
        Tf = np.where(Tf > 0, np.clip(Tf, t_glob * lo, t_glob * hi), -1.0)
        ni = (src < Tf) & sel

        lab, n = ndi.label(ni, structure=np.ones((3, 3), bool))
        if n:
            sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sz >= 40
            ni = keep[lab]
        # 桥接 1-2px 断口（治"短虚线"观感），再清一次碎点
        ni = ndi.binary_closing(ni, structure=disk(1))
        lab, n = ndi.label(ni, structure=np.ones((3, 3), bool))
        if n:
            sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sz >= 40
            ni = keep[lab]

        a3 = ni.astype(np.float32)[..., None]
        out = _base * (1.0 - a3) + _blk * a3
        res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
        res.save(str(OUT / f"{name}.jpg"), quality=95)
        rows.append((name, ni, Tf, block_density_mae(ni, ink, B), B, lo))

    # ---- QC 表 ----
    lines = []
    o_edt = ndi.distance_transform_edt(ink)
    o_w = 2.0 * float(np.median(o_edt[ink]))
    _, o_n = ndi.label(ink, structure=np.ones((3, 3), bool))
    o_sz = ndi.sum(np.ones_like(_), _, range(1, o_n + 1))
    lines.append(f"原图      墨={100*ink.mean():6.2f}%  笔宽median={o_w:5.2f}px  "
                 f"连通块={o_n:5d}(≥40px {int((o_sz >= 40).sum()):4d})")
    for name, mask, _t, mae, B, lo in rows:
        e = ndi.distance_transform_edt(mask)
        wd = 2.0 * float(np.median(e[mask])) if mask.any() else 0.0
        _, nn = ndi.label(mask, structure=np.ones((3, 3), bool))
        sz = ndi.sum(np.ones_like(_), _, range(1, nn + 1)) if nn else []
        big = int((sz >= 40).sum()) if nn else 0
        lines.append(f"{name:24s} 墨={100*mask.mean():6.2f}%  笔宽={wd:5.2f}px  "
                     f"块密度MAE={mae:.4f}  连通块={nn:5d}(≥40px {big:4d})")
    txt = "\n".join(lines)
    (OUT / "QC_v350.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
