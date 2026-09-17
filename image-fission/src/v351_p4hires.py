"""v351: p4 重生分辨率对照（1536 vs 2048）—— 唯一未实测的精修路线。

背景：p4 图 1242x1754，重生裁块 ~1382x1894。`max_side=1536` 会把它**降采样到
0.811x** 再升回 → 最细 7px 的叶片线宽只剩 ~5.7px 的有效像素 → 二值化后必然
发毛。v345_p4sweep 里写过 `F_..._2048` 变体，但那次扫描只跑到 C 就停了，
**2048 从未真正跑过**（产物目录只有 A/B/C）。

本脚本补齐：
  H1_ms1536 = 交付基准（复现 B1）
  H2_ms2048 = 原生 1:1 渲染（1382x1894，2.6M px）

两者都套**管线同款密度匹配 snap**（全局阈值 t50 + 去碎点≥60px）再合成，
保证对比只反映"重生分辨率"这一个自由度。
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp            # noqa: E402
import v342_rebirth as R                               # noqa: E402
from v345_p4sweep import P_ST, N_ST, CANNY             # noqa: E402

OUT = ROOT / "jobs" / "v351_p4hires"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
WARP = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
INK_RGB = np.array([12.0, 10.0, 9.0], np.float32)

BASE = dict(prompt=P_ST, neg=N_ST, denoise=0.90, cn_name=CANNY,
            cn_strength=0.35, cn_end=0.45, wide=120, bg_gate=120.0,
            subject_dark=True, seed=7)
VARIANTS = [
    ("H1_ms1536", dict(max_side=1536), 1.0),
    ("H2_ms2048", dict(max_side=2048), 1.0),
    ("H3_pre15_ms4096", dict(max_side=4096), 1.5),
    ("H4_pre20_ms4096", dict(max_side=4096), 2.0),
]


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def snap(gen, base, ink, W, H):
    lg = R._lum(np.asarray(gen, np.float32))
    sel = ndi.binary_dilation(ink, structure=disk(4))
    frac = min(0.95, max(0.02, float(ink.sum()) / float(sel.sum())))
    t = float(np.clip(np.percentile(lg[sel], 100.0 * frac), 20.0, 200.0))
    ni = (lg < t) & sel
    lab, n = ndi.label(ni, structure=np.ones((3, 3), bool))
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sz >= 60
        ni = keep[lab]
    a3 = ni.astype(np.float32)[..., None]
    res = Image.fromarray(np.clip(base * (1 - a3) + INK_RGB[None, None, :] * a3,
                                  0, 255).astype(np.uint8), "RGB")
    e = ndi.distance_transform_edt(ni)
    _, nn = ndi.label(ni, structure=np.ones((3, 3), bool))
    return res, dict(t=t, ink=100 * float(ni.mean()),
                     w=2.0 * float(np.median(e[ni])) if ni.any() else 0.0, blocks=nn)


def main():
    only = sys.argv[1:] or None
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    mask = R.mask_p4(img)
    base = np.asarray(Image.open(WARP).convert("RGB").resize((W, H), Image.LANCZOS),
                      np.float32)
    ink = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    eo = ndi.distance_transform_edt(ink)
    _, o_n = ndi.label(ink, structure=np.ones((3, 3), bool))
    print(f"[v351] 原图 墨={100*ink.mean():.2f}% 笔宽={2*np.median(eo[ink]):.2f} "
          f"块数={o_n}")
    for name, kw, pre in VARIANTS:
        if only and name not in only:
            continue
        t0 = time.time()
        if pre > 1.0:
            iw, ih = int(round(W * pre)), int(round(H * pre))
            inj = img.resize((iw, ih), Image.LANCZOS)
            msk = Image.fromarray((mask.astype(np.uint8) * 255), "L").resize(
                (iw, ih), Image.NEAREST)
            msk = np.asarray(msk) > 127
        else:
            inj, msk = img, mask
        gen = R.rebirth_subject(inj, msk, tag=name, **{**BASE, **kw})
        if gen.size != (W, H):
            gen = gen.resize((W, H), Image.LANCZOS)
        gen.save(str(OUT / f"{name}_raw.jpg"), quality=95)
        res, q = snap(gen, base, ink, W, H)
        res.save(str(OUT / f"{name}.jpg"), quality=95)
        print(f"[{name}] {time.time()-t0:.0f}s t={q['t']:.1f} 墨={q['ink']:.2f}% "
              f"笔宽={q['w']:.2f} 块数={q['blocks']}")


if __name__ == "__main__":
    main()
