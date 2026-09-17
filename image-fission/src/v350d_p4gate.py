"""v350d: p4 「笔画厚度门控」局部阈值 —— 分治治毛。

【v350 首轮结论（肉眼 + 量化）】
  · 全局阈值 G0：笔宽 4.47px（=原图，✓ 保住树干细环纹），但连通块 1648（原图 91）
    → 叶片被切成大量细须 → 肉眼"发毛"。
  · 局部密度匹配 L1/L4：连通块降到 153（✓ 叶片合并），但笔宽升到 6.00px
    → **树干斜环纹被压成粗梯子**（更糊）。
  · 定位到两个元凶：
    ① `binary_closing(disk(1))`：树干环纹是**斜向细条**，3x3 闭运算斜向粘连
       → 梯子变粗块（这也是笔宽 4.47→6.00 的主因）。
    ② 局部密度匹配在**环纹区**也抬阈值：环纹本来就密，抬阈值 = 细条变粗。

【修法】按块内**原图笔画厚度**门控阈值上限：
    th_b = 2 * median(EDT(原图墨))[block∩ink]
    · th_b 小（≈2px，树干环纹 = 线稿）→ cap 压到 ≈1.05×全局 → 保留细条；
    · th_b 大（≥6px，叶片 = 实心面）→ cap 放到 ≈1.85×全局 → 细须合并成实心叶片。
    th_b ∈ (2,6) 线性过渡。同时**取消 closing**（改由阈值本身完成连接）。
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

OUT = ROOT / "jobs" / "v350d_p4gate"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
REB = ROOT / "jobs" / "router_out_v329" / "_p4_rebirth_raw.jpg"
WARP = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
INK_RGB = np.array([12.0, 10.0, 9.0], np.float32)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


VARIANTS = [
    # name, kind, B, lo, hi(或 (hi_thin, hi_thick)), th_range, ps
    ("M0_global_noclose", "glob", 48, 0.0, 0.0, None, 0.0),
    ("M1_loc48_070_160", "loc", 48, 0.70, 1.60, None, 0.0),
    ("M2_loc48_070_200", "loc", 48, 0.70, 2.00, None, 0.0),
    ("M3_gate48_105_185", "gate", 48, 0.70, (1.05, 1.85), (3.0, 6.0), 0.0),
    ("M5_gate48_105_160", "gate", 48, 0.70, (1.05, 1.60), (3.0, 6.0), 0.0),
    ("M4_gate64_105_185", "gate", 64, 0.70, (1.05, 1.85), (3.0, 6.0), 0.60),
]


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
    eo = ndi.distance_transform_edt(ink)

    frac = float(ink.sum()) / max(1, int(sel.sum()))
    t_glob = float(np.clip(np.percentile(lg[sel], 100.0 * frac), 20.0, 200.0))
    print(f"[v350d] {W}x{H} 原墨={100*ink.mean():.2f}% t_glob={t_glob:.1f}")

    # 原图连通块基准
    _, o_n = ndi.label(ink, structure=np.ones((3, 3), bool))
    o_w = 2.0 * float(np.median(eo[ink]))

    results = []
    for name, kind, B, lo, hi, thr, ps in VARIANTS:
        if kind == "glob":
            ni = (lg < t_glob) & sel
            tinfo = f"t={t_glob:.1f}"
        else:
            src = ndi.gaussian_filter(lg, ps) if ps > 0 else lg
            ny, nx = (H + B - 1) // B, (W + B - 1) // B
            Tg = np.zeros((ny, nx), np.float32)
            capp = np.zeros((ny, nx), np.float32)
            for iy in range(ny):
                y0, y1 = iy * B, min(H, (iy + 1) * B)
                for ix in range(nx):
                    x0, x1 = ix * B, min(W, (ix + 1) * B)
                    s = sel[y0:y1, x0:x1]
                    k = ink[y0:y1, x0:x1]
                    ns = int(s.sum())
                    if ns < 250:
                        Tg[iy, ix] = t_glob
                        capp[iy, ix] = t_glob * (hi[1] if kind == "gate" else hi)
                        continue
                    d = float((k & s).sum()) / ns
                    tloc = (float(np.percentile(src[y0:y1, x0:x1][s], 100.0 * (1.0 - d)))
                            if d >= 0.004 else 0.0)
                    if kind == "gate":
                        t0, t1 = float(thr[0]), float(thr[1])
                        ki = k
                        ei = eo[y0:y1, x0:x1]
                        th = 2.0 * float(np.median(ei[ki])) if ki.any() else t0
                        u = float(np.clip((th - t0) / max(1e-6, t1 - t0), 0.0, 1.0))
                        cap = t_glob * (hi[0] + (hi[1] - hi[0]) * u)
                    else:
                        cap = t_glob * hi
                    Tg[iy, ix] = tloc
                    capp[iy, ix] = cap
            w = (Tg > 0).astype(np.float32)
            z = lambda a: ndi.gaussian_filter(ndi.zoom(a, B, order=1)[:H, :W], B / 2.5)
            den = z(w)
            Tf = np.where(den > 0.15, z(Tg * w) / np.maximum(den, 1e-6), 0.0)
            capf = z(capp * w) / np.maximum(den, 1e-6)
            Tf = np.where(Tf > 0, np.clip(Tf, t_glob * lo, np.maximum(capf, t_glob * lo)),
                          -1.0)
            ni = (src < Tf) & sel
            _hmax = hi[1] if isinstance(hi, tuple) else hi
            tinfo = f"t=[{t_glob*lo:.0f},{t_glob*_hmax:.0f}]"

        lab, n = ndi.label(ni, structure=np.ones((3, 3), bool))
        if n:
            sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sz >= 40
            ni = keep[lab]
        a3 = ni.astype(np.float32)[..., None]
        res = Image.fromarray(np.clip(_base * (1.0 - a3) + INK_RGB[None, None, :] * a3,
                                      0, 255).astype(np.uint8), "RGB")
        res.save(str(OUT / f"{name}.jpg"), quality=95)
        results.append((name, ni, tinfo))

    lines = [f"原图                墨={100*ink.mean():6.2f}%  笔宽median={o_w:5.2f}px  "
             f"连通块={o_n:5d}"]
    for name, mask, tinfo in results:
        e = ndi.distance_transform_edt(mask)
        wd = 2.0 * float(np.median(e[mask])) if mask.any() else 0.0
        _, nn = ndi.label(mask, structure=np.ones((3, 3), bool))
        lines.append(f"{name:20s} 墨={100*mask.mean():6.2f}%  笔宽={wd:5.2f}px  "
                     f"连通块={nn:5d}   {tinfo}")
    txt = "\n".join(lines)
    (OUT / "QC_v350d.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
