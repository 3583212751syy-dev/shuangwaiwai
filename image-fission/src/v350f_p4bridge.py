"""v350f: p4 「原图厚度门控闭运算」——把 SDXL 的细须叶合并成实心叶片，同时保住树干环纹。

【诊断链（v350 → v350d → 本脚本）】
  · 旧交付 = 全局阈值，笔宽 4.47px（=原图 ✓）但连通块 197（原图仅 91）→ 叶片被
    SDXL 画成**一排细须**（原图是**实心带齿叶片**）→ 肉眼"发毛"。
  · v350 局部密度阈值 + 全图 closing(disk1)：连通块 → 153（叶片合并了 ✓），
    但笔宽 → 6.00px ✗ —— 树干环纹是**斜向细条**，全图闭运算把相邻细条斜向粘连
    → 环纹被压成"粗梯子"（比原来更糊）。
  · v350d 去掉 closing 后笔宽回到 4.47px，但叶片须没合并（连通块 249）。
  ⇒ 结论：**闭运算是需要的，但必须按区域门控** —— 只在"原图本身就是实心面"的地方
    做闭运算（叶片），"原图本身是细线稿"的地方（树干环纹/细枝）绝不动。

【判据】原图墨迹厚度场 th0 = 2·EDT(ink0)（平滑后）：
    · th0 ≥ g_thr（≈4px）= 实心面 → 允许 closing
    · th0 <  g_thr           = 线稿   → 保持全局阈值原样
  门控掩膜做 disk(gd) 膨胀，避免分区接缝处出现"半边合并"的硬边。
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

OUT = ROOT / "jobs" / "v350f_p4bridge"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
REB = ROOT / "jobs" / "router_out_v329" / "_p4_rebirth_raw.jpg"
WARP = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
INK_RGB = np.array([12.0, 10.0, 9.0], np.float32)

# name, close_r, gate_thr, gate_dilate, presmooth
VARIANTS = [
    ("B0_none(旧交付)", 0, 0.0, 0, 0.0),
    ("B1_r1_g40_d8", 1, 4.0, 8, 0.0),
    ("B2_r2_g40_d8", 2, 4.0, 8, 0.0),
    ("B3_r2_g30_d8", 2, 3.0, 8, 0.0),
    ("B4_r3_g45_d10", 3, 4.5, 10, 0.0),
    ("B5_r2_g50_d6", 2, 5.0, 6, 0.6),
]


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


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

    eo = ndi.distance_transform_edt(ink)
    th0 = ndi.gaussian_filter(2.0 * eo, 2.0)
    # 分区（用于 QC）：细线稿区 / 实心面区
    r_thin = ndi.binary_dilation(th0 <= 3.0, structure=disk(5))
    r_thick = ndi.binary_dilation(th0 >= 5.0, structure=disk(5))

    lines = []
    for name, cr, gthr, gd, ps in VARIANTS:
        src = ndi.gaussian_filter(lg, ps) if ps > 0 else lg
        ni = (src < t_glob) & sel
        if cr > 0:
            cl = ndi.binary_closing(ni, structure=disk(cr))
            gate = ndi.binary_dilation(th0 >= gthr, structure=disk(gd))
            ni = np.where(gate, cl, ni) & ndi.binary_dilation(sel, structure=disk(cr))
        lab, n = ndi.label(ni, structure=np.ones((3, 3), bool))
        if n:
            sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sz >= 40
            ni = keep[lab]
        a3 = ni.astype(np.float32)[..., None]
        Image.fromarray(np.clip(_base * (1.0 - a3) + INK_RGB[None, None, :] * a3,
                                0, 255).astype(np.uint8), "RGB").save(
            str(OUT / f"{name}.jpg"), quality=95)
        e = ndi.distance_transform_edt(ni)
        wd = 2.0 * float(np.median(e[ni])) if ni.any() else 0.0
        # 分区域厚度：新墨落在该区内的中位厚度（原图同区作为目标）
        def regw(mask, reg):
            m = mask & reg
            if not m.any():
                return 0.0
            ee = ndi.distance_transform_edt(mask)
            return 2.0 * float(np.median(ee[m]))
        _, nn = ndi.label(ni, structure=np.ones((3, 3), bool))
        lines.append(
            f"{name:16s} 墨={100*ni.mean():6.2f}% 笔宽={wd:4.2f} 块数={nn:4d} | "
            f"细线区厚={regw(ni, r_thin):4.2f}(原{regw(ink, r_thin):4.2f}) "
            f"实心区厚={regw(ni, r_thick):4.2f}(原{regw(ink, r_thick):4.2f})")

    _, o_n = ndi.label(ink, structure=np.ones((3, 3), bool))
    hdr = (f"[v350f] {W}x{H} t_glob={t_glob:.1f} | 原图 墨={100*ink.mean():.2f}% "
           f"笔宽={2*np.median(eo[ink]):.2f} 块数={o_n}\n")
    txt = hdr + "\n".join(lines)
    (OUT / "QC_v350f.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
