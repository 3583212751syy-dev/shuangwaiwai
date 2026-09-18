# -*- coding: utf-8 -*-
"""v394_p4dual — pinterest4 双层裂变定稿（树剪影 + 迷彩底 一起换）。

用户第 20 轮否决 v393 的两条（原话）：
  ❌「自己看的到碎的吗」——成品里仍有与树干脱节的悬空叶丛碎块。
  ❌「这是张迷彩图，背景湖泊形的色块区域也要裂变」——v346 直接复用管线的
    `_p4_warped.jpg`（= 原迷彩原样抠掉树），底纹从没动过。

本版两层都裂：
  ① 树层：复用 v346 全套（原子归属/相似变换/四方环绕/预乘原色）——经 v387~v392
     五轮打磨的线质保真链一根线都不动；只把变化幅度调大（max_ang 12→20、
     scl_var 0.06→0.08），并新增【大件碎块清理】：合成后连通件 <1500px 且离任何
     ≥1500px 大件 >30px 的独立叶丛 → 判为"与树干脱节的悬空碎块"，直接剔除
     （底是干净迷彩，剔了无痕）。旧阈值(<4px)放走了几十倍大的碎块。
  ② 迷彩底层：`_p4_warped.jpg` 按调色板(k=5)量化成 label map → 平滑随机场
     (sigma≈W/9, 振幅≈7%W) warp label map → 新的湖泊形色块（同色系/同比例，
     边界形状全变）。树贴回时底已是新迷彩 ⇒ 一张图里"树"和"迷彩"同时裂变。

用法：
    python src/v394_p4dual.py [--seed 20260918] [--ang 20] [--scl-var 0.08]
产出 jobs/v394_p4dual/_swap.jpg（树层，底已是新迷彩）+ _camo_new.jpg（新迷彩底预览）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import v346_p4aff as b346                                # noqa: E402  (collect/place/_aff/SRC)
from styles import camo_palm_pattern as cpp              # noqa: E402

OUT = ROOT / "jobs" / "v394_p4dual"
WRAP = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"


# ---------------------------------------------------------------- 迷彩底裂变
def kmeans_palette(base, k=5, n_sample=24000, iters=18, seed=7):
    """从干净迷彩底采色做 k-means → 调色板（含占优的卡其底色）。"""
    rng = np.random.default_rng(seed)
    a = np.asarray(base, np.float32)
    idx = rng.choice(a.shape[0] * a.shape[1], n_sample, replace=False)
    X = a.reshape(-1, 3)[idx]
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    # 按像素占比降序，保证 0 号 = 底色
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
    lab = d.argmin(1)
    order = np.argsort(-np.bincount(lab, minlength=k))
    return C[order]


def camo_labels_safe(base, pal):
    a = np.asarray(base, np.float32)
    H, W = a.shape[:2]
    flat = a.reshape(-1, 3)
    out = np.empty(len(flat), np.uint8)
    CH = 1 << 20
    for i in range(0, len(flat), CH):
        blk = flat[i:i + CH]
        out[i:i + CH] = ((blk[:, None, :] - pal[None, :, :]) ** 2).sum(2).argmin(1)
    return out.reshape(H, W)


def fission_camo(base, seed=11, amp_frac=0.07, sigma_frac=1 / 9.0):
    """label-map 平滑场 warp → 新湖泊形色块。返回 (新底图, label map)。"""
    pal = kmeans_palette(base, k=5, seed=seed)
    lab = camo_labels_safe(base, pal)
    H, W = lab.shape
    rng = np.random.default_rng(seed)
    sx = max(8, int(W * sigma_frac))
    sy = max(8, int(H * sigma_frac))
    amp = amp_frac * min(W, H)
    dx = ndi.gaussian_filter(rng.standard_normal((H, W)), sx).astype(np.float32)
    dy = ndi.gaussian_filter(rng.standard_normal((H, W)), sy).astype(np.float32)
    dx *= amp / max(1e-6, np.abs(dx).max())
    dy *= amp / max(1e-6, np.abs(dy).max())
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lab2 = ndi.map_coordinates(lab.astype(np.float32),
                               [ (yy + dy).ravel(), (xx + dx).ravel() ],
                               order=0, mode="nearest").reshape(H, W).astype(np.uint8)
    # 边界不规则化：对 2 个随机非底色类做随机膨胀/腐蚀(±3px)，湖泊形更碎
    for k in rng.choice(range(1, len(pal)), size=min(2, len(pal) - 1), replace=False):
        m = lab2 == k
        if not m.any():
            continue
        r = int(rng.integers(2, 6))
        op = ndi.binary_dilation if rng.random() < 0.5 else ndi.binary_erosion
        m2 = op(m, structure=b346._disk(r))
        lab2[m2 & (lab2 != 0)] = k          # 只抢别类的地，不动底色
        m3 = (~op(m, structure=b346._disk(r))) & m
        lab2[m3] = int(np.bincount(lab2[~m]).argmax()) if False else lab2[m3]
        lab2[m3] = 0 if 0 == np.bincount(lab.ravel()).argmax() else lab2[m3]
    new = pal[lab2]
    return Image.fromarray(new.astype(np.uint8), "RGB"), lab2


# ---------------------------------------------------------------- 碎块清理
def drop_orphans(alpha, min_big=1500, gap=30.0):
    """连通件 <min_big 且离任何 ≥min_big 大件 >gap px → 悬空碎块，剔除。"""
    m = alpha > 0.5
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if not n:
        return alpha, 0, 0
    sz = np.bincount(lab.ravel())
    big = sz >= min_big
    big[0] = False
    bigm = big[lab]
    if not bigm.any():
        return alpha, 0, n
    d = ndi.distance_transform_edt(~bigm)
    small_ids = np.where((sz > 0) & (sz < min_big))[0]
    drop_ids = []
    for sid in small_ids:
        if d[lab == sid].min() > gap:
            drop_ids.append(sid)
    if drop_ids:
        alpha = alpha.copy()
        alpha[np.isin(lab, drop_ids)] = 0.0
    return alpha, len(drop_ids), n


# ---------------------------------------------------------------- 主流程
def build(seed=20260918, ang=20.0, scl_var=0.08, camo_seed=11,
          min_big=1500.0, gap=30.0, outdir=None):
    outdir = Path(outdir) if outdir else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    # ① 树层：v346 战斗链原样跑（角度/缩放加大），产出 alpha + 预乘原色层
    tmp = outdir / "_tree"
    b346.build(seed=seed, max_ang=ang, scl_var=scl_var, outdir=str(tmp))
    alpha = np.asarray(Image.open(tmp / "_swap_alpha.png"), np.float32) / 255.0
    rgbp = np.asarray(Image.open(tmp / "_swap_rgb.jpg").convert("RGB"), np.float32)

    # ② 碎块清理（用户「自己看的到碎的吗」）：<1500px 且 >30px 离大件 → 剔
    alpha, ndrop, ntot = drop_orphans(alpha, min_big=min_big, gap=gap)
    print(f"[v394] 悬空碎块清理：{ndrop}/{ntot} 块剔除 (<{int(min_big)}px 且离大件>{gap:.0f}px)")
    # 清完再扫一遍 <4px 微粒
    _l, _n = ndi.label(alpha > 0.5, structure=np.ones((3, 3), bool))
    if _n:
        _s = np.bincount(_l.ravel()); _s[0] = 0
        _bad = np.where(_s < 4)[0]
        if len(_bad):
            alpha[np.isin(_l, _bad)] = 0.0

    # ③ 迷彩底裂变（用户「背景湖泊形的色块区域也要裂变」）
    if WRAP.exists():
        base0 = Image.open(WRAP).convert("RGB")
    else:
        img0 = Image.open(b346.SRC).convert("RGB")
        ink0 = cpp.tree_ink_mask(img0, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
        base0 = cpp.tf.erase(img0, ndi.binary_dilation(ink0, structure=b346._disk(4)),
                             method="nn", nn_median=21, margin=24)
    W, H = Image.open(b346.SRC).size
    base0 = base0.resize((W, H), Image.LANCZOS)
    camo_new, _lab = fission_camo(base0, seed=camo_seed)
    camo_new.save(str(outdir / "_camo_new.jpg"), quality=96)

    # ④ 合成：新迷彩底 × (1-α) + 预乘树色（v346 的 rgbp 已预乘 α，直接加）
    nb = np.asarray(camo_new, np.float32)
    aa = alpha[..., None]
    outv = nb * (1 - aa) + rgbp
    pv = Image.fromarray(np.clip(outv, 0, 255).astype(np.uint8), "RGB")
    pv.save(str(outdir / "_swap.jpg"), quality=96)
    Image.fromarray((alpha > 0.5).astype(np.uint8) * 255, "L").save(str(outdir / "_swap_ink.png"))
    ink_pct = 100 * (alpha > 0.5).mean()
    print(f"[v394] 新墨={ink_pct:.1f}%  新迷彩底 seed={camo_seed}  → {outdir / '_swap.jpg'}")
    return outdir / "_swap.jpg"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--ang", type=float, default=20.0)
    ap.add_argument("--scl-var", type=float, default=0.08)
    ap.add_argument("--camo-seed", type=int, default=11)
    ap.add_argument("--min-big", type=float, default=1500.0)
    ap.add_argument("--gap", type=float, default=30.0)
    ap.add_argument("--outdir", type=str, default=None)
    a = ap.parse_args()
    build(seed=a.seed, ang=a.ang, scl_var=a.scl_var, camo_seed=a.camo_seed,
          min_big=a.min_big, gap=a.gap, outdir=a.outdir)
