# -*- coding: utf-8 -*-
"""v395_p4trees — pinterest4「整树对象」裂变（第 21 轮，用户否决 v394）。

用户原话：「为什么更碎了…树木剪影裂变会不会，结构合理连贯不要碎。」

v394 更碎的两个真凶（写死在这里防回退）：
  ① v346 链按**单笔画**做原子归属/搬移 —— 原稿里树干和树冠本来就是两坨笔画，
     靠细连接线成一体；单笔画搬移天然把树拆散。
  ② v394 的 drop_orphans(<1500px 且离大件>30px 剔除) 把**树干-树冠的连接笔画**
     当"悬空碎块"删了 —— 树就只剩漂浮叶丛 + 断成虚线的干。

v395 根治：**整树对象**裂变，永不碎——
  ① ink 膨胀 group_r px → label → 同组**全部原笔画**划为一棵树（干冠连接线
     天然同组）。不 closing、不填孔 —— 纹理=原笔画，风格零损伤。
  ② 每棵作为**整体**刚体变换：旋转(±14°，贴边树±4°)/缩放 0.90~1.12/镜像 50%/
     位移抖动 ±30px —— 结构天然连贯，物理上不可能碎。
  ③ 迷彩底沿用 v394 的 label-map warp 湖泊形裂变（用户认可的部分）。

用法：python src/v395_p4trees.py [--seed 5] [--camo-seed 11]
产出 jobs/v395_p4trees/_swap.jpg + _camo_new.jpg
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

from v346_p4aff import _disk, SRC as SRC4                   # noqa: E402
from v394_p4dual import fission_camo, WRAP                  # noqa: E402
from styles import camo_palm_pattern as cpp                 # noqa: E402

OUT = ROOT / "jobs" / "v395_p4trees"


def tree_objects(ink, color, min_px=400, group_r=7):
    """墨迹 → 整树对象 [(alpha裁剪, color裁剪, bbox, touch_border)]。
    ink 膨胀 group_r → label → 同组全部原笔画=一棵树（干冠连接线天然同组）；
    <min_px 的组=孤立草屑，整组剔除；大树内部一根笔画都不动。"""
    dil = ndi.binary_dilation(ink, structure=_disk(group_r))
    lab, n = ndi.label(dil, structure=np.ones((3, 3), bool))
    if not n:
        return []
    sz_ink = np.bincount(lab[ink].ravel(), minlength=n + 1)
    ys_all, xs_all = np.nonzero(ink)
    ids_all = lab[ys_all, xs_all]
    H, W = ink.shape
    objs = []
    for j in range(1, n + 1):
        if sz_ink[j] < min_px:
            continue
        sel = ids_all == j
        ys, xs = ys_all[sel], xs_all[sel]
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        pa = ink[y0:y1, x0:x1].astype(np.float32)
        pc = color[y0:y1, x0:x1].copy()
        objs.append((pa, pc, (y0, y1, x0, x1),
                     y0 <= 2 or x0 <= 2 or y1 >= H - 2 or x1 >= W - 2))
    return objs


def transform_obj(alpha, col, rng, touch_border):
    """整棵树单一刚体变换：旋转+缩放+镜像（坐标重采样一次完成，结构不变）。"""
    ang = rng.uniform(-4, 4) if touch_border else rng.uniform(-14, 14)
    sx = rng.uniform(0.90, 1.12)
    flip = rng.random() < 0.5
    a = ndi.rotate(alpha, ang, reshape=True, order=1, mode="constant", cval=0.0)
    c = ndi.rotate(col, ang, reshape=True, order=1, mode="nearest", cval=0.0)
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    ys = np.clip((cy + (yy - cy) / sx).astype(int), 0, H - 1)
    xs = np.clip((cx + (xx - cx) / sx).astype(int), 0, W - 1)
    a, c = a[ys, xs], c[ys, xs]
    if flip:
        a, c = a[:, ::-1], c[:, ::-1]
    return np.clip(a, 0, 1), c


def build(seed=5, camo_seed=11, group_r=7, min_px=400, jitter=30, outdir=None):
    outdir = Path(outdir) if outdir else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    img0 = Image.open(SRC4).convert("RGB")
    W, H = img0.size
    ink = cpp.tree_ink_mask(img0, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    color = np.asarray(img0, np.float32)
    objs = tree_objects(ink, color, min_px=min_px, group_r=group_r)
    print(f"[v395] 整树对象={len(objs)}  原墨={100 * ink.mean():.1f}%")

    # 迷彩底裂变（v394 已验证的部分原样保留）
    if WRAP.exists():
        base0 = Image.open(WRAP).convert("RGB")
    else:
        base0 = cpp.tf.erase(img0, ndi.binary_dilation(ink, structure=_disk(4)),
                             method="nn", nn_median=21, margin=24)
    base0 = base0.resize((W, H), Image.LANCZOS)
    camo_new, _ = fission_camo(base0, seed=camo_seed)
    camo_new.save(str(outdir / "_camo_new.jpg"), quality=96)

    out = np.asarray(camo_new, np.float32).copy()
    cov = np.zeros((H, W), np.float32)            # 贴树 α 覆盖（度量用）
    rng = np.random.default_rng(seed)
    for pa, pc, (y0, y1, x0, x1), tb in objs:
        pa, pc = transform_obj(pa, pc, rng, tb)
        ph, pw = pa.shape
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        cy = int(np.clip(cy + rng.integers(-jitter, jitter + 1), ph // 2, H - ph // 2))
        cx = int(np.clip(cx + rng.integers(-jitter, jitter + 1), pw // 2, W - pw // 2))
        ty, tx = cy - ph // 2, cx - pw // 2
        sy0, sy1 = max(0, ty), min(H, ty + ph)
        sx0, sx1 = max(0, tx), min(W, tx + pw)
        if sy1 <= sy0 or sx1 <= sx0:
            continue
        va = pa[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx]
        vc = pc[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx]
        reg = out[sy0:sy1, sx0:sx1]
        out[sy0:sy1, sx0:sx1] = reg * (1 - va[..., None]) + vc * va[..., None]
        cov[sy0:sy1, sx0:sx1] = np.maximum(cov[sy0:sy1, sx0:sx1], va)

    pv = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    pv.save(str(outdir / "_swap.jpg"), quality=96)
    print(f"[v395] 贴树覆盖={100 * (cov > 0.5).mean():.1f}%（原墨 22.6%）"
          f"  seed={seed} camo_seed={camo_seed} → {outdir / '_swap.jpg'}")
    return outdir / "_swap.jpg"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--camo-seed", type=int, default=11)
    ap.add_argument("--group-r", type=int, default=7)
    ap.add_argument("--min-px", type=int, default=400)
    a = ap.parse_args()
    build(seed=a.seed, camo_seed=a.camo_seed, group_r=a.group_r, min_px=a.min_px)
