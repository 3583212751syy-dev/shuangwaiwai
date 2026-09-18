# -*- coding: utf-8 -*-
"""p4 碎片溯源诊断：把 NEW 里"与原图逐像素相同"的墨（= 留在原位的残件）
与"底板自带墨"分离出来，定位用户红框那 4 处碎片到底来自哪一环。
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp  # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
OUTP = ROOT / "jobs" / "router_out_v329" / "Pinterest (4)_variant.jpg"
BASE = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
ALPHA = ROOT / "jobs" / "v346_p4aff" / "_swap_alpha.png"
CMP = ROOT / "jobs" / "_cmpcrop"
CMP.mkdir(parents=True, exist_ok=True)
LUM = np.array([0.299, 0.587, 0.114], np.float32)


def ink_of(a, lum_thr=55.0, sat_thr=14.0):
    lum = a @ LUM
    sat = a.max(2) - a.min(2)
    return (lum < lum_thr) & (sat < sat_thr)


o = Image.open(SRC).convert("RGB")
W, H = o.size
ao = np.asarray(o, np.float32)
ink0 = cpp.tree_ink_mask(o, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
print(f"图像 {W}x{H}   ORIG ink={100*ink0.mean():.2f}%")

# ── ① 底板有没有自带残墨？────────────────────────────────────────────────
if BASE.exists():
    b = np.asarray(Image.open(BASE).convert("RGB").resize((W, H), Image.LANCZOS), np.float32)
    bi = ink_of(b)
    lab, n = ndi.label(bi, np.ones((3, 3), bool))
    sz = np.bincount(lab.ravel()); sz[0] = 0
    big = np.where(sz >= 40)[0]
    print(f"[底板 {BASE.name}] 墨={100*bi.mean():.3f}%  连通块={n}  >=40px 的块={len(big)}  "
          f"这些块合计={100*(np.isin(lab, big)).mean():.3f}%")
    if len(big):
        for i in big[:12]:
            ys, xs = np.where(lab == i)
            print(f"    底板墨块 #{i}: {sz[i]}px  bbox x[{xs.min()}-{xs.max()}] "
                  f"y[{ys.min()}-{ys.max()}]  亮度均值={b[lab==i]@LUM.mean():.0f}")
else:
    print("[底板] _p4_warped.jpg 不存在")

# ── ② NEW 里有多少"与原图逐像素相同"的墨（= 原位残件）────────────────────
new = np.asarray(Image.open(OUTP).convert("RGB").resize((W, H), Image.LANCZOS), np.float32)
d = np.abs(new - ao).max(2)
same = d <= 16
ln = new @ LUM
resi = ink0 & (ln < 60) & same
lab2, n2 = ndi.label(resi, np.ones((3, 3), bool))
sz2 = np.bincount(lab2.ravel()); sz2[0] = 0
keep = np.where(sz2 >= 30)[0]
print(f"\n[NEW 原位残件] 残余墨 {100*resi.mean():.3f}%（{int(resi.sum())}px）")
print(f"  >=30px 的残件块 {len(keep)} 个，合计 {100*np.isin(lab2,keep).mean():.3f}%")
keep = keep[np.argsort(-sz2[keep])]
for i in keep[:14]:
    ys, xs = np.where(lab2 == i)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    print(f"    残件 #{i}: {sz2[i]:5d}px  bbox {w}x{h} @ x[{xs.min()}-{xs.max()}] "
          f"y[{ys.min()}-{ys.max()}]  中心=({xs.mean():.0f},{ys.mean():.0f})")

# ── ③ α 层在残件处是多少？─────────────────────────────────────────────────
if ALPHA.exists():
    a = np.asarray(Image.open(ALPHA).convert("L").resize((W, H), Image.LANCZOS), np.float32) / 255.0
    print(f"\n[α层] α 覆盖(>=0.5)={100*(a>=0.5).mean():.2f}%  "
          f"残件处 α 均值={a[resi].mean() if resi.any() else -1:.3f}")

# ── ④ 可视化：ORIG 灰底 + 残件染红 ────────────────────────────────────────
g = (ao @ LUM)
vis = np.stack([g, g, g], -1).astype(np.uint8)
vis[resi] = np.array([255, 40, 40], np.uint8)
Image.fromarray(vis, "RGB").save(str(CMP / "p4_residual_map.jpg"), quality=94)

# ── ⑤ 红框区域 1:1 ORIG | NEW ─────────────────────────────────────────────
BOXES = [(0.34, 0.10, 0.52, 0.28), (0.06, 0.40, 0.44, 0.72)]
for k, (x0, y0, x1, y1) in enumerate(BOXES):
    bx = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
    oc = o.crop(bx); nc = Image.open(OUTP).convert("RGB").resize((W, H), Image.LANCZOS).crop(bx)
    sc = 2
    oc = oc.resize((oc.width * sc, oc.height * sc), Image.LANCZOS)
    nc = nc.resize((nc.width * sc, nc.height * sc), Image.LANCZOS)
    gap = np.full((oc.height, 12, 3), 255, np.uint8)
    out = np.concatenate([np.asarray(oc, np.uint8), gap, np.asarray(nc, np.uint8)], 1)
    Image.fromarray(out, "RGB").save(str(CMP / f"p4_frag{k}.jpg"), quality=95)
    print(f"saved p4_frag{k}.jpg  {out.shape}  box={bx}")
