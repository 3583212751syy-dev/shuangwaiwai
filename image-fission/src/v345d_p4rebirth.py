"""v345: p4 棕榈树「蝙蝠式」逐棵重生（定稿版）。

用户第 14 轮：
  「裂变没看到明显变化，并且比原图丑，这种手法严格禁止」
  「蝙蝠设计可以，树木的处理按照蝙蝠的裂变去试试」

**真正根因（本轮查实，之前 3 轮都没抓到）**：
  旧掩膜 = 原墨迹 dilate(1) → 新墨迹被 `lg<t & mask` 死死限制在原剪影**内部**
  → **剪影在数学上不可能变大或换形**。所以无论换多少 prompt / denoise / LoRA，
  结果都只能"轻微侵蚀原剪影"，用户读成"没看到明显变化"，而侵蚀产生的毛刺又
  读成"比原图丑"。这就是为什么换了 6 组参数（v345/v345b 两轮扫参）全部失败。

**解法（= 蝙蝠的做法，蝙蝠的掩膜正是"实心块"）**：
  ① 用树干 `closing(vline25)+opening(vline60)` 定位每一棵树；
  ② 按"最近树干"把墨迹 Voronoi 切成 34 份（互不交叉）；
  ③ 每棵树取**实心包络**（本树墨迹 dilate9 → closing14 → fill_holes）
     → 模型有整块空白可以重新设计剪影；
  ④ 每棵树的包络单独裁块（~200x600）→ **上采样到 1024 长边**再重生 → 缩回
     → SDXL 在 1024 甜点尺作画 = 硬边清晰（蝙蝠也是这个尺度）；
  ⑤ 密度匹配阈值（新墨量 == 原墨量）+ 阈值硬上限 52（迷彩深棕 lum≈60~90，
     不压住就会把迷彩暗块吃成纯黑）+ 去碎点 → 纯黑硬边剪影。

输出 = 新墨迹掩膜（H×W bool），交给 camo_palm_pattern.fission 的
`tree_ink_override` 消费：out = 重 blob 迷彩底 ⊕ 纯黑新剪影（文字/版式不动）。
"""
import argparse
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
from v345_p4tree import inpaint_crop, PALM_POS, PALM_NEG   # noqa: E402
from v345c_p4seg import trunks, partition, _disk, keep_big  # noqa: E402

OUT = ROOT / "jobs" / "v345_p4tree"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
BLACK = np.array([12.0, 10.0, 9.0], np.float32)


def envelope(own, dil=9, close=14, min_px=900):
    e = ndi.binary_closing(ndi.binary_dilation(own, structure=_disk(dil)),
                           structure=_disk(close))
    e = ndi.binary_fill_holes(e)
    e, _, _ = keep_big(e, min_px)
    return e


def build(margin=40, target=1024, denoise=0.88, cn_strength=0.35, cn_end=0.45,
          seed=7, limit=None, tcap=52.0, gain=1.0, edil=9, eclose=14, min_own=400,
          ckpt="ProteusV0.4.safetensors", pos=None, neg=None, out_tag="", smooth=0.0):
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    o = np.asarray(img, np.float32)
    ink0 = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    _, segs = trunks(ink0, H, W, 25, 60)
    segs = [s for s in segs if 0.06 * H < s["h"] < 0.62 * H]
    win = partition(ink0, segs, H, W)
    trees = []
    for i, s in enumerate(segs):
        own = (win == i) & ink0
        if own.sum() < min_own:
            continue
        e = envelope(own, edil, eclose)
        if e.sum() < 800:
            continue
        trees.append(dict(own=own, env=e, x=s["x"]))
    print(f"[p4tree] 树干={len(segs)} 可用树={len(trees)} 图={W}x{H} 原墨={100*ink0.mean():.1f}%")

    # 重叠像素判给最近树干
    cost = np.full((len(trees), H, W), 1e18, np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i, t in enumerate(trees):
        cost[i] = (xx - t["x"]) ** 2 + (yy - t["x"]) ** 2 * 0
    stack = np.stack([t["env"] for t in trees])
    cost = np.where(stack, cost, 1e18)
    winE = cost.argmin(0)
    for i, t in enumerate(trees):
        t["env"] = t["env"] & (winE == i)
    env_all = np.zeros((H, W), bool)
    for t in trees:
        env_all |= t["env"]
    print(f"[p4tree] 包络合计={100*env_all.mean():.1f}%")

    if limit:
        trees = trees[:limit]

    gen = o.copy()
    t0 = time.time()
    ok = 0
    for i, t in enumerate(trees):
        e = t["env"]
        ys, xs = np.where(e)
        x0 = max(0, xs.min() - margin); x1 = min(W, xs.max() + 1 + margin)
        y0 = max(0, ys.min() - margin); y1 = min(H, ys.max() + 1 + margin)
        ec = e[y0:y1, x0:x1]
        crop = img.crop((x0, y0, x1, y1))
        try:
            r = inpaint_crop(crop, ec, pos or PALM_POS, neg or PALM_NEG, target=target,
                             denoise=denoise, cn_strength=cn_strength, cn_end=cn_end,
                             seed=seed + i * 13, tag=f"t{i}", ckpt=ckpt)
        except Exception as ex:
            print(f"   [{i}] FAIL {ex}")
            continue
        if r is None:
            continue
        ra = np.asarray(r, np.float32)
        ee = ec[..., None]
        gen[y0:y1, x0:x1] = gen[y0:y1, x0:x1] * (1 - ee) + ra * ee
        ok += 1
        if ok % 6 == 0:
            print(f"   [p4tree] {ok}/{len(trees)} {time.time()-t0:.0f}s")
    print(f"[p4tree] 重生 {ok} 棵，用时 {time.time()-t0:.0f}s")
    Image.fromarray(np.clip(gen, 0, 255).astype(np.uint8), "RGB").save(
        str(OUT / f"_gen{out_tag}.jpg"), quality=93)

    lg = R._lum(gen)
    sel = ndi.binary_erosion(env_all, structure=_disk(6))
    n_in = float((ink0 & sel).sum())
    frac = min(0.9, max(0.02, n_in / float(max(1, sel.sum())))) * gain
    t = float(np.percentile(lg[sel], 100.0 * min(0.9, frac))) if sel.any() else 55.0
    t = float(min(max(t, 12.0), tcap))
    new_ink = (lg < t) & sel
    new_ink, _, _ = keep_big(ndi.binary_closing(new_ink, structure=_disk(2)), 70)
    if smooth and smooth > 0:                       # 边界曲率流：把 SDXL 的锯齿磨成矢量硬边
        for _ in range(2):
            new_ink = ndi.gaussian_filter(new_ink.astype(np.float32), smooth) > 0.5
    new_ink = ndi.binary_opening(new_ink, structure=_disk(1))
    ink_out = (ink0 & ~env_all) | new_ink
    print(f"[p4tree] thr={t:.1f} 新墨={100*ink_out.mean():.1f}% (原 {100*ink0.mean():.1f}%)")

    idx = ndi.distance_transform_edt(ink0, return_distances=False, return_indices=True)
    base = o[idx[0], idx[1]]
    a = ink_out.astype(np.float32)[..., None]
    prev = base * (1 - a) + BLACK[None, None, :] * a
    pv = Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8), "RGB")
    pv.save(str(OUT / f"_preview{out_tag}.jpg"), quality=94)
    Image.fromarray((ink_out.astype(np.uint8) * 255), "L").save(str(OUT / f"_ink{out_tag}.png"))
    cb = Image.new("RGB", (W * 2 + 12, H), (20, 20, 20))
    cb.paste(img, (0, 0))
    cb.paste(pv, (W + 12, 0))
    cb.resize(((W * 2 + 12) // 2, H // 2), Image.LANCZOS).save(
        str(OUT / f"_cmp{out_tag}.jpg"), quality=92)
    print("[p4tree] ->", OUT / "_ink.png")
    return OUT / "_ink.png"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target", type=int, default=1024)
    ap.add_argument("--denoise", type=float, default=0.88)
    ap.add_argument("--cn", type=float, default=0.35)
    ap.add_argument("--cne", type=float, default=0.45)
    ap.add_argument("--tcap", type=float, default=52.0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--ckpt", type=str, default="ProteusV0.4.safetensors")
    ap.add_argument("--out-tag", type=str, default="")
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--edil", type=int, default=9)
    ap.add_argument("--eclose", type=int, default=14)
    a = ap.parse_args()
    build(limit=a.limit, target=a.target, denoise=a.denoise, cn_strength=a.cn,
          cn_end=a.cne, tcap=a.tcap, gain=a.gain, edil=a.edil, eclose=a.eclose,
          ckpt=a.ckpt, out_tag=a.out_tag, smooth=a.smooth)
