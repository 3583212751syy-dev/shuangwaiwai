"""v345: p4 棕榈树「蝙蝠式」重生扫参。

用户第 14 轮原话：
  「裂变没看到明显变化，并且比原图丑，这种手法严格禁止」
  「蝙蝠设计可以，树木的处理按照蝙蝠的裂变去试试」

诊断（本文件落地前已查实）：
  ① 模糊根因 A：p4 尺寸 1242x1754 > max_side 1536 → 降采样 0.876 重绘再升回 → 线稿发虚。
  ② 模糊根因 B：SDXL(ProteusV0.4) 对「细线稿棕榈」的固有画法是**软气刷/水彩**，
     不是硬边墨迹（见 jobs/v342_rebirth/_p4_p4A_dm_noLora.jpg）。二值化后软边变
     「毛球/绒毛」→ 用户读作"比原图丑"。
  ③ 变化小的根因：cn_strength 0.35 太弱又 max_side 降采样 → 剪影只是轻微摆动。

本脚本对 (prompt 语言 / denoise / ControlNet 强度+收束 / LoRA / 出图分辨率) 扫参，
并对每个结果跑三种**墨迹正则化**后处理，输出对照表供肉眼终验。
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import v342_rebirth as R
from styles import camo_palm_pattern as cpp

OUT = ROOT / "jobs" / "v345_p4"
OUT.mkdir(parents=True, exist_ok=True)
CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
TILE = "controlnet-tile-sdxl-1.0.safetensors"

# ---------------------------------------------------------------- prompt 变体
P_OLD = R.PROMPTS["p4"][0]
N_OLD = R.PROMPTS["p4"][1]

# 「石印/镂空/矢量」语言：把 SDXL 从"气刷水彩"推向"硬边墨迹"
P_ST = ("seamless camouflage textile pattern with palm tree silhouettes, "
        "solid black ink palm trees, bold clean stencil shapes, crisp hard edges, "
        "high contrast two-tone, flat matte posterized, minimal interior detail, "
        "screen print, woodcut look, thin elegant curving trunks, wide open fronds "
        "with clear empty gaps between them, black on camouflage green brown tan, "
        "no text, no letters")
N_ST = ("text, letters, words, color photo, realistic, 3d render, blurry, soft, "
        "airbrush, watercolor, gradient, shading, gray, smudge, blotch, hairy, "
        "fuzzy, feathery, painterly, noise, grain, chunky, cluttered, dense")
# 更强：直接点名 "fan of separate leaflets"，逼模型留白
P_ST2 = ("camouflage pattern of black palm tree silhouettes, flat vector clipart, "
         "each frond a separate solid black shape with white gaps between them, "
         "stencil cut-out, hard jagged edges, pure black and camouflage only, "
         "two-tone screen print, no shading, no gradient, minimal detail, "
         "no text, no letters")
N_ST2 = N_ST + ", soft edges, glow, bloom, spray, rough brush, textured, grainy"


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _ink_stats(ink):
    sk = skeletonize(ink)
    if sk.sum() < 20:
        return 0.0, sk
    d = ndi.distance_transform_edt(ink)
    return float(np.median(d[sk])), sk


def _despeck(ink, min_px):
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    if n == 0:
        return ink
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sz >= min_px
    return keep[lab]


def _compose(o, ink, mask, ink0, black=(12.0, 10.0, 9.0)):
    """掩膜外原图；掩膜内 = 干净迷彩底 ⊕ 新墨迹纯黑。"""
    if isinstance(o, Image.Image):
        o = np.asarray(o, np.float32)
    idx = ndi.distance_transform_edt(ink0, return_distances=False, return_indices=True)
    base = o[idx[0], idx[1]]
    a = ink.astype(np.float32)[..., None] * mask[..., None]
    blk = np.array(black, np.float32)[None, None, :]
    inside = base * (1.0 - a) + blk * a
    mm = mask[..., None]
    return Image.fromarray(np.clip(o * (1 - mm) + inside * mm, 0, 255).astype(np.uint8), "RGB")


def _thr_density(lg, mask, ink0, lo=20.0, hi=200.0):
    ns = int(mask.sum())
    frac = min(0.95, max(0.02, float(ink0.sum()) / float(ns))) if ns else 0.25
    return float(np.clip(np.percentile(lg[mask], 100.0 * frac), lo, hi))


def snaps(gen, mask, ink0, o):
    """三种墨迹正则化。返回 dict[name] = PIL。"""
    lg = R._lum(np.asarray(gen, np.float32))
    t = _thr_density(lg, mask, ink0)
    raw = (lg < t) & mask
    res = {}

    # --- dm：现状（密度匹配 + 轻闭开）---
    ink = _despeck(ndi.binary_opening(ndi.binary_closing(raw, _disk(2)), _disk(1)), 60)
    res["dm"] = _compose(o, ink, mask, ink0)

    # --- op：开运算去绒毛（毛球 killer）---
    ink = ndi.binary_closing(raw, _disk(2))
    ink = ndi.binary_opening(ink, _disk(4))
    ink = ndi.binary_closing(ink, _disk(3))
    ink = _despeck(ink, 90)
    res["op"] = _compose(o, ink, mask, ink0)

    # --- rg：骨架 + 统一线宽（原图线宽定标）---
    ink = ndi.binary_closing(raw, _disk(2))
    ink = _despeck(ink, 70)
    w0, _ = _ink_stats(ink0)                    # 原图半宽
    r = int(max(1, min(6, round(w0))))
    sk = skeletonize(ink)
    sk = _despeck(sk, 18)                       # 剪掉短毛刺
    ink = ndi.binary_dilation(sk, _disk(r))
    ink &= ndi.binary_dilation(mask, _disk(2))
    res["rg"] = _compose(o, ink, mask, ink0)

    frac = 100.0 * raw.mean()
    return res, t, frac, w0


def load_p4():
    src = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
    img = Image.open(src).convert("RGB")
    mask = R.mask_p4(img)
    ink0 = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    return img, mask, ink0


VARIANTS = [
    ("A_old_dn85_cn35e45", dict(prompt=P_OLD, neg=N_OLD, denoise=0.85,
                                cn_name=CANNY, cn_strength=0.35, cn_end=0.45, max_side=1536)),
    ("B_st_dn85_cn35e45", dict(prompt=P_ST, neg=N_ST, denoise=0.85,
                               cn_name=CANNY, cn_strength=0.35, cn_end=0.45, max_side=1536)),
    ("C_st_dn65_cn60e65", dict(prompt=P_ST, neg=N_ST, denoise=0.65,
                               cn_name=CANNY, cn_strength=0.60, cn_end=0.65, max_side=1536)),
    ("D_st_vec_dn85", dict(prompt=P_ST, neg=N_ST, denoise=0.85, cn_name=CANNY,
                           cn_strength=0.35, cn_end=0.45, max_side=1536,
                           lora=[("DD-vector-v2.safetensors", 0.85)])),
    ("E_st2_dn88_cn40e50", dict(prompt=P_ST2, neg=N_ST2, denoise=0.88,
                                cn_name=CANNY, cn_strength=0.40, cn_end=0.50, max_side=1536)),
    ("F_st_dn88_cn35e45_2048", dict(prompt=P_ST, neg=N_ST, denoise=0.88,
                                    cn_name=CANNY, cn_strength=0.35, cn_end=0.45, max_side=2048)),
    ("G_tile_dn70", dict(prompt=P_ST, neg=N_ST, denoise=0.70, cn_name=TILE,
                         cn_strength=0.70, cn_end=0.80, cn_pre="tile", max_side=1536)),
]


def main():
    only = sys.argv[1:] or None
    img, mask, ink0 = load_p4()
    w0, sk0 = _ink_stats(ink0)
    print(f"[p4] size={img.size} mask={100*mask.mean():.1f}% ink0={100*ink0.mean():.1f}% "
          f"原图线宽半宽={w0:.2f}px")
    rows = []
    for name, kw in VARIANTS:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            gen = R.rebirth_subject(img, mask, kw.pop("prompt"), kw.pop("neg"),
                                    tag=name, seed=7, **kw)
        except Exception as e:
            print(f"[{name}] FAIL {e}")
            continue
        gen.save(str(OUT / f"{name}_raw.jpg"), quality=93)
        res, t, frac, _ = snaps(gen, mask, ink0, img)
        for k, v in res.items():
            v.save(str(OUT / f"{name}_{k}.jpg"), quality=93)
        print(f"[{name}] {time.time()-t0:.0f}s thr={t:.0f} raw_ink={frac:.1f}%")
        rows.append((name, [gen] + [res[k] for k in ("dm", "op", "rg")]))
    # 对照表
    if rows:
        cell = (300, 424)
        labels = ["RAW(SDXL)", "dm", "op", "rg"]
        sheet = Image.new("RGB", (cell[0] * 4 + 30, cell[1] * len(rows) + 10), (30, 30, 30))
        for i, (nm, imgs) in enumerate(rows):
            for j, im in enumerate(imgs):
                sheet.paste(im.resize(cell, Image.LANCZOS), (j * cell[0] + 6, i * cell[1] + 5))
        sheet.save(str(OUT / "_sheet.jpg"), quality=90)
        print("[sheet]", OUT / "_sheet.jpg")
        print("labels:", labels)


if __name__ == "__main__":
    main()
