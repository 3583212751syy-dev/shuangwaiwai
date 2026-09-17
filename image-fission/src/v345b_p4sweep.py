"""v345b: p4 SDXL 重生扫参第 2 轮 —— 换掉三个关键变量。

第 1 轮（v345_p4sweep）结论：
  · 旧 prompt（"elegant palm trees drawn as fine black line art... delicate fronds"）
    在 SDXL 里被画成**软气刷水彩**（见 _p4_p4A_dm_noLora.jpg）→ 二值化后成"毛球"。
  · 单纯 stencil 语言（P_ST）也只改善一点，仍带绒毛。

第 2 轮换三个变量：
  ① **矢量化 LoRA**（DD-vector-v2 0.9）—— 专门训过的 flat vector 风格，最可能给出
     "硬边纯色剪影"而不是气刷；
  ② **镂空语言 P_ST2**（"each frond a separate solid black shape with white gaps"）；
  ③ **实心掩膜**（dilate20+closing30）—— 让模型看到整块空白可以重新设计整棵树，
     而不是往细缝里塞纹理（细带掩膜 = 气刷绒毛的直接原因）。

阈值一律 **密度匹配 + 硬上限 52**（迷彩深棕 lum≈60~90，阈值必须压在其下，
否则会把迷彩暗块吃成纯黑）。
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import v342_rebirth as R
from styles import camo_palm_pattern as cpp

OUT = ROOT / "jobs" / "v345b_p4"
OUT.mkdir(parents=True, exist_ok=True)
CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
VEC = "DD-vector-v2.safetensors"

P_ST2 = ("camouflage pattern of black palm tree silhouettes, flat vector clipart, "
         "each frond a separate solid black shape with white gaps between them, "
         "stencil cut-out, hard jagged edges, pure black on camouflage, "
         "two-tone screen print, no shading, no gradient, minimal detail, "
         "no text, no letters")
N_ST2 = ("soft, airbrush, watercolor, gradient, shading, grey, gray, blurry, "
         "fuzzy, hairy, feathery, painterly, sketch, noise, grain, smudge, blotch, "
         "text, letters, words, color photo, realistic, 3d render, glow, bloom, "
         "soft edges, dense, cluttered")
P_ST = R.PROMPTS["p4"][0].replace("elegant tropical palm trees drawn as fine black line art",
                                  "black palm tree silhouettes, flat vector stencil")
N_ST = ("soft, airbrush, watercolor, gradient, shading, gray, blurry, fuzzy, hairy, "
        "feathery, painterly, noise, grain, smudge, text, letters, color photo, "
        "realistic, chunky, cluttered")


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def solid_mask(ink, dil=20, close=30):
    m = ndi.binary_dilation(ink, structure=_disk(dil))
    m = ndi.binary_closing(m, structure=_disk(close))
    m = ndi.binary_fill_holes(m)
    return ndi.binary_dilation(m, structure=_disk(2))


def snap(gen, mask, ink0, o, tcap=52.0):
    """密度匹配阈值（新墨量 == 原墨量）→ 去碎点 → 纯黑硬边。"""
    lg = R._lum(np.asarray(gen, np.float32))
    ns = int(mask.sum())
    frac = min(0.9, max(0.02, float(ink0.sum()) / float(ns))) if ns else 0.25
    t = float(min(max(np.percentile(lg[mask], 100.0 * frac), 12.0), tcap))
    ink = (lg < t) & mask
    ink = ndi.binary_closing(ink, structure=_disk(2))
    ink = ndi.binary_opening(ink, structure=_disk(1))
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        k = np.zeros(n + 1, bool)
        k[1:] = sz >= 70
        ink = k[lab]
    idx = ndi.distance_transform_edt(ink0, return_distances=False, return_indices=True)
    base = np.asarray(o if not isinstance(o, Image.Image) else o, np.float32)
    base = base[idx[0], idx[1]]
    a = ink.astype(np.float32)[..., None] * mask[..., None]
    blk = np.array([12.0, 10.0, 9.0], np.float32)[None, None, :]
    out = np.asarray(o, np.float32) * (1 - mask[..., None]) + \
        (base * (1 - a) + blk * a) * mask[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), t


VARIANTS = [
    ("H1_solid_vec_dn90_cn25", dict(mk="solid", pr="st2", dn=0.90, cs=0.25, ce=0.35,
                                    ms=1536, lora=VEC)),
    ("H2_ink_vec_dn90_cn25", dict(mk="ink", pr="st2", dn=0.90, cs=0.25, ce=0.35,
                                  ms=1536, lora=VEC)),
    ("H3_solid_vec_dn65_cn55", dict(mk="solid", pr="st2", dn=0.65, cs=0.55, ce=0.60,
                                    ms=1536, lora=VEC)),
    ("H4_ink_st_vec_dn88_2048", dict(mk="ink", pr="st", dn=0.88, cs=0.30, ce=0.40,
                                     ms=2048, lora=VEC)),
    ("H5_solid_nolora_dn88_2048", dict(mk="solid", pr="st2", dn=0.88, cs=0.30, ce=0.40,
                                       ms=2048, lora=None)),
    ("H6_ink_nolora_dn70_cn60", dict(mk="ink", pr="st2", dn=0.70, cs=0.60, ce=0.65,
                                     ms=1536, lora=None)),
]


def main():
    only = sys.argv[1:] or None
    img = Image.open("E:/Desktop/图裂变测试图/Pinterest (4).jpg").convert("RGB")
    ink0 = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    ink_m = ndi.binary_dilation(ink0, structure=np.ones((3, 3), bool))
    sol_m = solid_mask(ink0)
    print(f"[p4b] ink掩膜={100*ink_m.mean():.1f}% 实心掩膜={100*sol_m.mean():.1f}% "
          f"原墨={100*ink0.mean():.1f}%")
    rows = []
    for name, v in VARIANTS:
        if only and name not in only:
            continue
        mask = sol_m if v["mk"] == "solid" else ink_m
        pr, ng = (P_ST2, N_ST2) if v["pr"] == "st2" else (P_ST, N_ST)
        t0 = time.time()
        try:
            gen = R.rebirth_subject(
                img, mask, pr, ng, tag=name, seed=7, denoise=v["dn"],
                cn_name=CANNY, cn_strength=v["cs"], cn_end=v["ce"],
                max_side=v["ms"], color_match=0.0,
                lora=([(v["lora"], 0.9)] if v["lora"] else None))
        except Exception as e:
            print(f"[{name}] FAIL {e}")
            continue
        gen.save(str(OUT / f"{name}_raw.jpg"), quality=93)
        sn, t = snap(gen, mask, ink0, img, tcap=52.0)
        sn.save(str(OUT / f"{name}_snap.jpg"), quality=93)
        print(f"[{name}] {time.time()-t0:.0f}s thr={t:.1f}")
        rows.append((name, [gen, sn]))
    if rows:
        cell = (400, 565)
        sheet = Image.new("RGB", (cell[0] * 2 + 18, cell[1] * len(rows) + 8), (25, 25, 25))
        for i, (nm, ims) in enumerate(rows):
            for j, im in enumerate(ims):
                sheet.paste(im.resize(cell, Image.LANCZOS), (j * cell[0] + 6, i * cell[1] + 4))
        sheet.save(str(OUT / "_sheet.jpg"), quality=90)
        # 原图也放一张供对照
        img.save(str(OUT / "_orig.jpg"), quality=93)
        print("[sheet]", OUT / "_sheet.jpg")


if __name__ == "__main__":
    main()
