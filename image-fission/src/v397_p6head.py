# -*- coding: utf-8 -*-
"""v397_p6head —— p6「主体裂变」第①步的**两通道**实现（用户第22轮指令：分步骤）

用户第 22 轮原话：
  「主体生成错乱，要求有逻辑符合正常事物的元素，文本裂变代码要求记得，
    在主体老鹰和骷髅头裂变后进行文本裂变，通过分步骤来做」

诊断（本轮实测，见 jobs/_probe/p6_head_ms.jpg）：
  · 主体错乱发生在**重生本身**，不是文本步骤擦坏的。
  · s888 @2048（原图 3543px 的 0.58×）→ 鹰头退化成白噪点团、黄喙脱离脸部沉成黄块。
  · s888 @2560 → 明显好转（眼睛可见、喙回到脸上），但喙仍偏大偏低 = 解剖仍不合理。
  ⇒ 两个真因：① 重生分辨率低于原图（🔴22）；② 整幅"一把 guid"→ 翼/骷髅要换形，
     鹰头却必须守住解剖，二者对 canny 强度的要求相反。

本脚本 = **通道分离**（治 ②）+ 鹰头原生分辨率重画（治 ①）：
  A 通道 主体整体：松引导（cn .20/.55/dn .98）→ 翼/骷髅真换形（沿用 v390/v393 路线）
  B 通道 鹰头局部：裁鹰头 bbox → 原生 1024² 渲染（不降采样）→ 中紧引导
        （cn .32/.50/dn .88）+ prompt 写死解剖（喙长在脸上/眼可见/头转新角度）
  C 合成：B 以软 alpha 贴回 A（alpha = 高斯羽化的鹰头掩膜），翼/骷髅保留 A 的新形状。

用法：
    python src/v397_p6head.py                       # A 用现成 v397 ms2560 产物
    python src/v397_p6head.py --a-pass            # 先跑 A 通道（~6min）
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

from v342_rebirth import mask_p6, rebirth_subject          # noqa: E402
from v390_p6canny import CN, CKPT, POS, NEG                # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
OUT = ROOT / "jobs" / "v397_p6head"
OUT.mkdir(parents=True, exist_ok=True)
A_DEFAULT = ROOT / "jobs" / "v397_p6hi" / "p6_v397_ms2560_s888_snap.jpg"

# 鹰头 A 通道参数（沿用 v393 定稿：让模型真换形）
A_CN, A_END, A_DN, A_MS = 0.20, 0.55, 0.98, 2560
# 鹰头 B 通道：中紧引导 + 原生分辨率 → 解剖守得住又换细节
B_CN, B_END, B_DN, B_MS = 0.32, 0.50, 0.88, 1024
B_SEED = 888

POS_HEAD = ("bald eagle head close-up, fierce bald eagle face, "
            "white feathered head and neck, "
            "head turned to one side at a slightly new angle, "
            "bright golden-yellow hooked beak firmly attached to the eagle's face, "
            "beak above the chin, sharp brow, visible round eye, "
            "flat vector illustration, bold screen print, hard clean edges, "
            "crisp black linework, solid flat colors, pure black background, "
            "no text, no letters")
NEG_HEAD = ("beak on the chest, beak below the chin, disconnected beak, "
            "yellow blob, yellow patches, extra beak, second beak, deformed beak, "
            "missing eye, closed eye, blurry, low quality, photo, realistic, 3d render, "
            "airbrush, painterly, soft gradients, speckle, noise, gray haze, fog, "
            "text, letters, words, watermark")


def head_mask(img):
    """鹰头 = 上中部「白羽头颈 + 黄喙」实体，取最大连通域。"""
    a = np.asarray(img, np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = a.max(2) - a.min(2)
    H, W = lum.shape
    white = (lum > 140) & (sat < 70)
    yellow = (a[..., 0] > 140) & (a[..., 1] > 110) & (a[..., 2] < 120)
    m = white | yellow
    keep = np.zeros_like(m)
    keep[1050:2750, 1050:2500] = True            # 鹰头/颈/喙所在窗口
    m &= keep
    m = ndi.binary_closing(m, structure=_disk(9))
    m = ndi.binary_opening(m, structure=_disk(4))
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        m = lab == (int(np.argmax(sz)) + 1)
    m = ndi.binary_fill_holes(m)
    return m


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-pass", action="store_true", help="先跑 A 通道（整体重生）")
    ap.add_argument("--a-file", type=str, default=str(A_DEFAULT))
    ap.add_argument("--b-seed", type=int, default=B_SEED)
    ap.add_argument("--b-cn", type=float, default=B_CN)
    ap.add_argument("--b-end", type=float, default=B_END)
    ap.add_argument("--b-dn", type=float, default=B_DN)
    ap.add_argument("--b-ms", type=int, default=B_MS)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    src = Image.open(SRC).convert("RGB")
    t0 = time.time()

    # ── A 通道：主体整体（松引导，真换形） ────────────────────────────────
    if args.a_pass:
        a_img = rebirth_subject(src, mask_p6(src), POS, NEG, ckpt=CKPT,
                                denoise=A_DN, ipa_weight=0.0, color_match=0.0,
                                cn_name=CN, cn_strength=A_CN, cn_pre="canny",
                                cn_end=A_END, margin=60, grow=10, seed=888,
                                tag="A", max_side=A_MS)
        a_img.save(str(OUT / "A_subject.jpg"), quality=93)
        print(f"[A] 整体重生 {time.time()-t0:.0f}s")
    else:
        a_img = Image.open(args.a_file).convert("RGB")
        print(f"[A] 复用 {Path(args.a_file).name} {a_img.size}")

    # ── B 通道：鹰头原生 1024² 重画 ───────────────────────────────────────
    hm = head_mask(src)
    ys, xs = np.where(hm)
    print(f"[B] 鹰头掩膜 px={int(hm.sum())} bbox x[{xs.min()},{xs.max()}] "
          f"y[{ys.min()},{ys.max()}]  → {xs.max()-xs.min()+1}x{ys.max()-ys.min()+1}")
    t1 = time.time()
    hm_img = Image.fromarray((hm * 255).astype(np.uint8), "L")
    b_img = rebirth_subject(src, hm_img, POS_HEAD, NEG_HEAD, ckpt=CKPT,
                            denoise=args.b_dn, ipa_weight=0.0, color_match=0.0,
                            cn_name=CN, cn_strength=args.b_cn, cn_pre="canny",
                            cn_end=args.b_end, margin=50, grow=8, seed=args.b_seed,
                            tag="B", max_side=args.b_ms)
    b_img.save(str(OUT / "B_head.jpg"), quality=93)
    print(f"[B] 鹰头重画 seed={args.b_seed} cn={args.b_cn}/{args.b_end} "
          f"dn={args.b_dn} ms={args.b_ms} → {time.time()-t1:.0f}s")

    # ── C 合成：B 以软 alpha 贴回 A ───────────────────────────────────────
    soft = ndi.gaussian_filter(ndi.binary_dilation(hm, structure=_disk(7)).astype(np.float32), 5.0)
    alpha = np.clip(soft, 0.0, 1.0)[..., None]
    aa = np.asarray(a_img, np.float32)
    ba = np.asarray(b_img, np.float32)
    out = aa * (1.0 - alpha) + ba * alpha
    tag = args.tag or f"head_s{args.b_seed}_cn{args.b_cn}"
    dst = OUT / f"p6_step1_{tag}.jpg"
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(str(dst), quality=93)
    print(f"[C] 合成 → {dst}  (总 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
