# -*- coding: utf-8 -*-
"""底色精确回填：把 IPAdapter/FLUX 抬亮的底色压回原图实测色值。
★只处理"底色像素"，金属元素/主体形态/色相一律不碰——纯 LOCK 修复，改色相=违规。

原理：
  1. 原图与成品构图已对齐（IPAdapter 锁了构图）→ 用原图定义"哪些像素属于底色"
  2. 原图底色掩膜 = 色距原图四角底色 < tol 的连通区域（只取四角向内，避开主体）
  3. 成品对应像素从 c_from 映射到 c_to，保留少量底色颗粒（scale<1）
  4. 掩膜羽化 2px 避免硬边

用法:
  python _zai_bg_restore.py --orig <原图> --final <成品> --out <输出> [--tol 42] [--grain 0.35]
"""
import argparse, os, sys
import numpy as np
from PIL import Image, ImageFilter

def corner_bg(a, k=24):
    """四角采样底色中位数（避开主体）。"""
    h, w = a.shape[:2]
    patches = np.concatenate([
        a[:k, :k].reshape(-1,3), a[:k, -k:].reshape(-1,3),
        a[-k:, :k].reshape(-1,3), a[-k:, -k:].reshape(-1,3)], axis=0)
    return np.median(patches, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=46, help="底色色距容差")
    ap.add_argument("--grain", type=float, default=0.35, help="保留底色颗粒比例")
    ap.add_argument("--feather", type=int, default=2)
    a = ap.parse_args()

    orig = np.asarray(Image.open(a.orig).convert("RGB"), dtype=np.float32)
    fin_img = Image.open(a.final).convert("RGB")
    fin = np.asarray(fin_img, dtype=np.float32)
    oh, ow = orig.shape[:2]
    if fin.shape[:2] != orig.shape[:2]:
        fin_img = fin_img.resize((ow, oh), Image.LANCZOS)
        fin = np.asarray(fin_img, dtype=np.float32)

    c_to = corner_bg(orig)       # 原图底色（目标）
    c_from = corner_bg(fin)      # 成品底色（来源）
    print(f"[bg] orig_corner={c_to.astype(int).tolist()} final_corner={c_from.astype(int).tolist()}",
          file=sys.stderr)

    # 原图底色判定：色距四角底色 < tol（底色是低饱和暗色，金属是高饱和暖金，天然分开）
    d_orig = np.sqrt(((orig - c_to) ** 2).sum(2))
    mask = (d_orig < a.tol).astype(np.uint8) * 255

    # 形态学：只保留大块连通底色，去掉元素内部小洞（避免误伤装饰细节）
    try:
        import cv2
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > 2000:   # 只保留大块底色
                keep[lab == i] = 255
        mask = keep
    except Exception:
        pass

    # 羽化
    if a.feather > 0:
        mask = np.asarray(Image.fromarray(mask).filter(
            ImageFilter.GaussianBlur(a.feather)), dtype=np.float32) / 255.0
    else:
        mask = mask.astype(np.float32) / 255.0
    m = mask[..., None]

    # 映射：new = c_to + (fin - c_from) * grain，保留底色颗粒但压回亮度
    mapped = c_to + (fin - c_from) * a.grain
    mapped = np.clip(mapped, 0, 255)
    out = fin * (1 - m) + mapped * m
    out_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    out_img.save(a.out, quality=95)

    # 验证
    d2 = np.asarray(out_img, dtype=np.float32)
    print(f"[bg] after_corner={corner_bg(d2).astype(int).tolist()} "
          f"target={c_to.astype(int).tolist()} mask_cover={float((mask>0.5).mean())*100:.1f}%",
          file=sys.stderr)
    print(f"[saved] {a.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
