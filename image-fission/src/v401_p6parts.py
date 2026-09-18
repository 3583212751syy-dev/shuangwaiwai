# -*- coding: utf-8 -*-
"""v401_p6parts —— p6 主体裂变：**多部件通道**合成器（C 档「AI 保色彩保构图、元素重绘」）

背景（用户第 24 轮口谕 → MEMORY 🔴38）：
  判据 = 该元素有没有「剪影辨识 / 品牌标识」功能。
  · 自由型（老鹰 + 骷髅 + 牛角）→ **C 档**：AI 保色彩 + 保构图，元素整只重绘。
  · 标识型（蝙蝠徽章、棕榈矢量）→ A~B 档：在原图基础上裂变。

为什么需要"多部件通道"：
  整幅一把 canny 强度时，各部件对"自由度"的要求是**相反**的：
    · 翼/鹰身 要换形 → 需要松引导（cn ~0.12 / dn 0.98）
    · 鹰头   要守解剖 → 需要紧引导（cn 0.32 / dn 0.88）  ← v397 已修
    · 骷髅   要守"骨"的逻辑（颅形不能变毛发）→ 需要中紧
    · 牛角   要守"平涂分节"的设计语言（原图签名）→ 需要中紧
  一度用统一参数，结果：v400C-a 的骷髅颅盖被画成波浪毛发、角被翼羽埋掉（1:1 实测）。

做法：base（整幅 C 档重绘）→ 逐个部件用**原图**取结构 + 紧引导重画 → 按部件软 alpha 贴回。
  贴回顺序 = 骷髅 → 角 → 鹰头（后贴者优先级高，解决掩膜重叠）。

用法：
    python src/v401_p6parts.py --mask-only                 # 只建掩膜并出可视化
    python src/v401_p6parts.py --base jobs/v400_p6free/p6_v400C_a_snap.jpg --tag v402
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
from v397_p6head import head_mask, POS_HEAD, NEG_HEAD, _disk  # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
OUT = ROOT / "jobs" / "v401_p6parts"
OUT.mkdir(parents=True, exist_ok=True)


def _lum(a):
    return a @ np.array([0.299, 0.587, 0.114], np.float32)


def _biggest(m, min_px=2000):
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return m
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    if sz.max() < min_px:
        return np.zeros_like(m)
    return lab == (int(np.argmax(sz)) + 1)


def skull_mask(img):
    """骷髅 = 白色骨壳实体（颅盖 + 下颌/牙列）。

    难点：骨壳被内部黑色线稿（颅缝/网格阴影）切成多个连通块，且窗口内还有
    鹰的白色胸羽（左右对称、贴着窗口边）与细白射线。取法 = 组件规则：
      · 阈值 lum>135 & sat<70（白/近白）
      · 闭 9 → 开 6（去细射线/星点）
      · 组件筛选：area≥40000 **且** bbox 完整落在 x[900,2700] 内 **且** y_min>2450
        （白胸羽 y_min≈2300 或触窗口边 → 天然被排除；射线面积不够 → 排除）
      · 并集 fill_holes
    """
    a = np.asarray(img, np.float32)
    lum, sat = _lum(a), a.max(2) - a.min(2)
    m = (lum > 135) & (sat < 70)
    keep = np.zeros_like(m)
    keep[2300:4850, 800:2900] = True               # 搜索窗（含爪抓的颅顶与下颌）
    m &= keep
    m = ndi.binary_closing(m, structure=_disk(9))
    m = ndi.binary_opening(m, structure=_disk(6))
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    out = np.zeros_like(m)
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        for i in range(1, n + 1):
            if int(sz[i - 1]) < 40000:
                continue
            mm = lab == i
            ys, xs = np.where(mm)
            if xs.min() < 900 or xs.max() > 2700 or ys.min() < 2450:
                continue                            # 触窗口边/偏上 → 翅膀白羽，弃
            out |= mm
    # 并集可能仍破碎（颅缝）→ 再闭一次把碎块连成一体，然后只留大件
    out = ndi.binary_closing(out, structure=_disk(13))
    out = ndi.binary_fill_holes(out)
    lab2, n2 = ndi.label(out, structure=np.ones((3, 3), bool))
    if n2:
        sz2 = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        keep2 = np.zeros(n2 + 1, bool)
        keep2[1:] = sz2 >= 30000
        out = keep2[lab2]
    return out


def horn_mask(img, side):
    """牛角 = 棕色平涂分节实心体。左/右各取窗口内棕区最大连通域。"""
    a = np.asarray(img, np.float32)
    sat = a.max(2) - a.min(2)
    br = (sat > 40) & (a[..., 0] > a[..., 2] + 20)
    keep = np.zeros_like(br)
    if side == "L":
        keep[2150:3500, 250:1550] = True
    else:
        keep[2150:3500, 2050:3400] = True
    m = br & keep
    m = ndi.binary_closing(m, structure=_disk(11))
    m = _biggest(m, min_px=8000)
    return ndi.binary_fill_holes(m)


def talon_mask(img):
    """黄爪 = 抓在颅骨上的亮黄实体（原图设计签名，重生常把它做丢）。"""
    a = np.asarray(img, np.float32)
    yel = (a[..., 0] > 150) & (a[..., 1] > 110) & (a[..., 2] < 130)
    keep = np.zeros_like(yel)
    keep[2450:3300, 1000:2500] = True
    m = yel & keep
    return ndi.binary_fill_holes(ndi.binary_closing(m, structure=_disk(7)))


# 部件表：name → (mask_fn, POS, NEG, cn, cn_end, dn, max_side, margin, grow, seed)
PART_ORDER = ["skull", "hornL", "hornR", "head"]

POS_SKULL = ("clean white skull, smooth rounded cranium of a human skull, "
             "two large hollow black eye sockets, nasal cavity, upper teeth row, "
             "fine ink hatching lines on the bone, "
             "flat vector illustration, bold screen print, hard clean edges, "
             "crisp black linework, solid flat colors, pure black background, "
             "horror illustration, no text, no letters")
NEG_SKULL = ("hair, fur, wool, mane, shaggy strands, wavy hair strokes, "
             "feathers, beak, eyes inside sockets, flesh, brain, "
             "blurry, low quality, photo, realistic, 3d render, airbrush, painterly, "
             "soft gradients, gray haze, fog, text, letters, watermark")

POS_HORN = ("smooth curved animal horn, flat solid brown horn with crisp black "
            "segment lines and clean highlight, segmented husk texture, "
            "flat vector illustration, bold screen print, hard clean edges, "
            "crisp black linework, solid flat colors, pure black background, "
            "no text, no letters")
NEG_HORN = ("feathers, plumage, fur, hair, wood grain, moss, leaves, "
            "blurry, low quality, photo, realistic, 3d render, airbrush, painterly, "
            "soft gradients, gray haze, fog, text, letters, watermark")


def get_parts(img):
    """返回 {name: (mask, POS, NEG, cn, end, dn, ms, margin, grow, seed)}"""
    return {
        "skull": (skull_mask(img), POS_SKULL, NEG_SKULL, 0.30, 0.55, 0.78, 1024, 45, 8, 4242),
        "hornL": (horn_mask(img, "L"), POS_HORN, NEG_HORN, 0.34, 0.60, 0.75, 1024, 40, 7, 5151),
        "hornR": (horn_mask(img, "R"), POS_HORN, NEG_HORN, 0.34, 0.60, 0.75, 1024, 40, 7, 6161),
        "head": (head_mask(img), POS_HEAD, NEG_HEAD, 0.32, 0.50, 0.88, 1024, 50, 8, 888),
    }


def build_masks(src):
    parts = get_parts(src)
    return {k: v[0] for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str,
                    default=str(ROOT / "jobs" / "v400_p6free" / "p6_v400C_a_snap.jpg"))
    ap.add_argument("--parts", type=str, default="skull,hornL,hornR,head")
    ap.add_argument("--mask-only", action="store_true")
    ap.add_argument("--tag", type=str, default="v402")
    args = ap.parse_args()

    src = Image.open(SRC).convert("RGB")
    parts = get_parts(src)

    # 掩膜统计 + 可视化
    print("[masks]")
    for k in PART_ORDER:
        m = parts[k][0]
        if m.sum() == 0:
            print(f"  {k:7s} EMPTY (!)")
            continue
        ys, xs = np.where(m)
        print(f"  {k:7s} px={int(m.sum()):8d} bbox x[{xs.min()},{xs.max()}] "
              f"y[{ys.min()},{ys.max()}]  {xs.max()-xs.min()+1}x{ys.max()-ys.min()+1}")
    # 重叠检查
    for i in range(len(PART_ORDER)):
        for j in range(i + 1, len(PART_ORDER)):
            a, b = PART_ORDER[i], PART_ORDER[j]
            ov = int((parts[a][0] & parts[b][0]).sum())
            if ov > 200:
                print(f"  [overlap] {a} ∩ {b} = {ov} px")

    if args.mask_only:
        vis = np.asarray(src).copy()
        cols = {"skull": (255, 60, 60), "hornL": (60, 255, 120), "hornR": (60, 180, 255),
                "head": (255, 220, 60)}
        for k in PART_ORDER:
            m = parts[k][0]
            vis[m] = (vis[m] * 0.35 + np.array(cols[k]) * 0.65).astype(np.uint8)
        Image.fromarray(vis).resize((760, 1064), Image.LANCZOS).save(
            str(OUT / "masks_view.jpg"), quality=92)
        print("[masks] → masks_view.jpg")
        return

    base = Image.open(args.base).convert("RGB")
    print(f"[base] {Path(args.base).name} {base.size}")
    cur = np.asarray(base, np.float32)

    want = [p for p in args.parts.split(",") if p.strip()]
    for name in PART_ORDER:
        if name not in want:
            continue
        m, pos, neg, cn, cend, dn, ms, mg, gr, seed = parts[name]
        if m.sum() == 0:
            print(f"[{name}] 掩膜为空 → 跳过")
            continue
        t0 = time.time()
        r = rebirth_subject(src, Image.fromarray((m * 255).astype(np.uint8), "L"),
                            pos, neg, ckpt=CKPT, denoise=dn, ipa_weight=0.0,
                            color_match=0.0, cn_name=CN, cn_strength=cn, cn_pre="canny",
                            cn_end=cend, margin=mg, grow=gr, seed=seed,
                            tag=name, max_side=ms)
        # 软 alpha 贴回当前合成
        soft = ndi.gaussian_filter(ndi.binary_dilation(m, structure=_disk(6)).astype(np.float32), 5.0)
        alpha = np.clip(soft, 0.0, 1.0)[..., None]
        cur = cur * (1.0 - alpha) + np.asarray(r, np.float32) * alpha
        Image.fromarray(cur.clip(0, 255).astype(np.uint8)).save(
            str(OUT / f"{args.tag}_{name}.jpg"), quality=93)
        print(f"[{name}] cn={cn}/{cend} dn={dn} seed={seed} → {time.time()-t0:.0f}s")

    dst = OUT / f"p6_{args.tag}_final.jpg"
    Image.fromarray(cur.clip(0, 255).astype(np.uint8)).save(str(dst), quality=93)
    print(f"[done] → {dst}")


if __name__ == "__main__":
    main()
