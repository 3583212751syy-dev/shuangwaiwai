# -*- coding: utf-8 -*-
"""v394_p6split — pinterest6「主体/文本分离」两段式裂变定稿。

用户第 20 轮否决 v393（原话「你到底改变了什么」）+ 流程指令：
  「主体跟文本分开处理，先处理图片裂变出来再把文本裂变添加进去。」

v393 观感≈原图的两个真凶（都写死在这里防回退）：
  ① rebirth_subject 从没传 wide → 默认 wide=0：噪声掩膜=原剪影膨胀
     (grow+6+0)≈±16px —— 新剪影**物理上长不出原轮廓**，只有轮廓内纹理在换
     → 骷髅/双角/闪电/鹰的构图全部锁死 = 用户看到的"没变"。
  ② 标题直接沿用原图像素 / 叠同款字体字 → 文本层从没裂变。

本版两段式：
  Stage A 主体裂变（无文本参与）：wide=140 解锁剪影 + cn 0.15/end 0.50/dn 1.00
    + prompt 要求构图级改变（翼上扬 V 形、鹰头下俯、颅骨侧倾、双角外扫、
    闪电布局重排）。protect 标题带 y<1500 禁重生侵入。
  Stage B 文本裂变：取**原标题**白色尖刺字 → 列投影切字母簇 → 每簇随机
    (旋转±8°/纵缩放0.85~1.15/垂直抖动±60px/间隙抖动) 重排 → 铺满标题带
    → 叠回 Stage A 成品。同风格（原字身），新排列 = 文本也裂变。

用法：
    python src/v394_p6split.py --stage subject --seed 777 --tag v394_s777
    python src/v394_p6split.py --stage final --rebirth jobs/.../xxx_snap.jpg --seed 11
产出 jobs/v394_p6split/。
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

from v342_rebirth import mask_p6, rebirth_subject, _lum   # noqa: E402
from v377_p6rb import snap_p6                             # noqa: E402
from styles import camo_palm_pattern as cpp               # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CKPT = "juggernautXL_ragnarokBy.safetensors"
OUT = ROOT / "jobs" / "v394_p6split"
BAND = 1500                     # 标题带下缘（全分辨率 px）

# v395：v394 seed 2026 解剖崩坏（鹰头太小埋进翼里/黄爪悬空）后的修复版。
# 部位归属锁保留；新增「大头/爪抓颅骨/翼比例」约束；dn 0.92+wide 90 收贴原解剖。
POS = ("fierce bald eagle with a LARGE detailed white feathered head held high, "
       "bright golden-yellow hooked beak attached to the eagle's face, "
       "fierce visible eye, yellow eagle talons gripping the top of the skull, "
       "broad dark chocolate brown wings spread behind the eagle, "
       "different feather arrangement and new wing angle, "
       "large cracked horned demon skull tilted at an angle below the eagle, "
       "clean white bone skull with even fine cross-hatch shading, "
       "empty hollow pitch-black eye sockets and pitch-black nasal cavity, "
       "two long curved ribbed horns sweeping outward, "
       "jagged white lightning bolts scattered in a new irregular pattern, "
       "flat vector illustration, bold screen print, hard clean edges, crisp linework, "
       "solid flat colors, tan horns, pure black background, "
       "centered composition, high contrast, no text, no letters")
NEG = ("yellow patches, yellow stains, yellow spots on skull, yellow feathers on skull, "
       "yellow nose, yellow nasal cavity, yellow teeth, beak on the skull, second beak, "
       "dirty bone, dark stains, smudges, blotches, mud, grime, "
       "glowing eyes, luminous eyes, eye light, text, letters, words, watermark, "
       "blurry, low quality, photo, realistic, 3d render, airbrush, painterly, "
       "soft gradients, smooth shading, extra heads, extra skulls, extra birds, "
       "deformed, gray haze, fog, noise, speckle, dirty background, "
       "halo, glow, pale beak, white beak")


# ------------------------------------------------------------- Stage A 主体
def subject(seed=777, tag=None, cn_strength=0.20, cn_end=0.50, denoise=0.92,
            wide=90, max_side=2048):
    out_dir = OUT / "subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"v394_s{seed}"
    src = Image.open(SRC).convert("RGB")
    W, H = src.size
    mask = mask_p6(src)
    prot = np.zeros((H, W), bool)
    prot[:BAND, :] = True                      # 标题带禁重生侵入
    t = time.time()
    full = rebirth_subject(src, mask, POS, NEG, ckpt=CKPT, denoise=denoise,
                           ipa_weight=0.0, color_match=0.0, cn_name=CN,
                           cn_strength=cn_strength, cn_pre="canny", cn_end=cn_end,
                           margin=60, grow=10, seed=seed, tag=tag, max_side=max_side,
                           wide=wide, bg_gate=40, protect=prot)
    snap = snap_p6(src, full, mask)
    dst = out_dir / f"p6_{tag}_snap.jpg"
    snap.save(str(dst), quality=93)
    print(f"[v394-subject] seed={seed} wide={wide} cn={cn_strength}/{cn_end} dn={denoise} "
          f"→ {dst}  ({time.time() - t:.0f}s)")
    return dst


# ------------------------------------------------------------- Stage B 文本
def _segments(ink, min_gap=12):
    """列投影切字母簇：全零列隙 ≥min_gap 切开；不够切则按低谷补切。"""
    prof = ink.sum(0).astype(np.float32)
    W = len(prof)
    cuts = [0]
    x = 0
    while x < W:
        if prof[x] == 0:
            x0 = x
            while x < W and prof[x] == 0:
                x += 1
            if x - x0 >= min_gap and x0 > cuts[-1] and x < W:
                cuts.append(x0 + (x - x0) // 2)
        else:
            x += 1
    cuts.append(W)
    if len(cuts) < 8:                          # 尖刺连体 → 低谷补切到 ≥8 段
        sm = ndi.gaussian_filter(prof, 25)
        while len(cuts) - 1 < 8:
            segs = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
            segs = [s for s in segs if s[1] - s[0] > 200]
            if not segs:
                break
            a, b = max(segs, key=lambda s: s[1] - s[0])
            mid = a + int(np.argmin(sm[a + 80:b - 80])) + 80
            cuts = sorted(cuts + [int(mid)])
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
            if cuts[i + 1] - cuts[i] > 40]


def _title_ink(bandc):
    """标题带内的**标题笔画**。⚠️ 标题与半调扇纹是同一个连通件（实测
    rows 88-1499 一整块），且扇纹中部是实心楔形、腐蚀分不开。
    分离策略：腐蚀 disk5 → 连通块里 min_row<300 的是**标题字身**（从画布顶
    长出来），扇纹核 min_row≈1085 → 排除；字身核回胀 disk8 取回笔画主体。"""
    lum = _lum(bandc)
    sat = bandc.max(2) - bandc.min(2)
    ink = (lum > 170) & (sat < 50)
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    if n:                                       # 去星点/碎屑(<60px)
        sz = np.bincount(lab.ravel())
        ink &= sz[lab] >= 60
    core = ndi.binary_erosion(ink, structure=cpp.tf._disk(3))
    cl, cn = ndi.label(core, structure=np.ones((3, 3), bool))
    if not cn:
        return np.zeros_like(ink)
    cs = np.bincount(cl.ravel())
    rr, cc = np.nonzero(cl)
    ids = cl[rr, cc]
    minrow = np.full(cn + 1, 10 ** 9, np.int64)
    np.minimum.at(minrow, ids, rr)
    keep = (cs >= 200) & (minrow < 300)        # 标题字身从画布顶长出（min_row<300）
    keep[0] = False                            # 扇纹核 min_row≈1085 → 被此条排除
    body = keep[cl]
    return ndi.binary_dilation(body, structure=cpp.tf._disk(6)) & ink


def _band_erase(bandc):
    """擦除用：标题带内亮白件（标题+扇纹+带内闪电，≥60px）∪ 蓝烟
    （do_pinterest6 v336 判据 b>r+6 & b>22）→ 全部清黑，对齐 v393 观感。"""
    lum = _lum(bandc)
    sat = bandc.max(2) - bandc.min(2)
    ink = (lum > 170) & (sat < 50)
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    if n:
        sz = np.bincount(lab.ravel())
        ink &= sz[lab] >= 60
    blue = (bandc[..., 2] > bandc[..., 0] + 6.0) & (bandc[..., 2] > 22.0)
    return ink | blue


def fission_text(seed=11, band=BAND):
    """原标题白色尖刺字 → 字母簇重排 → 新标题带（**RGBA**：只含文字，底透明，
    以便叠回重生图时不砍掉扇纹/翼尖）。"""
    src = np.asarray(Image.open(SRC).convert("RGB"), np.float32)
    bandc = src[:band]
    ink = _title_ink(bandc)
    segs = _segments(ink)
    print(f"[v394-text] 字母簇={len(segs)}  标题墨={100 * ink.mean():.2f}%")
    rng = np.random.default_rng(seed)

    alpha_f = np.clip((_lum(bandc) - 100.0) / 110.0, 0.0, 1.0) * ink
    # v395 布局（修 v394 叠压乱码）：先旋转 reshape → **实测**每簇宽高 →
    # 再按实测总宽归一化缩放铺满 → 顺序铺放。旧版先算宽后旋转 → 变宽叠压。
    raw = []
    for (a, b) in segs:
        cols = slice(a, b)
        rows = np.where(ink[:, cols].any(1))[0]
        if len(rows) == 0:
            continue
        r0, r1 = max(0, rows.min() - 6), min(band, rows.max() + 6)
        pa = alpha_f[r0:r1, cols]
        pc = bandc[r0:r1, cols]
        sy = rng.uniform(0.90, 1.12)
        ang = rng.uniform(-6, 6)
        h1 = max(8, int(pa.shape[0] * sy))
        w1 = max(8, int(pa.shape[1]))
        pa_s = np.asarray(Image.fromarray((pa * 255).astype(np.uint8), "L")
                          .resize((w1, h1), Image.BILINEAR), np.float32) / 255.0
        pc_s = np.asarray(Image.fromarray(pc.astype(np.uint8), "RGB")
                          .resize((w1, h1), Image.BILINEAR), np.float32)
        pa_r = ndi.rotate(pa_s, ang, reshape=True, order=1, mode="constant", cval=0.0)
        pc_r = ndi.rotate(pc_s, ang, reshape=True, order=1, mode="constant", cval=0.0)
        raw.append((pa_r, pc_r))
    if not raw:
        return np.zeros((band, src.shape[1], 3), np.float32), np.zeros((band, src.shape[1]), np.float32)
    gaps = [int(rng.integers(30, 90)) for _ in range(len(raw) + 1)]
    total = sum(p[0].shape[1] for p in raw) + sum(gaps)
    kfill = (src.shape[1] * 0.985) / total
    kfill = min(kfill, 1.0)                    # 只缩不放，保证字身不糊
    cy0 = int(band * 0.42)                     # 标题字身垂直中心（原题≈600px 处）
    new = np.zeros((band, src.shape[1], 3), np.float32)
    na = np.zeros((band, src.shape[1]), np.float32)
    xcur = int(gaps[0] * kfill)
    placed = 0
    for i, (pa_i, pc_i) in enumerate(raw):
        ph, pw = pa_i.shape
        w2, h2 = max(8, int(pw * kfill)), max(8, int(ph * kfill))
        pa_i = np.asarray(Image.fromarray((pa_i * 255).astype(np.uint8), "L")
                          .resize((w2, h2), Image.BILINEAR), np.float32) / 255.0
        pc_i = np.asarray(Image.fromarray(pc_i.astype(np.uint8), "RGB")
                          .resize((w2, h2), Image.BILINEAR), np.float32)
        dy = int(rng.integers(-40, 41))
        y0 = int(np.clip(cy0 - h2 // 2 + dy, 0, max(0, band - h2)))
        x1 = min(src.shape[1], xcur + w2)
        if x1 - xcur < 8:
            break
        va = pa_i[:, :x1 - xcur]
        vc = pc_i[:, :x1 - xcur]
        hh = min(va.shape[0], band - y0)
        va, vc = va[:hh], vc[:hh]
        sa = na[y0:y0 + hh, xcur:x1]
        sa[:] = np.maximum(sa, va)
        new[y0:y0 + hh, xcur:x1] = (new[y0:y0 + hh, xcur:x1] * (1 - va[..., None])
                                    + vc * va[..., None])
        placed += 1
        xcur = x1 + int(gaps[i + 1] * kfill)
        if xcur >= src.shape[1]:
            break
    print(f"[v395-text] 铺了 {placed}/{len(raw)} 簇  kfill={kfill:.2f}  "
          f"α覆盖={100 * (na > 0.3).mean():.1f}%")
    return new, na                        # (RGB 文字色, α) —— 底透明，叠回用


def final(rebirth_path, text_seed=11, out_name=None):
    """Stage A 成品 → 擦标题带全部亮件(nn 回填成黑底，对齐 v393 观感)
    → 以 α 叠回裂变文本 → p6 完整成品。"""
    rp = Path(rebirth_path)
    img = Image.open(rp).convert("RGB")
    W, H = img.size
    # ① 只擦**标题笔画**（_title_ink 已排除扇纹），LaMa 结构重建扇纹。
    #    ⚠️ v395 教训：把蓝烟并入擦除会让 LaMa 在大块蓝烟区幻化白色雾团吞掉
    #    新字母 —— 蓝烟是原设计元素，保留不擦。
    src = np.asarray(Image.open(SRC).convert("RGB"), np.float32)
    tink = _title_ink(src[:BAND])
    mask = np.zeros((H, W), bool)
    mask[:BAND] = ndi.binary_dilation(tink, structure=cpp.tf._disk(6))
    mask[BAND:] = False                        # 严守标题带，不碰重生主体
    img = cpp.tf.erase(img, mask, method="lama", max_side=1100, margin=40)
    # ② 叠裂变文本（α 合成）
    tc, ta = fission_text(seed=text_seed)
    a = np.asarray(img, np.float32)
    aa = ta[..., None]
    a[:BAND] = a[:BAND] * (1 - aa) + tc * aa
    out = OUT / "final" / (out_name or f"p6_v394_final_{rp.stem}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB").save(str(out), quality=95)
    print(f"[v394-final] → {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    ps = sub.add_parser("subject")
    ps.add_argument("--seed", type=int, default=777)
    ps.add_argument("--tag", type=str, default=None)
    ps.add_argument("--cn", type=float, default=0.20)
    ps.add_argument("--cn-end", type=float, default=0.50)
    ps.add_argument("--dn", type=float, default=0.92)
    ps.add_argument("--wide", type=int, default=90)
    pf = sub.add_parser("final")
    pf.add_argument("--rebirth", type=str, required=True)
    pf.add_argument("--text-seed", type=int, default=11)
    pf.add_argument("--out-name", type=str, default=None)
    a = ap.parse_args()
    if a.stage == "subject":
        subject(seed=a.seed, tag=a.tag, cn_strength=a.cn, cn_end=a.cn_end,
                denoise=a.dn, wide=a.wide)
    else:
        final(a.rebirth, text_seed=a.text_seed, out_name=a.out_name)
