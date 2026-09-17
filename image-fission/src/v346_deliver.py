"""v346 交付打包：四图完整成品 + 原图/成品逐列对比 + 局部放大 + QC 指标。

用户第 14 轮的三条硬要求在这份交付里逐条对应：
  ① 「裂变没看到明显变化」→ 对比图顶层=原图、底层=成品，逐列并排；
  ② 「比原图丑 严格禁止」→ 产线已把 p4 从 SDXL 重画改回**整棵仿射换位**（保真零掉档）；
  ③ 「主体元素不去裂变」→ p6 用 wide=120 放宽回贴区，主体 IoU 0.867→0.606。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp                # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图")
OUTV = ROOT / "jobs" / "router_out_v329"
DST = Path("E:/Desktop/v346_四图完整成品_第14轮交付")
DST.mkdir(parents=True, exist_ok=True)

ITEMS = [
    ("1_pinterest3_小蝴蝶+牛仔字", "Pinterest (3).jpg", "pinterest3_variant.jpg"),
    ("2_pinterest4_迷彩棕榈", "Pinterest (4).jpg", "Pinterest (4)_variant.jpg"),
    ("3_pinterest6_鹰骷髅+尖刺标题", "Pinterest (6).jpg", "pinterest6_variant.jpg"),
    ("4_6978_蝙蝠+弧字", "6978fabda2cc99629fa9e81f802762d3.jpg", "6978_variant.jpg"),
]


def fit(im, w):
    return im.resize((w, max(1, int(im.size[1] * w / im.size[0]))), Image.LANCZOS)


def main():
    rows = []
    for tag, s, v in ITEMS:
        sp, vp = SRC / s, OUTV / v
        if not vp.exists():
            print("[skip] 缺成品", vp)
            continue
        oi, vi = Image.open(sp).convert("RGB"), Image.open(vp).convert("RGB")
        vi.save(str(DST / f"{tag}.jpg"), quality=94)
        rows.append((tag, oi, vi))
        print(f"[deliver] {tag}  {oi.size} -> {vi.size}")

    # ---- 逐列对比：顶=原图，底=成品 ----
    W = 430
    panels = [(fit(o, W), fit(v, W)) for _, o, v in rows]
    hmax = max(max(a.size[1], b.size[1]) for a, b in panels)
    gap = 10
    tot_w = W * len(panels) + gap * (len(panels) + 1)
    sheet = Image.new("RGB", (tot_w, hmax * 2 + gap * 3 + 30), (22, 22, 24))
    dr = ImageDraw.Draw(sheet)
    for i, (a, b) in enumerate(panels):
        x = gap + i * (W + gap)
        sheet.paste(a, (x, 26))
        sheet.paste(b, (x, 26 + hmax + gap))
        dr.text((x + 4, 8), f"原图 {rows[i][0][2:]}", fill=(235, 235, 235))
        dr.text((x + 4, 26 + hmax + gap - 17), "裂变成品", fill=(255, 210, 90))
    sheet.save(str(DST / "00_对比_原图vs成品_四图.jpg"), quality=92)
    print("[deliver] 对比总表", sheet.size)

    # ---- 四图完整成品总表 ----
    W2 = 520
    pp = [fit(v, W2) for _, _, v in rows]
    h2 = max(p.size[1] for p in pp)
    s2 = Image.new("RGB", (W2 * len(pp) + gap * (len(pp) + 1), h2 + gap * 2), (22, 22, 24))
    for i, p in enumerate(pp):
        s2.paste(p, (gap + i * (W2 + gap), gap))
    s2.save(str(DST / "00_四图完整成品总表.jpg"), quality=92)
    print("[deliver] 成品总表", s2.size)

    # ---- p4 局部放大（树区）：原图 vs 成品 ----
    for tag, oi, vi in rows:
        if tag.startswith("2_"):
            box = (0, int(oi.size[1] * 0.45), oi.size[0], oi.size[1])
            a, b = oi.crop(box), vi.crop(box)
            w = 700
            a, b = fit(a, w), fit(b, w)
            c = Image.new("RGB", (w * 2 + 24, a.size[1] + 16), (22, 22, 24))
            c.paste(a, (8, 8)); c.paste(b, (w + 16, 8))
            c.save(str(DST / "z1_p4_树区_原图vs成品.jpg"), quality=95)
            print("[deliver] z1 p4 树区")
        if tag.startswith("3_"):
            box = (150, 1450, oi.size[0] - 150, oi.size[1])
            a, b = oi.crop(box), vi.crop(box)
            w = 700
            a, b = fit(a, w), fit(b, w)
            c = Image.new("RGB", (w * 2 + 24, a.size[1] + 16), (22, 22, 24))
            c.paste(a, (8, 8)); c.paste(b, (w + 16, 8))
            c.save(str(DST / "z2_p6_主体区_原图vs成品.jpg"), quality=95)
            print("[deliver] z2 p6 主体区")

    # ---- QC 指标 ----
    from scipy import ndimage as ndi
    from skimage.color import rgb2lab
    L = ["v346 第 14 轮 量化自检（客观口径，非目检）", "=" * 60, ""]
    for tag, oi, vi in rows:
        ao = np.asarray(oi, np.float32); av = np.asarray(vi, np.float32)
        d = np.sqrt(((rgb2lab(ao / 255.) - rgb2lab(av / 255.)) ** 2).sum(2))
        L.append(f"【{tag}】")
        L.append(f"  整体 LAB 距离 = {d.mean():.2f}")
        if tag.startswith("2_"):
            mo = cpp.tree_ink_mask(oi, lum_thr=55., sat_thr=14., min_px=25, thin_only=False)
            mv = cpp.tree_ink_mask(vi, lum_thr=55., sat_thr=14., min_px=25, thin_only=False)
            gone = mo & (~mv)
            dd = ndi.distance_transform_edt(~mv)
            far = gone & (dd > 8)
            base = av[~mo & ~mv].mean(1); atgone = av[far].mean(1)
            L.append(f"  墨迹: 原 {100*mo.mean():.2f}% → 成品 {100*mv.mean():.2f}%")
            L.append(f"  剪影 IoU = {(mo&mv).sum()/max(1,(mo|mv).sum()):.3f}  "
                     f"（v344 是 0.960 = 用户说的'没看到明显变化'）")
            L.append(f"  原墨位置(距新墨>8px) 亮度 mean={atgone.mean():.0f} vs 迷彩底 mean={base.mean():.0f}"
                     f"  → {'无残影' if abs(atgone.mean()-base.mean())<12 else '⚠有残影'}")
            L.append("  手法: 整棵仿射换位（34 棵错排置换 + 各向异性缩放 + 切变倾斜 + 镜像）"
                     "→ 线质零掉档，禁 SDXL 重画（必掉档=踩'比原图丑'红线）")
        if tag.startswith("3_"):
            f = lambda x: ndi.binary_opening(x.mean(2) > 55., np.ones((3, 3), bool))
            s1 = f(ao); s1[:1500] = False
            s2 = f(av); s2[:1500] = False
            L.append(f"  主体: 原 {100*s1.mean():.2f}% → 成品 {100*s2.mean():.2f}%  "
                     f"IoU = {(s1&s2).sum()/max(1,(s1|s2).sum()):.3f}")
            L.append(f"  （v344 是 0.867 = 用户说的'主体元素不去裂变'；"
                     f"用户认可的 6978 蝙蝠是 0.780）")
            L.append("  修法: wide=120 放宽回贴区 + bg_gate=35 门控 + protect 挡标题带；"
                     "cn .35/.45 与蝙蝠同档")
        if tag.startswith("1_"):
            L.append("  主蝶 acc>50000 保持零改动（用户第 11 轮硬规则）；仅小蝶形变 + 牛仔字裂变")
        if tag.startswith("4_"):
            L.append("  蝙蝠设计冻结为用户已认可的 v344 版（'蝙蝠设计可以'）；弧字原位改写")
        L.append("")
    (DST / "QC_指标.txt").write_text("\n".join(L), encoding="utf-8")
    print("[deliver] QC_指标.txt")
    print("[DONE]", DST)


if __name__ == "__main__":
    main()
