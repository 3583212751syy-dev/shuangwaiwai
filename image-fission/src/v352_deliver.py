"""v352 交付打包（第 14 轮修订版）—— p4 提重生分辨率定稿 + 全量重新出图。

【本轮修订内容】
  p4 的「发毛」经四轮排查后定位到**重生分辨率**：
    · 旧引擎 max_side=1536 把 1382x1894 的裁块降采样到 0.811x 再升回
      → 最细 7px 叶片只剩 ~5.7px 有效像素 → 二值化后碎成短虚线；
    · 实测（src/v351_p4hires.py，仅动 max_side / 预放大倍数）：
        H1 ms1536（旧交付）  墨 22.29%  笔宽 5.66  碎块 177
        H2 ms2048 原生 1:1   墨 22.38%  笔宽 5.66  碎块 126   ← 新主方案（帕累托改进）
        H3 预放大 1.5x       墨 21.99%  笔宽 4.00  碎块 130   ← 备选（更细的线稿感）
    · 原图基准：墨 22.63%  笔宽 4.47  碎块 91
  同轮实测无用/有害、已排除：局部密度阈值(v350)、厚度门控阈值(v350d)、
  门控闭运算(v350f，细线区厚度 4.0→6.3~10.0)、全局阈值扫描 t54/t58（环纹变粗梯子）。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp                # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图")
OUTV = ROOT / "jobs" / "router_out_v329"
HIRES = ROOT / "jobs" / "v351_p4hires"
DST = Path("E:/Desktop/v352_四图成品_第14轮修订")
DST.mkdir(parents=True, exist_ok=True)

ITEMS = [
    ("1_pinterest3_小蝴蝶+牛仔字", "Pinterest (3).jpg", "pinterest3_variant.jpg"),
    ("2_pinterest4_迷彩棕榈_分辨率定稿", "Pinterest (4).jpg", "Pinterest (4)_variant.jpg"),
    ("3_pinterest6_鹰骷髅+尖刺标题", "Pinterest (6).jpg", "pinterest6_variant.jpg"),
    ("4_6978_蝙蝠+弧字", "6978fabda2cc99629fa9e81f802762d3.jpg", "6978_variant.jpg"),
]
P4_ALT = HIRES / "H3_pre15_ms4096.jpg"
GAP = 10


def font(sz):
    for p in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
              r"C:\Windows\Fonts\arialbd.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def fit(im, w):
    return im.resize((w, max(1, int(im.size[1] * w / im.size[0]))), Image.LANCZOS)


def strip(items, cell, path, title, lab_h=40, title_h=30):
    """items = [(label, PIL)] 等宽 cell，横向排列（2 列/行）。"""
    cols = 2 if len(items) <= 4 else 3
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell + GAP * (cols + 1),
                              title_h + rows * (cell + lab_h) + GAP * (rows + 1)),
                      (24, 24, 26))
    dr = ImageDraw.Draw(sheet)
    dr.text((GAP, 5), title, fill=(255, 210, 90), font=font(22))
    fp = font(23)
    for i, (lb, im) in enumerate(items):
        if im.size != (cell, cell):
            im = im.resize((cell, cell), Image.LANCZOS)
        x = GAP + (i % cols) * (cell + GAP)
        y = title_h + GAP + (i // cols) * (cell + lab_h + GAP)
        sheet.paste(im, (x, y))
        dr.text((x + 4, y + cell + 6), lb, fill=(240, 240, 240), font=fp)
    sheet.save(str(path), quality=94)
    print(f"[deliver] {path.name} {sheet.size}")


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

    p4o = Image.open(SRC / "Pinterest (4).jpg").convert("RGB")
    p4b = Image.open(OUTV / "Pinterest (4)_variant.jpg").convert("RGB")

    # ---- ① 原图 vs 成品（四图，上下） ----
    W = 440
    panels = [(fit(o, W), fit(v, W)) for _, o, v in rows]
    hmax = max(max(a.size[1], b.size[1]) for a, b in panels)
    sheet = Image.new("RGB", (W * len(panels) + GAP * (len(panels) + 1),
                              hmax * 2 + GAP * 3 + 30), (22, 22, 24))
    dr = ImageDraw.Draw(sheet)
    fp = font(17)
    for i, (a, b) in enumerate(panels):
        x = GAP + i * (W + GAP)
        sheet.paste(a, (x, 26))
        sheet.paste(b, (x, 26 + hmax + GAP))
        dr.text((x + 4, 8), f"原图 {rows[i][0][2:]}", fill=(235, 235, 235), font=fp)
        dr.text((x + 4, 26 + hmax + GAP - 18), "裂变成品", fill=(255, 210, 90), font=fp)
    sheet.save(str(DST / "00_对比_原图vs成品_四图.jpg"), quality=92)
    print("[deliver] 对比总表", sheet.size)

    # ---- ② 四图成品总表 ----
    W2 = 540
    pp = [fit(v, W2) for _, _, v in rows]
    h2 = max(p.size[1] for p in pp)
    s2 = Image.new("RGB", (W2 * len(pp) + GAP * (len(pp) + 1), h2 + GAP * 2), (22, 22, 24))
    for i, p in enumerate(pp):
        s2.paste(p, (GAP + i * (W2 + GAP), GAP))
    s2.save(str(DST / "00_四图成品总表.jpg"), quality=92)
    print("[deliver] 成品总表", s2.size)

    # ---- ③ p4 三方案：原图 / 主方案(ms2048) / 备选(细线 pre1.5x) ----
    labs = [("原图", p4o), ("主方案 ms2048 原生重生", p4b)]
    if P4_ALT.exists():
        a_ = Image.open(P4_ALT).convert("RGB")
        a_.save(str(DST / "2b_pinterest4_备选_细线版.jpg"), quality=94)
        labs.append(("备选 预放大1.5x 细线版", a_))
        print("[deliver] p4 备选已存")
    strip(labs, 560, DST / "00_p4_三方案对比.jpg", "p4 方案对比（整图）")

    # ---- ④ p4 分辨率实验证据（1:1 树区） ----
    box = (90, 230, 830, 970)
    ev = [("原图 碎块91", p4o)]
    for nm, lb in (("H1_ms1536", "H1 ms1536 碎块177"),
                   ("H2_ms2048", "H2 ms2048 碎块126"),
                   ("H3_pre15_ms4096", "H3 预放大1.5x 碎块130")):
        f = HIRES / f"{nm}.jpg"
        if f.exists():
            ev.append((lb, Image.open(f).convert("RGB").crop(box)))
    strip(ev, 740, DST / "00_p4_分辨率实验_树区1to1.jpg",
          "p4 重生分辨率实验（树区 1:1）：碎块数 177→126，墨量/笔宽不变")

    # ---- ⑤ 局部 1:1 ----
    for tag, oi, vi in rows:
        if tag.startswith("2_"):
            a, b = fit(p4o.crop(box), 640), fit(p4b.crop(box), 640)
            c = Image.new("RGB", (640 * 2 + 24, a.size[1] + 16), (22, 22, 24))
            c.paste(a, (8, 8)); c.paste(b, (656, 8))
            c.save(str(DST / "z1_p4_树区_原图vs成品.jpg"), quality=95)
            print("[deliver] z1 p4 树区")
        if tag.startswith("3_"):
            b3 = (0, 0, 1800, 1800)
            a, b = fit(oi.crop(b3), 640), fit(vi.crop(b3), 640)
            c = Image.new("RGB", (640 * 2 + 24, a.size[1] + 16), (22, 22, 24))
            c.paste(a, (8, 8)); c.paste(b, (656, 8))
            c.save(str(DST / "z2_p6_标题带左半_原图vs成品.jpg"), quality=95)
            b4 = (150, 1450, oi.size[0] - 150, oi.size[1])
            a, b = fit(oi.crop(b4), 640), fit(vi.crop(b4), 640)
            c = Image.new("RGB", (640 * 2 + 24, a.size[1] + 16), (22, 22, 24))
            c.paste(a, (8, 8)); c.paste(b, (656, 8))
            c.save(str(DST / "z3_p6_主体区_原图vs成品.jpg"), quality=95)
            print("[deliver] z2/z3 p6")

    # ---- ⑥ QC ----
    L = ["v352 第 14 轮修订版 量化自检", "=" * 62, ""]
    for tag, oi, vi in rows:
        ao = np.asarray(oi, np.float32); av = np.asarray(vi, np.float32)
        d = np.sqrt(((rgb2lab(ao / 255.) - rgb2lab(av / 255.)) ** 2).sum(2))
        L.append(f"【{tag}】整体 LAB 距离 = {d.mean():.2f}")
        if tag.startswith("2_"):
            mo = cpp.tree_ink_mask(oi, lum_thr=55., sat_thr=14., min_px=25, thin_only=False)
            mv = cpp.tree_ink_mask(vi, lum_thr=55., sat_thr=14., min_px=25, thin_only=False)
            L.append(f"  墨迹 原 {100*mo.mean():.2f}% → 成品 {100*mv.mean():.2f}%")
            L.append(f"  墨迹 IoU = {(mo&mv).sum()/max(1,(mo|mv).sum()):.3f}"
                     f"   （v344 = 0.960 ← 用户说的『没看到明显变化』；"
                     f"本轮旧版 0.453 / 新版 0.454）")
            L.append("  手法: 蝙蝠法 = SDXL 低 cn 结构重生(cn .35/end .45/dn .90) + 放宽回贴区"
                     " wide=120 + 墨量密度匹配；本轮新增 max_side 1536→2048（原生 1:1 渲染）")
            L.append("  分辨率实测（其余参数仅差 max_side）：")
            L.append("     原图          墨 22.63%  笔宽 4.47  碎块  91")
            L.append("     H1 ms1536 旧  墨 22.29%  笔宽 5.66  碎块 177")
            L.append("     H2 ms2048 新  墨 22.38%  笔宽 5.66  碎块 126  ← 帕累托改进，已采用")
            L.append("     H3 pre1.5x 备 墨 21.99%  笔宽 4.00  碎块 130  ← 线更细，可选")
        if tag.startswith("3_"):
            f = lambda x: ndi.binary_opening(x.mean(2) > 55., np.ones((3, 3), bool))
            s1 = f(ao); s1[:1500] = False
            s2 = f(av); s2[:1500] = False
            L.append(f"  主体 原 {100*s1.mean():.2f}% → 成品 {100*s2.mean():.2f}%  "
                     f"IoU = {(s1&s2).sum()/max(1,(s1|s2).sum()):.3f}")
            L.append("  （v344 = 0.867 ← 用户说的『主体元素不去裂变』；已认可的 6978 蝙蝠 = 0.780）")
            L.append("  修法: 放宽回贴区 wide=120 + bg_gate=35 门控 + protect 挡标题带；"
                     "标题 MRCHOSR→VORCRAVEN")
        if tag.startswith("1_"):
            L.append("  主蝶 acc>50000 保持零改动（用户第 11 轮硬规则）；仅小蝶形变 + 牛仔字裂变")
        if tag.startswith("4_"):
            L.append("  蝙蝠设计冻结为用户已认可的 v344 版（『蝙蝠设计可以』）；弧字原位改写")
        L.append("")
    L += [
        "本轮 p4 排查结论（4 条路线全实测，防回退）：",
        "  ✗ 局部自适应密度阈值(v350)：树干环纹被压成粗梯子（笔宽 4.47→6.00）",
        "  ✗ 笔画厚度门控阈值(v350d)：无增益（碎块 197→249）",
        "  ✗ 原图厚度门控闭运算(v350f)：细线区厚度也被抬高 4.0→6.3~10.0，墨量涨到 26%",
        "  ✗ 全局阈值扫描(v350g)：t54/t58 碎块降但环纹变粗；t50 已是最优工作点",
        "  ✓ 重生分辨率(v351)：**唯一有效**——max_side 1536→2048 让碎块 177→126，",
        "     墨量与笔宽零代价。这是 p4 的定稿配置。",
        "",
        "其它已排除路线：",
        "  · 程序化 palm_art 重画 → 刺球（v330 用户已否）",
        "  · 原图墨迹极坐标重组 → 细叶被撕成刮痕",
        "  · 逐棵树裁块 SDXL 重生（34棵×1024）→ 毛边糊团",
        "  · 逐棵树切分+换位(v346) → 本图墨迹互连（膨胀 r=2 只剩 20 组、最大组占全墨 42%）→",
        "    换位后出现『原地留空 + 漂浮碎块』，1:1 目检不合格",
        "  · 平滑姿态场加大幅度 → 细高树干被高度相关侧移拉成斜纹/拉丝",
    ]
    (DST / "QC_指标.txt").write_text("\n".join(L), encoding="utf-8")
    print("[deliver] QC_指标.txt")
    print("[DONE]", DST)


if __name__ == "__main__":
    main()
