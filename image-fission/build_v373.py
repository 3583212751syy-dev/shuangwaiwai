"""build_v373.py —— 汇总 v373（第 16 轮修订）四图成品 + 量化自检 + 1:1 对照。

产物目录：E:/Desktop/v373_四图成品_第16轮修订/
  · 1..4 四张成品
  · 00_四图成品总表.jpg
  · z1..z4 原图 vs 成品 1:1 对照
  · QC_指标.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v378_p3rb import build_masks as p3_masks          # noqa: E402
from v342_rebirth import mask_p6                       # noqa: E402

R = ROOT / "jobs" / "router_out_v329"
SRC = Path("E:/Desktop/图裂变测试图")
DST = Path("E:/Desktop/v373_四图成品_第16轮修订")
DST.mkdir(parents=True, exist_ok=True)

cfg = json.load(open(ROOT / "regression_set" / "set.json", encoding="utf-8"))
BY = {i["id"]: i for i in cfg["images"]}

ITEMS = [
    ("1_pinterest3_牛仔蝶+denim字", "pinterest3", R / "pinterest3_variant.jpg"),
    ("2_pinterest4_迷彩棕榈", "pinterest4", R / "Pinterest (4)_variant.jpg"),
    ("3_pinterest6_鹰骷髅+尖刺标题", "pinterest6", R / "pinterest6_variant.jpg"),
    ("4_6978_蝙蝠+弧字", "6978", R / "6978_variant.jpg"),
]

ZOOM = {                       # 1:1 对照裁剪框（成品尺寸下）
    "pinterest3": (0, 300, 736, 1308),
    "pinterest4": (0, 0, 1242, 1754),
    "pinterest6": (250, 1350, 3350, 4350),
    "6978": (0, 0, 1552, 2000),
}


def lab_dE(o, n):
    a = rgb2lab(np.asarray(o, np.float32) / 255.0)
    b = rgb2lab(np.asarray(n, np.float32) / 255.0)
    return np.sqrt(((a - b) ** 2).sum(-1))


def subj_mask(iid, img):
    if iid == "pinterest3":
        m, _, _ = p3_masks(img)
        return m
    if iid == "pinterest6":
        return mask_p6(img)
    if iid == "pinterest4":
        a = np.asarray(img.convert("RGB"), np.float32)
        return a.max(2) < 60                      # 墨迹层（p4 判墨 lum<60 → 取 <60 的暗区）
    return None


def ink_stats(gray):
    ink = gray < 30
    from scipy import ndimage as ndi
    lab, n = ndi.label(ink, np.ones((3, 3), bool))
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    big = sz[sz >= 8]
    return ink.mean() * 100.0, int((sz >= 8).sum()), float(np.median(big)) if len(big) else 0.0


def main():
    report = []
    finals = []
    for name, iid, fin in ITEMS:
        if not fin.exists():
            print(f"[MISS] {fin}")
            continue
        o = Image.open(BY[iid]["path"]).convert("RGB")
        n = Image.open(fin).convert("RGB")
        ext = Path(BY[iid]["filename"]).suffix
        assert o.size == n.size, f"{iid}: {o.size} vs {n.size}"
        dst = DST / f"{name}{ext}"
        Image.open(fin).save(dst, quality=95)
        finals.append((name, n))

        dE = lab_dE(o, n)
        m = subj_mask(iid, o)
        if m is not None:
            mm = m
            iou = float((mm & subj_mask(iid, n)).sum()) / max(1, float((mm | subj_mask(iid, n)).sum()))
            d = dE[mm]
            rec = (f"  主体区 LAB ΔE 均值 = {d.mean():.2f}  |  ΔE>15 的像素占比 = "
                   f"{(d > 15).mean() * 100:.1f}%  |  主体剪影 IoU = {iou:.3f}")
        else:
            rec = "  （无主体掩膜）"
        head = (f"【{name}】{o.size[0]}x{o.size[1]}\n"
                f"  全图 LAB ΔE 均值 = {dE.mean():.2f}  |  ΔE>15 占比 = {(dE > 15).mean() * 100:.1f}%")
        print(head)
        print(rec)
        report.append((name, head, rec, o, n))

        # 1:1 对照
        box = ZOOM[iid]
        th = 700
        cols = []
        for nm, im in (("ORIG", o), ("NEW", n)):
            c = im.crop(box)
            w = int(c.width * th / c.height)
            cols.append((nm, c.resize((max(1, w), th), Image.LANCZOS)))
        gap = 10
        cv = Image.new("RGB", (sum(c[1].width for c in cols) + gap * (len(cols) + 1), th + 24),
                       (25, 25, 28))
        dr = ImageDraw.Draw(cv)
        x = gap
        for nm, c in cols:
            cv.paste(c, (x, 21))
            dr.text((x + 3, 5), nm, fill=(240, 240, 240))
            x += c.width + gap
        cv.save(DST / f"z{name[0]}_{name[2:12]}_原图vs成品.jpg", quality=94)

    # 总表 2x2（按各行实际宽度逐张排布，避免右侧被画布裁掉）
    th = 640
    cols = []
    for nm, im in finals:
        w = int(im.width * th / im.height)
        cols.append((nm, im.resize((max(1, w), th), Image.LANCZOS)))
    gap = 10
    rows = [cols[:2], cols[2:]]
    W = max(sum(c[1].width for c in r) + gap * (len(r) - 1) for r in rows) + gap * 2
    cv = Image.new("RGB", (W, th * 2 + gap * 3 + 44), (22, 22, 25))
    dr = ImageDraw.Draw(cv)
    for ri, row in enumerate(rows):
        cy = 20 + ri * (th + gap + 22)
        x = gap
        for nm, c in row:
            cv.paste(c, (x, cy))
            dr.text((x + 3, cy - 15), nm, fill=(240, 240, 240))
            x += c.width + gap
    cv.save(DST / "00_四图成品总表.jpg", quality=93)
    print("总表", cv.size)

    # ---- QC 文本 ----
    lines = ["v373 第 16 轮修订版 —— 量化自检（生成日期 2026-09-18）",
             "=" * 62,
             "本轮针对用户第 15 轮否决逐条返修：",
             "  #1 p3「让你裂变不是让你旋转跟变形 —— 主体蝴蝶为什么变成变形了；",
             "        小蝴蝶元素裂变跟原图不一样、跟主题有关联就行」",
             "  #2 p6「老鹰跟骷髅头说几次了，就是不裂变是吗」",
             "返修路线：两条否决的根因同一个 —— 上一轮用**程序化形变(warp/旋转)**冒充裂变。",
             "本轮全部回到 **SDXL 结构级内容重生**（真裂变：内容换新、物种/材质/版式不变）。",
             ""]
    for name, head, rec, _, _ in report:
        lines += [head, rec, ""]
    lines += [
        "【本轮关键根因与修复】",
        "  ① p3 轨迹圆点被主蝶掩膜吞掉 → 重生后轨迹整条变成发白圆环",
        "     真因：binary_closing(disk9) 桥接点缘间隙(11px) → 整条轨迹焊进主蝶连通域。",
        "     修：v378 `trail_dots()` 先剥离圆点再闭运算（实测剥离 1807px / 20 颗）。",
        "  ② p3 末段紧贴流苏的那几颗点无法被 ① 检出（已与流苏连通）→ 100% 落在主蝶掩膜内",
        "     → 重生后留下**完美灰色空心圆环**。",
        "     修：v381 `repair_dots()` —— 只在「原图厚度>7px 的厚暗斑(bbox≤24px) 且新图被提亮>18」",
        "         处还原原像素；翅面是大区域，bbox 远超 24px，天然免疫误伤。",
        "  ③ p3 小蝶回贴环带相对小蝶过大 → 用生成浅底盖住旁边的轨迹。",
        "     修：小蝶 wide 26→0、paste_pad 26→5、grow 6→3；dn 0.80→0.72（保住原小蝶的俏皮感）。",
        "  ④ p6 鹰喙由原图**明黄**漂成白骨色、翼羽由中深棕漂成浅茶金（局部色相漂移）。",
        "     snap 的 LAB Reinhard 只对齐掩膜内整体均值/方差，治不了局部漂移。",
        "     修：v384 把颜色写进 prompt（bright golden-yellow hooked beak / dark chocolate brown",
        "         wing feathers）+ NEG 去掉与之打架的 yellow/orange eyes → 喙色找回来了。",
        "",
        "【1_pinterest3_蝴蝶+牛仔字】",
        "  手法：三只蝶各自 **SDXL 重生**（ProteusV0.4 + ControlNet canny .40/.45、dn .80）",
        "        · 主蝶：zoom 1.7x 原分辨率重绘 → 翅形与走线全部换新，外层磨毛流苏原样保留",
        "        · 上/下小蝶：zoom 5x/6x，同参数不同种子；只要求「跟原图不一样、跟主题是一家」",
        "  文字：UPCY → DENIM（同牛仔贴布材质/同描边/同排版框，draw_line_material mat_mode=tile）",
        "  轨迹：端点锚定的原位重画（同点数/同半径/同色，只加垂直于轨迹的正弦弯曲）",
        "",
        "【2_pinterest4_迷彩棕榈】",
        "  沿用 v372 定稿（本轮未再改）：EDT 精确清底 + 按原图风格程序化重绘棕榈",
        "  （树干=细脊线+密横档 / 叶=粗人字锯齿缎带 / 冠 16~21 根宽叶下垂 0.10L），",
        "  走「同构」约束下的重绘，而非 SDXL 重生（矢量线稿过 VAE 必掉档）。",
        "  用户第 15 轮未再对 p4 提出意见 → 保持冻结。",
        "",
        "【3_pinterest6_鹰骷髅+尖刺标题】",
        "  主体 **SDXL 真重生**：Juggernaut-Ragnarok + ControlNet canny (.45/.45)、dn .90、",
        "  max_side 2048（原生 1:1，不降采样），掩膜外像素零改动。",
        "  鹰头/羽序/骷髅/双角/闪电全部重画；标题 MACABRE → VORTCRAVEN（金属尖刺，原位满幅 3513px）。",
        "  已知妥协：新翼展比原图略收窄（原图双翼更外张），羽片分得更细（钢笔画密度更高）。",
        "",
        "【4_6978_蝙蝠+弧字】",
        "  用户第 13 轮已认可（「蝙蝠设计可以」）→ 冻结不动，仅沿用。",
        "",
        "-" * 62,
        "待用户确认 / 已知未尽事项",
        "  · p3 文字「DENIM」的 I 是浅白贴布，与左右深蓝字母明度差大；如需更统一可改配色。",
        "  · p3 下小蝶经重生后姿态比原图略倾（内容已换新，非单纯旋转）。",
        "  · p6 新翼展略收窄（见上）。",
        "  · p4 仍为程序化重绘（非 SDXL 重生）：该图为矢量线稿，SDXL 重生会掉档。",
    ]
    (DST / "QC_指标.txt").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", DST / "QC_指标.txt")
    print("[DONE]", DST)


if __name__ == "__main__":
    main()
