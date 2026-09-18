"""v381：p3 三只蝴蝶全部 SDXL 真重生 → 合成 p3 底图（供 make_v329 文字流程使用）。

- 主蝶：v380 定稿参数（dn.80/cn.40/e45 + 保流苏覆盖层）
- 上/下小蝶：同参数不同种子；用户标准「跟原图不一样跟主题有关联就行」
输出：jobs/v381_p3final/p3_base.jpg（= 原图 + 三只蝶换新，其余像素零改动）

用法：
  python src/v381_p3final.py             # 出 p3_base.jpg（BOT_SEED 可选，默认 922）
  python src/v381_p3final.py sweep       # 下小蝶多候选择优（出 s3_bot_<seed>.jpg + 对比表）
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v378_p3rb import build_masks, rebirth_zoom, SRC, _fg, _disk, _lum, trail_dots  # noqa: E402
from v380_p3fringe import overlay_fringe                                           # noqa: E402

OUT = ROOT / "jobs" / "v381_p3final"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_KW = dict(denoise=0.80, cn_strength=0.40, cn_end=0.45, seed=101)
BOT_SEED = int(os.environ.get("BOT_SEED", "922"))


def protect_for(fg_all, m, letters, dots=None):
    """保护「非当前主体」的一切前景元素（虚线轨迹圆点 / 另两只蝶 / 字母）。

    根因（实测）：重生噪声掩膜 = dilate(mask, grow+6)，主体边缘外扩 16~36px，
    会把**紧邻的虚线轨迹圆点**一并重绘成背景 → 轨迹断成发白虚影。
    """
    others = fg_all if dots is None else (fg_all | dots)
    others = others & (~ndi.binary_dilation(m, structure=_disk(4)))
    return ndi.binary_dilation(others, structure=_disk(4)) | letters


def repair_dots(orig, gen):
    """🔴30 轨迹**末段紧贴流苏**的那几颗点会被 closing 焊进主蝶掩膜 → 无法被
    `trail_dots()` 检出 → 重生后在原位留下**完美灰色空心圆环**（点被生成浅底顶掉，
    只有抗锯齿边被羽化 alpha 部分混回）。

    判别式（只认「原图里尺度过大的厚暗斑 + 新图被明显提亮」）：
      · `lum(orig) < 185` 二值化后 `erode(disk 3)` → 只有**厚度 >7px** 的实心区残留
        （流苏细丝 2~5px 全部消失，圆点直径 ~13px 完整保留）；
      · 保留 bbox ≤24px、面积 10~400px 的孤立厚斑 = 圆点候选；
      · 候选外扩 3px 内，凡 `lum(gen) - lum(orig) > 18`（= 被生成内容提亮覆盖）→ 还原原像素。
    绝不会误伤新翅面：翅面是**大区域**厚暗，bbox 远超 24px 被直接排除。
    """
    o = np.asarray(orig, np.float32)
    g = np.asarray(gen, np.float32)
    lo, lg = _lum(o), _lum(g)
    core = ndi.binary_erosion(lo < 185, structure=_disk(3))
    lab, n = ndi.label(core, np.ones((3, 3), bool))
    cand = np.zeros_like(core)
    for i, sl in enumerate(ndi.find_objects(lab)):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if w > 24 or h > 24:
            continue
        m = lab[sl] == (i + 1)
        if not (10 <= int(m.sum()) <= 400):
            continue
        cc = np.zeros_like(core)
        cc[sl] |= m
        cand |= ndi.binary_dilation(cc, structure=_disk(4))
    rep = ndi.binary_dilation(cand, structure=_disk(3)) & ((lg - lo) > 18)
    if not rep.any():
        return gen, 0
    out = g.copy()
    out[rep] = o[rep]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), int(rep.sum())


def _stage_main_top(img):
    """跑主蝶 + 上小蝶，返回中间结果与掩膜。"""
    main_m, top, bot = build_masks(img)
    fg_all = _fg(img)
    fg_all[:360, :] = False
    dots = trail_dots(fg_all)
    print(f"[dots] 轨迹圆点 {int(dots.sum())}px 纳入保护池")
    letters = np.zeros((img.height, img.width), bool)
    letters[:368, :] = True                                # UPCY 字母带严格保护

    cur = rebirth_zoom(img, main_m, zoom=1.7, margin=60, wide=0, grow=10,
                       protect_full=protect_for(fg_all, main_m, letters, dots),
                       tag="main", **MAIN_KW)
    cur = overlay_fringe(img, cur, main_m, core_shrink=34)
    cur, _ = repair_dots(img, cur)
    cur.save(str(OUT / "s1_main.jpg"), quality=95)

    cur = rebirth_zoom(cur, top, zoom=5.0, margin=20, wide=0, paste_pad=5, grow=3,
                       protect_full=protect_for(fg_all, top, letters, dots),
                       tag="top", seed=911, denoise=0.72, cn_strength=0.45, cn_end=0.50)
    cur = overlay_fringe(img, cur, top, core_shrink=7)
    cur, _ = repair_dots(img, cur)
    cur.save(str(OUT / "s2_top.jpg"), quality=95)
    return img, cur, bot, fg_all, dots, letters


def do_bot(img, cur, bot, fg_all, dots, letters, seed):
    """下小蝶重生（⚠️ wide=0 / paste_pad=5 / grow=3：小蝶尺寸小，环带相对就是巨环，
    首版 wide=26 把旁边的虚线轨迹用生成浅底盖住 → 轨迹变成一排发白圆斑）。

    ⚠️ dn 降到 0.72：原小蝶是「拼布+金色铆钉」的俏皮感，dn 0.80 会把它重生成一枚
    普通刺绣蝶，失掉原设计语言的趣味。
    """
    out = rebirth_zoom(cur, bot, zoom=6.0, margin=20, wide=0, paste_pad=5, grow=3,
                       protect_full=protect_for(fg_all, bot, letters, dots),
                       tag="bot", seed=seed, denoise=0.72, cn_strength=0.45, cn_end=0.50)
    out = overlay_fringe(img, out, bot, core_shrink=6)
    out, n = repair_dots(img, out)
    print(f"[bot seed={seed}] 轨迹修复 {n}px")
    return out


def main():
    img = Image.open(SRC).convert("RGB")
    img, cur, bot, fg_all, dots, letters = _stage_main_top(img)
    cur = do_bot(img, cur, bot, fg_all, dots, letters, BOT_SEED)
    cur.save(str(OUT / "p3_base.jpg"), quality=96)
    print("saved", OUT / "p3_base.jpg")

    items = [("ORIG", img), ("main", Image.open(str(OUT / 's1_main.jpg'))),
             ("+top", Image.open(str(OUT / 's2_top.jpg'))),
             ("+bot(FINAL)", cur)]
    _sheet(items, (0, 330, 736, 1308), "S_p3_steps.jpg", th=640)


def sweep():
    """下小蝶多种子择优：姿态倾斜度是主要观感项，出对比表人工/量化选。"""
    seeds = [int(s) for s in os.environ.get("BOT_SEEDS", "922,1207,3311,4242").split(",")]
    img = Image.open(SRC).convert("RGB")
    img, cur, bot, fg_all, dots, letters = _stage_main_top(img)
    items = [("ORIG", img)]
    for s in seeds:
        o = do_bot(img, cur, bot, fg_all, dots, letters, s)
        o.save(str(OUT / f"s3_bot_{s}.jpg"), quality=95)
        items.append((f"bot{s}", o))
    _sheet(items, (150, 880, 420, 1140), "S_p3bot_sweep.jpg", th=520)


def _sheet(items, box, name, th=640):
    cols = []
    for nm, im in items:
        c = im.crop(box)
        w = int(c.width * th / c.height)
        cols.append((nm, c.resize((max(1, w), th), Image.LANCZOS)))
    gap = 8
    cv = Image.new("RGB", (sum(c[1].width for c in cols) + gap * (len(cols) + 1), th + 22),
                   (25, 25, 28))
    dr = ImageDraw.Draw(cv)
    x = gap
    for nm, c in cols:
        cv.paste(c, (x, 19))
        dr.text((x + 3, 4), nm, fill=(240, 240, 240))
        x += c.width + gap
    cv.save(str(OUT / name), quality=95)
    print("sheet", name, cv.size)


if __name__ == "__main__":
    (sweep if len(sys.argv) > 1 and sys.argv[1] == "sweep" else main)()
