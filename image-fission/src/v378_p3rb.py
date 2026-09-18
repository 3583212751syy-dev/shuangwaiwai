"""v378：p3 主蝶 / 小蝶 **SDXL 结构级重生**。

用户第 16 轮：「让你裂变不是让你旋转跟变形就行了 …… 小蝴蝶元素裂变跟原图不一样跟主题有关联就行」。
上一轮（v370 轮廓再设计）只改了轮廓 → 用户判定"变形"。本轮走 SDXL 重生：
主体掩膜内重绘成**另一只牛仔蝴蝶**（新翅形 + 新走线），掩膜外（浅灰底 / UPCY 字母）零改动。

用法：
  python src/v378_p3rb.py main     # 主蝶 3 个变体
  python src/v378_p3rb.py small    # 两只小蝶（用主蝶定稿参数，各自不同种子）
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v342_rebirth import rebirth_subject, _lum, _disk        # noqa: E402

OUT = ROOT / "jobs" / "v378_p3rb"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (3).jpg")
CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"

POS = ("a denim butterfly applique patch, dust blue and faded navy denim fabric, "
       "satin stitch embroidery wing veins, visible woven denim texture and stitching, "
       "raw frayed fabric edge, symmetrical spread wings, different wing silhouette "
       "and a different embroidery pattern, sharp focus product photo, plain light gray background, "
       "high detail, crisp")
NEG = ("text, letters, words, watermark, blurry, low quality, illustration, drawing, painting, "
       "cartoon, sketch, colorful, red, green, yellow, orange, pink, glow, "
       "extra wings, extra legs, extra antenna, deformed, cropped, shadow, border, frame")


# ------------------------------------------------------------------ 掩膜
def _fg(img):
    a = np.asarray(img, np.float32)
    sat = a.max(2) - a.min(2)
    return (_lum(a) < 210) | (sat > 26)


def _cc_in(fgc, box, min_px=300):
    x0, y0, x1, y1 = box
    sub = np.zeros_like(fgc)
    sub[y0:y1, x0:x1] = fgc[y0:y1, x0:x1]
    lab, n = ndi.label(sub, structure=np.ones((3, 3), bool))
    if n == 0:
        return None
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    k = int(np.argmax(sz))
    if sz[k] < min_px:
        return None
    return ndi.binary_fill_holes(lab == k + 1)


def trail_dots(fg):
    """找出「虚线飞行轨迹」圆点，从 fg 里剥掉。

    🔴29 根因（实测）：圆点圆心间距 ~20px、半径 ~4.6px → 点缘间隙 ~11px，
    `binary_closing(disk 9)` 能桥接 ≤18px 间隙 → 整条轨迹被**焊进主蝶连通域**
    （实测 30/35 个点落在 main 掩膜内）。主蝶重生时把轨迹当翅膀一起重画 →
    「轨迹变成一排发白圆环」。判别：面积 20~260、bbox 4~20px、圆度 >0.45 的孤立圆斑。
    """
    lab, n = ndi.label(fg, structure=np.ones((3, 3), bool))
    out = np.zeros_like(fg)
    if n == 0:
        return out
    for i, sl in enumerate(ndi.find_objects(lab)):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if not (4 <= w <= 20 and 4 <= h <= 20):
            continue
        m = lab[sl] == (i + 1)
        ar = int(m.sum())
        if not (20 <= ar <= 260):
            continue
        if ar / (math.pi * (0.25 * (w + h)) ** 2) < 0.45:
            continue
        out[sl] |= m
    return out


def build_masks(img):
    fg = _fg(img)
    fg[:360, :] = False                        # 剔除 UPCY 字母带
    dots = trail_dots(fg)
    if dots.any():
        fg = fg & (~ndi.binary_dilation(dots, structure=_disk(2)))
    fgc = ndi.binary_closing(fg, structure=_disk(9))
    main = _cc_in(fgc, (0, 380, 736, 1130), 50000)
    top = _cc_in(fgc, (420, 330, 600, 500), 3000)
    bot = _cc_in(fgc, (200, 1140, 360, 1290), 3000)
    print(f"[masks] 轨迹圆点 {int(dots.sum())}px 已从前景剥离")
    return main, top, bot


def grow(m, k):
    return ndi.binary_dilation(m, structure=_disk(k))


# ------------------------------------------------------------------ 带缩放的贴回
def rebirth_zoom(img, mask, zoom=2.0, margin=40, protect_full=None, paste_pad=26, **kw):
    a = np.asarray(img, np.float32)
    ys, xs = np.where(mask)
    x0 = max(0, xs.min() - margin); x1 = min(img.width, xs.max() + 1 + margin)
    y0 = max(0, ys.min() - margin); y1 = min(img.height, ys.max() + 1 + margin)
    crop = img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    iw = max(8, (int(round(cw * zoom)) // 8) * 8)
    ih = max(8, (int(round(ch * zoom)) // 8) * 8)
    up = crop.resize((iw, ih), Image.LANCZOS)
    m_up = np.asarray(Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255), "L")
                      .resize((iw, ih), Image.NEAREST)) > 127
    prot = None
    if protect_full is not None:
        prot = np.asarray(Image.fromarray((protect_full[y0:y1, x0:x1].astype(np.uint8) * 255), "L")
                          .resize((iw, ih), Image.NEAREST)) > 127
    print(f"[zoom] crop {cw}x{ch} -> inj {iw}x{ih}  mask {int(m_up.sum())}px")
    res_up = rebirth_subject(up, m_up, POS, NEG, ckpt="ProteusV0.4.safetensors",
                             ipa_weight=0.0, color_match=0.85, cn_name=CN, cn_pre="canny",
                             margin=24, protect=prot, max_side=0, **kw)
    res = res_up.resize((cw, ch), Image.LANCZOS)
    pad = paste_pad + int(kw.get("wide", 0))
    al = np.clip(ndi.gaussian_filter(grow(mask[y0:y1, x0:x1], pad).astype(np.float32), 3.0), 0, 1)
    out = a.copy()
    r = np.asarray(res, np.float32)
    out[y0:y1, x0:x1] = a[y0:y1, x0:x1] * (1 - al[..., None]) + r * al[..., None]
    # 🔴29：回贴环带内若压到「非本主体」的元素（虚线轨迹点 / 字母 / 另两只蝶），
    #        一律从原图**强制还原**。回归贴 alpha 只保证掩膜内，环带是无条件贴的，
    #        所以必须在这里兜底，否则轨迹永远会被生成的浅底吃掉。
    if protect_full is not None:
        _p = protect_full[y0:y1, x0:x1]
        if _p.any():
            out[y0:y1, x0:x1][_p] = a[y0:y1, x0:x1][_p]
            print(f"[{kw.get('tag', '?')}] 保护还原 {int(_p.sum())}px")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


MAIN_VARIANTS = [
    ("M1_dn80_cn40e45", dict(denoise=0.80, cn_strength=0.40, cn_end=0.45, seed=101)),
    ("M2_dn88_cn30e38", dict(denoise=0.88, cn_strength=0.30, cn_end=0.38, seed=202)),
    ("M3_dn74_cn55e55", dict(denoise=0.74, cn_strength=0.55, cn_end=0.55, seed=303)),
]


def run_main():
    img = Image.open(SRC).convert("RGB")
    main, top, bot = build_masks(img)
    print("main", int(main.sum()), "top", int(top.sum()), "bot", int(bot.sum()))
    vis = np.asarray(img).copy()
    vis[grow(main, 3)] = [255, 60, 60]
    Image.fromarray(vis).save(str(OUT / "p3_main_mask.jpg"), quality=92)
    keep = [("ORIG", img)]
    for nm, kw in MAIN_VARIANTS:
        out = rebirth_zoom(img, main, zoom=1.7, margin=60, wide=70,
                           bg_gate=216.0, subject_dark=True, grow=12, tag=nm, **kw)
        out.save(str(OUT / f"p3main_{nm}.jpg"), quality=94)
        keep.append((nm, out))
    _sheet(keep, (10, 380, 736, 1160), "S_p3main_cands.jpg")


def run_small():
    img = Image.open(SRC).convert("RGB")
    _, top, bot = build_masks(img)
    prot = np.zeros((img.height, img.width), bool)
    prot[:368, :] = True                       # 字母带严格保护
    cfg = dict(denoise=float(sys.argv[2]) if len(sys.argv) > 2 else 0.80,
               cn_strength=float(sys.argv[3]) if len(sys.argv) > 3 else 0.40,
               cn_end=float(sys.argv[4]) if len(sys.argv) > 4 else 0.45)
    keep = [("ORIG", img)]
    for i, (m, box, zoom, seed) in enumerate(
            [(top, (400, 320, 620, 520), 5.0, 911),
             (bot, (180, 1130, 380, 1300), 6.0, 922)]):
        out = rebirth_zoom(img, m, zoom=zoom, margin=26, wide=26,
                           bg_gate=216.0, subject_dark=True, grow=6,
                           tag=f"small{i}", seed=seed, **cfg)
        out.save(str(OUT / f"p3small{i}.jpg"), quality=94)
        keep.append((f"small{i}", out))
    _sheet(keep, (100, 300, 660, 1308), "S_p3small_cands.jpg", th=760)


def _sheet(items, box, name, th=470):
    cols = []
    for nm, im in items:
        c = im.crop(box)
        w = int(c.width * th / c.height)
        cols.append((nm, c.resize((max(1, w), th), Image.LANCZOS)))
    gap = 8
    Wt = sum(c[1].width for c in cols) + gap * (len(cols) + 1)
    cv = Image.new("RGB", (Wt, th + 24), (25, 25, 28))
    dr = ImageDraw.Draw(cv)
    x = gap
    for nm, c in cols:
        cv.paste(c, (x, 20)); dr.text((x + 3, 5), nm, fill=(235, 235, 235))
        x += c.width + gap
    cv.save(str(OUT / name), quality=93)
    print("sheet", name, cv.size)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    (run_main if mode == "main" else run_small)()
