"""v380：p3 主蝶 —— **保流苏**的 SDXL 重生。

v378 重生把蝴蝶换成了另一只（好），但把原图最有辨识度的**狂野磨毛流苏边**重生成了
整齐锯齿轮廓 —— 那是本设计语言的签名（用户在第 16 轮说 p4 时同义反复强调过
"根据原图风格去裂变"）。本脚本：
  ① 重生只发生在**躯干核**（wide=0，剪影留在原位附近）；
  ② 把**原图的流苏丝**（环带内非背景像素）抠成 RGBA 覆盖层，盖在重生结果之上
     → 新蝶翼 + 原流苏，风格语言一致、内容已换新。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v378_p3rb import build_masks, grow, rebirth_zoom, SRC, _lum, _disk   # noqa: E402

OUT = ROOT / "jobs" / "v380_p3fringe"
OUT.mkdir(parents=True, exist_ok=True)


def bg_color(img):
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    corners = np.concatenate([a[:20, :20].reshape(-1, 3), a[:20, -20:].reshape(-1, 3),
                              a[-20:, :20].reshape(-1, 3), a[-20:, -20:].reshape(-1, 3)])
    return np.median(corners, 0)


def overlay_fringe(orig, gen, mask, core_shrink=34):
    """把原图环带内的「流苏丝」按软 alpha 盖到 gen 上（只覆盖丝线，不覆盖实布）。

    alpha 曲线：与底色差 d 小 → 近乎透明（那是底色，交给重生图）；
    d 中段（20~110，= 丝线）→ 接近 1（取原图，保住狂野边）；
    d 很大（>130，= 实心深色布）→ 回落 0（交给重生图，避免旧翅面花纹在边缘成环）。
    """
    o = np.asarray(orig, np.float32)
    g = np.asarray(gen, np.float32)
    bg = bg_color(orig)
    ring = mask & (~ndi.binary_erosion(mask, structure=_disk(core_shrink)))
    ring_soft = ndi.gaussian_filter(ring.astype(np.float32), 4.0)
    d = np.abs(o - bg[None, None, :]).max(2)
    a_silk = np.clip((d - 8.0) / 38.0, 0.0, 1.0) * (1.0 - np.clip((d - 130.0) / 80.0, 0.0, 1.0))
    a_silk = ndi.gaussian_filter(a_silk, 0.7)
    al = (a_silk * ring_soft)[..., None]
    out = g * (1 - al) + o * al
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


CFGS = [
    ("F1_M1core", dict(denoise=0.80, cn_strength=0.40, cn_end=0.45, seed=101)),
    ("F2_M2core", dict(denoise=0.88, cn_strength=0.30, cn_end=0.38, seed=202)),
]


def main():
    img = Image.open(SRC).convert("RGB")
    main_m, top, bot = build_masks(img)
    vis = np.asarray(img).copy()
    ring = main_m & (~ndi.binary_erosion(main_m, structure=_disk(30)))
    vis[ring] = [255, 80, 80]
    Image.fromarray(vis).save(str(OUT / "fringe_ring.jpg"), quality=92)
    keep = [("ORIG", img)]
    for nm, kw in CFGS:
        base = rebirth_zoom(img, main_m, zoom=1.7, margin=60, wide=0, grow=10,
                            tag=nm, **kw)
        base.save(str(OUT / f"p3main_{nm}_core.jpg"), quality=94)
        fin = overlay_fringe(img, base, main_m, core_shrink=34)
        fin.save(str(OUT / f"p3main_{nm}_fringe.jpg"), quality=94)
        keep += [(nm + "_core", base), (nm + "_fringe", fin)]
    _sheet(keep, (0, 420, 736, 1160), "S_p3fringe.jpg", 640)


def _sheet(items, box, name, th=470):
    cols = []
    for nm, im in items:
        c = im.crop(box); w = int(c.width * th / c.height)
        cols.append((nm, c.resize((max(1, w), th), Image.LANCZOS)))
    gap = 6
    Wt = sum(c[1].width for c in cols) + gap * (len(cols) + 1)
    cv = Image.new("RGB", (Wt, th + 22), (25, 25, 28)); dr = ImageDraw.Draw(cv); x = gap
    for nm, c in cols:
        cv.paste(c, (x, 19)); dr.text((x + 3, 4), nm, fill=(235, 235, 235)); x += c.width + gap
    cv.save(str(OUT / name), quality=94); print("sheet", name, cv.size)


if __name__ == "__main__":
    main()
