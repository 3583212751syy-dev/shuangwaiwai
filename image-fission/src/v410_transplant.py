# -*- coding: utf-8 -*-
"""v410_transplant.py — 把新版「鹰踩角骷髅」按原主体位贴回 p6（第 29 轮）

流程（背景纯黑 ⇒ 腾空区自动变黑，无需 inpaint，🔴25）
  ① 原图主体掩膜（`mask_p6`）→ 扩张 6px 得"腾空区"
  ② 腾空区整块抹黑（不是色块遮盖 —— 因为版面本来就是纯黑底）
  ③ 新生成图按**自有主体 bbox** 缩放，**contain** 进原主体 bbox（不越界、不吃射线）
  ④ 用新图自有的主体掩膜（羽化 2px）贴回
  ⑤ LAB Reinhard 把新主体的`均值/方差`拉回原主体配色（🔴4 / 🔴20 不用 IPAdapter 的替代）
  ⑥ `snap_p6` 做亮度直方图规定化，找回平涂四阶
  ⑦ 标题带（y<1500）强制还原原图 —— 新字标仍由 v407 通道贴回（🔴37 两步走）

用法
  python src/v410_transplant.py --pick jobs/v410_t2i/t2i_T1_s111.jpg --tag v410a
  python src/v410_transplant.py --pick <图> --tag v410b --fit cover --scale 1.06
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from v342_rebirth import _disk, _lum, _match_region, mask_p6     # noqa: E402
from v377_p6rb import snap_p6                                     # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
OUT = ROOT / "jobs" / "v410_transplant"
OUT.mkdir(parents=True, exist_ok=True)
UP = ROOT / "jobs" / "v410_up"
TITLE_LINE = 1500
UPSCALE_MODEL = "4x-UltraSharp.pth"

from v342_rebirth import CUI_IN, comfy_submit, _fetch          # noqa: E402


def upscale(pick: Path, model=UPSCALE_MODEL) -> Path:
    """4x-UltraSharp 放大（1024→4096），否则 3 倍上采样贴回必糊（🔴22）。"""
    UP.mkdir(parents=True, exist_ok=True)
    dst = UP / f"{pick.stem}_4x.jpg"
    if dst.exists():
        return dst
    name = f"v410up_{pick.stem}.png"
    Image.open(pick).convert("RGB").save(str(CUI_IN / name))
    wf = {
        "1": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model}},
        "2": {"class_type": "LoadImage", "inputs": {"image": name}},
        "3": {"class_type": "ImageUpscaleWithModel",
              "inputs": {"upscale_model": ["1", 0], "image": ["2", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0],
                                                    "filename_prefix": f"v410up_{pick.stem}"}},
    }
    im = None
    for _n, o in comfy_submit(wf).items():
        if "images" in o:
            im = _fetch(o); break
    if im is None:
        raise RuntimeError("upscale failed")
    im.convert("RGB").save(dst, quality=96)
    print(f"[v410] upscale {pick.name} → {im.size}")
    return dst


def subject_mask(img: Image.Image) -> np.ndarray:
    """新图的"自有主体"掩膜。

    ⚠️ 第 29 轮踩到的坑：文生图底色是**纯黑**，所以判据必须是"**非黑即主体**"。
    照抄 `mask_p6` 的 `lum>55` 会把深棕鹰身（实测 lum 30~55）当背景剔掉 →
    贴回后鹰身整块留黑，只剩白头/白翅/白骷髅的碎片（"金丝"感）。
    取 `lum>22` + 闭 3 / 开 3 去噪点 + 留大分量 + 膨胀 3。
    """
    a = np.asarray(img, np.float32)
    lum = _lum(a)
    nb = lum > 22.0
    m = ndi.binary_closing(nb, structure=_disk(3))
    m = ndi.binary_opening(m, structure=_disk(3))
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n:
        sz = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, bool)
        keep[1:] = sz[1:] >= 2000
        m = keep[lab]
    return ndi.binary_dilation(m, structure=_disk(3))


def _bbox(m: np.ndarray):
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def erase_mask(orig: Image.Image, disk_thin: int = 6, min_area: int = 15000) -> np.ndarray:
    """**擦除掩膜**：要擦掉的原主体（含深棕翼），但要**放过细长的白色闪电**。

    ⚠️ 不能直接用 `mask_p6`（`lum>55`）—— 它漏掉深棕鹰身（实测 lum 30~55），
    残留后会在新主体周围显出"金丝"；也不能无脑擦亮部（会把原图闪电一起擦掉，
    新主体又自带另一套射线 → 接缝处断裂）。
    做法：低阈值取全部非黑 → 闭 5 → **开 disk(6) 去掉厚度 <12px 的细长闪电** →
    只留 ≥15000px 的实体分量 → 膨胀 6。
    """
    a = np.asarray(orig, np.float32)
    lum = _lum(a)
    low = lum > 22.0
    m = ndi.binary_closing(low, structure=_disk(5))
    m = ndi.binary_opening(m, structure=_disk(disk_thin))
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n:
        sz = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, bool)
        keep[1:] = sz[1:] >= min_area
        m = keep[lab]
    return ndi.binary_dilation(m, structure=_disk(6))


def transplant(orig: Image.Image, mask: np.ndarray, pick: Path, *,
               fit="contain", scale=1.0, grow=6, feather=2.0,
               color_alpha=0.85, erase=None, verbose=True) -> Image.Image:
    W, H = orig.size
    tb = _bbox(mask)
    tw_, th_ = tb[2] - tb[0], tb[3] - tb[1]

    new = Image.open(pick).convert("RGB")
    nm = subject_mask(new)
    nb = _bbox(nm)
    nw, nh = nb[2] - nb[0], nb[3] - nb[1]

    k = (max(tw_ / nw, th_ / nh) if fit == "cover" else min(tw_ / nw, th_ / nh)) * scale
    nw2, nh2 = max(1, int(round(nw * k))), max(1, int(round(nh * k)))
    sub = new.crop(nb).resize((nw2, nh2), Image.LANCZOS)
    sm = Image.fromarray((nm[nb[1]:nb[3], nb[0]:nb[2]] * 255).astype(np.uint8), "L")
    sm = sm.resize((nw2, nh2), Image.LANCZOS)

    px = (tb[0] + tb[2]) // 2 - nw2 // 2
    py = (tb[1] + tb[3]) // 2 - nh2 // 2
    if verbose:
        print(f"[v410] 原主体 bbox {tb} {tw_}x{th_} | 新主体 {nb} {nw}x{nh} k={k:.3f} "
              f"→ 贴于 ({px},{py}) 尺寸 {nw2}x{nh2}")

    # 腾空区抹黑（纯黑底图 ⇒ 等价于把旧主体擦掉；**放过闪电**，见 erase_mask）
    out = np.array(orig).copy()
    hole = erase if erase is not None else ndi.binary_dilation(mask, _disk(grow))
    out[hole] = 0
    canvas = np.zeros((H, W, 3), np.uint8)
    a = np.asarray(sub, np.uint8)
    canvas[py:py + nh2, px:px + nw2] = a
    am = np.zeros((H, W), np.float32)
    am[py:py + nh2, px:px + nw2] = np.asarray(sm, np.float32) / 255.0

    # 羽化
    am = np.asarray(Image.fromarray((am * 255).astype(np.uint8), "L")
                    .filter(ImageFilter.GaussianBlur(feather)), np.float32) / 255.0
    am = np.clip(am * 1.35, 0, 1)                    # 提高实心度，抵掉模糊带来的半透明
    out = (out.astype(np.float32) * (1 - am[..., None])
           + canvas.astype(np.float32) * am[..., None])

    # 配色对齐（🔴4）
    if color_alpha and float(color_alpha) > 0:
        rc = _match_region(out, np.asarray(orig, np.float32),
                           am > 0.5, alpha=float(color_alpha))
        out = np.asarray(rc, np.float32)
    # 标题带保护
    tb_row = np.arange(H) < TITLE_LINE
    out[tb_row, :] = np.asarray(orig, np.float32)[tb_row, :]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def run(pick: Path, tag: str, out_dir=None, snap=True, up=True, **kw) -> Path:
    out_dir = Path(out_dir) if out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    src = upscale(pick) if up else pick
    orig = Image.open(SRC).convert("RGB")
    mask = mask_p6(orig)
    em = erase_mask(orig) if up else None
    full = transplant(orig, mask, src, erase=em, **kw)
    if snap:
        full = snap_p6(orig, full, mask)
    dst = out_dir / f"p6_{tag}.jpg"
    full.save(dst, quality=94)
    print(f"[v410] → {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True, help="新图路径[,...]")
    ap.add_argument("--tags", default=None, help="标签[,...]（缺省用文件名）")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fit", default="contain", choices=["contain", "cover"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--grow", type=int, default=6)
    ap.add_argument("--feather", type=float, default=2.0)
    ap.add_argument("--nosnap", action="store_true")
    ap.add_argument("--no-up", action="store_true", help="不做 4x 放大（会糊，仅调试用）")
    a = ap.parse_args()
    picks = [Path(x.strip()) for x in a.picks.split(",") if x.strip()]
    tags = [x.strip() for x in a.tags.split(",")] if a.tags else [p.stem for p in picks]
    for p, t in zip(picks, tags):
        if not p.exists():
            print(f"! 缺 {p}")
            continue
        run(p, t, out_dir=a.out_dir, snap=not a.nosnap, up=not a.no_up, fit=a.fit,
            scale=a.scale, grow=a.grow, feather=a.feather)


if __name__ == "__main__":
    main()
