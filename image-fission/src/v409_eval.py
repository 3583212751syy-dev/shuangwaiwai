# -*- coding: utf-8 -*-
"""v409_eval.py —— p6 主体裂变候选评估（量化 + 拼版），第 29 轮

量化只作辅助（🔴 目检才是终判），但本轮的**主判据**是：

  · `wcorr` **逐行宽度相关** —— 对每个 y 行统计"亮部像素数"得到轮廓宽度曲线，
    与原图同一曲线求相关。**这是"轮廓是否与原图同构"的直接量化**：
      - v342 时代的失败案例 IoU 0.867 / 逐行宽度相关 0.99+ → 用户读作"没变化"；
      - 本轮目标：wcorr 明显低于 0.90（宽度剖面被重排，而不只是缩放）。
  · `sd` 结构差异（1 - MSE/255²）—— 内容换了多少
  · `iou` 亮度剪影 IoU
  · `yellow` 颅骨窗黄像素占比（🔴35：不得高于原图）
  · `edge` 边缘密度（过低 = 糊）

用法：
  python src/v409_eval.py --cands "标签=路径,标签=路径" --out jobs/_probe/sheet.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from v342_rebirth import _lum, mask_p6                      # noqa: E402

ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
BOX = (420, 1180, 3120, 4760)          # 主体评估窗（避开标题带）
YWIN = (slice(2800, 4700), slice(800, 2800))   # 颅骨窗（黄像素红线）


def metrics(im: Image.Image, o: Image.Image, m: np.ndarray, box=BOX) -> dict:
    if im.size != o.size:
        im = im.resize(o.size, Image.LANCZOS)
    a = np.asarray(im, np.float32)
    b = np.asarray(o, np.float32)
    x0, y0, x1, y1 = box
    g1 = _lum(a)[y0:y1, x0:x1]
    g2 = _lum(b)[y0:y1, x0:x1]
    mm = m[y0:y1, x0:x1]

    mse = float(((g1[mm] - g2[mm]) ** 2).mean())
    sd = 1.0 - mse / (255.0 ** 2)

    t = 55.0
    s1 = (g1 > t) & mm
    s2 = (g2 > t) & mm
    iou = float((s1 & s2).sum()) / max(1.0, float((s1 | s2).sum()))

    # 逐行宽度相关（轮廓同构度）
    ra = (g1 > t).sum(axis=1).astype(np.float64)
    rb = (g2 > t).sum(axis=1).astype(np.float64)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    wcorr = float((ra * rb).sum() / den) if den > 1e-6 else 1.0

    aw = a[YWIN]
    y = ((aw[..., 0] > 150) & (aw[..., 1] > 110) & (aw[..., 2] < 110)
         & ((aw[..., 0] - aw[..., 2]) > 60))
    gx = np.abs(np.diff(g1, axis=1)).mean()
    gy = np.abs(np.diff(g1, axis=0)).mean()
    return dict(sd=sd, iou=iou, wcorr=wcorr, yellow=float(y.mean()) * 100,
                edge=(gx + gy) / 2.0)


def _cr(im: Image.Image, box, w=420):
    k = im.crop(box)
    return k.resize((w, int(k.height * w / k.width)), Image.LANCZOS)


def sheet(items, out: Path, title: str, cols=4, w=420, box=BOX):
    ims = [(_cr(Image.open(p).convert("RGB"), box, w), lab, mm) for lab, p, mm in items]
    H = ims[0][0].height
    rows = (len(ims) + cols - 1) // cols
    cap, gap = 30, 8
    sh = Image.new("RGB", (w * cols + gap * (cols + 1),
                           (H + cap) * rows + gap + 26), (18, 18, 22))
    d = ImageDraw.Draw(sh)
    d.text((10, 7), title, fill=(255, 235, 170))
    for i, (im, lab, mm) in enumerate(ims):
        r, c = divmod(i, cols)
        x = gap + c * (w + gap); y = 26 + gap + r * (H + cap)
        sh.paste(im, (x, y + cap))
        d.text((x + 3, y + 4), lab, fill=(255, 214, 110))
        d.text((x + 3, y + 16),
               f"IoU{mm['iou']:.2f} wcorr{mm['wcorr']:.2f} Y{mm['yellow']:.1f}%",
               fill=(150, 220, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    sh.save(out, quality=92)
    print("sheet:", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True, help="标签=路径[,标签=路径...]")
    ap.add_argument("--out", default="jobs/_probe/v409_eval_sheet.jpg")
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()

    o = Image.open(ORIG).convert("RGB")
    m = mask_p6(o)
    mb = metrics(o, o, m)
    items = [("ORIG", str(ORIG), mb)]
    print(f"{'ORIG':24s} sd={mb['sd']:.3f} IoU={mb['iou']:.3f} "
          f"wcorr={mb['wcorr']:.3f} 黄={mb['yellow']:.2f}% 边缘={mb['edge']:.2f}")
    for it in a.cands.split(","):
        if not it.strip():
            continue
        lab, p = it.split("=", 1)
        p = Path(p.strip())
        if not p.exists():
            print(f"! 缺 {p}")
            continue
        mm = metrics(Image.open(p).convert("RGB"), o, m)
        items.append((lab, str(p), mm))
        print(f"{lab:24s} sd={mm['sd']:.3f} IoU={mm['iou']:.3f} "
              f"wcorr={mm['wcorr']:.3f} 黄={mm['yellow']:.2f}% 边缘={mm['edge']:.2f}")
    sheet(items, Path(a.out), "p6 主体裂变评估（wcorr 越低 = 轮廓越拉开）", cols=a.cols)


if __name__ == "__main__":
    main()
