# -*- coding: utf-8 -*-
"""v408_eval.py —— p6 区域裂变候选评估（量化 + 拼版）

量化（只作辅助，🔴 目检才是终判）：
  · 主体剪影 IoU（vs 原图）—— 越低 = 轮廓变化越大
  · 主体区灰度结构差异 (1 - MSE/255²) —— 越低 = 内容换得越多
  · 黄像素占比（颅骨窗）—— 🔴35：不得高于原图
  · 边缘密度（细节量）—— 过低 = 糊

用法：python src/v408_eval.py --cands "标签=路径,标签=路径" --out 拼版.jpg
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


def metrics(im: Image.Image, o: Image.Image, m: np.ndarray) -> dict:
    if im.size != o.size:
        im = im.resize(o.size, Image.LANCZOS)
    a = np.asarray(im, np.float32)
    b = np.asarray(o, np.float32)
    # ① 原剪影内的内容差异
    g1 = _lum(a); g2 = _lum(b)
    mse = float(((g1[m] - g2[m]) ** 2).mean())
    sd = 1.0 - mse / (255.0 ** 2)
    # ② 亮度剪影 IoU（主体 = 亮部，p6 主体白/棕 on 黑底）
    t = 55.0
    s1 = (g1 > t) & (m | _dil(m, 60))
    s2 = (g2 > t) & (m | _dil(m, 60))
    iou = float((s1 & s2).sum()) / max(1.0, float((s1 | s2).sum()))
    # ③ 黄像素（颅骨窗）
    win = (slice(2700, 4600), slice(800, 2800))
    aw = a[win]
    y = ((aw[..., 0] > 150) & (aw[..., 1] > 110) & (aw[..., 2] < 110)
         & ((aw[..., 0] - aw[..., 2]) > 60))
    # ④ 边缘密度
    gx = np.abs(np.diff(g1, axis=1)).mean()
    gy = np.abs(np.diff(g1, axis=0)).mean()
    return dict(sd=sd, iou=iou, yellow=float(y.mean()) * 100, edge=(gx + gy) / 2.0)


def _dil(m, it):
    from scipy import ndimage as ndi
    return ndi.binary_dilation(m, iterations=it)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True, help="标签=路径[,标签=路径...]")
    ap.add_argument("--out", default="jobs/_probe/v408_eval_sheet.jpg")
    ap.add_argument("--box", default="500,1000,3050,4600")
    a = ap.parse_args()
    o = Image.open(ORIG).convert("RGB")
    m = mask_p6(o)
    x0, y0, x1, y1 = [int(v) for v in a.box.split(",")]
    W = 420

    def cr(im):
        k = im.crop((x0, y0, x1, y1))
        return k.resize((W, int(k.height * W / k.width)), Image.LANCZOS)

    mb = metrics(o, o, m)
    tiles = [("ORIG", cr(o), mb)]
    for item in a.cands.split(","):
        if not item.strip():
            continue
        lab, p = item.split("=", 1)
        p = Path(p.strip())
        if not p.exists():
            print(f"! 缺 {p}")
            continue
        im = Image.open(p).convert("RGB")
        mm = metrics(im, o, m)
        tiles.append((lab, cr(im), mm))
        print(f"{lab:22s} 结构差异={mm['sd']:.3f}  剪影IoU={mm['iou']:.3f}  "
              f"黄={mm['yellow']:.2f}%  边缘={mm['edge']:.2f}")
    print(f"{'ORIG':22s} 结构差异={mb['sd']:.3f}  剪影IoU={mb['iou']:.3f}  "
          f"黄={mb['yellow']:.2f}%  边缘={mb['edge']:.2f}")

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    H = tiles[0][1].height
    cap = 20; gap = 8
    sh = Image.new("RGB", (W * cols + gap * (cols + 1), (H + cap) * rows + gap), (20, 20, 24))
    d = ImageDraw.Draw(sh)
    for i, (lab, im, mm) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = gap + c * (W + gap); y = gap + r * (H + cap)
        sh.paste(im, (x, y + cap))
        d.text((x + 3, y + 3), f"{lab}  IoU{mm['iou']:.2f} Y{mm['yellow']:.1f}%",
               fill=(255, 214, 110))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    sh.save(out, quality=92)
    print("sheet:", out)


if __name__ == "__main__":
    main()
