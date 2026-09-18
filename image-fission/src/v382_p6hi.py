"""v382：p6 重生 **高分辨率 + 调色板量化**（治「笔触比原图软」）。

诊断：p6 原图主体裁块 3291x3546，SDXL 只会跑 1536 长边 → 0.47x 降采样再升回 2.15x，
线稿必然发软（🔴28：降采样 = 线稿掉档）。且原图是**平色钢笔画**（黑/白/骨/棕/褐），
SDXL 输出却是连续调 → 语言不符。

两步修：
  ① max_side 1536 → 2048（0.62x，降采样损失减半）；
  ② snap 之后再做**原图调色板量化**：从原图主体区取 6 个主色，逐像素映射过去
     （mix 混回一点连续调防色带）→ 平色 + unsharp = 硬边线稿语言。
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v342_rebirth import mask_p6, rebirth_subject              # noqa: E402
from v377_p6rb import snap_p6                                   # noqa: E402
from v379_p6rb2 import POS, NEG, CN                             # noqa: E402

OUT = ROOT / "jobs" / "v382_p6hi"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")

VARIANTS = [
    ("H1_ragn2048", "juggernautXL_ragnarokBy.safetensors", 0.90, 0.45, 0.45, 91),
    ("H2_prot2048", "ProteusV0.4.safetensors", 0.90, 0.45, 0.45, 77),
]


def palette(o, mask, n=6):
    px = o[mask].astype(np.uint8)
    step = max(1, len(px) // 200000)
    sub = px[::step]
    im = Image.fromarray(sub.reshape(-1, 1, 3))
    q = im.quantize(colors=n, method=Image.Quantize.MEDIANCUT).convert("RGB")
    cols = np.unique(np.asarray(q).reshape(-1, 3), axis=0).astype(np.float32)
    print("palette", cols.astype(int).tolist())
    return cols


def quantize_to(gen, cols, mix=0.78):
    best = np.full(gen.shape[:2], 1e18, np.float32)
    qc = np.zeros_like(gen)
    for c in cols:
        d = ((gen - c[None, None, :]) ** 2).sum(-1)
        m = d < best
        best[m] = d[m]
        qc[m] = c
    return gen * (1.0 - mix) + qc * mix


def main():
    src = Image.open(SRC).convert("RGB")
    mask = mask_p6(src)
    o = np.asarray(src, np.float32)
    cols = palette(o, mask, 6)
    for name, ckpt, dn, cns, cne, seed in VARIANTS:
        t = time.time()
        try:
            full = rebirth_subject(src, mask, POS, NEG, ckpt=ckpt, denoise=dn, ipa_weight=0.0,
                                   color_match=0.0, cn_name=CN, cn_strength=cns, cn_pre="canny",
                                   cn_end=cne, margin=60, grow=10, seed=seed, tag=name,
                                   max_side=2048)
        except Exception as e:
            print(f"[{name}] FAILED {type(e).__name__}: {e}")
            continue
        snap_p6(src, full, mask).save(str(OUT / f"p6_{name}_snap.jpg"), quality=93)
        g = np.asarray(snap_p6(src, full, mask), np.float32)
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        blk = quantize_to(g[y0:y1, x0:x1], cols)
        mc = mask[y0:y1, x0:x1][..., None]
        g[y0:y1, x0:x1] = g[y0:y1, x0:x1] * (1 - mc) + blk * mc
        Image.fromarray(np.clip(g, 0, 255).astype(np.uint8), "RGB").save(
            str(OUT / f"p6_{name}_snap_q.jpg"), quality=93)
        print(f"[{name}] {time.time()-t:.0f}s")
    sheet()


def sheet():
    src = Image.open(SRC).convert("RGB")
    items = [("ORIG", src),
             ("J1_1536", Image.open(ROOT / "jobs/v379_p6rb2/p6_J1_ragn_dn90_cn45e45_snap.jpg")),
             ("J1_q", Image.open(ROOT / "jobs/v379_p6rb2/p6_J1_ragn_dn90_cn45e45_snap.jpg"))]
    items = [("ORIG", src)]
    for p in sorted(OUT.glob("p6_*_snap*.jpg")):
        items.append((p.stem.replace("p6_", ""), Image.open(p).convert("RGB")))
    box = (250, 1350, 3350, 4350)
    th = 760
    cols = []
    for nm, im in items:
        c = im.crop(box); w = int(c.width * th / c.height)
        cols.append((nm, c.resize((max(1, w), th), Image.LANCZOS)))
    gap = 8
    cv = Image.new("RGB", (sum(c[1].width for c in cols) + gap * (len(cols) + 1), th + 22),
                   (25, 25, 28))
    dr = ImageDraw.Draw(cv); x = gap
    for nm, c in cols:
        cv.paste(c, (x, 19)); dr.text((x + 3, 4), nm[:14], fill=(240, 240, 240)); x += c.width + gap
    cv.save(str(OUT / "S_p6_hi.jpg"), quality=94)
    print("sheet", cv.size)


if __name__ == "__main__":
    main()
