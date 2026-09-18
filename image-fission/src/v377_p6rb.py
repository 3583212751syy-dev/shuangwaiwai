"""v377：p6 鹰+骷髅 —— **真结构级重生**扫描。

用户第 16 轮原话「老鹰跟骷髅头说几次了，就是不裂变是吗」。
上一轮（v372）我做成了纯几何位移 → 鹰/骷髅像素级是原图本体，用户判定"不裂变"。
回到 SDXL 重生，但解决两个历史失败：
  · v344（cn .60 / end .75 / dn .85）→ 结构锁太死，重生结果几乎是原图描一遍（用户仍判"不裂变"）
  · v346（wide 回贴）→ 模型自由发挥 → 白羽乱飞/骷髅崩坏（用户判"这个主体是什么东西"）
本轮思路：**降 CN 强度 + 提早 cn_end + 抬 denoise** 让形体真正换新，
再靠 prompt 锁"物种/配色/构图"，最后用直方图匹配 + 硬边化把 SDXL 的糊边拉回硬边线稿语言。
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v342_rebirth import mask_p6, rebirth_subject, _lum, _match_region   # noqa: E402

OUT = ROOT / "jobs" / "v377_p6rb"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")

POS = ("a fierce bald eagle with spread wings perched on a horned demon skull, "
       "different wing shape and different feather arrangement, different skull shape "
       "with different eye sockets and teeth, different curved horns, "
       "heavy black ink engraving illustration, bold cross-hatched shading, "
       "brown feathers, bone-white skull, warm tan horns, pure black background, "
       "symmetrical centered composition, high contrast, no text, no letters")
NEG = ("text, letters, words, watermark, blurry, low quality, photo, realistic, "
       "extra heads, extra skulls, extra birds, deformed, asymmetric, "
       "gray haze, fog, noise, speckle, dirty background, white glow, halo")

CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"

#  name, denoise, cn_strength, cn_end, seed
VARIANTS = [
    ("A_dn90_cn45e45", 0.90, 0.45, 0.45, 11),
    ("B_dn92_cn35e40", 0.92, 0.35, 0.40, 23),
    ("C_dn95_cn30e35", 0.95, 0.30, 0.35, 37),
    ("D_dn88_cn55e55", 0.88, 0.55, 0.55, 51),
]


def snap_p6(orig, gen, mask, sharpen=True):
    """把重生区的**色调语言**拉回原图：LAB Reinhard + 亮度直方图规定化 + unsharp。

    SDXL 出来的笔触必然比原图软；用户读作"质量掉 / 糊"。这里不动结构，只把
    亮度分布对齐原图（分布一致 → 黑/白/棕/骨四阶重新分明）、再轻度 unsharp。
    ⚠️ 只算掩膜 bbox 内的裁块：全图 3543x4961 上跑 rgb2lab / unsharp 会慢到分钟级，
    且 `np.quantile(x, q_array)` 传 n 个分位点是 O(n·log n) 的假象真 O(n²) —— 必须
    改成「排序 + 索引取值」。
    """
    o = np.asarray(orig, np.float32)
    g = np.asarray(gen, np.float32)
    if int(mask.sum()) < 100:
        return Image.fromarray(np.clip(g, 0, 255).astype(np.uint8), "RGB")
    ys, xs = np.where(mask)
    x0 = max(0, xs.min() - 4); x1 = min(orig.width, xs.max() + 5)
    y0 = max(0, ys.min() - 4); y1 = min(orig.height, ys.max() + 5)
    oc = o[y0:y1, x0:x1]; gc = g[y0:y1, x0:x1]; mc = mask[y0:y1, x0:x1]
    # ① LAB Reinhard（均值/方差对齐，全量）
    gg = _match_region(gc, oc, mc, alpha=1.0)
    # ② 亮度直方图规定化（掩膜内）：gen 排序后逐位取 ref 的对应分位
    lg = _lum(gg); lo = _lum(oc)
    gv = lg[mc]; ov = lo[mc]
    ov_s = np.sort(ov)
    rank = np.argsort(np.argsort(gv))                     # 0..n-1
    tgt = ov_s[np.clip((rank.astype(np.int64) * (len(ov_s) - 1)) // max(1, len(gv) - 1),
                       0, len(ov_s) - 1)]
    lg2 = lg.copy(); lg2[mc] = tgt
    gain = np.clip(np.where(np.abs(lg) > 1e-3, lg2 / np.maximum(lg, 1e-3), 1.0), 0.0, 8.0)
    gg = np.clip(gg * gain[..., None], 0, 255)
    if sharpen:
        from PIL import ImageFilter
        im = Image.fromarray(np.clip(gg, 0, 255).astype(np.uint8), "RGB")
        im = im.filter(ImageFilter.UnsharpMask(radius=3, percent=110, threshold=2))
        gg = np.asarray(im, np.float32)
    out = o.copy()
    blk = oc * (1.0 - mc[..., None]) + gg * mc[..., None]
    out[y0:y1, x0:x1] = blk
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def main():
    src = Image.open(SRC).convert("RGB")
    print("src", src.size)
    mask = mask_p6(src)
    print("mask px", int(mask.sum()), "frac %.3f" % (mask.mean()))
    Image.fromarray((mask.astype(np.uint8) * 255), "L").save(str(OUT / "p6_mask.png"))
    gains = []
    for name, dn, cns, cne, seed in VARIANTS:
        t = time.time()
        try:
            full = rebirth_subject(src, mask, POS, NEG, ckpt="ProteusV0.4.safetensors",
                                   denoise=dn, ipa_weight=0.0, color_match=0.0,
                                   cn_name=CN, cn_strength=cns, cn_pre="canny", cn_end=cne,
                                   margin=60, grow=10, seed=seed, tag=name, max_side=1536)
        except Exception as e:
            print(f"[{name}] FAILED {e}")
            continue
        full.save(str(OUT / f"p6_{name}.jpg"), quality=93)
        snap = snap_p6(src, full, mask)
        snap.save(str(OUT / f"p6_{name}_snap.jpg"), quality=93)
        print(f"[{name}] {time.time()-t:.0f}s saved")
        gains.append((name, full, snap))


def sheet(TH=470):
    src = Image.open(SRC).convert("RGB")
    items = [("ORIG", src)]
    for p in sorted(OUT.glob("p6_[A-D]_*_snap.jpg")):
        items.append((p.stem.replace("p6_", "").replace("_snap", ""),
                      Image.open(p).convert("RGB")))
    box = (150, 1350, 3450, 4100)
    cols = []
    for nm, im in items:
        c = im.crop(box)
        w = int(c.width * TH / c.height)
        cols.append((nm, c.resize((w, TH), Image.LANCZOS)))
    gap = 8
    Wt = sum(c[1].width for c in cols) + gap * (len(cols) + 1)
    cv = Image.new("RGB", (Wt, TH + 26), (25, 25, 28))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(cv)
    x = gap
    for nm, c in cols:
        cv.paste(c, (x, 22)); dr.text((x + 3, 6), nm, fill=(235, 235, 235))
        x += c.width + gap
    cv.save(str(OUT / "S_p6_cands.jpg"), quality=93)
    print("sheet", cv.size)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sheet":
        sheet()
    else:
        main()
        sheet()
