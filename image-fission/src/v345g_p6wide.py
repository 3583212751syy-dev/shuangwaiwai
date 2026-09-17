"""v345g: p6 鹰+骷髅 —— 「放宽回贴区」扫参（v345 真根因修复后的第一轮）。

根因链（本轮实测坐实）：
  ① `rebirth_subject` 把 SDXL 的 inpainting 噪声掩膜设成 dilate(mask, 18)、回贴羽化
     dilate(mask, 12) → **新剪影物理上长不出原剪影 ±18px**；
  ② 于是实测 v344 成品与原图：p6 主体集合 IoU=0.867 / 逐行宽度相关 0.995，
     p4 更极端 IoU=0.960 / 0.997 → 用户"没看到明显变化"是**几何必然**，不是模型问题，
     更不是调参能救的（前 13 组 cn/denoise/ckpt 扫参全部无效，就是这个原因）；
  ③ 对比用户认可的 6978 蝙蝠：IoU=0.780 → 差值主要来自"掩膜外的形状自由度"，
     蝙蝠掩膜小(3.1%)、外扩 18px 相对它自身尺度很大 → 形状能动；p6 掩膜 22%、
     主体框 3130x3388，18px 相对可忽略 → 形状锁死。

修法：`wide` 放宽噪声掩膜与回贴区，`bg_gate` 只让"判为主体"的环带像素贴回（防灰雾），
`protect` 挡住标题带（y<1500，主体上沿 1495 紧贴它，不挡会被擦掉）。

判据（客观、不靠肉眼）：与原图比 主体集合 IoU（目标 ~0.78 档）+ 逐行宽度相关（<0.99）。
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import v342_rebirth as R            # noqa: E402

OUT = ROOT / "jobs" / "v345_p6wide"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
TITLE_Y = 1500                       # mask_p6 里排除的标题带

P6 = ("a bald eagle with wide spread wings perched on top of a horned animal skull, "
      "red-brown and white feathers, ornate dark engraving illustration, bold clean shapes, "
      "high contrast, sharp hooked beak, fierce eye, bone-white cracked skull with curved horns, "
      "a completely different wing silhouette with a different feather arrangement, "
      "differently curved horns and different skull proportions, "
      "black background, brown tan white and black, no text, no letters")
N6 = ("text, letters, words, watermark, blurry, low quality, photo, realistic, soft, "
      "airbrush, mushy, noise, sketch, messy, deformed, extra wings, extra legs, "
      "two heads, two skulls, three eyes, white background, gray background, "
      "glow, neon, sparkle, flat plain shape")

VARIANTS = [
    ("W1_wide120_cn35e45_dn90", dict(wide=120, cn_strength=0.35, cn_end=0.45, denoise=0.90)),
    ("W2_wide200_cn35e45_dn90", dict(wide=200, cn_strength=0.35, cn_end=0.45, denoise=0.90)),
    ("W3_wide120_cn25e35_dn92", dict(wide=120, cn_strength=0.25, cn_end=0.35, denoise=0.92)),
    ("W4_wide60_cn35e45_dn90", dict(wide=60, cn_strength=0.35, cn_end=0.45, denoise=0.90)),
]
MS = 2048


def subject(a):
    """p6 主体集合：非黑实体 − 标题带 − 细射线/星点。"""
    lum = a.mean(2)
    s = lum > 55.0
    s[:TITLE_Y, :] = False
    s = ndi.binary_opening(s, structure=np.ones((3, 3), bool))
    return s


def diff(tag, so, sr):
    iou = (so & sr).sum() / max(1, (so | sr).sum())
    r1, r2 = np.where(so.any(1))[0], np.where(sr.any(1))[0]
    lo, hi = max(r1.min(), r2.min()), min(r1.max(), r2.max())
    c = float(np.corrcoef(so.sum(1)[lo:hi + 1], sr.sum(1)[lo:hi + 1])[0, 1])
    print(f"[{tag}] 主体 {100*so.mean():.2f}%→{100*sr.mean():.2f}%  "
          f"IoU={iou:.3f} 行宽相关={c:.3f}  {'★够裂变' if iou < 0.84 else '⚠仍太像'}")
    return iou, c


def main():
    only = sys.argv[1:] or None
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    mask = R.mask_p6(img)
    protect = np.zeros((H, W), bool)
    protect[:TITLE_Y, :] = True
    so = subject(np.asarray(img, np.float32))
    print(f"[p6] size={img.size} mask={100*mask.mean():.1f}% 原主体={100*so.mean():.2f}%")

    rows = []
    for name, v in VARIANTS:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            out = R.rebirth_subject(img, mask, P6, N6, tag=name, seed=7,
                                    ipa_weight=0.0, color_match=0.90,
                                    cn_name=CANNY, cn_pre="canny", max_side=MS,
                                    bg_gate=35.0, subject_dark=False, protect=protect,
                                    **v)
        except Exception as e:
            print(f"[{name}] FAIL {type(e).__name__}: {e}")
            continue
        p = OUT / f"{name}.jpg"
        out.save(str(p), quality=93)
        print(f"[{name}] {time.time()-t0:.0f}s -> {p.name}")
        diff(name, so, subject(np.asarray(out.convert("RGB"), np.float32)))
        rows.append((name, out))

    if rows:
        w = 430
        h = int(img.size[1] * w / img.size[0])
        sheet = Image.new("RGB", (w * (len(rows) + 1) + 8 * (len(rows) + 2), h + 8),
                          (25, 25, 25))
        sheet.paste(img.resize((w, h), Image.LANCZOS), (8, 4))
        for i, (nm, im) in enumerate(rows):
            sheet.paste(im.resize((w, h), Image.LANCZOS), ((i + 1) * (w + 8) + 8, 4))
        sheet.save(str(OUT / "_sheet.jpg"), quality=90)
        print("[sheet]", OUT / "_sheet.jpg")


if __name__ == "__main__":
    main()
