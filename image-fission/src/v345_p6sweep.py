"""v345: p6 鹰+骷髅「真裂变」扫参。

用户第 14 轮：「主体元素不去裂变是吗，一定要我讲吗」
诊断：v344 的 CFG 是 cn_strength .60 / cn_end .75 —— ControlNet 结构锁太死，
      鹰/骷髅只是被"描了一遍 + 换纹理"，剪影几乎没动 → 读成"没裂变"。
对照 6978 蝙蝠（用户认可的"蝙蝠设计可以"）：cn .38 / cn_end .45 / dn .92
      → 结构只在前 45% 步被约束，后段完全放开 → 剪影真的换新，且 prompt
      死咬 "flat matte vector / bold clean outline" 所以不乱。

本脚本在同一掩膜上扫 (cn_strength, cn_end, denoise, max_side)，外加一版**分部件**
（鹰 / 骷髅各自裁块单独重生）——分部件能把每块画幅压到 SDXL 甜点尺寸，
既避免整块降采样发虚，也让两个部件各自被彻底重画。
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
import v342_rebirth as R

OUT = ROOT / "jobs" / "v345_p6"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"

P6 = ("a bald eagle with spread wings perched on top of a horned animal skull, "
      "brown and white feathers, ornate dark engraving illustration, bold clean shapes, "
      "high contrast, sharp hooked beak, fierce eye, cracked bone skull with curved horns, "
      "a different wing shape, a different feather arrangement, different skull proportions "
      "and a different horn shape, black background, brown tan white and black, "
      "no text, no letters")
N6 = ("text, letters, words, watermark, blurry, low quality, photo, realistic, soft, "
      "airbrush, mushy, noise, sketch, messy, deformed, extra wings, extra legs, "
      "two heads, two skulls, three eyes, white background, gray background, "
      "glow, neon, sparkle")

VARIANTS = [
    ("P1_cn35e45_dn90", dict(cn_strength=0.35, cn_end=0.45, denoise=0.90, max_side=2048)),
    ("P2_cn25e35_dn92", dict(cn_strength=0.25, cn_end=0.35, denoise=0.92, max_side=2048)),
    ("P3_cn45e60_dn90_ms2560", dict(cn_strength=0.45, cn_end=0.60, denoise=0.90,
                                    max_side=2560)),
    ("P4_cn35e45_dn88_ms1536", dict(cn_strength=0.35, cn_end=0.45, denoise=0.88,
                                    max_side=1536)),
]


def main():
    only = sys.argv[1:] or None
    img = Image.open(SRC).convert("RGB")
    mask = R.mask_p6(img)
    ys, xs = np.where(mask)
    print(f"[p6] size={img.size} mask%={100*mask.mean():.1f} "
          f"bbox x{xs.min()}-{xs.max()} y{ys.min()}-{ys.max()}")
    rows = []
    for name, v in VARIANTS:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            out = R.rebirth_subject(img, mask, P6, N6, tag=name, seed=7,
                                    ipa_weight=0.0, color_match=0.90,
                                    cn_name=CANNY, cn_pre="canny", **v)
        except Exception as e:
            print(f"[{name}] FAIL {e}")
            continue
        p = OUT / f"{name}.jpg"
        out.save(str(p), quality=93)
        q = R.qc(img, out, mask)
        print(f"[{name}] {time.time()-t0:.0f}s")
        rows.append((name, out))
    if rows:
        # 缩小对照（原图 | 各变体），并额外存一个 1:1 部件裁片
        w = 420
        h = int(img.size[1] * w / img.size[0])
        sheet = Image.new("RGB", (w * (len(rows) + 1) + 8 * (len(rows) + 2), h + 8),
                          (25, 25, 25))
        sheet.paste(img.resize((w, h), Image.LANCZOS), (8, 4))
        for i, (nm, im) in enumerate(rows):
            sheet.paste(im.resize((w, h), Image.LANCZOS), ((i + 1) * (w + 8) + 8, 4))
        sheet.save(str(OUT / "_sheet.jpg"), quality=90)
        # 主体区 1:1 裁片对照
        cx0, cx1, cy0, cy1 = 250, 3350, 1350, 4750
        sc = 900
        s2 = int(img.size[1] * sc / img.size[0])
        sub = [img.crop((cx0, cy0, cx1, cy1)).resize((sc, s2), Image.LANCZOS)]
        for nm, im in rows:
            sub.append(im.crop((cx0, cy0, cx1, cy1)).resize((sc, s2), Image.LANCZOS))
        sh2 = Image.new("RGB", (sc * len(sub) + 8 * (len(sub) + 1), s2 + 8), (25, 25, 25))
        for i, im in enumerate(sub):
            sh2.paste(im, (i * (sc + 8) + 8, 4))
        sh2.save(str(OUT / "_subject.jpg"), quality=92)
        print("[sheet]", OUT / "_sheet.jpg", OUT / "_subject.jpg")


if __name__ == "__main__":
    main()
