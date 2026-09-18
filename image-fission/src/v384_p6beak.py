"""v384：p6 主体重生 **喙色/羽色校正**（v373 遗留项）。

v382 定稿的 `p6_H1_ragn2048_snap.jpg` 结构裂变已达标（鹰头羽序/骷髅/双角全部换新），
但两处**配色漂移**：
  ① 鹰喙 由原图**明黄**变成白／浅骨色（原图最亮的彩色点缀丢了）；
  ② 翼羽 由原图**中深棕**整体偏成**浅茶金**（色族仍在棕系，但明度抬高）。
`snap_p6` 的 LAB Reinhard 只对齐**掩膜内整体均值/方差**，局部物体的色相漂移治不了。

修法：把颜色写进 prompt（SDXL 靠文本锁色），并把 NEG 里与之打架的
"yellow eyes / orange eyes" 换成不含颜色的同义否定（眼窝由 POS 的
"pitch-black eye sockets" 保证），避免 CLIP 把 yellow 绑到眼睛上。
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

OUT = ROOT / "jobs" / "v384_p6beak"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"
BASE_SNAP = ROOT / "jobs" / "v382_p6hi" / "p6_H1_ragn2048_snap.jpg"

POS = ("a fierce bald eagle with spread wings perched on a horned demon skull, "
       "different wing shape and different feather arrangement, different skull shape, "
       "empty hollow pitch-black eye sockets and nasal cavity, no glowing eyes, "
       "bright golden-yellow hooked beak, dark chocolate brown wing feathers, "
       "different curved horns, heavy black ink engraving illustration, bold cross-hatched shading, "
       "white skull bone, tan horns, pure black background, "
       "symmetrical centered composition, high contrast, no text, no letters")
NEG = ("glowing eyes, luminous eyes, eye light, text, letters, words, watermark, "
       "blurry, low quality, photo, realistic, extra heads, extra skulls, extra birds, "
       "deformed, asymmetric, gray haze, fog, noise, speckle, dirty background, halo, glow, "
       "pale beak, white beak")

VARIANTS = [
    ("K1_ragn_beak_s91", 0.90, 0.45, 0.45, 91),
    ("K2_ragn_beak_s41", 0.90, 0.45, 0.45, 41),
]


def main():
    src = Image.open(SRC).convert("RGB")
    mask = mask_p6(src)
    for name, dn, cns, cne, seed in VARIANTS:
        t = time.time()
        try:
            full = rebirth_subject(src, mask, POS, NEG,
                                   ckpt="juggernautXL_ragnarokBy.safetensors", denoise=dn,
                                   ipa_weight=0.0, color_match=0.0, cn_name=CN,
                                   cn_strength=cns, cn_pre="canny", cn_end=cne,
                                   margin=60, grow=10, seed=seed, tag=name, max_side=2048)
        except Exception as e:
            print(f"[{name}] FAILED {type(e).__name__}: {e}")
            continue
        snap_p6(src, full, mask).save(str(OUT / f"p6_{name}_snap.jpg"), quality=93)
        print(f"[{name}] {time.time() - t:.0f}s")
    sheet()


def sheet():
    src = Image.open(SRC).convert("RGB")
    items = [("ORIG", src)]
    if BASE_SNAP.exists():
        items.append(("H1_2048(v373)", Image.open(BASE_SNAP).convert("RGB")))
    for p in sorted(OUT.glob("p6_*_snap.jpg")):
        items.append((p.stem.replace("p6_", ""), Image.open(p).convert("RGB")))
    box = (700, 1050, 2950, 2650)          # 鹰头 + 双翼
    th = 560
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
        dr.text((x + 3, 4), nm[:16], fill=(240, 240, 240))
        x += c.width + gap
    cv.save(str(OUT / "S_p6_beak.jpg"), quality=94)
    print("sheet", cv.size)


if __name__ == "__main__":
    main()
