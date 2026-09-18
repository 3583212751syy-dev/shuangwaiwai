"""v379：p6 重生第二轮（改善眼窝黄光 + 试 Juggernaut-Ragnarok 暗黑金属底模 + 矢量 LoRA）。"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v342_rebirth import mask_p6, rebirth_subject            # noqa: E402
from v377_p6rb import snap_p6                                 # noqa: E402

OUT = ROOT / "jobs" / "v379_p6rb2"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"

POS = ("a fierce bald eagle with spread wings perched on a horned demon skull, "
       "different wing shape and different feather arrangement, different skull shape, "
       "empty hollow pitch-black eye sockets and nasal cavity, no glowing eyes, "
       "different curved horns, heavy black ink engraving illustration, bold cross-hatched shading, "
       "white skull bone, warm brown feathers, tan horns, pure black background, "
       "symmetrical centered composition, high contrast, no text, no letters")
NEG = ("glowing eyes, yellow eyes, orange eyes, eye light, text, letters, words, watermark, "
       "blurry, low quality, photo, realistic, extra heads, extra skulls, extra birds, "
       "deformed, asymmetric, gray haze, fog, noise, speckle, dirty background, halo, glow")

VARIANTS = [
    ("P1_prot_dn90_cn45e45", "ProteusV0.4.safetensors", 0.90, 0.45, 0.45, 77, None),
    ("P2_prot_dn88_cn50e50", "ProteusV0.4.safetensors", 0.88, 0.50, 0.50, 88, None),
    ("J1_ragn_dn90_cn45e45", "juggernautXL_ragnarokBy.safetensors", 0.90, 0.45, 0.45, 91, None),
    ("J2_ragn_dn85_cn55e55", "juggernautXL_ragnarokBy.safetensors", 0.85, 0.55, 0.55, 92, None),
    ("V1_prot_vec_dn90", "ProteusV0.4.safetensors", 0.90, 0.45, 0.45, 95,
     [("DD-vector-v2.safetensors", 0.55)]),
]


def main():
    src = Image.open(SRC).convert("RGB")
    mask = mask_p6(src)
    for name, ckpt, dn, cns, cne, seed, lora in VARIANTS:
        t = time.time()
        try:
            full = rebirth_subject(src, mask, POS, NEG, ckpt=ckpt, denoise=dn, ipa_weight=0.0,
                                   color_match=0.0, cn_name=CN, cn_strength=cns, cn_pre="canny",
                                   cn_end=cne, lora=lora, margin=60, grow=10, seed=seed,
                                   tag=name, max_side=1536)
        except Exception as e:
            print(f"[{name}] FAILED {type(e).__name__}: {e}")
            continue
        full.save(str(OUT / f"p6_{name}.jpg"), quality=93)
        snap_p6(src, full, mask).save(str(OUT / f"p6_{name}_snap.jpg"), quality=93)
        print(f"[{name}] {time.time()-t:.0f}s")
    sheet()


def sheet():
    TH = 430
    src = Image.open(SRC).convert("RGB")
    items = [("ORIG", src)]
    ref = Image.open(ROOT / "jobs" / "v377_p6rb" / "p6_A_dn90_cn45e45_snap.jpg").convert("RGB")
    items.append(("A(round1)", ref))
    for p in sorted(OUT.glob("p6_*_snap.jpg")):
        items.append((p.stem.replace("p6_", "").replace("_snap", ""), Image.open(p).convert("RGB")))
    box = (150, 1350, 3450, 4100)
    cols = []
    for nm, im in items:
        c = im.crop(box); w = int(c.width * TH / c.height)
        cols.append((nm, c.resize((max(1, w), TH), Image.LANCZOS)))
    gap = 6
    Wt = sum(c[1].width for c in cols) + gap * (len(cols) + 1)
    cv = Image.new("RGB", (Wt, TH + 22), (25, 25, 28)); dr = ImageDraw.Draw(cv); x = gap
    for nm, c in cols:
        cv.paste(c, (x, 19)); dr.text((x + 3, 4), nm, fill=(235, 235, 235)); x += c.width + gap
    cv.save(str(OUT / "S_p6_cands2.jpg"), quality=93)
    print("sheet", cv.size)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sheet":
        sheet()
    else:
        main()
        sheet()
