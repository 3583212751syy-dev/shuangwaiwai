"""v389：p6 找平衡点 —— 「真裂变(内容换新) + 平涂硬边(不糊) + 逻辑正常(不散)」。

用户第 18 轮两条否决：
  ① p6「有逻辑是正常元素的要求，做的代码错乱就是不合格」→ 上一版 mode1 低分辨率
     （880×1240）被 4× 照片放大到 3543×4961 → 鹰糊成一团 = "错乱"。
  ② v373 的 canny 重生(dn.90/cn.38/end.45 @2048)清晰，但**结构锁太死** →
     用户判「这点裂变跟原图的区别是什么」（近乎复制）。

本轮取二者之长：**保留 @2048 的原生清晰度**，把 canny 锁放松（降 strength/end）
让模型真的重画内容，同时 prompt 补上「flat vector / 硬边 / 印刷」压制气刷发虚。
"""
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v342_rebirth import mask_p6, rebirth_subject          # noqa: E402
from v377_p6rb import snap_p6                              # noqa: E402

OUT = ROOT / "jobs" / "v389_p6crisp"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"

POS = ("bald eagle with spread wings perched on a horned demon skull, "
       "new wing shape and different feather arrangement, redesigned skull, "
       "empty hollow pitch-black eye sockets and nasal cavity, no glowing eyes, "
       "bright golden-yellow hooked beak, dark chocolate brown wing feathers, "
       "different curved ribbed horns, "
       "flat vector illustration, bold screen print, hard clean edges, crisp linework, "
       "cross-hatched engraving shading, solid flat colors, "
       "white skull bone, tan horns, pure black background, white lightning bolts, "
       "symmetrical centered composition, high contrast, no text, no letters")
NEG = ("glowing eyes, luminous eyes, eye light, text, letters, words, watermark, "
       "blurry, low quality, photo, realistic, 3d render, airbrush, painterly, "
       "soft gradients, smooth shading, extra heads, extra skulls, extra birds, "
       "deformed, asymmetric, gray haze, fog, noise, speckle, dirty background, "
       "halo, glow, pale beak, white beak")

VARIANTS = [
    ("C1_cn28_e35_dn92", 0.28, 0.35, 0.92, 91),
    ("C2_cn18_e28_dn95", 0.18, 0.28, 0.95, 41),
    ("C3_cn35_e45_dn95", 0.35, 0.45, 0.95, 91),
    ("C4_cn22_e32_dn98", 0.22, 0.32, 0.98, 7),
]


def main():
    src = Image.open(SRC).convert("RGB")
    mask = mask_p6(src)
    for name, cns, cne, dn, seed in VARIANTS:
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


if __name__ == "__main__":
    main()
