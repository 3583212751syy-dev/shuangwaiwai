"""v390：p6 定稿精选 —— 在 C3 参数（cn.35/e.45/dn.95 @2048）附近多种子扫描。

v389 结论：
  · C1(cn.28/e.35) 变化大，但骷髅颅面出现**深色污斑**、颊侧**黄色杂块**；
  · C2(cn.18/e.28) 鹰头糊成黄色团块 → 无逻辑（用户红线）；
  · C4(cn.22/e.32/dn.98) 鹰头退化成褐色团 → 无逻辑；
  · C3(cn.35/e.45/dn.95) 白头鹰+黄喙+重构翼羽/分节肋角/重构颅骨，**逻辑正确且清晰** ← 基准。
本轮在 C3 附近扫描种子 + 加负提示压「脏斑/黄渍」，选最干净的一张定稿。
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

OUT = ROOT / "jobs" / "v390_p6final"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"

POS = ("bald eagle with spread wings perched on a horned demon skull, "
       "new wing shape and different feather arrangement, redesigned skull, "
       "clean white bone skull with even fine cross-hatch shading, "
       "empty hollow pitch-black eye sockets and nasal cavity, no glowing eyes, "
       "bright golden-yellow hooked beak, dark chocolate brown wing feathers, "
       "different curved ribbed horns, "
       "flat vector illustration, bold screen print, hard clean edges, crisp linework, "
       "solid flat colors, tan horns, pure black background, white lightning bolts, "
       "symmetrical centered composition, high contrast, no text, no letters")
NEG = ("yellow patches, yellow stains, yellow spots on skull, yellow feathers on skull, "
       "dirty bone, dark stains, smudges, blotches, mud, grime, "
       "glowing eyes, luminous eyes, eye light, text, letters, words, watermark, "
       "blurry, low quality, photo, realistic, 3d render, airbrush, painterly, "
       "soft gradients, smooth shading, extra heads, extra skulls, extra birds, "
       "deformed, asymmetric, gray haze, fog, noise, speckle, dirty background, "
       "halo, glow, pale beak, white beak")

# (name, cn_strength, cn_end, denoise, seed)
VARIANTS = [
    ("F1_s91", 0.35, 0.45, 0.95, 91),
    ("F2_s41", 0.35, 0.45, 0.95, 41),
    ("F3_s7", 0.35, 0.45, 0.95, 7),
    ("F4_s2026", 0.35, 0.45, 0.95, 2026),
    ("F5_s777", 0.32, 0.42, 0.94, 777),
    ("F6_s1234", 0.38, 0.48, 0.95, 1234),
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
