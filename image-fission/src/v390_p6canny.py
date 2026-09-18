# -*- coding: utf-8 -*-
"""v390_p6canny — pinterest6 鹰骷髅 定稿裂变代码（**单图专用**）

五图 ↔ 代码命名（严格记住，每图独立代码，禁止一码通用）：
  ① 6978        → bat_badge          styles/subject_badge_text.py
  ② b78e60      → dogtag_camo        styles/camo_pattern.py
  ③ pinterest3  → denim_butterfly    src/v381_p3final.py
  ④ pinterest4  → camo_palm_swap     styles/camo_palm_pattern.py + src/v346_p4aff.py
  ⑤ pinterest6  → eagle_skull_canny  src/v390_p6canny.py   ← 本文件

────────────────────────────────────────────────────────────────────────────
用户对 p6 的三次否决（每次都被逼出一条铁律）：

  ❌ v373「本地 inpaint 重生」（掩膜内小改）
     → 主体≈原图复制。用户 z3 对照原话：「这点裂变跟原图的区别是什么」
     铁律：**禁本地 inpaint 当裂变** —— 那只是局部修补，不是换内容。

  ❌ v386「mode1」（880×1240 txt2img + IPAdapter 双锁 + Canny 0.5）
     → 生成图原生仅 880px 宽，被 4× 放大回 3543px → 鹰头糊成一团。
     用户原话：「有逻辑是正常元素的要求，做的代码错乱就是不合格，严重不合格」
     实测：鹰头区 Laplacian 方差 461 vs 原图 2715（软 5.9 倍）= 糊。
     铁律：**重生分辨率不得低于原图** —— 降采样/放大 = 糊 = "错乱"。

  ✅ v390「Canny 结构引导 SDXL 原生 2048 重绘」（本文件）
     · 用 ControlNet-Canny 出原图线稿骨架，只做**构图骨架**引导（不锁死轮廓）
     · v393 放松：cn_strength 0.20（弱引导）/ cn_end 0.55（骨架沿用更久）/
       denoise 0.98（高噪声→真换形，不只描羽片）
     · prompt 加「不同翼展(wider/narrower) + 鹰头转向一侧」以逼出结构级变化
     · max_side=2048 原生渲染（原图 3543px 的 0.58×，不放大）
     · prompt 锁「flat vector illustration / hard clean edges / solid flat colors」
       防 SDXL 把平涂制版风画成照片写实（v384 的病）
     · 结果：鹰（翼羽改人字纹/翼展姿态/白首黄喙黄眼）、骷髅（颅形/裂纹/下颌）、
             双角（加粗分节）全部重构 = 真·异内容同构，且硬边清晰、逻辑正常。
     · seed 777：6 变体（F1~F6）中唯一同时满足
       「变化量最大 + 无黄色杂斑 + 无深色污斑」（见 _p6_final.py 的扫描结论）。

用法：
    python src/v390_p6canny.py            # 用定稿参数出图 → jobs/v390_p6canny/
    python src/v390_p6canny.py --seed 91  # 换种子
产出可直接喂给 `make_v329.py` 的 REBIRTH_FILES['pinterest6']。
"""
import argparse
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v342_rebirth import mask_p6, rebirth_subject          # noqa: E402
from v377_p6rb import snap_p6                              # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"

# ── 定稿参数（v393 放松：让 canny 只当构图骨架、模型真换形）─────────────────
# 旧 F5(0.32/0.42/0.94) 被 user 判"canny 锁太死→只描羽片不换形"。
# 本轮按 user 指示放松：cn_strength 0.18~0.22（弱引导）、cn_end 0.55（骨架沿用更久）、
# denoise 0.97~1.0（高噪声→真重画）。取区间中值 0.20 / 0.98。
CKPT = "juggernautXL_ragnarokBy.safetensors"
DEF_CN_STRENGTH = 0.20
DEF_CN_END = 0.55
DEF_DENOISE = 0.98
DEF_SEED = 777
DEF_MAX_SIDE = 2048

POS = ("bald eagle with spread wings perched on a horned demon skull, "
       "white feathered eagle head turned to one side at a new angle, "
       "bright golden-yellow hooked beak, fierce visible eye, "
       "new wing shape with a different wingspan wider or narrower than before, "
       "different feather arrangement, very dark chocolate brown wing feathers, "
       "redesigned skull, "
       "clean white bone skull with even fine cross-hatch shading, "
       "empty hollow pitch-black eye sockets and nasal cavity, no glowing eyes, "
       "different curved ribbed horns, "
       "flat vector illustration, bold screen print, hard clean edges, crisp linework, "
       "solid flat colors, tan horns, pure black background, white lightning bolts, "
       "centered composition, high contrast, no text, no letters")
NEG = ("yellow patches, yellow stains, yellow spots on skull, yellow feathers on skull, "
       "dirty bone, dark stains, smudges, blotches, mud, grime, "
       "glowing eyes, luminous eyes, eye light, text, letters, words, watermark, "
       "blurry, low quality, photo, realistic, 3d render, airbrush, painterly, "
       "soft gradients, smooth shading, extra heads, extra skulls, extra birds, "
       "deformed, asymmetric, gray haze, fog, noise, speckle, dirty background, "
       "halo, glow, pale beak, white beak")


def run(seed=DEF_SEED, cn_strength=DEF_CN_STRENGTH, cn_end=DEF_CN_END,
        denoise=DEF_DENOISE, max_side=DEF_MAX_SIDE, out_dir=None, tag=None):
    """重生 p6 主体，返回 snap 后成品路径（含原图标题带）。"""
    out_dir = Path(out_dir) if out_dir else (ROOT / "jobs" / "v390_p6canny")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"canny_s{seed}"
    src = Image.open(SRC).convert("RGB")
    mask = mask_p6(src)
    t = time.time()
    full = rebirth_subject(src, mask, POS, NEG, ckpt=CKPT, denoise=denoise,
                           ipa_weight=0.0, color_match=0.0, cn_name=CN,
                           cn_strength=cn_strength, cn_pre="canny", cn_end=cn_end,
                           margin=60, grow=10, seed=seed, tag=tag, max_side=max_side)
    snap = snap_p6(src, full, mask)
    dst = out_dir / f"p6_{tag}_snap.jpg"
    snap.save(str(dst), quality=93)
    print(f"[eagle_skull_canny] seed={seed} cn={cn_strength}/{cn_end} dn={denoise} "
          f"ms={max_side} → {dst}  ({time.time() - t:.0f}s)")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEF_SEED)
    ap.add_argument("--cn-strength", type=float, default=DEF_CN_STRENGTH)
    ap.add_argument("--cn-end", type=float, default=DEF_CN_END)
    ap.add_argument("--denoise", type=float, default=DEF_DENOISE)
    ap.add_argument("--max-side", type=int, default=DEF_MAX_SIDE)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)
    a = ap.parse_args()
    run(seed=a.seed, cn_strength=a.cn_strength, cn_end=a.cn_end,
        denoise=a.denoise, max_side=a.max_side, out_dir=a.out_dir, tag=a.tag)


if __name__ == "__main__":
    main()
