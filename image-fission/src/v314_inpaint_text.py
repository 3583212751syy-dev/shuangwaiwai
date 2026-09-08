"""
v314 — 文字带本地 ComfyUI inpaint（彻底去灰矩形）+ Anton stencil 字体写相似词。

路线（本地 ComfyUI，零下载，符合"做图只许 ComfyUI"硬规则）：
  1. 用 PIL 生成文字带 mask（三条 band 白块，alpha=255），存 ComfyUI/input
  2. 以 v313 AI 底图(01_custom_1.jpg) 为 image，Proteus(SDXL) + Canny 锁(保狗牌/链布局)
     + SetLatentNoiseMask 只重绘文字带 → inpaint 出与四周无缝融合的迷彩（无矩形）
  3. PIL 用 Anton-Regular.ttf(stencil 军事风) 写相似词：
       small = "WE HONOR OUR HEROES"
       big1  = "BRAVE"
       big2  = "LEGION"

用法：
  python v314_inpaint_text.py <base_img> <out_img> [--seed N]
"""
import sys
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMFY_INPUT = os.path.join(ROOT, "ComfyUI", "input")
sys.path.insert(0, HERE)

from engine.comfy_client import ComfyClient
from pipelines.build import (
    _checkpoint_node, _load_image_node, _clip_nodes, _sampler_node,
    _vae_decode, _save_node, _canny_node, _controlnet_loader_node,
    _controlnet_apply,
)

# 文字带（相对原图比例，基于 v313 AI 底图实测文字位置 y0.22~0.47）
# 比 v312_text_fission 略放宽，确保 AI 废字整块被 inpaint 覆盖
BANDS = [
    # (y0, y1, x0, x1)  相对比例
    (0.218, 0.292, 0.24, 0.76),   # small: WE HONOR OUR HEROES
    (0.292, 0.378, 0.22, 0.78),   # big1:  BRAVE
    (0.378, 0.474, 0.18, 0.82),   # big2:  LEGION
]

SIMILAR_WORDS = [
    "WE HONOR OUR HEROES",
    "BRAVE",
    "LEGION",
]


def make_mask(base_img: str, mask_path: str):
    """生成文字带 mask（RGBA，alpha=0 在 bands 内=透明=inpaint，alpha=255=不透明=保留）。
    本机 ComfyUI 链路下 alpha=1 区域被原样保留、alpha=0 区域被 SetLatentNoiseMask 重绘。"""
    im = Image.open(base_img).convert("RGB")
    W, H = im.size
    # 全图 alpha=255（保留），bands 区改为 alpha=0（重绘）
    mask = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    px = mask.load()
    for (y0, y1, x0, x1) in BANDS:
        yy0, yy1 = int(y0 * H), int(y1 * H)
        xx0, xx1 = int(x0 * W), int(x1 * W)
        for y in range(yy0, yy1):
            for x in range(xx0, xx1):
                px[x, y] = (0, 0, 0, 0)
    mask.save(mask_path, "PNG")
    print(f"[mask] saved {mask_path}  size={W}x{H}  bands={len(BANDS)} (inverted: alpha=0=inpaint)")
    return W, H


def build_inpaint(base_name: str, mask_name: str, params: dict, job_id: str) -> dict:
    style = params.get("style_prompt",
        "gray pixel camouflage pattern background, brushed steel dog tag on ball chain, "
        "military commemorative emblem, flat vector illustration, muted military gray tones, "
        "clean empty area, no text")
    neg = params.get("negative_prompt",
        "text, letters, words, writing, caption, watermark, signature, blurry, low quality, "
        "deformed, extra chain, oversaturated, jpeg artifacts")
    steps = int(params.get("steps", 30))
    cfg = float(params.get("cfg", 5.0))
    denoise = float(params.get("denoise", 0.62))
    seed = int(params.get("seed", 0))
    cn_strength = float(params.get("controlnet_strength", 0.5))

    g = {}
    g.update(_checkpoint_node(1))                 # (1,0)MODEL (1,1)CLIP (1,2)VAE
    g.update(_load_image_node(2, base_name))      # AI 底图 IMAGE
    g.update(_load_image_node(3, mask_name))      # mask IMAGE+MASK
    # VAEEncode 底图
    g.update({str(4): {"class_type": "VAEEncode",
                       "inputs": {"pixels": [str(2), 0], "vae": [str(1), 2]}}})
    # SetLatentNoiseMask：只重绘 mask 区
    g.update({str(5): {"class_type": "SetLatentNoiseMask",
                       "inputs": {"samples": [str(4), 0],
                                  "mask": [str(3), 1]}}})
    # Canny 锁（保狗牌/链布局，避免 inpaint 时把下方狗牌也改了）
    cn_name = "controlnet-canny-sdxl-1.0.fp16.safetensors"
    g.update(_controlnet_loader_node(70, cn_name))
    g.update(_canny_node(71, 2, low=0.4, high=0.8))
    g.update(_clip_nodes(7, 8, 1, style, neg))
    g.update(_controlnet_apply(72, 7, 8, 70, 71, cn_strength,
                               start_percent=0.0, end_percent=0.9))
    # KSampler 只重绘 mask 区（denoise<1 → 未遮区域原样保留）
    g.update(_sampler_node(10, 1, 72, 72, 5,
                           {"seed": seed, "steps": steps, "cfg": cfg, "denoise": denoise}))
    g.update(_vae_decode(12, 10, 1))
    g.update(_save_node(15, 12, f"{job_id}/v314"))
    return g


def draw_words(out_base: str, out_final: str, W: int, H: int):
    """在 inpaint 后的干净 camo 上用 Anton stencil 写相似词。"""
    im = Image.open(out_base).convert("RGB")
    d = ImageDraw.Draw(im)
    font_dir = os.path.join(ROOT, "ComfyUI", "models", "fonts")

    def font(sz):
        for cand in ["Anton-Regular.ttf", "Arial_Unicode.ttf", "arial.ttf"]:
            p = os.path.join(font_dir, cand)
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, sz)
                except Exception:
                    pass
        return ImageFont.load_default()

    ink = (22, 22, 22)  # 近黑，军事 stencil 风
    for (y0, y1, x0, x1), word in zip(BANDS, SIMILAR_WORDS):
        band_h = (y1 - y0) * H
        band_w = (x1 - x0) * W
        cy = (y0 + y1) / 2 * H
        cx = (x0 + x1) / 2 * W
        # 字号：大字约占 band 高度的 72%，小字约 45%（小字更长需更窄）
        fs = int(band_h * (0.72 if band_h > 140 else 0.42))
        f = font(fs)
        # 自适应收窄字号直到文本宽度不超出 band 宽
        for _ in range(12):
            bb = d.textbbox((0, 0), word, font=f, anchor="mm")
            tw = bb[2] - bb[0]
            if tw <= band_w * 0.94:
                break
            fs = int(fs * 0.92)
            f = font(fs)
        d.text((cx, cy), word, fill=ink, font=f, anchor="mm")
        print(f"[text] '{word}'  fontsize={fs}  center=({cx:.0f},{cy:.0f})")
    im.save(out_final, "JPEG", quality=92)
    print(f"[out] saved {out_final}")


def main():
    if len(sys.argv) < 3:
        print("用法: python v314_inpaint_text.py <base_img> <out_img> [--seed N]")
        sys.exit(1)
    base_img = sys.argv[1]
    out_final = sys.argv[2]
    seed = 0
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])

    mask_name = "v314_mask.png"
    mask_path = os.path.join(COMFY_INPUT, mask_name)
    base_name = os.path.basename(base_img)
    # 确保底图也在 ComfyUI/input
    if not os.path.dirname(os.path.abspath(base_img)).lower().startswith(COMFY_INPUT.lower()):
        import shutil
        dst = os.path.join(COMFY_INPUT, base_name)
        if not os.path.exists(dst):
            shutil.copy(base_img, dst)
        base_name = dst if False else base_name

    W, H = make_mask(base_img, mask_path)

    client = ComfyClient()
    params = dict(seed=seed, steps=30, cfg=5.0, denoise=0.62,
                 controlnet_strength=0.5)
    g = build_inpaint(base_name, mask_name, params, "v314")
    print("[run] queue inpaint workflow ...")
    res = client.run(g, timeout=600)
    # 取 SaveImage(15) 产出（client.run 返回的已是图片 bytes）
    imgs = []
    for node, lst in res.items():
        imgs.extend(lst)
    if not imgs:
        raise RuntimeError("inpaint 无产出")
    os.makedirs(os.path.dirname(out_final), exist_ok=True)
    tmp = os.path.join(os.path.dirname(out_final), "_v314_inpaint_raw.png")
    with open(tmp, "wb") as f:
        f.write(imgs[0])
    print(f"[run] inpaint raw saved {tmp}")

    draw_words(tmp, out_final, W, H)
    # 清理临时 mask / raw
    try:
        os.remove(mask_path)
    except Exception:
        pass
    print("[done]")


if __name__ == "__main__":
    main()
