r"""v10.0 同风格+同主题类裂变（用户 2026-08-25 17:30 拍板方向 B）。

策略：完全脱开参考图，靠 prompt 主体词轮换 + VECTOR LoRA 自洽。
- 不用 IPAdapter / 不用 ControlNet / 不用参考图（之前 style 模式反而把"主体"也锁死）
- LoRA: VECTOR(DD-vector-v2) @ 0.55 提供矢量线稿风
- 底模: Proteus v0.4
- Prompt 主体在长尾鸟主题类轮换（phoenix / crane / heron / hawk / raven / swallow）
- HiRes: denoise 0.20, steps 40
- 超分: 4x_NMKD-Siax（黑白矢量专用，治 v9.8 锐利度不够）
- 黑度/边框/锐利度 必须 8/10+ 才 present（AI 自检仅作快速筛选，最终用户验证）

管线：
  Checkpoint(Proteus) → LoraLoader(VECTOR 0.55)
  → CLIP×2 → EmptyLatentImage(1024²)
  → KSampler1(steps=45, cfg=7.5, denoise=1.0)
  → KSampler2(steps=40, cfg=7.5, denoise=0.20, hires 细化)
  → VAEDecode (vae from ckpt)
  → 4x_NMKD-Siax_200k 真实超分 → SaveImage
"""
import os
import sys
import time
import random
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import demo_batch_6x4 as db
from engine.comfy_client import ComfyClient


def _checkpoint_node(n, ckpt):
    return {str(n): {"class_type": "CheckpointLoaderSimple",
                     "inputs": {"ckpt_name": ckpt}}}


def _lora_loader(n, model_from, clip_from, lora_name, strength=0.55):
    return {str(n): {"class_type": "LoraLoader",
                     "inputs": {"model": [str(model_from), 0],
                                "clip": [str(clip_from), 1],
                                "lora_name": lora_name,
                                "strength_model": strength,
                                "strength_clip": strength}}}


def _clip_encode(n, clip_from, text):
    return {str(n): {"class_type": "CLIPTextEncode",
                     "inputs": {"text": text, "clip": [str(clip_from), 1]}}}


def _empty_latent(n, w, h):
    return {str(n): {"class_type": "EmptyLatentImage",
                     "inputs": {"width": w, "height": h, "batch_size": 1}}}


def _ksampler(n, model_from, pos_from, neg_from, latent_from, seed, steps, cfg, denoise=1.0):
    return {str(n): {"class_type": "KSampler",
                     "inputs": {
                         "seed": seed,
                         "steps": steps,
                         "cfg": cfg,
                         "sampler_name": "dpmpp_2m",
                         "scheduler": "karras",
                         "denoise": denoise,
                         "model": [str(model_from), 0],
                         "positive": [str(pos_from), 0],
                         "negative": [str(neg_from), 0],
                         "latent_image": [str(latent_from), 0],
                     }}}


def _vae_decode(n, samples_from, vae_from):
    return {str(n): {"class_type": "VAEDecode",
                     "inputs": {"samples": [str(samples_from), 0],
                                "vae": [str(vae_from), 2]}}}


def _upscale_loader(n, model_name):
    return {str(n): {"class_type": "UpscaleModelLoader",
                     "inputs": {"model_name": model_name}}}


def _image_upscale(n, model_from, image_from):
    return {str(n): {"class_type": "ImageUpscaleWithModel",
                     "inputs": {"upscale_model": [str(model_from), 0],
                                "image": [str(image_from), 0]}}}


def _save_image(n, images_from, prefix):
    return {str(n): {"class_type": "SaveImage",
                     "inputs": {"images": [str(images_from), 0],
                                "filename_prefix": prefix}}}


def build_v100_workflow(params: dict) -> dict:
    """v10.0 T2I + VECTOR LoRA + hires + NMKD-Siax。"""
    g = {}
    # 1: Checkpoint (Proteus)
    g.update(_checkpoint_node(1, params["ckpt"]))
    # 2: LoraLoader (VECTOR @ 0.55, 接节点 1 的 model/clip)
    g.update(_lora_loader(2, 1, 1, "DD-vector-v2.safetensors",
                          strength=params["lora_strength"]))
    # 3, 4: CLIP encode (clip 来自 LoRA 节点 2 的输出 1)
    g.update(_clip_encode(3, 2, params["positive"]))
    g.update(_clip_encode(4, 2, params["negative"]))
    # 5: EmptyLatentImage (1024²)
    g.update(_empty_latent(5, 1024, 1024))
    # 6: KSampler 1 (满采, denoise 1.0, steps 45, cfg 7.5) — 接 LoRA model
    g.update(_ksampler(6, 2, 3, 4, 5, params["seed"],
                       steps=45, cfg=7.5, denoise=1.0))
    # 7: KSampler 2 (hires 细化, denoise 0.20, steps 40, cfg 7.5)
    g.update(_ksampler(7, 2, 3, 4, 6, params["seed"] + 1,
                       steps=40, cfg=7.5, denoise=0.20))
    # 8: VAEDecode (vae from ckpt 1, 索引 2)
    g.update(_vae_decode(8, 7, 1))
    # 9: UpscaleModelLoader (NMKD-Siax 插画超分)
    g.update(_upscale_loader(9, "4x_NMKD-Siax_200k.pth"))
    # 10: ImageUpscaleWithModel
    g.update(_image_upscale(10, 9, 8))
    # 11: SaveImage
    g.update(_save_image(11, 10, params["filename_prefix"]))
    return g


def prompt_for_subject(bird: str) -> str:
    """长尾鸟主题 prompt 模板——同风格（矢量+黑底+雄壮+卷草花卉），换内容。"""
    return (
        f"vector illustration t-shirt graphic, {bird}, "
        "majestic long-tailed bird with flowing detailed feathers, "
        "tropical plumage, sharp beak, fierce eyes, "
        "surrounded by 5-petal flowers, lily blossoms, "
        "decorative ornamental swirls, vine scrolls, filigree accents, "
        "ornamental baroque pattern, intricate detail, "
        "sharp clean edges, refined linework, "
        "high contrast black and white, "
        "pure black background, white linework only, no color, "
        "no gradient, no shading, no halftone, no painterly texture, "
        "edge-to-edge, full bleed, fills entire frame, no margins, "
        "centered subject, t-shirt apparel print design, "
        "professional vector graphic, 4k detail"
    )


NEGATIVE = (
    "color, colorful, gradient, shading, halftone, painterly texture, "
    "photographic, 3d render, realistic texture, fabric folds, wrinkles, shadows, "
    "white border, frame, edge, margin, padding, empty corners, letterbox, "
    "blurry, noisy, grain, film grain, sensor noise, compression artifacts, "
    "speckles, dust spots, harsh jagged edges, fuzzy halftone, "
    "rough sketch, pencil smudge, dirty paper texture, "
    "uneven ink, broken outlines, scratchy strokes, "
    "depth of field, bokeh, lens flare, "
    "deformed, low quality, watermark, copyright logo, "
    "text, letters, words, alphabet, typography, font, "
    "garbled text, gibberish text, pseudo-script, fake characters, "
    "runic nonsense, occult sigils, runes, talisman symbols, "
    "malformed letters, repeating letters, double letters, "
    "illegible text, scribbles resembling text, signature, "
    "background scenery, environment, landscape, sky, ground"
)


def run_one(bird: str, seed: int, out_dir: str, idx: int) -> str:
    """跑一张 — bird 是长尾鸟主体词（phoenix / crane / heron / hawk / raven / swallow）。"""
    prefix = f"v100_{bird}"
    params = {
        "ckpt": "ProteusV0.4.safetensors",
        "seed": seed,
        "lora_strength": 0.55,
        "positive": prompt_for_subject(bird),
        "negative": NEGATIVE,
        "filename_prefix": prefix,
    }
    g = build_v100_workflow(params)
    print(f"\n[v10.0 #{idx}] {bird}  seed={seed}")
    print(f"  ckpt=Proteus  VECTOR=0.55  steps=45/40 cfg=7.5")
    print(f"  prompt: {bird} + 矢量+黑底+卷草花卉")
    client = ComfyClient()
    t0 = time.time()
    try:
        res = client.run(g, timeout=600)
        data = next(iter(res.values()))[0]
        out_path = os.path.join(out_dir, f"{bird}_{seed}.jpg")
        with open(out_path, "wb") as f:
            f.write(data)
        dt = time.time() - t0
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  [OK] {dt:.0f}s  {size_mb:.1f}MB  {out_path}")
        return out_path
    except Exception as e:
        print(f"  [FAIL] {repr(e)}")
        return None


def main():
    # 第一步：先 3 张不同主体验证（让用户看"换内容"是否到位）
    # 用户认可后跑剩下 3 张
    pilots = [
        ("phoenix", 100001),
        ("crane",   100002),
        ("hawk",    100003),
    ]
    out_dir = os.path.join(db.JOBS_BASE, f"smoke_v100_pilots_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[plan] 先跑 3 张 pilot（phoenix/crane/hawk）让用户验")
    print(f"[plan] 用户认可后再跑 6 张全量")
    ok = []
    for i, (bird, seed) in enumerate(pilots, 1):
        p = run_one(bird, seed, out_dir, i)
        if p:
            ok.append(p)
    print(f"\n[v10.0 done] {len(ok)}/{len(pilots)} pilots OK")
    for p in ok:
        print(f"  {p}")


if __name__ == "__main__":
    main()
