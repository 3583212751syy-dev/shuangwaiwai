r"""v9.8 illust_1：img2img 直接重画（路线 A，用户 2026-08-25 17:0x 拍板）。

策略：把 HD 参考图直接喂进 VAEEncode → KSampler denoise 0.55 重画。
- 不用 ControlNet（之前 Canny 0.85 + IPAdapter 0.6 仍锁不住孔雀，4 版未达标）
- 不用 IPAdapter（img2img 本身就以原图为起点）
- 不用 LoRA（原图已是矢量风，img2img 自然保留）
- 不依赖 comfyui_controlnet_aux 节点

管线：
  LoadImage(HD) → ImageScale(1024²) → VAEEncode → KSampler(denoise 0.55, steps 40)
  → KSampler(denoise 0.20, steps 30, hires 细化) → VAEDecode
  → 4x_NMKD-Siax_200k 真实超分 → SaveImage

prompt：A 风格（黑白矢量）+ 边框强约束（与 v9.7 相同的 POSITIVE/NEGATIVE
  但去掉 "no halftone" 强调 + 加 "no text/letters" 强约束避免 AI 拼字）。
"""
import os
import sys
import time
import random
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import demo_batch_6x4 as db
from engine.comfy_client import ComfyClient


def _checkpoint_node(n, ckpt):
    return {str(n): {"class_type": "CheckpointLoaderSimple",
                     "inputs": {"ckpt_name": ckpt}}}


def _load_image(n, filename):
    return {str(n): {"class_type": "LoadImage",
                     "inputs": {"image": filename, "upload": "image"}}}


def _image_scale(n, image_from, w, h, method="lanczos", crop="center"):
    return {str(n): {"class_type": "ImageScale",
                     "inputs": {"image": [str(image_from), 0],
                                "width": w,
                                "height": h,
                                "upscale_method": method,
                                "crop": crop}}}


def _vae_encode(n, pixels_from, vae_from):
    return {str(n): {"class_type": "VAEEncode",
                     "inputs": {"pixels": [str(pixels_from), 0],
                                "vae": [str(vae_from), 2]}}}


def _clip_encode(n, clip_from, text):
    return {str(n): {"class_type": "CLIPTextEncode",
                     "inputs": {"text": text, "clip": [str(clip_from), 1]}}}


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


def build_v98_workflow(seed_name: str, params: dict) -> dict:
    """v9.8 纯 img2img + hires + NMKD-Siax。"""
    g = {}
    # 1: Checkpoint (Proteus)
    g.update(_checkpoint_node(1, params["ckpt"]))
    # 2: LoadImage (HD 参考)
    g.update(_load_image(2, seed_name))
    # 3: ImageScale → 1024x1024 (center crop 居中裁剪)
    g.update(_image_scale(3, 2, 1024, 1024, method="lanczos", crop="center"))
    # 4: VAEEncode (vae 从节点 1 取，索引 2)
    g.update(_vae_encode(4, 3, 1))
    # 5, 6: CLIP encode
    g.update(_clip_encode(5, 1, params["positive"]))
    g.update(_clip_encode(6, 1, params["negative"]))
    # 7: KSampler 1 (img2img, denoise 0.55, steps 40, cfg 7.0)
    g.update(_ksampler(7, 1, 5, 6, 4, params["seed"],
                       steps=40, cfg=7.0, denoise=0.55))
    # 8: KSampler 2 (hires 细化, denoise 0.20, steps 30, cfg 7.0)
    g.update(_ksampler(8, 1, 5, 6, 7, params["seed"] + 1,
                       steps=30, cfg=7.0, denoise=0.20))
    # 9: VAEDecode
    g.update(_vae_decode(9, 8, 1))
    # 10: UpscaleModelLoader (NMKD-Siax 插画超分)
    g.update(_upscale_loader(10, "4x_NMKD-Siax_200k.pth"))
    # 11: ImageUpscaleWithModel
    g.update(_image_upscale(11, 10, 9))
    # 12: SaveImage
    g.update(_save_image(12, 11, "v98_illust_1"))
    return g


# A 风格（黑白矢量）+ 边框强约束（不再强写"peacock"，因为原图已有，
# 让 img2img 自己从参考保留主体；prompt 重点约束"矢量风+无色+无边框"）
POSITIVE = (
    "vector illustration, clean sharp linework, intricate detail, "
    "edge-to-edge, full bleed, fills entire frame, no margins, "
    "high contrast black and white, no color, no gradient, no shading, no halftone"
)

NEGATIVE = (
    "color, colorful, gradient, shading, halftone, painterly, photographic, "
    "3d render, realistic texture, fabric folds, wrinkles, shadows, "
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
    "illegible text, scribbles resembling text, signature"
)


def main():
    orig_seed = "pinterest_illust_1"
    hd = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}_hd.png")
    jpg = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}.jpg")
    if os.path.exists(hd):
        ref = hd
        ext = ".png"
        print(f"[v9.8] img2img HD参考 {os.path.basename(hd)} "
              f"({os.path.getsize(hd)/1024/1024:.1f}MB)")
    else:
        ref = jpg
        ext = ".jpg"
        print(f"[v9.8] fallback 原图 {os.path.basename(jpg)}")

    out_dir = os.path.join(db.JOBS_BASE, f"smoke_v98_illust_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    seed_name = f"smoke_v98_{int(time.time()*1000)}{ext}"
    shutil.copy(ref, os.path.join(db.COMFYUI_INPUT, seed_name))

    seed = random.randint(1, 999999999)
    params = {
        "ckpt": "ProteusV0.4.safetensors",
        "seed": seed,
        "positive": POSITIVE,
        "negative": NEGATIVE,
    }

    g = build_v98_workflow(seed_name, params)
    print(f"[v9.8 cfg] ckpt={params['ckpt']}  seed={seed}")
    print(f"[v9.8 cfg] img2img denoise=0.55 steps=40 cfg=7.0")
    print(f"[v9.8 cfg] hires  denoise=0.20 steps=30 cfg=7.0")
    print(f"[v9.8 cfg] upscale=4x_NMKD-Siax_200k (4096² target)")
    print(f"[v9.8 cfg] positive=B&W vector + edge-to-edge  ({len(POSITIVE)} chars)")
    print(f"[v9.8 cfg] negative=de-color/de-border/de-text/de-noise  ({len(NEGATIVE)} chars)")

    client = ComfyClient()
    t0 = time.time()
    try:
        res = client.run(g, timeout=600)
        data = next(iter(res.values()))[0]
        out_path = os.path.join(out_dir, f"illust_1_peacock_floral.jpg")
        with open(out_path, "wb") as f:
            f.write(data)
        dt = time.time() - t0
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"\n[OK] {dt:.0f}s  {size_mb:.1f}MB  {out_path}")
        return out_path
    except Exception as e:
        print(f"\n[FAIL] {repr(e)}")
        return None


if __name__ == "__main__":
    main()
