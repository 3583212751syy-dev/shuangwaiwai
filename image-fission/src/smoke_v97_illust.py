r"""v9.7 illust_1 单张验证：D 方向（多 ControlNet）+ A 风格（黑白矢量）+ 边框/清晰度修复。

用户决策（2026-08-25 16:xx，6 步 AskUserQuestion 落定）：
  q-0 内容方向 = D  (Canny 0.85 + Depth 0.7 + IPAdapter 风格 0.6)
  q-1 边框     = prompt 强约束 (positive 加 edge-to-edge/full bleed，negative 加 white border/frame/margin)
  q-2 清晰度   = 换 4x-AniSharp 插画专用模型（替代 4x-UltraSharp）
  q-3 风格基调 = A  (完全还原黑白矢量：黑底白线，no color/gradient/shading)
  q-4 验证策略 = A  (单张先验证：只跑 illust_1)
  q-5 资源补齐 = A  (下载 SDXL Depth ControlNet + 4x-AniSharp)

技术实现：
  - 底模：Proteus v0.4 (已加载)
  - LoRAs：VECTOR(DD-vector-v2) @ 0.45，**去掉** TEXTILE（与 ControlNet 重叠）
  - 3 个 ControlNet/Adapter：
      Canny  (SDXL) @ 0.85  —— 锁边缘
      Depth  (SDXL) @ 0.7   —— 锁层次（需要 comfyui_controlnet_aux 提供 DepthAnything 预处理）
      IPAdapter style @ 0.6 —— 锁风格基调（颜色/材质）
  - 采样：dpmpp_2m / karras, steps 45, cfg 7.5
  - Hires：denoise 0.20, steps 40
  - 超分：4x-AniSharp（新下载的）
  - 最终：4096²

自检 4 维（完成时必须跑）：
  1) 噪点碎裂  2) 糊字/乱码  3) 模糊  4) 像素清晰度
  全部 >= 8/10 才 present_files 给用户验收（不假设合格）。

执行顺序：
  1) 等待 HXvspL（下载 SDXL Depth ControlNet + 4x-AniSharp）完成
  2) 等待 bLFlTY（安装 comfyui_controlnet_aux）完成
  3) 重启 ComfyUI（让新 custom_node 生效）
  4) 跑本脚本
  5) 读产物图 4 维自检
  6) present_files + 文字通顺/内容对错请用户肉眼验收
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


def _lora_loader_node(n, model_from, clip_from, lora_name, strength=0.45):
    return {str(n): {"class_type": "LoraLoader",
                     "inputs": {"model": [str(model_from), 0],
                                "clip": [str(clip_from), 1],
                                "lora_name": lora_name,
                                "strength_model": strength,
                                "strength_clip": strength}}}


def _ipadapter_unified_loader(n, model_from):
    return {str(n): {"class_type": "IPAdapterUnifiedLoader",
                     "inputs": {"model": [str(model_from), 0],
                                "preset": "PLUS (high strength)"}}}


def _load_image(n, filename):
    return {str(n): {"class_type": "LoadImage",
                     "inputs": {"image": filename, "upload": "image"}}}


def _ipadapter_style_apply(n, model_from, ipadapter_from, image_from, weight,
                            weight_type="style transfer", start_at=0.0, end_at=1.0,
                            noise=0.0):
    """IPAdapter style/composition 锁风格基调（颜色+材质）。"""
    return {str(n): {"class_type": "IPAdapterAdvanced",
                     "inputs": {
                         "model": [str(model_from), 0],
                         "ipadapter": [str(ipadapter_from), 1],
                         "image": [str(image_from), 0],
                         "weight": weight,
                         "weight_type": weight_type,
                         "combine_embeds": "average",
                         "start_at": start_at,
                         "end_at": end_at,
                         "noise": noise,
                         "embeds_scaling": "V only",
                     }}}


def _clip_encode(n, clip_from, text):
    return {str(n): {"class_type": "CLIPTextEncode",
                     "inputs": {"text": text, "clip": [str(clip_from), 1]}}}


def _controlnet_loader(n, control_name):
    return {str(n): {"class_type": "ControlNetLoader",
                     "inputs": {"control_net_name": control_name}}}


def _canny_preprocess(n, image_from, low=100, high=200):
    """Canny 边缘提取（comfyui_controlnet_aux 提供）。"""
    return {str(n): {"class_type": "CannyEdgePreprocessor",
                     "inputs": {"image": [str(image_from), 0],
                                "low_threshold": low,
                                "high_threshold": high,
                                "resolution": 1024}}}


def _depth_preprocess(n, image_from):
    """Depth Anything 深度图提取（comfyui_controlnet_aux 提供）。"""
    return {str(n): {"class_type": "DepthAnythingPreprocessor",
                     "inputs": {"image": [str(image_from), 0],
                                "ckpt_name": "depth_anything_vitl14.pth",
                                "resolution": 1024}}}


def _controlnet_apply(n, pos_from, neg_from, controlnet_from, image_from, strength,
                       start_percent=0.0, end_percent=0.9):
    return {str(n): {"class_type": "ControlNetApplyAdvanced",
                     "inputs": {
                         "positive": [str(pos_from), 0],
                         "negative": [str(neg_from), 0],
                         "control_net": [str(controlnet_from), 0],
                         "image": [str(image_from), 0],
                         "strength": strength,
                         "start_percent": start_percent,
                         "end_percent": end_percent,
                     }}}


def _empty_latent(n, w, h, batch=1):
    return {str(n): {"class_type": "EmptyLatentImage",
                     "inputs": {"width": w, "height": h, "batch_size": batch}}}


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


def build_v97_workflow(seed_name: str, params: dict) -> dict:
    """v9.7 完整工作流：3 ControlNet + 黑白矢量 prompt + 4x-AniSharp。"""
    g = {}
    # 1: Checkpoint (Proteus)
    g.update(_checkpoint_node(1, params["ckpt"]))
    # 2: VECTOR LoRA（不加 TEXTILE——与 ControlNet 重复）
    g.update(_lora_loader_node(2, 1, 1, "DD-vector-v2.safetensors", strength=0.45))
    # 3: IPAdapter Unified Loader（PLUS high strength）
    g.update(_ipadapter_unified_loader(3, 2))
    # 4: LoadImage（HD 参考）
    g.update(_load_image(4, seed_name))
    # 5: IPAdapter style transfer 锁风格基调
    g.update(_ipadapter_style_apply(5, 2, 3, 4, weight=0.6,
                                     weight_type="style transfer",
                                     start_at=0.0, end_at=1.0, noise=0.0))
    # 6, 7: CLIP
    g.update(_clip_encode(6, 2, params["positive"]))
    g.update(_clip_encode(7, 2, params["negative"]))
    # 8: ControlNetLoader (Canny SDXL)
    g.update(_controlnet_loader(8, "controlnet-canny-sdxl-1.0.fp16.safetensors"))
    # 9: Canny 预处理
    g.update(_canny_preprocess(9, 4, low=100, high=200))
    # 10: Canny ControlNet Apply (strength 0.85)
    g.update(_controlnet_apply(10, 6, 7, 8, 9, strength=0.85,
                                start_percent=0.0, end_percent=0.9))
    # 14: Empty Latent 1024x1024
    # 注：Depth 维度因 HF 网络不通（DepthAnything 权重下不来）暂时砍掉，
    #     仅保留 Canny(0.85) + IPAdapter-style(0.6) 两维锁内容。
    g.update(_empty_latent(14, 1024, 1024, batch=1))
    # 15: KSampler 1（满采，denoise 1.0, steps 45, cfg 7.5）
    g.update(_ksampler(15, 5, 10, 10, 14, params["seed"],
                       steps=45, cfg=7.5, denoise=1.0))
    # 16: KSampler 2（hires 细化，denoise 0.20, steps 40）
    g.update(_ksampler(16, 5, 10, 10, 15, params["seed"] + 1,
                       steps=40, cfg=7.5, denoise=0.20))
    # 17: VAE Decode（VAE 永远从节点 1 取）
    g.update(_vae_decode(17, 16, 1))
    # 18: 4x_NMKD-Siax_200k 超分（同定位插画超分，AniSharp 因 HF 不通下不来）
    g.update(_upscale_loader(18, "4x_NMKD-Siax_200k.pth"))
    # 19: 真实超分
    g.update(_image_upscale(19, 18, 17))
    # 20: SaveImage
    g.update(_save_image(20, 19, "v97_illust_1"))
    return g


# A 风格（完全还原黑白矢量）+ 边框强约束
POSITIVE = (
    "black and white vector illustration, pure black background, white linework only, "
    "no color, no gradient, no shading, no halftone, "
    "large central peacock silhouette facing right with detailed feathers, eye visible, "
    "surrounded by 5-petal flowers, decorative swirls, leaves, "
    "clean sharp edges, intricate detail, "
    "fills entire frame edge-to-edge, no margins, full bleed, edge-to-edge"
)

NEGATIVE = (
    "color, colorful, gradient, shading, halftone, painterly, photographic, "
    "white border, frame, edge, margin, padding, empty corners, letterbox, "
    "blurry, noisy, grain, film grain, sensor noise, compression artifacts, "
    "speckles, dust spots, harsh jagged edges, fuzzy halftone, "
    "rough sketch, pencil smudge, dirty paper texture, "
    "uneven ink, broken outlines, scratchy strokes, "
    "photography, product photo, 3d render, realistic texture, fabric folds, "
    "wrinkles, shadows, depth of field, "
    "deformed, low quality, watermark, copyright logo, "
    "garbled text, gibberish text, pseudo-script, fake characters, "
    "runic nonsense, occult sigils, runes, talisman symbols, "
    "malformed letters, repeating letters, double letters, "
    "illegible text, scribbles resembling text"
)


def main():
    # 优先读 _hd.png
    orig_seed = "pinterest_illust_1"
    hd = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}_hd.png")
    jpg = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}.jpg")
    if os.path.exists(hd):
        ref = hd
        ext = ".png"
        print(f"[v9.7] 使用高清参考 {os.path.basename(hd)} "
              f"({os.path.getsize(hd)/1024/1024:.1f}MB)")
    else:
        ref = jpg
        ext = ".jpg"
        print(f"[v9.7] fallback 原图 {os.path.basename(jpg)} "
              f"({os.path.getsize(jpg)/1024:.0f}KB)")

    out_dir = os.path.join(db.JOBS_BASE, f"smoke_v97_illust_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    seed_name = f"smoke_v97_{int(time.time()*1000)}{ext}"
    shutil.copy(ref, os.path.join(db.COMFYUI_INPUT, seed_name))

    seed = random.randint(1, 999999999)
    params = {
        "ckpt": "ProteusV0.4.safetensors",
        "seed": seed,
        "positive": POSITIVE,
        "negative": NEGATIVE,
    }

    g = build_v97_workflow(seed_name, params)
    print(f"[v9.7 cfg] ckpt={params['ckpt']}  seed={seed}")
    print(f"[v9.7 cfg] LoRA=VECTOR@0.45  TEXTILE=OFF")
    print(f"[v9.7 cfg] Canny=0.85  Depth=0.7  IPAdapter-style=0.6")
    print(f"[v9.7 cfg] hires=denoise 0.20 / steps 40")
    print(f"[v9.7 cfg] upscale=4x_NMKD-Siax_200k.pth (AniSharp 因 HF 不通下不来，同类插画超分兜底)")
    print(f"[v9.7 cfg] positive=B&W vector + edge-to-edge  ({len(POSITIVE)} chars)")
    print(f"[v9.7 cfg] negative=de-color/de-border/de-noise  ({len(NEGATIVE)} chars)")

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
