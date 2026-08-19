"""
ComfyUI 工作流构造器（API Format）。
生成 MVP 的两种裂变模式：
  - mode1 换景换风格：IP-Adapter(相似度滑杆) + 风格 prompt + KSampler
  - mode2 内容重绘：img2img(重绘幅度) + IP-Adapter(保主体) + KSampler
每批 4 张：EmptyLatentImage.batch_size=4（mode2 用 VAEEncode 代替）。
注：节点 class_type 以 cubiq/ComfyUI_IPAdapter_plus 为准；启动后可用
    ComfyClient 查询 /object_info 校验，如有出入再微调。
"""
from config import (SDXL_CHECKPOINT, SDXL_VAE, IPADAPTER_SDXL,
                    DEFAULTS)


def _checkpoint_node(n, ckpt=SDXL_CHECKPOINT):
    return {str(n): {"class_type": "CheckpointLoaderSimple",
                     "inputs": {"ckpt_name": ckpt}}}


def _ipadapter_loader_node(n, model_name=IPADAPTER_SDXL, weight=0.6):
    return {str(n): {"class_type": "IPAdapterModelLoader",
                     "inputs": {"preset": "STANDARD (medium strength)",
                                "model_name": model_name,
                                "weight": weight}}}


def _load_image_node(n, filename):
    return {str(n): {"class_type": "LoadImage",
                     "inputs": {"image": filename, "upload": "image"}}}


def _clip_nodes(n_pos, n_neg, clip_from, pos_text, neg_text):
    return {
        str(n_pos): {"class_type": "CLIPTextEncode",
                     "inputs": {"text": pos_text, "clip": [str(clip_from), 0]}},
        str(n_neg): {"class_type": "CLIPTextEncode",
                     "inputs": {"text": neg_text, "clip": [str(clip_from), 0]}},
    }


def _sampler_node(n, model_from, pos_from, neg_from, latent_from, params):
    return {str(n): {"class_type": "KSampler",
                     "inputs": {
                         "seed": params.get("seed", 0),
                         "steps": params.get("steps", DEFAULTS["steps"]),
                         "cfg": params.get("cfg", DEFAULTS["cfg"]),
                         "sampler_name": "euler",
                         "scheduler": "normal",
                         "denoise": params.get("denoise", 1.0),
                         "model": [str(model_from), 0],
                         "positive": [str(pos_from), 0],
                         "negative": [str(neg_from), 0],
                         "latent_image": [str(latent_from), 0],
                     }}}


def _vae_decode(n, samples_from, vae_from):
    return {str(n): {"class_type": "VAEDecode",
                     "inputs": {"samples": [str(samples_from), 0],
                                "vae": [str(vae_from), 1]}}}


def _save_node(n, images_from, prefix):
    return {str(n): {"class_type": "SaveImage",
                     "inputs": {"images": [str(images_from), 0],
                                "filename_prefix": prefix}}}


def _ipadapter_apply(n, model_from, ipadapter_from, image_from, weight,
                     start_at=0.0, end_at=1.0):
    return {str(n): {"class_type": "IPAdapter",
                     "inputs": {
                         "model": [str(model_from), 0],
                         "ipadapter": [str(ipadapter_from), 0],
                         "image": [str(image_from), 0],
                         "weight": weight,
                         "start_at": start_at,
                         "end_at": end_at,
                     }}}


def build_mode1(original_filename: str, params: dict, job_id: str) -> dict:
    """换景换风格：保持与原图相似度(weight)，用 prompt 换场景/风格。"""
    w = params.get("similarity", DEFAULTS["similarity"])
    style = params.get("style_prompt", "")
    neg = params.get("negative_prompt", "low quality, blurry, deformed, watermark, text")
    g = {}
    g.update(_checkpoint_node(1))
    g.update(_ipadapter_loader_node(2, weight=w))
    g.update(_load_image_node(3, original_filename))
    g.update(_ipadapter_apply(4, 1, 2, 3, w))
    g.update(_clip_nodes(5, 6, 1, style, neg))
    g.update({str(7): {"class_type": "EmptyLatentImage",
                       "inputs": {"width": params.get("width", DEFAULTS["width"]),
                                  "height": params.get("height", DEFAULTS["height"]),
                                  "batch_size": params.get("batch_per_run", DEFAULTS["batch_per_run"])}}})
    g.update(_sampler_node(8, 4, 5, 6, 7, {**params, "denoise": 1.0}))
    g.update(_vae_decode(9, 8, 1))
    g.update(_save_node(10, 9, f"{job_id}/mode1"))
    return g


def build_mode2(original_filename: str, params: dict, job_id: str) -> dict:
    """内容重绘：img2img(重绘幅度 denoise) + IP-Adapter(保主体相似度)。"""
    w = params.get("similarity", DEFAULTS["similarity"])
    denoise = params.get("redraw_amount", DEFAULTS["redraw_amount"])
    style = params.get("style_prompt", "")
    neg = params.get("negative_prompt", "low quality, blurry, deformed, watermark, text")
    g = {}
    g.update(_checkpoint_node(1))
    g.update(_ipadapter_loader_node(2, weight=w))
    g.update(_load_image_node(3, original_filename))
    g.update(_ipadapter_apply(4, 1, 2, 3, w))
    g.update(_clip_nodes(5, 6, 1, style, neg))
    # img2img：原图 VAE 编码为初始 latent
    g.update({str(7): {"class_type": "VAEEncode",
                       "inputs": {"pixels": [str(3), 0], "vae": [str(1), 1]}}})
    g.update(_sampler_node(8, 4, 5, 6, 7, {**params, "denoise": denoise}))
    g.update(_vae_decode(9, 8, 1))
    g.update(_save_node(10, 9, f"{job_id}/mode2"))
    return g


def build(mode: str, original_filename: str, params: dict, job_id: str) -> dict:
    if mode == "mode1":
        return build_mode1(original_filename, params, job_id)
    elif mode == "mode2":
        return build_mode2(original_filename, params, job_id)
    raise ValueError(f"未知模式: {mode}")
