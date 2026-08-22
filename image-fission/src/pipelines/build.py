"""
ComfyUI 工作流构造器（API Format）。
生成 MVP 的两种裂变模式：
  - mode1 换景换风格：IP-Adapter(相似度滑杆) + 风格 prompt + KSampler
  - mode2 内容重绘：img2img(重绘幅度) + IP-Adapter(保主体) + KSampler
每批 4 张：EmptyLatentImage.batch_size=4（mode2 用 VAEEncode 代替）。
注：节点 class_type 以 cubiq/ComfyUI_IPAdapter_plus 为准；启动后可用
    ComfyClient 查询 /object_info 校验，如有出入再微调。
"""
from config import (SDXL_VAE, DEFAULTS)
import config as _cfg

# IP-Adapter 统一加载预设：plus 模型配 "PLUS (high strength)"，最契合 SDXL+plus
IPADAPTER_PRESET = "PLUS (high strength)"


def _checkpoint_node(n, ckpt=None):
    if ckpt is None:
        ckpt = _cfg.SDXL_CHECKPOINT
    return {str(n): {"class_type": "CheckpointLoaderSimple",
                     "inputs": {"ckpt_name": ckpt}}}


def _ipadapter_loader_node(n, model_from):
    """IPAdapterUnifiedLoader：输入 checkpoint 的 MODEL，输出带 ipadapter 的
    MODEL(0) 与 IPADAPTER(1) 对象（自动加载 clip_vision 编码器）。"""
    return {str(n): {"class_type": "IPAdapterUnifiedLoader",
                     "inputs": {"model": [str(model_from), 0],
                                "preset": IPADAPTER_PRESET}}}


def _load_image_node(n, filename):
    return {str(n): {"class_type": "LoadImage",
                     "inputs": {"image": filename, "upload": "image"}}}


def _clip_nodes(n_pos, n_neg, clip_from, pos_text, neg_text):
    return {
        str(n_pos): {"class_type": "CLIPTextEncode",
                     "inputs": {"text": pos_text, "clip": [str(clip_from), 1]}},
        str(n_neg): {"class_type": "CLIPTextEncode",
                     "inputs": {"text": neg_text, "clip": [str(clip_from), 1]}},
    }


def _sampler_node(n, model_from, pos_from, neg_from, latent_from, params):
    return {str(n): {"class_type": "KSampler",
                     "inputs": {
                         "seed": params.get("seed", 0),
                         "steps": params.get("steps", DEFAULTS["steps"]),
                         "cfg": params.get("cfg", DEFAULTS["cfg"]),
                        "sampler_name": "dpmpp_2m",
                        "scheduler": "karras",
                         "denoise": params.get("denoise", 1.0),
                         "model": [str(model_from), 0],
                         "positive": [str(pos_from), 0],
                         "negative": [str(neg_from), 0],
                         "latent_image": [str(latent_from), 0],
                     }}}


def _latent_upscale_by(n, samples_from, scale_by=1.5, method="nearest-exact"):
    """LatentUpscaleBy：潜空间放大（hires fix 第一步），无需额外模型。"""
    return {str(n): {"class_type": "LatentUpscaleBy",
                     "inputs": {"samples": [str(samples_from), 0],
                                "upscale_method": method,
                                "scale_by": scale_by}}}


def _vae_decode(n, samples_from, vae_from):
    # VAE 始终从 CheckpointLoader（节点 1）取，因为 LoRA 节点只输出 MODEL/CLIP 不带 VAE
    return {str(n): {"class_type": "VAEDecode",
                     "inputs": {"samples": [str(samples_from), 0],
                                "vae": [str(1), 2]}}}


def _save_node(n, images_from, prefix):
    return {str(n): {"class_type": "SaveImage",
                     "inputs": {"images": [str(images_from), 0],
                                "filename_prefix": prefix}}}


def _ipadapter_apply(n, model_node, ipadapter_node, image_node, weight,
                     start_at=0.0, end_at=1.0, weight_type="linear"):
    """IPAdapterAdvanced：输入 unified loader 的 MODEL(输出0) 与 IPADAPTER(输出1)，
    输出带参考图条件的 MODEL(0)。weight=与原图相似度滑杆。"""
    return {str(n): {"class_type": "IPAdapterAdvanced",
                     "inputs": {
                         "model": [str(model_node), 0],
                         "ipadapter": [str(ipadapter_node), 1],
                         "image": [str(image_node), 0],
                         "weight": weight,
                         "weight_type": weight_type,
                         "combine_embeds": "average",
                         "start_at": start_at,
                         "end_at": end_at,
                         "embeds_scaling": "V only",
                     }}}


def _lora_loader_node(n, model_from, clip_from, lora_name, strength_model=0.85, strength_clip=0.85):
    """LoRA loader — 必须在 checkpoint 之后、IPAdapter 之前（IPAdapter 需要 LoRA 后的 MODEL/CLIP）。
    lora_name 为空字符串时返回空 dict（passthrough 行为，下游直接用 checkpoint 节点）。
    model_from/clip_from 接受 (node_id, output_index) 元组或纯节点编号（默认用 0 号输出）。"""
    def _ref(src, default_out=0):
        if isinstance(src, tuple):
            return [str(src[0]), src[1]]
        return [str(src), default_out]
    if not lora_name:
        return {}
    return {str(n): {"class_type": "LoraLoader",
                     "inputs": {"model": _ref(model_from, 0),
                                "clip": _ref(clip_from, 1),       # CLIP 永远用节点 1 的输出 1
                                "lora_name": lora_name,
                                "strength_model": strength_model,
                                "strength_clip": strength_clip}}}


def _resolve_model_clip(lora_strength):
    """根据 lora 强度返回 (effective_model_node, effective_clip_node)。
    0 = 不用 LoRA（用 checkpoint 节点 1），>0 = 用了 LoRA 节点 2。"""
    if lora_strength and lora_strength > 0:
        return 2, 2
    return 1, 1


def build_mode1(original_filename: str, params: dict, job_id: str) -> dict:
    """换景换风格：保持与原图相似度(weight)，用 prompt 换场景/风格。
    可选 LoRA：通过 params["lora_name"] + params["lora_strength"] 启用（默认 0.85）。
    """
    w = params.get("similarity", DEFAULTS["similarity"])
    style = params.get("style_prompt", "")
    neg = params.get("negative_prompt", "low quality, blurry, deformed, watermark, text")
    lora_name = params.get("lora_name", "")
    lora_strength = float(params.get("lora_strength", 0.85))
    model_node, clip_node = _resolve_model_clip(lora_strength)
    g = {}
    g.update(_checkpoint_node(1))
    g.update(_lora_loader_node(2, 1, 1, lora_name, lora_strength, lora_strength))
    g.update(_ipadapter_loader_node(3, model_node))
    g.update(_load_image_node(4, original_filename))
    g.update(_ipadapter_apply(5, model_node, 3, 4, w))
    g.update(_clip_nodes(6, 7, clip_node, style, neg))
    g.update({str(8): {"class_type": "EmptyLatentImage",
                       "inputs": {"width": params.get("width", DEFAULTS["width"]),
                                  "height": params.get("height", DEFAULTS["height"]),
                                  "batch_size": params.get("batch_per_run", DEFAULTS["batch_per_run"])}}})
    g.update(_sampler_node(9, 5, 6, 7, 8, {**params, "denoise": 1.0,
                                           "steps": params.get("steps_base", DEFAULTS["steps"])}))
    g.update(_latent_upscale_by(10, 9, scale_by=params.get("hires_scale", 1.5)))
    g.update(_sampler_node(11, 5, 6, 7, 10,
                           {**params, "denoise": params.get("hires_denoise", 0.35),
                            "steps": params.get("hires_steps", 20)}))
    g.update(_vae_decode(12, 11, model_node))
    g.update(_save_node(13, 12, f"{job_id}/mode1"))
    return g


def build_mode2(original_filename: str, params: dict, job_id: str) -> dict:
    """内容重绘：img2img(重绘幅度 denoise) + IP-Adapter(保主体相似度)。"""
    w = params.get("similarity", DEFAULTS["similarity"])
    denoise = params.get("redraw_amount", DEFAULTS["redraw_amount"])
    style = params.get("style_prompt", "")
    neg = params.get("negative_prompt", "low quality, blurry, deformed, watermark, text")
    width = params.get("width", DEFAULTS["width"])
    height = params.get("height", DEFAULTS["height"])
    lora_name = params.get("lora_name", "")
    lora_strength = float(params.get("lora_strength", 0.85))
    model_node, clip_node = _resolve_model_clip(lora_strength)
    g = {}
    g.update(_checkpoint_node(1))
    g.update(_lora_loader_node(2, 1, 1, lora_name, lora_strength, lora_strength))
    g.update(_ipadapter_loader_node(3, model_node))
    g.update(_load_image_node(4, original_filename))
    g.update(_ipadapter_apply(5, model_node, 3, 4, w))
    g.update(_clip_nodes(6, 7, clip_node, style, neg))
    g.update({str(8): {"class_type": "ImageScale",
                       "inputs": {"image": [str(4), 0],
                                  "upscale_method": "lanczos",
                                  "width": width, "height": height,
                                  "crop": "disabled"}}})
    g.update({str(9): {"class_type": "VAEEncode",
                       "inputs": {"pixels": [str(8), 0], "vae": [str(1), 2]}}})
    g.update(_sampler_node(10, 5, 6, 7, 9, {**params, "denoise": denoise}))
    g.update(_latent_upscale_by(11, 10, scale_by=params.get("hires_scale", 1.5)))
    g.update(_sampler_node(12, 5, 6, 7, 11,
                           {**params, "denoise": params.get("hires_denoise", 0.40),
                            "steps": params.get("hires_steps", 25)}))
    g.update(_vae_decode(13, 12, model_node))
    g.update(_save_node(14, 13, f"{job_id}/mode2"))
    return g


def build_bgswap(original_filename: str, params: dict, job_id: str) -> dict:
    """背景替换（显式主体锁定）：BiRefNet 抠出主体蒙版 -> 反选为背景蒙版
    -> SetLatentNoiseMask 只重绘背景区 -> KSampler 换新场景，主体像素级保留。
    链路：LoadImage -> ImageScale -> BiRefNetRMBG -> InvertMask -> VAEEncode
          -> SetLatentNoiseMask -> KSampler(denoise=1.0) -> VAEDecode -> Save。
    """
    scene = params.get("style_prompt", "a clean white studio background, soft light")
    neg = params.get("negative_prompt", "low quality, blurry, deformed, watermark, text")
    width = params.get("width", DEFAULTS["width"])
    height = params.get("height", DEFAULTS["height"])
    model = params.get("matting_model", "BiRefNet-matting")
    g = {}
    g.update(_checkpoint_node(1))                            # (1,0)MODEL (1,1)CLIP (1,2)VAE
    g.update(_load_image_node(2, original_filename))
    g.update({str(3): {"class_type": "ImageScale",
                       "inputs": {"image": [str(2), 0],
                                  "upscale_method": "lanczos",
                                  "width": width, "height": height,
                                  "crop": "disabled"}}})
    # BiRefNet 主体分割：输出 (4,0)IMAGE 抠图 (4,1)MASK 主体 (4,2)MASK_IMAGE
    # 注意：该节点内部直接访问 params["background"]/["mask_blur"]/["mask_offset"]
    #       /["invert_output"]，API 格式省略可选参数不会自动补默认值，必须全传。
    g.update({str(4): {"class_type": "BiRefNetRMBG",
                       "inputs": {"image": [str(3), 0],
                                  "model": model,
                                  "sensitivity": 1.0,
                                  "mask_blur": 0,
                                  "mask_offset": 0,
                                  "invert_output": False,
                                  "refine_foreground": False,
                                  "background": "Alpha",
                                  "background_color": "#222222"}}})
    g.update({str(5): {"class_type": "InvertMask",
                       "inputs": {"mask": [str(4), 1]}}})    # 背景蒙版
    g.update(_clip_nodes(6, 7, 1, scene, neg))
    g.update({str(8): {"class_type": "VAEEncode",
                       "inputs": {"pixels": [str(3), 0], "vae": [str(1), 2]}}})
    g.update({str(9): {"class_type": "SetLatentNoiseMask",
                       "inputs": {"samples": [str(8), 0],
                                  "mask": [str(5), 0]}}})
    g.update(_sampler_node(10, 1, 6, 7, 9, {**params, "denoise": 1.0}))
    g.update(_vae_decode(11, 10, 1))
    g.update(_save_node(12, 11, f"{job_id}/bgswap"))
    return g


def build(mode: str, original_filename: str, params: dict, job_id: str) -> dict:
    if mode == "mode1":
        return build_mode1(original_filename, params, job_id)
    elif mode == "mode2":
        return build_mode2(original_filename, params, job_id)
    elif mode == "bgswap":
        return build_bgswap(original_filename, params, job_id)
    raise ValueError(f"未知模式: {mode}")
