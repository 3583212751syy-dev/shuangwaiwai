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


def _usdu_only(n_start, decoded_image_node, upscale_model_name, save_n, save_prefix):
    """真实放大链（精简版）：ImageUpscaleWithModel(2x) -> SaveImage。
    替代 LatentUpscaleBy 的"潜空间插值放大"，治"不增加新细节"的根本问题。
    不加细化 KSampler（真实超分本身已加细节，加了会拖慢 2-3 倍且容易改变原图）。
    """
    g = {}
    n = n_start
    g.update({str(n): {"class_type": "UpscaleModelLoader",
                       "inputs": {"model_name": upscale_model_name}}})
    n_up = n; n += 1
    g.update({str(n): {"class_type": "ImageUpscaleWithModel",
                       "inputs": {"upscale_model": [str(n_up), 0],
                                  "image": [str(decoded_image_node), 0]}}})
    n_img = n; n += 1
    g.update(_save_node(save_n, n_img, save_prefix))
    return g, n


def _ipadapter_apply(n, model_node, ipadapter_node, image_node, weight,
                     start_at=0.0, end_at=1.0, weight_type="linear",
                     noise=0.0, combine_embeds="average"):
    """IPAdapterAdvanced：输入 unified loader 的 MODEL(输出0) 与 IPADAPTER(输出1)，
    输出带参考图条件的 MODEL(0)。
    weight_type (SDXL 关键)：
      - linear: 通用混合
      - style transfer (SDXL): 专锁风格（颜色/材质/笔触）— 用户要的颜色参考
      - composition (SDXL): 专锁构图（布局/位置）— 用户要的构图参考
    noise: 0.05-0.15 加噪防止完全照搬（"换内容"留更多空间）"""
    return {str(n): {"class_type": "IPAdapterAdvanced",
                     "inputs": {
                         "model": [str(model_node), 0],
                         "ipadapter": [str(ipadapter_node), 1],
                         "image": [str(image_node), 0],
                         "weight": weight,
                         "weight_type": weight_type,
                         "combine_embeds": combine_embeds,
                         "start_at": start_at,
                         "end_at": end_at,
                         "noise": noise,
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


def _controlnet_loader_node(n, control_name):
    return {str(n): {"class_type": "ControlNetLoader",
                     "inputs": {"control_net_name": control_name}}}


def _canny_node(n, image_from, low=0.4, high=0.8):
    """原生 Canny 边缘提取：从原图得到线稿，作为 ControlNet 构图控制图。"""
    return {str(n): {"class_type": "Canny",
                     "inputs": {"image": [str(image_from), 0],
                                "low_threshold": low,
                                "high_threshold": high}}}


def _controlnet_apply(n, pos_from, neg_from, controlnet_from, image_from, strength,
                      start_percent=0.0, end_percent=0.9):
    """ControlNetApplyAdvanced：本机 ComfyUI 版本作用在 CONDITIONING 上（输入/输出都是
    positive/negative CONDITIONING，不是 MODEL），因此插在 CLIP 与 KSampler 之间：
      CLIP(6/7) → ControlNetApplyAdvanced(52) → KSampler positive/negative。
    start/end_percent 控制影响区间；strength 即构图参考强度滑杆。输出 [n,0]=positive', [n,1]=negative'。"""
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


def build_mode1(original_filename: str, params: dict, job_id: str) -> dict:
    """换景换风格：保持与原图相似度(weight)，用 prompt 换场景/风格。
    可选 LoRA（链式加载，最多 3 个）：
      - params["lora_name"]   + params["lora_strength"]   第一个 LoRA
      - params["lora_name_2"] + params["lora_strength_2"] 第二个 LoRA（链在第一个之后）
      - params["lora_name_3"] + params["lora_strength_3"] 第三个 LoRA（链在第二个之后）
        用于文字可控生成（如 Harrlogos XL）：让徽章/横幅区出现的文字是真实可读英文，
        而非 SDXL 默认生成的伪字母乱码。
    """
    w = params.get("similarity", DEFAULTS["similarity"])
    style = params.get("style_prompt", "")
    neg = params.get("negative_prompt", "low quality, blurry, deformed, watermark, text")

    # ---- 三 LoRA 链式加载 ----
    lora1_name = params.get("lora_name", "")
    lora1_strength = float(params["lora_strength"]) if "lora_strength" in params and params["lora_strength"] is not None else 0.0
    lora2_name = params.get("lora_name_2", "")
    lora2_strength = float(params["lora_strength_2"]) if "lora_strength_2" in params and params["lora_strength_2"] is not None else 0.0
    lora3_name = params.get("lora_name_3", "")
    lora3_strength = float(params["lora_strength_3"]) if "lora_strength_3" in params and params["lora_strength_3"] is not None else 0.0

    g = {}
    g.update(_checkpoint_node(1))
    last_model, last_clip = 1, 1
    if lora1_name and lora1_strength > 0:
        g.update(_lora_loader_node(2, last_model, last_clip, lora1_name, lora1_strength, lora1_strength))
        last_model, last_clip = 2, 2
    if lora2_name and lora2_strength > 0:
        g.update(_lora_loader_node(3, last_model, last_clip, lora2_name, lora2_strength, lora2_strength))
        last_model, last_clip = 3, 3
    # 第三个 LoRA 用节点 17（与 IPAdapter 4/5/6、CLIP 7/8、ControlNet 70/71/72、
    # USDU 的 UpscaleModelLoader(13)/ImageUpscaleWithModel(14)/SaveImage(15)、
    # VAEDecode(12)/KSampler(10/11) 均不冲突；17 仅作 LoRA 链式上游输入）。
    if lora3_name and lora3_strength > 0:
        g.update(_lora_loader_node(17, last_model, last_clip, lora3_name, lora3_strength, lora3_strength))
        last_model, last_clip = 17, 17

    # ---- IPAdapter 加载（必须在 LoRA 之后）----
    g.update(_ipadapter_loader_node(4, last_model))
    g.update(_load_image_node(5, original_filename))

    # ---- 三通道参考控制：颜色 + 构图 + 内容 ----
    # 用户需求：裂变图「有区别」（内容由 prompt/seed 变化），但「保留原图颜色与构图参考」。
    # 做法：同一张原图挂两个 IPAdapter——
    #   · 颜色锁 node6  weight_type="style transfer" → 锁住配色/材质/笔触
    #   · 构图锁 node60 weight_type="composition"    → 锁住布局/结构/位置
    # 两者叠加后，内容(prompt)自由变化 → 得到「同色同构、异内容」的裂变图。
    # color_strength / composition_strength 都为 0 时，回退到单 IPAdapter linear
    # （兼容只传 similarity 的旧 demo：demo_pattern_fission / demo_fission）。
    color_w = float(params.get("color_strength", 0.0) or 0.0)
    comp_w = float(params.get("composition_strength", 0.0) or 0.0)
    ipa_noise = params.get("ipadapter_noise", 0.05)
    ipa_end = params.get("ipadapter_end", 0.85)

    model_after_ip = last_model
    if color_w > 0:
        g.update(_ipadapter_apply(6, last_model, 4, 5, color_w,
                                 weight_type="style transfer",
                                 start_at=params.get("ipadapter_start", 0.0),
                                 end_at=ipa_end,
                                 noise=ipa_noise,
                                 combine_embeds="average"))
        model_after_ip = 6
    if comp_w > 0:
        # 构图锁叠加在「颜色锁之后的模型」上，end 取 max(颜色锁, 0.9) 保证布局收尾稳定
        g.update(_ipadapter_apply(60, model_after_ip, 4, 5, comp_w,
                                 weight_type="composition",
                                 start_at=0.0, end_at=max(ipa_end, 0.9),
                                 noise=0.0, combine_embeds="average"))
        model_after_ip = 60
    if color_w <= 0 and comp_w <= 0:
        # 旧路径：单 IPAdapter linear（仅传 similarity 的 demo 用）
        g.update(_ipadapter_apply(6, last_model, 4, 5, w,
                                 weight_type="linear",
                                 start_at=0.0, end_at=1.0,
                                 noise=params.get("ipadapter_noise", 0.0),
                                 combine_embeds="average"))
        model_after_ip = 6

    model_pre_cn = model_after_ip

    # ---- CLIP 编码（供 prompt 使用）----
    g.update(_clip_nodes(7, 8, last_clip, style, neg))
    samp_pos, samp_neg = 7, 8   # KSampler 默认直接取 CLIP 编码

    # ---- ControlNet Canny（更强构图锁）—— 可选 ----
    # 本机 ComfyUI 的 ControlNetApplyAdvanced 作用在 CONDITIONING 上（输出 CONDITIONING
    # 而非 MODEL），因此插在 CLIP(7/8) 与 KSampler(10) 之间，model 仍走 IPAdapter 主线。
    model_for_sampler = model_pre_cn
    cn_name = params.get("controlnet_name", "")
    cn_strength = float(params.get("controlnet_strength", 0.0) or 0.0)
    if cn_name and cn_strength > 0:
        g.update(_controlnet_loader_node(70, cn_name))
        g.update(_canny_node(71, 5,
                             low=params.get("controlnet_low_threshold", 100),
                             high=params.get("controlnet_high_threshold", 200)))
        g.update(_controlnet_apply(72, 7, 8, 70, 71, cn_strength,
                                   start_percent=params.get("controlnet_start", 0.0),
                                   end_percent=params.get("controlnet_end", 0.9)))
        samp_pos, samp_neg = 72, 72   # 改用 ControlNet 输出的 positive/negative

    g.update({str(9): {"class_type": "EmptyLatentImage",
                       "inputs": {"width": params.get("width", DEFAULTS["width"]),
                                  "height": params.get("height", DEFAULTS["height"]),
                                  "batch_size": params.get("batch_per_run", DEFAULTS["batch_per_run"])}}})

    # ---- KSampler + 放大 ----
    usdu_model = params.get("usdu_model", "4x_NMKD-Siax_200k.pth")
    use_usdu = bool(usdu_model)
    if use_usdu:
        # USDU 路径（治「中央噪点」关键修正）：
        #   KSampler 1（满采去噪, steps） → 二阶细化 KSampler（latent 二次清理,
        #   denoise=hires_denoise, steps=hires_steps） → VAE Decode → 真实 4x 超分 → Save。
        # 二阶细化在 base latent(1024) 上以低 denoise 重采，清掉潜空间残留噪点，
        # 使 4x 真实超分不会把噪点放大成可见碎裂感；同时保留 4096px 高清晰度。
        g.update(_sampler_node(10, model_for_sampler, samp_pos, samp_neg, 9,
                               {**params, "denoise": 1.0,
                                "steps": params.get("steps",
                                                    params.get("steps_base", DEFAULTS["steps"]))}))
        g.update(_sampler_node(11, model_for_sampler, samp_pos, samp_neg, 10,
                               {**params,
                                "denoise": params.get("hires_denoise", 0.28),
                                "steps": params.get("hires_steps", 30)}))
        g.update(_vae_decode(12, 11, last_model))
        usdu_nodes, _ = _usdu_only(13, 12, usdu_model, save_n=15,
                                   save_prefix=f"{job_id}/mode1")
        g.update(usdu_nodes)
    else:
        # 旧路径：KSampler 1 → LatentUpscale 1.5x → KSampler 2 → Save
        g.update(_sampler_node(10, model_for_sampler, samp_pos, samp_neg, 9, {**params, "denoise": 1.0,
                                               "steps": params.get("steps",
                                                                   params.get("steps_base", DEFAULTS["steps"]))}))
        g.update(_latent_upscale_by(11, 10, scale_by=params.get("hires_scale", 1.5)))
        g.update(_sampler_node(12, model_for_sampler, samp_pos, samp_neg, 11,
                               {**params, "denoise": params.get("hires_denoise", 0.35),
                                "steps": params.get("hires_steps", 20)}))
        g.update(_vae_decode(13, 12, last_model))
        g.update(_save_node(14, 13, f"{job_id}/mode1"))
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
    lora_strength = float(params["lora_strength"]) if "lora_strength" in params and params["lora_strength"] is not None else 0.0
    if lora_name and lora_strength > 0:
        model_node, clip_node = 2, 2
    else:
        model_node, clip_node = 1, 1
    g = {}
    g.update(_checkpoint_node(1))
    if lora_name and lora_strength > 0:
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
