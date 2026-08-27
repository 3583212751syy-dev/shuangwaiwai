r"""v10.5 验证图：IPAdapter style 锁原图画风+配色 + 换内容 + 文字可控 LoRA。

对齐用户 2026-08-25 17:39 需求：
- 风格颜色多参考原图（IPAdapter style transfer 锁画风+配色，喂高清化参考图）
- 内容裂变有区别有意义（prompt 写新主体 phoenix，不写孔雀）
- 文字清晰同类型（Harrlogos XL v2 真实拼 MAJESTY，匹配威严鸟风格）
- 清晰有质量（4x_NMKD-Siax 超分）

管线：
  Checkpoint(Proteus) -> LoraLoader(VECTOR 0.55) -> LoraLoader(Harrlogos 0.65)
  -> IPAdapterUnifiedLoader(PLUS) -> LoadImage(HD参考) -> IPAdapterAdvanced(style, w=0.55)
  -> CLIP encode -> EmptyLatent(1024) -> KSampler1(45/7.5/1.0)
  -> KSampler2(hires 0.20/40) -> VAEDecode -> 4x_NMKD-Siax -> Save
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import demo_batch_6x4 as db
from engine.comfy_client import ComfyClient
from pipelines.build import (
    IPADAPTER_PRESET,
)


def _checkpoint_node(n, ckpt):
    return {str(n): {"class_type": "CheckpointLoaderSimple",
                     "inputs": {"ckpt_name": ckpt}}}


def _lora_loader(n, model_from, clip_from, lora_name, sm=0.55, sc=0.55):
    return {str(n): {"class_type": "LoraLoader",
                     "inputs": {"model": [str(model_from), 0],
                                "clip": [str(clip_from), 1],
                                "lora_name": lora_name,
                                "strength_model": sm,
                                "strength_clip": sc}}}


def _ipadapter_loader(n, model_from):
    return {str(n): {"class_type": "IPAdapterUnifiedLoader",
                     "inputs": {"model": [str(model_from), 0],
                                "preset": IPADAPTER_PRESET}}}


def _load_image(n, filename):
    return {str(n): {"class_type": "LoadImage",
                     "inputs": {"image": filename, "upload": "image"}}}


def _ipadapter_apply(n, model_node, ipadapter_node, image_node, weight,
                     weight_type="style transfer", noise=0.1,
                     start_at=0.0, end_at=0.9):
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
                         "noise": noise,
                         "embeds_scaling": "V only",
                     }}}


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


def build_v105(params):
    g = {}
    # 1: Checkpoint (Proteus)
    g.update(_checkpoint_node(1, params["ckpt"]))
    # 2: VECTOR LoRA 0.55
    g.update(_lora_loader(2, 1, 1, "DD-vector-v2.safetensors", sm=0.55, sc=0.55))
    # 3: Harrlogos XL v2 文字 LoRA 0.65
    g.update(_lora_loader(3, 2, 2, "Harrlogos_XL_v2.safetensors", sm=0.65, sc=0.65))
    # 4: IPAdapterUnifiedLoader
    g.update(_ipadapter_loader(4, 3))
    # 5: LoadImage (高清化参考图)
    g.update(_load_image(5, params["ref_image"]))
    # 6: IPAdapterAdvanced (style transfer 锁画风+配色, 不锁内容)
    g.update(_ipadapter_apply(6, 3, 4, 5, params["ipa_weight"],
                              weight_type="style transfer",
                              noise=params.get("ipa_noise", 0.1),
                              start_at=0.0, end_at=0.9))
    # 7/8: CLIP encode (clip 来自 Harrlogos 节点 3 输出 1)
    g.update(_clip_encode(7, 3, params["positive"]))
    g.update(_clip_encode(8, 3, params["negative"]))
    # 9: EmptyLatent 1024
    g.update(_empty_latent(9, 1024, 1024))
    # 10: KSampler1
    g.update(_ksampler(10, 6, 7, 8, 9, params["seed"],
                       steps=45, cfg=7.5, denoise=1.0))
    # 11: KSampler2 hires
    g.update(_ksampler(11, 6, 7, 8, 10, params["seed"] + 1,
                       steps=40, cfg=7.5, denoise=0.20))
    # 12: VAEDecode (vae from ckpt 1)
    g.update(_vae_decode(12, 11, 1))
    # 13/14: 4x NMKD-Siax
    g.update(_upscale_loader(13, "4x_NMKD-Siax_200k.pth"))
    g.update(_image_upscale(14, 13, 12))
    # 15: Save
    g.update(_save_image(15, 14, params["filename_prefix"]))
    return g


POSITIVE = (
    "YOURTEXT text logo MAJESTY, "
    "vector illustration t-shirt graphic, phoenix, "
    "majestic long-tailed bird with flowing detailed feathers, "
    "tropical plumage, sharp beak, fierce eyes, "
    "surrounded by 5-petal flowers, lily blossoms, "
    "decorative ornamental swirls, vine scrolls, filigree accents, "
    "ornamental baroque pattern, intricate detail, "
    "sharp clean edges, refined linework, "
    "high contrast, pure black background, white linework, "
    "edge-to-edge, full bleed, fills entire frame, no margins, "
    "centered subject, t-shirt apparel print design, "
    "professional vector graphic, 4k detail"
)

NEGATIVE = (
    "colorful gradient shading halftone painterly texture photographic 3d render "
    "realistic texture fabric folds wrinkles shadows, "
    "white border frame edge margin padding empty corners letterbox, "
    "blurry noisy grain film grain sensor noise compression artifacts "
    "speckles dust spots harsh jagged edges fuzzy halftone "
    "rough sketch pencil smudge dirty paper texture uneven ink "
    "broken outlines scratchy strokes, "
    "deformed low quality watermark copyright logo, "
    "garbled text gibberish text pseudo-script fake characters "
    "runic nonsense occult sigils runes talisman symbols "
    "malformed letters repeating letters double letters "
    "illegible text scribbles resembling text signature, "
    "background scenery environment landscape sky ground"
)


def run_one(out_dir, seed):
    params = {
        "ckpt": "ProteusV0.4.safetensors",
        "ref_image": "pinterest_illust_1_hd.png",
        "ipa_weight": 0.55,
        "ipa_noise": 0.1,
        "seed": seed,
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "filename_prefix": "v105_illust_phoenix",
    }
    g = build_v105(params)
    print(f"\n[v10.5] illust_1 + phoenix  seed={seed}")
    print(f"  ckpt=Proteus  VECTOR=0.55  Harrlogos=0.65")
    print(f"  IPAdapter=style transfer w=0.55 noise=0.1  ref=pinterest_illust_1_hd.png")
    print(f"  text=MAJESTY (YOURTEXT text logo)")
    client = ComfyClient()
    t0 = time.time()
    try:
        res = client.run(g, timeout=600)
        data = next(iter(res.values()))[0]
        out_path = os.path.join(out_dir, f"illust_phoenix_{seed}.jpg")
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
    out_dir = os.path.join(db.JOBS_BASE, f"smoke_v105_illust_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    p = run_one(out_dir, 100101)
    if p:
        print(f"\n[v10.5 done] {p}")


if __name__ == "__main__":
    main()
