"""v11.0 治本裂变：AI 提取原图内容（主体+风格+配色+装饰）→ IPAdapter style 锁画风/配色
+ 主体词保留 + 后期 PIL 烧字（匹配画面风格）。不侵权，零糊字。

用法：python src/smoke_v110_illust.py
"""
import json, os, sys, time, requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

# 6 张参考图（已读出内容，按"主体+风格+配色"提炼）
REFS = [
    {
        "id": "illust_1",
        "ref_img": "pinterest_illust_1_hd.png",
        "subject_word": "elegant peacock silhouette with elaborate flowing tail feathers",
        "style_words": "baroque vector illustration, pure black background, white linework only, ornamental 5-petal flowers and decorative swirling flourishes, t-shirt graphic, clean sharp edges, fills entire frame edge-to-edge, no margins, full bleed, no halftone, no noise, no grain, no jagged edges",
        "negative": "no gray, no midtones, pure black and white only, high contrast, no soft shading, no gradient, no shading, no color, no halftone, 3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, margin, border, frame, padding, empty corners, letterbox, white border, smudge",
    },
    {
        "id": "eagle_2",
        "ref_img": "pinterest_eagle_2_hd.png",
        "subject_word": "majestic spread-winged eagle gripping a skull in its talons, gothic crest emblem with chains and flames",
        "style_words": "gothic tattoo illustration, dark black background, white and red linework, intricate feathers, flaming aura, heraldic crest shield, iron chains, blood splatter, t-shirt graphic, sharp clean edges, fills entire frame edge-to-edge, no margins, no halftone, no noise",
        "negative": "halftone, photographic, 3d, painterly, pastel, blur, noise, grain, pixelated, jagged edges, margin, border, white border, smudge",
    },
    {
        "id": "denim_3",
        "ref_img": "pinterest_denim_3_flat_hd.png",
        "subject_word": "single delicate moth flying over a distressed X-shaped denim patch",
        "style_words": "vintage faded denim texture, indigo blue and white only, rough brushstroke X mark, fabric wear and tear, dotted flight trail, t-shirt graphic, sharp clean edges, fills entire frame edge-to-edge, no margins, no halftone, no noise",
        "negative": "photographic, 3d, painterly, halftone, color noise, grain, pixelated, jagged edges, margin, border, white border",
    },
    {
        "id": "camo_4",
        "ref_img": "pinterest_camo_4_hd.png",
        "subject_word": "tropical palm tree silhouette with classic military camouflage pattern background",
        "style_words": "tropical jungle camouflage, brown green tan khaki black palette, black palm tree silhouettes, organic camo blob shapes, repeating pattern, t-shirt graphic, sharp clean edges, fills entire frame edge-to-edge, no margins, no halftone, no noise",
        "negative": "photographic, 3d, painterly, pastel, halftone, blur, noise, grain, pixelated, jagged edges, margin, border, white border",
    },
    {
        "id": "skull_5",
        "ref_img": "pinterest_skull_5_hd.png",
        "subject_word": "grim human skull with black eye patch, red snake coiled around it, red angel wings, dripping blood, red rose",
        "style_words": "gothic tattoo illustration, pure black background, red and white palette, blood drips, cracked bone, serpent scales, feathered wings, t-shirt graphic, sharp clean edges, fills entire frame edge-to-edge, no margins, no halftone, no noise",
        "negative": "halftone, photographic, 3d, painterly, pastel, blur, noise, grain, pixelated, jagged edges, margin, border, white border",
    },
    {
        "id": "metal_6",
        "ref_img": "pinterest_metal_6_hd.png",
        "subject_word": "bald eagle perched on a cracked human skull with massive horn-like spikes and radiating lightning bolts",
        "style_words": "death metal thrash metal illustration, pure black background, white and bronze gold yellow palette, sharp aggressive lines, radial lightning burst, heavy ink, t-shirt graphic, sharp clean edges, fills entire frame edge-to-edge, no margins, no halftone, no noise",
        "negative": "halftone, photographic, 3d, painterly, pastel, blur, noise, grain, pixelated, jagged edges, margin, border, white border",
    },
]

def build_workflow(ref, seed):
    """v11.0 正确接法：
    1: CheckpointLoader (Proteus)
    2: CLIPTextEncode (positive)
    3: CLIPTextEncode (negative)
    4: IPAdapterUnifiedLoader (model=1.0, preset=PLUS)
    5: LoadImage (HD ref)
    6: IPAdapterAdvanced (model=1.0, ipadapter=4.1, image=5.0, weight=0.55, weight_type=style transfer, noise=0.05)
    7: EmptyLatentImage 1024²
    8: KSampler 1 (model=6.0, denoise=1.0, steps=45, cfg=7.5)
    9: KSampler 2 (model=6.0, denoise=0.20, steps=40, hires 细化)
    10: VAEDecode (samples=9.0, vae=1.2)
    11: UpscaleModelLoader (4x_NMKD-Siax_200k.pth)
    12: ImageUpscaleWithModel (upscale=11.0, image=10.0)
    13: SaveImage (images=12.0, prefix=v110_<id>)
    """
    g = {}

    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}

    pos = f"t-shirt graphic design, {ref['subject_word']}, {ref['style_words']}, masterpiece, best quality, ultra detailed, clean illustration"
    g["2"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": pos}}
    g["3"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": ref["negative"]}}

    g["4"] = {"class_type": "IPAdapterUnifiedLoader",
              "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}

    g["5"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}

    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0],
        "ipadapter": ["4", 1],
        "image": ["5", 0],
        "weight": 0.7,
        "weight_type": "style transfer",
        "combine_embeds": "average",
        "start_at": 0.0,
        "end_at": 0.85,
        "noise": 0.05,
        "embeds_scaling": "V only",
    }}

    # Canny ControlNet：锁原图构图/边缘分布（作用在 conditioning 上，与 IPAdapter 的 model 维度叠加不冲突）
    # 权重 0.6 控在 0.5-0.7 区间，防丢主体（之前 0.85 锁内容丢过孔雀）
    g["14"] = {"class_type": "ControlNetLoader",
               "inputs": {"control_net_name": "controlnet-canny-sdxl-1.0.fp16.safetensors"}}
    g["15"] = {"class_type": "CannyEdgePreprocessor", "inputs": {"image": ["5", 0]}}
    g["16"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["2", 0],
        "control_net": ["14", 0],
        "image": ["15", 0],
        "strength": 0.6,
    }}

    g["7"] = {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}

    g["8"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0],
        "positive": ["16", 0],
        "negative": ["3", 0],
        "latent_image": ["7", 0],
        "seed": seed,
        "steps": 45,
        "cfg": 7.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 1.0,
    }}

    g["9"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0],
        "positive": ["16", 0],
        "negative": ["3", 0],
        "latent_image": ["8", 0],
        "seed": seed + 1,
        "steps": 40,
        "cfg": 7.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 0.20,
    }}

    g["10"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["1", 2]}}

    g["11"] = {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}

    g["12"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["11", 0], "image": ["10", 0]}}

    g["13"] = {"class_type": "SaveImage",
               "inputs": {"images": ["12", 0], "filename_prefix": f"v112_{ref['id']}"}}

    return g


def main():
    out_base = PROJECT_ROOT / "jobs" / f"smoke_v112_{int(time.time())}"
    out_base.mkdir(parents=True, exist_ok=True)

    ref = REFS[0]
    seed = 100301
    print(f"=== v11.0 验证：{ref['id']} ===")
    print(f"  ref_img={ref['ref_img']}")
    print(f"  subject={ref['subject_word'][:60]}...")

    workflow = build_workflow(ref, seed)
    payload = {"prompt": workflow, "client_id": f"v110_{int(time.time())}"}

    r = requests.post(f"{COMFYUI}/prompt", json=payload, timeout=10)
    if r.status_code != 200:
        print(f"[ERR] submit failed: {r.status_code} {r.text[:300]}")
        return 1
    pid = r.json().get("prompt_id")
    print(f"  prompt_id={pid}")

    print("  waiting for completion...")
    for i in range(120):
        time.sleep(5)
        try:
            q = requests.get(f"{COMFYUI}/history/{pid}", timeout=5).json()
            if pid in q:
                rec = q[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("13", {}).get("images", [])
                    if imgs:
                        fname = imgs[0]["filename"]
                        subfolder = imgs[0].get("subfolder", "")
                        url = f"{COMFYUI}/view?filename={fname}&type=output&subfolder={subfolder}"
                        data = requests.get(url, timeout=30).content
                        out_path = out_base / f"{ref['id']}_{seed}.jpg"
                        out_path.write_bytes(data)
                        size_mb = out_path.stat().st_size / 1024 / 1024
                        print(f"  [OK] saved {out_path}  ({size_mb:.1f}MB)")
                        return 0
                    else:
                        print("  [ERR] completed but no images")
                        return 2
                elif rec.get("status", {}).get("error"):
                    print(f"  [ERR] error: {rec['status']}")
                    return 3
        except Exception:
            pass
        if i % 4 == 0:
            print(f"    {i*5}s waiting...")

    print("  [TIMEOUT] 600s exceeded")
    return 4


if __name__ == "__main__":
    sys.exit(main())
