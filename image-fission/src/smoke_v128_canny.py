"""v128: 加 ControlNet Canny 锁主结构 + 降 denoise 0.55 + 保留 IPAdapter style.

按用户最新目标:
- 同源 (颜色/风格/大主题) - IPAdapter style 锁
- 不同 (内部细节重画) - Canny 锁骨架 + denoise 0.55 留 45% 自由度
- 不侵权 - 不在图内造字 (PIL 后期烧)
"""
import json, time, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REFS = [
    {"id": "illust_1", "filename": "pinterest_illust_1.jpg", "weight": 0.55, "cn_weight": 0.55,
     "subject": "ornate symmetrical black scrollwork floral filigree frame on dark canvas",
     "palette": "monochrome ink illustration, pure black background, white filigree, no color, no text"},
    {"id": "eagle_2",  "filename": "pinterest_eagle_2.jpg",  "weight": 0.55, "cn_weight": 0.55,
     "subject": "double-headed eagle crest with flaming aura and iron chains, skull throne at base, heraldic shield with banner ribbon, symmetrical",
     "palette": "gothic tattoo illustration, dark black background, vivid orange flames, white and red linework, intricate feathers"},
    {"id": "denim_3",  "filename": "pinterest_denim_3.jpg",  "weight": 0.55, "cn_weight": 0.50,
     "subject": "delicate moth flying over a distressed X-shaped denim patch with fabric wear and stitch holes",
     "palette": "vintage faded denim texture, indigo blue and white only, rough brushstroke X mark"},
    {"id": "camo_4",   "filename": "pinterest_camo_4.jpg",   "weight": 0.55, "cn_weight": 0.55,
     "subject": "palm leaves and tropical fronds in military camouflage pattern, dense dark foliage",
     "palette": "dark green brown black camouflage, matte, shadowy, no text"},
    {"id": "skull_5",  "filename": "pinterest_skull_5.jpg",  "weight": 0.55, "cn_weight": 0.55,
     "subject": "grim human skull with black eye patch, snake coiled around it, wings, rose",
     "palette": "gothic tattoo illustration, pure dark black background, red and white accents"},
    {"id": "metal_6",  "filename": "pinterest_metal_6.jpg",  "weight": 0.55, "cn_weight": 0.55,
     "subject": "bald eagle perched on cracked human skull with massive horn-like spikes and radiating lightning bolts",
     "palette": "death metal thrash metal illustration, pure black background, white and bronze gold yellow palette"},
]

CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"
SEED = 700301

POS_TEMPLATE = "masterpiece, best quality, ultra detailed, clean illustration, sharp clean edges, {subject}, intricate linework, redesigned internal details, framed composition, {palette}"
NEG = "text, words, letters, typography, font, alphabet, watermark, signature, logo, badge, frame, border, blurry, smudged, deformed, disfigured, low quality, bad anatomy, mutation, cropped, jpeg artifacts, noise"


def build_graph(ref):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["filename"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["4"] = {"class_type": "Canny", "inputs": {"image": ["3", 0], "low_threshold": 0.08, "high_threshold": 0.18}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = POS_TEMPLATE.format(subject=ref["subject"], palette=ref["palette"])
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": ref["cn_weight"], "start_percent": 0.0, "end_percent": 0.85,
        "control_net": ["5", 0]}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1],
        "image": ["2", 0],
        "weight": ref["weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "embeds_scaling": "V only",
        "start_at": 0.0, "end_at": 0.85, "noise": 0.08}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["7", 0],
        "seed": SEED, "steps": 40, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.55, "noise_mask": None}}
    g["13"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0],
        "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["12", 0],
        "seed": SEED + 1, "steps": 28, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.18, "noise_mask": None}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["16"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["15", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": "v128_clean"}}
    return g


def run_one(ref, out_dir):
    import requests
    tag = ref["id"]
    out_path = out_dir / f"v128_clean_{tag}.png"
    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[skip] {tag}: 已存在 {out_path.stat().st_size//1024}KB", flush=True); return True
    print(f"[run] {tag}  cn={ref['cn_weight']}  ipa={ref['weight']}  denoise=0.55", flush=True)
    g = build_graph(ref)
    try:
        r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v128_{int(time.time())}"}, timeout=15)
    except Exception as e:
        print(f"  ERR submit conn: {e}", flush=True); return False
    j = r.json() if r.status_code == 200 else {}
    if r.status_code != 200 or "error" in j:
        print(f"  ERR submit: {r.status_code} {str(j)[:300]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"  ERR no prompt_id: {str(j)[:300]}", flush=True); return False
    t0 = time.time()
    print(f"  pid={pid}", flush=True)
    for i in range(40):  # 200s 硬超时
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                rec = h[pid]
                status = rec.get("status", {})
                if status.get("completed"):
                    imgs = rec.get("outputs", {}).get("17", {}).get("images", [])
                    if not imgs:
                        print(f"  ERR no images in output", flush=True); return False
                    url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                    data = requests.get(url, timeout=120).content
                    out_path.write_bytes(data)
                    print(f"  OK {out_path.stat().st_size//1024//1024}MB  (waited {int(time.time()-t0)}s)", flush=True); return True
                if status.get("error"):
                    print(f"  ERR exec: {status.get('error')}", flush=True); return False
        except Exception as e:
            print(f"  poll err: {e}", flush=True)
    print(f"  TIMEOUT after {int(time.time()-t0)}s", flush=True); return False


def main():
    out = PROJECT_ROOT / "jobs" / f"smoke_v128_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== v128 (Canny cn=0.55 + IPA style=0.55 + denoise=0.55) -> {out} ===", flush=True)
    ok = 0
    for ref in REFS:
        if run_one(ref, out):
            ok += 1
    print(f"\n=== done v128: {ok}/{len(REFS)} ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
