"""v128 仅重生 camo_4 和 skull_5 (受源图映射反掉影响的两张)."""
import json, time, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

# 用现有 v128 目录
OUT_DIR = Path(r"E:\Desktop\双接口\image-fission\jobs\smoke_v128_1787814169")

TARGETS = [
    {"id": "camo_4", "filename": "pinterest_camo_4.jpg", "weight": 0.55, "cn_weight": 0.55,
     "subject": "dense palm tree fronds in military camouflage pattern, dense dark foliage",
     "palette": "dark green brown black camouflage, matte, shadowy, no text"},
    {"id": "skull_5", "filename": "pinterest_skull_5.jpg", "weight": 0.55, "cn_weight": 0.55,
     "subject": "grim human skull with black eye patch, snake coiled around it, dark wings, dark rose, gothic composition",
     "palette": "dark gothic illustration, pure dark black background, dark red and white accents"},
]

CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"
SEED = 700301

POS_TEMPLATE = "masterpiece, best quality, ultra detailed, clean illustration, sharp clean edges, {subject}, intricate linework, redesigned internal details, {palette}"
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


def run_one(ref):
    import requests
    tag = ref["id"]
    out_path = OUT_DIR / f"v128_clean_{tag}.png"
    print(f"[run] {tag}  cn={ref['cn_weight']}  ipa={ref['weight']}  denoise=0.55  src={ref['filename']}", flush=True)
    g = build_graph(ref)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v128r_{int(time.time())}"}, timeout=15)
    j = r.json() if r.status_code == 200 else {}
    if r.status_code != 200 or "error" in j:
        print(f"  ERR: {r.status_code} {str(j)[:200]}", flush=True); return False
    pid = j.get("prompt_id")
    print(f"  pid={pid}", flush=True)
    for i in range(60):  # 5 分钟超时 (含模型预热)
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                rec = h[pid]
                status = rec.get("status", {})
                if status.get("completed"):
                    imgs = rec.get("outputs", {}).get("17", {}).get("images", [])
                    if not imgs: return False
                    url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                    data = requests.get(url, timeout=120).content
                    out_path.write_bytes(data)
                    print(f"  OK {out_path.stat().st_size//1024//1024}MB", flush=True); return True
                if status.get("error"):
                    print(f"  ERR exec: {status.get('error')}", flush=True); return False
        except Exception as e:
            print(f"  poll err: {e}", flush=True)
    print(f"  TIMEOUT", flush=True); return False


def main():
    print(f"=== 仅重生 camo_4 + skull_5 → {OUT_DIR} ===", flush=True)
    for ref in TARGETS:
        run_one(ref)


if __name__ == "__main__":
    sys.exit(main())
