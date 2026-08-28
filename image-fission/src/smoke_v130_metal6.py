"""v130 metal_6 专跑: 17.6MP 超高清, 降到 4MP 基准裂变 -> PIL 重采样回原图尺寸.

原因: ImageScaleToTotalPixels(16MP) latent 过大 + 4x NMKD 超分爆显存/超时.
方案: cap megapixels=4.0 (保 AR) -> 出图后 PIL resize 回 3543x4961 (等原图尺寸).
"""
import json, time, sys
from pathlib import Path
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = Path("E:/Desktop/图裂变测试图")
COMFYUI = "http://127.0.0.1:8188"
CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"
SEED = 700301

OUT_DIR = PROJECT_ROOT / "jobs" / "smoke_v130_metal6_4mp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RAW = OUT_DIR / "v130_metal_6_raw.png"
OUT_FINAL = OUT_DIR / "v130_metal_6.png"

REF = {
    "id": "metal_6", "filename": "pinterest_metal_6.jpg",
    "cn_weight": 0.55, "ipa_weight": 0.50,
    "subject": ("ornate weathered bronze medallion with carved stone relief, classical barbarian metalwork, "
                "REDESIGNED carved motif, completely different ornamental pattern"),
    "palette": "oxidized bronze patina, aged tarnished metal texture, sepia brown gold antiqued finish",
}

POS_TEMPLATE = ("masterpiece, best quality, ultra detailed, sharp clean edges, intricate linework, "
                "{subject}, completely redrawn internal details, redesigned color placement, "
                "framed composition, {palette}, {hard_avoid}")
HARD_AVOID = ("no logos, no celebrities, no national flags, no political figures, "
              "no copyrighted characters, no trademarked text, no readable slogans")
NEG = ("blurry, smudged, deformed, disfigured, low quality, bad anatomy, mutation, "
       "cropped, jpeg artifacts, noise, watermark, signature, "
       "illegible text, garbled letters, random text, gibberish words, distorted text")


def build_graph(mp):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": REF["filename"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"upscale_method": "lanczos", "megapixels": round(mp, 4),
                         "image": ["2", 0], "resolution_steps": 8}}
    g["4"] = {"class_type": "Canny",
              "inputs": {"image": ["3", 0], "low_threshold": 0.08, "high_threshold": 0.18}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = POS_TEMPLATE.format(subject=REF["subject"], palette=REF["palette"], hard_avoid=HARD_AVOID)
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": REF["cn_weight"], "start_percent": 0.0, "end_percent": 0.85,
        "control_net": ["5", 0]}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader",
               "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1], "image": ["2", 0],
        "weight": REF["ipa_weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "embeds_scaling": "V only",
        "start_at": 0.0, "end_at": 0.85, "noise": 0.08}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["7", 0], "seed": SEED, "steps": 40, "cfg": 7.0,
        "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.72, "noise_mask": None}}
    g["13"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["12", 0], "seed": SEED + 1, "steps": 28, "cfg": 6.5,
        "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.18, "noise_mask": None}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["16"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["15", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage",
               "inputs": {"images": ["16", 0], "filename_prefix": "v130_metal6_4mp"}}
    return g


def main():
    if OUT_FINAL.exists() and OUT_FINAL.stat().st_size > 100_000:
        print(f"[skip] {OUT_FINAL} 已存在"); return 0
    mp = 4.0  # 4MP 基准 (保 AR), 原图 17.6MP 太高清跑不动
    g = build_graph(mp)
    r = requests.post(f"{COMFYUI}/prompt",
                      json={"prompt": g, "client_id": f"v130m6_{int(time.time())}"}, timeout=15)
    if r.status_code != 200:
        print(f"ERR submit: {r.status_code} {r.text[:300]}"); return 1
    pid = r.json().get("prompt_id")
    print(f"pid={pid} mp={mp}", flush=True)
    t0 = time.time()
    for _ in range(40):
        time.sleep(5)
        h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed"):
                imgs = h[pid].get("outputs", {}).get("17", {}).get("images", [])
                if not imgs:
                    print("ERR no images"); return 1
                url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                OUT_RAW.write_bytes(requests.get(url, timeout=120).content)
                # PIL 重采样回原图精确尺寸 (3543x4961)
                src = Image.open(OUT_RAW)
                src = src.resize((3543, 4961), Image.LANCZOS)
                src.save(OUT_FINAL)
                print(f"OK {OUT_FINAL} {src.size} (waited {int(time.time()-t0)}s)")
                return 0
            if st.get("error"):
                print(f"ERR exec: {st.get('error')}"); return 1
    print(f"TIMEOUT {int(time.time()-t0)}s"); return 1


if __name__ == "__main__":
    sys.exit(main())
