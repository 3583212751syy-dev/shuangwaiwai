"""v127 干净裂变（鲁棒+续跑版）：6 张无文字无LoRA裂变图。
修复 v126 卡死根因：
- /prompt 响应含 {"error":...} 也返回 200 → 必须查 json 里的 error 字段。
- 取图 /view 失败被裸 except 吞掉 → 改为显式失败即返回。
- 每张硬超时（40*5s=200s），超时即跳下一张，不无限等。
- 续跑：输出已存在的图直接跳过。
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REFS = [
    {"id": "illust_1", "ref_img": "pinterest_illust_1.jpg", "weight": 0.55,
     "subject": "ornate black and white scrolling acanthus foliage border frame, baroque floral linework",
     "palette": "monochrome ink line illustration, pure black background, white filigree, no color"},
    {"id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg", "weight": 0.55,
     "subject": ("double-headed eagle crest emblem with flaming aura and iron chains, "
                 "skull throne at base, heraldic shield with banner ribbon, symmetrical"),
     "palette": "gothic tattoo illustration, dark black background, white and red linework, intricate feathers"},
    {"id": "denim_3", "ref_img": "pinterest_denim_3.jpg", "weight": 0.55,
     "subject": "delicate moth flying over a distressed X-shaped denim patch with fabric wear and stitch holes",
     "palette": "vintage faded denim texture, indigo blue and white only, rough brushstroke X mark"},
    {"id": "camo_4", "ref_img": "pinterest_camo_4.jpg", "weight": 0.55,
     "subject": "palm leaves and tropical fronds in military camouflage pattern, camouflage texture",
     "palette": "green brown black camouflage, matte, no text"},
    {"id": "skull_5", "ref_img": "pinterest_skull_5.jpg", "weight": 0.55,
     "subject": "grim human skull with black eye patch, red snake coiled around it, red angel wings, red rose",
     "palette": "gothic tattoo illustration, pure black background, red and white palette"},
    {"id": "metal_6", "ref_img": "pinterest_metal_6.jpg", "weight": 0.55,
     "subject": "bald eagle perched on cracked human skull with massive horn-like spikes and radiating lightning bolts",
     "palette": "death metal thrash metal illustration, pure black background, white and bronze gold yellow palette"},
]

POS_TAIL = (
    "same composition and layout as reference image, "
    "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
    "fills entire frame edge-to-edge, full bleed, no halftone, no noise, no grain"
)
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, "
    "no gray, no midtones, no gradient, no soft shading, no halftone, "
    "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, smudge, "
    "duplicate image, exact copy, watermark"
)
SEED = 700401


def build_clean(ref, seed):
    pos = (f"t-shirt graphic design, {ref['subject']}, {ref['palette']}, {POS_TAIL}")
    neg = NEG_BASE
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": ref["weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0], "positive": ["8", 0], "negative": ["9", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 32, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.72}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0], "positive": ["8", 0], "negative": ["9", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 28, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0],
                       "filename_prefix": f"v127_clean_{ref['id']}"}}
    return g


def gen_clean(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v127_clean_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": build_clean(ref, seed),
                    "client_id": f"v127_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {str(j)[:300]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {tag}: 无 prompt_id {str(j)[:300]}", flush=True); return False
    print(f"  [{tag}] pid={pid}", flush=True)
    for i in range(40):  # 200s 硬超时
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                rec = h[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
                    if imgs:
                        url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                        try:
                            data = requests.get(url, timeout=60).content
                        except Exception as e:
                            print(f"  [{tag}] 取图失败 {e}", flush=True); return False
                        out.write_bytes(data)
                        print(f"  [{tag}] OK {out.stat().st_size/1024/1024:.1f}MB", flush=True); return True
                elif rec.get("status", {}).get("error"):
                    print(f"  [{tag}] COMFY错误 {rec['status'].get('error')}", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] 轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT 跳过重试", flush=True); return False


def main():
    out = PROJECT_ROOT / "jobs" / f"smoke_v127_{int(time.time())}"
    # 续跑：复用已存在的 v127 目录
    existing = sorted([d for d in (PROJECT_ROOT/"jobs").iterdir() if d.is_dir() and d.name.startswith("smoke_v127")])
    if existing:
        out = existing[-1]
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== v127 干净裂变（续跑） 输出={out} ===", flush=True)
    for ref in REFS:
        gen_clean(ref, SEED, out)
    print("done clean", flush=True)


if __name__ == "__main__":
    sys.exit(main())
