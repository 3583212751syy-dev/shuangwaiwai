"""v138: ComfyUI fallback for skull_5 + denim_3 (ImageGen 后端屏蔽这2张).

设计:
  - 复用 v137 build_graph/run_one 骨架
  - DENOISE 0.66 -> 0.72 (内容真改, 不像 v137 几乎复制)
  - cn_weight 0.62 -> 0.50 (弱锁构图, 释放内容自由度)
  - 文字 PIL 后期烧 (确定性拼写, SDXL 拼字不可靠)
  - 单图 ~3-4 分钟, 2 张总 ~8 分钟
"""
import json, time, sys, requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REFS = [
    {"id": "skull_5", "filename": "pinterest_skull_5_hd.png", "weight": 0.45, "cn_weight": 0.50,
     "subject": ("central ornate skull on a symmetric layout, replaced decorative elements: "
                 "spiked iron halo above, geometric thorn-line patterns on both sides, "
                 "ash-grey petals instead of red roses, geometric angular shard wings, "
                 "wilted orchid instead of the red rose, "
                 "banner at the bottom for text, "
                 "keep the exact central skull placement and symmetric composition"),
     "palette": "gothic tattoo illustration, pure deep black background, dark crimson and bone-white accents only, no other color"},
    {"id": "denim_3", "filename": "pinterest_denim_3_flat_hd.png", "weight": 0.45, "cn_weight": 0.50,
     "subject": ("tall portrait layout with large word area at the top and a central denim patchwork subject, "
                 "replaced the central butterfly with an OWL silhouette made of the same denim patchwork pieces, "
                 "keep the small flying butterfly and dotted flight trail in the corners, "
                 "keep the exact rectangular portrait composition"),
     "palette": "vintage faded denim texture, indigo blue and white stitching only, neutral light background, no color tint, no other color"},
]

CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"
SEED = 700801
DENOISE = 0.72

POS_TEMPLATE = ("masterpiece, best quality, ultra detailed, clean illustration, sharp clean edges, "
                "{subject}, intricate linework, {palette}")
NEG = ("text, words, letters, typography, font, alphabet, watermark, signature, logo, badge, "
       "frame, border, blurry, smudged, deformed, disfigured, low quality, bad anatomy, mutation, "
       "cropped, jpeg artifacts, noise, color shift")


def build_graph(ref):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["filename"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
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
        "denoise": DENOISE, "noise_mask": None}}
    g["13"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0],
        "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["12", 0],
        "seed": SEED + 1, "steps": 28, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.18, "noise_mask": None}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["16"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["15", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": "v138_fb_clean"}}
    return g


def run_one(ref, out_dir):
    tag = ref["id"]
    out_path = out_dir / f"v138_fb_clean_{tag}.png"
    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[skip] {tag}: 已存在 {out_path.stat().st_size//1024}KB", flush=True); return True
    print(f"[run] {tag}  cn={ref['cn_weight']}  ipa={ref['weight']}  denoise={DENOISE}", flush=True)
    g = build_graph(ref)
    try:
        r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v138_{int(time.time())}"}, timeout=15)
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
    for i in range(60):
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
    out = PROJECT_ROOT / "jobs" / f"smoke_v138_fb_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== v138 fallback (denoise 0.72, cn 0.50, skull_5 + denim_3) -> {out} ===", flush=True)
    ok = 0
    for ref in REFS:
        if run_one(ref, out):
            ok += 1
    print(f"\n=== done v138: {ok}/{len(REFS)} ===", flush=True)
    return 0 if ok == len(REFS) else 1


if __name__ == "__main__":
    sys.exit(main())
