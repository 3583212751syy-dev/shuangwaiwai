"""v137: 锁结构裂变 v2 — HD高清参考 + 强Canny(0.62) + 强denoise(0.66) + 具体替换prompt.

v135 (denoise 0.62, vague prompt) 裂变太弱 (SSIM 0.85+ ≈ 原图复制).
v137 改进:
  1. 用 *_hd.png (4x-UltraSharp) 替代 jpg (v128 教训: 小尺寸压缩图锁进风格向量)
  2. Canny cn 0.55→0.62 (强锁构图, 抗 denoise 提升的形变)
  3. denoise 0.62→0.66 (给内容裂变更多空间, 仍 < 0.72 的崩结构阈值)
  4. 关键: subject prompt 改为 ImageGen 风格的"具体替换指令"
     — 明确说"把 X 换成 Y", 而非笼统"redesign internals"
  5. 文字: 后期 PIL 烧 (确定性拼写, 不靠 LoRA)
"""
import json, time, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

# Concrete replacement prompts: KEEP composition+palette, REPLACE internal X with Y.
# (借鉴 ImageGen 验证: 具体替换 > 笼统 redesign)
REFS = [
    {"id": "illust_1", "filename": "pinterest_illust_1_hd.png", "weight": 0.55, "cn_weight": 0.62,
     "subject": ("ornate symmetrical decorative border frame on dark canvas; "
                 "REPLACE the internal magnolia floral motif with a different botanical: "
                 "lotus blossoms, bamboo stalks and willow tendrils in the same filigree linework style; "
                 "keep the exact outer frame shape and layout"),
     "palette": "monochrome ink illustration, pure black background, white filigree only, no color, no text"},
    {"id": "eagle_2",  "filename": "pinterest_eagle_2_hd.png",  "weight": 0.55, "cn_weight": 0.62,
     "subject": ("spread-wing bird of prey perched on heraldic shield with three skulls at base and chain; "
                 "REPLACE the bird's natural feathers with overlapping geometric triangular scales (sacred-geometry mandala style); "
                 "REPLACE the flames with stylized flowing ribbons with rune-like patterns; "
                 "REPLACE the shield engravings with bold Celtic knotwork; "
                 "keep exact composition, keep the small raven above"),
     "palette": "gothic tattoo illustration, pure black background, vivid orange-red flames, white bird and skulls, high contrast"},
    {"id": "denim_3",  "filename": "pinterest_denim_3_flat_hd.png", "weight": 0.55, "cn_weight": 0.62,
     "subject": ("distressed X-shaped denim patch collage with fabric wear and visible stitch holes; "
                 "REPLACE the X shape with an OWL silhouette made of the same denim patchwork pieces; "
                 "keep the small flying moth and the dotted flight trail; "
                 "keep the exact rectangular portrait composition"),
     "palette": "vintage faded denim texture, indigo blue and white stitching only, neutral light background, no color tint"},
    {"id": "camo_4",   "filename": "pinterest_camo_4_hd.png",   "weight": 0.55, "cn_weight": 0.62,
     "subject": ("dense military camouflage pattern of overlapping foliage shapes; "
                 "REPLACE the tropical palm fronds with pine branches and snow-dusted fir needles; "
                 "keep the exact camouflage density and overall rectangular composition; "
                 "no text anywhere"),
     "palette": "dark olive green, brown, black camouflage only, matte and shadowy, no bright color"},
    {"id": "skull_5",  "filename": "pinterest_skull_5_hd.png",  "weight": 0.55, "cn_weight": 0.62,
     "subject": ("central human skull composition with serpent and wings; "
                 "REPLACE the eye patch with an ornate iron crown; "
                 "REPLACE the coiled snake with a thorny vine wreath; "
                 "REPLACE the feathered wings with geometric angular shard wings; "
                 "REPLACE the rose with a wilted orchid; "
                 "keep exact central skull placement and symmetrical layout"),
     "palette": "gothic tattoo illustration, pure deep black background, crimson red and bone-white accents only"},
    {"id": "metal_6",  "filename": "pinterest_metal_6_hd.png",  "weight": 0.55, "cn_weight": 0.62,
     "subject": ("death metal crest with central eagle-and-skull motif and radiating spikes; "
                 "REPLACE the lightning bolts with heavy industrial chain links; "
                 "REPLACE the horn spikes with rusted gears and bolts; "
                 "keep the central eagle-on-skull placement and the radiating composition"),
     "palette": "death metal thrash metal illustration, pure black background, white and bronze gold only, no other color"},
]

CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"
SEED = 700701
DENOISE = 0.66

POS_TEMPLATE = ("masterpiece, best quality, ultra detailed, clean illustration, sharp clean edges, "
                "{subject}, intricate linework, framed composition, {palette}")
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
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": "v137_clean"}}
    return g


def run_one(ref, out_dir):
    import requests
    tag = ref["id"]
    out_path = out_dir / f"v137_clean_{tag}.png"
    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[skip] {tag}: 已存在 {out_path.stat().st_size//1024}KB", flush=True); return True
    print(f"[run] {tag}  cn={ref['cn_weight']}  ipa={ref['weight']}  denoise={DENOISE}  (HD+具体prompt)", flush=True)
    g = build_graph(ref)
    try:
        r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v137_{int(time.time())}"}, timeout=15)
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
    for i in range(60):  # 300s 硬超时 (HD+双KSampler 较慢)
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
    out = PROJECT_ROOT / "jobs" / f"smoke_v137_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== v137 (HD ref + Canny 0.62 + denoise {DENOISE} + 具体替换prompt) -> {out} ===", flush=True)
    ok = 0
    for ref in REFS:
        if run_one(ref, out):
            ok += 1
    print(f"\n=== done v137: {ok}/{len(REFS)} ===", flush=True)
    return 0 if ok == len(REFS) else 1


if __name__ == "__main__":
    sys.exit(main())
