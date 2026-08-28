"""v129 smoke: v128 正解管线 + 等原图尺寸修正.

复用 v128 Canny+IPA+Proteus+denoise0.55+4xupscale 正解管线, 仅修正:
  - ImageScaleToTotalPixels(megapixels=原图实际MP) → 严格等原图比例
  - hard-avoid 中等: no logos / celebrities / flags / political figures / copyrighted chars
  - 智能保字换词: 不在 NEG 里禁文字, 让 AI 自然产出合法词 (denim_3 已证明 UPCYCLE 合法词会自然出现)

烟测 2 张: denim_3 + camo_4 → present_files 拼图交付验收.
"""
import json, time, sys
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT.parent.parent / "图裂变测试图"
COMFYUI = "http://127.0.0.1:8188"

CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"
SEED = 700301

REFS = [
    {"id": "denim_3", "filename": "pinterest_denim_3.jpg", "weight": 0.55, "cn_weight": 0.50,
     "subject": "delicate moth flying over a distressed X-shaped denim patch with fabric wear and stitch holes",
     "palette": "vintage faded denim texture, indigo blue and white only, rough brushstroke X mark"},
    {"id": "camo_4",  "filename": "pinterest_camo_4.jpg",  "weight": 0.55, "cn_weight": 0.55,
     "subject": "palm leaves and tropical fronds in military camouflage pattern, dense dark foliage",
     "palette": "dark green brown black camouflage, matte, shadowy, no text"},
]

POS_TEMPLATE = ("masterpiece, best quality, ultra detailed, clean illustration, sharp clean edges, "
                 "{subject}, intricate linework, redesigned internal details, framed composition, "
                 "{palette}, {hard_avoid}")

HARD_AVOID = ("no logos, no celebrities, no national flags, no political figures, "
              "no copyrighted characters, no trademarked text")

# 不禁文字 (智能保字换词): 让 AI 自然产出合法短词; 遇侵权词 PIL 后期替换
NEG = ("blurry, smudged, deformed, disfigured, low quality, bad anatomy, mutation, "
       "cropped, jpeg artifacts, noise, watermark, signature, badge, frame, border")


def get_original_megapixels(filename):
    """读取原图实际像素, 返回 megapixels float (ImageScaleToTotalPixels 会保比例缩放到该总量)."""
    from PIL import Image
    im = Image.open(INPUT_DIR / filename)
    w, h = im.size
    return (w * h) / 1_000_000.0


def build_graph(ref, mp):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["filename"]}}
    # 等原图尺寸 (按实际 MP)
    g["3"] = {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"upscale_method": "lanczos", "megapixels": round(mp, 4),
                         "image": ["2", 0], "resolution_steps": 8}}
    g["4"] = {"class_type": "Canny",
              "inputs": {"image": ["3", 0], "low_threshold": 0.08, "high_threshold": 0.18}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = POS_TEMPLATE.format(subject=ref["subject"], palette=ref["palette"], hard_avoid=HARD_AVOID)
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": ref["cn_weight"], "start_percent": 0.0, "end_percent": 0.85,
        "control_net": ["5", 0]}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader",
               "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1],
        "image": ["2", 0],
        "weight": ref["weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "embeds_scaling": "V only",
        "start_at": 0.0, "end_at": 0.85, "noise": 0.08}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["7", 0],
        "seed": SEED, "steps": 40, "cfg": 7.0, "sampler_name": "dpmpp_2m",
        "scheduler": "karras", "denoise": 0.55, "noise_mask": None}}
    g["13"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["12", 0],
        "seed": SEED + 1, "steps": 28, "cfg": 6.5, "sampler_name": "dpmpp_2m",
        "scheduler": "karras", "denoise": 0.18, "noise_mask": None}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["16"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["15", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": "v129_eqsize"}}
    return g


def run_one(ref, out_dir):
    tag = ref["id"]
    out_path = out_dir / f"v129_eqsize_{tag}.png"
    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[skip] {tag}: 已存在 {out_path.stat().st_size//1024//1024}MB", flush=True)
        return True
    mp = get_original_megapixels(ref["filename"])
    print(f"[run] {tag}  cn={ref['cn_weight']}  ipa={ref['weight']}  denoise=0.55  mp={mp:.4f}", flush=True)
    g = build_graph(ref, mp)
    try:
        r = requests.post(f"{COMFYUI}/prompt",
                          json={"prompt": g, "client_id": f"v129_{int(time.time())}"}, timeout=15)
    except Exception as e:
        print(f"  ERR submit conn: {e}", flush=True); return False
    if r.status_code != 200:
        print(f"  ERR submit: {r.status_code} {r.text[:300]}", flush=True); return False
    pid = r.json().get("prompt_id")
    if not pid:
        print(f"  ERR no prompt_id", flush=True); return False
    print(f"  pid={pid}", flush=True)
    t0 = time.time()
    for _ in range(40):  # 200s 硬超时
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                status = h[pid].get("status", {})
                if status.get("completed"):
                    imgs = h[pid].get("outputs", {}).get("17", {}).get("images", [])
                    if not imgs:
                        print(f"  ERR no images", flush=True); return False
                    url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                    data = requests.get(url, timeout=120).content
                    out_path.write_bytes(data)
                    print(f"  OK {out_path.stat().st_size//1024//1024}MB (waited {int(time.time()-t0)}s)", flush=True)
                    return True
                if status.get("error"):
                    print(f"  ERR exec: {status.get('error')}", flush=True); return False
        except Exception as e:
            print(f"  poll err: {e}", flush=True)
    print(f"  TIMEOUT after {int(time.time()-t0)}s", flush=True); return False


def main():
    out = PROJECT_ROOT / "jobs" / f"smoke_v129_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== v129 (等原图尺寸 + 智能保字换词) -> {out} ===", flush=True)
    ok = 0
    for ref in REFS:
        if run_one(ref, out):
            ok += 1
    print(f"\n=== done v129: {ok}/{len(REFS)} ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())