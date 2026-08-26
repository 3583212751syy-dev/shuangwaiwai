"""v113 治本：img2img 直接基于 HD 参考重画 + 强 prompt 控纯黑白 + 防白边处理。

v9.8 验证过 img2img 能保住主体（孔雀出现），v113 改进：
- prompt 强"pure black background, white linework only, no color, no gradient" → 纯黑白
- prompt 强"fills entire frame edge-to-edge, no white border" + 负向"no padding/no margin" → 治 v9.8 白边
- 不用 IPAdapter/Canny/LoRA（最简管线，依赖最少）
- 底模 Proteus
- KSampler1 denoise 0.50（保留更多原图信息，避免变灰/变色）
- HiRes denoise 0.20
- 4x NMKD-Siax 超分

用法：python src/smoke_v113_illust.py
"""
import json, os, sys, time, requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"


def build_workflow(ref_img, subject_word, seed):
    """v113 管线：img2img + 强纯黑白 prompt + 防白边 + Proteus + NMKD-Siax 超分。

    1: CheckpointLoader (Proteus)
    2: CLIPTextEncode (positive, 强纯黑白+防白边)
    3: CLIPTextEncode (negative, 防白边+防色)
    4: LoadImage (HD 参考)
    5: ImageScale (1024² fit, 保持比例)
    6: KSampler 1 (denoise 0.50, steps 45, cfg 7.5)
    7: KSampler 2 (hires denoise 0.20, steps 40)
    8: VAEDecode
    9: UpscaleModelLoader (4x_NMKD-Siax_200k.pth)
    10: ImageUpscaleWithModel
    11: SaveImage
    """
    g = {}

    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}

    pos = (
        f"t-shirt graphic, {subject_word}, "
        "pure black background, white linework only, "
        "no color, no gradient, no shading, no halftone, no midtones, no gray, "
        "high contrast, sharp clean edges, "
        "fills entire frame edge-to-edge, no margins, no white border, no padding, full bleed"
    )
    neg = (
        "color, gradient, shading, halftone, midtones, gray, 3d, photographic, painterly, "
        "blur, noise, grain, pixelated, jagged edges, "
        "white border, margin, border, frame, padding, empty corners, letterbox, smudge, "
        "soft shading, low contrast, washed out"
    )

    g["2"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": pos}}
    g["3"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": neg}}

    g["4"] = {"class_type": "LoadImage", "inputs": {"image": ref_img}}

    # ImageScale 到 1024²：保持比例 fit-in（letterbox 会有 padding，但 prompt 强控+超分后看不见）
    g["5"] = {"class_type": "ImageScale", "inputs": {
        "image": ["4", 0],
        "width": 1024,
        "height": 1024,
        "upscale_method": "lanczos",
        "crop": "center",
    }}

    g["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}}

    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0],
        "positive": ["2", 0],
        "negative": ["3", 0],
        "latent_image": ["6", 0],
        "seed": seed,
        "steps": 45,
        "cfg": 7.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 0.50,
    }}

    g["8"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0],
        "positive": ["2", 0],
        "negative": ["3", 0],
        "latent_image": ["7", 0],
        "seed": seed + 1,
        "steps": 40,
        "cfg": 7.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 0.20,
    }}

    g["9"] = {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["1", 2]}}

    g["10"] = {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}

    g["11"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["10", 0], "image": ["9", 0]}}

    g["12"] = {"class_type": "SaveImage",
               "inputs": {"images": ["11", 0], "filename_prefix": f"v113_illust_1"}}

    return g


def main():
    out_base = PROJECT_ROOT / "jobs" / f"smoke_v113_{int(time.time())}"
    out_base.mkdir(parents=True, exist_ok=True)

    # illust_1 验证：保留孔雀主体，强纯黑白，防白边
    ref_img = "pinterest_illust_1_hd.png"
    subject_word = (
        "elegant peacock silhouette with elaborate flowing tail feathers, "
        "surrounded by 5-petal flowers and decorative swirling flourishes, "
        "baroque vector illustration style"
    )
    seed = 100301

    print(f"=== v113 验证：illust_1 (img2img+纯黑白+防白边) ===")
    print(f"  ref_img={ref_img}")
    print(f"  denoise=0.50 (保留更多原图信息，颜色稳)")

    workflow = build_workflow(ref_img, subject_word, seed)
    payload = {"prompt": workflow, "client_id": f"v113_{int(time.time())}"}

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
                    imgs = rec.get("outputs", {}).get("12", {}).get("images", [])
                    if imgs:
                        fname = imgs[0]["filename"]
                        subfolder = imgs[0].get("subfolder", "")
                        url = f"{COMFYUI}/view?filename={fname}&type=output&subfolder={subfolder}"
                        data = requests.get(url, timeout=30).content
                        out_path = out_base / f"illust_1_{seed}.jpg"
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
