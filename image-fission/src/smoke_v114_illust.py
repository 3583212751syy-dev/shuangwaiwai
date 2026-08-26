"""v114 裂变平衡点：txt2img + IPAdapter style 降权 0.40（用户拍板方向）
v11.0 太变（区别大）/ v113 不变（长得一样）→ 降权 + 贴近原图构图词找中间点。
- IPAdapter weight 0.55 → 0.40（让画面更贴近原图，但仍是 AI 自由重画）
- 去掉 Canny（v111 变灰根因）
- prompt 加原图构图/布局描述词（主体位置、尾羽展开方向、装饰分布）

用法：python src/smoke_v114_illust.py
"""
import json, os, sys, time, requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REFS = [
    {
        "id": "illust_1",
        "ref_img": "pinterest_illust_1_hd.png",
        "subject_word": "elegant peacock silhouette with elaborate flowing tail feathers",
        # 构图词：贴近原图布局（主体中下、尾羽扇形上撑满、五瓣花点缀四周、竖构图满幅）
        "composition_words": "peacock positioned at lower center, tail feathers fanned upward filling the upper two thirds of the frame, ornate 5-petal flowers scattered around the feathers, vertical composition, subject large and centered, fills entire frame edge-to-edge, full bleed",
        "style_words": "baroque vector illustration, pure black background, white linework only, ornamental 5-petal flowers and decorative swirling flourishes, t-shirt graphic, clean sharp edges, no halftone, no noise, no grain, no jagged edges",
        "negative": "no gray, no midtones, pure black and white only, high contrast, no soft shading, no gradient, no shading, no color, no halftone, 3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, margin, border, frame, padding, empty corners, letterbox, white border, smudge",
    },
]


def build_workflow(ref, seed):
    """v114 接法（回到 v11.0 无 Canny 结构，仅权重降到 0.40 + 构图词）：
    1: CheckpointLoader (Proteus)
    2: CLIPTextEncode (positive)
    3: CLIPTextEncode (negative)
    4: IPAdapterUnifiedLoader (model=1.0, preset=PLUS)
    5: LoadImage (HD ref)
    6: IPAdapterAdvanced (model=1.0, ipadapter=4.1, image=5.0, weight=0.40, style transfer)
    7: EmptyLatentImage 1024²
    8: KSampler 1 (model=6.0, denoise=1.0, steps=45, cfg=7.5)
    9: KSampler 2 (model=6.0, denoise=0.20, hires 细化)
    10: VAEDecode
    11: UpscaleModelLoader (4x_NMKD-Siax_200k.pth)
    12: ImageUpscaleWithModel
    13: SaveImage
    """
    g = {}

    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}

    pos = (f"t-shirt graphic design, {ref['subject_word']}, {ref['composition_words']}, "
           f"{ref['style_words']}, masterpiece, best quality, ultra detailed, clean illustration")
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
        "weight": 0.40,
        "weight_type": "style transfer",
        "combine_embeds": "average",
        "start_at": 0.0,
        "end_at": 0.85,
        "noise": 0.05,
        "embeds_scaling": "V only",
    }}

    g["7"] = {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}

    g["8"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0],
        "positive": ["2", 0],
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
        "positive": ["2", 0],
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
               "inputs": {"images": ["12", 0], "filename_prefix": f"v114_{ref['id']}"}}

    return g


def main():
    out_base = PROJECT_ROOT / "jobs" / f"smoke_v114_{int(time.time())}"
    out_base.mkdir(parents=True, exist_ok=True)

    ref = REFS[0]
    seed = 100301
    print(f"=== v114 验证：{ref['id']} ===")
    print(f"  ref_img={ref['ref_img']}")
    print(f"  subject={ref['subject_word'][:60]}...")

    workflow = build_workflow(ref, seed)
    payload = {"prompt": workflow, "client_id": f"v114_{int(time.time())}"}

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
