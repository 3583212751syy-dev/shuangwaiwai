"""v118 全量裂变：img2img(原图比例) + IPAdapter 锁配色 + denoise 0.72 裂变
v117 已验收 illust_1（竖图 3072×5376 纯黑白 12.3MB）→ 全量跑剩余 5 张。

每张保持各自原图宽高比（ImageScaleToTotalPixels 1MP）+ 各自配色 prompt：
- eagle_2: 白红哥特（有字→后期烧）
- denim_3: 靛蓝白牛仔
- camo_4: 棕绿迷彩
- skull_5: 红白哥特（有字→后期烧）
- metal_6: 白青铜金死亡金属（有字→后期烧）

用法：python src/smoke_v118_all.py
"""
import time, requests, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

REFS = [
    {
        "id": "eagle_2",
        "ref_img": "pinterest_eagle_2.jpg",
        "pos": (
            "t-shirt graphic design, majestic spread-winged eagle gripping a skull in its talons, "
            "gothic crest emblem with chains and flames, "
            "same composition and layout as reference image, "
            "dark black background, white and red linework, "
            "high contrast, "
            "REDESIGNED internal details, new intricate feather patterns, "
            "beautifully reimagined crest ornaments, elegant decorative swirls, "
            "rich ornate flourishes, refined artistic variations, "
            "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
            "fills entire frame edge-to-edge, no margins, full bleed, no halftone, no noise, no grain"
        ),
        "neg": (
            "no gray, no midtones, no gradient, no soft shading, no halftone, "
            "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, "
            "margin, border, frame, padding, empty corners, letterbox, white border, smudge, "
            "text, letters, watermark, duplicate image, exact copy"
        ),
    },
    {
        "id": "denim_3",
        "ref_img": "pinterest_denim_3.jpg",
        "pos": (
            "t-shirt graphic design, single delicate moth flying over a distressed X-shaped denim patch, "
            "same composition and layout as reference image, "
            "vintage faded denim texture, indigo blue and white only, "
            "rough brushstroke X mark, fabric wear and tear, dotted flight trail, "
            "high contrast, "
            "REDESIGNED internal details, new intricate denim stitches, "
            "beautifully reimagined moth wing patterns, refined artistic variations, "
            "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
            "fills entire frame edge-to-edge, no margins, full bleed, no halftone, no noise, no grain"
        ),
        "neg": (
            "no gray, no midtones, no gradient, no soft shading, no halftone, "
            "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, "
            "margin, border, frame, padding, empty corners, letterbox, white border, smudge, "
            "text, letters, watermark, duplicate image, exact copy"
        ),
    },
    {
        "id": "camo_4",
        "ref_img": "pinterest_camo_4.jpg",
        "pos": (
            "t-shirt graphic design, tropical palm tree silhouette with classic military camouflage pattern, "
            "same composition and layout as reference image, "
            "tropical jungle camouflage, brown green tan khaki black palette, "
            "black palm tree silhouettes, organic camo blob shapes, "
            "high contrast, "
            "REDESIGNED internal details, new intricate palm leaf textures, "
            "beautifully reimagined camo patterns, refined artistic variations, "
            "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
            "fills entire frame edge-to-edge, no margins, full bleed, no halftone, no noise, no grain"
        ),
        "neg": (
            "no gray, no midtones, no gradient, no soft shading, no halftone, "
            "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, "
            "margin, border, frame, padding, empty corners, letterbox, white border, smudge, "
            "text, letters, watermark, duplicate image, exact copy"
        ),
    },
    {
        "id": "skull_5",
        "ref_img": "pinterest_skull_5.jpg",
        "pos": (
            "t-shirt graphic design, grim human skull with red snake coiled around it and red angel wings, "
            "same composition and layout as reference image, "
            "gothic tattoo illustration, pure black background, red and white palette, "
            "blood drips, cracked bone, serpent scales, feathered wings, "
            "high contrast, "
            "REDESIGNED internal details, new intricate serpent scale patterns, "
            "beautifully reimagined wing feathers, refined artistic variations, "
            "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
            "fills entire frame edge-to-edge, no margins, full bleed, no halftone, no noise, no grain"
        ),
        "neg": (
            "no gray, no midtones, no gradient, no soft shading, no halftone, "
            "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, "
            "margin, border, frame, padding, empty corners, letterbox, white border, smudge, "
            "text, letters, watermark, duplicate image, exact copy"
        ),
    },
    {
        "id": "metal_6",
        "ref_img": "pinterest_metal_6.jpg",
        "pos": (
            "t-shirt graphic design, bald eagle perched on a cracked human skull with horn-like spikes and lightning bolts, "
            "same composition and layout as reference image, "
            "death metal thrash metal illustration, pure black background, white and bronze gold yellow palette, "
            "sharp aggressive lines, radial lightning burst, heavy ink, "
            "high contrast, "
            "REDESIGNED internal details, new intricate skull cracks, "
            "beautifully reimagined lightning patterns, refined artistic variations, "
            "masterpiece, best quality, ultra detailed, clean vector illustration, sharp clean edges, "
            "fills entire frame edge-to-edge, no margins, full bleed, no halftone, no noise, no grain"
        ),
        "neg": (
            "no gray, no midtones, no gradient, no soft shading, no halftone, "
            "3d, photographic, painterly, blur, noise, grain, pixelated, jagged edges, "
            "margin, border, frame, padding, empty corners, letterbox, white border, smudge, "
            "text, letters, watermark, duplicate image, exact copy"
        ),
    },
]


def build_workflow(ref, seed, denoise):
    """同 v117：原始图 -> ImageScaleToTotalPixels(1MP 保比例) -> img2img + IPAdapter 0.55"""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0],
        "upscale_method": "lanczos",
        "megapixels": 1.0,
        "resolution_steps": 64,
    }}
    g["4"] = {"class_type": "VAEEncode",
              "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader",
              "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0],
        "ipadapter": ["5", 1],
        "image": ["3", 0],
        "weight": 0.55,
        "weight_type": "style transfer",
        "combine_embeds": "average",
        "start_at": 0.0,
        "end_at": 0.85,
        "noise": 0.05,
        "embeds_scaling": "V only",
    }}
    g["7"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": ref["pos"]}}
    g["8"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": ref["neg"]}}
    g["9"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0],
        "positive": ["7", 0],
        "negative": ["8", 0],
        "latent_image": ["4", 0],
        "seed": seed,
        "steps": 45,
        "cfg": 7.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": denoise,
    }}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0],
        "positive": ["7", 0],
        "negative": ["8", 0],
        "latent_image": ["9", 0],
        "seed": seed + 1,
        "steps": 40,
        "cfg": 7.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 0.20,
    }}
    g["11"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["13"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["12", 0], "image": ["11", 0]}}
    g["14"] = {"class_type": "SaveImage",
               "inputs": {"images": ["13", 0],
                          "filename_prefix": f"v118_{ref['id']}"}}
    return g


def run_one(ref, seed, denoise, out_base):
    tag = ref["id"]
    workflow = build_workflow(ref, seed, denoise)
    payload = {"prompt": workflow, "client_id": f"v118_{int(time.time())}"}
    r = requests.post(f"{COMFYUI}/prompt", json=payload, timeout=10)
    if r.status_code != 200:
        print(f"[ERR] submit {tag} failed: {r.status_code} {r.text[:300]}", flush=True)
        return False
    pid = r.json().get("prompt_id")
    print(f"  [{tag}] prompt_id={pid}", flush=True)

    for i in range(140):
        time.sleep(5)
        try:
            q = requests.get(f"{COMFYUI}/history/{pid}", timeout=5).json()
            if pid in q:
                rec = q[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("14", {}).get("images", [])
                    if imgs:
                        fname = imgs[0]["filename"]
                        sub = imgs[0].get("subfolder", "")
                        url = f"{COMFYUI}/view?filename={fname}&type=output&subfolder={sub}"
                        data = requests.get(url, timeout=30).content
                        out = out_base / f"{tag}.jpg"
                        out.write_bytes(data)
                        print(f"  [{tag}] OK {out} ({out.stat().st_size/1024/1024:.1f}MB)", flush=True)
                        return True
                elif rec.get("status", {}).get("error"):
                    print(f"  [{tag}] ERR {rec['status']}", flush=True)
                    return False
        except Exception:
            pass
        if i % 12 == 0:
            print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT", flush=True)
    return False


def main():
    out_base = PROJECT_ROOT / "jobs" / f"smoke_v118_{int(time.time())}"
    out_base.mkdir(parents=True, exist_ok=True)
    denoise = 0.72
    seed = 100301
    print(f"=== v118 全量 5 张（denoise {denoise}，各保原图比例）===", flush=True)
    for ref in REFS:
        ok = run_one(ref, seed, denoise, out_base)
        seed += 100
        if not ok:
            print(f"  [FAIL] {ref['id']} 继续下一张", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
