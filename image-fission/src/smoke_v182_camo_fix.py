# -*- coding: utf-8 -*-
"""v182 camo_4 修复版：v181 颜色漂移（HEX 偏冷）+ 椰子树乱码（具体角度指定）的根因修复

v181 翻车根因:
  1. 颜色: 区域4写了具体 HEX "#C2B280" (偏冷米黄), SDXL 不严格按 HEX 渲染, 被引导到亮黄绿色调
  2. 椰子树乱码: 5 个独立区域提示里具体指定"弯30°/垂直/左倾20°", 孤立区域没连贯参考, 每棵变扭曲黑色团

v182 修复策略:
  - 颜色: 全部 language 描述 (warm deep olive green / warm earth brown / warm sand beige /
    pure black silhouette), 不写 HEX
  - 椰子树: 全局统一描述 "tall straight black palm tree silhouette with clean fan-shaped
    fronds" -- 每棵都画标准椰子树剪影 (只大小/数量不同, 角度靠种子自然变化)
  - 迷彩斑块: 圆润度+4色配色锁死, 只改大小/分布
  - 严格遵守红线: denoise/tile/canny/IPA/LORA_DETAIL 全不动

用法: python smoke_v182_camo_fix.py
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700407
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 0.0  # v174 防碎裂
MEGA_PIXELS = 1.2

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, alphabet, "
    "banner, banner inscription, engraved lettering, runic text, readable text, glyphs, calligraphy, "
    "3d, photographic, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "melted, fused, smudged, bleeding, water damaged, anatomically incorrect, "
    "extra wings, asymmetric error, garbled forms, nonsense, AI artifact, tiling, repeating pattern, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "elements touching each other, elements touching neighboring elements, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "crowded center, cluttered middle area, "
    "low contrast between adjacent elements, no clear black separating outline, "
    "new colors, different color palette, extra colors, color shift, "
    "bright yellow green, lime green, acid green, neon green, "
    "cool muted beige, cold pale tan, washed out color, desaturated, "
    "sharp geometric edges, polygonal color blocks, fractured camouflage, shattered pattern, "
    "abstract black blobs, twisted black shapes, deformed tree silhouettes, "
    "pastel color, washed out color, desaturated"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

# ===== v182 camo_4 5 区域：圆润 + 4色 language + 椰子树剪影 =====
REFS = [
    {
        "id": "camo_4", "ref_img": "pinterest_camo_4.jpg",
        "global_pos": ("bold organic rounded military camouflage print pattern, "
                       "vector illustration style, "
                       "large SOFT ROUNDED ORGANIC BLOB color blocks "
                       "in WARM deep olive green, WARM earth brown, and WARM sand beige "
                       "(exactly 3 camouflage colors only, no other colors, no yellow green), "
                       "plus pure black palm tree silhouettes with crisp clean outlines, "
                       "no text, no letters, no words anywhere, "
                       "fabric print quality, repeatable seamless pattern feel, "
                       "blob edges are soft organic curves not sharp geometric polygons"),
        "regions": [
            # 区域1 中央主棕榈 (大) - 不指定具体角度, 只描述大小
            {"x": 0.30, "y": 0.00, "w": 0.40, "h": 0.50, "strength": 1.25,
             "prompt": ("ONE tall straight black palm tree silhouette in clean vector style, "
                        "tall straight vertical trunk with subtle horizontal bark segment lines, "
                        "topped with a full crown of natural fan-shaped palm fronds spreading outward, "
                        "LARGER than usual taking most of the vertical space, "
                        "pure black silhouette with crisp clean outlines, "
                        "tropical military style. " + COHESIVE)},
            # 区域2 右侧椰子树 (中) - 不指定角度, 只描述大小
            {"x": 0.62, "y": 0.20, "w": 0.30, "h": 0.50, "strength": 1.20,
             "prompt": ("ONE secondary straight black palm tree silhouette in clean vector style, "
                        "tall straight trunk, smaller crown of natural fan-shaped palm fronds, "
                        "MEDIUM size, "
                        "pure black silhouette with crisp outline, "
                        "tropical military style. " + COHESIVE)},
            # 区域3 左下小棕榈 (小) - 不指定角度, 只描述更小
            {"x": 0.05, "y": 0.45, "w": 0.30, "h": 0.55, "strength": 1.10,
             "prompt": ("ONE small straight black palm tree silhouette in clean vector style, "
                        "short straight trunk, smaller crown of natural fan-shaped palm fronds, "
                        "YOUNG and SLENDER, "
                        "pure black silhouette, "
                        "tropical military style. " + COHESIVE)},
            # 区域4 迷彩斑块 (4色, language 描述, 不写 HEX)
            {"x": 0.00, "y": 0.20, "w": 1.0, "h": 0.30, "strength": 1.30,
             "prompt": ("large rounded organic camouflage blobs in MIXED SIZES, "
                        "some as large as palm tree crown, some as small as frond tip, "
                        "three WARM camouflage colors ONLY: warm deep olive green, warm earth brown, "
                        "and warm sand beige (no yellow green, no lime, no neon, no pastel), "
                        "all blob edges are SOFT ROUNDED ORGANIC CURVES not sharp geometric, "
                        "no gradient, no soft airbrush, "
                        "fabric-print-ready military camouflage pattern. " + COHESIVE)},
            # 区域5 底部队列 (数量增 + 大小递减, 角度自然不指定)
            {"x": 0.00, "y": 0.92, "w": 1.0, "h": 0.08, "strength": 1.05,
             "prompt": ("a thin horizontal band of EIGHT small straight black palm tree silhouettes in a row at the very bottom, "
                        "each tree with straight trunk and natural fan-shaped fronds, "
                        "sizes GRADUALLY DECREASING from left to right, "
                        "pure black silhouettes with crisp outlines, "
                        "tactical collar-band decoration. " + COHESIVE)},
        ],
    },
]


def scaled_region_strengths(ref):
    return [{**r, "strength": r["strength"] * REGION_STRENGTH_SCALE} for r in ref["regions"]]


def build(ref, seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": MEGA_PIXELS, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}

    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1],
        "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}

    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": CANNY_STRENGTH}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": TILE_STRENGTH}}

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": ref["global_pos"]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    region_nodes = []
    for i, r in enumerate(scaled_region_strengths(ref)):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"]}}
        region_nodes.append(sk)

    comb_in = {"global_cond": ["pg", 0]}
    for i, sk in enumerate(region_nodes):
        comb_in[f"region{i+1}"] = [sk, 0]
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 24, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v182_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v182_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在 {out.stat().st_size/1024/1024:.1f}MB，跳过", flush=True); return True
    if not ref.get("regions"):
        print(f"  [{tag}] 无 regions 配置，跳过", flush=True); return True

    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v182_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {str(j)[:1500]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {tag}: 无 prompt_id {str(j)[:400]}", flush=True); return False
    print(f"  [{tag}] pid={pid} running...", flush=True)
    for i in range(72):
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
                        try:
                            from PIL import Image, ImageFilter
                            im = Image.open(out).convert('RGB')
                            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                            sharp.save(out, 'JPEG', quality=95, optimize=True)
                            print(f"  [{tag}] USM锐化 {out.stat().st_size/1024/1024:.1f}MB", flush=True)
                        except Exception as e:
                            print(f"  [{tag}] USM失败 原图保留 {e}", flush=True)
                        print(f"  [{tag}] OK {out.stat().st_size/1024/1024:.1f}MB", flush=True); return True
                elif rec.get("status", {}).get("error"):
                    err = rec["status"].get("error")
                    print(f"  [{tag}] COMFY错误 {str(err)[:600]}", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] 轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT 跳过重试", flush=True); return False


def main():
    wants = sys.argv[1:] if len(sys.argv) > 1 else [r["id"] for r in REFS]
    out = PROJECT_ROOT / "jobs" / "smoke_v182"
    out.mkdir(parents=True, exist_ok=True)
    for want in wants:
        ref = next((r for r in REFS if r["id"] == want), None)
        if not ref:
            print(f"未知 ref_id={want}，可选: {[r['id'] for r in REFS]}"); continue
        print(f"\n=== {want} ===", flush=True)
        gen(ref, SEED, out)
    print("\nALL done", flush=True)


if __name__ == "__main__":
    sys.exit(main())