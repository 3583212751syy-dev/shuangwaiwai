"""v149 忠实原图设计（本机 ComfyUI / SDXL）：v145→v148 区域提示逐元素强改连续三版失真后，
用户 08-29 指示"按照原图设计就好"。本版回到"忠实 + 清洁重渲染"路线：
- 强 Canny (0.85) 锁死原图构图骨架（不再焊死，但保留所有元素在原位）
- 高 IP-Adapter style (0.60) 锁配色画风
- 低 denoise (0.45) 只做"清洁+锐化+去毛边+强化细节"，不重排元素
- 1.5MP 高分辨率 + Detail Tweaker 1.0 + 4x NMKD-Siax 超分 + USM 后期
- prompt 直接描述原图（保留原构图/元素/姿态），要求"keep the exact same composition"
- 不使用任何区域提示（区域提示是 v145-v148 失真根因）

用法：python smoke_v149_faithful.py [ref_id]   (默认 eagle_2)
输出：jobs/smoke_v149/v149_{id}.jpg
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

# v149 忠实调参（用户要求"按原图设计"，反过来强锁一切）：
# - CN 0.50→0.85：强锁原图骨架（元素位置/相对关系/比例都不动）
# - IPA 0.32→0.60：强锁配色画风
# - denoise 0.78→0.45：低重绘，只清洁锐化
# - Detail Tweaker 1.0 + 1.5MP + 4x 超分 + USM：抗失真、提清晰
SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.45
CN_STRENGTH = 0.85
IPA_WEIGHT = 0.60
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.5

# 忠实原图设计：subject 字段就是原图本身的元素描述
REFS = [
    {"id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg", "weight": IPA_WEIGHT,
     "subject": ("a gothic heraldic crest on pure black background, "
                 "a large bald eagle with spread silver-white wings and sharp yellow beak at the center, "
                 "a small bald eagle flying just above the main eagle, "
                 "three human skulls at the bottom (one large central skull flanked by two side skulls), "
                 "heavy iron chains draped on both the left and right sides, "
                 "red and orange flames wrapping around the wings and behind the skulls, "
                 "a black heraldic shield with an emblem at the center between the eagle and the skulls, "
                 "gothic dark tattoo illustration aesthetic, same composition and layout as the reference")},
    {"id": "camo_4", "ref_img": "pinterest_camo_4.jpg", "weight": IPA_WEIGHT,
     "subject": ("palm trees and tropical fronds arranged in a military camouflage pattern, "
                 "green brown black matte camouflage with palm leaves, "
                 "same composition and layout as the reference")},
    {"id": "illust_1", "ref_img": "pinterest_illust_1.jpg", "weight": IPA_WEIGHT,
     "subject": ("ornate black and white scrolling acanthus foliage with flowers and decorative leaves, "
                 "symmetrical border frame, monochrome ink line illustration, pure black background, white filigree, "
                 "same composition and layout as the reference")},
    {"id": "denim_3", "ref_img": "pinterest_denim_3.jpg", "weight": IPA_WEIGHT,
     "subject": ("a butterfly perched on a distressed X-shaped denim patch with fabric wear and stitch holes, "
                 "vintage faded denim texture, indigo blue and white, "
                 "same composition and layout as the reference")},
    {"id": "skull_5", "ref_img": "pinterest_skull_5.jpg", "weight": IPA_WEIGHT,
     "subject": ("a grim human skull with black eye patch, a red snake coiled around it, "
                 "red feathered wings, a red rose, gothic tattoo illustration, pure black background, red and white, "
                 "same composition and layout as the reference")},
    {"id": "metal_6", "ref_img": "pinterest_metal_6.jpg", "weight": IPA_WEIGHT,
     "subject": ("a bald eagle perched on a cracked human skull with horn-like spikes, "
                 "radiating lightning bolts, death metal illustration, pure black background, white and bronze, "
                 "same composition and layout as the reference")},
]

POS_TAIL = (
    "FAITHFUL clean re-render of the reference design, "
    "KEEP the exact same composition, layout, element positions, poses, proportions, and element category "
    "as the reference image, do NOT rearrange, recompose, or change the design, "
    "only clean up noise, sharpen edges, enhance fine detail, and remove artifacts, "
    "every element rendered with PIN-SHARP precision and CLEAN READABLE FORMS, "
    "professional high-end commercial apparel graphic, "
    "masterpiece best quality ultra detailed, "
    "anatomically correct and physically coherent subjects, "
    "intricate craftsmanship and fine engraved details on every element, "
    "ULTRA sharp crisp high-contrast edges, "
    "every element surrounded by a CLEAR solid-black separating outline silhouette, "
    "elements NEVER bleed into neighbors, "
    "bold graphic t-shirt print composition full bleed edge-to-edge, "
    "no halftone no noise no grain no smudge no watercolor no soft airbrush"
)
NEG_BASE = (
    "rearranged composition, new layout, different angle, different pose, changed proportions, "
    "added element, removed element, extra subject, "
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "readable text, readable letters, readable words, readable alphabet, readable banner inscription, "
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
    "new colors, different color palette, extra colors, color shift"
)


def build(ref, seed):
    pos = f"t-shirt graphic design, {ref['subject']}, {POS_TAIL}"
    neg = NEG_BASE
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": MEGA_PIXELS, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": ref["weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    # Detail Tweaker XL LoRA
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1],
        "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": neg}}
    # Canny 强锁：0.85 锁死原图骨架（元素位置/比例/相对关系全不动）
    g["16"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "controlnet-canny-sdxl-1.0.fp16.safetensors"}}
    g["17"] = {"class_type": "Canny", "inputs": {
        "image": ["2", 0], "low_threshold": 0.08, "high_threshold": 0.24, "resolution": 1024}}
    g["18"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["8", 0],
        "control_net": ["16", 0], "image": ["17", 0],
        "strength": CN_STRENGTH}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 28, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    # 第二次 KSampler 微细化：denoise 0.15，专注修细节不动构图
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 22, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.15}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0],
                       "filename_prefix": f"v149_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v149_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": build(ref, seed),
                    "client_id": f"v149_{int(time.time())}"}, timeout=15)
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
    for i in range(60):
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                rec = h[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
                    if imgs:
                        url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                        data = requests.get(url, timeout=60).content
                        out.write_bytes(data)
                        # USM 锐化
                        from PIL import Image, ImageFilter
                        im = Image.open(out).convert('RGB')
                        sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
                        sharp.save(out, 'JPEG', quality=95, optimize=True)
                        print(f"  [{tag}] OK {out.stat().st_size/1024/1024:.1f}MB", flush=True); return True
                elif rec.get("status", {}).get("error"):
                    print(f"  [{tag}] COMFY错误 {rec['status'].get('error')}", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] 轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT", flush=True); return False


def main():
    wants = sys.argv[1:] if len(sys.argv) > 1 else ["eagle_2"]
    out = PROJECT_ROOT / "jobs" / "smoke_v149"
    out.mkdir(parents=True, exist_ok=True)
    for want in wants:
        ref = next((r for r in REFS if r["id"] == want), None)
        if not ref:
            print(f"未知 ref_id={want}，可选: {[r['id'] for r in REFS]}"); continue
        print(f"--- {want} ---", flush=True)
        gen(ref, SEED, out)
    print("ALL done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
