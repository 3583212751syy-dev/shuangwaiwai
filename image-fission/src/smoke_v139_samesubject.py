"""v139 同主体裂变（本机 ComfyUI / SDXL）：img2img(VAEEncode 锁构图) + IPAdapter style 0.55(锁配色)
+ 双 KSampler(denoise 改内部细节)。新规：同类型主体不变、只改内部细节、配色/构图/结构保留、不加新色。

用法：python smoke_v139_samesubject.py [ref_id]   (默认 eagle_2)
输出：jobs/smoke_v139/v139_{id}.jpg
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

# 同主体 prompt：subject 严格描述原图“同一类”元素（鹰还是鹰/棕榈还是棕榈/蝴蝶还是蝴蝶），
# 不替换为别的主体；palette 锁死原图配色，pos/neg 都强调“不加新色、不换主体”。
REFS = [
    {"id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg", "weight": 0.42,
     "subject": ("a bald eagle with spread wings and talons, surrounded by red flames, "
                 "three human skulls below, draped iron chains, gothic heraldic crest"),
     "palette": ("gothic tattoo illustration, pure black background, white and silver eagle, "
                 "red and orange flames, gray iron chains")},
    {"id": "camo_4", "ref_img": "pinterest_camo_4.jpg", "weight": 0.42,
     "subject": "palm trees and tropical fronds arranged in a military camouflage pattern",
     "palette": "green brown black camouflage, matte, no text"},
    {"id": "illust_1", "ref_img": "pinterest_illust_1.jpg", "weight": 0.42,
     "subject": ("ornate black and white scrolling acanthus foliage, flowers and decorative "
                 "leaves, symmetrical border frame"),
     "palette": "monochrome ink line illustration, pure black background, white filigree, no color"},
    {"id": "denim_3", "ref_img": "pinterest_denim_3.jpg", "weight": 0.42,
     "subject": ("a butterfly perched on a distressed X-shaped denim patch with fabric wear "
                 "and stitch holes"),
     "palette": "vintage faded denim texture, indigo blue and white only"},
    {"id": "skull_5", "ref_img": "pinterest_skull_5.jpg", "weight": 0.42,
     "subject": ("a grim human skull with black eye patch, a red snake coiled around it, "
                 "red feathered wings, a red rose"),
     "palette": "gothic tattoo illustration, pure black background, red and white palette"},
    {"id": "metal_6", "ref_img": "pinterest_metal_6.jpg", "weight": 0.42,
     "subject": ("a bald eagle perched on a cracked human skull with horn-like spikes, "
                 "radiating lightning bolts"),
     "palette": "death metal illustration, pure black background, white and bronze palette"},
]

POS_TAIL = (
    "same composition and layout as the reference image, "
    "redraw ONLY the internal details and micro-patterns of these same subjects, "
    "keep the exact same color palette as the reference, do not change subject types, "
    "do not introduce new subjects, do not add new colors, "
    "masterpiece, best quality, ultra detailed, "
    "anatomically correct and physically coherent subjects "
    "(real-looking animal anatomy, real-looking metal chains, real-looking fire, real-looking bone), "
    "intricate craftsmanship and fine engraved details on every element, "
    "ULTRA sharp crisp high-contrast edges, "
    "every element surrounded by a CLEAR solid-black separating outline silhouette, "
    "elements NEVER bleed into neighbors (fire does NOT melt into eagle, "
    "skull does NOT blend into background, chains do NOT dissolve into feathers), "
    "each major element occupies 75 to 90 percent of its allocated area "
    "(large dramatic macroperspective), "
    "bold graphic t-shirt print composition, full bleed edge-to-edge, "
    "no halftone, no noise, no grain, no smudge, no watercolor, no soft airbrush"
)
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, "
    "3d, photographic, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "new colors, different color palette, extra colors, color shift, changed subject, "
    "different subject type, new subject, replaced main subject"
)
SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.58


def build(ref, seed):
    pos = (f"t-shirt graphic design, {ref['subject']}, {ref['palette']}, {POS_TAIL}")
    neg = NEG_BASE
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
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
    # ControlNet Canny：锁元素轮廓，防止 IPAdapter 把相邻元素融一起
    g["16"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "controlnet-canny-sdxl-1.0.safetensors"}}
    g["17"] = {"class_type": "Canny", "inputs": {
        "image": ["2", 0], "low_threshold": 0.06, "high_threshold": 0.20, "resolution": 1024}}
    g["18"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["8", 0],
        "control_net": ["16", 0], "image": ["17", 0],
        "strength": 0.40}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 32, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 28, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0],
                       "filename_prefix": f"v139_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v139_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": build(ref, seed),
                    "client_id": f"v139_{int(time.time())}"}, timeout=15)
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
    for i in range(60):  # 300s 硬超时
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
                        # ====== 后期 PIL USM 锐化（补 SDXL 4x 超分后的边缘软化） ======
                        try:
                            from PIL import Image, ImageFilter
                            im = Image.open(out).convert('RGB')
                            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
                            sharp.save(out, 'JPEG', quality=95, optimize=True)
                            import os
                            os.replace(out, out)  # 刷新大小
                            print(f"  [{tag}] USM锐化 {out.stat().st_size/1024/1024:.1f}MB", flush=True)
                        except Exception as e:
                            print(f"  [{tag}] USM失败 原图保留 {e}", flush=True)
                        print(f"  [{tag}] OK {out.stat().st_size/1024/1024:.1f}MB", flush=True); return True
                elif rec.get("status", {}).get("error"):
                    print(f"  [{tag}] COMFY错误 {rec['status'].get('error')}", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] 轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT 跳过重试", flush=True); return False


def main():
    wants = sys.argv[1:] if len(sys.argv) > 1 else ["eagle_2"]
    out = PROJECT_ROOT / "jobs" / "smoke_v139"
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
