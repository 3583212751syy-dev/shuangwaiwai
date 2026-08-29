"""v148 区域合并 5→3 + 1.5MP 高分 + 真实感 prompt（本机 ComfyUI / SDXL）。

用户 08-29 反馈截图：「失真问题依然存在，加强元素主体处理，保留其元素真实感」。
红框框出三处糊/穿模/失真：①顶部皇冠区 ②中间骷髅+铁链穿模 ③右下糊区。

v147 (5 区域 / 1.2MP) 同根因失真连续两版未达标 → 用户授权我自主选择走「区域合并 + 高分辨率 + 真实感强化」路线。

v148 改动：
- 5 区域 → 3 区域（每区域得更多像素预算，根本性缓解糊）
  * R1 (0.10, 0.00, 0.80, 0.45): 主体老鹰 + 顶部皇冠（连续一气，皇冠和老鹰头顶自然衔接）
  * R2 (0.30, 0.40, 0.70, 0.60): 骷髅 + 铁链（大区域覆盖，让两者空间关系内生一致，治穿模）
  * R3 (0.00, 0.30, 0.30, 0.70): 火焰独立燃烧走势
- 渲染 1.2MP → 1.5MP（约 1740x870，给每元素更多像素）
- Tile CN 0.60 → 0.70：拉强细节/边缘清晰度（治穿模）
- Canny 0.25（保持轻骨架：不焊轮廓但锁定相对位置）
- denoise 0.80 → 0.78：略降保真实感
- Detail Tweaker LoRA 1.0 → 1.2：试探上限强化元素细节真实感
- 加 Detail Enhancer XL LoRA (来自 civitai "Detail Enhancer XL") +0.3：再叠一层细节精修
- 每条区域 prompt 强化「真实感」：
  * 去掉「gothic tattoo style / illustration / stylized」等 AI 风词汇
  * 加 "photo-realistic / photorealistic / lifelike / fine engraved detail / real feather texture / real bone surface texture / real metal surface texture / fine texture / crisp material definition"
- NEG 加 "AI generated, AI art, oversaturated, plastic, cartoon, simplified, smoothed over, painterly rendering, illustration style, stylized"

用法：python smoke_v148_3region.py [ref_id]   (默认 eagle_2)
输出：jobs/smoke_v148/v148_{id}.jpg
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.78                      # v147:0.80 → 0.78 略降（更高让区域提示独占、更低保真实）
IPA_WEIGHT = 0.18
LORA_DETAIL_STRENGTH = 1.2          # v147:1.0 → 1.2（Detail Tweaker 试探上限强化真实感）
LORA_DETAIL_2 = ("add-detail-xl.safetensors", 0.30)  # 再叠一个 LoRA 把细节刻死
MEGA_PIXELS = 1.5                   # v147:1.2 → 1.5 高分辨率

# 空间一致性 ControlNet
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.70                # v147:0.60 → 0.70 强化细节/边缘
REGION_STRENGTH_SCALE = 0.55

# 全局负向：锁配色 + 无字 + 抗失真 + 抗穿模 + 抗元素融合 + 抗 AI 风（v148 加强）
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, alphabet, "
    "banner, banner inscription, engraved lettering, runic text, readable text, glyphs, calligraphy, "
    "3d, photographic, painterly, illustration by child, beginner drawing, "
    "AI generated, AI art, AI artifact, oversaturated, plastic look, cartoon, "
    "simplified, simplified shapes, smoothed over, illustration style, stylized, stylized rendering, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "melted, fused, smudged, bleeding, water damaged, anatomically incorrect, "
    "extra wings, asymmetric error, garbled forms, nonsense, tiling, repeating pattern, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "elements touching each other, elements touching neighboring elements, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "crowded center, cluttered middle area, "
    "low contrast between adjacent elements, no clear black separating outline, "
    "new colors, different color palette, extra colors, color shift"
)

# 全局正向（画风基调）
GLOBAL_POS = (
    "gothic dark t-shirt graphic print on pure black background, "
    "red and orange flames, white and silver feathered eagle and bone skull, gray iron, "
    "high contrast, sharp crisp edges, no text, no letters, no words, no banner, no inscription anywhere, "
    "cohesive composition, all elements connected and spatially consistent, "
    "photorealistic tattoo print quality, fine engraved detail on every element"
)

# 衔接短语
COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

REFS = [
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "regions": [
            # ===== R1 (大区域): 主体老鹰 + 顶部皇冠（让两者自然衔接，不挤压糊化）=====
            {"x": 0.10, "y": 0.00, "w": 0.80, "h": 0.45, "strength": 1.20,
             "prompt": (
                "a bald eagle FACING THE CAMERA head-on (NOT a side profile, NOT a top-down dive), "
                "both wings spread wide symmetrically outward from a central body, "
                "wings and body anatomically connected as one continuous bird, no wing-body separation, "
                "every feather photorealistic with fine individual feather barbs visible, real feather texture, "
                "fierce open beak with sharp detailed beak ridges, piercing eyes, "
                "white and silver plumage with subtle gray shading, black background, "
                "AT THE VERY TOP CENTER above the eagle's head sits a SINGLE small gothic iron crown with 5 visible spikes, "
                "the crown SITS ALONE with a CLEAR BLACK GAP between it and the eagle's head (crown does not touch eagle), "
                "the crown is small, clean, simple, decorative, no chains, no skulls, "
                "detailed photorealistic engraved iron crown. "
                + COHESIVE + ", photorealistic tattoo print, fine engraved detail, real material texture")},
            # ===== R2 (大区域): 骷髅 + 铁链 —— 大区域覆盖让两者空间关系内生一致，根本治穿模 =====
            {"x": 0.30, "y": 0.40, "w": 0.70, "h": 0.60, "strength": 1.30,
             "prompt": (
                "ONE LARGE human skull (only one, not three) turned at a three-quarter angle, "
                "deep realistic cracks and hairline fractures across the bone surface, photorealistic bone texture, "
                "no crown, no hat, just natural cracked bone, gray bone with realistic shadow and bone dimension, "
                "the skull occupies the center-lower area, "
                "AROUND the skull and BEHIND it drapes heavy iron chain with sharp metal SPIKES bolted through the links, "
                "the chain links are bold solid outlines with photorealistic metal sheen, gray iron, "
                "the chain naturally wraps AROUND but clearly OUTSIDE the skull (chain does NOT pierce skull bone), "
                "the chain links touch the skull surface from outside, never penetrate inside, "
                "every chain link distinct and clearly outlined, "
                "sharp edges, crisp material definition. "
                + COHESIVE + ", photorealistic tattoo print, fine engraved detail, real bone and metal texture")},
            # ===== R3: 火焰独立燃烧走势 =====
            {"x": 0.00, "y": 0.30, "w": 0.28, "h": 0.70, "strength": 1.15,
             "prompt": (
                "red and orange flames rising as a DIAGONAL sweeping column flowing from lower-left toward upper-right, "
                "dynamic motion, asymmetric, no wrapping symmetry, "
                "individual flame tongues are sharp and distinct with photorealistic flame shape, "
                "fires DO NOT touch the skull, separated by black space, flames distinct from skull. "
                + COHESIVE + ", photorealistic flame detail, real fire appearance")},
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

    # IPAdapter style 锁配色（沿用）
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    # 双 LoRA 链：v148 新增第二个 LoRA 叠层
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1],
        "lora_name": LORA_DETAIL_2[0],
        "strength_model": LORA_DETAIL_STRENGTH, "strength_clip": LORA_DETAIL_STRENGTH}}

    # ===== 空间一致性双 ControlNet =====
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

    # 全局正负向
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    # 3 区域独立编码 + 内置 ConditioningSetAreaPercentage
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

    # 双 KSampler (28/24 步 高分辨率需更多)
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 28, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 24, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v148_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v148_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    import json as _json
    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v148_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {_json.dumps(j)[:1500]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {tag}: 无 prompt_id {str(j)[:400]}", flush=True); return False
    print(f"  [{tag}] pid={pid}", flush=True)
    for i in range(80):
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
                            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
                            sharp.save(out, 'JPEG', quality=95, optimize=True)
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
    out = PROJECT_ROOT / "jobs" / "smoke_v148"
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
