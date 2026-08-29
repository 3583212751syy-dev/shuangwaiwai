"""v159 — v157 + 极致精修 prompt：
- 区域 prompt 极度简短明确（不让 AI 自由发挥，每条只描述"区域里有什么"）
- NEG 列举所有见过的失真（城堡/塔/教堂/鸟结构/翅膀穿模/怪鸟等）
- 锁结构 Canny 0.85 + 锁风格 IPA 0.60（保持 v157 强锁）
- denoise 0.62（比 v157 略降，给"无区域自由发挥"留更多 base 主控）
- 区域缩小（更紧密的 mask fit）
"""
import time, requests, sys, os, json as _json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.62     # 比 v157 0.70 略降，给 base prompt 主导权
CN_STRENGTH = 0.78 # 锁结构（v155 同档）
IPA_WEIGHT = 0.55
TILE_STRENGTH = 0.50
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.5
REGION_STRENGTH_SCALE = 0.65

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"

NEG_BASE = (
    # 边框/外框
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    # 文字
    "text, letters, words, writing, typography, signature, caption, label, paragraph, alphabet, glyphs, calligraphy, "
    # 渲染风格
    "3d, painterly, illustration by child, beginner drawing, photorealistic 3d render, "
    # 模糊/糊
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    # 灰底/污染
    "gray background, gray backdrop, dark gray, light gray, ash gray, gradient gray, charcoal gray, "
    "foggy background, smoky background, hazy background, dim gray paneling, "
    "yellow background, brown background, blue background, purple background, green background, "
    "color tint in background, tinted background, washed background, off-black background, "
    "scattered composition, chaotic layout, debris, flying fragments, broken composition, "
    # 元素互穿
    "elements touching, elements adjacent, elements overlapping, elements intersecting, "
    "merged elements, melting into each other, blending into each other, "
    "crowded center, cluttered middle, "
    "chain piercing through skull, chain piercing through eagle, "
    "wings extending below upper half, eagle feathers in skull region, "
    "bird anatomy inside skull region, bird anatomy inside crown, "
    # 多余元素
    "extra bird, second eagle, multiple skulls in foreground, multiple eagles overlapping, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    # 噪点/伪影
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "garbled forms, nonsense, AI artifact, tiling, repeating pattern, "
    "plastic look, fake, synthetic, low detail, simplified, cartoonish, "
    # 配色
    "new colors, different color palette, extra colors, color shift, recolored, hue shift, "
    # 建筑/非生物结构
    "castle, tower, fortress, dome, cathedral, architecture, spire roof, "
    "turrets, battlement, keep, citadel, stronghold, palace, mansion, "
    "elaborate structure on top, gothic cathedral element, "
    # 火焰
    "flames spreading across full image, fire dominating composition, "
    "fire in center, fire covering skull, fire covering eagle, fire everywhere, "
    "inferno, wildfire, conflagration, "
    # 鸟（防止怪鸟/多鸟）
    "weird bird, abstract bird, malformed bird, twisted bird, headless bird, "
    "spread wings, open wings in upper region, two birds together, multiple birds, "
    "alien creature, mutant, chimera"
)

# ★ v159 终极精修：极简短 prompt，让 AI 只改局部细节，全局不自由发挥
GLOBAL_POS = (
    "gothic heraldic crest on PURE BLACK background, "
    "SAME layout and SAME element positions as reference, "
    "SAME black silver red-orange color palette, "
    "ONLY mutate the internal TEXTURE and DETAIL of each existing element, "
    "do not add new elements, do not remove existing elements, "
    "each element surrounded by a CLEAR solid-black separating outline silhouette, "
    "elements NEVER touch or bleed into neighbors, "
    "PIN-SHARP precision and CLEAN READABLE forms, "
    "ULTRA sharp crisp high-contrast edges, "
    "professional high-end commercial apparel graphic, masterpiece, ultra detailed"
)

# 简短明确的区域 prompt：每条只描述该区域里有什么 + 改什么 + 不要什么
COHESIVE = (
    "ONLY contains what's in this region, NO other elements or anatomy, "
    "KEEP position and size, SAME color palette, photorealistic fine detail"
)

REFS = [
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "regions": [
            # 主鹰：仅描述上方鹰区，禁任何下方穿模
            {"x": 0.20, "y": 0.05, "w": 0.60, "h": 0.40, "strength": 1.10,
             "prompt": (
                 "ONLY the single bald eagle: render with MORE aggressive, denser, sharper silver-white feather detail, "
                 "fierce expression, deep eye socket, prominent curved talons, "
                 "wings FULLY CONTAINED within this region (y top 0.05 to y 0.45), "
                 "NEVER extend wings or feathers below y=0.45, "
                 "NO bird structures, wings, or feathers below this region. "
                 + COHESIVE
             )},
            # 中央骷髅王冠：明确禁鸟结构/禁建筑
            {"x": 0.22, "y": 0.50, "w": 0.56, "h": 0.40, "strength": 1.30,
             "prompt": (
                 "ONLY human skull with crown: stack of classic skulls with deep dark cracks on cranium, "
                 "more weathered bone texture, missing teeth, "
                 "on top sits a SMALL simple iron band crown with 2-3 short blunt spikes, "
                 "the crown is a small CROWN clearly visible, "
                 "NOT castle, NOT tower, NOT fortress, NOT building, NOT architecture, NOT cathedral, "
                 "NO wings or feathers or eagle anatomy in this region, "
                 "NO bird structure inside the skull or crown, "
                 + COHESIVE
             )},
            # 左铁链
            {"x": 0.00, "y": 0.32, "w": 0.12, "h": 0.55, "strength": 1.20,
             "prompt": (
                 "ONLY thick iron chain links along LEFT EDGE with ADDED sharp metal spikes and barbs protruding, "
                 "heavy industrial gothic metal look, realistic metallic surface, "
                 "NO eagle, NO bird, NO feathers, NO wings in this region. "
                 + COHESIVE
             )},
            # 右铁链
            {"x": 0.88, "y": 0.32, "w": 0.12, "h": 0.55, "strength": 1.20,
             "prompt": (
                 "ONLY thick iron chain links along RIGHT EDGE with ADDED sharp metal spikes and barbs protruding, "
                 "matching gothic industrial metallic look, "
                 "NO eagle, NO bird, NO feathers, NO wings in this region. "
                 + COHESIVE
             )},
            # 顶部小乌（如有）：明确单只紧凑乌鸦
            {"x": 0.36, "y": 0.00, "w": 0.28, "h": 0.10, "strength": 0.95,
             "prompt": (
                 "ONLY one small compact black raven with FOLDED wings (NOT spread, NOT multiple), "
                 "clearly recognizable single bird silhouette at top center, "
                 "NOT abstract, NOT malformed, NOT two birds, NOT spread wings. "
                 + COHESIVE
             )},
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
        "image": ["3", 0], "low_threshold": 0.08, "high_threshold": 0.24, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": CN_STRENGTH}}

    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": TILE_STRENGTH}}

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
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
        "latent_image": ["4", 0], "seed": seed, "steps": 26, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.15}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v159_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v159_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v159_{int(time.time())}"}, timeout=15)
    j = r.json()
    if 'prompt_id' not in j:
        print(f"!! prompt failed: {j}"); return False
    pid = j['prompt_id']; print(f'  prompt_id={pid}')
    t0 = time.time()
    while time.time() - t0 < 360:
        time.sleep(2)
        r2 = requests.get(f"{COMFYUI}/history/{pid}", timeout=10)
        d = r2.json()
        h = d.get(pid)
        if not h: continue
        outs = h.get('outputs', {})
        if outs:
            for nid, outp in outs.items():
                if 'images' in outp:
                    for img in outp['images']:
                        nm = img['filename']
                        url = f"{COMFYUI}/view?filename={nm}&subfolder={img.get('subfolder','')}&type=output"
                        local = out_base / nm
                        try:
                            import urllib.request
                            urllib.request.urlretrieve(url, str(local))
                            print(f"  saved {local} ({local.stat().st_size/1024/1024:.1f}MB)", flush=True)
                        except Exception as e:
                            print(f'  ! download fail: {e}')
            return True
        status = h.get('status', {})
        if status.get('errored'):
            print(f"!! error: {status}"); return False
    print('!! TIMEOUT')
    return False


if __name__ == '__main__':
    out_base = PROJECT_ROOT / "jobs" / "smoke_v159"
    out_base.mkdir(parents=True, exist_ok=True)
    keys = sys.argv[1:] if len(sys.argv) > 1 else [r['id'] for r in REFS]
    for ref in REFS:
        if ref['id'] not in keys: continue
        print(f"\n=== v159 run {ref['id']} (seed={SEED}) ===", flush=True)
        gen(ref, SEED, out_base)
