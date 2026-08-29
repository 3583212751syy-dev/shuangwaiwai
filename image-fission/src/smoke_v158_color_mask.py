# -*- coding: utf-8 -*-
"""v158 — Regional Mask Prompt（颜色 mask 架构）drop-in 替代 ConditioningSetArea+RegionalListCombine

基于 v155 验证过的 IPAdapter+ControlNet+LoRA+超分框架，
把 5 个 ConditioningSetArea + RegionalListCombine 替换成单一 RegionalMaskPromptEncode 节点。
颜色 mask 程序化生成（4 区域：主鹰 / 骷髅王冠 / 左铁链 / 右铁链）。

继承 v155 的强 NEG/防失真约束（GLOBAL_POS 防失真重写）但叠加上 v156 的裂变强度。
"""
import time, sys, os, json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw
import urllib.request, urllib.error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700501
CKPT = "ProteusV0.4.safetensors"

# Lock controls
CN_STRENGTH = 0.78
IPA_WEIGHT = 0.50
TILE_STRENGTH = 0.50
DENOISE = 0.62
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.5

# Regional Mask Prompt params
REGION_STRENGTH = 0.95
BASE_STRENGTH = 0.70
MASK_FEATHER = 8
MASK_TOLERANCE = 24
MIN_MASK_ALPHA = 0.05

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"

# Palette (Regional Prompter deterministic HSV)
REGION_COLORS = ["#804040", "#408080", "#608040", "#604080"]  # r1, r2, r3, r4

# === Prompts ===
BASE_PROMPT = (
    "gothic heraldic crest t-shirt graphic on pure black background, "
    "SAME composition, layout, element positions, poses and proportions as the reference image, "
    "SAME color palette and art style (black background, white-silver eagle, "
    "gray skull, gray iron chains, red-orange flames, gothic iron crown), "
    "every element rendered with PIN-SHARP precision and CLEAN READABLE FORMS, "
    "professional high-end commercial apparel graphic, masterpiece best quality ultra detailed, "
    "ULTRA sharp crisp high-contrast edges, "
    "every element surrounded by a CLEAR solid-black separating outline silhouette, "
    "elements NEVER bleed into neighbors, "
    "no halftone no noise no grain no smudge"
)

NEG = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "gray background, gray backdrop, dark gray, light gray, ash gray, gradient gray, charcoal gray, "
    "scattered composition, chaotic layout, debris, flying fragments, broken composition, "
    "elements touching, elements adjacent, elements overlapping, elements intersecting, "
    "merged elements, melting into each other, blending into each other, "
    "chain piercing through skull, chain piercing through eagle, "
    "extra bird, second eagle, multiple skulls in foreground, multiple eagles overlapping, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, "
    "garbled forms, nonsense, AI artifact, plastic look, fake, synthetic, low detail, "
    "new colors, different color palette, extra colors, color shift, recolored, hue shift, "
    "castle, tower, fortress, dome, cathedral, architecture, spire roof, "
    "turrets, battlement, keep, citadel, stronghold, palace, mansion, "
    "elaborate structure on top, gothic cathedral element, "
    "flames spreading across full image, fire dominating composition, "
    "inferno, wildfire, conflagration, "
    "weird bird, abstract bird, malformed bird, twisted bird, headless bird, "
    "alien creature, mutant, chimera"
)

# 4 个区域 prompt
R1_EAGLE = (
    "the bald eagle at center-top: render with MORE aggressive, denser, sharper silver-white feather detail, "
    "fierce expression, deep carved eye socket, prominent curved talons, "
    "wings spread and extending outward, "
    "KEEP position, pose and size, SAME color palette, photorealistic fine detail"
)
R2_SKULL_CROWN = (
    "the human skull stack at bottom-center: render with DEEPER and MORE NUMEROUS cracks and fractures "
    "across the cranium, more weathered bone texture, missing teeth, "
    "on top sits a small simple GOTHIC IRON BAND CROWN with 3 short blunt spikes, "
    "the crown is a CROWN not a building, NOT castle NOT tower NOT architecture NOT fortress, "
    "KEEP position and size, SAME color palette (gray skull + dark iron crown)"
)
R3_CHAIN_LEFT = (
    "the thick iron chain along the LEFT EDGE: render each link with ADDED sharp metal spikes and barbs, "
    "heavier industrial gothic look, realistic metallic surface with reflections, "
    "KEEP position along left edge and length, SAME color (dark gray iron)"
)
R4_CHAIN_RIGHT = (
    "the thick iron chain along the RIGHT EDGE: render each link with ADDED sharp metal spikes and barbs, "
    "matching gothic industrial style, realistic metallic surface, "
    "KEEP position along right edge and length, SAME color (dark gray iron)"
)

REF_ID = "eagle_2"
REF_IMG = "pinterest_eagle_2.jpg"

# 4 区域几何（按 eagle_2 原图实际布局 y=0~1）
REGIONS = [
    # r1 主鹰区域：占据上半部
    {"x_min": 0.05, "x_max": 0.95, "y_min": 0.04, "y_max": 0.48},
    # r2 骷髅王冠：占据下半部中央
    {"x_min": 0.15, "x_max": 0.85, "y_min": 0.45, "y_max": 0.80},
    # r3 左铁链：左边竖条
    {"x_min": 0.0,  "x_max": 0.18, "y_min": 0.30, "y_max": 0.85},
    # r4 右铁链：右边竖条
    {"x_min": 0.82, "x_max": 1.0,  "y_min": 0.30, "y_max": 0.85},
]


def hex_to_rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))


def make_mask_png(W, H):
    img = Image.new('RGB', (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, r in enumerate(REGIONS):
        c = hex_to_rgb(REGION_COLORS[i])
        x0 = int(r['x_min'] * W); x1 = int(r['x_max'] * W)
        y0 = int(r['y_min'] * H); y1 = int(r['y_max'] * H)
        draw.rectangle([x0, y0, x1, y1], fill=c)
    out_path = 'E:/Desktop/双接口/image-fission/jobs/smoke_v158/_tmp_mask.png'
    img.save(out_path, 'PNG')
    return out_path


def upload_to_input(local_path, comfy_input='E:/Desktop/双接口/image-fission/ComfyUI/input'):
    name = os.path.basename(local_path)
    target = os.path.join(comfy_input, name)
    shutil.copy(local_path, target)
    return name


def build(ref_id=REF_ID, seed=SEED):
    ref_local = f"E:/Desktop/图裂变测试图/{REF_IMG}"
    W_ref, H_ref = Image.open(ref_local).size
    target_mp = MEGA_PIXELS * 1_000_000
    scale = (target_mp / (W_ref * H_ref)) ** 0.5
    W = max(64, int(W_ref * scale / 64) * 64)
    H = max(64, int(H_ref * scale / 64) * 64)
    print(f'  size: {W}x{H}')

    # 生成颜色 mask (匹配 WxH)
    mask_local = make_mask_png(W, H)
    ref_name = upload_to_input(ref_local)
    mask_name = upload_to_input(mask_local)

    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref_name}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}

    # IPAdapter (load+apply)
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["2", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}

    # LoRA on top of IPAdapter (clip from CheckpointLoaderSimple[1])
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1],
        "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}

    # Canny preprocessor + ControlNet (链到 conditioning)
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["2", 0], "low_threshold": 0.08, "high_threshold": 0.24, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["40", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": CN_STRENGTH}}

    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["2", 0], "strength": TILE_STRENGTH}}

    # CLIP text encode for base/neg
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG}}

    # ★ v158 核心：Regional Mask Prompt Encode（替代 5x ConditioningSetArea + RegionalListCombine）
    # 输入是 mask_image (RGB 图，已在节点 3) + 区域 prompt 字符串
    # 输出：CONDITIONING（综合）+ 8 个 MASK + info STRING
    g["40"] = {"class_type": "RegionalMaskPromptEncode", "inputs": {
        "clip": ["7", 1],
        "mask_image": ["3", 0],
        "base_prompt": BASE_PROMPT,
        "region_1_prompt": R1_EAGLE,
        "region_2_prompt": R2_SKULL_CROWN,
        "region_3_prompt": R3_CHAIN_LEFT,
        "region_4_prompt": R4_CHAIN_RIGHT,
        "region_count": 4,
        "region_strength": REGION_STRENGTH,
        "base_strength": BASE_STRENGTH,
        "mask_tolerance": MASK_TOLERANCE,
        "mask_feather": MASK_FEATHER,
        "min_mask_alpha": MIN_MASK_ALPHA,
        "set_cond_area": "default"
    }}

    # Latent: 原图 encode
    g["50"] = {"class_type": "ImageScale", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos",
        "width": W, "height": H, "crop": "center"}}
    g["51"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["50", 0], "vae": ["1", 2]}}

    # 双 KSampler（v155 那套）
    g["60"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["40", 0], "negative": ["ng", 0],
        "latent_image": ["51", 0], "seed": seed, "steps": 26, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["61"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["40", 0], "negative": ["ng", 0],
        "latent_image": ["60", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.15}}

    g["62"] = {"class_type": "VAEDecode", "inputs": {"samples": ["61", 0], "vae": ["1", 2]}}
    g["63"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["64"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["63", 0], "image": ["62", 0]}}
    g["65"] = {"class_type": "SaveImage", "inputs": {"images": ["64", 0], "filename_prefix": f"v158_{REF_ID}"}}
    return g


def gen(seed=SEED):
    g = build(seed=seed)
    payload = {"prompt": g, "client_id": f"v158_{int(time.time())}"}
    req = urllib.request.Request(f"{COMFYUI}/prompt", data=json.dumps(payload).encode())
    req.add_header('Content-Type', 'application/json')
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        r = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"!! prompt submit HTTP {e.code}")
        print(body[:2500])
        return False
    if 'prompt_id' not in r:
        print(f"!! no prompt_id: {r}")
        return False
    pid = r['prompt_id']
    print(f'  prompt_id={pid}')

    out_dir = 'E:/Desktop/双接口/image-fission/jobs/smoke_v158'
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    while time.time() - t0 < 360:
        try:
            req2 = urllib.request.Request(f"{COMFYUI}/history/{pid}")
            r2 = urllib.request.urlopen(req2, timeout=10)
            d = json.loads(r2.read().decode())
        except Exception:
            time.sleep(2); continue
        h = d.get(pid)
        if h:
            outs = h.get('outputs', {})
            if outs:
                saved = False
                for nid, out in outs.items():
                    if 'images' in out:
                        for img in out['images']:
                            nm = img['filename']
                            sub = img.get('subfolder', '')
                            url = f"{COMFYUI}/view?filename={nm}&subfolder={sub}&type=output"
                            local = os.path.join(out_dir, nm)
                            try:
                                urllib.request.urlretrieve(url, local)
                                print(f'  saved {local} ({os.path.getsize(local)/1024/1024:.1f}MB)')
                                saved = True
                            except Exception as e:
                                print(f'  ! download fail {nm}: {e}')
                if saved:
                    return True
            status = h.get('status', {})
            if status.get('errored') or status.get('completed_str', '').startswith('error'):
                err = status.get('errored') or status.get('completed_str')
                print(f'!! job error: {err}')
                if 'exception_message' in status:
                    print(f"   msg: {status['exception_message'][:500]}")
                return False
        time.sleep(2)
    print('!! TIMEOUT after 360s')
    return False


if __name__ == '__main__':
    print(f'=== v158 run {REF_ID} (seed={SEED}) ===')
    gen()
