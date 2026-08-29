"""v152 紧凑徽章式裂变 + 强抗灰色（按 B 方案）。

方向：用户要求「按原图风格 + 减少失真」。
原图是紧凑徽章式（单鹰 + 单骷髅下 + 徽章插画 + 边框铁链 + 火焰背景）。
v151 把徽章拆散了 + 背景发灰。
v152 重画区域为徽章式紧凑构图，强约束纯黑底，强 NEG 禁灰色禁堆砌。

v152 vs v151 关键改动：
1. DENOISE 0.78→0.66：让元素听话，少堆砌
2. IPA_WEIGHT 0.50→0.60：更锁元素真实身份/配色
3. CANNY_STRENGTH 0.22→0.32：锁徽章构图
4. TILE_STRENGTH 0.75→0.55：不再填灰暗部
5. 区域重画成徽章式（取消 v151 的「3-4 骷髅堆」「3 段穿插铁链」等堆砌设定）
6. NEG 强加禁灰色禁堆砌禁穿模
7. POS 强加 pure black background + 紧凑徽章

输出：jobs/smoke_v152/v152_{id}.jpg

用法：python smoke_v152_badge_compact.py [ref_id]   (默认 eagle_2)
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.66                       # v151:0.78 → 0.66 元素听话少堆砌
IPA_WEIGHT = 0.60                    # v151:0.50 → 0.60 强锁元素真实身份/配色
LORA_DETAIL = 1.0                    # 抗失真
MEGA_PIXELS = 1.5

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
CANNY_STRENGTH = 0.32                # v151:0.22 → 0.32 锁徽章构图
TILE_STRENGTH = 0.55                 # v151:0.75 → 0.55 不再填灰暗部
REGION_STRENGTH_SCALE = 0.55         # v151:0.65 → 0.55 让全局不让区域喧宾夺主

# 强 NEG：核心是「禁灰色背景 + 禁元素堆砌 + 禁穿模 + 禁融合」
NEG_BASE = (
    # 边框/排版
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    # 文字（不让 AI 烧字）
    "text, letters, words, writing, typography, signature, caption, label, paragraph, alphabet, glyphs, calligraphy, "
    # 风格错误
    "3d, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    # ★ 禁灰（用户投诉：v151 背景发灰）
    "gray background, gray backdrop, dark gray, light gray, ash gray, gradient gray, charcoal gray, "
    "foggy background, smoky background, hazy background, dim gray paneling, "
    "yellow background, brown background, blue background, purple background, green background, "
    # ★ 禁堆砌（v151 元素乱堆）
    "scattered composition, chaotic layout, debris, flying fragments, broken composition, "
    "elements touching, elements adjacent, elements overlapping, elements intersecting, "
    "merged elements, melting into each other, blending into each other, "
    "crowded center, cluttered middle, "
    "multiple skulls in foreground, multiple eagles overlapping, "
    "chain piercing through skull, chain piercing through eagle, "
    # 主体过小
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    # 失真/低质
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "extra wings, asymmetric error, garbled forms, nonsense, AI artifact, tiling, repeating pattern, "
    # 假感
    "plastic look, fake, synthetic, low detail, simplified, cartoonish, "
    "new colors, different color palette, extra colors, color shift"
)

# 强 POS：紧凑徽章 + 纯黑底
GLOBAL_POS = (
    "compact gothic badge illustration, tight framed centered composition, "
    "PURE JET BLACK BACKGROUND, pure deep black void backdrop, solid black ground, "
    "red and orange flames along the LEFT and RIGHT sides as a backdrop arc behind the eagle and skull, "
    "white and silver eagle, gray skull, gray iron chains, gold-silver crown with spikes, "
    "high contrast, sharp edges, "
    "EACH ELEMENT CLEARLY SEPARATED BY BLACK SPACE, clear black gap between every pair of elements, "
    "balanced symmetrical badge layout, "
    "photorealistic detail on every element, realistic textures, "
    "no text, no letters, no words, no banner, no inscription anywhere, "
    "emblem layout: main eagle on top, single crowned skull below center"
)

# 衔接短语：每条区域 prompt 都拼上，强调清晰分离
COHESIVE = (
    "isolated against pure black background, CLEAR BLACK SPACE on all four sides of this element, "
    "does NOT touch any other element, no overlap, no merge, no clipping through, "
    "does NOT pierce through adjacent objects, no intersecting geometry, "
    "photorealistic, fine realistic detail, real surface texture"
)

REFS = [
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "regions": [
            # 主鹰：徽章中央上半，正面，单只
            {"x": 0.18, "y": 0.10, "w": 0.64, "h": 0.42, "strength": 1.20,
             "prompt": ("ONE single photorealistic bald eagle FACING THE CAMERA head-on (NOT a side profile), "
                        "both wings spread symmetrically outward from a central body, "
                        "wings and body anatomically connected as one continuous bird, "
                        "fierce open beak, real white and silver feathers with fine realistic texture, "
                        "CENTERED upper area of the badge composition, "
                        "isolated against pure black background. "
                        + COHESIVE)},
            # 顶部小元素：把后面小鹰换成黑乌鸦（明显种类改变）
            {"x": 0.34, "y": 0.00, "w": 0.32, "h": 0.10, "strength": 1.00,
             "prompt": ("ONE single small photorealistic black raven (Corvus corax) with folded wings, "
                        "fully black plumage with subtle blue sheen, NO bald eagle, NO white feathers, "
                        "perched alone at the top center, small fine bird with real plumage detail, "
                        "isolated from the eagle below by a clear black gap. "
                        + COHESIVE)},
            # 主骷髅+王冠：徽章下方中央，单颅，单冠
            {"x": 0.22, "y": 0.58, "w": 0.56, "h": 0.42, "strength": 1.30,
             "prompt": ("ONE SINGLE photorealistic human skull at the bottom center of the badge, "
                        "facing forward, slightly tilted up, "
                        "on top of the skull sits ONE matching gothic iron crown with 5 sharp spikes, "
                        "deep realistic cracks across the skullcap, "
                        "gray bone with realistic shadow and fine bone pore texture, "
                        "CLEAR BLACK GAP above the skull separating it from the eagle area, "
                        "photorealistic bones, real osteological detail. "
                        + COHESIVE)},
            # 左铁链：徽章左边垂坠
            {"x": 0.00, "y": 0.30, "w": 0.12, "h": 0.60, "strength": 1.20,
             "prompt": ("photorealistic heavy gothic iron chain hanging down along the LEFT EDGE, "
                        "vertical column of solid interlocking iron links with sharp metal spikes, "
                        "industrial gothic, gray metal with realistic metallic surface and highlights, "
                        "chain stays at the very left edge, DOES NOT extend toward center, "
                        "DOES NOT touch skull, eagle or flames, ISOLATED along the left edge. "
                        + COHESIVE)},
            # 右铁链：徽章右边垂坠
            {"x": 0.88, "y": 0.30, "w": 0.12, "h": 0.60, "strength": 1.20,
             "prompt": ("photorealistic heavy gothic iron chain hanging down along the RIGHT EDGE, "
                        "vertical column of solid interlocking iron links with sharp metal spikes, "
                        "industrial gothic, gray metal with realistic metallic surface and highlights, "
                        "chain stays at the very right edge, DOES NOT extend toward center, "
                        "DOES NOT touch skull, eagle or flames, ISOLATED along the right edge. "
                        + COHESIVE)},
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

    # IPAdapter style 锁配色+元素身份
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

    # === 双 ControlNet ===
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

    # 双 KSampler
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 26, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v152_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v152_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    import json as _json
    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v152_{int(time.time())}"}, timeout=15)
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
                    print(f"  [{tag}] COMFY错误 {rec['status'].get('error')}", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] 轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT 跳过重试", flush=True); return False


def main():
    wants = sys.argv[1:] if len(sys.argv) > 1 else ["eagle_2"]
    out = PROJECT_ROOT / "jobs" / "smoke_v152"
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
