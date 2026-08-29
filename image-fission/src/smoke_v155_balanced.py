"""v155 平衡版：v153 的裂变强度 + v154 的防失真约束（不砍裂变本身）。

v154 过度修正：为了修失真，把三处裂变点一起砍了 → 用户反馈"裂变效果又没了"。
v155 收手：裂变加回来，但用 v154 的 NEG/约束防住已知的失真类型（城堡/塔/火焰铺满/怪鸟），不禁止裂变本身。

相对 v154 的恢复点：
1. 恢复顶部小乌区域 → 换物种成清晰乌鸦（非抽象），保留"物种裂变"
2. 中央王冠恢复 gothic 5 尖刺（裂变视觉点），但加 "NOT a castle/tower/architecture" 防建筑感
3. 全局火焰保留"thin side arc"不铺满，但加 "MORE energetic, varied, taller flame tongues than reference" 保留走势裂变

锁参数继承 v153/v154：Canny 0.85 / IPA 0.60 / denoise 0.55 / Tile 0.50 / REGION_STRENGTH_SCALE 0.55
输出：jobs/smoke_v155/v155_{id}.jpg
"""
import time, requests, sys, os, json as _json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.55
CN_STRENGTH = 0.85
IPA_WEIGHT = 0.60
TILE_STRENGTH = 0.50
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.5
REGION_STRENGTH_SCALE = 0.55

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, alphabet, glyphs, calligraphy, "
    "3d, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "gray background, gray backdrop, dark gray, light gray, ash gray, gradient gray, charcoal gray, "
    "foggy background, smoky background, hazy background, dim gray paneling, "
    "yellow background, brown background, blue background, purple background, green background, "
    "scattered composition, chaotic layout, debris, flying fragments, broken composition, "
    "elements touching, elements adjacent, elements overlapping, elements intersecting, "
    "merged elements, melting into each other, blending into each other, "
    "crowded center, cluttered middle, "
    "chain piercing through skull, chain piercing through eagle, "
    "extra bird, second eagle, multiple skulls in foreground, multiple eagles overlapping, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "extra wings, asymmetric error, garbled forms, nonsense, AI artifact, tiling, repeating pattern, "
    "plastic look, fake, synthetic, low detail, simplified, cartoonish, "
    "new colors, different color palette, extra colors, color shift, recolored, hue shift, "
    # 防中央王冠被渲染成建筑（但仍允许尖刺王冠裂变）
    "castle, tower, fortress, dome, cathedral, architecture, spire roof, "
    "turrets, battlement, keep, citadel, stronghold, palace, mansion, "
    "elaborate structure on top, gothic cathedral element, "
    # 防火焰整图扩散（但仍允许侧弧火焰有走势裂变）
    "flames spreading across full image, fire dominating composition, "
    "fire in center, fire covering skull, fire covering eagle, fire everywhere, "
    "inferno, wildfire, conflagration, "
    # 防顶部小乌失真（但仍允许换物种成清晰乌鸦）
    "weird bird, abstract bird, malformed bird, twisted bird, headless bird, "
    "alien creature, mutant, chimera"
)

GLOBAL_POS = (
    "gothic heraldic crest t-shirt graphic, "
    "SAME composition, layout, element positions, poses and proportions as the reference image, "
    "KEEP the exact same arrangement, "
    "SAME color palette and art style as the reference (black background, white-silver eagle, "
    "gray skull, gray iron chains, red-orange flames, gothic iron crown), "
    "only MUTATE the INTERNAL DETAIL and TEXTURE of each element, do not alter positions or colors, "
    # ★ v155：火焰保留 side arc 不铺满，但保留"走势裂变"——比原图更有能量/多样/更高火舌
    "flames stay as a THIN SIDE ARC BACKDROP along the left and right sides only, "
    "with MORE energetic, varied and taller flame tongues than the reference (mutated flame shape), "
    "NOT a solid wall of fire, NOT covering the eagle or skulls, NOT spreading across the center, "
    "every element rendered with PIN-SHARP precision and CLEAN READABLE FORMS, "
    "professional high-end commercial apparel graphic, masterpiece best quality ultra detailed, "
    "anatomically correct and physically coherent subjects, "
    "intricate craftsmanship and fine engraved details on every element, "
    "ULTRA sharp crisp high-contrast edges, "
    "every element surrounded by a CLEAR solid-black separating outline silhouette, "
    "elements NEVER bleed into neighbors, "
    "bold graphic t-shirt print composition full bleed edge-to-edge, "
    "no halftone no noise no grain no smudge no watercolor no soft airbrush"
)

COHESIVE = (
    "KEEP its original position and size exactly, only MUTATE internal detail and texture, "
    "SAME color palette as reference, isolated against the black background, "
    "does NOT touch any other element, no overlap, no merge, no clipping through, "
    "photorealistic, fine realistic detail, real surface texture"
)

REFS = [
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "regions": [
            # 主鹰：更狂羽翼纹理/更深眼窝/更凶（v153 效果好的保留）
            {"x": 0.18, "y": 0.10, "w": 0.64, "h": 0.42, "strength": 1.20,
             "prompt": ("the bald eagle at the center-top: render with MORE aggressive, denser, sharper "
                        "silver-white feather detail, deeper carved eye socket, more pronounced curved talons, "
                        "fiercer expression, add subtle new fracture lines in the plumage, "
                        "MORE dynamic and detailed than the reference but SAME pose and SAME position. "
                        + COHESIVE)},
            # ★ v155 恢复：顶部小乌区域 → 换物种成清晰乌鸦（非抽象），保留"物种裂变"
            {"x": 0.34, "y": 0.00, "w": 0.32, "h": 0.12, "strength": 1.00,
             "prompt": ("the small bird at the very top center: change species into ONE small photorealistic "
                        "black raven (Corvus corax) with a clearly readable folded-wing bird silhouette and "
                        "glossy black plumage, NOT white, NOT an eagle, a clear recognizable bird shape, "
                        "NOT abstract, NOT malformed, KEEP its position at top center and its size. "
                        + COHESIVE)},
            # ★ v155 恢复裂变：中央王冠改回 gothic 5 尖刺（裂变视觉点），但禁建筑感
            {"x": 0.22, "y": 0.58, "w": 0.56, "h": 0.42, "strength": 1.30,
             "prompt": ("the human skull with crown at bottom center: render with DEEPER and MORE NUMEROUS "
                        "cracks and fractures across the cranium, more weathered bone texture, "
                        "a few missing teeth, more menacing realistic osteology, "
                        "on top sits a GOTHIC IRON CROWN with 5 sharp upward spikes and intricate gothic "
                        "metalwork, the crown is clearly a CROWN not a building, "
                        "NOT a castle, NOT a tower, NOT architecture, NOT a fortress, NOT a cathedral, "
                        "KEEP the crown size relative to the skull similar to reference. "
                        + COHESIVE)},
            # 左铁链：加尖刺倒钩（保留）
            {"x": 0.00, "y": 0.30, "w": 0.12, "h": 0.60, "strength": 1.20,
             "prompt": ("the iron chain along the LEFT EDGE: render each link with ADDED sharp metal spikes "
                        "and barbs, heavier industrial gothic look, more realistic metallic surface, "
                        "KEEP its position along the left edge and its length. "
                        + COHESIVE)},
            # 右铁链：加尖刺倒钩（保留）
            {"x": 0.88, "y": 0.30, "w": 0.12, "h": 0.60, "strength": 1.20,
             "prompt": ("the iron chain along the RIGHT EDGE: render each link with ADDED sharp metal spikes "
                        "and barbs, heavier industrial gothic look, more realistic metallic surface, "
                        "KEEP its position along the right edge and its length. "
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v155_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v155_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v155_{int(time.time())}"}, timeout=15)
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
    out = PROJECT_ROOT / "jobs" / "smoke_v155"
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
