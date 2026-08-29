"""v156 放松锁大裂变（用户选 B 路线）：接近 v147 级逐元素裂变。

相对 v155 的改动（解锁裂变空间）：
- Canny 0.85 → 0.50：位置约束放松，允许鹰翻转正对/元素移位
- IPA 0.60 → 0.42：风格配色仍锁，但给裂变更多主导权
- denoise 0.55 → 0.70：大裂变空间
- Tile 0.50 → 0.45：略降，给更多自由
- RegionScale 0.55 → 0.75：区域提示更强，主导裂变
- GLOBAL_POS 放宽为 "MUTATE poses, orientation, size, number AND internal detail"
- 区域 prompt 改为激进裂变（鹰正对、王冠7尖刺、铁链多尖刺、换物种）

保留的硬约束 NEG：黑底（禁灰）、色彩不变（禁改色）、禁建筑/怪鸟/穿模/融合/新增元素。

风险：放松锁会重新引入失真，需逐张验收再修。
输出：jobs/smoke_v156/v156_{id}.jpg
"""
import time, requests, sys, os, json as _json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.70                        # ★ 大裂变空间
CN_STRENGTH = 0.50                    # ★ 放松位置锁
IPA_WEIGHT = 0.42                     # ★ 放松风格锁（仍锁配色画风）
TILE_STRENGTH = 0.45                  # ★ 略降
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.5
REGION_STRENGTH_SCALE = 0.75          # ★ 区域主导裂变

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, alphabet, glyphs, calligraphy, "
    "3d, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    # 黑底硬约束（用户要求）：禁灰
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
    # 色彩不变硬约束（用户要求）：禁改色
    "new colors, different color palette, extra colors, color shift, recolored, hue shift, "
    # 防中央王冠被渲染成建筑（但仍允许尖刺王冠裂变）
    "castle, tower, fortress, dome, cathedral, architecture, spire roof, "
    "turrets, battlement, keep, citadel, stronghold, palace, mansion, "
    "elaborate structure on top, gothic cathedral element, "
    # 防顶部小乌失真（但仍允许换物种成清晰乌鸦/猛禽）
    "weird bird, abstract bird, malformed bird, twisted bird, headless bird, "
    "alien creature, mutant, chimera"
)

# ★ 放宽为允许改姿态/朝向/大小/数量
GLOBAL_POS = (
    "gothic heraldic crest t-shirt graphic, "
    "SAME color palette and art style as the reference (black background, white-silver eagle, "
    "gray skull, gray iron chains, red-orange flames, gothic iron crown), "
    "MUTATE the poses, orientation, size, NUMBER and internal detail of each element for a bold fission variant, "
    "flames rendered as dynamic energetic curved flame shapes forming a side arc backdrop on left and right, "
    "with varied taller flame tongues than reference, NOT a solid wall of fire, NOT covering eagle or skull, "
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
    "SAME color palette as reference, isolated against the black background, "
    "does NOT touch other elements, no overlap, no merge, no clipping through, "
    "photorealistic, fine realistic detail, real surface texture"
)

REFS = [
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "regions": [
            # ★ 主鹰：翻转正对镜头（v147 核心要求），更狂羽翼
            {"x": 0.18, "y": 0.08, "w": 0.64, "h": 0.44, "strength": 1.30,
             "prompt": ("the bald eagle: render FACING THE CAMERA head-on with wings spread in a symmetric "
                        "front-facing heraldic pose, MORE aggressive denser silver-white feathers, deeper "
                        "carved eye sockets, fiercer expression, add NEW fracture lines in plumage, "
                        "a bold reinterpretation in SAME white-silver color, within the upper-center area. "
                        + COHESIVE)},
            # ★ 顶部小乌：换物种成清晰猛禽（飞行姿态）
            {"x": 0.32, "y": 0.00, "w": 0.36, "h": 0.13, "strength": 1.10,
             "prompt": ("the small bird at the very top: change into ONE small photorealistic black raven "
                        "with spread wings in flight, clearly readable bird silhouette, glossy black plumage, "
                        "NOT white, NOT an eagle, NOT abstract, within the top-center area. "
                        + COHESIVE)},
            # ★ 中央骷髅+王冠：加深裂痕+旋转+7尖刺王冠（大裂变）
            {"x": 0.20, "y": 0.56, "w": 0.60, "h": 0.44, "strength": 1.40,
             "prompt": ("the human skull: DEEPER and MORE NUMEROUS cracks and fractures, rotated slightly, "
                        "more weathered bone, a few missing teeth, more menacing osteology; on top sits a "
                        "GOTHIC IRON CROWN with 7 sharp upward spikes and intricate gothic metalwork, "
                        "clearly a CROWN not a building, NOT castle/tower/architecture/fortress, "
                        "within the bottom-center area. "
                        + COHESIVE)},
            # ★ 左铁链：大量尖刺+更多链节
            {"x": 0.00, "y": 0.28, "w": 0.13, "h": 0.64, "strength": 1.30,
             "prompt": ("the iron chain along the LEFT EDGE: ADD MANY sharp metal spikes and barbs along each "
                        "link, heavier industrial gothic look, MORE visible links, realistic metallic surface, "
                        "along the left edge. "
                        + COHESIVE)},
            # ★ 右铁链：大量尖刺+更多链节
            {"x": 0.87, "y": 0.28, "w": 0.13, "h": 0.64, "strength": 1.30,
             "prompt": ("the iron chain along the RIGHT EDGE: ADD MANY sharp metal spikes and barbs along each "
                        "link, heavier industrial gothic look, MORE visible links, realistic metallic surface, "
                        "along the right edge. "
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
        "latent_image": ["4", 0], "seed": seed, "steps": 28, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.15}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v156_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v156_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v156_{int(time.time())}"}, timeout=15)
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
    out = PROJECT_ROOT / "jobs" / "smoke_v156"
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
