"""v146 逐元素区域控制裂变 + 空间一致性防穿模（本机 ComfyUI / SDXL, v147 微调）。

v147 修复（针对 v146 反馈）：
- 取消骷髅头上的皇冠（避免和顶部小鹰皇冠撞车成糊状团）
- 顶部皇冠区域扩大，与下方元素间留 black gap
- 链条改为清晰 3 段沿画面右边沿，不贯穿中央、不接触骷髅/火焰
- Tile CN 0.45→0.60、Canny 0.18→0.25 强化元素边界清晰度
- 区域 strength 微涨 0.5→0.55
- NEG 加"elements touching neighboring elements / crowded center / no clear black separating outline"
- 每条区域 prompt 显式加 "CLEAR BLACK GAP between this and any other element"

根因：v145 只用 ConditioningSetAreaPercentage 改每块元素的提示词，但模型在画区域 A 时不知道区域 B 长什么样，导致元素之间穿模/脱节。

v146 修法：
- 加 Tile ControlNet fp16（strength 0.45）：保局部细节一致性，治穿模+治区域衔接断
- 加轻 Canny ControlNet（strength 0.18）：只锁「大位置/相对关系」不焊轮廓
- ConditioningCombine 把两条 CN conditioning 合并
- 区域 strength 1.3/1.4 → 0.65：不让提示词压垮空间一致性
- 每条区域 prompt 加「cohesive/connected to neighbors」衔接约束
- steps 32→24 提速（用户铁律：只跑 1 张等验收）

用法：python smoke_v146_region_tile.py [ref_id]   (默认 eagle_2)
输出：jobs/smoke_v146/v146_{id}.jpg
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.80                      # 略降（更高让区域提示独占；0.80 给 CN 一点空间）
IPA_WEIGHT = 0.18                   # 同 v145（锁配色画风）
LORA_DETAIL = 1.0                   # 同 v145（抗失真）
MEGA_PIXELS = 1.2

# 空间一致性 ControlNet（v147 增强：Tile 提到 0.60、Canny 提到 0.25 防元素边界融化）
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
CANNY_STRENGTH = 0.25   # v146:0.18 → 0.25 防元素边界融
TILE_STRENGTH = 0.60    # v146:0.45 → 0.60 拉强细节/边缘
REGION_STRENGTH_SCALE = 0.55  # v146:0.5 → 0.55 微涨

# 全局负向：锁配色 + 无字 + 抗失真 + 抗穿模 + 抗元素融合（v147 新增）
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
    "new colors, different color palette, extra colors, color shift"
)

# 全局正向：只给画风/配色基调
GLOBAL_POS = (
    "gothic tattoo illustration, pure black background, "
    "red and orange flames, white and silver eagle and skull, gray iron, "
    "bold t-shirt graphic print, high contrast, sharp edges, "
    "no text, no letters, no words, no banner, no inscription anywhere, "
    "cohesive composition, all elements connected and spatially consistent"
)

# 衔接短语：每条区域 prompt 都拼上 —— 模型才知道区域之间要连成一体
COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

REFS = [
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "regions": [
            # 老鹰：正对镜头、双翼对称展开（明显不同于原图的侧俯冲姿态）—— 改朝向/姿态
            # v147 强化：明确"翅膀和身子连一体"+"头顶与下方骷髅间留出黑色呼吸带（防止两皇冠/元素挤压）"
            {"x": 0.10, "y": 0.00, "w": 0.80, "h": 0.42, "strength": 1.25,
             "prompt": ("a bald eagle FACING THE CAMERA head-on (NOT a side profile), "
                        "both wings spread symmetrically outward from a central body, "
                        "wings and body anatomically connected as one continuous bird, no wing-body separation, "
                        "fierce open beak, sharp white and silver feathers with fine detail, "
                        "centered dominant focal element at the top with a CLEAR BLACK GAP between it and any lower elements, "
                        "gothic tattoo style, black background. "
                        + COHESIVE)},
            # 骷髅：3/4 朝向、深裂痕、单颗（v147 不再戴皇冠，避免和顶部皇冠撞车）
            {"x": 0.30, "y": 0.50, "w": 0.40, "h": 0.46, "strength": 1.35,
             "prompt": ("ONE single large human skull turned at a three-quarter angle (NOT a front view), "
                        "deep realistic cracks and fractures across the bone surface, "
                        "no crown, no hat, just bone with cracked skullcap, "
                        "gray bone with realistic shadow, "
                        "CLEAR BLACK GAP between this skull and any element above it, "
                        "centered lower area, gothic. "
                        + COHESIVE)},
            # 火焰：左下向右上斜升的火柱（改燃烧走势/方向）
            {"x": 0.00, "y": 0.22, "w": 0.22, "h": 0.74, "strength": 1.15,
             "prompt": ("red and orange flames rising as a DIAGONAL sweeping column flowing from lower-left toward upper-right, "
                        "dynamic motion, asymmetric, no wrapping symmetry, "
                        "fires DO NOT touch the skull, separate with black space. "
                        + COHESIVE)},
            # 铁链：v147 重写——清晰 3 段铁链沿画面右下边→向上→顶部回折，不再贯穿整个右侧中间
            {"x": 0.62, "y": 0.42, "w": 0.38, "h": 0.55, "strength": 1.30,
             "prompt": ("THREE DISTINCT segments of heavy iron chain with sharp metal SPIKES, "
                        "running ALONG the right edge: (1) bottom-right corner curling up, "
                        "(2) middle-right straight vertical drop, (3) top-right diagonal back toward center but stopping before skull, "
                        "industrial gothic, gray metal, bold SOLID LINK outlines, "
                        "each chain segment SEPARATED from skull and flame by clear black space, "
                        "CHAIN DOES NOT TOUCH the skull or eagle at all. "
                        + COHESIVE)},
            # 顶部小元素：原图后面小鹰换成哥特铁王冠（v147 加大区域，与下方骷髅留清晰黑色间隔）
            {"x": 0.36, "y": 0.00, "w": 0.28, "h": 0.12, "strength": 1.10,
             "prompt": ("a SINGLE clean gothic iron crown with 5 visible spikes at the very top center, "
                        "no birds anywhere near this area, the crown SITS ALONE in its own black region, "
                        "decorative only, no chains attached, no skulls attached, "
                        "visually SEPARATED from the eagle below by a clear black gap. "
                        + COHESIVE)},
        ],
    },
]


def scaled_region_strengths(ref):
    """每条区域 strength 乘 REGION_STRENGTH_SCALE，减少提示词对空间一致性的压垮。"""
    return [{**r, "strength": r["strength"] * REGION_STRENGTH_SCALE} for r in ref["regions"]]


def build(ref, seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": MEGA_PIXELS, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}

    # IPAdapter style 锁配色（沿用 v145）
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

    # === v146 新增：空间一致性双 ControlNet ===
    # Canny 边缘预处理器 + Canny ControlNet（轻骨架 0.18）
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": CANNY_STRENGTH}}

    # Tile ControlNet（保细节防穿模 0.45）
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": TILE_STRENGTH}}

    # 全局正负向编码（node "pg"/"ng" 在 24 之前被引用，ComfyUI 按 id 排序自动执行顺序正确）
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    # 每个区域独立编码 + 内置 ConditioningSetAreaPercentage
    region_nodes = []
    for i, r in enumerate(scaled_region_strengths(ref)):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"]}}
        region_nodes.append(sk)

    # 合并：global + 各区域（各区域已带 area），靠 RegionalListCombine 把列表追加
    comb_in = {"global_cond": ["pg", 0]}
    for i, sk in enumerate(region_nodes):
        comb_in[f"region{i+1}"] = [sk, 0]
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}

    # 双 KSampler（减速: 24/20 步）
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v146_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v146_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    import json as _json
    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v146_{int(time.time())}"}, timeout=15)
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
    out = PROJECT_ROOT / "jobs" / "smoke_v146"
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
