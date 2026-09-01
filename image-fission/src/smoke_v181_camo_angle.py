"""v181 camo_4 一版裂变：基于 v147/v164 管线锁死参数 + v174 的 LORA_DETAIL=0（防迷彩碎裂），
只改 camo_4 的 5 区域提示词，严格遵守新红线：
  - 圆润有机迷彩色块（去掉 v164 写的 "sharp edges"，改成 soft rounded organic blob edges）
  - 严格 4 色配色：深橄榄绿 / 棕 / 沙（迷彩）+ 黑色（椰子树轮廓），不出现第 5 色
  - 椰子树矢量风（黑色 silhouette + crisp outline）不变
  - 只改：椰子树角度/大小/数量；迷彩斑块大小/分布（保持圆润）
  - 无文本
技术参数全锁（铁律1）：
  - DENOISE=0.80  TILE=0.60  CANNY=0.25  IPA=0.18 (style transfer)
  - LORA_DETAIL=0（v174 验证，防碎裂）
  - ProteusV0.4 / KSampler 24+20 / 4x_NMKD-Siax / MEGA_PIXELS=1.2

用法：python smoke_v181_camo_angle.py  (默认跑 camo_4)
"""
import time, requests, sys, os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700405
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

# 共享 NEG（与 v164 一致）—— 已含 "new colors, different color palette, extra colors, color shift" 防配色漂移
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
    "sharp geometric edges, polygonal color blocks, fractured camouflage, shattered pattern, "
    "pastel color, washed out color, desaturated"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

# ===== camo_4 5 区域：圆润 + 严格 4 色 + 椰子树改角度/大小/数量 =====
REFS = [
    {
        "id": "camo_4", "ref_img": "pinterest_camo_4.jpg",
        "global_pos": ("bold organic rounded military camouflage print pattern, "
                       "vector illustration style, "
                       "large SOFT ROUNDED ORGANIC BLOB color blocks "
                       "in olive drab green, dark brown, and dusty tan "
                       "(exactly 3 camouflage colors only, no other colors), "
                       "black palm tree silhouettes with crisp clean outline, "
                       "no text, no letters, no words anywhere, "
                       "fabric print quality, repeatable seamless pattern feel, "
                       "blob edges are soft organic curves not sharp geometric polygons"),
        "regions": [
            # 区域1 中央大棕榈：弯 30 度向右（角度变化）+ 加大 60% 垂直空间（大小变化）
            {"x": 0.30, "y": 0.00, "w": 0.40, "h": 0.50, "strength": 1.25,
             "prompt": ("a TALL bold royal palm tree centered in the design, "
                        "thin curving trunk BENT SHARPLY to the right at 30 degree angle by strong wind, "
                        "TOP CROWN of wide fan-shaped fronds spreading in 8 to 10 distinct plumes "
                        "tilted right following the wind, "
                        "LARGER than usual taking 60 percent of vertical space, "
                        "pure black silhouette with crisp clean outlines, "
                        "tropical military style. " + COHESIVE)},
            # 区域2 右侧次椰子树：完美垂直 90 度（角度变化对比）+ 中等大小 70% 主棕榈（大小变化）
            {"x": 0.62, "y": 0.20, "w": 0.30, "h": 0.50, "strength": 1.20,
             "prompt": ("a SECONDARY tall straight coconut palm tree in the right portion, "
                        "perfectly VERTICAL trunk at 90 degrees with NO bending at all, "
                        "smaller curved fronds in 5 plumes, "
                        "MEDIUM size about 70 percent of the main palm, "
                        "pure black silhouette with crisp outline, "
                        "contrasts strongly with the main palm bent to the right. " + COHESIVE)},
            # 区域3 左下小棕榈：左倾 20 度（角度变化）+ 更小 40% 主棕榈（大小变化）+ 数量少 4 片叶
            {"x": 0.05, "y": 0.45, "w": 0.30, "h": 0.55, "strength": 1.10,
             "prompt": ("a small SLENDER palm tree in the bottom-left corner, "
                        "YOUNG sapling style only 40 percent height of main palm, "
                        "trunk LEANING LEFT 20 degrees, "
                        "only 4 drooping fronds, "
                        "tucked behind a camo color block, "
                        "pure black silhouette. " + COHESIVE)},
            # 区域4 迷彩斑块：圆润有机，大小混合（一些大如树冠一些小如叶尖），严格 3 色（绿/棕/沙），无黑色
            {"x": 0.00, "y": 0.20, "w": 1.0, "h": 0.30, "strength": 1.30,
             "prompt": ("large rounded organic camouflage blobs in MIXED SIZES, "
                        "some as large as palm tree crown, some as small as frond tip, "
                        "3 colors ONLY: olive drab green #4B5320, dark brown #4A2C2A, "
                        "and tan #C2B280 (no black, no other colors), "
                        "all blob edges are SOFT ROUNDED ORGANIC CURVES not sharp geometric, "
                        "no gradient, no soft airbrush, "
                        "fabric-print-ready military camouflage pattern. " + COHESIVE)},
            # 区域5 底部装饰带：8 棵小棕榈（数量 6→8 增），每棵角度不同（角度变化），大小递减（大小变化）
            {"x": 0.00, "y": 0.92, "w": 1.0, "h": 0.08, "strength": 1.05,
             "prompt": ("a thin horizontal band of EIGHT tiny palm tree silhouettes in a row at the very bottom, "
                        "each at a DIFFERENT ANGLE: some leaning right some leaning left some perfectly straight, "
                        "sizes GRADUALLY DECREASING from left to right, each 85 percent of previous, "
                        "pure black silhouettes with no detail, "
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v181_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v181_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在 {out.stat().st_size/1024/1024:.1f}MB，跳过", flush=True); return True
    if not ref.get("regions"):
        print(f"  [{tag}] 无 regions 配置，跳过", flush=True); return True

    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v181_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {json.dumps(j)[:1500]}", flush=True); return False
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
    out = PROJECT_ROOT / "jobs" / "smoke_v181"
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