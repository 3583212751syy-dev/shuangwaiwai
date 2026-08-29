"""v145 逐元素区域控制裂变（本机 ComfyUI / SDXL）。

用户 08-29 最终裂变定义：每个元素独立改变「角度/朝向/姿态/大小/数量/主题」——
不再是整体照搬或只换纹理。具体对 eagle_2 的要求：
- 老鹰正对镜头（不再是侧俯冲），朝向可改
- 骷髅头朝向改 + 裂痕增加 + 数量可改
- 火焰燃烧走势/方向改
- 铁链多加 / 减少 / 加金属尖刺
- 后面小老鹰换成别的元素
- 配色（黑/红/银）锁死，不换物种大类

实现：img2img(VAEEncode 锁大构图) + IPAdapter style 0.18(锁配色) + Detail Tweaker 1.0(抗失真)
+ 区域提示（ComfyUI-Lora-Pipeline 的 ConditioningPipelineSetArea，git 安装的节点）
  分别给 老鹰/骷髅/火焰/铁链/顶部小元素 各写独立 prompt，
  实现「每个元素角度/朝向/数量/主题」独立裂变。
不锁 Canny（Canny 从原图提取会焊死原姿态，与“改朝向”冲突）。

节点来源：git clone https://github.com/andreszs/ComfyUI-Lora-Pipeline (custom_nodes/ComfyUI-Lora-Pipeline)

用法：python smoke_v145_region.py [ref_id]   (默认 eagle_2)
输出：jobs/smoke_v145/v145_{id}.jpg
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.85
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.2

# 全局负向（锁配色/无字/抗失真）
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
    "new colors, different color palette, extra colors, color shift"
)

# 全局正向（只给画风/配色/无字基调，具体元素交给区域 prompt）
GLOBAL_POS = (
    "gothic tattoo illustration, pure black background, "
    "red and orange flames, white and silver eagle and skull, gray iron, "
    "bold t-shirt graphic print, high contrast, sharp edges, "
    "no text, no letters, no words, no banner, no inscription anywhere"
)

# 每张图的「区域裂变」定义：regions 内每个元素独立 prompt + 归一化 x/y/w/h + strength
REFS = [
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        # 区域坐标基于 latent（归一化 0~1）。原图竖构图：鹰在上方、骷髅在下方、火焰两侧、铁链斜穿。
        "regions": [
            # 老鹰：正对镜头、双翼对称展开 —— 直接改朝向
            {"x": 0.12, "y": 0.00, "w": 0.76, "h": 0.46, "strength": 1.3,
             "prompt": ("a bald eagle FACING THE CAMERA head-on, both wings spread symmetrically outward, "
                        "fierce open beak, sharp white and silver feathers with fine detail, "
                        "centered dominant focal element at the top, gothic tattoo style, black background")},
            # 骷髅：3/4 朝向、深裂痕、单颗 + 铁皇冠 —— 改朝向/裂痕/数量
            {"x": 0.28, "y": 0.46, "w": 0.44, "h": 0.46, "strength": 1.4,
             "prompt": ("ONE single large human skull turned at a three-quarter angle, "
                        "deep realistic cracks and fractures across the bone surface, "
                        "wearing an iron spike crown on top, gray bone color with shadow, "
                        "centered lower area, gothic")},
            # 火焰：左下向右上斜升的火柱 —— 改燃烧走势
            {"x": 0.00, "y": 0.22, "w": 0.26, "h": 0.74, "strength": 1.1,
             "prompt": ("red and orange flames rising as a DIAGONAL sweeping column flowing toward the upper left, "
                        "dynamic motion, asymmetric, no wrapping symmetry")},
            # 铁链：右侧斜穿 + 金属尖刺 —— 加尖刺/改量
            {"x": 0.74, "y": 0.10, "w": 0.26, "h": 0.80, "strength": 1.3,
             "prompt": ("heavy iron chains with sharp metal SPIKES crisscrossing diagonally across the right side, "
                        "industrial gothic, gray metal, bold links")},
            # 顶部小元素：把原图后面小鹰换成哥特铁王冠 —— 换元素/主题
            {"x": 0.40, "y": 0.00, "w": 0.20, "h": 0.14, "strength": 1.2,
             "prompt": ("a gothic iron crown with small hanging chains at the very top center, "
                        "decorative, replacing any small bird, no eagle here")},
        ],
    },
]


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

    # 全局正负向编码
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    # 每个区域独立编码 + 内置 ConditioningSetAreaPercentage 设置 area（归一化坐标）
    region_nodes = []
    for i, r in enumerate(ref["regions"]):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"]}}
        region_nodes.append(sk)

    # 合并：global + 各区域（各区域已带自己的 area），靠 RegionalListCombine 把列表追加起来
    comb_in = {"global_cond": ["pg", 0]}
    for i, sk in enumerate(region_nodes):
        comb_in[f"region{i+1}"] = [sk, 0]
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 32, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 28, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v145_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v145_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": build(ref, seed),
                    "client_id": f"v145_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        import json as _json
        print(f"[ERR] {tag}: {r.status_code} {_json.dumps(j)[:1200]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {tag}: 无 prompt_id {str(j)[:400]}", flush=True); return False
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
    out = PROJECT_ROOT / "jobs" / "smoke_v145"
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
