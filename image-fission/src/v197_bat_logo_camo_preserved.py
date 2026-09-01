"""
v197 — bat_logo 重做：保留原图迷彩底色 + 真清晰可读英文单词

v193 的两个错：
1. ref_img 用了 test_6978fab（紫底 BACARDÍ 图），不是 test_5784eab（迷彩底）
   → SDXL 复刻紫底，整张变紫，迷彩材质/画风完全丢失
2. prompt 写 "STYLIZED LETTERING shaped into the badge as graphical elements
   (looks like abstract letter-shaped ribbons woven into the badge,
   NOT readable English words)" → SDXL 拼伪词（SEARTUEES / STIRPUA LEF / STARI / SIPITR）

v197 反着做：
- ref_img = test_5784eab（迷彩底原图）
- global_pos 明确写 PRESERVED camo 4-color palette + centered circular badge with bat
- 真清晰可读英文单词：MOONCREST (拱形) / CURSE (中央) / EST. MMXXVI (下方)
- v147 锁死基线参数（不动 denoise/tile/canny/IPA）
- workflow 完全对齐 v195（RegionalListCombine + ControlNetApply）
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import requests
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v197"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

# ==================== v147 锁死基线参数 ====================
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

# ==================== NEG_BASE（v147 标准 + 颜色防漂）====================
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "purple background, violet background, lavender background, magenta background, pink background, "
    "BACARDI logo, Bacardi bat, branded bat emblem, branded logo, branded badge, "
    "soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
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

# ==================== bat_logo 提示词（v197 修正版）====================
REFS = [
    {
        "id": "bat_logo",
        # 关键修复：用对的 ref_img——纯迷彩底原图
        "ref_img": "test_5784eab326634d17573b469e91cdc565.jpg",
        "global_pos": (
            # 反 v193：第 1 句就锁死迷彩底色 + 明示徽章叠加
            "bold military camouflage print PRESERVED as the FULL BACKGROUND, "
            "classic 4-color camo palette exactly: olive green #4B5320, "
            "tan/khaki #C2B280, dark brown #4A2C2A, and black color blocks, "
            "irregular organic camo blob shapes filling every corner of the canvas, "
            # 徽章叠加中央
            "with a centered circular vintage military badge in the middle, "
            "the badge has a dark olive green and tan color scheme matching the camo, "
            # 真清晰可读英文单词——明示是单词不是装饰
            "the badge displays the clearly readable English words "
            "'MOONCREST' (curved along the upper arc of the badge) "
            "'CURSE' (large bold letters across the middle of the badge) "
            "'EST. MMXXVI' (smaller letters at the bottom of the badge), "
            # 徽章主体
            "the badge contains a stylized bat silhouette rotated 30 degrees, "
            "small geometric star accents around the badge ring, "
            "sharp clean edges, vintage military emblem style, "
            "PRESERVE camo background fully — do NOT replace with solid purple or any other color"
        ),
        "regions": [
            # 徽章主体：蝙蝠 + 真清晰字 + 风格保持
            {"x": 0.18, "y": 0.18, "w": 0.64, "h": 0.62, "strength": 1.30,
             "prompt": (
                 "a centered circular military badge with a stylized bat silhouette inside, "
                 "the bat is rotated 30 degrees clockwise with wings spread 1.2x wider, "
                 "the badge has a clearly readable curved English word 'MOONCREST' "
                 "along the upper arc of the badge ring, "
                 "a clearly readable bold English word 'CURSE' across the middle, "
                 "a clearly readable smaller English text 'EST. MMXXVI' at the bottom, "
                 "the words are SHARP, LEGIBLE, READABLE English letters (not decorative ribbons, "
                 "not stylized abstract letter shapes, not illegible ornament), "
                 "small star accents around the badge ring, "
                 "dark olive green and tan badge color scheme, "
                 "vintage military emblem style with sharp clean edges. "
                 "cohesive with the rest of the design, anatomically connected, "
                 "no floating disconnected parts, no clipping through other elements, "
                 "natural overlap hierarchy, fits the overall composition"
             )},
            # 顶部飘带：EST. MMXXVI
            {"x": 0.10, "y": 0.02, "w": 0.80, "h": 0.13, "strength": 1.05,
             "prompt": (
                 "a flat top decorative banner with clearly readable English text 'EST. MMXXVI' "
                 "in sharp legible bold letters, "
                 "tan and olive green banner color matching the camo palette. "
                 "cohesive with the rest of the design"
             )},
            # 底部装饰带
            {"x": 0.20, "y": 0.88, "w": 0.60, "h": 0.10, "strength": 1.00,
             "prompt": (
                 "a flat bottom decorative banner with subtle ornamental patterns, "
                 "tan and dark brown color matching the camo palette. "
                 "cohesive with the rest of the design"
             )},
        ],
    },
]


# ==================== Workflow 生成（对齐 v195 节点接法）====================
def build_workflow(ref, seed=700970):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScale", "inputs": {
        "image": ["2", 0], "width": 1024, "height": 1024,
        "upscale_method": "lanczos", "crop": "center"}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {
        "model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["5", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "embeds_scaling": "V only",
        "start_at": 0.0, "end_at": 0.85, "noise": 0.05}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1],
        "lora_name": LORA, "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}

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
    for i, r in enumerate(ref["regions"]):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"] * REGION_STRENGTH_SCALE}}
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
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
        "upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {
        "images": ["14", 0], "filename_prefix": f"v197_{ref['id']}"}}
    return g


# ==================== Run ====================
def submit(wf):
    try:
        r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": wf}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERR submit] {e} | body: {getattr(r, 'text', '')[:2500]}")
        return None


def wait_and_download(prompt_id, ref_id):
    for i in range(180):
        try:
            r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
            r.raise_for_status()
            hist = r.json()
            if prompt_id in hist:
                outputs = hist[prompt_id].get("outputs", {})
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        for img in node_out["images"]:
                            fn = img["filename"]
                            sub = img.get("subfolder", "")
                            url = f"{COMFY_URL}/view"
                            params = {"filename": fn, "type": "output"}
                            if sub:
                                params["subfolder"] = sub
                            img_bytes = requests.get(url, params=params, timeout=60).content
                            dst = JOB / f"v197_{ref_id}.jpg"
                            try:
                                Image.open(__import__("io").BytesIO(img_bytes)).save(dst)
                            except Exception:
                                dst.write_bytes(img_bytes)
                            return dst
        except Exception:
            pass
        time.sleep(2)
    print("[ERR] timeout")
    return None


def main():
    if not COMFY_INPUT.exists():
        raise SystemExit(f"ComfyUI input not found: {COMFY_INPUT}")

    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
        print(f"[comfyui] OK ({r.status_code})")
    except Exception as e:
        raise SystemExit(f"ComfyUI not running: {e}")

    ref = REFS[0]
    src_path = COMFY_INPUT / ref["ref_img"]
    if not src_path.exists():
        raise SystemExit(f"ref image missing: {src_path}")
    print(f"[ref] {src_path}")

    wf = build_workflow(ref)
    res = submit(wf)
    if not res:
        return
    prompt_id = res["prompt_id"]
    print(f"[prompt] {prompt_id}")
    print(f"[job dir] {JOB}")
    print(f"[wait] ~80s for KSampler 24+20 + 4x upscale...")

    result_path = wait_and_download(prompt_id, ref["id"])
    if result_path:
        print(f"\n[done] {result_path}")


if __name__ == "__main__":
    main()