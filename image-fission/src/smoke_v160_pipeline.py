#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v160 多阶段裂变-精修管线（重写版）
====================================
思路（用户指令：先大裂变再精修，精修没过就再精修，找最好模型/技能做到结果）：
- Stage A : 大裂变。采用 v156 已验证的「区域提示逐元素裂变」写法
           (ConditioningSetAreaPercentage + RegionalListCombine)，配合
           Canny 0.50(松锁位置) + IPAdapter 0.42(锁配色/画风) + denoise 0.70(大裂变空间)。
           同时产出 work-res(给 Stage B) 与 4x(交付) 两版。
- Stage B : 遮罩精修。ComfyUI 原生 InpaintModelConditioning 实现"遮罩内重绘/遮罩外锁死"，
           在 work-res 上只重绘失真区（遮罩外像素保持 Stage A 结果），IPAdapter 锁原图色彩。
           内部两遍 KSampler(denoise 0.55 -> 0.30) 即「再精修」，最后只 4x 超分一次。

为何不用 BrushNet/PowerPaint：二者权重均托管 HuggingFace，被本机代理 MITM 掐断
(TLS 握手失败/SSLEOFError)，无法落地；原生 inpaint 节点无需外部权重，同样达成精细化重绘。

节点全在 ComfyUI v0.33.0 内置/已装 custom_nodes（IPAdapter_plus / Regional-Mask-Prompt /
comfyui-inpaint-nodes），无新依赖。
"""
import os, sys, time, json, shutil
import requests
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = Path(r"E:/Desktop/图裂变测试图")
INPUT_DIR = PROJECT_ROOT / "ComfyUI" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
COMFYUI = "http://127.0.0.1:8188"
CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA_DETAIL = 1.0
WORK_LONG = 1216  # 工作分辨率长边（裂变/精修都在此分辨率，最后统一 4x 超分一次）

# ---- Stage A 大裂变参数（v156 已验证）----
IPA_WEIGHT = 0.42
CN_STRENGTH = 0.50
TILE_STRENGTH = 0.45
DENOISE = 0.70
REGION_STRENGTH_SCALE = 0.75

# ---- prompts ----
COHESIVE = (
    "SAME color palette as reference, isolated against the black background, "
    "does NOT touch other elements, no overlap, no merge, no clipping through, "
    "photorealistic, fine realistic detail, real surface texture"
)
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
    "castle, tower, fortress, dome, cathedral, architecture, spire roof, "
    "turrets, battlement, keep, citadel, stronghold, palace, mansion, "
    "elaborate structure on top, gothic cathedral element, "
    "weird bird, abstract bird, malformed bird, twisted bird, headless bird, "
    "alien creature, mutant, chimera"
)
A_GLOBAL_POS = (
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
REFINE_POS = (
    "a single small black raven with neatly folded or spread wings perched at the top; "
    "a clean human skull with deep realistic cracks wearing a gothic iron crown with 7 sharp upward "
    "spikes below, no castle, no tower, no architecture, no wings growing from the skull; "
    "consistent dark gothic emblem style, sharp clean details, pure black background"
)

# eagle_2 区域提示（v156 已验证出大裂变，corr≈0.43）
EAGLE_REGIONS = [
    {"x": 0.18, "y": 0.08, "w": 0.64, "h": 0.44, "strength": 1.30,
     "prompt": ("the bald eagle: render FACING THE CAMERA head-on with wings spread in a symmetric "
                "front-facing heraldic pose, MORE aggressive denser silver-white feathers, deeper "
                "carved eye sockets, fiercer expression, add NEW fracture lines in plumage, "
                "a bold reinterpretation in SAME white-silver color, within the upper-center area. "
                + COHESIVE)},
    {"x": 0.32, "y": 0.00, "w": 0.36, "h": 0.13, "strength": 1.10,
     "prompt": ("the small bird at the very top: change into ONE small photorealistic black raven "
                "with spread wings in flight, clearly readable bird silhouette, glossy black plumage, "
                "NOT white, NOT an eagle, NOT abstract, within the top-center area. "
                + COHESIVE)},
    {"x": 0.20, "y": 0.56, "w": 0.60, "h": 0.44, "strength": 1.40,
     "prompt": ("the human skull: DEEPER and MORE NUMEROUS cracks and fractures, rotated slightly, "
                "more weathered bone, a few missing teeth, more menacing osteology; on top sits a "
                "GOTHIC IRON CROWN with 7 sharp upward spikes and intricate gothic metalwork, "
                "clearly a CROWN not a building, NOT castle/tower/architecture/fortress, "
                "within the bottom-center area. "
                + COHESIVE)},
    {"x": 0.00, "y": 0.28, "w": 0.13, "h": 0.64, "strength": 1.30,
     "prompt": ("the iron chain along the LEFT EDGE: ADD MANY sharp metal spikes and barbs along each "
                "link, heavier industrial gothic look, MORE visible links, realistic metallic surface, "
                "along the left edge. "
                + COHESIVE)},
    {"x": 0.87, "y": 0.28, "w": 0.13, "h": 0.64, "strength": 1.30,
     "prompt": ("the iron chain along the RIGHT EDGE: ADD MANY sharp metal spikes and barbs along each "
                "link, heavier industrial gothic look, MORE visible links, realistic metallic surface, "
                "along the right edge. "
                + COHESIVE)},
]

REFS = {
    "eagle_2": {
        "img": "pinterest_eagle_2.jpg",
        "regions": EAGLE_REGIONS,                       # 逐元素大裂变 + 失真区精修
        "mask": [(0.30,0.03,0.70,0.19), (0.24,0.45,0.76,0.83)],  # 失真区(顶小乌 + 中骷髅王冠)
    },
    "camo_4":  {"img": "pinterest_camo_4.jpg",  "regions": [], "mask": []},
    "illust_1":{"img": "pinterest_illust_1.jpg","regions": [], "mask": []},
    "denim_3": {"img": "pinterest_denim_3.jpg", "regions": [], "mask": []},
    "skull_5": {"img": "pinterest_skull_5.jpg", "regions": [], "mask": []},
    "metal_6": {"img": "pinterest_metal_6.jpg", "regions": [], "mask": []},
}

def work_size(img_path):
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    scale = WORK_LONG / max(w, h)
    nw = max(8, round(w * scale / 8) * 8)
    nh = max(8, round(h * scale / 8) * 8)
    return nw, nh

def preprocess(ref_id, img_path, out_dir):
    im = Image.open(img_path).convert("RGB")
    nw, nh = work_size(img_path)
    im = im.resize((nw, nh), Image.LANCZOS)
    p = INPUT_DIR / f"work_{ref_id}.png"
    im.save(p, "PNG")
    return p.name, nw, nh

def make_mask(regions, W, H, ref_id):
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    for (x0, y0, x1, y1) in regions:
        d.rectangle([int(x0*W), int(y0*H), int(x1*W), int(y1*H)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
    p = INPUT_DIR / f"mask_{ref_id}.png"
    mask.save(p, "PNG")
    return p.name

def build_stageA(work_png, seed, ref):
    g = {}
    g["1"]  = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"]  = {"class_type": "LoadImage", "inputs": {"image": work_png}}
    g["4"]  = {"class_type": "VAEEncode", "inputs": {"pixels": ["2",0], "vae": ["1",2]}}
    g["5"]  = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1",0], "preset": "PLUS (high strength)"}}
    g["6"]  = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1",0], "ipadapter": ["5",1], "image": ["2",0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"]  = {"class_type": "LoraLoader", "inputs": {
        "model": ["6",0], "clip": ["1",1], "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["2",0], "low_threshold": 0.08, "high_threshold": 0.24, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg",0], "control_net": ["21",0], "image": ["20",0], "strength": CN_STRENGTH}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22",0], "control_net": ["23",0], "image": ["2",0], "strength": TILE_STRENGTH}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7",1], "text": A_GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7",1], "text": NEG_BASE}}

    # 区域提示逐元素裂变（v156 式）
    reg = ref.get("regions", [])
    if reg:
        region_nodes = []
        for i, r in enumerate(reg):
            rk = f"rp{i}"
            g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7",1], "text": r["prompt"]}}
            sk = f"sa{i}"
            g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
                "conditioning": [rk,0], "width": r["w"], "height": r["h"],
                "x": r["x"], "y": r["y"], "strength": r["strength"] * REGION_STRENGTH_SCALE}}
            region_nodes.append(sk)
        comb_in = {"global_cond": ["pg",0]}
        for i, sk in enumerate(region_nodes):
            comb_in[f"region{i+1}"] = [sk,0]
        g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}
        pos_node = "comb"
    else:
        pos_node = "pg"

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7",0], "positive": [pos_node,0], "negative": ["ng",0],
        "latent_image": ["4",0], "seed": seed, "steps": 28, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7",0], "positive": [pos_node,0], "negative": ["ng",0],
        "latent_image": ["10",0], "seed": seed+1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.15}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11",0], "vae": ["1",2]}}
    # 同时输出 work-res(给 Stage B) 与 4x(交付)
    g["12b"] = {"class_type": "SaveImage", "inputs": {"images": ["12",0], "filename_prefix": "v160_stageA_work"}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13",0], "image": ["12",0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14",0], "filename_prefix": "v160_stageA"}}
    return g

def build_stageB(work_png, stageA_work_png, mask_png, seed):
    g = {}
    g["1"]  = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"]  = {"class_type": "LoadImage", "inputs": {"image": stageA_work_png}}   # work-res 待精修图
    g["3"]  = {"class_type": "LoadImage", "inputs": {"image": work_png}}          # 原图(锁色彩/风格)
    g["2m"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_png, "channel": "red"}}
    g["5"]  = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1",0], "preset": "PLUS (high strength)"}}
    g["6"]  = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1",0], "ipadapter": ["5",1], "image": ["3",0],
        "weight": 0.55, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"]  = {"class_type": "LoraLoader", "inputs": {
        "model": ["6",0], "clip": ["1",1], "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["2",0], "low_threshold": 0.08, "high_threshold": 0.24, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg",0], "control_net": ["21",0], "image": ["20",0], "strength": 0.55}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22",0], "control_net": ["23",0], "image": ["2",0], "strength": 0.45}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7",1], "text": REFINE_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7",1], "text": NEG_BASE}}
    g["inc"] = {"class_type": "InpaintModelConditioning", "inputs": {
        "positive": ["24",0], "negative": ["ng",0], "vae": ["1",2],
        "pixels": ["2",0], "mask": ["2m",0], "noise_mask": True}}
    # 内部两遍精修（即「再精修」）：第一遍重绘失真区，第二遍轻修收尾
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7",0], "positive": ["inc",0], "negative": ["inc",1],
        "latent_image": ["inc",2], "seed": seed, "steps": 28, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.55}}
    g["10b"] = {"class_type": "KSampler", "inputs": {
        "model": ["7",0], "positive": ["inc",0], "negative": ["inc",1],
        "latent_image": ["10",0], "seed": seed+1, "steps": 18, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.30}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10b",0], "vae": ["1",2]}}
    # 只超分一次
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13",0], "image": ["12",0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14",0], "filename_prefix": "v160_stageB"}}
    return g

def submit_and_fetch(g, reads, tag):
    """reads: list of (node_id, out_path, usm_bool)"""
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v160_{int(time.time()*1000)}"}, timeout=20)
    try: j = r.json()
    except Exception: j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {json.dumps(j)[:1500]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {tag}: no prompt_id {str(j)[:400]}", flush=True); return False
    print(f"  [{tag}] pid={pid}", flush=True)
    for i in range(110):
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                rec = h[pid]
                if rec.get("status", {}).get("completed"):
                    ok = True
                    for (node_id, out_path, usm) in reads:
                        imgs = rec.get("outputs", {}).get(node_id, {}).get("images", [])
                        if not imgs:
                            ok = False; continue
                        url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                        data = requests.get(url, timeout=90).content
                        out_path.write_bytes(data)
                        if usm:
                            try:
                                im = Image.open(out_path).convert("RGB")
                                im.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3)).save(out_path, "JPEG", quality=95, optimize=True)
                            except Exception as e:
                                print(f"  [{tag}] USM跳过 {e}", flush=True)
                        print(f"    node{node_id} -> {out_path.name} {out_path.stat().st_size/1024/1024:.1f}MB", flush=True)
                    if ok:
                        return True
                    print(f"  [{tag}] 缺节点输出", flush=True); return False
                elif rec.get("status", {}).get("error"):
                    print(f"  [{tag}] COMFY错误 {rec['status'].get('error')}", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] poll异常 {e}", flush=True)
        if i % 6 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT", flush=True); return False

def metrics(src_p, a_p, b_p):
    import numpy as np
    def load(p): return np.asarray(Image.open(p).convert("L")).astype(np.float64)
    def black(p, thr=24):
        im = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
        return float((im.mean(axis=2) < thr).mean())
    def sharp(p):
        im = np.asarray(Image.open(p).convert("L")).astype(np.float64)
        gx = np.gradient(im, axis=1); gy = np.gradient(im, axis=0)
        return float((gx**2+gy**2).mean())
    def corr(a,b):
        a=a.ravel(); b=b.ravel(); return float(np.corrcoef(a,b)[0,1])
    src=load(src_p); H,W=src.shape
    def align(p): return np.asarray(Image.open(p).convert("L").resize((W,H), Image.LANCZOS)).astype(np.float64)
    a,b=align(a_p),align(b_p)
    return (f"corr(原图,StageA)={corr(src,a):.3f}  corr(原图,StageB)={corr(src,b):.3f}  "
            f"(越低=裂变越大) | 黑底: 原图={black(src_p):.3f} A={black(a_p):.3f} B={black(b_p):.3f} | "
            f"锐度: 原图={sharp(src_p):.0f} A={sharp(a_p):.0f} B={sharp(b_p):.0f}")

def main():
    wants = sys.argv[1:] if len(sys.argv) > 1 else ["eagle_2"]
    out = PROJECT_ROOT / "jobs" / "smoke_v160"; out.mkdir(parents=True, exist_ok=True)
    for want in wants:
        if want not in REFS:
            print(f"未知 {want}"); continue
        ref = REFS[want]
        print(f"=== {want} ===", flush=True)
        src_img = SRC / ref["img"]
        work_name, W, H = preprocess(want, src_img, out)
        print(f"  work size={W}x{H} name={work_name}", flush=True)
        # Stage A
        a_out = out / f"stageA_{want}.jpg"          # 4x 交付
        a_work = out / f"stageA_{want}_work.png"    # work-res 给 Stage B
        if not (a_out.exists() and a_out.stat().st_size > 100000):
            print("  >> Stage A 大裂变", flush=True)
            if not submit_and_fetch(build_stageA(work_name, 700401, ref),
                                    [("15", a_out, True), ("12b", a_work, False)], f"{want}/A"):
                print("  Stage A 失败, 跳过", flush=True); continue
        else:
            print("  Stage A 已存在, 跳过", flush=True)
        # Stage B
        b_out = out / f"stageB_{want}.jpg"
        if ref["mask"]:
            mask_name = make_mask(ref["mask"], W, H, want)
            stageA_work_input = INPUT_DIR / f"stageA_work_{want}.png"
            shutil.copy(a_work, stageA_work_input)
            print("  >> Stage B 遮罩精修(两遍)", flush=True)
            if not submit_and_fetch(build_stageB(work_name, stageA_work_input.name, mask_name, 700402),
                                    [("15", b_out, True)], f"{want}/B"):
                print("  Stage B 失败, 用 Stage A 顶替", flush=True)
                shutil.copy(a_out, b_out)
        else:
            print("  无失真区, 跳过 Stage B", flush=True)
            shutil.copy(a_out, b_out)
        # 指标
        print("  " + metrics(str(src_img), str(a_out), str(b_out)), flush=True)
        # 拼图
        try:
            from PIL import ImageDraw as ID, ImageFont as IF
            labels = ["原始参考", "Stage A 大裂变", "Stage B 精修"]
            ps = [str(src_img), str(a_out), str(b_out)]
            ch, pad, lh = 900, 16, 50
            def rp(p):
                im=Image.open(p).convert("RGB"); iw,ih=im.size
                s=min(ch/ih, (ch*0.62)/iw); nw,nh=int(iw*s),int(ih*s)
                c=Image.new("RGB",(nw,ch),(10,10,12)); c.paste(im,((nw-c.width)//2,0)); return c
            cells=[rp(p) for p in ps]
            gw=sum(c.width for c in cells)+pad*(len(cells)+1)
            big=Image.new("RGB",(gw,ch+lh+pad),(12,12,14)); d=ID.Draw(big)
            try: f=IF.truetype(r"C:/Windows/Fonts/msyhbd.ttc",24)
            except: f=IF.load_default()
            x=pad
            for i,c in enumerate(cells):
                big.paste(c,(x,lh)); d.text((x+6,12),labels[i],fill=(225,225,225),font=f); x+=c.width+pad
            strip=out/f"compare_{want}_A_B.png"; big.save(strip,"PNG",optimize=True)
            print(f"  拼图 {strip}", flush=True)
        except Exception as e:
            print(f"  拼图失败 {e}", flush=True)
    print("ALL done", flush=True)

if __name__ == "__main__":
    sys.exit(main())
