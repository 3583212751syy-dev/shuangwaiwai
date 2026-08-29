#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v161 逐元素 BrushNet 精修
================================
用户指令：「去git里找更能精细化处理的模型...去给每张图手工标元素 mask 逐个重画」

管线（eagle_2 单图先验）：
- Stage A：复用 v160 区域提示大裂变（区域 prompt + 区域 strength 0.75，Canny 0.50 / IPA 0.42 / denoise 0.70）
- Stage B：逐元素 BrushNet SDXL 精修
  · 每个元素一个独立 mask + 独立 prompt（消除 v160 凭空加 3 骷髅/夹克/权杖的失真）
  · 上一步输出作为下一步输入（链式精修）
  · BrushNet (nullquant) + random_mask SDXL — 专门为 inpaint 训练，原生实现最细
  · denoise 0.50（BrushNet 专用模型可比基模用更低 denoise 也出真细节）
  · IPAdapter 锁原图配色画风

为何此管线胜于 v160（InpaintModelConditioning 路径）：
- BrushNet 权重专为 inpaint 训练，遮罩边界更自然、无明显拼接缝
- 每个元素单独 prompt+单独 mask → 模型只在指定区域生成指定内容，**不会越界**
- 用 random_mask（不是 segmentation_mask）因为我们的 mask 是矩形，不依赖形状先验
- 用户可「手工标 mask」控制每个元素的范围和 prompt（per-image per-element 可定制）
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
WORK_LONG = 1216

BRUSHNET_MODEL = "diffusion_pytorch_model.safetensors"  # 已在 models/inpaint/

# Stage A (与 v160 相同，复制过来不依赖)
IPA_WEIGHT = 0.42
CN_STRENGTH = 0.50
TILE_STRENGTH = 0.45
DENOISE_A = 0.70
REGION_STRENGTH_SCALE = 0.75

# Stage B (逐元素精修)
DENOISE_B = 0.50          # BrushNet 专用模型用 0.50 足矣
BRUSHNET_SCALE = 0.80
BRUSHNET_START = 0
BRUSHNET_END = 10000

NEG_BASE = (
    "frame, border, white border, edge outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, "
    "3d, painterly, illustration by child, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "gray background, gray backdrop, dark gray, light gray, ash gray, gradient gray, charcoal gray, "
    "foggy background, smoky background, hazy background, "
    "yellow background, brown background, blue background, purple background, green background, "
    "scattered composition, chaotic layout, debris, flying fragments, broken composition, "
    "elements touching, elements adjacent, elements overlapping, elements intersecting, "
    "merged elements, melting into each other, blending into each other, "
    "crowded center, cluttered middle, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, "
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
COHESIVE = (
    "SAME color palette as reference, isolated against the black background, "
    "does NOT touch other elements, no overlap, no merge, no clipping through, "
    "photorealistic, fine realistic detail, real surface texture"
)

EAGLE_A_REGIONS = [
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

# 逐元素精修（mask + prompt）— 专门治 v160 过度裂变（多了 3 骷髅/夹克/权杖）
# 每个元素：rect + 严格 prompt（只生成指定内容，禁止越界）
EAGLE_B_ELEMENTS = [
    {
        "name": "top_raven",
        "rect": (0.30, 0.03, 0.70, 0.19),
        "prompt": ("ONE single small photorealistic black raven with spread wings in flight, "
                   "clean simple bird silhouette, glossy black plumage, clean readable form, "
                   "isolated in pure black background, do NOT touch the eagle below, do NOT overlap, "
                   "do NOT grow horns or antlers, do NOT become a skull, do NOT be an eagle, do NOT be white, "
                   "centered at top"),
        "neg": NEG_BASE + ", horns, antlers, skull, dragon, phoenix, white bird, two birds, multiple birds, flock",
    },
    {
        "name": "central_skull",
        "rect": (0.18, 0.40, 0.82, 0.85),
        "prompt": ("ONE single bare weathered human skull with deep realistic cracks and fractures, "
                   "wearing a SIMPLE gothic iron crown with 7 SHARP upward spikes, "
                   "BARE BONE ONLY, no clothing, no coat, no jacket, no armor, no vest, no robe, no cape, "
                   "no neck chain, no pentagram, no pendant, no medallion, no shirt, no tie, "
                   "just skull and crown in this region, do NOT add extra skulls, do NOT add weapons, "
                   "do NOT add scepter, do NOT add staff, do NOT add any extra human-like figure, "
                   "isolated in pure black background, do NOT touch the eagle above or the chains, "
                   "centered in the middle area"),
        "neg": NEG_BASE + ", coat, jacket, armor, vest, robe, cape, shirt, tie, neck chain, pentagram, pendant, medallion, "
                       "multiple skulls, three skulls, two skulls, weapon, sword, dagger, axe, scepter, staff, "
                       "human figure, person, body, hand, fingers, arm",
    },
    {
        "name": "bottom_clear",
        "rect": (0.18, 0.86, 0.82, 1.00),
        "prompt": ("clean pure black void, empty black background, "
                   "do NOT add any element, do NOT add skulls, do NOT add scepter, do NOT add figures, "
                   "just deep pure black"),
        "neg": NEG_BASE + ", skull, scepter, staff, weapon, figure, person, object, item, element, decoration",
    },
]

# REFS 结构：每张图的 Stage A 区域 + Stage B 逐元素 mask+prompt
REFS = {
    "eagle_2": {
        "img": "pinterest_eagle_2.jpg",
        "a_regions": EAGLE_A_REGIONS,
        "b_elements": EAGLE_B_ELEMENTS,
    },
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

def make_mask(rect, W, H, name):
    x0, y0, x1, y1 = rect
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    d.rectangle([int(x0*W), int(y0*H), int(x1*W), int(y1*H)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=8))
    p = INPUT_DIR / f"mask_{name}.png"
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

    reg = ref["a_regions"]
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
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE_A}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7",0], "positive": [pos_node,0], "negative": ["ng",0],
        "latent_image": ["10",0], "seed": seed+1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.15}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11",0], "vae": ["1",2]}}
    g["12b"] = {"class_type": "SaveImage", "inputs": {"images": ["12",0], "filename_prefix": "v161_stageA_work"}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13",0], "image": ["12",0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14",0], "filename_prefix": "v161_stageA"}}
    return g

def build_brushnet_element(input_png, mask_png, prompt, neg, seed, prefix, denoise=DENOISE_B):
    """单元素 BrushNet inpaint 图（BrushNet + KSampler + VAEDecode + SaveImage）"""
    g = {}
    g["1"]  = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"]  = {"class_type": "LoadImage", "inputs": {"image": input_png}}      # 上一步输出
    g["3"]  = {"class_type": "LoadImageMask", "inputs": {"image": mask_png, "channel": "red"}}
    g["4"]  = {"class_type": "BrushNetLoader", "inputs": {"brushnet": BRUSHNET_MODEL, "dtype": "float16"}}
    g["5"]  = {"class_type": "LoraLoader", "inputs": {
        "model": ["1",0], "clip": ["1",1], "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}
    g["6"]  = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5",1], "text": prompt}}
    g["7"]  = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5",1], "text": neg}}
    g["8"]  = {"class_type": "BrushNet", "inputs": {
        "model": ["5",0], "vae": ["1",2], "image": ["2",0], "mask": ["3",0],
        "brushnet": ["4",0], "positive": ["6",0], "negative": ["7",0],
        "scale": BRUSHNET_SCALE, "start_at": BRUSHNET_START, "end_at": BRUSHNET_END}}
    g["9"]  = {"class_type": "KSampler", "inputs": {
        "model": ["8",0], "positive": ["8",1], "negative": ["8",2], "latent_image": ["8",3],
        "seed": seed, "steps": 24, "cfg": 6.5, "sampler_name": "euler",
        "scheduler": "normal", "denoise": denoise}}
    g["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9",0], "vae": ["1",2]}}
    g["11"] = {"class_type": "SaveImage", "inputs": {"images": ["10",0], "filename_prefix": prefix}}
    return g

def build_upscale(input_png, prefix):
    g = {}
    g["1"] = {"class_type": "LoadImage", "inputs": {"image": input_png}}
    g["2"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["3"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2",0], "image": ["1",0]}}
    g["4"] = {"class_type": "SaveImage", "inputs": {"images": ["3",0], "filename_prefix": prefix}}
    return g

def submit_and_fetch(g, reads, tag):
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v161_{int(time.time()*1000)}"}, timeout=20)
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
                    if ok: return True
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
    return (f"corr(原图,StageA)={corr(src,a):.3f}  corr(原图,Final)={corr(src,b):.3f}  "
            f"(越低=裂变越大) | 黑底: 原图={black(src_p):.3f} A={black(a_p):.3f} Final={black(b_p):.3f} | "
            f"锐度: 原图={sharp(src_p):.0f} A={sharp(a_p):.0f} Final={sharp(b_p):.0f}")

def main():
    wants = sys.argv[1:] if len(sys.argv) > 1 else ["eagle_2"]
    out = PROJECT_ROOT / "jobs" / "smoke_v161"; out.mkdir(parents=True, exist_ok=True)
    for want in wants:
        if want not in REFS:
            print(f"未知 {want}"); continue
        ref = REFS[want]
        print(f"=== {want} ===", flush=True)
        src_img = SRC / ref["img"]
        work_name, W, H = preprocess(want, src_img, out)
        print(f"  work size={W}x{H} name={work_name}", flush=True)
        # ---- Stage A ----
        a_out = out / f"stageA_{want}.jpg"
        a_work = out / f"stageA_{want}_work.png"
        if not (a_out.exists() and a_out.stat().st_size > 100000):
            print("  >> Stage A 大裂变 (v160 区域提示)", flush=True)
            if not submit_and_fetch(build_stageA(work_name, 700401, ref),
                                    [("15", a_out, True), ("12b", a_work, False)], f"{want}/A"):
                print("  Stage A 失败, 跳过", flush=True); continue
        else:
            print("  Stage A 已存在, 跳过", flush=True)
        # ---- Stage B 逐元素 BrushNet 精修（链式）----
        current_input = a_work
        b_chains = []  # 保存每步中间结果用于诊断
        for i, el in enumerate(ref["b_elements"]):
            mask_name = make_mask(el["rect"], W, H, f"{want}_{el['name']}")
            current_input_dest = INPUT_DIR / f"v161_step_{want}_{i}.png"
            shutil.copy(current_input, current_input_dest)
            step_out = out / f"step{i+1}_{want}_{el['name']}.png"
            print(f"  >> Step {i+1}: {el['name']} rect={el['rect']}", flush=True)
            if not submit_and_fetch(
                build_brushnet_element(current_input_dest.name, mask_name,
                                       el["prompt"], el["neg"],
                                       700410 + i*10, f"v161_{want}_{el['name']}"),
                [("11", step_out, False)], f"{want}/S{i+1}"
            ):
                print(f"  Step {i+1} 失败, 跳过剩余", flush=True); break
            current_input = step_out
            b_chains.append(step_out)
        # current_input 已是工作分辨率；最后超分
        final_work = current_input
        b_out = out / f"final_{want}.jpg"
        # 把 final_work 放到 INPUT_DIR 再超分
        final_input_name = f"v161_final_{want}.png"
        shutil.copy(final_work, INPUT_DIR / final_input_name)
        print(f"  >> Final 4x 超分", flush=True)
        if not submit_and_fetch(build_upscale(final_input_name, f"v161_final_{want}"),
                                [("4", b_out, True)], f"{want}/F"):
            print("  Final 超分失败, 用 stageA 顶替", flush=True)
            shutil.copy(a_out, b_out)
        # ---- 指标 ----
        print("  " + metrics(str(src_img), str(a_out), str(b_out)), flush=True)
        # ---- 拼图 ----
        try:
            from PIL import ImageDraw as ID, ImageFont as IF
            cells = [str(src_img), str(a_out), str(b_out)]
            labels = ["原始参考", "Stage A 大裂变", "v161 最终 (逐元素 BrushNet)"]
            ch, pad, lh = 900, 16, 50
            def rp(p):
                im=Image.open(p).convert("RGB"); iw,ih=im.size
                s=min(ch/ih, (ch*0.62)/iw); nw,nh=int(iw*s),int(ih*s)
                c=Image.new("RGB",(nw,ch),(10,10,12)); c.paste(im,((nw-c.width)//2,0)); return c
            cells_im=[rp(p) for p in cells]
            gw=sum(c.width for c in cells_im)+pad*(len(cells_im)+1)
            big=Image.new("RGB",(gw,ch+lh+pad),(12,12,14)); d=ID.Draw(big)
            try: f=IF.truetype(r"C:/Windows/Fonts/msyhbd.ttc",24)
            except: f=IF.load_default()
            x=pad
            for i,c in enumerate(cells_im):
                big.paste(c,(x,lh)); d.text((x+6,12),labels[i],fill=(225,225,225),font=f); x+=c.width+pad
            strip=out/f"compare_{want}_v161.png"; big.save(strip,"PNG",optimize=True)
            print(f"  拼图 {strip}", flush=True)
        except Exception as e:
            print(f"  拼图失败 {e}", flush=True)
    print("ALL done", flush=True)

if __name__ == "__main__":
    sys.exit(main())
