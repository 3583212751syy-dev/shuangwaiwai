"""v173 eagle_2 选 A：主体保原姿态 + 装饰大换（骷髅/闪电/蛇 → 黑曜石/水晶/钢链）

v171/v172 主体换姿态在该全图 img2img 管线硬止损（多鹰重叠）。回退 A：
- 鹰 = 居中正面+展翅（与原图姿态一致，主体铁律保）
- 装饰母题大换（让「裂变更大」的目标通过周围元素差异化达成）
- 参数中等：denoise 0.82 / tile 0.55 / canny 0.25 / region_scale 0.70
- 输出 3 路对照：原图 | v172 翻车 | v173 选 A
"""
import time, requests, sys, os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.82
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.2

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.55
REGION_STRENGTH_SCALE = 0.70

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
    "second eagle, duplicate eagle, two eagles, multiple eagles, twin eagles, paired eagles, "
    "second head, duplicate head, mirror head, phantom figure, ghost figure, second bird"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

REFS = [
    # === eagle_2（v173 选 A：主体保原姿态 + 装饰大换）===
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "global_pos": ("gothic tattoo illustration, pure black background, "
                       "red and orange flames, white and silver eagle, gray iron, "
                       "bold t-shirt graphic print, high contrast, sharp edges, "
                       "no text, no letters, no words, no banner, no inscription anywhere, "
                       "EXACTLY ONE bald eagle centered facing the camera, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            # 主体：鹰（保原姿态——居中正面+展翅）
            {"x": 0.30, "y": 0.00, "w": 0.40, "h": 0.55, "strength": 1.30,
             "prompt": ("EXACTLY ONE HUGE bald eagle centered FACING THE CAMERA, "
                        "wings spread wide symmetrically outward, "
                        "head slightly tilted down with piercing orange-yellow eyes "
                        "and half-open yellow beak, "
                        "white head feathers with sharp hatch lines, "
                        "brown BODY feathers with bold geometric block-shading, "
                        "yellow legs with sharp talons gripping forward, "
                        "fierce intense front-facing stare, this is the ONLY eagle. " + COHESIVE)},
            # 装饰（原骷髅位置）→ 熔岩裂缝黑曜石
            {"x": 0.25, "y": 0.50, "w": 0.50, "h": 0.40, "strength": 1.25,
             "prompt": ("a CLEARLY VISIBLE massive cracked OBSIDIAN monolith with glowing molten LAVA veins "
                        "splitting through it from top to bottom, "
                        "the fissure glowing hot orange and red, "
                        "CLEARLY DISTINGUISHABLE from the eagle above, "
                        "jagged black rock shards flaking off the edges, "
                        "no skull, no bone, no human face. " + COHESIVE)},
            # 装饰（原左侧闪电）→ 白色水晶簇
            {"x": 0.00, "y": 0.20, "w": 0.22, "h": 0.65, "strength": 1.20,
             "prompt": ("a burst of 7 sharp WHITE CRYSTAL QUARTZ shards CLEARLY VISIBLE against the pure black background, "
                        "fanning out from the eagle's left wing tip, "
                        "each shard long and faceted with hard geometric edges, "
                        "pure white with crisp outline, no gradient, no lightning bolt. " + COHESIVE)},
            # 装饰（原右侧闪电）→ 镜像水晶簇
            {"x": 0.78, "y": 0.20, "w": 0.22, "h": 0.65, "strength": 1.20,
             "prompt": ("a MIRROR burst of 7 sharp WHITE CRYSTAL QUARTZ shards CLEARLY VISIBLE on the right side, "
                        "symmetric to the left, faceted geometric edges, pure white. " + COHESIVE)},
            # 装饰（原蛇位置）→ 钢链框
            {"x": 0.10, "y": 0.90, "w": 0.80, "h": 0.10, "strength": 1.00,
             "prompt": ("a row of heavy interlocking STEEL CHAIN links curving along the bottom edge "
                        "like a metallic crown frame, "
                        "each link a thick gray iron oval with bolt details, "
                        "industrial brutalist feel, no serpent, no snake. " + COHESIVE)},
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v173_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v173_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在 {out.stat().st_size/1024/1024:.1f}MB，跳过", flush=True); return True
    if not ref.get("regions"):
        print(f"  [{tag}] 无 regions 配置，跳过", flush=True); return True

    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v173_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {json.dumps(j)[:1500]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {tag}: 无 prompt_id {str(j)[:400]}", flush=True); return False
    print(f"  [{tag}] pid={pid} running (denoise={DENOISE}, tile={TILE_STRENGTH}, canny={CANNY_STRENGTH}, region_scale={REGION_STRENGTH_SCALE})...", flush=True)
    for i in range(90):
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
                        print(f"  [{tag}] 完成 -> {out} ({out.stat().st_size/1024/1024:.1f}MB)", flush=True)
                        return True
                    else:
                        print(f"  [{tag}] 完成但无图", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] 轮询异常 {e}", flush=True)
        if i % 12 == 11:
            print(f"  [{tag}] 已等待 {(i+1)*5}s ...", flush=True)
    print(f"  [{tag}] 超时未完成", flush=True); return False


def make_3way_compare(orig_path, mid_path, new_path, out_path, mid_label, new_label):
    from PIL import Image, ImageDraw, ImageFont
    a = Image.open(orig_path).convert("RGB")
    b = Image.open(mid_path).convert("RGB")
    c = Image.open(new_path).convert("RGB")
    H = 1024
    def fit(im):
        w, h = im.size
        return im.resize((int(w * H / h), H))
    a2, b2, c2 = fit(a), fit(b), fit(c)
    gap = 20
    W = a2.width + b2.width + c2.width + gap * 2
    canvas = Image.new("RGB", (W, H + 60), (20, 20, 20))
    x = 0
    canvas.paste(a2, (x, 60)); x += a2.width + gap
    canvas.paste(b2, (x, 60)); x += b2.width + gap
    canvas.paste(c2, (x, 60))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        f = ImageFont.load_default()
    def lab(text, x, w):
        d.text((x + w // 2 - len(text) * 7, 16), text, fill=(220, 220, 220), font=f)
    lab("原图", 0, a2.width)
    lab(mid_label, a2.width + gap, b2.width)
    lab(new_label, a2.width + gap + b2.width + gap, c2.width)
    canvas.save(out_path)
    print(f"  3 路对照图 -> {out_path}", flush=True)


if __name__ == "__main__":
    out_base = PROJECT_ROOT / "outputs" / "v173"
    out_base.mkdir(parents=True, exist_ok=True)
    ref = REFS[0]
    ok = gen(ref, SEED, out_base)
    if ok:
        orig = PROJECT_ROOT / "ComfyUI" / "input" / ref["ref_img"]
        new = out_base / f"v173_{ref['id']}.jpg"
        v172_path = PROJECT_ROOT / "outputs" / "v172" / f"v172_{ref['id']}.jpg"
        if orig.exists() and new.exists() and v172_path.exists():
            make_3way_compare(orig, v172_path, new,
                              out_base / f"v173_{ref['id']}_3way.jpg",
                              "v172 翻车 6.5/10", "v173 选 A")
    print("DONE", flush=True)
