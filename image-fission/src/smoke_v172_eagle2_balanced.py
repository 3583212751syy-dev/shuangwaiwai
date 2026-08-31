"""v172 eagle_2 平衡版：装饰大换 + 主体换姿态，但参数收更紧 + 强防多鹰

vs v171（翻车 6/10）的修正：
- DENOISE 0.88 -> 0.82（不再让原图 latent 完全被噪声覆盖）
- TILE_STRENGTH 0.45 -> 0.55（锁回纹理，防止主体复数化）
- CANNY_STRENGTH 0.20 -> 0.22（保留布局/位置，让姿态变化受控）
- REGION_STRENGTH_SCALE 0.85 -> 0.70（区域不抢主体）
- 关键：加「SINGLE eagle only」强约束（区域提示 + global_pos + NEG 三重保险）
- 装饰母题措辞加强「CLEARLY VISIBLE」便于在羽毛背景中分离出来

主体姿态保留 v171 的换姿态设计（3/4 侧脸 + 收翅 + 抓破齿轮），但约束到只一只鹰。
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
CANNY_STRENGTH = 0.22
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
    # v172 新增：防多鹰（v171 翻车根因）
    "second eagle, duplicate eagle, two eagles, multiple eagles, twin eagles, paired eagles, "
    "second head, duplicate head, mirror head, phantom figure, ghost figure, second bird"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

REFS = [
    # === eagle_2（v172 平衡版：装饰大换 + 主体换姿态 + 防多鹰）===
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "global_pos": ("gothic tattoo illustration, pure black background, "
                       "red and orange flames, white and silver eagle, gray iron, "
                       "bold t-shirt graphic print, high contrast, sharp edges, "
                       "no text, no letters, no words, no banner, no inscription anywhere, "
                       "EXACTLY ONE bald eagle in the entire image, NO second eagle, "
                       "no duplicate eagle, no twin eagle, no paired eagle, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            # 主体：唯一一只鹰（3/4 侧脸 + 收翅 + 抓破齿轮）— 强 SINGLE 约束
            {"x": 0.30, "y": 0.00, "w": 0.40, "h": 0.50, "strength": 1.40,
             "prompt": ("EXACTLY ONE SINGLE bald eagle, NO second eagle anywhere, "
                        "in a dramatic 3/4 SIDE PROFILE pose viewed from the right, "
                        "head turned sharply to its left with a hooked yellow beak shown in PROFILE, "
                        "ONE piercing orange-yellow eye visible, "
                        "BOTH WINGS TUCKED and SWEPT UPWARD into a sharp chevron behind the back, "
                        "wing tips pointing UP and BEHIND, NOT spread downward, "
                        "talons extended downward CLUTCHING a shattered broken metal GEAR, "
                        "white head and tail feathers with bold hatch-line shading, brown body, "
                        "fierce intense profile silhouette, "
                        "this is the ONLY eagle in the image, no other eagle head or body. " + COHESIVE)},
            # 装饰：熔岩裂缝黑曜石（强调 CLEARLY VISIBLE）
            {"x": 0.25, "y": 0.45, "w": 0.50, "h": 0.40, "strength": 1.25,
             "prompt": ("a CLEARLY VISIBLE massive cracked OBSIDIAN monolith with glowing molten LAVA veins "
                        "splitting through it from top to bottom, "
                        "the fissure glowing hot orange and red, "
                        "CLEARLY DISTINGUISHABLE from the eagle above, "
                        "jagged black rock shards flaking off the edges, "
                        "no skull, no bone, no human face. " + COHESIVE)},
            # 装饰：左侧白色水晶簇（强调 CLEARLY VISIBLE / 远离羽毛）
            {"x": 0.00, "y": 0.20, "w": 0.22, "h": 0.65, "strength": 1.20,
             "prompt": ("a burst of 7 sharp WHITE CRYSTAL QUARTZ shards CLEARLY VISIBLE against the pure black background, "
                        "fanning out from the eagle's left shoulder, "
                        "each shard long and faceted with hard geometric edges, "
                        "pure white with crisp outline, no gradient, no lightning bolt. " + COHESIVE)},
            # 装饰：右侧镜像水晶簇
            {"x": 0.78, "y": 0.20, "w": 0.22, "h": 0.65, "strength": 1.20,
             "prompt": ("a MIRROR burst of 7 sharp WHITE CRYSTAL QUARTZ shards CLEARLY VISIBLE on the right side, "
                        "symmetric to the left, faceted geometric edges, pure white. " + COHESIVE)},
            # 装饰：底部钢链框
            {"x": 0.10, "y": 0.88, "w": 0.80, "h": 0.12, "strength": 1.00,
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v172_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v172_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在 {out.stat().st_size/1024/1024:.1f}MB，跳过", flush=True); return True
    if not ref.get("regions"):
        print(f"  [{tag}] 无 regions 配置，跳过", flush=True); return True

    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v172_{int(time.time())}"}, timeout=15)
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


def make_3way_compare(orig_path, mid_path, new_path, out_path):
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
    lab("v171 翻车 6/10", a2.width + gap, b2.width)
    lab("v172 平衡版", a2.width + gap + b2.width + gap, c2.width)
    canvas.save(out_path)
    print(f"  3 路对照图 -> {out_path}", flush=True)


if __name__ == "__main__":
    out_base = PROJECT_ROOT / "outputs" / "v172"
    out_base.mkdir(parents=True, exist_ok=True)
    ref = REFS[0]
    ok = gen(ref, SEED, out_base)
    if ok:
        orig = PROJECT_ROOT / "ComfyUI" / "input" / ref["ref_img"]
        new = out_base / f"v172_{ref['id']}.jpg"
        # v171 6/10 翻车版也在 outputs/v171/，用于 3 路对照
        v171_path = PROJECT_ROOT / "outputs" / "v171" / f"v171_{ref['id']}.jpg"
        if orig.exists() and new.exists() and v171_path.exists():
            make_3way_compare(orig, v171_path, new, out_base / f"v172_{ref['id']}_3way.jpg")
        elif orig.exists() and new.exists():
            from PIL import Image, ImageDraw, ImageFont
            a = Image.open(orig).convert("RGB")
            b = Image.open(new).convert("RGB")
            H = 1024
            a2 = a.resize((int(a.width * H / a.height), H))
            b2 = b.resize((int(b.width * H / b.height), H))
            canvas = Image.new("RGB", (a2.width + b2.width + 20, H + 60), (20, 20, 20))
            canvas.paste(a2, (0, 60)); canvas.paste(b2, (a2.width + 20, 60))
            d = ImageDraw.Draw(canvas)
            try: f = ImageFont.truetype("arial.ttf", 28)
            except Exception: f = ImageFont.load_default()
            d.text((20, 16), "原图", fill=(220, 220, 220), font=f)
            d.text((a2.width + 40, 16), "v172 平衡版", fill=(220, 220, 220), font=f)
            canvas.save(out_base / f"v172_{ref['id']}_compare.jpg")
    print("DONE", flush=True)
