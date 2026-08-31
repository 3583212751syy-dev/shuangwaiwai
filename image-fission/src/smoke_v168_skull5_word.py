"""v168 skull_5 整体裂变 + 单词也裂变（v167 同款架构）

按用户 2026-08-29 终态指令：原图 logo "TRUE / NEVER / DIES" 是真品牌引用，
- 1:1 复刻三行红色血滴哥特衬线美术形态
- 字母替换为 BONE / BLOOM / ASH（与图元素隐喻同构：骨/玫瑰盛开/燃尽，0 侵权）
- 元素全部由 AI 重画：戴眼罩血污骷髅 + 血红双翼（合区对称）+ 红蛇 + 血玫瑰
- 风格由 IPA 从原图 skull_5 携带，裂变图天然继承黑底血红哥特纹身质感
"""
import time, requests, sys, json
from pathlib import Path
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700501
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.2

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.saf16.safetensors".replace(".fp16.safetensors", ".safetensors")  # 容错
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

# 与 v167 一致：去掉 v164 的 text/letters/words 抑制项，让单词能裂变
NEG_WORD = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
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

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

# skull_5 配置：戴眼罩血污骷髅 + 血红双翼 + 红蛇 + 血玫瑰（保留主体+数量），
# 顶部 / 中下 / 底部三行红色血滴哥特衬线单词 BONE / BLOOM / ASH
REF = {
    "id": "skull_5", "ref_img": "pinterest_skull_5.jpg",
    "global_pos": ("gothic tattoo illustration on pure black background, "
                   "white skull with deep cracks and dried blood splatter, "
                   "blood-red ceremonial angel wings, red scaled serpent, dark blood-red rose, "
                   "bold RED blood-drip gothic serif lettering as part of the tattoo design, "
                   "high contrast, sharp edges, cohesive dark composition, "
                   "extreme line weight contrast, harsh jagged edges, fabric-print quality"),
    "regions": [
        # ① 顶部单词 BONE（1:1 复刻原 TRUE 行的位置 + 红色血滴哥特衬线美术形态）
        {"x": 0.10, "y": 0.02, "w": 0.80, "h": 0.10, "strength": 1.40,
         "prompt": ("the word 'BONE' in bold RED blood-drip gothic SERIF lettering across the very top, "
                    "crimson red letters with thick dark-red outline, "
                    "blood drips dripping DOWN from the bottoms of each letter, "
                    "sharp classical serifs, red on pure black, extreme high-contrast, "
                    "reading clearly as B-O-N-E. " + COHESIVE)},
        # ② 血红双翼（左右合并为对称对，节省 1 个 region 位给三行单词）
        {"x": 0.02, "y": 0.10, "w": 0.96, "h": 0.40, "strength": 1.25,
         "prompt": ("a PERFECTLY SYMMETRIC pair of blood-red CEREMONIAL angel wings spreading from behind the skull crown, "
                    "one wing filling the upper-left and one filling the upper-right, "
                    "both TALL and NARROW with long sharp pointed feathers, "
                    "5 distinct feather rows separated by hard black gaps, "
                    "deep crimson red with darker shading along feather spines, "
                    "left wing tip pointing upper-left corner, right wing tip pointing upper-right corner, "
                    "mirror-symmetric pair. " + COHESIVE)},
        # ③ 戴眼罩血污骷髅（保留主体 + 眼罩 + 裂纹 + 血污）
        {"x": 0.30, "y": 0.28, "w": 0.40, "h": 0.40, "strength": 1.35,
         "prompt": ("ONE large forward-facing human skull at the center, "
                    "realistic cracks across the forehead and cheekbones, "
                    "a single black leather EYE PATCH covering the right eye socket, "
                    "dried blood splatter across the cheek bone and forehead, "
                    "bleached white bone with realistic gray shadow in eye sockets, "
                    "mouth slightly open showing teeth, no hat no helmet no crown, just bone and patch. " + COHESIVE)},
        # ④ 血红玫瑰（左侧贴面颊，保留主体 + 数量 1 大朵 + 散落花瓣）
        {"x": 0.04, "y": 0.50, "w": 0.26, "h": 0.32, "strength": 1.10,
         "prompt": ("ONE large dark blood-red rose on the left side near the skull cheek, "
                    "fully bloomed with layered petals and visible thorns on the stem, "
                    "a few scattered loose blood-red petals falling downward, "
                    "deep crimson only, no white no pink. " + COHESIVE)},
        # ⑤ 红蛇缠绕（保留主体，改缠绕方式：颈部→头骨下方 S 型）
        {"x": 0.25, "y": 0.62, "w": 0.50, "h": 0.22, "strength": 1.15,
         "prompt": ("a blood-red SCALED SERPENT wrapping around the BOTTOM of the skull, "
                    "coiled in a tight S-shape with the body crossing the chin twice, "
                    "head lifted on the right side facing the skull's mouth, "
                    "forked tongue flicking outward toward the skull teeth, "
                    "red scale texture clearly visible, forked tongue in darker crimson. " + COHESIVE)},
        # ⑥ 中下单词 BLOOM（1:1 复刻原 NEVER 行的位置 + 红色血滴哥特衬线美术形态）
        {"x": 0.10, "y": 0.78, "w": 0.80, "h": 0.10, "strength": 1.40,
         "prompt": ("the word 'BLOOM' in bold RED blood-drip gothic SERIF lettering across the lower-middle, "
                    "matching the top word exactly in style and color, "
                    "crimson red letters with thick dark-red outline, blood drips from letter bottoms, "
                    "sharp classical serifs, red on pure black, reading clearly as B-L-O-O-M. " + COHESIVE)},
        # ⑦ 底部单词 ASH（1:1 复刻原 DIES 行的位置 + 红色血滴哥特衬线美术形态）
        {"x": 0.25, "y": 0.90, "w": 0.50, "h": 0.09, "strength": 1.40,
         "prompt": ("the word 'ASH' in bold RED blood-drip gothic SERIF lettering across the very bottom, "
                    "matching the top word exactly in style and color, "
                    "crimson red letters with thick dark-red outline, blood drips from letter bottoms, "
                    "sharp classical serifs, red on pure black, reading clearly as A-S-H. " + COHESIVE)},
    ],
}


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
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_WORD}}

    region_nodes = []
    for i, r in enumerate(ref["regions"]):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        st = r["strength"] * REGION_STRENGTH_SCALE
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": st}}
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v168_{ref['id']}"}}
    return g


def main():
    out = PROJECT_ROOT / "jobs" / "smoke_v168"
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"v168_skull_5.jpg"

    g = build(REF, SEED)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v168_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {r.status_code} {json.dumps(j)[:1500]}", flush=True); return
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] 无 prompt_id {str(j)[:400]}", flush=True); return
    print(f"[skull_5] pid={pid} running...", flush=True)
    for i in range(150):  # 放宽到 750s=12.5min (v168 7区域+三单词+4x超分 比 v167 慢)
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                rec = h[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
                    if imgs:
                        url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                        data = requests.get(url, timeout=60).content
                        out_file.write_bytes(data)
                        try:
                            im = Image.open(out_file).convert('RGB')
                            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                            sharp.save(out_file, 'JPEG', quality=95, optimize=True)
                        except Exception as e:
                            print(f"  USM失败 {e}", flush=True)
                        print(f"[skull_5] OK {out_file.stat().st_size/1024/1024:.1f}MB -> {out_file}", flush=True)
                        return
                elif rec.get("status", {}).get("error"):
                    print(f"[ERR] COMFY错误 {str(rec['status'].get('error'))[:600]}", flush=True); return
        except Exception as e:
            print(f"  轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"  {i*5}s...", flush=True)
    print("[TIMEOUT]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
