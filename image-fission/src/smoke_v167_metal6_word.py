"""v167 metal_6 直接裂变 + 单词也裂变

按用户 2026-08-29 最终指令："不要 polygon，就直接将整体内容裂变，单词也裂变"。
- 复用 v164 的 v147 技术管线（ProteusV0.4 / IPA 0.18 style / add-detail-xl / Tile0.60 + Canny0.25 / 双 KSampler / 4x）
- 与 v164 的区别：不再"禁文字"，而是新增一个【顶部 death-metal logo 区域】，
  prompt 直接让 AI 把单词 SKULLWING（骷髅+翼，扣图元素，0 侵权）以原图 death-metal logo 美术形态裂变出来。
- 风格由 IPA（style transfer）从原图 metal_6 携带（原图自带 MRCHGSR death-metal logo），
  所以裂变出的单词会天然带原图那种刺冠/尖刺 logo 形态，而不是标准字体。
- 字母拼写由 SDXL 在裂变中生成（即"单词也被裂变"），AI 尽量贴近 SKULLWING。
"""
import time, requests, sys, json
from pathlib import Path
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700406
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.2

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

# 注意：这里去掉了 v164 NEG_BASE 里的 text/letters/words 抑制项，让单词能裂变出来
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

# metal_6 配置：主体保留（鹰+角骷髅+闪电），顶部新增 death-metal logo 单词区域
REF = {
    "id": "metal_6", "ref_img": "pinterest_metal_6.jpg",
    "global_pos": ("brutal death metal band illustration, pure black background, "
                   "white eagle, brown feather shading, white skull with brown horns, "
                   "high contrast sharp extreme detail, bold graphic print, "
                   "extreme line weight contrast, harsh jagged edges, "
                   "a death-metal band logo word across the top in spiky gothic lettering"),
    "regions": [
        # === 顶部 death-metal logo 单词（裂变出来，拼 SKULLWING）===
        {"x": 0.02, "y": 0.00, "w": 0.96, "h": 0.16, "strength": 1.40,
         "prompt": ("a BRUTAL death-metal band LOGO TEXT across the very top, spiky blackletter gothic "
                    "lettering spelling 'SKULLWING', bold white letters with thick black outline, "
                    "every letter topped with sharp triangular spikes and bottom barbs, "
                    "extreme high-contrast, reading clearly, "
                    "the signature jagged crown-shaped death-metal wordmark style. " + COHESIVE)},
        # 主体鹰
        {"x": 0.30, "y": 0.10, "w": 0.40, "h": 0.40, "strength": 1.30,
         "prompt": ("a HUGE bald eagle FACING THE CAMERA with wings half-spread downward, "
                    "both shoulder blades visible, "
                    "head slightly tilted down with piercing orange-yellow eyes and half-open yellow beak, "
                    "white head feathers with sharp hatch lines, "
                    "brown BODY feathers with very bold geometric block-shading, "
                    "yellow legs with sharp talons gripping forward, "
                    "fierce intense stare. " + COHESIVE)},
        # 角骷髅
        {"x": 0.25, "y": 0.45, "w": 0.50, "h": 0.40, "strength": 1.30,
         "prompt": ("a large white human skull DIRECTLY BELOW the eagle, "
                    "mouth wide open in a roar (no helmet on), "
                    "FOUR long curved horns sprouting from the skull crown, "
                    "two main horns curling outward and up to either side, "
                    "two secondary shorter horns rising straight up between them, "
                    "brown horns with rough texture. " + COHESIVE)},
        # 左侧闪电荆棘
        {"x": 0.00, "y": 0.15, "w": 0.25, "h": 0.75, "strength": 1.15,
         "prompt": ("a burst of 7 sharp white metal lightning spikes fanning out from "
                    "the eagle-skull junction on the left side, "
                    "spikes of varying lengths the longest 1.5x the shortest, "
                    "each spike has 3-4 short perpendicular barbs, "
                    "pure white with crisp outline, no gradient. " + COHESIVE)},
        # 右侧闪电荆棘
        {"x": 0.75, "y": 0.15, "w": 0.25, "h": 0.75, "strength": 1.15,
         "prompt": ("a MIRROR burst of 7 sharp white metal lightning spikes on the right side, "
                    "symmetric to the left, "
                    "matching length variation. " + COHESIVE)},
        # 底部羽毛笔触装饰带
        {"x": 0.10, "y": 0.88, "w": 0.80, "h": 0.12, "strength": 1.00,
         "prompt": ("a thin band of 5 sharp downward white spikes at the very bottom edge, "
                    "like a jagged maw of teeth, "
                    "each spike 3x as tall as wide, "
                    "creates a metallic crown frame for the whole design. " + COHESIVE)},
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v167_{ref['id']}"}}
    return g


def main():
    out = PROJECT_ROOT / "jobs" / "smoke_v167"
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"v167_metal_6.jpg"

    g = build(REF, SEED)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v167_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {r.status_code} {json.dumps(j)[:1500]}", flush=True); return
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] 无 prompt_id {str(j)[:400]}", flush=True); return
    print(f"[metal_6] pid={pid} running...", flush=True)
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
                        data = requests.get(url, timeout=60).content
                        out_file.write_bytes(data)
                        try:
                            im = Image.open(out_file).convert('RGB')
                            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                            sharp.save(out_file, 'JPEG', quality=95, optimize=True)
                        except Exception as e:
                            print(f"  USM失败 {e}", flush=True)
                        print(f"[metal_6] OK {out_file.stat().st_size/1024/1024:.1f}MB -> {out_file}", flush=True)
                        return
                elif rec.get("status", {}).get("error"):
                    print(f"[ERR] COMFY错误 {str(rec['status'].get('error'))[:600]}", flush=True); return
        except Exception as e:
            print(f"  轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"  {i*5}s...", flush=True)
    print("[TIMEOUT]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
