"""
v228 — bat_logo 真裂变（v226 同源，配色锁加强消残留紫 flag）

v226 已干净 PASS（edge 1.12×），唯一残留 flag = 右半多一种同色相紫暗调 (163,109,156)（Reinhard 不锁 a/b，SDXL 生成的轻微色相漂移）。
v228 同 seed 220001 受控，仅加强配色锁（POS+NEG 双加 "uniform flat purple, NO tonal variation, NO lighter purple patch"），
试消灭该 false-positive flag，目标达成 100% 零 flag 的干净报告。构图/姿态/技术栈与 v226 完全一致。
"""
import os, sys, json, time, shutil, uuid, urllib.request
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageFilter

# ==================== 路径/常量 ====================
PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v228"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.fp16.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 220001  # 同 v220，受控对比

# ==================== v147 锁死基线（与 v220 一致） ====================
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 0.50
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

# ==================== NEG：v220 全禁词 + 配色锁 ====================
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    # —— 禁炸裂星形 ——
    "starburst, exploding star, spiky star, firework, sunburst, radiant spikes, sharp burst rays, "
    "comic burst, explosion shape, star shaped emblem, "
    # —— 禁可读字 / 单词含义 ——
    "legible text, readable text, recognizable letters, clear word, English word, brand name, logo word, "
    "spelled out words, specific brand word, recognizable brand, BACARDI, Mooncrest, Curse, NOCTURNE, "
    "TMG, MME, CENT, all readable word content, "
    # —— 禁 3D 珠宝 / 项链 / 吊坠 / 写实蝙蝠 ——
    "3d, photographic, painterly, photorealistic bat, photoreal, hyperrealistic, "
    "jewelry, necklace, pendant, pendant gem, teardrop, teardrop gem, hanging gem, drop shape, "
    "gem, gemstone, jewelry setting, earring, ring, brooch, locket, charm, dangling ornament, "
    "metal frame, gold frame, silver frame, 3d rendering, cgi render, "
    "depth of field, blurry background, shallow focus, "
    # —— 禁构图偏 / 不对称 / 底部悬挂物 ——
    "off-center, asymmetric composition, crooked, tilted, skewed, not centered, "
    "floating object below the badge, dangling element below, hanging charm below, "
    # —— 标准防劣化 ——
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "anatomically incorrect, extra wings, asymmetric error, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "new colors, different color palette, extra colors, color shift, "
    # —— v221 新增：配色锁，禁灰紫/低饱和 ——
    "desaturated purple, muted gray, grayish purple, dull tones, washed out, "
    "pale gray, off-palette gray, low-saturation, gray tint, ashy purple, "
    "beige, tan, brown, green, blue, cyan, orange, yellow, "
    "uneven purple tone, lighter purple patch, tonal gradient in background, two-tone purple, mixed purple shades, "
    "v196-v212 style word swap, identical layout to source"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "natural overlap hierarchy, fits the overall composition, perfectly centered"
)

# ==================== bat_logo 参考图 =================
REF_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

# ==================== Prompt 框架（v225：v221 + 更干净构图） ====================
GLOBAL_POS = (
    "a perfectly centered and symmetrical vintage gothic purple spirit brand emblem badge, "
    "occupying the full frame center on a soft purple background, "
    "purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F color blocks only, "
    "STRICTLY saturated purple / violet / black / white palette ONLY, "
    "NO gray, NO desaturated muted purple, NO grayish tones, NO washed-out pale purple, "
    "NO beige, NO brown, NO green, NO blue, NO gray tint, "
    "uniform flat saturated purple background, NO tonal variation, NO lighter purple patches, NO purple gradient, "
    "a 2D flat vintage circular badge with ONE bold evolved vector silhouette bat in the center, "
    "the bat is a single stylized 2D wing-spread silhouette in pure black (NOT realistic, NOT 3D, NOT photographic), "
    "a thin clean minimal geometric circular ring outline framing the badge as purely graphical motif "
    "(NO elaborate scrollwork, NO letterform-shaped flourishes, NOT forming any readable word, NOT spelling anything), "
    "the emblem ring is a thin clean minimal circular outline, very slightly rotated, minimal baroque hint, "
    "NO crescent moon, NO star, clean empty top above the ring, "
    "vintage craft spirits label art style, sharp clean edges, bold emblem, "
    "flat 2D illustration style, symmetric composition, "
    "CLEAN minimal bottom, NO beads, NO floating pendant below the badge, NO dangling charm, "
    "clean empty purple space below the badge, "
    "NO starburst, NO exploding star, NO firework, NO photographic, NO jewelry, NO pendant, NO teardrop"
)

REGIONS = [
    # 中心徽章：单只大胆剪影蝙蝠 + 干净巴洛克框（去 two-bats / 去 hanging beads）
    {
        "x": 0.05, "y": 0.10, "w": 0.90, "h": 0.80, "strength": 1.30,
        "prompt": (
            "a perfectly centered flat 2D vintage circular purple badge with ONE bold evolved vector silhouette bat inside, "
            "the bat is a single stylized 2D wing-spread silhouette in pure black (NOT photographic, NOT 3D, NOT realistic), "
    "a thin clean minimal circular ring outline framing the badge as purely abstract graphical motif "
    "(NO elaborate scrollwork, NOT readable letters, NOT spelling any word, "
    "NOT any brand name, minimal clean geometric frame), "
    "NO crescent moon, NO star, clean empty top, "
    "the emblem ring is a thin clean minimal circular outline, NO hanging beads, NO dangling charms, "
            "keep STRICTLY saturated purple/black/white palette, NO gray, NO desaturated muted purple, "
            "flat 2D illustration, perfectly centered and symmetrical, "
            "CLEAN minimal bottom, NO pendant, NO teardrop. " + COHESIVE
        )
    },
    # 顶部弧形带：装饰性图案带
    {
        "x": 0.10, "y": 0.00, "w": 0.80, "h": 0.15, "strength": 1.10,
        "prompt": (
    "a single thin clean arched line above the emblem top, "
    "minimal abstract pattern ONLY "
            "(NOT readable text, NOT letter-shaped wordmarks, NOT any word, NOT starburst), "
            "STRICTLY saturated purple and black color blocks with sharp edges and baroque flourishes, "
            "NO gray, NO desaturated purple, perfectly centered. " + COHESIVE
        )
    },
    # 底部徽章饰带：装饰性图案（minimal）
    {
        "x": 0.35, "y": 0.85, "w": 0.30, "h": 0.10, "strength": 0.90,
        "prompt": (
            "a small flat 2D bottom decorative element with minimal abstract ornamental pattern "
            "(NOT readable English words, NOT spelled-out words, NOT starburst, NOT jewelry, "
            "NOT pendant, NOT teardrop, NO hanging charm), kept STRICTLY saturated purple and black palette, "
            "NO gray, NO desaturated purple, flat 2D vintage style, perfectly centered, MINIMAL. " + COHESIVE
        )
    },
]


# ==================== build workflow ====================
def build(seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMG}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}

    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
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

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    region_nodes = []
    for i, r in enumerate(REGIONS):
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
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "v228_bat_logo"}}
    return g


# ==================== 后置: Reinhard LAB 配色迁移 + 轻 USM ====================
def color_transfer(src_bgr, dst_bgr, alpha=1.0):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = dst.copy()
    for i in range(3):
        s_mean, s_std = src[:, :, i].mean(), src[:, :, i].std() + 1e-6
        d_mean, d_std = dst[:, :, i].mean(), dst[:, :, i].std() + 1e-6
        out[:, :, i] = (dst[:, :, i] - d_mean) * (s_std / d_std) + s_mean
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if alpha >= 1.0:
        return out
    blended = dst.astype(np.float32) * (1 - alpha) + out.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def unsharp(rgb_img, radius=1.5, percent=50, threshold=2):
    return rgb_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def hist_intersection(src_bgr, dst_bgr, bins=32):
    inter = 0.0
    for c in range(3):
        hs = cv2.calcHist([src_bgr], [c], None, [bins], [0, 256])
        hd = cv2.calcHist([dst_bgr], [c], None, [bins], [0, 256])
        cv2.normalize(hs, hs); cv2.normalize(hd, hd)
        inter += cv2.compareHist(hs, hd, cv2.HISTCMP_INTERSECT)
    return inter / 3.0


def structural_diff(src_bgr, dst_bgr):
    h, w = min(src_bgr.shape[0], dst_bgr.shape[0]), min(src_bgr.shape[1], dst_bgr.shape[1])
    s = cv2.resize(src_bgr, (w, h)); d = cv2.resize(dst_bgr, (w, h))
    mse = ((s.astype(np.float32) - d.astype(np.float32)) ** 2).mean()
    return float(np.clip(1 - mse / (255.0 ** 2), 0, 1))


# ==================== ComfyUI API ====================
def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def history(pid):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
        return json.loads(r.read())


def gen(seed):
    g = build(seed)
    r = submit(g)
    pid = r["prompt_id"]
    print(f"[v228] submitted pid={pid[:8]} denoise={DENOISE} lora={LORA_DETAIL} (v221 + cleaner composition, same seed, controlled)")
    for _ in range(120):  # 10 分钟超时
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        print(f"[v228] TIMEOUT")
        return None
    outputs = h[pid].get("outputs", {})
    raw_path = None
    for node_id, info in outputs.items():
        for img in info.get("images", []):
            src = COMFY_OUTPUT / img["filename"]
            if not src.exists():
                src = Path(img.get("abs_path", src))
            if not src.exists():
                continue
            raw_path = str(src)
            break
        if raw_path:
            break
    if not raw_path:
        cands = sorted(COMFY_OUTPUT.glob("v228_bat_logo*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        print(f"[v228] NOT FOUND raw_path")
        return None
    return raw_path


def side_by_side(orig_path, gen_path, out_path):
    orig = Image.open(orig_path).convert("RGB")
    gen = Image.open(gen_path).convert("RGB")
    if orig.size != gen.size:
        gen = gen.resize(orig.size)
    out = Image.new("RGB", (orig.width * 2 + 30, orig.height), "white")
    out.paste(orig, (0, 0))
    out.paste(gen, (orig.width + 30, 0))
    out.save(out_path, quality=95)


def ocr_check(path):
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        res = reader.readtext(path)
        return [(t, round(c, 2)) for _, t, c in res]
    except Exception as e:
        return [("OCR_FAIL", str(e))]


def main():
    out_final = JOB / "v228_bat_logo.jpg"
    if out_final.exists() and out_final.stat().st_size > 100000:
        print(f"[skip] already exists {out_final}")
    else:
        raw = gen(SEED)
        if not raw:
            print("[v228] FAIL gen")
            return
        src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF_IMG).convert("RGB")), cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(np.array(Image.open(raw).convert("RGB")), cv2.COLOR_RGB2BGR)
        matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
        matched_rgb = Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
        matched_rgb = unsharp(matched_rgb, radius=1.5, percent=50, threshold=2)
        out_raw = JOB / "v228_bat_logo_raw.jpg"
        Image.fromarray(cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)).save(str(out_raw), quality=95)
        matched_bgr = cv2.cvtColor(np.array(matched_rgb), cv2.COLOR_RGB2BGR)
        Image.fromarray(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)).save(str(out_final), quality=95)
        hi_before = hist_intersection(src_bgr, dst_bgr)
        hi_after = hist_intersection(src_bgr, matched_bgr)
        sd = structural_diff(src_bgr, matched_bgr)
        print(f"  saved v228_bat_logo.jpg ({out_final.stat().st_size//1024} KB)")
        print(f"    配色: 迁移前={hi_before:.3f} → 迁移后={hi_after:.3f} | 结构差异={sd:.3f}")

    # 对照拼图
    out_cmp = JOB / "_compare_v228.jpg"
    side_by_side(COMFY_INPUT / REF_IMG, out_final, out_cmp)
    print(f"  saved compare: {out_cmp}")

    # OCR 自检
    txt = ocr_check(str(out_final))
    print(f"  OCR 读到的可读字: {txt}")


if __name__ == "__main__":
    main()
