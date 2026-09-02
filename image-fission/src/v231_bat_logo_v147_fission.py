"""
v231 — bat_logo 真裂变（v147 锁基线 + v230 prompt 强化 + canny strength 0.55 锁轮廓）

v230 跑出来还是不够对：出现 "SDRUEIAMM GA FFHI" 等乱字符，bat 倒挂。
v231 修复策略：
  - canny_strength 从 0.25 提到 0.55（构图轮廓强制锁紧）
  - tile_strength 从 0.60 提到 0.85（整体结构强化）
  - 加 start at 0.0 / end at 0.7 让 ipa 早收
  - NEG 加 "wild lettering", "random characters", "illegible squiggles"
  - 徽章 prompt 强化 "bat wings pointing UPWARD" 强制方向
  - 顶部 + 底部区域均禁止出 "任何线条/任何文字"
  - bat 主体中心居中、展翅左右对称、翅膀末端离 ring 等距
"""
import os, sys, json, time, shutil, uuid, urllib.request
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageFilter

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v231"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.fp16.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 231001

# v147 锁基线 + canny 强化
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 0.50
CANNY_STRENGTH = 0.55  # v230 是 0.25, v231 提到 0.55 锁轮廓
TILE_STRENGTH = 0.85   # v230 是 0.60, v231 提到 0.85 锁结构
REGION_STRENGTH_SCALE = 0.55

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "legible text, readable text, recognizable letters, clear word, English word, brand name, logo word, "
    "spelled out words, specific brand word, recognizable brand, BACARDI, Mooncrest, Curse, NOCTURNE, "
    "TMG, MME, CENT, all readable word content, "
    "wild lettering, pseudo letters, accidental letters, fragmented letters, jumbled letters, "
    "letter fragments, half letters, nonsense text, scrambled letters, "
    "random characters, random squiggles, accidental squiggles, glyphs, runic marks, "
    "illegible scribbles, fringe characters, scribble lines, smudge letters, "
    "ribbon banner with text, scroll with text, ribbon with words, "
    "3d, photographic, painterly, photorealistic bat, photoreal, hyperrealistic, "
    "trophy, cup, teacup, cup shape, chalice, goblet, mug, tumbler, flagon, "
    "vase, urn, amphora, jar, teapot, kettle, pitcher, jug, decanter, carafe, "
    "plate, platter, bowl, dish, saucer, "
    "china, porcelain, ceramic, enamel, glazed pottery, ceramic mug, ceramic plate, "
    "reflection, mirrored surface, reflective surface, glossy ceramic surface, polished surface, "
    "shadow cast, ground shadow, drop shadow, cast shadow, contact shadow, "
    "depth of field, blurry background, shallow focus, "
    "off-center, asymmetric composition, crooked, tilted, skewed, not centered, misaligned, "
    "floating object below the badge, dangling element below, hanging charm below, "
    "object casting reflection on ground, ground plane, floor, table top, "
    "inverted bat, upside down bat, bat hanging upside down, bat facing down, wings drooping down, "
    "wings pointing down, wings downward, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "anatomically incorrect, extra wings, asymmetric error, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "new colors, different color palette, extra colors, color shift, "
    "desaturated purple, muted gray, grayish purple, dull tones, washed out, "
    "pale gray, off-palette gray, low-saturation, gray tint, ashy purple, "
    "beige, tan, brown, green, blue, cyan, orange, yellow, "
    "v196-v212 style word swap, identical layout to source, "
    "baroque on the ring, scrollwork on the ring, flames on the ring, relief on the ring, "
    "shapes on the ring, patterns on the ring, decoration on the ring, anything ON the ring line, "
    "multiple bats, second bat, bats below, small bats flanking, extra bats, "
    "pendant gems below, floating gems, hanging gems, drop gems, teardrop gems below, "
    "floating stars, scattered stars, stars below, stars around, "
    "floating objects below, anything below the ring, elements below, "
    "3d rendered, glossy highlights, gem appearance, jewelry appearance, necklace, medallion, chandelier, crystal, glass dome, "
    "shading gradient, glossy surface, polished surface"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "natural overlap hierarchy, fits the overall composition, perfectly centered, "
    "no ground plane, no reflection, no shadow, no 3D form"
)

REF_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

GLOBAL_POS = (
    "strict 2D flat printed vintage craft spirits label art, NO 3D, NO cup, NO trophy, NO teapot, NO vase, NO chalice, NO mug, "
    "NO ceramic, NO porcelain, NO trophy cup, NO goblet, NO metal container, NO bowl, NO jar, NO scroll, NO ribbon, "
    "a perfectly centered circular emblem with a single thin clean minimal ring outline on a flat saturated purple background, "
    "ONE single stylized 2D bat silhouette INSIDE the ring (NOT realistic, NOT photographic, NOT 3D), "
    "the bat wings point SYMMETRICALLY UPWARD and SPREAD EQUAL distance to the ring, bat facing 12 o'clock, "
    "STRICTLY 2D flat vector vintage craft spirits label print style, no shading, no gradient, no reflection, no shadow, no ground plane, "
    "STRICTLY saturated purple / violet / black / white palette ONLY, "
    "NO gray, NO desaturated muted purple, NO washed-out pale purple, NO beige, NO brown, NO green, NO blue, NO gray tint, "
    "uniform flat saturated purple background, NO tonal variation, NO lighter purple patches, NO gradient background, "
    "NO letters, NO text, NO words, NO pseudo-letters, NO letter fragments, NO scribbles, NO glyphs, NO marks, NO squiggles, NO random characters ANYWHERE, "
    "sharp clean edges, bold emblem, symmetric composition, perfectly centered"
)

REGIONS = [
    {
        "x": 0.05, "y": 0.10, "w": 0.90, "h": 0.80, "strength": 1.30,
        "prompt": (
            "a perfectly centered flat 2D circular printed emblem with ONE single bold 2D black bat silhouette centered inside the ring, "
            "the bat wings spread SYMMETRICALLY with one wing-tip to the left ring edge and the other wing-tip to the right ring edge, both wing-tips at equal distance from the ring, "
            "bat head pointing UP toward 12 o'clock, wings CLEARLY visible spreading outward in 2D flat printed style, NOT drooping, NOT hanging, NOT inverted, "
            "a thin clean minimal ring outline as a pure abstract graphical frame — NO decoration on the ring, NO lettering on the ring, NO baroque, NO scrollwork, NO flame, NO relief, NO shapes ON the ring, "
            "NO elements below the ring, NO gems, NO pendants, NO bats below, NO stars, NO floating objects, "
            "keep STRICTLY saturated purple/black/white palette ONLY, flat 2D print, NO 3D, NO trophy, NO cup, NO ceramic, NO vase, "
            "perfectly centered and symmetrical, NO shadow, NO reflection, NO ground plane, NO ground shadow, "
            "CLEAN minimal bottom (empty purple space). " + COHESIVE
        )
    },
    {
        "x": 0.10, "y": 0.00, "w": 0.80, "h": 0.15, "strength": 1.10,
        "prompt": (
            "completely EMPTY purple space above the badge, NO arched band, NO curved line, NO banner, NO ribbon, NO scroll, NO ring on top, NO border band, "
            "NO letters, NO text, NO words, NO pseudo-letters, NO letter fragments, NO logo, NO monogram, NO glyphs, NO random characters, NO marks, NO squiggles, "
            "NO decoration, NO wreath, NO laurel, NO filigree, NO crescent, "
            "STRICTLY saturated purple background, perfectly centered. " + COHESIVE
        )
    },
    {
        "x": 0.35, "y": 0.85, "w": 0.30, "h": 0.10, "strength": 0.90,
        "prompt": (
            "completely EMPTY purple space below the badge, NO banner, NO ribbon, NO scroll, NO pendant, NO triangle, NO drop shape, NO laurel, "
            "NO letters, NO words, NO logos, NO gems, NO stars, NO floating object, NO crescent, NO monogram, NO glyphs, NO marks, NO random characters, "
            "STRICTLY saturated purple background, perfectly centered. " + COHESIVE
        )
    },
]


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
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.70,  # 早收
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "v231_bat_logo"}}
    return g


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
    print(f"[v231] submitted pid={pid[:8]} denoise={DENOISE} canny={CANNY_STRENGTH} (v230 + canny 锁轮廓)")
    for _ in range(120):
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        print(f"[v231] TIMEOUT")
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
        cands = sorted(COMFY_OUTPUT.glob("v231_bat_logo*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        print(f"[v231] NOT FOUND raw_path")
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
    out_final = JOB / "v231_bat_logo.jpg"
    if out_final.exists() and out_final.stat().st_size > 100000:
        print(f"[skip] already exists {out_final}")
    else:
        raw = gen(SEED)
        if not raw:
            print("[v231] FAIL gen")
            return
        src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF_IMG).convert("RGB")), cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(np.array(Image.open(raw).convert("RGB")), cv2.COLOR_RGB2BGR)
        matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
        matched_rgb = Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
        matched_rgb = unsharp(matched_rgb, radius=1.5, percent=50, threshold=2)
        out_raw = JOB / "v231_bat_logo_raw.jpg"
        Image.fromarray(cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)).save(str(out_raw), quality=95)
        matched_bgr = cv2.cvtColor(np.array(matched_rgb), cv2.COLOR_RGB2BGR)
        Image.fromarray(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)).save(str(out_final), quality=95)
        hi_before = hist_intersection(src_bgr, dst_bgr)
        hi_after = hist_intersection(src_bgr, matched_bgr)
        sd = structural_diff(src_bgr, matched_bgr)
        print(f"  saved v231_bat_logo.jpg ({out_final.stat().st_size//1024} KB)")
        print(f"    配色: 迁移前={hi_before:.3f} → 迁移后={hi_after:.3f} | 结构差异={sd:.3f}")

    out_cmp = JOB / "_compare_v231.jpg"
    side_by_side(COMFY_INPUT / REF_IMG, out_final, out_cmp)
    print(f"  saved compare: {out_cmp}")

    txt = ocr_check(str(out_final))
    print(f"  OCR 读到的可读字: {txt}")


if __name__ == "__main__":
    main()
