"""
v242 — bat_logo 真裂变（按 v147 baseline 重做，给用户看真裂变版）

= 起因 =
用户 17:48 反馈 v212 PIL 烧字版"跟原图又没区别了，知道什么叫裂变图吗"——
v210-v212 都是"擦 BACARDI → PIL 烧新字"，本质是文字替换不是裂变，
构图/配色/风格/结构都跟原图一字不差，根本不叫裂变。

= v242 路线 =
完全复用 v232 的 v147 真裂变 baseline (commit 8285e76 用户硬指令版)：
  • SDXL Proteus v0.4 全图重画 (不是 PIL 烧字)
  • IPAdapter PLUS high-strength style transfer (0.30 锁风格关联)
  • Canny 0.65 (锁轮廓防 v230 bat 倒挂)
  • Tile 0.95 (锁纹理防 v229 跑题 3D)
  • Reinhard LAB 1:1 紫色锁 (history 0.601 → 迁移后 1.342)
  • NEG 全方位拦截崩坏（字符/3D/容器/反射/构图/配色漂移）
  • 字作为装饰图形，不 PIL 烧字（v193 哲学 + 规则 4）

= 用户硬指令 16:16 四条 v242 严格执行 =
  1) 结构保留     —— 三层布局：顶部弧形带 / 中间圆形徽章含 bat / 底部饰带
  2) 主体方向一致 —— bat 居中朝上展翅
  3) 风格保留     —— 2D flat printed vintage craft spirits label
  4) 颜色不改     —— 紫色调 1:1 锁死 (Reinhard LAB)

v242 vs v232 只换 seed (232001 → 242001)，让 SDXL 出不一样的新徽章。
v232 跑出的 HAGARTA bat badge 已存档证明 v147 真裂变生效。
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
JOB = PROJECT / "jobs" / "smoke_v242"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.fp16.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 242001

# v147 锁基线 + 强化锁定
DENOISE = 0.70        # v229 是 0.80, 略降防过度漂移
IPA_WEIGHT = 0.30     # v229 是 0.18, 略升保风格关联
LORA_DETAIL = 0.50
CANNY_STRENGTH = 0.65  # v147 0.25, v232 提到 0.65 锁轮廓（防 v230 bat 倒挂）
TILE_STRENGTH = 0.95   # v147 0.60, v232 提到 0.95 锁纹理（防 v229 跑题 3D）
REGION_STRENGTH_SCALE = 0.55

# NEG = 全方位拦截 v229-v231 崩坏
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    # —— 禁字符（v230/v231 都出过）——
    "legible text, readable text, recognizable letters, clear word, English word, brand name, logo word, "
    "spelled out words, specific brand word, recognizable brand, BACARDI, Mooncrest, Curse, NOCTURNE, "
    "TMG, MME, CENT, all readable word content, "
    "wild lettering, pseudo letters, accidental letters, fragmented letters, jumbled letters, "
    "letter fragments, half letters, nonsense text, scrambled letters, "
    "random characters, random squiggles, accidental squiggles, glyphs, runic marks, "
    "illegible scribbles, fringe characters, scribble lines, smudge letters, "
    "ribbon banner with text, scroll with text, ribbon with words, "
    # —— 禁 3D / 容器 / 茶器 / 瓷器（v229 跑题 3D 茶杯）——
    "3d, photographic, painterly, photorealistic bat, photoreal, hyperrealistic, "
    "trophy, cup, teacup, cup shape, chalice, goblet, mug, tumbler, flagon, "
    "vase, urn, amphora, jar, teapot, kettle, pitcher, jug, decanter, carafe, "
    "plate, platter, bowl, dish, saucer, "
    "china, porcelain, ceramic, enamel, glazed pottery, ceramic mug, ceramic plate, "
    # —— 禁反射 / 倒影（v229 也出过）——
    "reflection, mirrored surface, reflective surface, glossy ceramic surface, polished surface, "
    "shadow cast, ground shadow, drop shadow, cast shadow, contact shadow, "
    "depth of field, blurry background, shallow focus, "
    # —— 禁构图偏 / 不对称 —
    "off-center, asymmetric composition, crooked, tilted, skewed, not centered, misaligned, "
    "floating object below the badge, dangling element below, hanging charm below, "
    "object casting reflection on ground, ground plane, floor, table top, "
    # —— 禁 bat 倒挂 / 翅膀朝下（v230 出现过）——
    "inverted bat, upside down bat, bat hanging upside down, bat facing down, wings drooping down, "
    "wings pointing down, wings downward, bat sideways, bat profile, bat from side, "
    # —— 标准防劣化 —
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "anatomically incorrect, extra wings, asymmetric error, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "adjacent objects merged, adjacent objects blending into each other, "
    # —— 配色锁（用户硬指令：颜色不改）——
    "desaturated purple, muted gray, grayish purple, dull tones, washed out, "
    "pale gray, off-palette gray, low-saturation, gray tint, ashy purple, "
    "beige, tan, brown, green, blue, cyan, orange, yellow, "
    "v196-v212 style word swap, identical layout to source, "
    # —— v229 三大崩坏禁词 —
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
    "no ground plane, no reflection, no shadow, no 3D form, visually related to the source composition"
)

REF_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

# GLOBAL PROMPT —— 严格保留原图结构 (三层布局 + 圆环 + bat + 月牙 + 三角饰带)
GLOBAL_POS = (
    "a vintage craft spirits label illustration with the same THREE-LAYER composition as the source, "
    "TOP LAYER: a thin curved arched banner above the circular badge filled with stylized abstract geometric ornaments (NOT letters, NOT readable text), "
    "MIDDLE LAYER: a circular emblem with a thin clean ring outline containing ONE single stylized 2D bat silhouette centered inside, "
    "a small abstract crescent moon directly behind the bat at 6 o'clock inside the ring (decorative graphic element, NOT a separate object), "
    "BOTTOM LAYER: a small flat triangular decorative pennant below the badge with stylized abstract geometric ornaments (NOT letters), "
    "STRICTLY 2D flat vector vintage craft spirits label print style, no shading, no gradient, no reflection, no shadow, no ground plane, "
    "the bat is a 2D flat printed black bat silhouette facing the viewer with wings spread symmetrically upward, head at top 12 o'clock, NOT inverted, NOT drooping, NOT sideways, "
    "STRICTLY saturated purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F color blocks ONLY, "
    "NO gray, NO desaturated muted purple, NO washed-out pale purple, NO beige, NO brown, NO green, NO blue, NO gray tint, "
    "uniform flat saturated purple background, NO tonal variation, NO lighter purple patches, NO gradient background, "
    "NO letters, NO text, NO words, NO pseudo-letters, NO letter fragments, NO scribbles, NO glyphs, NO marks, NO squiggles, NO random characters ANYWHERE, "
    "sharp clean edges, bold emblem, symmetric composition, perfectly centered, same composition as the source illustration"
)

REGIONS = [
    # 中心徽章：ring + bat + 月牙
    {
        "x": 0.05, "y": 0.10, "w": 0.90, "h": 0.80, "strength": 1.30,
        "prompt": (
            "the CENTER circular emblem with a thin clean ring outline as the same abstract frame, "
            "ONE single stylized 2D flat black bat silhouette in dead center inside the ring, "
            "bat wings spread SYMMETRICALLY upward with both wing-tips touching the upper ring interior, "
            "bat head at 12 o'clock top of ring, body centered, tail at bottom 6 o'clock, NOT inverted, NOT sideways, NOT drooping, "
            "a single small stylized crescent moon shape at the bottom inside the ring acting as decorative ground for the bat (NOT a separate object outside), "
            "ring is a clean thin LINE outline, no decoration on ring, no ornament on ring line, "
            "CLEAN minimal interior, NO gems, NO stars, NO bats besides the main one, NO floating objects inside, "
            "keep STRICTLY saturated purple/black/white palette ONLY, flat 2D print, NO 3D, NO trophy, NO cup, NO ceramic, NO vase, "
            "perfectly centered and symmetrical, NO shadow, NO reflection, NO ground plane, NO ground shadow. " + COHESIVE
        )
    },
    # 顶部弧形带
    {
        "x": 0.10, "y": 0.00, "w": 0.80, "h": 0.15, "strength": 1.10,
        "prompt": (
            "the TOP curved arched banner above the circular badge, same position as source, "
            "filled with stylized abstract geometric ornaments and small abstract dots and shapes "
            "(NOT letters, NOT text, NOT words, NOT pseudo-letters, NOT letter fragments, NOT logos, NOT monograms, NOT readable), "
            "an abstract decorative banner matching the source style, "
            "STRICTLY saturated purple and black color blocks, perfectly centered. " + COHESIVE
        )
    },
    # 底部三角饰带
    {
        "x": 0.30, "y": 0.88, "w": 0.40, "h": 0.10, "strength": 1.00,
        "prompt": (
            "the BOTTOM small triangular decorative pennant below the badge, same position as source, "
            "filled with stylized abstract geometric ornaments and small abstract shapes "
            "(NOT letters, NOT text, NOT words, NOT pseudo-letters, NOT letter fragments, NOT logos, NOT readable), "
            "a small flat triangle shape, STRICTLY saturated purple and black color blocks, perfectly centered, MINIMAL. " + COHESIVE
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
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.75,
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "v242_bat_logo"}}
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
    print(f"[v242] submitted pid={pid[:8]} denoise={DENOISE} canny={CANNY_STRENGTH} tile={TILE_STRENGTH} ipa={IPA_WEIGHT}")
    for _ in range(120):
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        print(f"[v232] TIMEOUT")
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
        cands = sorted(COMFY_OUTPUT.glob("v242_bat_logo*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        print(f"[v232] NOT FOUND raw_path")
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
    out_final = JOB / "v242_bat_logo.jpg"
    if out_final.exists() and out_final.stat().st_size > 100000:
        print(f"[skip] already exists {out_final}")
    else:
        raw = gen(SEED)
        if not raw:
            print("[v232] FAIL gen")
            return
        src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF_IMG).convert("RGB")), cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(np.array(Image.open(raw).convert("RGB")), cv2.COLOR_RGB2BGR)
        matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
        matched_rgb = Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
        matched_rgb = unsharp(matched_rgb, radius=1.5, percent=50, threshold=2)
        out_raw = JOB / "v242_bat_logo_raw.jpg"
        Image.fromarray(cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)).save(str(out_raw), quality=95)
        matched_bgr = cv2.cvtColor(np.array(matched_rgb), cv2.COLOR_RGB2BGR)
        Image.fromarray(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)).save(str(out_final), quality=95)
        hi_before = hist_intersection(src_bgr, dst_bgr)
        hi_after = hist_intersection(src_bgr, matched_bgr)
        sd = structural_diff(src_bgr, matched_bgr)
        print(f"  saved v232_bat_logo.jpg ({out_final.stat().st_size//1024} KB)")
        print(f"    配色: 迁移前={hi_before:.3f} → 迁移后={hi_after:.3f} | 结构差异={sd:.3f}")

    out_cmp = JOB / "_compare_v242.jpg"
    side_by_side(COMFY_INPUT / REF_IMG, out_final, out_cmp)
    print(f"  saved compare: {out_cmp}")

    txt = ocr_check(str(out_final))
    print(f"  OCR 读到的可读字: {txt}")


if __name__ == "__main__":
    main()
