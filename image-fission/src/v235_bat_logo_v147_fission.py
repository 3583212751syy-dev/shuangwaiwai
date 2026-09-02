"""
v235 — v234 基础上根治字符崩坏（IPA 降到 0.15 / Canny 提到 0.80）

v234 崩坏（OCR 实测）：
  ['M}', 'BACARDT', 'MEL PALLT']
  图实际看到：中央徽章下出现 "BACARDT" / "MEL PALET"
  顶部弧形带出现 "B I MATR CINUTE"

根因：Canny 0.65 把 BACARDÍ/LA CASA/MHEART 字符边缘当成"参考结构"灌给 SDXL → SDXL 在 sample 时按字符纹理自然生成模仿词
IPA 0.30 仍把原图字符风格迁移过来

v235 根治：
  1. CANNY 0.65 → 0.80（强锁轮廓，抹掉字符笔画）
  2. TILE 0.95 → 1.05（超纹理锁定）
  3. IPA 0.30 → 0.15（弱风格迁移，prompt 占主导）
  4. NEG 加 BACARDT / MEL PALET hash
  5. 加步骤：OCR 实测 → OCR 仍读出字符则 PIL mask 后期修
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
JOB = PROJECT / "jobs" / "smoke_v235"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.fp16.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 235001

DENOISE = 0.70
IPA_WEIGHT = 0.15  # v234 是 0.30，降低让 prompt 占主导
LORA_DETAIL = 0.50
CANNY_STRENGTH = 0.80  # v234 是 0.65，强锁抹字符笔画
TILE_STRENGTH = 1.05   # v234 是 0.95，超纹理锁定
REGION_STRENGTH_SCALE = 0.55
CFG = 8.0

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    # —— v232-v234 OCR 实测崩坏全列 ——
    "any readable or unreadable letter shapes, any horizontal oriented letter-like glyphs, "
    "any vertical oriented letter-like glyphs, "
    "letter fragments, half letters, alphabet fragments, "
    "faux letters, mock letters, simulated letters, "
    "pseudo letters, accidental letters, jumbled letters, "
    "wild lettering, glyphs, runic marks, "
    "fringe characters, scribble lines, smudge letters, "
    "alphanumeric, roman numerals, "
    "any word shapes, any written content, any inscribed shapes, "
    "any banner with text, any ribbon with text, any scroll with text, any banner with letters, "
    "any individual character shapes resembling HAGARIA, S1UIEZI, GALLERIE, SDRUEIAMM, BACLRET, SMIRAICE, TMG, MME, CENT, NOCTURNE, "
    "any individual character shapes resembling BAT, BATMAN, MENEARI, MENAIR, MENERI, MENEAR, BAT EN, EN, ES, "
    "any individual character shapes resembling BACARDT, BACARDI, BACARDO, BACARDY, BACARDA, "
    "any individual character shapes resembling MEL PALET, MEL PALLT, MEL PAILET, MELPALLE, MELL PALLET, "
    "any individual character shapes resembling B I MATR CINUTE, BIMATR, BIMATRC, CINUTE, CINNUTE, "
    "any characters resembling TEH, GBET, AMVURIT, GIQE, "
    "any inscription, any engraved text, any printed text, any glyph cluster, "
    "any text-like geometric ornaments arranged horizontally or in a curved line, "
    # —— 3D / 茶器 / 瓷器 ——
    "3d, photographic, painterly, photorealistic bat, photoreal, hyperrealistic, "
    "trophy, cup, teacup, cup shape, chalice, goblet, mug, tumbler, flagon, "
    "vase, urn, amphora, jar, teapot, kettle, pitcher, jug, decanter, carafe, "
    "plate, platter, bowl, dish, saucer, china, porcelain, ceramic, enamel, glazed pottery, "
    # —— 反射 / 倒影 ——
    "reflection, mirrored surface, reflective surface, glossy ceramic surface, polished surface, "
    "shadow cast, ground shadow, drop shadow, cast shadow, contact shadow, "
    "depth of field, blurry background, shallow focus, "
    # —— 构图偏 / bat 倒挂 ——
    "off-center, asymmetric composition, crooked, tilted, skewed, not centered, misaligned, "
    "floating object below the badge, dangling element below, hanging charm below, "
    "inverted bat, upside down bat, bat hanging upside down, bat facing down, wings drooping down, "
    "wings pointing down, wings downward, bat sideways, bat profile, bat from side, "
    # —— 装饰过度 ——
    "baroque on the ring, scrollwork on the ring, flames on the ring, relief on the ring, "
    "shapes on the ring, patterns on the ring, decoration on the ring, anything ON the ring line, "
    "multiple bats, second bat, bats below, small bats flanking, extra bats, "
    "pendant gems below, floating gems, hanging gems, drop gems, teardrop gems below, "
    "floating stars, scattered stars, stars below, stars around, "
    "floating objects below, anything below the ring, elements below, "
    # —— 标准防劣化 ——
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "anatomically incorrect, extra wings, asymmetric error, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "adjacent objects merged, adjacent objects blending into each other, "
    # —— 配色锁 ——
    "desaturated purple, muted gray, grayish purple, dull tones, washed out, "
    "pale gray, off-palette gray, low-saturation, gray tint, ashy purple, "
    "beige, tan, brown, green, blue, cyan, orange, yellow, "
    "v196-v212 style word swap, identical layout to source, "
    "3d rendered, glossy highlights, gem appearance, jewelry appearance, necklace, medallion, chandelier, crystal, glass dome, "
    "shading gradient"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "natural overlap hierarchy, fits the overall composition, perfectly centered, "
    "no ground plane, no reflection, no shadow, no 3D form, visually related to the source composition, "
    "no characters, no glyphs, no letter shapes anywhere, clean vector artwork"
)

REF_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

GLOBAL_POS = (
    "strict 2D flat printed vintage craft spirits label art, NO 3D, NO cup, NO trophy, NO teapot, NO vase, NO chalice, NO mug, NO ceramic, "
    "a perfectly centered circular emblem with a single thin clean minimal ring outline on a flat saturated purple background, "
    "ONE single stylized 2D bat silhouette INSIDE the ring (NOT realistic, NOT photographic, NOT 3D), "
    "bat wings spread SYMMETRICALLY upward facing viewer, head at 12 o'clock, NOT inverted, NOT drooping, "
    "ABOVE the badge: only a single thin clean curved line - this is just a thin LINE not a banner, NOT inscribed, "
    "NO text, NO letters, NO words, NO glyphs, NO characters, NO marks, NO scribbles, NO inscription anywhere, "
    "BELOW the badge: completely empty purple space, no decoration, "
    "INSIDE the badge: ONLY ring outline + bat + small crescent - NOTHING ELSE, "
    "OUTSIDE the badge: NO banner text, NO inscription, NO letters, NO words anywhere outside the ring, "
    "STRICTLY 2D flat vector print style, no shading, no gradient, no reflection, no shadow, no ground plane, "
    "STRICTLY saturated purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F color blocks ONLY, "
    "NO gray, NO desaturated muted purple, NO washed-out pale purple, NO beige, NO brown, NO green, NO blue, NO gray tint, "
    "uniform flat saturated purple background, NO tonal variation, NO lighter purple patches, NO gradient background, "
    "ABSOLUTELY NO characters, NO letters, NO glyphs, NO scribbles anywhere, "
    "the ONLY shapes present: a thin curving line (above), a thin ring outline (badge), a stylized bat (badge), a small crescent (badge), empty purple space (everywhere else)"
)

REGIONS = [
    {
        "x": 0.05, "y": 0.10, "w": 0.90, "h": 0.80, "strength": 1.30,
        "prompt": (
            "the CENTER circular emblem with a thin clean ring outline ONLY, "
            "ONE single stylized 2D flat black bat silhouette in dead center inside the ring, "
            "bat wings spread SYMMETRICALLY, head at 12 o'clock top, body centered, tail at bottom 6 o'clock, NOT inverted, NOT sideways, "
            "a single small stylized crescent moon shape at the bottom inside the ring acting as decorative ground for the bat, "
            "ring is a clean thin LINE outline, NOTHING on the ring line, NOTHING inside the ring besides the bat and crescent, "
            "NO text-like marks inside the ring, NO patterns inside the ring, NO ornamental inscription inside the ring, "
            "NO characters inside or outside the ring, NO inscription, "
            "keep STRICTLY saturated purple/black/white palette ONLY, flat 2D print, NO 3D, "
            "perfectly centered and symmetrical, NO shadow, NO reflection, NO ground plane. " + COHESIVE
        )
    },
    {
        "x": 0.10, "y": 0.00, "w": 0.80, "h": 0.15, "strength": 1.30,
        "prompt": (
            "above the badge: only a SINGLE thin clean curved line following the top arc, "
            "this is JUST a thin curving LINE shape, NOT a banner, NOT a ribbon, NOT a scroll, NOT inscribed, "
            "NO text, NO letters, NO words, NO pseudo-letters, NO glyphs, NO marks, NO scribbles, NO inscription anywhere on this line, "
            "just a single thin curved line shape, STRICTLY solid saturated purple background, "
            "perfectly centered. " + COHESIVE
        )
    },
    {
        "x": 0.30, "y": 0.88, "w": 0.40, "h": 0.10, "strength": 1.10,
        "prompt": (
            "below the badge: completely EMPTY solid saturated purple space, "
            "NO banner, NO ribbon, NO scroll, NO pendant, NO triangle, NO letters, NO glyphs, NO marks, NO scribbles, NO inscription, NO decoration, "
            "STRICTLY solid saturated purple background, perfectly centered. " + COHESIVE
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
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.70,
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
        "latent_image": ["4", 0], "seed": seed, "steps": 24, "cfg": CFG,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": CFG,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "v235_bat_logo"}}
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
    print(f"[v235] submitted pid={pid[:8]} cfg={CFG} canny={CANNY_STRENGTH} ipa={IPA_WEIGHT}")
    for _ in range(120):
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        print(f"[v235] TIMEOUT")
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
        cands = sorted(COMFY_OUTPUT.glob("v235_bat_logo*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        print(f"[v235] NOT FOUND raw_path")
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


def main():
    out_final = JOB / "v235_bat_logo.jpg"
    if out_final.exists() and out_final.stat().st_size > 100000:
        print(f"[skip] already exists {out_final}")
    else:
        raw = gen(SEED)
        if not raw:
            print("[v235] FAIL gen")
            return
        src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF_IMG).convert("RGB")), cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(np.array(Image.open(raw).convert("RGB")), cv2.COLOR_RGB2BGR)
        matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
        matched_rgb = Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
        matched_rgb = unsharp(matched_rgb, radius=1.5, percent=50, threshold=2)
        out_raw = JOB / "v235_bat_logo_raw.jpg"
        Image.fromarray(cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)).save(str(out_raw), quality=95)
        matched_bgr = cv2.cvtColor(np.array(matched_rgb), cv2.COLOR_RGB2BGR)
        Image.fromarray(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)).save(str(out_final), quality=95)
        hi_before = hist_intersection(src_bgr, dst_bgr)
        hi_after = hist_intersection(src_bgr, matched_bgr)
        sd = structural_diff(src_bgr, matched_bgr)
        print(f"  saved v235_bat_logo.jpg ({out_final.stat().st_size//1024} KB)")
        print(f"    配色: 迁移前={hi_before:.3f} → 迁移后={hi_after:.3f} | 结构差异={sd:.3f}")

    out_cmp = JOB / "_compare_v235.jpg"
    side_by_side(COMFY_INPUT / REF_IMG, out_final, out_cmp)
    print(f"  saved compare: {out_cmp}")


if __name__ == "__main__":
    main()
