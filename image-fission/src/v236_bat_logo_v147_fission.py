"""
v236 — v234 baseline + PIL mask 后处理根治字符崩坏

OCR 实证 v234 残留字符:
  ['M}', 'BACARDT', 'MEL PALLT']
  顶部有 "B I MATR CINUTE"

根因：v234 IPA 0.30 + Canny 0.65 已是最优 SDXL 输出，但 SDXL 在 "vintage spirit label" 视觉锚下，
总是习惯性填几个 pseudo-alphabet tokens，与 prompt 强禁词对决胜负各半 → 永远剩 5-15% 字符残留。
继续 prompt 工程化边际收益很低（v235 已证降低 IPA 反而恶化）。

v236 终极根治：保留 v234 SDXL 出图不变，后处理 OCR bbox 抹字 + 周边背景色融合
  1. SDXL 出 v234_bat_logo_raw.jpg（重生成或不重生成均可）
  2. easyocr 找出所有字符 bbox（XYXY 格式）
  3. 每个 bbox：扩大 25%，取周边 4 边背景色均值 fill 进去，inpaint 边缘自然
  4. 输出 v236_bat_logo.jpg
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
JOB = PROJECT / "jobs" / "smoke_v236"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.fp16.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 234001  # 与 v234 同 seed 受控（用 v234 的产出当 base）

# v234 出图路径
V234_JPG = PROJECT / "jobs" / "smoke_v234" / "v234_bat_logo.jpg"

# v234 baseline 参数（同 v234）
DENOISE = 0.70
IPA_WEIGHT = 0.30
LORA_DETAIL = 0.50
CANNY_STRENGTH = 0.65
TILE_STRENGTH = 0.95
REGION_STRENGTH_SCALE = 0.55
CFG = 8.0

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, characters, alphabet, "
    "any readable or unreadable letter shapes, any horizontal oriented letter-like glyphs, "
    "faux letters, mock letters, pseudo letters, accidental letters, jumbled letters, "
    "wild lettering, glyphs, runic marks, fringe characters, scribble lines, smudge letters, "
    "alphanumeric, roman numerals, any word shapes, any written content, any inscribed shapes, "
    "HAGARIA, S1UIEZI, GALLERIE, SDRUEIAMM, BACLRET, SMIRAICE, TMG, MME, CENT, NOCTURNE, "
    "BAT, BATMAN, MENEARI, BAT EN, EN, ES, BACARDT, BACARDI, BACARDO, "
    "MEL PALET, MEL PALLT, B I MATR CINUTE, BIMATR, CINUTE, "
    "any banner with text, any ribbon with text, any scroll with text, "
    "3d, photographic, painterly, photorealistic bat, photoreal, hyperrealistic, "
    "trophy, cup, teacup, cup shape, chalice, goblet, mug, tumbler, flagon, "
    "vase, urn, amphora, jar, teapot, kettle, pitcher, jug, decanter, carafe, "
    "plate, platter, bowl, dish, saucer, china, porcelain, ceramic, enamel, glazed pottery, "
    "reflection, mirrored surface, glossy ceramic surface, polished surface, "
    "shadow cast, ground shadow, drop shadow, cast shadow, contact shadow, "
    "depth of field, blurry background, shallow focus, "
    "off-center, asymmetric composition, crooked, tilted, skewed, not centered, misaligned, "
    "floating object below the badge, dangling element below, hanging charm below, "
    "inverted bat, upside down bat, bat facing down, wings drooping down, wings pointing down, "
    "baroque on the ring, scrollwork on the ring, flames on the ring, relief on the ring, "
    "shapes on the ring, patterns on the ring, decoration on the ring, anything ON the ring line, "
    "multiple bats, second bat, bats below, small bats flanking, extra bats, "
    "pendant gems below, floating gems, hanging gems, drop gems, teardrop gems below, "
    "floating stars, scattered stars, stars below, stars around, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "anatomically incorrect, extra wings, asymmetric error, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "desaturated purple, muted gray, grayish purple, dull tones, washed out, "
    "pale gray, off-palette gray, low-saturation, gray tint, ashy purple, "
    "beige, tan, brown, green, blue, cyan, orange, yellow, "
    "v196-v212 style word swap, identical layout to source, "
    "3d rendered, glossy highlights, gem appearance, jewelry appearance, necklace, medallion, "
    "shading gradient"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "perfectly centered, no ground plane, no reflection, no shadow, no 3D form, "
    "visually related to the source composition, clean vector artwork"
)

REF_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

GLOBAL_POS = (
    "strict 2D flat printed vintage craft spirits label art, NO 3D, NO cup, NO trophy, NO teapot, NO vase, NO chalice, NO mug, NO ceramic, "
    "a perfectly centered circular emblem with a single thin clean minimal ring outline on a flat saturated purple background, "
    "ONE single stylized 2D bat silhouette INSIDE the ring (NOT realistic, NOT photographic, NOT 3D), "
    "bat wings spread SYMMETRICALLY upward facing viewer, head at 12 o'clock, NOT inverted, NOT drooping, "
    "ABOVE the badge: only a single thin clean curved line following the top arc of the badge - this is a thin LINE not a banner, NOT a banner with text, NOT a ribbon, NOT an inscribed banner, NOT a decorative banner, just a thin curving line, "
    "BELOW the badge: completely empty purple space, no decoration, no elements, no marks, "
    "INSIDE the badge: ONLY the ring outline + the bat silhouette + a single small abstract crescent moon shape at the bottom near the bat - NOTHING ELSE inside the ring, NO decorative text-like marks inside, NO ornamental patterns inside, "
    "OUTSIDE the badge: NO banner text, NO ribbon text, NO inscription, NO letters, NO words anywhere outside the ring, "
    "STRICTLY 2D flat vector vintage craft spirits label print style, no shading, no gradient, no reflection, no shadow, no ground plane, "
    "STRICTLY saturated purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F color blocks ONLY, "
    "NO gray, NO desaturated muted purple, NO washed-out pale purple, NO beige, NO brown, NO green, NO blue, NO gray tint, "
    "uniform flat saturated purple background, NO tonal variation, NO lighter purple patches, NO gradient background, "
    "ABSOLUTELY NO characters, NO letters, NO glyphs, NO scribbles anywhere in the entire image"
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
            "keep STRICTLY saturated purple/black/white palette ONLY, flat 2D print, NO 3D, "
            "perfectly centered and symmetrical, NO shadow, NO reflection, NO ground plane. " + COHESIVE
        )
    },
    {
        "x": 0.10, "y": 0.00, "w": 0.80, "h": 0.15, "strength": 1.25,
        "prompt": (
            "above the badge: a single thin clean curved line following the top arc of the badge, "
            "this is JUST a thin curving LINE shape, NOT a banner, NOT a ribbon, NOT a scroll, NOT an inscribed banner, "
            "NO text, NO letters, NO words, NO pseudo-letters, NO glyphs, NO marks, NO scribbles, NO decoration on this line, "
            "just a single thin curved line shape following the arc, STRICTLY solid saturated purple background, "
            "perfectly centered. " + COHESIVE
        )
    },
    {
        "x": 0.30, "y": 0.88, "w": 0.40, "h": 0.10, "strength": 1.00,
        "prompt": (
            "below the badge: completely EMPTY solid saturated purple space, "
            "NO banner, NO ribbon, NO scroll, NO pendant, NO triangle, NO drop shape, NO letters, NO glyphs, NO marks, NO scribbles, NO decoration, "
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "v236_bat_logo"}}
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
    print(f"[v236-SDXL] submitted pid={pid[:8]} cfg={CFG} canny={CANNY_STRENGTH} (v234 baseline 重生成)")
    for _ in range(120):
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        print(f"[v236-SDXL] TIMEOUT")
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
        cands = sorted(COMFY_OUTPUT.glob("v236_bat_logo*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        print(f"[v236-SDXL] NOT FOUND raw_path")
        return None
    return raw_path


def ocr_erase_text(image_path, out_path, expand=0.25, min_area=200):
    """
    OCR 检测字符位置 → 用周边背景色填充抹除
    用于清除 SDXL 不自觉生成的 pseudo-alphabet tokens
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    res = reader.readtext(bgr, decoder='beamsearch')
    res += reader.readtext(bgr, decoder='greedy')

    # 收集所有 bbox（去重：IoU>0.5 合并）
    boxes = []
    seen = []
    for (_, t, c) in res:
        if c < 0.10:
            continue
        # res 是 [[box, text, conf]] box 是 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        pass
    # 重新调一次拿到 box
    res_b = []
    for dec in ['beamsearch', 'greedy']:
        r = reader.readtext(bgr, decoder=dec)
        for item in r:
            if len(item) >= 3:
                res_b.append((item[0], item[1], float(item[2])))

    for box_pts, text, conf in res_b:
        if conf < 0.10:
            continue
        pts = np.array(box_pts).astype(int)
        x1 = pts[:, 0].min(); x2 = pts[:, 0].max()
        y1 = pts[:, 1].min(); y2 = pts[:, 1].max()
        bw = x2 - x1; bh = y2 - y1
        if bw * bh < min_area:
            continue
        # 扩 bbox 25%
        px = int(bw * expand); py = int(bh * expand)
        x1 = max(0, x1 - px); y1 = max(0, y1 - py)
        x2 = min(w, x2 + px); y2 = min(h, y2 + py)

        # 跳过明显是 bat 剪影的大区域(>1500px 高的单 box)
        if (x2 - x1) > 0.6 * w or (y2 - y1) > 0.4 * h:
            print(f"  [OCR-skip] too big ({x2-x1}x{y2-y1}) for '{text}'")
            continue
        boxes.append((x1, y1, x2, y2, text))

    print(f"  [OCR] found {len(boxes)} char cluster(s)")
    for x1, y1, x2, y2, t in boxes:
        print(f"    bbox=({x1},{y1},{x2},{y2}) text='{t}'")

    if not boxes:
        img.save(out_path, quality=95)
        return 0

    # 取每个 bbox 周边背景色 (环外 +1.5× 区域)
    for x1, y1, x2, y2, t in boxes:
        ring_pixels = []
        rh, rw = (y2 - y1), (x2 - x1)
        # 上下左右各取 rh/2 / rw/2 距离外的背景区域
        for sy, ey in [(0, max(0, y1 - rh)), (min(h, y2 + rh), h)]:
            if ey > sy:
                ring_pixels.append(arr[sy:ey, max(0, x1-rw):min(w, x2+rw)])
        for sx, ex in [(0, max(0, x1 - rw)), (min(w, x2 + rw), w)]:
            if ex > sx:
                ring_pixels.append(arr[max(0, y1-rh):min(h, y2+rh), sx:ex])
        if not ring_pixels:
            continue
        bg = np.concatenate([p.reshape(-1, 3) for p in ring_pixels if p.size > 0], axis=0)
        if len(bg) < 100:
            continue
        # k-means (k=1) 取主色
        from collections import Counter
        bg_q = (bg // 8) * 8  # 量化到 8-step 减少噪声
        bg_tuples = [tuple(p) for p in bg_q]
        most_common = Counter(bg_tuples).most_common(1)[0][0]
        fill_color = np.array(most_common, dtype=np.uint8)
        # 用周边色均值更稳健
        fill_color = np.median(bg, axis=0).astype(np.uint8)

        # cv2.inpaint 周边区域
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        # 给 mask 一点 feathering
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        # inpaint (TELEA mode, radius 5)
        bgr_inpainted = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
        # 颜色向背景微调
        # 写入回 bgr
        bgr = bgr_inpainted

    out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out.save(out_path, quality=95)
    return len(boxes)


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
    out_final = JOB / "v236_bat_logo.jpg"
    if out_final.exists() and out_final.stat().st_size > 100000:
        print(f"[skip] already exists {out_final}")
    else:
        # 路线 A: 复用 v234 已有 raw,只做 OCR 抹字
        if V234_JPG.exists():
            print(f"[v236] 复用 v234 baseline 做 OCR 抹字")
            n_erased = ocr_erase_text(str(V234_JPG), str(out_final))
        else:
            # 路线 B: 重跑 SDXL
            print(f"[v236] 重跑 SDXL 然后 OCR 抹字")
            raw = gen(SEED)
            if not raw:
                print("[v236] FAIL gen")
                return
            src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF_IMG).convert("RGB")), cv2.COLOR_RGB2BGR)
            dst_bgr = cv2.cvtColor(np.array(Image.open(raw).convert("RGB")), cv2.COLOR_RGB2BGR)
            matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
            matched_rgb = Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
            matched_rgb = unsharp(matched_rgb, radius=1.5, percent=50, threshold=2)
            tmp_path = JOB / "v236_sdxl_before_ocr.jpg"
            matched_rgb.save(str(tmp_path), quality=95)
            n_erased = ocr_erase_text(str(tmp_path), str(out_final))

        print(f"[v236] erased {n_erased} char cluster(s)")

    out_cmp = JOB / "_compare_v236.jpg"
    side_by_side(COMFY_INPUT / REF_IMG, out_final, out_cmp)
    print(f"  saved compare: {out_cmp}")

    # 量化 QC
    src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF_IMG).convert("RGB")), cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(np.array(Image.open(out_final).convert("RGB")), cv2.COLOR_RGB2BGR)
    if src_bgr.shape != dst_bgr.shape:
        dst_bgr = cv2.resize(dst_bgr, (src_bgr.shape[1], src_bgr.shape[0]))
    hi = hist_intersection(src_bgr, dst_bgr)
    sd = structural_diff(src_bgr, dst_bgr)
    print(f"  QC: 配色 hist={hi:.3f} | 结构差异={sd:.3f}")

    # OCR 验最终
    print("  OCR final:")
    img = Image.open(out_final).convert("RGB")
    arr = np.array(img)
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    for dec in ['beamsearch', 'greedy']:
        r = reader.readtext(arr, decoder=dec)
        print(f"    {dec}: {[(t, round(c, 2)) for _, t, c in r if float(c) > 0.10]}")


if __name__ == "__main__":
    main()
