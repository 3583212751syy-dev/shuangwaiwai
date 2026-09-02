"""
v243 — bat_logo 真裂变(用户 17:57 二次修正)
= 两个明确投诉 =
1) 主体蝙蝠要更像原图  -> 用【原图蝙蝠剪影提取+仿射变换换角度】作 Canny 姿态硬约束
   (蝙蝠就是原图蝙蝠旋转后重绘, 材质/形态 100% 保留) + IPA 提到 0.40 锁材质/紫调
2) 文字糊/缺字依然存在 -> 不再让 SDXL 生字(必糊必缺), 改为【PIL 弧字烧字】
   (arc_text.py + PirataOne 哥特字体, 矢量清晰渲染绝不缺笔) 烧回干净非侵权词

架构: 去蝙蝠Canny(姿态参考=原图蝙蝠旋转) + IPAdapter PLUS 0.40 + Tile + Reinhard LAB 紫锁
+ OCR 抹除 SDXL 残余字 + arc_text 烧回清晰新词(NOCTWING, 蝙蝠隐喻, 非 BACARDÍ 侵权词)
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
JOB = PROJECT / "jobs" / "smoke_v243"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.fp16.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

DENOISE = 0.70
IPA_WEIGHT = 0.40
LORA_DETAIL = 0.50
CANNY_STRENGTH = 0.65
TILE_STRENGTH = 0.95
REGION_STRENGTH_SCALE = 0.55
CFG = 8.0

MASK_REGION = (0.24, 0.12, 0.56, 0.48)
CENTER_KEEP = MASK_REGION

REF_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"
MASKED_REF = "v243_masked_ref.png"

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, characters, alphabet, "
    "any readable or unreadable letter shapes, any horizontal oriented letter-like glyphs, "
    "any vertical oriented letter-like glyphs, "
    "faux letters, mock letters, pseudo letters, accidental letters, jumbled letters, "
    "wild lettering, glyphs, runic marks, fringe characters, scribble lines, smudge letters, "
    "alphanumeric, roman numerals, any word shapes, any written content, any inscribed shapes, "
    "HAGARIA, S1UIEZI, GALLERIE, SDRUEIAMM, BACLRET, SMIRAICE, TMG, MME, CENT, NOCTURNE, "
    "BAT, BATMAN, MENEARI, BAT EN, EN, ES, BACARDT, BACARDI, BACARDO, "
    "MEL PALET, MEL PALLT, B I MATR CINUTE, BIMATR, CINUTE, "
    "any banner with text, any ribbon with text, any scroll with text, "
    "internal lines on the bat, letter-like curves on the wings, scrollwork inside the bat silhouette, patterns inside the bat, fine line details resembling text on the bat, "
    "3d, photographic, painterly, photorealistic bat, photoreal, hyperrealistic, "
    "trophy, cup, teacup, cup shape, chalice, goblet, mug, tumbler, flagon, "
    "vase, urn, amphora, jar, teapot, kettle, pitcher, jug, decanter, carafe, "
    "plate, platter, bowl, dish, saucer, china, porcelain, ceramic, enamel, glazed pottery, "
    "reflection, mirrored surface, glossy ceramic surface, polished surface, "
    "shadow cast, ground shadow, drop shadow, cast shadow, contact shadow, "
    "depth of field, blurry background, shallow focus, "
    "off-center, asymmetric composition, crooked, tilted, skewed, not centered, misaligned, "
    "floating object below the badge, dangling element below, hanging charm below, "
    "inverted bat, upside down bat, bat hanging upside down, bat facing down, wings drooping down, "
    "wings pointing down, wings downward, bat sideways profile, "
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
    "shading gradient, duplicate of source pose"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "perfectly centered, no ground plane, no reflection, no shadow, no 3D form, "
    "visually related to the source composition, clean vector artwork"
)

# 两段【轻微】姿态(像原图, 但明显是裂变: 同源蝙蝠/向上感/2D平面紫锁, 只换小角度)
# (tag, GLOBAL_POS 蝙蝠行, REGION[0] 蝙蝠行)
POSES = [
    ("a_bank",
     "the bat is gently BANKED and tilted to its RIGHT - its entire body rotated about 20 degrees from the source, the LEFT wing slightly RAISED and the RIGHT wing slightly LOWERED, the head turned a little toward the viewer; a subtle dynamic turning pose, clearly a NEW angle from the source, but unmistakably the SAME bat species and clearly related to the source bat, NOT inverted, NOT drooping, NOT a flat symmetric copy of the source pose, "
     "the bat is a CLEAN smooth solid silhouette with NO internal lines, NO letter-like curves, NO scrollwork, NO patterns inside, smooth wing membranes only, ",
     "the bat is in a NEW but subtle pose: gently BANKED to its RIGHT, body rotated ~20 degrees, LEFT wing up slightly and RIGHT wing down slightly, head turned a little toward viewer - a related VARIANT pose of the source heraldic bat, NOT a flat front-on symmetric copy, NOT inverted, NOT drooping, "
     "the bat is a CLEAN smooth solid black silhouette, NO internal detail lines, NO letter-like shapes on the wings, NO scrollwork inside the silhouette, smooth wing membranes, "),
    ("b_tilt",
     "the bat is gently TILTED and mirrored to its LEFT side - body rotated about 18 degrees the other way, wings slightly asymmetric with a natural turn, head looking slightly outward; a subtle alternative angle, clearly a NEW pose from the source, but unmistakably the SAME bat species and clearly related to the source bat, NOT inverted, NOT drooping, NOT a flat symmetric copy of the source pose, "
     "the bat is a CLEAN smooth solid silhouette with NO internal lines, NO letter-like curves, NO scrollwork, NO patterns inside, smooth wing membranes only, ",
     "the bat is in a NEW but subtle pose: gently TILTED to its LEFT side, body rotated ~18 degrees, wings slightly asymmetric, head looking slightly outward - a related VARIANT pose of the source heraldic bat, NOT a flat front-on symmetric copy, NOT inverted, NOT drooping, "
     "the bat is a CLEAN smooth solid black silhouette, NO internal detail lines, NO letter-like shapes on the wings, NO scrollwork inside the silhouette, smooth wing membranes, "),
]

SEEDS = {"a_bank": 243001, "b_tilt": 243002}

# 从原图提取真实蝙蝠剪影, 用仿射变换换【小】角度, 作为 Canny 姿态硬约束
# (rot=度数, flip_h/flip_v=镜像, scale=缩放) —— 同源物种、画风材质不变, 只换小角度(像原图)
BAT_BOX = (0.28, 0.16, 0.52, 0.46)   # 提取蝙蝠本体的内框(避开顶部弧线文字)
POSE_XFORM = {
    "a_bank": (20, False, False, 1.00),   # 向右轻倾 20°
    "b_tilt": (-18, True, False, 1.00),   # 向左轻倾 18° + 镜像(另一种角度)
}


def make_masked_ref():
    img = Image.open(COMFY_INPUT / REF_IMG).convert("RGB")
    arr = np.array(img).astype(np.int32)
    h, w = arr.shape[:2]
    x0 = int(MASK_REGION[0] * w); y0 = int(MASK_REGION[1] * h)
    x1 = int(MASK_REGION[2] * w); y1 = int(MASK_REGION[3] * h)
    mask_full = np.zeros((h, w), dtype=bool)
    mask_full[y0:y1, x0:x1] = True
    bg = arr[~mask_full]
    purple = np.median(bg.reshape(-1, 3), axis=0).astype(np.int32)
    arr[y0:y1, x0:x1] = purple
    Image.fromarray(arr.astype(np.uint8)).save(str(COMFY_INPUT / MASKED_REF), "PNG")
    print(f"[v243] masked ref saved (central filled purple {tuple(purple)})")


def extract_bat_mask():
    """从原图中央内框提取真实蝙蝠剪影(深色像素), 返回全图尺寸 bool 掩码。"""
    img = Image.open(COMFY_INPUT / REF_IMG).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    x0 = int(BAT_BOX[0] * w); y0 = int(BAT_BOX[1] * h)
    x1 = int(BAT_BOX[2] * w); y1 = int(BAT_BOX[3] * h)
    crop = arr[y0:y1, x0:x1].astype(np.int32)
    lum = crop.sum(axis=2)
    # 蝙蝠/月亮深色(lum 低), 紫底 lum 高 -> 阈值提取前景
    bat = lum < 230
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    bat = cv2.morphologyEx(bat.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    bat = cv2.morphologyEx(bat, cv2.MORPH_CLOSE, kernel)
    full = np.zeros((h, w), dtype=bool)
    full[y0:y1, x0:x1] = bat.astype(bool)
    ys, xs = np.where(full)
    if len(xs) == 0:
        raise RuntimeError("[v243] bat mask empty - 阈值可能失效")
    print(f"[v243] bat mask: {len(xs)}px bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})")
    return full


def make_pose_refs():
    """对原图蝙蝠剪影做仿射变换, 贴回紫底(保留环/弧), 生成每姿态 Canny 参考图。"""
    base = np.array(Image.open(COMFY_INPUT / MASKED_REF).convert("RGB")).copy()
    mask_full = extract_bat_mask()
    ys, xs = np.where(mask_full)
    bx0, bx1 = xs.min(), xs.max(); by0, by1 = ys.min(), ys.max()
    sh, sw = by1 - by0 + 1, bx1 - bx0 + 1
    sil = np.zeros((sh, sw, 4), np.uint8)
    m = mask_full[by0:by1 + 1, bx0:bx1 + 1]
    sil[m, 0] = 0; sil[m, 1] = 0; sil[m, 2] = 0; sil[m, 3] = 255  # 黑剪影 + alpha
    for tag, (rot, fh, fv, sc) in POSE_XFORM.items():
        si = Image.fromarray(sil, "RGBA")
        if fh:
            si = si.transpose(Image.FLIP_LEFT_RIGHT)
        if fv:
            si = si.transpose(Image.FLIP_TOP_BOTTOM)
        if sc != 1.0:
            si = si.resize((max(1, int(si.width * sc)), max(1, int(si.height * sc))), Image.BILINEAR)
        if rot != 0:
            si = si.rotate(rot, expand=True, resample=Image.BILINEAR)
        bcx, bcy = (bx0 + bx1) // 2, (by0 + by1) // 2
        base_img = Image.fromarray(base).convert("RGBA")
        base_img.alpha_composite(si, (bcx - si.width // 2, bcy - si.height // 2))
        out = base_img.convert("RGB")
        out.save(str(COMFY_INPUT / f"v243_pose_{tag}.png"), "PNG")
        print(f"[v243] pose ref saved v243_pose_{tag}.png (rot={rot} fh={fh} fv={fv} sc={sc})")


def build(seed, pose_global, pose_region, tag):
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
    g["3b"] = {"class_type": "LoadImage", "inputs": {"image": f"v243_pose_{tag}.png"}}
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3b", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": CANNY_STRENGTH}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": TILE_STRENGTH}}

    global_pos = (
        "strict 2D flat printed vintage craft spirits label art, NO 3D, NO cup, NO trophy, NO teapot, NO vase, NO chalice, NO mug, NO ceramic, "
        "a perfectly centered circular emblem with a single thin clean minimal ring outline on a flat saturated purple background, "
        "ONE single stylized 2D bat silhouette INSIDE the ring (NOT realistic, NOT photographic, NOT 3D), "
        + pose_global +
        "ABOVE the badge: only a single thin clean curved line following the top arc of the badge - this is a thin LINE not a banner, NOT a banner with text, NOT a ribbon, NOT an inscribed banner, NOT a decorative banner, just a thin curving line, "
        "BELOW the badge: completely empty purple space, no decoration, no elements, no marks, "
        "INSIDE the badge: ONLY the ring outline + the bat silhouette + a single small abstract crescent moon shape near the bat - NOTHING ELSE inside the ring, NO decorative text-like marks inside, NO ornamental patterns inside, "
        "OUTSIDE the badge: NO banner text, NO ribbon text, NO inscription, NO letters, NO words anywhere outside the ring, "
        "STRICTLY 2D flat vector vintage craft spirits label print style, no shading, no gradient, no reflection, no shadow, no ground plane, "
        "STRICTLY saturated purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F color blocks ONLY, "
        "NO gray, NO desaturated muted purple, NO washed-out pale purple, NO beige, NO brown, NO green, NO blue, NO gray tint, "
        "uniform flat saturated purple background, NO tonal variation, NO lighter purple patches, NO gradient background, "
        "ABSOLUTELY NO characters, NO letters, NO glyphs, NO scribbles anywhere in the entire image, "
        "the ONLY shapes present are: a thin curving line (above), a thin ring outline (badge), a stylized bat in a NEW dynamic pose (badge), a small crescent (badge), empty purple space (everywhere else)"
    )
    region0 = (
        "the CENTER circular emblem with a thin clean ring outline ONLY, "
        "ONE single stylized 2D flat black bat silhouette in dead center inside the ring, "
        + pose_region +
        "a single small stylized crescent moon shape near the bat acting as decorative ground, "
        "ring is a clean thin LINE outline, NOTHING on the ring line, NOTHING inside the ring besides the bat and crescent, "
        "NO text-like marks inside the ring, NO patterns inside the ring, NO ornamental inscription inside the ring, "
        "keep STRICTLY saturated purple/black/white palette ONLY, flat 2D print, NO 3D, "
        "perfectly centered, NO shadow, NO reflection, NO ground plane. " + COHESIVE
    )

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": global_pos}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    regions = [
        {"x": 0.05, "y": 0.10, "w": 0.90, "h": 0.80, "strength": 1.30, "prompt": region0},
        {"x": 0.10, "y": 0.00, "w": 0.80, "h": 0.15, "strength": 1.25,
         "prompt": ("above the badge: a single thin clean curved line following the top arc of the badge, "
                    "this is JUST a thin curving LINE shape, NOT a banner, NOT a ribbon, NOT a scroll, NOT an inscribed banner, "
                    "NO text, NO letters, NO words, NO pseudo-letters, NO glyphs, NO marks, NO scribbles, NO decoration on this line, "
                    "just a single thin curved line shape following the arc, STRICTLY solid saturated purple background, "
                    "perfectly centered. " + COHESIVE)},
        {"x": 0.30, "y": 0.88, "w": 0.40, "h": 0.10, "strength": 1.00,
         "prompt": ("below the badge: completely EMPTY solid saturated purple space, "
                    "NO banner, NO ribbon, NO scroll, NO pendant, NO triangle, NO drop shape, NO letters, NO glyphs, NO marks, NO scribbles, NO decoration, "
                    "STRICTLY solid saturated purple background, perfectly centered. " + COHESIVE)},
    ]
    region_nodes = []
    for i, r in enumerate(regions):
        rk = f"rp{i}"; sk = f"sa{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v243_{tag}_bat_logo"}}
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
    return out if alpha >= 1.0 else np.clip(dst.astype(np.float32) * (1 - alpha) + out.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def unsharp(rgb_img, radius=1.5, percent=50, threshold=2):
    return rgb_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")
        except Exception:
            pass
        print("===== ComfyUI /prompt HTTP 400 body =====")
        print(body[:4000])
        print("===========================================")
        raise


def history(pid):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
        return json.loads(r.read())


def gen(seed, pose_global, pose_region, tag):
    g = build(seed, pose_global, pose_region, tag)
    r = submit(g); pid = r["prompt_id"]
    print(f"[v243-SDXL] pid={pid[:8]} seed={seed}")
    for _ in range(180):
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        return None
    for node_id, info in h[pid].get("outputs", {}).items():
        for img in info.get("images", []):
            src = COMFY_OUTPUT / img["filename"]
            if not src.exists():
                src = Path(img.get("abs_path", src))
            if src.exists():
                return str(src)
    return None


def ocr_erase_text(image_path, out_path, center_keep=CENTER_KEEP, expand=0.22, min_area=150):
    import easyocr
    img = Image.open(image_path).convert("RGB"); arr = np.array(img)
    h, w = arr.shape[:2]; bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    res_b = []
    for dec in ['beamsearch', 'greedy']:
        for item in reader.readtext(bgr, decoder=dec):
            if len(item) >= 3:
                res_b.append((item[0], item[1], float(item[2])))
    boxes = []
    for box_pts, text, conf in res_b:
        if conf < 0.10:
            continue
        pts = np.array(box_pts).astype(int)
        x1 = pts[:, 0].min(); x2 = pts[:, 0].max(); y1 = pts[:, 1].min(); y2 = pts[:, 1].max()
        bw = x2 - x1; bh = y2 - y1
        if bw * bh < min_area:
            continue
        px = int(bw * expand); py = int(bh * expand)
        x1 = max(0, x1 - px); y1 = max(0, y1 - py); x2 = min(w, x2 + px); y2 = min(h, y2 + py)
        cx = (x1 + x2) / 2 / w; cy = (y1 + y2) / 2 / h
        if center_keep[0] <= cx <= center_keep[2] and center_keep[1] <= cy <= center_keep[3]:
            continue
        boxes.append((x1, y1, x2, y2, text))
    for x1, y1, x2, y2, t in boxes:
        ring_pixels = []
        rh, rw = (y2 - y1), (x2 - x1)
        for sy, ey in [(0, max(0, y1 - rh)), (min(h, y2 + rh), h)]:
            if ey > sy:
                ring_pixels.append(arr[sy:ey, max(0, x1 - rw):min(w, x2 + rw)])
        for sx, ex in [(0, max(0, x1 - rw)), (min(w, x2 + rw), w)]:
            if ex > sx:
                ring_pixels.append(arr[max(0, y1 - rh):min(h, y2 + rh), sx:ex])
        if not ring_pixels:
            continue
        bg = np.concatenate([p.reshape(-1, 3) for p in ring_pixels if p.size > 0], axis=0)
        if len(bg) < 100:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        bgr = cv2.inpaint(bgr, mask, 6, cv2.INPAINT_TELEA)
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(out_path, quality=95)
    return len(boxes)


def burn_clean_text(image_path, out_path, word="NOCTWING",
                    font_name="PirataOne-Regular.ttf", color=(245, 235, 210)):
    """PIL 弧字烧字(铁律#4 / 图裂变单词铁律#4): 矢量清晰渲染, 绝不糊/缺笔。
    烧回【蝙蝠隐喻非侵权词】(替换 BACARDÍ 商标), 沿徽章顶部弧线。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    draw_arc_text = None
    try:
        from arc_text import draw_arc_text
    except Exception as e:
        print(f"[v243] arc_text import 失败: {e}; 回退直烧")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    center = (int(w * 0.50), int(h * 0.50))
    radius = int(h * 0.30)
    font_path = str(PROJECT / "fonts" / font_name)
    if draw_arc_text is not None:
        out = draw_arc_text(img, word, font_path, int(h * 0.05), color,
                            center, radius, 200, 340,
                            char_spacing_px=max(2, int(w * 0.004)), flip_180=False)
    else:
        from PIL import ImageDraw, ImageFont
        d = ImageDraw.Draw(img)
        f = ImageFont.truetype(font_path, int(h * 0.05))
        tw = f.getlength(word)
        d.text(((w - tw) / 2, h * 0.40), word, font=f, fill=color)
        out = img
    out.convert("RGB").save(out_path, quality=95)
    print(f"[v243] 弧字烧回: word={word} font={font_name} center={center} radius={radius}")
    return out


def ocr_verify_word(image_path, expect):
    """核验烧字是否清晰可读(无缺笔/无糊): easyocr 读图, 看期望词字母是否齐全。"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        res = reader.readtext(str(image_path))
        flat = " ".join(t for _, t, _ in res).upper()
        ok = all(ch in flat for ch in expect.upper() if ch.isalpha())
        print(f"[v243] OCR 核验: 读到={[t for _, t, _ in res][:6]} | 期望含={expect} | 字母齐全={ok}")
        return ok, [t for _, t, _ in res]
    except Exception as e:
        return False, [f"OCR_FAIL:{e}"]


def side_by_side(orig_path, gen_path, out_path):
    orig = Image.open(orig_path).convert("RGB"); gen = Image.open(gen_path).convert("RGB")
    if orig.size != gen.size:
        gen = gen.resize(orig.size)
    out = Image.new("RGB", (orig.width * 2 + 30, orig.height), "white")
    out.paste(orig, (0, 0)); out.paste(gen, (orig.width + 30, 0))
    out.save(out_path, quality=95)


# 烧回的清晰新词: 蝙蝠隐喻 + 非 BACARDÍ 侵权词(铁律#4 AI 自主判定商标需换词)
BURN_WORD = "NOCTWING"   # NOCT(夜) + WING(翼) = 蝙蝠隐喻, 8 字母贴近原词长度


def main():
    make_masked_ref()
    make_pose_refs()
    src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF_IMG).convert("RGB")), cv2.COLOR_RGB2BGR)
    results = {}
    for tag, pose_global, pose_region in POSES:
        seed = SEEDS[tag]
        raw = gen(seed, pose_global, pose_region, tag)
        if not raw:
            print(f"[v243] {tag} FAIL"); continue
        dst_bgr = cv2.cvtColor(np.array(Image.open(raw).convert("RGB")), cv2.COLOR_RGB2BGR)
        matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
        matched_rgb = unsharp(Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)))
        tmp = JOB / f"v243_{tag}_pre.jpg"; matched_rgb.save(str(tmp), quality=95)
        out = JOB / f"v243_{tag}.jpg"
        n = ocr_erase_text(str(tmp), str(out))
        # 烧回清晰弧字(解决文字糊/缺字) + OCR 核验
        out_text = JOB / f"v243_{tag}_text.jpg"
        burn_clean_text(str(out), str(out_text), word=BURN_WORD)
        ok, _ = ocr_verify_word(str(out_text), BURN_WORD)
        side_by_side(COMFY_INPUT / REF_IMG, out_text, JOB / f"_compare_v243_{tag}.jpg")
        d = cv2.cvtColor(np.array(Image.open(out_text).convert("RGB")), cv2.COLOR_RGB2BGR)
        if src_bgr.shape != d.shape:
            d = cv2.resize(d, (src_bgr.shape[1], src_bgr.shape[0]))
        # 量化 QC
        hi = float(np.mean([cv2.compareHist(cv2.calcHist([src_bgr], [c], None, [32], [0, 256]),
                                          cv2.calcHist([d], [c], None, [32], [0, 256]), cv2.HISTCMP_INTERSECT) for c in range(3)]))
        h0, w0 = min(src_bgr.shape[0], d.shape[0]), min(src_bgr.shape[1], d.shape[1])
        mse = ((cv2.resize(src_bgr, (w0, h0)).astype(np.float32) - cv2.resize(d, (w0, h0)).astype(np.float32)) ** 2).mean()
        sd = float(np.clip(1 - mse / (255.0 ** 2), 0, 1))
        print(f"[v243] {tag}: erased={n} | 烧字OCR字母齐全={ok} | 配色 hist={hi:.3f} | 结构差异={sd:.3f}")
        results[tag] = out_text
        print(f"  saved v243_{tag}_text.jpg")

    # 3 联对照: 原图 + 两变体
    if results:
        orig = Image.open(COMFY_INPUT / REF_IMG).convert("RGB")
        imgs = [orig] + [Image.open(results[t]).convert("RGB").resize(orig.size) for (t, _, _) in POSES if t in results]
        gap = 20; W = orig.width * len(imgs) + gap * (len(imgs) - 1)
        grid = Image.new("RGB", (W, orig.height), "white")
        x = 0
        for im in imgs:
            grid.paste(im, (x, 0)); x += orig.width + gap
        grid.save(str(JOB / "_v243_grid.jpg"), quality=95)
        print(f"[v243] grid saved _v243_grid.jpg ({len(imgs)} cols)")


if __name__ == "__main__":
    main()
