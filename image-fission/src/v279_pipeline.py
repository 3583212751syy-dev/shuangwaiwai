"""
v279 — 严格保留背景 + 仅蝙蝠区裂变 + 无遮挡清字

修复 v278 被用户指出的问题：
  1. 主体裂变没拉开 → 蝙蝠只做 mask 内重绘（denoise 0.90 / IPAdapter 0.10 / Canny 0.12 / Tile 0.35），
     三变体 up/spread/fold 姿态差异明显。
  2. 背景颜色改变 → VAEEncodeForInpaint + 精确蝙蝠 mask，SDXL 只能碰蝙蝠；背景像素 100% 保留原图。
  3. 文字区遮挡 → 不用扁平色块硬填，改用全局背景色填充 + 25px 高斯羽化，文字像素与周围背景无缝过渡。

流程：
  1. 生成 text_free_source.png：弧带/主/副/Est/1862 全部用全局背景色填充 + 区域羽化，蝙蝠/圆环保留。
  2. 生成 bat_mask.png：仅覆盖蝙蝠剪影（暗+中性灰像素），边缘羽化，严格限定 inner badge 圆内。
  3. ComfyUI：text_free_source + bat_mask → VAEEncodeForInpaint，只生成蝙蝠区。
  4. 将生成后的蝙蝠区按 mask 合成回 text_free_source（背景绝对不变）。
  5. PIL AbrilFatface 原位烧新字。
"""
import argparse
import json
import math
import sys
import time
import uuid
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "v279"
JOB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "src"))
import arc_text

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"

TARGET_W, TARGET_H = 1024, 1280
INK = (26, 10, 31)

# 裂变参数：拉开蝙蝠姿态，同时用 IPAdapter 锁死黑色/紫色材质
DENOISE = 0.90
IPA_WEIGHT = 0.10
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.12
TILE_STRENGTH = 0.35

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  279001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   279101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", 279201),
]

NEG = (
    "text, letters, words, readable text, brand name, BACARDI, logo, watermark, "
    "tarot, mandala, chandelier, filigree, ornamental frame, excessive decoration, "
    "green, brown, gray, beige, blue, color shift, blur, watercolor, photorealistic, "
    "multiple bats, small bat, realistic fur"
)


def _ring_sector_mask(h, w, cx, cy, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) - a0) % 360
    sweep = (a1 - a0) % 360
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    in_angle = ang <= sweep
    return in_ring & in_angle


def _global_bg_color(arr):
    """从四角安全区采样原图背景色（排除中心 logo 区域）。"""
    h, w = arr.shape[:2]
    # 四角 15% 边距区域
    regions = [
        arr[:int(h*0.15), :int(w*0.15)],
        arr[:int(h*0.15), int(w*0.85):],
        arr[int(h*0.85):, :int(w*0.15)],
        arr[int(h*0.85):, int(w*0.85):],
    ]
    samples = np.concatenate([r.reshape(-1, 3) for r in regions], axis=0)
    return np.median(samples, axis=0)


def make_text_free_source():
    """全清字底图：弧带/主/副/Est/1862 全部用全局背景色填充 + 区域羽化。
    蝙蝠/圆环保留；文字区无可见遮挡痕迹。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]
    cx, cy = 776, 746
    ring_outer = 421
    bg = _global_bg_color(arr)

    masks = []

    # 弧带：宽环带确保全覆盖
    arc_mask = _ring_sector_mask(h, w, cx, cy, ring_outer - 90, ring_outer + 70, 190, 350)
    masks.append(arc_mask)

    # 主/副/Est/1862
    rects = [
        (1000, 1180, 150, 1400),  # BACARDI
        (1180, 1350, 350, 1200),  # MCHEART
        (720,  880,  300, 620),   # Est.
        (720,  880,  930, 1250),  # 1862
    ]
    for y0, y1, x0, x1 in rects:
        m = np.zeros((h, w), dtype=np.uint8)
        m[y0:y1, x0:x1] = 255
        masks.append(m)

    # 逐个填充 + 区域羽化（核 25px，避免泄漏到蝙蝠）
    for m in masks:
        m = cv2.dilate(m.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1)
        arr[m > 0] = bg
        m_blur = cv2.GaussianBlur(m.astype(np.float32), (25, 25), 0) / 255.0
        blurred = cv2.GaussianBlur(arr.astype(np.uint8), (25, 25), 0).astype(np.float32)
        for c in range(3):
            arr[:, :, c] = arr[:, :, c] * (1 - m_blur) + blurred[:, :, c] * m_blur

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_bat_mask(size=(1552, 2000)):
    """只覆盖蝙蝠剪影的 mask；背景+内徽章圆+ring 装饰全部保留不动。"""
    w, h = size
    src = np.array(make_text_free_source()).astype(np.float32)
    cx, cy = 776, 746

    # 限制在 inner badge 内，排除外圈 ring 装饰
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), 300, 255, -1)

    # 蝙蝠：暗且接近中性灰（与紫色 badge 区分）
    R, G, B = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    stdc = np.std(src, axis=2)
    dark_neutral = (maxc < 80) & (stdc < 25)

    mask = dark_neutral.astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, badge)
    # 去掉小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # 膨胀覆盖蝙蝠边缘抗锯齿
    mask = cv2.dilate(mask, np.ones((12, 12), np.uint8), iterations=1)
    # 羽化
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return Image.fromarray(mask)


def build(seed, tag):
    bat_desc = {
        "up": (
            "bat wings raised high upward pointing to the sky, wings angled steeply up, "
            "head tilted back, talons extended downward, tall vertical dramatic pose"
        ),
        "spread": (
            "bat wings spread wide horizontally in a powerful open stance, "
            "broad wingspan stretched left and right, wings flat and wide, horizontal pose"
        ),
        "fold": (
            "bat wings folded downward draping like a dark cloak, wings wrapped around body, "
            "compact hunched mysterious posture, closed downward pose"
        ),
    }[tag]

    pos = (
        f"vintage gothic purple spirit brand emblem centered on soft purple background, "
        f"circular dark purple badge with black bat silhouette, {bat_desc}, "
        f"NO text, NO letters, NO words, NO brand name, NO readable glyphs, "
        f"ornamental geometric flourishes around the ring, baroque accents, "
        f"purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F only, "
        f"vintage craft spirits label art, sharp clean edges, bold emblem, "
        f"same material and art style as original, only pose changed"
    )

    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v279_text_free_source.png"}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}

    # mask 需要与图片同尺寸；ImageScale 输出 IMAGE，再用 ImageToMask 转成 MASK
    g["mask_load"] = {"class_type": "LoadImage", "inputs": {"image": "v279_bat_mask.png"}}
    g["mask"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["mask_load", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}
    g["mask_conv"] = {"class_type": "ImageToMask", "inputs": {"image": ["mask", 0], "channel": "red"}}

    # 只编码蝙蝠区，背景完全不动
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["3", 0], "vae": ["1", 2], "mask": ["mask_conv", 0], "grow_mask_by": 8}}

    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.80,
        "noise": 0.08, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}

    # Canny 用全清字底图，避免旧字边缘干扰蝙蝠生成
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": 50, "high_threshold": 100, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": CANNY_STRENGTH}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": TILE_STRENGTH}}

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": pos}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG}}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["24", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 32, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["24", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.10}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v279_{tag}"}}
    return g


def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def poll(pid, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(5)
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=20) as r:
                h = json.loads(r.read())
        except Exception:
            continue
        if pid in h:
            e = h[pid]
            if e.get("status", {}).get("completed"):
                return e
            if e.get("status", {}).get("error"):
                raise RuntimeError(f"ComfyUI error: {e['status']['error']}")
    raise TimeoutError("poll timeout")


def _calibrate_font(text, font_path, start_size, max_w):
    lo, hi = 8, start_size
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        if font.getlength(text) <= max_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def burn_text(img, arc, big, sub, color=INK):
    img = img.convert("RGB")
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img)

    fs_arc = int(w * 0.045)
    radius = int(w * 0.46)
    cy = int(h * 0.13) + radius
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT, fs_arc, radius, char_spacing_px=3)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 170:
            break
        fs_arc = int(fs_arc * 0.92)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    img = arc_text.draw_arc_text(img, arc, FONT, fs_arc, color,
                                 center=(w // 2, cy), radius=radius,
                                 start_angle_deg=start, end_angle_deg=end,
                                 char_spacing_px=3, flip_180=False)

    fs_big = _calibrate_font(big, FONT, int(h * 0.145), max_w=int(w * 0.82))
    font = ImageFont.truetype(FONT, fs_big)
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, int(h * 0.555)), big, font=font, fill=color, anchor="mm")

    fs_sub = _calibrate_font(sub, FONT, int(h * 0.075), max_w=int(w * 0.60))
    font = ImageFont.truetype(FONT, fs_sub)
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, int(h * 0.655)), sub, font=font, fill=color, anchor="mm")

    f_side = ImageFont.truetype(FONT, int(h * 0.028))
    draw.text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    draw.text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_path = COMFY_INPUT / "v279_text_free_source.png"
    mask_path = COMFY_INPUT / "v279_bat_mask.png"

    if not src_path.exists() or args.force:
        src = make_text_free_source()
        src.save(src_path, quality=98)
        print(f"[v279] text-free source -> {src_path}")
    else:
        print(f"[v279] reuse text-free source")

    if not mask_path.exists() or args.force:
        mask = make_bat_mask()
        # 保存为带 alpha 的 PNG，ComfyUI LoadImage 可输出 MASK
        rgba = Image.merge("RGBA", (mask, mask, mask, mask))
        rgba.save(mask_path)
        print(f"[v279] bat mask -> {mask_path}")
    else:
        print(f"[v279] reuse bat mask")

    text_free = Image.open(src_path).convert("RGB")
    bat_mask = Image.open(mask_path).convert("L")

    files = []
    for tag, big, sub, arc, seed in SUBJECT_VARIANTS:
        cands = sorted(COMFY_OUTPUT.glob(f"v279_{tag}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands or args.force:
            print(f"[v279] generating {tag} ...")
            g = build(seed, tag)
            r = submit(g)
            pid = r["prompt_id"]
            print(f"  pid={pid[:8]}")
            entry = poll(pid, timeout=1800)
            outs = []
            for n in entry.get("outputs", {}).values():
                if "images" in n:
                    outs += [COMFY_OUTPUT / im["filename"] for im in n["images"]]
            if not outs:
                raise RuntimeError(f"{tag}: no output")
            raw_path = outs[0]
        else:
            raw_path = cands[0]
            print(f"[v279] reuse {tag}: {raw_path.name}")

        # 使用原图（弧带已清）作为背景，把生成后的蝙蝠区合成回去
        raw = Image.open(raw_path).convert("RGB")
        raw = raw.resize((1552, 2000), Image.LANCZOS)

        # 将生成后的蝙蝠区严格合成回 text_free_source，背景绝对不变
        raw_arr = np.array(raw).astype(np.float32)
        bg_arr = np.array(text_free).astype(np.float32)
        mask_arr = np.array(bat_mask).astype(np.float32) / 255.0
        mask_arr = np.stack([mask_arr] * 3, axis=-1)
        composed = raw_arr * mask_arr + bg_arr * (1 - mask_arr)
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v279_{tag}_composed.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v279_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"  {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v279.png"
    grid.save(gp, quality=92)
    print(f"  grid -> {gp}")


if __name__ == "__main__":
    main()
