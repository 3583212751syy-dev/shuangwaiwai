"""
v282 — 用统一背景+大羽化+整体模糊彻底消除文字区遮挡痕迹，严格保留内徽章/圆环。

修复 v281 肉眼问题：
  1. EST./1862 及弧带仍有 smudge/区块感 → 不用 cv2.inpaint 抹大片，改用"全局背景色填充 + 高斯模糊 + 均匀噪点"，
     让文字区与周围背景完全融为一体。
  2. 圆环/内徽章/装饰卷草必须保留 → 用 preserve mask（圆 r=450 但挖去 EST./1862 小块）只保护非文字区域。
  3. 蝙蝠裂变保持 v281 的极端骨架。

流程：
  1. text_free_source：
     - 文字区（弧带、主/副/Est./1862/三角）统一用全局背景色填充。
     - 整图大高斯模糊（31px）消去所有填充边界。
     - 加均匀高斯噪点匹配原图纹理。
     - 用 preserve mask（羽化）把内徽章+圆环+装饰卷草从原图粘回。
  2. bat_mask：同 v281，精确蝙蝠剪影。
  3. inpaint_mask：bat_mask 外扩 40px。
  4. pose guides / ComfyUI / 合成 / 烧字：同 v281。
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
JOB = PROJECT / "jobs" / "v282"
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

cx, cy, r_badge = 776, 746, 300
ring_outer = 421

DENOISE = 0.92
REFINE_DENOISE = 0.12
IPA_WEIGHT = 0.12
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.85
TILE_STRENGTH = 0.45

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  282001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   282101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", 282201),
]

NEG = (
    "text, letters, words, readable text, brand name, BACARDI, logo, watermark, "
    "tarot, mandala, chandelier, filigree, ornamental frame, excessive decoration, "
    "green, brown, gray, beige, blue, color shift, blur, watercolor, photorealistic, "
    "multiple bats, small bat, realistic fur, white bat, red bat, orange bat"
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
    h, w = arr.shape[:2]
    regions = [
        arr[:int(h*0.18), :int(w*0.18)],
        arr[:int(h*0.18), int(w*0.82):],
        arr[int(h*0.82):, :int(w*0.18)],
        arr[int(h*0.82):, int(w*0.82):],
    ]
    samples = np.concatenate([r.reshape(-1, 3) for r in regions], axis=0)
    return np.median(samples, axis=0)


def make_text_free_source():
    """统一背景填充 + 大模糊去边界 + 保留内徽章/圆环/卷草。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]
    bg = _global_bg_color(arr)

    # 1. 文字 mask
    text_mask = np.zeros((h, w), dtype=np.uint8)

    # 弧带（覆盖 LA CASA DEL MURCIELAGO + 横幅）
    arc = _ring_sector_mask(h, w, cx, cy, ring_outer - 110, ring_outer + 95, 180, 360)
    text_mask = cv2.bitwise_or(text_mask, arc.astype(np.uint8) * 255)

    # 主/副/Est./1862/三角
    rects = [
        (1000, 1185, 120, 1430),  # BACARDÍ
        (1185, 1370, 300, 1250),  # MCHEART
        (735,  895,  270,  650),  # Est.
        (735,  895,  900, 1280),  # 1862
        (1460, 1545, 660,  890),  # 底部三角周围
    ]
    for y0, y1, x0, x1 in rects:
        text_mask[y0:y1, x0:x1] = 255

    # 2. 填充文字区为全局背景色
    filled = arr.copy()
    filled[text_mask > 127] = bg

    # 3. 大模糊消去所有边界，得到平滑底图
    blur = cv2.GaussianBlur(filled.astype(np.uint8), (31, 31), 0).astype(np.float32)

    # 4. 加均匀噪点匹配原图背景纹理
    # 采样四角背景方差
    corners = np.concatenate([
        arr[:int(h*0.18), :int(w*0.18)].reshape(-1, 3),
        arr[:int(h*0.18), int(w*0.82):].reshape(-1, 3),
        arr[int(h*0.82):, :int(w*0.18)].reshape(-1, 3),
        arr[int(h*0.82):, int(w*0.82):].reshape(-1, 3),
    ], axis=0)
    std = corners.std(axis=0)
    noise = np.random.normal(0, std * 0.7, size=(h, w, 3)).astype(np.float32)
    blur = np.clip(blur + noise, 0, 255)

    # 5. preserve mask：保护内徽章+圆环+附近卷草，但挖去 EST./1862 文字区
    preserve = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(preserve, (cx, cy), ring_outer + 30, 255, -1)
    # 挖去 EST./1862
    for y0, y1, x0, x1 in [(735, 895, 270, 650), (735, 895, 900, 1280)]:
        preserve[y0:y1, x0:x1] = 0
    preserve = cv2.GaussianBlur(preserve.astype(np.float32), (25, 25), 0) / 255.0

    # 6. 合成：preserve 区域取原图，其余取模糊底图
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * preserve + blur[:, :, c] * (1 - preserve)

    # 7. 最后再把 EST./1862 小字区用模糊底图重新盖一次（防止原图残字漏出）
    est_mask = np.zeros((h, w), dtype=np.uint8)
    for y0, y1, x0, x1 in [(735, 895, 270, 650), (735, 895, 900, 1280)]:
        est_mask[y0:y1, x0:x1] = 255
    est_mask = cv2.dilate(est_mask, np.ones((9, 9), np.uint8), iterations=1)
    est_blur = cv2.GaussianBlur(est_mask.astype(np.float32), (21, 21), 0) / 255.0
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - est_blur) + blur[:, :, c] * est_blur

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_bat_mask(size=(1552, 2000)):
    w, h = size
    src = np.array(make_text_free_source()).astype(np.float32)
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge, 255, -1)

    R, G, B = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    stdc = np.std(src, axis=2)
    dark_neutral = (maxc < 80) & (stdc < 25)

    mask = dark_neutral.astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, badge)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((8, 8), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return Image.fromarray(mask)


def _transform_silhouette(mask, sx, sy, rot_deg=0.0):
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
    M[0, 0] *= sx
    M[0, 1] *= sx
    M[1, 0] *= sy
    M[1, 1] *= sy
    # 中心保持
    M[0, 2] = cx - sx * (cx * math.cos(math.radians(rot_deg)) - cy * math.sin(math.radians(rot_deg)))
    M[1, 2] = cy - sy * (cx * math.sin(math.radians(rot_deg)) + cy * math.cos(math.radians(rot_deg)))
    out = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def make_pose_guides():
    bat = np.array(make_bat_mask().convert("L")).astype(np.uint8)
    _, bat = cv2.threshold(bat, 127, 255, cv2.THRESH_BINARY)

    bat = cv2.morphologyEx(bat, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    bat = cv2.morphologyEx(bat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    poses = {}
    up = _transform_silhouette(bat, sx=0.65, sy=1.45, rot_deg=-3)
    spread = _transform_silhouette(bat, sx=1.70, sy=0.60, rot_deg=0)
    fold_base = _transform_silhouette(bat, sx=1.05, sy=0.90, rot_deg=2)
    h, w = fold_base.shape
    ys = np.arange(h).reshape(-1, 1).astype(np.float32)
    factor = 1.0 + 0.30 * np.clip((ys - cy) / (h - cy), 0, 1) - 0.10 * np.clip((cy - ys) / cy, 0, 1)
    x_map = (np.arange(w).astype(np.float32)[None, :] - cx) / factor + cx
    y_map = np.tile(ys, (1, w))
    fold = cv2.remap(fold_base, x_map.astype(np.float32), y_map.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    for tag, m in (("up", up), ("spread", spread), ("fold", fold)):
        badge = np.zeros_like(m)
        cv2.circle(badge, (cx, cy), r_badge, 255, -1)
        m = cv2.bitwise_and(m, badge)
        edges = cv2.Canny(m, 50, 150)
        edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)
        poses[tag] = Image.fromarray(edges).convert("RGB")
        path = COMFY_INPUT / f"v282_pose_{tag}.png"
        poses[tag].save(path, quality=95)
        print(f"[v282] pose guide {tag} -> {path}")
    return poses


def make_inpaint_mask(size=(1552, 2000)):
    w, h = size
    bat = np.array(make_bat_mask().convert("L")).astype(np.uint8)
    _, bat = cv2.threshold(bat, 80, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(bat, np.ones((40, 40), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return Image.fromarray(mask)


def build(seed, tag, pose_img):
    bat_desc = {
        "up": (
            "bat wings raised steeply upward, tall vertical dramatic pose, "
            "elongated silhouette, wings pointing to top center"
        ),
        "spread": (
            "bat wings spread wide horizontally, powerful broad wingspan, "
            "wings flat and stretched left and right, open stance"
        ),
        "fold": (
            "bat wings folded downward draping like dark cloak, "
            "wings wrapped around lower body, compact hunched posture"
        ),
    }[tag]

    pos = (
        f"black bat silhouette only, centered in dark purple circular badge, {bat_desc}, "
        f"NO text, NO letters, NO words, NO background change, "
        f"gothic emblem style, sharp clean edges, deep violet and black only"
    )

    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v282_text_free_source.png"}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}

    g["mask_load"] = {"class_type": "LoadImage", "inputs": {"image": "v282_inpaint_mask.png"}}
    g["mask"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["mask_load", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}
    g["mask_conv"] = {"class_type": "ImageToMask", "inputs": {"image": ["mask", 0], "channel": "red"}}

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

    g["20"] = {"class_type": "LoadImage", "inputs": {"image": f"v282_pose_{tag}.png"}}
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
        "sampler_name": "euler", "scheduler": "normal", "denoise": REFINE_DENOISE}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v282_{tag}"}}
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
    cy_arc = int(h * 0.13) + radius
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT, fs_arc, radius, char_spacing_px=3)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 170:
            break
        fs_arc = int(fs_arc * 0.92)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    img = arc_text.draw_arc_text(img, arc, FONT, fs_arc, color,
                                 center=(w // 2, cy_arc), radius=radius,
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
    ap.add_argument("--source-only", action="store_true", help="只生成 text_free_source 并出 check 图，方便自检")
    args = ap.parse_args()

    src_path = COMFY_INPUT / "v282_text_free_source.png"
    mask_path = COMFY_INPUT / "v282_inpaint_mask.png"
    bat_mask_path = COMFY_INPUT / "v282_bat_mask.png"

    if not src_path.exists() or args.force:
        src = make_text_free_source()
        src.save(src_path, quality=98)
        orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(orig, (0, 0))
        check.paste(src, (1552 + 24, 0))
        check.save(JOB / "_source_check.png", quality=95)
        print(f"[v282] text-free source -> {src_path}")
        print(f"[v282] source check -> {JOB / '_source_check.png'}")
    else:
        print(f"[v282] reuse text-free source")

    if args.source_only:
        print("[v282] source-only mode, exit.")
        return

    if not bat_mask_path.exists() or args.force:
        bm = make_bat_mask()
        bm.save(bat_mask_path)
        print(f"[v282] bat mask -> {bat_mask_path}")
    else:
        print(f"[v282] reuse bat mask")

    pose_guides = make_pose_guides()

    if not mask_path.exists() or args.force:
        mask = make_inpaint_mask()
        rgba = Image.merge("RGBA", (mask, mask, mask, mask))
        rgba.save(mask_path)
        print(f"[v282] inpaint mask -> {mask_path}")
    else:
        print(f"[v282] reuse inpaint mask")

    text_free = Image.open(src_path).convert("RGB")
    bat_mask = Image.open(bat_mask_path).convert("L")
    bat_mask_arr = np.array(bat_mask).astype(np.float32) / 255.0
    bat_mask_arr = np.stack([bat_mask_arr] * 3, axis=-1)

    files = []
    for tag, big, sub, arc, seed in SUBJECT_VARIANTS:
        cands = sorted(COMFY_OUTPUT.glob(f"v282_{tag}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands or args.force:
            print(f"[v282] generating {tag} ...")
            g = build(seed, tag, pose_guides[tag])
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
            print(f"[v282] reuse {tag}: {raw_path.name}")

        raw = Image.open(raw_path).convert("RGB").resize((1552, 2000), Image.LANCZOS)
        raw_arr = np.array(raw).astype(np.float32)
        bg_arr = np.array(text_free).astype(np.float32)
        composed = raw_arr * bat_mask_arr + bg_arr * (1 - bat_mask_arr)
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v282_{tag}_composed.png", quality=95)

        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(text_free, (0, 0))
        check.paste(composed, (1552 + 24, 0))
        check.save(JOB / f"_compose_check_{tag}.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v282_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"  {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v282.png"
    grid.save(gp, quality=92)
    print(f"  grid -> {gp}")


if __name__ == "__main__":
    main()
