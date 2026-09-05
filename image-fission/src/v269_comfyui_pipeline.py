"""v269_comfyui_pipeline — 纯 ComfyUI 三阶段管线
Stage B: inpaint-dreamer SDXL ControlNet 生成式去字（先清原图）
Stage A: IPAdapter + Canny + Proteus v0.4 裂变 3 变体
Stage C: PIL 原位烧新字（底图已干净，不是色块遮盖）

解析度锁定 1024x1280（适配 RTX 4070 Ti 12G，避免 AnyText/大图 OOM）。
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
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "v269"
JOB.mkdir(parents=True, exist_ok=True)

ORIG_NAME = "6978fabda2cc99629fa9e81f802762d3.jpg"
TARGET_W, TARGET_H = 1024, 1280

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.safetensors"
CN_INPAINT = "controlnet-inpaint-dreamer-sdxl.fp16.safetensors"
FONT_PATH = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

# ---- 原图 1552x2000 上的文字区域（来自 v268 实测），缩放至 TARGET ----
SCALE_X = TARGET_W / 1552
SCALE_Y = TARGET_H / 2000


def sc(pts):
    return [(int(x * SCALE_X), int(y * SCALE_Y)) for x, y in pts]


# 更慷慨的 full ribbon polygon（覆盖整圈旧字带，含两侧 tab）
FULL_RIBBON_OUTER = [
    (120, 380), (120, 450), (140, 600), (180, 750), (250, 880),
    (450, 950), (700, 980), (850, 980), (1100, 950), (1300, 880),
    (1370, 750), (1410, 600), (1430, 450), (1430, 380),
    (1330, 310), (1100, 270), (776, 255), (450, 270), (220, 310)
]
FULL_RIBBON_INNER = [
    (600, 400), (776, 400), (950, 400),
    (1100, 520), (1100, 720), (1000, 800),
    (900, 850), (776, 900), (650, 850),
    (550, 800), (400, 720), (400, 520)
]


def s(x, y):
    return int(x * SCALE_X), int(y * SCALE_Y)


def make_text_mask():
    """生成 1024x1280 去字 mask：白=要重绘的文字区，黑=保留."""
    mask = Image.new("L", (TARGET_W, TARGET_H), 0)
    d = ImageDraw.Draw(mask)

    # 顶弧 ribbon（完整环形带）
    d.polygon(sc(FULL_RIBBON_OUTER) + sc(FULL_RIBBON_INNER)[::-1], fill=255)

    # 中心大字 BACARDÍ / MHEART
    d.rectangle([s(160, 980), s(1390, 1170)], fill=255)
    # 底部副字
    d.rectangle([s(360, 1180), s(1195, 1340)], fill=255)
    # Est. / 1862
    d.rectangle([s(350, 740), s(560, 860)], fill=255)
    d.rectangle([s(1000, 740), s(1220, 860)], fill=255)

    # 膨胀+羽化让生成边界自然
    mask = mask.filter(ImageFilter.MaxFilter(11))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    return mask


def save_inputs():
    orig = Image.open(COMFY_INPUT / ORIG_NAME).convert("RGB")
    resized = orig.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    resized.save(COMFY_INPUT / "v269_source.png", quality=98)

    mask = make_text_mask()
    mask.save(COMFY_INPUT / "v269_text_mask.png")
    mask.save(JOB / "v269_text_mask.png")

    # 预填底图：让 inpaint-dreamer 看不到旧字，避免它重建文字
    src_arr = np.array(resized)
    mask_bin = (np.array(mask) > 127).astype(np.uint8) * 255
    prefill = cv2.inpaint(src_arr, mask_bin, 9, cv2.INPAINT_TELEA)
    Image.fromarray(prefill).save(COMFY_INPUT / "v269_source_prefill.png", quality=98)
    print(f"[v269] source/mask/prefill saved to {COMFY_INPUT}")
    return resized


def node(id_, class_type, inputs):
    return {id_: {"class_type": class_type, "inputs": inputs}}


def build_clean_workflow(seed=269001):
    """Stage B：inpaint-dreamer 去字（预填后输入，避免模型重建旧字）."""
    g = {}
    g.update(node("1", "CheckpointLoaderSimple", {"ckpt_name": CKPT}))
    g.update(node("2", "LoadImage", {"image": "v269_source_prefill.png"}))
    g.update(node("3", "LoadImage", {"image": "v269_text_mask.png"}))
    g.update(node("4", "VAEEncodeForInpaint", {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 1], "grow_mask_by": 18
    }))
    g.update(node("pos_b", "CLIPTextEncode", {"clip": ["1", 1], "text": (
        "clean purple vintage label background, circular bat emblem, "
        "grunge scratched texture, violet and magenta color palette, "
        "no text, no letters, no words, no logo, seamless background"
    )}))
    g.update(node("neg_b", "CLIPTextEncode", {"clip": ["1", 1], "text": (
        "text, letters, words, readable text, brand name, BACARDI, logo, watermark, "
        "blurry, low quality, extra bat, distorted bat, gray, brown, green, blue"
    )}))
    g.update(node("5", "ControlNetLoader", {"control_net_name": CN_INPAINT}))
    g.update(node("6", "ControlNetApply", {
        "conditioning": ["pos_b", 0], "control_net": ["5", 0], "image": ["2", 0], "strength": 0.92
    }))
    g.update(node("7", "KSampler", {
        "model": ["1", 0], "positive": ["6", 0], "negative": ["neg_b", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 35, "cfg": 7.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0
    }))
    g.update(node("8", "VAEDecode", {"samples": ["7", 0], "vae": ["1", 2]}))
    g.update(node("9", "SaveImage", {"images": ["8", 0], "filename_prefix": "v269_clean"}))
    return g


def build_full_workflow(clean_image_name, seed_base=269100):
    """完整管线：先加载已干净的底图，再裂变 3 变体."""
    g = {}
    g.update(node("1", "CheckpointLoaderSimple", {"ckpt_name": CKPT}))
    g.update(node("ref", "LoadImage", {"image": clean_image_name}))

    # IPAdapter 风格锁
    g.update(node("ipl", "IPAdapterUnifiedLoader", {"model": ["1", 0], "preset": "PLUS (high strength)"}))
    g.update(node("ipa", "IPAdapterAdvanced", {
        "model": ["1", 0], "ipadapter": ["ipl", 1], "image": ["ref", 0],
        "weight": 0.52, "weight_type": "style transfer", "combine_embeds": "average",
        "start_at": 0.0, "end_at": 0.82, "noise": 0.05, "embeds_scaling": "V only"
    }))

    # Canny 锁排版
    g.update(node("canny", "CannyEdgePreprocessor", {
        "image": ["ref", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024
    }))
    g.update(node("cn_load", "ControlNetLoader", {"control_net_name": CN_CANNY}))

    # 共用 negative
    neg_text = (
        "text, letters, words, readable text, brand name, BACARDI, logo, watermark, "
        "3d, metallic, glossy, photorealistic, multiple bats, extra wings, deformed, "
        "blurry, low quality, gray, brown, green, blue, beige, desaturated"
    )
    g.update(node("neg", "CLIPTextEncode", {"clip": ["1", 1], "text": neg_text}))

    base_pos = (
        "a single stylized 2D SOLID FLAT BLACK bat silhouette, gothic vintage emblem, "
        "perfectly centered inside a circular ring, "
        "saturated purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F, "
        "SOLID FILLED FLAT SHAPE, clean outline, flat printed vector graphic, "
        "grunge scratched texture background, no text, no letters, no words"
    )

    variants = [
        ("up",    base_pos + ", wings RAISED UPWARD into a sharp V shape, angular wing tips, fierce silhouette"),
        ("wide",  base_pos + ", wings spread WIDE and horizontal with scalloped membrane edges, powerful silhouette"),
        ("fold",  base_pos + ", wings FOLDED DOWNWARD close to the body, slender body, calm elegant silhouette"),
    ]

    for i, (tag, pos) in enumerate(variants):
        sid = f"s_{tag}"
        g.update(node(f"pos_{tag}", "CLIPTextEncode", {"clip": ["1", 1], "text": pos}))
        g.update(node(f"cn_{tag}", "ControlNetApply", {
            "conditioning": [f"pos_{tag}", 0], "control_net": ["cn_load", 0], "image": ["canny", 0], "strength": 0.62
        }))
        g.update(node(f"lat_{tag}", "EmptyLatentImage", {"width": TARGET_W, "height": TARGET_H, "batch_size": 1}))
        g.update(node(f"sample_{tag}", "KSampler", {
            "model": ["ipa", 0], "positive": [f"cn_{tag}", 0], "negative": ["neg", 0],
            "latent_image": [f"lat_{tag}", 0], "seed": seed_base + i * 37,
            "steps": 32, "cfg": 6.5, "sampler_name": "euler", "scheduler": "normal", "denoise": 0.80
        }))
        g.update(node(f"dec_{tag}", "VAEDecode", {"samples": [f"sample_{tag}", 0], "vae": ["1", 2]}))
        g.update(node(f"save_{tag}", "SaveImage", {"images": [f"dec_{tag}", 0], "filename_prefix": f"v269_{tag}"}))
    return g


def submit(wf):
    data = json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def poll(pid, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(4)
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=20) as r:
                h = json.loads(r.read())
        except Exception:
            continue
        if pid in h:
            e = h[pid]
            if e.get("status", {}).get("completed"):
                return e
            err = e.get("status", {}).get("error")
            if err:
                raise RuntimeError(str(err))
    raise TimeoutError("timeout")


def collect_out(entry):
    outs = []
    for nid, node in entry.get("outputs", {}).items():
        if "images" in node:
            for im in node["images"]:
                outs.append(Path(COMFY_OUTPUT) / im["filename"])
    return outs


def calibrate_font(text, font_path, start_size, max_w):
    """二分找不超出 max_w 的最大字号."""
    lo, hi = 8, start_size
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        w = font.getlength(text)
        if w <= max_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def burn_text(img, big, arc, sub, color=(30, 20, 30)):
    """在干净裂变图上烧新字（无旧字，不是遮盖）."""
    sys.path.insert(0, str(PROJECT / "src"))
    import arc_text
    img = img.convert("RGB")
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img)

    # 顶弧字（沿用 v268 在 1552x2000 上的几何，按 TARGET_H 缩放）
    cy = int(820 * SCALE_Y)
    radius = int(520 * SCALE_Y)
    fs_arc = int(86 * SCALE_Y)
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT_PATH, fs_arc, radius, char_spacing_px=1)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 108:
            break
        fs_arc = int(fs_arc * 0.92)
    img = arc_text.draw_arc_text(img, arc, FONT_PATH, fs_arc, color,
                                 center=(w // 2, cy), radius=radius,
                                 start_angle_deg=225, end_angle_deg=315,
                                 char_spacing_px=1, flip_180=False)

    # 中心大字
    fs_big = calibrate_font(big, FONT_PATH, int(h * 0.16), max_w=int(w * 0.80))
    font = ImageFont.truetype(FONT_PATH, fs_big)
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, int(h * 0.555)), big, font=font, fill=color, anchor="mm")

    # 底部副字
    fs_sub = calibrate_font(sub, FONT_PATH, int(h * 0.085), max_w=int(w * 0.60))
    font = ImageFont.truetype(FONT_PATH, fs_sub)
    draw.text((w // 2, int(h * 0.655)), sub, font=font, fill=color, anchor="mm")

    # 两侧 EST. / 年份
    f_side = ImageFont.truetype(FONT_PATH, int(h * 0.032))
    draw.text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    draw.text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")
    return img


TEXT_VARIANTS = [
    ("NIGHTBAT", "WINGS OF SHADOW", "SILENT HUNT"),
    ("DUSKBAT",  "SHADOW OF THE WING", "ECHO FLIGHT"),
    ("MOONBAT",  "GUARDIAN OF THE DARK", "SONAR OATH"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="只跑 Stage B 去字测试")
    ap.add_argument("--seed", type=int, default=269001)
    ap.add_argument("--force", action="store_true", help="不复用已有 raw")
    args = ap.parse_args()

    save_inputs()

    if args.test:
        print("[v269 TEST] submit Stage B clean only ...")
        wf = build_clean_workflow(seed=args.seed)
        r = submit(wf)
        if "error" in r:
            print("COMFY ERROR:", r["error"])
            return
        entry = poll(r["prompt_id"])
        outs = collect_out(entry)
        print("clean outputs:", outs)
        return

    # ---- 1) Stage B: 生成干净底图 ----
    clean_file = COMFY_OUTPUT / f"v269_clean_{args.seed:06d}.png"
    existing_clean = sorted(COMFY_OUTPUT.glob("v269_clean_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if existing_clean and not args.force:
        clean_file = existing_clean[0]
        print(f"[v269] reuse clean {clean_file.name}")
    else:
        print("[v269] Stage B: inpaint-dreamer clean ...")
        wf = build_clean_workflow(seed=args.seed)
        r = submit(wf)
        if "error" in r:
            print("COMFY ERROR:", r["error"])
            return
        entry = poll(r["prompt_id"])
        outs = collect_out(entry)
        if not outs:
            print("no clean output"); return
        clean_file = outs[0]
        print(f"  clean -> {clean_file}")

    # 复制到 input 供 Stage A 引用
    clean_input = COMFY_INPUT / "v269_clean.png"
    Image.open(clean_file).convert("RGB").save(clean_input, quality=98)

    # ---- 2) Stage A: 裂变 3 变体 ----
    print("[v269] Stage A: IPAdapter+Canny fission ...")
    wf = build_full_workflow("v269_clean.png", seed_base=args.seed + 100)
    r = submit(wf)
    if "error" in r:
        print("COMFY ERROR:", r["error"])
        return
    entry = poll(r["prompt_id"])
    outs = collect_out(entry)
    print("fission outputs:", [o.name for o in outs])

    # ---- 3) Stage C: 烧新字 ----
    print("[v269] Stage C: burn new text ...")
    finals = []
    for i, path in enumerate(sorted(outs)):
        tag = ["up", "wide", "fold"][i % 3]
        big, arc, sub = TEXT_VARIANTS[i % 3]
        raw = Image.open(path).convert("RGB")
        final = burn_text(raw, big, arc, sub)
        out_path = JOB / f"v269_{tag}_final.png"
        final.save(out_path, quality=95)
        print(f"  {tag} -> {out_path}  ({big} / {arc} / {sub})")
        finals.append(out_path)

    # 拼对照图
    if finals:
        imgs = [Image.open(p).convert("RGB") for p in finals]
        w, h = imgs[0].size
        gap = 16
        grid = Image.new("RGB", (w * len(imgs) + gap * (len(imgs) + 1), h), "white")
        for i, im in enumerate(imgs):
            grid.paste(im, (gap + i * (w + gap), 0))
        grid_path = JOB / "_grid_v269.png"
        grid.save(grid_path, quality=92)
        print(f"\n[OK] grid -> {grid_path}")


if __name__ == "__main__":
    main()
