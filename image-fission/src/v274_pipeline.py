"""v274 — ComfyUI 整图裂变 + AbrilFatface 烧新字

基于 v147/v272 的整图 img2img 路线，但：
  - 更强的结构锁（Canny 0.65）+ 风格锁（IPAdapter style transfer 0.35）
  - denoise 0.72，在裂变蝙蝠姿态的同时避免飘成塔罗/曼陀罗
  - 负向强制排除可读字、品牌名、字母形
  - 输出 3 个变体，再烧新字
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
JOB = PROJECT / "jobs" / "v274"
JOB.mkdir(parents=True, exist_ok=True)

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
TARGET_W, TARGET_H = 1024, 1280

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

DENOISE = 0.72
IPA = 0.35
CANNY_S = 0.65

POS = (
    "vintage purple spirit brand emblem, a circular dark purple badge on a soft purple background, "
    "a stylized black bat silhouette in the center, wings spread, "
    "the top arched band and bottom banner contain PURE GEOMETRIC ORNAMENTS ONLY "
    "(concentric arcs, small triangles, dots, crescent moons) — "
    "ABSOLUTELY NO TEXT, NO LETTERS, NO WORDS, NO READABLE GLYPH, NO BRAND NAME, NO LOGO TEXT, "
    "grunge scratched texture, vintage craft spirits label art, "
    "saturated purple #6B2C8C and deep violet #2A0A3F and black only, "
    "sharp clean edges, bold emblem, high contrast"
)
NEG = (
    "text, letters, words, readable text, legible text, recognizable letters, "
    "brand name, BACARDI, logo word, spelled out words, letter-shaped forms, glyph, calligraphy, "
    "tarot, mandala, chandelier, ornamental frame, filigree, lace, "
    "3d render, photorealistic, painterly, watercolor, blur, soft focus, "
    "green, brown, gray, beige, orange, color shift, desaturated, "
    "deformed, mutated, extra wings, asymmetric error, duplicate of source"
)

VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING"),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT"),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK"),
]

INK = (28, 18, 30)

sys.path.insert(0, str(PROJECT / "src"))
import arc_text


def node(id_, class_type, inputs):
    return {id_: {"class_type": class_type, "inputs": inputs}}


def build(seed):
    g = {}
    g.update(node("1", "CheckpointLoaderSimple", {"ckpt_name": CKPT}))
    g.update(node("2", "LoadImage", {"image": ORIG}))
    g.update(node("3", "ImageScale", {"image": ["2", 0], "width": TARGET_W, "height": TARGET_H,
                                      "upscale_method": "lanczos", "crop": "disabled"}))
    g.update(node("4", "VAEEncode", {"pixels": ["3", 0], "vae": ["1", 2]}))
    g.update(node("5", "IPAdapterUnifiedLoader", {"model": ["1", 0], "preset": "PLUS (high strength)"}))
    g.update(node("6", "IPAdapterAdvanced", {"model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
                                             "weight": IPA, "weight_type": "style transfer",
                                             "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
                                             "noise": 0.03, "embeds_scaling": "V only"}))
    g.update(node("20", "CannyEdgePreprocessor", {"image": ["3", 0], "low_threshold": 0.10,
                                                 "high_threshold": 0.25, "resolution": 1024}))
    g.update(node("21", "ControlNetLoader", {"control_net_name": CN_CANNY}))
    g.update(node("22", "ControlNetApply", {"conditioning": ["pg", 0], "control_net": ["21", 0],
                                           "image": ["20", 0], "strength": CANNY_S}))
    g.update(node("pg", "CLIPTextEncode", {"clip": ["1", 1], "text": POS}))
    g.update(node("ng", "CLIPTextEncode", {"clip": ["1", 1], "text": NEG}))
    g.update(node("10", "KSampler", {"model": ["6", 0], "positive": ["22", 0], "negative": ["ng", 0],
                                    "latent_image": ["4", 0], "seed": seed, "steps": 32, "cfg": 7.0,
                                    "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}))
    g.update(node("12", "VAEDecode", {"samples": ["10", 0], "vae": ["1", 2]}))
    g.update(node("15", "SaveImage", {"images": ["12", 0], "filename_prefix": "v274_fission"}))
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
                p = Path(COMFY_OUTPUT) / im["filename"]
                if not p.exists():
                    p = Path(im.get("abs_path", p))
                if p.exists():
                    outs.append(p)
    return outs


def calibrate_font(text, font_path, start_size, max_w):
    lo, hi, best = 8, start_size, start_size
    while lo <= hi:
        mid = (lo + hi) // 2
        w = ImageFont.truetype(font_path, mid).getlength(text)
        if w <= max_w:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def burn_text(img, big, arc, sub, color=INK):
    img = img.convert("RGB")
    w, h = img.width, img.height

    fs_arc = int(w * 0.045)
    radius = int(w * 0.45)
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

    fs_big = calibrate_font(big, FONT, int(h * 0.16), max_w=int(w * 0.80))
    ImageDraw.Draw(img).text((w // 2, int(h * 0.555)), big,
                             font=ImageFont.truetype(FONT, fs_big), fill=color, anchor="mm")

    fs_sub = calibrate_font(sub, FONT, int(h * 0.085), max_w=int(w * 0.62))
    ImageDraw.Draw(img).text((w // 2, int(h * 0.655)), sub,
                             font=ImageFont.truetype(FONT, fs_sub), fill=color, anchor="mm")

    f_side = ImageFont.truetype(FONT, int(h * 0.032))
    ImageDraw.Draw(img).text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    ImageDraw.Draw(img).text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")
    return img


def ocr(path):
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return [(t, round(c, 2)) for _, t, c in reader.readtext(np.array(Image.open(path).convert("RGB")))]
    except Exception as e:
        return [("OCR_FAIL", str(e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=274001)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("[v274] ComfyUI 整图裂变 + 烧字 ...")
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H), "white")
    grid.paste(Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H)), (0, 0))

    fission_paths = {}
    for tag, _, _, _ in VARIANTS:
        cands = sorted(COMFY_OUTPUT.glob(f"v274_fission_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands or args.force:
            wf = build(args.seed + hash(tag) % 100000)
            r = submit(wf)
            e = poll(r["prompt_id"])
            outs = collect_out(e)
            if not outs:
                print(f"  {tag}: Stage A 无输出"); continue
            cands = outs
        # 取最新一张未分配的
        fp = cands[0]
        fission_paths[tag] = fp
        print(f"  {tag} fission -> {fp.name}")

    for tag, arc, big, sub in VARIANTS:
        if tag not in fission_paths:
            continue
        fission = Image.open(fission_paths[tag]).convert("RGB")
        final = burn_text(fission, big, arc, sub)
        fp = JOB / f"v274_{tag}_final.png"
        final.save(fp, quality=95)
        print(f"  {tag} final -> {fp.name}")
        print(f"    OCR: {ocr(fp)}")

        if tag == "up":
            grid.paste(final, (TARGET_W + 24, 0))

    grid.save(JOB / "_grid_v274.png", quality=92)
    print(f"  grid -> {JOB / '_grid_v274.png'}")

    # 三变体对照
    vg = Image.new("RGB", (TARGET_W * 3 + 24, TARGET_H), "white")
    for i, (tag, _, _, _) in enumerate(VARIANTS):
        p = JOB / f"v274_{tag}_final.png"
        if p.exists():
            vg.paste(Image.open(p).convert("RGB"), (i * (TARGET_W + 12), 0))
    vg.save(JOB / "_variants_v274.png", quality=92)
    print(f"  variants -> {JOB / '_variants_v274.png'}")


if __name__ == "__main__":
    main()
