"""v271 — 在 v264 基础上用 LaMa 只清主/副/Est 旧字 ghost，保留圆环结构。"""
import argparse
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import time
import uuid
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "v271"
JOB.mkdir(parents=True, exist_ok=True)
REF = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

sys.path.insert(0, str(PROJECT / "src"))
spec = importlib.util.spec_from_file_location("v253", str(PROJECT / "src" / "v253_bat_logo_inpaint.py"))
v253 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v253)

MODEL_PATH = PROJECT / "ComfyUI" / "models" / "RMBG" / "Lama" / "big-lama.pt"
FONT_PATH = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")
INK = (26, 10, 31)
_LAMA = None


def load_lama():
    global _LAMA
    if _LAMA is not None:
        return _LAMA
    src = str(MODEL_PATH)
    if any(ord(c) > 127 for c in src):
        tmp = Path(tempfile.gettempdir()) / "big-lama.pt"
        if not tmp.exists() or tmp.stat().st_size != Path(src).stat().st_size:
            shutil.copyfile(src, tmp)
        src = str(tmp)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(src, map_location=dev)
    model.eval().to(dev)
    _LAMA = (model, dev)
    return _LAMA


def pad_image(image, is_mask=False):
    w, h = image.size
    if w % 8 != 0: w += 8 - w % 8
    if h % 8 != 0: h += 8 - h % 8
    fill = 0 if is_mask else None
    padded = Image.new(image.mode, (w, h), color=fill)
    padded.paste(image, (0, 0))
    return padded


def lama_inpaint(src_pil, mask_pil, removal_strength=230, edge_smoothness=4):
    model, dev = load_lama()
    w, h = src_pil.size
    p_img = pad_image(src_pil)
    p_mask = pad_image(mask_pil, is_mask=True)
    if p_mask.size != p_img.size:
        p_mask = p_mask.resize(p_img.size, ImageFilter.LANCZOS)
    p_mask = ImageOps.invert(p_mask)
    p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=edge_smoothness))
    gray = p_mask.point(lambda x: 0 if x > removal_strength else 255)
    img_t = torch.from_numpy(np.array(p_img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    mask_t = torch.from_numpy(np.array(gray).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.inference_mode():
        res = model(img_t, mask_t)
    res_img = Image.fromarray((res[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    if res_img.width > w or res_img.height > h:
        res_img = res_img.crop((0, 0, w, h))
    return res_img.convert("RGB")


def make_style_ref():
    orig = Image.open(COMFY_INPUT / REF).convert("RGB")
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    base, bat_mask = v253.clean_base(orig, bgr)
    # LaMa 只清主/副/Est/1862，不动弧带和圆环
    mask = Image.new("L", base.size, 0)
    d = ImageDraw.Draw(mask)
    d.rectangle([180, 980, 1370, 1170], fill=255)   # main
    d.rectangle([360, 1180, 1195, 1340], fill=255)  # sub
    d.rectangle([350, 740, 560, 860], fill=255)      # EST.
    d.rectangle([1000, 740, 1220, 860], fill=255)   # 1862
    mask = mask.filter(ImageFilter.MaxFilter(15))
    base = lama_inpaint(base, mask, removal_strength=235, edge_smoothness=4)
    base.save(str(COMFY_INPUT / "v271_style_ref.png"), quality=98)
    base.save(str(JOB / "v271_style_ref.png"), quality=98)
    print(f"[v271] style_ref -> {COMFY_INPUT / 'v271_style_ref.png'}")
    return base, bat_mask


def make_bat_mask_image(bat_mask):
    h, w = bat_mask.shape[:2]
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = (bat_mask.astype(np.uint8) * 255)
    Image.fromarray(rgba, "RGBA").save(str(COMFY_INPUT / "v271_bat_mask.png"))
    return "v271_bat_mask.png"


BAT_STYLE = (
    "a single stylized 2D SOLID FLAT BLACK bat silhouette, gothic vintage emblem, "
    "SOLID FILLED FLAT SHAPE, NO internal detail, NO shading, NO gradient, NO texture, "
    "clean outline only, flat printed vector graphic, "
    "saturated purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F, "
    "perfectly centered inside the circular ring, NO shadow, NO ground plane, "
)

VARIANTS = [
    ("up",    BAT_STYLE + "wings RAISED UPWARD into a sharp V shape with pointed angular wing tips, fierce silhouette",
     "NIGHTBAT", "SHADOW OF THE WING", "ECHO HUNT"),
    ("spread", BAT_STYLE + "wings spread WIDE and horizontal with SCALLOPED membrane edges, powerful silhouette",
     "DUSKBAT", "WINGS OF TWILIGHT", "SILENT FLIGHT"),
    ("fold",  BAT_STYLE + "wings FOLDED DOWNWARD close to the body, calm elegant silhouette",
     "MOONBAT", "GUARDIAN OF THE DARK", "SONAR OATH"),
]
POS = {tag: pos for tag, pos, *_ in VARIANTS}

NEG = (
    "3d, metallic, glossy, jewelry, gradient shading, smooth shading, soft airbrush, "
    "photorealistic, realistic, hyperrealistic, watercolor, "
    "text, letters, words, readable text, brand name, BACARDI, logo, monogram, "
    "multiple bats, second bat, extra wings, extra limbs, deformed, mutated, malformed, "
    "blurry, soft focus, low quality, jagged, noise, grain, "
    "gray, grayish, brown, green, blue, cyan, beige, tan, desaturated, washed out, off-palette"
)


def build_workflow(tag, seed, ipa=0.5, denoise=0.80, canny=0.6):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": v253.CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v271_style_ref.png"}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": "v271_bat_mask.png"}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["2", 0],
        "weight": ipa, "weight_type": "style transfer", "combine_embeds": "average",
        "start_at": 0.0, "end_at": 0.82, "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": v253.LORA,
        "strength_model": v253.LORA_DETAIL, "strength_clip": v253.LORA_DETAIL}}
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["2", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": v253.CN_CANNY}}
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": POS[tag]}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pos", 0], "control_net": ["21", 0], "image": ["20", 0], "strength": canny}}
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 1], "grow_mask_by": 12}}
    g["41"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["4", 0], "mask": ["3", 1]}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["22", 0], "negative": ["neg", 0],
        "latent_image": ["41", 0], "seed": seed, "steps": 30, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": denoise}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v271_{tag}"}}
    return g


def submit(wf):
    data = json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def poll(pid, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(4)
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
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
    for node in entry.get("outputs", {}).values():
        if "images" in node:
            for im in node["images"]:
                outs.append(Path(COMFY_OUTPUT) / im["filename"])
    return outs


def composite_bat(style_ref, raw, bat_mask):
    h, w = bat_mask.shape[:2]
    raw_bgr = cv2.cvtColor(np.array(raw.convert("RGB").resize((w, h), Image.LANCZOS)), cv2.COLOR_RGB2BGR)
    orig_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF).convert("RGB")), cv2.COLOR_RGB2BGR)
    locked = v253.match_hist_lab(orig_bgr, raw_bgr)
    locked = v253.snap_to_original(locked, orig_bgr, v253.BAT_BBOX, n_colors=12)
    locked_rgb = Image.fromarray(cv2.cvtColor(locked, cv2.COLOR_BGR2RGB))
    alpha = Image.fromarray((bat_mask.astype(np.uint8) * 255), "L")
    comp = style_ref.convert("RGBA").copy()
    comp.paste(locked_rgb.convert("RGBA"), (0, 0), alpha)
    return comp.convert("RGB")


def calibrate(text, target_w, lo=4, hi=400):
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(FONT_PATH, mid)
        bb = ImageDraw.Draw(Image.new("RGB", (10, 10))).textbbox((0, 0), text, font=f)
        if (bb[2] - bb[0]) < target_w: lo = mid
        else: hi = mid
    return lo


def burn_text(img, big, arc, sub, est_year="1862"):
    img = img.convert("RGB")
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img)
    fs_arc = int(h * 0.045)
    from arc_text import draw_arc_text, fit_arc_text_width
    radius = int(w * 0.45)
    arc_len = fit_arc_text_width(arc, FONT_PATH, fs_arc, radius, char_spacing_px=8)
    total_deg = math.degrees(arc_len / radius)
    start = 270 - total_deg / 2; end = 270 + total_deg / 2
    img = draw_arc_text(img, arc, FONT_PATH, fs_arc, INK,
                        (w // 2, int(h * 0.13) + radius), radius,
                        start, end, char_spacing_px=8)
    fs_big = calibrate(big, int(w * 0.62))
    draw.text((w // 2, int(h * 0.535)), big, font=ImageFont.truetype(FONT_PATH, fs_big), fill=INK, anchor="mm")
    fs_sub = calibrate(sub, int(w * 0.54))
    draw.text((w // 2, int(h * 0.642)), sub, font=ImageFont.truetype(FONT_PATH, fs_sub), fill=INK, anchor="mm")
    f_side = ImageFont.truetype(FONT_PATH, int(h * 0.022))
    draw.text((int(w * 0.30), int(h * 0.745)), "EST.", font=f_side, fill=INK, anchor="mm")
    draw.text((int(w * 0.70), int(h * 0.745)), est_year, font=f_side, fill=INK, anchor="mm")
    return img


def qc_palette(img):
    rgb = np.array(img.convert("RGB")).reshape(-1, 3).astype(int)
    bright = rgb[(rgb[:, 0] > 45) & (rgb[:, 1] > 45) & (rgb[:, 2] > 45)]
    if len(bright) == 0: return 0.0
    r, g, b = bright[:, 0], bright[:, 1], bright[:, 2]
    off = ((r - b) < -25) | ((g - b) > 25) | ((abs(r - g) < 25) & (abs(g - b) < 25) & (abs(r - b) < 25) & (r > 90))
    return round(float(off.mean() * 100), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("[v271] make style ref + masks ...")
    style_ref, bat_mask = make_style_ref()
    make_bat_mask_image(bat_mask)

    finals = []
    for i, (tag, _pos, big, arc, sub) in enumerate(VARIANTS):
        seed = 271000 + i * 41
        existing = sorted(COMFY_OUTPUT.glob(f"v271_{tag}*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if existing and not args.force:
            raw = existing[0]
            print(f"\n=== [v271 {tag}] reuse raw {raw.name} ===")
        else:
            wf = build_workflow(tag, seed)
            print(f"\n=== [v271 {tag}] submit seed={seed} ===")
            try:
                r = submit(wf)
            except Exception as e:
                print(f"  submit FAIL: {e}"); continue
            if "error" in r:
                print(f"  COMFY ERROR: {r['error']}"); continue
            try:
                entry = poll(r["prompt_id"])
            except Exception as e:
                print(f"  poll FAIL: {e}"); continue
            outs = collect_out(entry)
            if not outs:
                print("  no output"); continue
            raw = outs[0]
            print(f"  raw -> {raw}")
        comp = composite_bat(style_ref, Image.open(raw), bat_mask)
        final = burn_text(comp, big, arc, sub)
        off = qc_palette(final)
        out = JOB / f"v271_{tag}_final.png"
        final.save(str(out), quality=95)
        print(f"  final -> {out}  off-palette={off}%  {big} / {arc} / {sub}")
        finals.append(out)

    if finals:
        imgs = [Image.open(p).convert("RGB") for p in finals]
        w, h = imgs[0].size
        gap = 14
        grid = Image.new("RGB", (w * len(imgs) + gap * (len(imgs) + 1), h), "white")
        for i, im in enumerate(imgs):
            grid.paste(im, (gap + i * (w + gap), 0))
        grid.save(str(JOB / "_grid_v271.png"), quality=92)
        print(f"\n[OK] grid -> {JOB / '_grid_v271.png'}")


if __name__ == "__main__":
    main()
