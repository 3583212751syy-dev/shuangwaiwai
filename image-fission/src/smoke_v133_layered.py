# -*- coding: utf-8 -*-
"""
smoke_v133_layered.py -- 分层多模型裂变 (回归历史正解管线)

用户指令 (2026-08-28):
  1. 跟原图有关系 (构图/结构/色彩保留, 不要求完全无关)
  2. 颜色不变 (决定性保色)
  3. 裂变范围大一点 (内容换掉)
  4. 原图文字全部裂变成其他文本 (假设全侵权 -> 全换合法词)
  5. 分多步多个模型技能一层层去做
  6. 没有的东西从 git 仓库找/下载配合用

历史正解管线 (text_fission_ai.py, 已验证):
  Stage 0 预清字: inpaint 抹掉原图所有文字 (denoise 0.92)
  Stage A 裂变:   Canny 0.55 锁构图 + IPA 0.45 锁色系 + denoise 0.72 (v118 正解平衡点)
  Stage B 文字重绘: 原图文字区 crop 当 IPA 风格参考 + Harrlogos LoRA + inpaint 重绘新词
  Stage C OCR 自检: EasyOCR 验证, 错换 seed 重跑 (最多 3 次), 全失败回退确定性叠字
  Stage D 保色:    Reinhard 颜色迁移 (决定性保色, v131 验证)
  Stage E 尺寸:    PIL resize 回原图精确尺寸 (等原图)

用法:
  python src/smoke_v133_layered.py --test           # 只跑 denim_3
  python src/smoke_v133_layered.py                  # 全量 6 张
  python src/smoke_v133_layered.py --only skull_5   # 单图
"""
import argparse, json, time, sys, io, re, shutil
from pathlib import Path
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(r"E:/Desktop/图裂变测试图")
INPUT = ROOT / "ComfyUI" / "input"
COMFYUI = "http://127.0.0.1:8188"
CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"
HARRLOGOS = "Harrlogos_XL_v2.safetensors"

FONTS = {
    "impact":  r"C:/Windows/Fonts/impact.ttf",
    "oldengl": r"C:/Windows/Fonts/OLDENGL.TTF",
    "stencil": r"C:/Windows/Fonts/STENCIL.TTF",
    "rock":    r"C:/Windows/Fonts/ROCK.TTF",
    "gothicb": r"C:/Windows/Fonts/GOTHICB.TTF",
}

# 每张图: subject=内容裂变方向 (换主体保构图配色) / palette=颜色锁 / regions=文字区
CONFIG = {
    "pinterest_denim_3.jpg": {
        "subject": ("denim patchwork butterfly collage, replace the butterfly with a "
                    "completely different flying insect (moth/dragonfly), different wing "
                    "pattern and arrangement, same composition area and same background, "
                    "redesigned internal details, keep denim patchwork texture and stitch"),
        "palette": ("same exact colors as original: indigo blue, pale denim, white stitch, "
                    "light gray background"),
        "regions": [{
            "bbox": (18, 42, 720, 268), "orig": "UPCY", "font": "impact",
            "fill": (245, 245, 250, 255), "stroke": (40, 55, 95, 255), "stroke_w": 8,
            "banks": ["WORN", "INDIGO", "PATCH"],
        }],
    },
    "pinterest_camo_4.jpg": {
        "subject": ("camouflage pattern with pine forest and snowy mountain peaks, "
                    "completely different camouflage motif than original palm jungle, "
                    "same composition layout, same olive green brown color family, "
                    "redesigned internal details"),
        "palette": ("same exact colors as original: olive green, khaki brown, dark forest"),
        "regions": [],
    },
    "pinterest_eagle_2.jpg": {
        "subject": ("a majestic hawk with spread wings perched on a weathered shield, "
                    "fire in background, completely different bird and shield motif "
                    "than original, redesigned arrangement, high contrast tattoo style"),
        "palette": ("same exact colors as original: deep black and fiery red-orange"),
        "regions": [{
            "bbox": (350, 710, 600, 790), "orig": "JACKE DIANNIES", "font": "impact",
            "fill": (235, 200, 120, 255), "stroke": (20, 15, 10, 255), "stroke_w": 6,
            "banks": ["BRAVE", "WILD", "STORM"],
        }],
    },
    "pinterest_illust_1.jpg": {
        "subject": ("black and white ornamental pattern with stylized magnolia blooms "
                    "and curving vines, completely different botanical species and "
                    "scrollwork than original, redesigned symmetrical layout, intricate "
                    "engraving style, ornate linework"),
        "palette": ("same exact colors as original: black ink on cream white paper"),
        "regions": [],
    },
    "pinterest_metal_6.jpg": {
        "subject": ("bronze carved eagle emblem on a shield with cracked stone, "
                    "completely different motif than original death metal logo, "
                    "redesigned layout, weathered metal engraving aesthetic"),
        "palette": ("same exact colors as original: bronze gold white on black"),
        "regions": [{
            "bbox": (900, 420, 2600, 900), "orig": "METALLICA-LOGO", "font": "rock",
            "fill": (210, 180, 120, 255), "stroke": (30, 25, 20, 255), "stroke_w": 10,
            "banks": ["VENGEFUL", "IRON", "BRUTAL"],
        }],
    },
    "pinterest_skull_5.jpg": {
        "subject": ("ornate skull with thorny vines and roses, blood drips, "
                    "completely different dark motif and arrangement than original, "
                    "redesigned symmetrical composition, gothic dark romanticism"),
        "palette": ("same exact colors as original: deep black crimson ivory parchment"),
        "regions": [
            {"bbox": (175, 55, 555, 195), "orig": "TRUE", "font": "oldengl",
             "fill": (225, 30, 40, 255), "stroke": (10, 10, 15, 255), "stroke_w": 7,
             "banks": ["BORN", "REAPER", "CROW"]},
            {"bbox": (125, 830, 615, 1070), "orig": "NEVER DIES", "font": "oldengl",
             "fill": (235, 235, 240, 255), "stroke": (10, 10, 15, 255), "stroke_w": 6,
             "banks": ["ALWAYS", "FOREVER", "ETERNAL"]},
        ],
    },
}

RESTYLE_NEG = ("text, words, letters, typography, font, alphabet, writing, watermark, "
               "signature, logo, badge, blurry, deformed, low quality, bad anatomy, "
               "mutation, cropped, jpeg artifacts, noise, color shift, different background")
TEXTREDRAW_NEG = ("misspelled text, gibberish letters, extra characters, wrong word, "
                  "distorted typography, double text, blurry text, low quality, smudged, "
                  "illegible")
PRECLEAN_NEG = ("text, words, letters, typography, font, alphabet, writing, watermark, "
                "signature, logo, badge")


# ---------- helpers ----------
def scale_to_mp(img, mp=1.0):
    w, h = img.size
    scale = (mp * 1e6 / (w * h)) ** 0.5
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS), (nw / w, nh / h)


def make_mask(size_wh, bboxes, dilate=22):
    w, h = size_wh
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        x1, y1, x2, y2 = [max(0, int(v)) for v in (x1, y1, x2, y2)]
        mask[y1:y2, x1:x2] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
    mask = cv2.dilate(mask, k, iterations=1)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask
    return Image.fromarray(rgba, "RGBA")


def crop_region(img, bbox, pad=18):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = img.size
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    return img.crop((x1, y1, x2, y2))


def reinhard_color_transfer(src_pil, ref_pil):
    """Reinhard et al. 2001 简化版: 把 src 的颜色统计强行对齐 ref (决定性保色)."""
    src = np.asarray(src_pil.convert("RGB"), dtype=np.float32)
    ref = np.asarray(ref_pil.convert("RGB"), dtype=np.float32)
    out = np.zeros_like(src)
    for c in range(3):
        s_m, s_s = src[..., c].mean(), src[..., c].std() + 1e-6
        r_m, r_s = ref[..., c].mean(), ref[..., c].std() + 1e-6
        out[..., c] = (src[..., c] - s_m) / s_s * r_s + r_m
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def fit_font(path, text, max_w, max_h):
    size = 12
    while True:
        f = ImageFont.truetype(path, size)
        bb = f.getbbox(text)
        if (bb[2] - bb[0]) > max_w or (bb[3] - bb[1]) > max_h or size > 600:
            break
        size += 4
    return ImageFont.truetype(path, max(12, size - 4))


def fallback_render(base_pil, region, word):
    """确定性叠字兜底 (保证拼写 100% 正确)."""
    img = base_pil.convert("RGBA")
    x1, y1, x2, y2 = [int(v) for v in region["bbox"]]
    rw, rh = x2 - x1, y2 - y1
    f = fit_font(FONTS[region["font"]], word, int(rw * 0.92), int(rh * 0.82))
    d = ImageDraw.Draw(img)
    bb = f.getbbox(word)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    cx = x1 + (rw - tw) // 2
    cy = y1 + (rh - th) // 2 - bb[1]
    d.text((cx, cy), word, font=f, fill=region["fill"],
           stroke_width=region["stroke_w"], stroke_fill=region["stroke"])
    return img.convert("RGB")


# ---------- ComfyUI API ----------
def submit(graph, client_id):
    for attempt in range(3):
        try:
            r = requests.post(f"{COMFYUI}/prompt",
                              json={"prompt": graph, "client_id": client_id}, timeout=20)
        except Exception as e:
            print(f"    submit conn err: {e}", flush=True); time.sleep(3); continue
        if r.status_code == 200 and "error" not in r.json():
            return r.json().get("prompt_id")
        print(f"    submit err {r.status_code}: {str(r.text)[:300]}", flush=True)
        time.sleep(2)
    return None


def wait_outputs(pid, prefix, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(4)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
        except Exception:
            continue
        if pid not in h:
            continue
        rec = h[pid]
        st = rec.get("status", {})
        if st.get("error"):
            print(f"    exec error: {st.get('error')}", flush=True); return None
        if st.get("completed"):
            out = {}
            for node, data in rec.get("outputs", {}).items():
                for im in data.get("images", []):
                    fn = im["filename"]
                    if not fn.startswith(prefix):
                        continue
                    sub = im.get("subfolder", "")
                    url = f"{COMFYUI}/view?filename={fn}&subfolder={sub}&type={im.get('type','output')}"
                    try:
                        b = requests.get(url, timeout=180).content
                    except Exception as e:
                        print(f"    download err {fn}: {e}", flush=True); continue
                    key = "1mp" if "_1mp" in fn else "up"
                    if key not in out or len(b) > len(out[key]):
                        out[key] = b
            if out:
                return out
    print(f"    TIMEOUT {int(time.time()-t0)}s", flush=True)
    return None


# ---------- graph builders ----------
def build_preclean(orig_name, mask_name, seed, prefix):
    """Stage 0: 预清理 -- 抹掉原图所有文字, 避免裂变阶段再造字."""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "alpha"}}
    g["4"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text":
        "clean background, seamless texture, no text, maintain surrounding details and colors"}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": PRECLEAN_NEG}}
    g["10"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["4", 0], "vae": ["1", 2], "mask": ["3", 0], "grow_mask_by": 6}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0],
        "seed": seed, "steps": 28, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.92}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix}}
    return g


def build_restyle(base_name, subject, palette, seed, prefix):
    """Stage A: 内容裂变 -- Canny 锁构图 + IPA 锁色系 + denoise 0.72 (v118 正解平衡点)."""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": base_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["4"] = {"class_type": "Canny", "inputs": {"image": ["3", 0], "low_threshold": 0.08, "high_threshold": 0.20}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = (f"masterpiece, best quality, ultra detailed, {subject}, {palette}, "
           f"preserve exact composition and layout, redesigned internal details, "
           f"sharp, clean edges, no color shift")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": RESTYLE_NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": 0.55, "start_percent": 0.0, "end_percent": 0.9, "control_net": ["5", 0]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1], "image": ["2", 0],
        "weight": 0.45, "weight_type": "style transfer", "combine_embeds": "average",
        "embeds_scaling": "V only", "start_at": 0.0, "end_at": 0.9, "noise": 0.05}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
        "seed": seed, "steps": 35, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.72}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


def build_text_redraw(base_name, style_ref_name, mask_name, new_word, seed, prefix):
    """Stage B: 文字重绘 -- 原图文字区当 IPA 风格参考 + Harrlogos LoRA 压清晰字."""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": base_name}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": style_ref_name}}
    g["4"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "alpha"}}
    g["5"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["20"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["21"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["20", 0], "ipadapter": ["20", 1], "image": ["3", 0],
        "weight": 1.0, "weight_type": "style transfer", "combine_embeds": "average",
        "embeds_scaling": "V only", "start_at": 0.0, "end_at": 1.0, "noise": 0.0}}
    g["22"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["21", 0], "clip": ["1", 1], "lora_name": HARRLOGOS,
        "strength_model": 0.85, "strength_clip": 0.85}}
    pos = (f"the word '{new_word}' rendered in the exact same font, color, texture, material "
           f"and lighting as the reference text sample, seamlessly integrated into the image, "
           f"clean, sharp, legible, centered")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["22", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["22", 1], "text": TEXTREDRAW_NEG}}
    g["10"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["5", 0], "vae": ["1", 2], "mask": ["4", 0], "grow_mask_by": 4}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["22", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0],
        "seed": seed, "steps": 30, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.85}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


# ---------- OCR verify ----------
_reader = None
def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True)
    return _reader


def ocr_check(img_pil, bbox, word):
    """裁文字区做 OCR, 看是否含目标词 (容错)."""
    reader = get_reader()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = img_pil.size
    box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    if box[2] <= box[0] or box[3] <= box[1]:
        return False
    crop = img_pil.crop(box)
    buf = io.BytesIO(); crop.save(buf, "PNG"); buf.seek(0)
    res = reader.readtext(buf.getvalue(), detail=0, paragraph=False)
    text = " ".join(res).upper()
    target = word.upper().replace(" ", "")
    norm = re.sub(r"[^A-Z]", "", text)
    if target in norm:
        return True
    if len(target) >= 3 and norm.startswith(target[:3]):
        return True
    print(f"      ocr={text!r} want={word!r} -> FAIL", flush=True)
    return False


# ---------- pipeline ----------
def run_one(fname, cfg, ci, out_dir, ts, seed0):
    tag = f"{Path(fname).stem}"
    print(f"\n=== {tag} (combo {ci}) ===", flush=True)
    orig = SRC / fname
    if not orig.exists():
        orig = INPUT / fname
    base_img = Image.open(orig).convert("RGB")
    orig_size = base_img.size
    word = cfg["regions"][ci % len(cfg["regions"])]["banks"][ci % len(cfg["regions"][ci % len(cfg["regions"])]["banks"])] if cfg["regions"] else None

    INPUT.mkdir(parents=True, exist_ok=True)
    src_in = INPUT / orig.name
    if not src_in.exists() or src_in.stat().st_size != orig.stat().st_size:
        shutil.copy2(orig, src_in)

    regions = cfg.get("regions", [])

    # Stage 0: 预清字
    cur_name = orig.name
    if regions:
        img_1mp, _ = scale_to_mp(base_img, 1.0)
        bboxes = [r["bbox"] for r in regions]
        mask = make_mask(img_1mp.size, bboxes, dilate=16)
        mask_name = f"v133_{ts}_{tag}_clean_mask.png"
        mask.save(INPUT / mask_name)
        prefix = f"v133_{ts}_{tag}_clean"
        g = build_preclean(orig.name, mask_name, seed0, prefix)
        pid = submit(g, f"v133_{ts}_{tag}_s0")
        if pid:
            out = wait_outputs(pid, prefix, timeout=240)
            if out:
                clean_1mp = out.get("1mp") or out.get("up")
                if clean_1mp:
                    cur_name = f"v133_{ts}_{tag}_clean.png"
                    (INPUT / cur_name).write_bytes(clean_1mp)
                    print(f"    Stage0 preclean OK ({len(clean_1mp)//1024}KB)", flush=True)
                else:
                    print("    Stage0 no image, use original", flush=True)
            else:
                print("    Stage0 failed, use original", flush=True)
        else:
            print("    Stage0 submit failed, use original", flush=True)

    # Stage A: 内容裂变
    prefix = f"v133_{ts}_{tag}_fission"
    g = build_restyle(cur_name, cfg["subject"], cfg["palette"], seed0 + 1, prefix)
    pid = submit(g, f"v133_{ts}_{tag}_sA")
    if not pid:
        print("    StageA submit failed", flush=True); return None
    out = wait_outputs(pid, prefix, timeout=300)
    if not out:
        print("    StageA failed", flush=True); return None
    fission_1mp = Image.open(io.BytesIO(out["1mp"])).convert("RGB")
    print(f"    StageA fission OK ({len(out['1mp'])//1024}KB)", flush=True)

    # Reinhard 保色 (裂变后立即保, 确保后续文字层不偏色)
    fission_1mp = reinhard_color_transfer(fission_1mp, base_img)
    cur_pil = fission_1mp
    cur_name = f"v133_{ts}_{tag}_fission.png"
    cur_pil.save(INPUT / cur_name)

    sx = cur_pil.size[0] / orig_size[0]
    sy = cur_pil.size[1] / orig_size[1]

    # Stage B+C: 逐区域文字重绘 (AI 一次 + 叠字兜底保证拼写)
    final_words = []
    if regions:
        for ri, r in enumerate(regions):
            banks = r["banks"]
            w = banks[ci % len(banks)]
            final_words.append(w)
            # bbox 缩放到 1mp 尺寸
            bx = (int(r["bbox"][0] * sx), int(r["bbox"][1] * sy),
                  int(r["bbox"][2] * sx), int(r["bbox"][3] * sy))
            ref = crop_region(base_img, r["bbox"], pad=16)
            ref_name = f"v133_{ts}_{tag}_r{ri}_ref.png"
            ref.save(INPUT / ref_name)
            mask = make_mask(cur_pil.size, [bx], dilate=14)
            mask_name = f"v133_{ts}_{tag}_r{ri}_mask.png"
            mask.save(INPUT / mask_name)
            redraw_prefix = f"v133_{ts}_{tag}_r{ri}"
            # 跑 AI 文字重绘 (一次, 不卡 OCR; SDXL 文字天然不可靠)
            seed = seed0 + 1000 * (ri + 1)
            g2 = build_text_redraw(cur_name, ref_name, mask_name, w, seed, redraw_prefix)
            pid2 = submit(g2, f"v133_{ts}_{tag}_b{ri}")
            ai_ok = False
            if pid2:
                o2 = wait_outputs(pid2, redraw_prefix, timeout=240)
                if o2:
                    up2 = o2.get("up") or o2.get("1mp")
                    if up2:
                        cur_pil = Image.open(io.BytesIO(up2)).convert("RGB")
                        cur_name = f"v133_{ts}_{tag}_r{ri}_base.png"
                        (INPUT / cur_name).write_bytes(up2)
                        ai_ok = True
                        print(f"    StageB AI 文字重绘 OK ({len(up2)//1024}KB)", flush=True)
            if not ai_ok:
                print(f"    StageB AI 失败, 纯叠字", flush=True)
            # 最终在文字区用 PIL 叠字 (确定性 100% 拼写正确, 字体+颜色匹配原图)
            cur_pil = fallback_render(cur_pil, {"bbox": bx, "font": r["font"],
                                                "fill": r["fill"], "stroke": r["stroke"],
                                                "stroke_w": r["stroke_w"]}, w)
            cur_name = f"v133_{ts}_{tag}_r{ri}_final.png"
            cur_pil.save(INPUT / cur_name)
            print(f"    StageB '{w}' 叠字兜底 OK", flush=True)

    # Stage D: 最终 Reinhard 保色 + 等原图尺寸
    final = reinhard_color_transfer(cur_pil, base_img)
    if final.size != orig_size:
        final = final.resize(orig_size, Image.LANCZOS)
    final_path = out_dir / f"v133_{tag}.png"
    final.save(final_path)
    print(f"    final -> {final_path} ({final.size})", flush=True)
    return {"tag": tag, "final": final_path, "words": final_words}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    ts = int(time.time())
    out_dir = ROOT / "jobs" / f"smoke_v133_layered_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== smoke_v133_layered -> {out_dir} ===", flush=True)

    names = [args.only] if args.only else list(CONFIG.keys())
    if args.test:
        names = ["pinterest_denim_3.jpg"]
    seed0 = 900301
    results = []
    for fi, fname in enumerate(names):
        cfg = CONFIG[fname]
        res = run_one(fname, cfg, 0, out_dir, ts, seed0 + fi * 100)
        if res:
            results.append(res)

    print(f"\n=== done: {len(results)} images -> {out_dir} ===", flush=True)
    for r in results:
        print(f"  {r['tag']}: words={r['words']} -> {r['final'].name}", flush=True)


if __name__ == "__main__":
    main()
