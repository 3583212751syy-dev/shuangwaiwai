# -*- coding: utf-8 -*-
"""
text_fission_ai.py -- AI 文本重绘 + 风格裂变 (用户 2026-08-27 指令)

规则:
1. 输入 = 桌面「图裂变测试图」原始 jpg, 绝不用裂变后的图当输入。
2. 图片裂变: 同主体、换明显不同的画风 (Canny 锁主体 + 高 denoise)。
3. 文本裂变: 新词提取「原图文字样式」-- 把原图文字区域 crop 当 IPAdapter 风格参考 +
   Harrlogos LoRA 压清晰字, AI 重绘进裂变图; 单词换成相关词 (侵权词必换)。
4. OCR 自检: 重绘后裁文字区跑 EasyOCR, 拼错换 seed 重跑(最多3次), 全失败回退确定性叠字。

用法:
  python src/text_fission_ai.py --test         # 只跑 denim_3 一个 combo 验证管线
  python src/text_fission_ai.py                # 全量 6 图
  python src/text_fission_ai.py --images denim_3 skull_5
"""
import argparse, json, time, sys, io, re, math
from pathlib import Path
import requests, numpy as np
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

RESTYLE_NEG = ("text, words, letters, typography, font, alphabet, writing, watermark, "
               "signature, logo, badge, blurry, deformed, low quality, bad anatomy, "
               "mutation, cropped, jpeg artifacts, noise")
TEXTREDRAW_NEG = ("misspelled text, gibberish letters, extra characters, wrong word, "
                  "distorted typography, double text, blurry text, low quality, smudged, "
                  "illegible")
PRECLEAN_NEG = ("text, words, letters, typography, font, alphabet, writing, watermark, "
                "signature, logo, badge")

# 每张图配置: styles=风格裂变列表; regions=文字区域(空=纯风格裂变)
# bbox 仅包文字 glyph 本身, 不要包整个横幅/背景, 避免 inpaint 走样。
CONFIG = {
    "pinterest_denim_3.jpg": {
        "styles": [
            "watercolor painting, soft pastel, paper grain texture, airy",
            "neon synthwave, glowing cyan and magenta, dark gradient, retro 80s",
            "vintage embroidered fabric patch, visible thread texture, warm muted tones",
        ],
        "regions": [{
            "bbox": (18, 42, 720, 268), "orig": "UPCY", "font": "impact",
            "fill": (245, 245, 250, 255), "stroke": (40, 55, 95, 255), "stroke_w": 8,
            "banks": ["JEANS", "PATCH", "MOTH", "INDIGO"],
        }],
    },
    "pinterest_skull_5.jpg": {
        "styles": [
            "stippled ink engraving, monochrome black and white, fine hatching",
            "vivid comic pop art, bold halftone dots, saturated colors",
            "iridescent oil paint, jewel tones, glossy reflective",
        ],
        "regions": [
            {"bbox": (175, 55, 555, 195), "orig": "TRUE", "font": "oldengl",
             "fill": (225, 30, 40, 255), "stroke": (10, 10, 15, 255), "stroke_w": 7,
             "banks": ["REAPER", "RAVEN", "BONES", "CROW"]},
            {"bbox": (125, 830, 615, 1070), "orig": "NEVER DIES", "font": "oldengl",
             "fill": (235, 235, 240, 255), "stroke": (10, 10, 15, 255), "stroke_w": 6,
             "banks": ["STILL BREATHES", "FOREVER MORE", "NEVER FADES", "ENDLESS"]},
        ],
    },
    "pinterest_eagle_2.jpg": {
        "styles": [
            "art deco, gold foil, geometric, luxurious",
            "high contrast woodcut linocut, bold black and cream",
            "cyberpunk neon, holographic, electric blue and pink",
        ],
        "regions": [{
            "bbox": (350, 710, 600, 790), "orig": "JACKE DIANNIES", "font": "impact",
            "fill": (235, 200, 120, 255), "stroke": (20, 15, 10, 255), "stroke_w": 6,
            "banks": ["RAPTOR", "EMPIRE", "CROWN", "STORM"],
        }],
    },
    "pinterest_metal_6.jpg": {
        "styles": [
            "acid green toxic biohazard palette, corroded metal",
            "blood red gore, visceral, dark crimson",
        ],
        "regions": [],
    },
    "pinterest_camo_4.jpg": {
        "styles": [
            "infrared thermal imaging palette, hot orange and blue",
            "blue digital pixel camouflage, crisp",
        ],
        "regions": [],
    },
    "pinterest_illust_1.jpg": {
        "styles": [
            "gold leaf gilded, luxurious metallic",
            "holographic iridescent, shifting rainbow sheen",
        ],
        "regions": [],
    },
}


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


def wait_outputs(pid, prefix, timeout=240):
    t0 = time.time()
    last = None
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
                        b = requests.get(url, timeout=120).content
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
def build_restyle(orig_name, style_prompt, seed, prefix):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["4"] = {"class_type": "Canny", "inputs": {"image": ["3", 0], "low_threshold": 0.10, "high_threshold": 0.22}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = (f"masterpiece, best quality, ultra detailed, {style_prompt}, "
           f"same subject and composition as the reference image, clean edges, sharp")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": RESTYLE_NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": 0.6, "start_percent": 0.0, "end_percent": 0.9, "control_net": ["5", 0]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1], "image": ["2", 0],
        "weight": 0.35, "weight_type": "style transfer", "combine_embeds": "average",
        "embeds_scaling": "V only", "start_at": 0.0, "end_at": 0.9, "noise": 0.1}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
        "seed": seed, "steps": 35, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.65, "noise_mask": None}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["16"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["15", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


def build_text_redraw(base_name, style_ref_name, mask_name, new_word, seed, prefix):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": base_name}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": style_ref_name}}
    g["4"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
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
        "pixels": ["5", 0], "vae": ["1", 2], "mask": ["4", 1], "grow_mask_by": 4}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["22", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0],
        "seed": seed, "steps": 30, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.85, "noise_mask": None}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["16"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["15", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


def build_preclean(orig_name, mask_name, seed, prefix):
    """预清理: 把原图所有文字区 inpaint 成干净背景, 避免风格裂变阶段再造字。"""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
    g["4"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text":
        "clean background, seamless texture, no text, maintain surrounding details and colors"}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": PRECLEAN_NEG}}
    g["10"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["4", 0], "vae": ["1", 2], "mask": ["3", 1], "grow_mask_by": 6}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0],
        "seed": seed, "steps": 28, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.92, "noise_mask": None}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix}}
    return g


# ---------- OCR verify ----------
_reader = None
def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True)
    return _reader


def ocr_check(orig_path, bbox, word, scale_x, scale_y):
    """裁 1mp 输出文字区做 OCR, 看是否含目标词(容错)。"""
    reader = get_reader()
    img = Image.open(orig_path).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
    w, h = img.size
    box = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
    if box[2] <= box[0] or box[3] <= box[1]:
        return False
    crop = img.crop(box)
    buf = io.BytesIO(); crop.save(buf, "PNG"); buf.seek(0)
    res = reader.readtext(buf.getvalue(), detail=0, paragraph=False)
    text = " ".join(res).upper()
    target = word.upper().replace(" ", "")
    # 容错: 目标词连续子串, 或首字母序列匹配
    norm = re.sub(r"[^A-Z]", "", text)
    if target in norm:
        return True
    # 首字母
    initials = "".join([c for c in target])
    if len(target) >= 4 and initials in norm:
        return True
    print(f"      ocr={text!r} want={word!r} -> FAIL", flush=True)
    return False


# ---------- deterministic fallback ----------
def fit_font(path, text, max_w, max_h):
    size = 12
    while True:
        f = ImageFont.truetype(path, size)
        bb = f.getbbox(text)
        if (bb[2] - bb[0]) > max_w or (bb[3] - bb[1]) > max_h or size > 600:
            break
        size += 4
    return ImageFont.truetype(path, max(12, size - 4))


def fallback_render(base_up_bytes, region, word, scale_x, scale_y, out_path):
    """AI 重绘全失败时用确定性叠字保拼写。"""
    img = Image.open(io.BytesIO(base_up_bytes)).convert("RGBA")
    x1, y1, x2, y2 = [int(v * (scale_x if i % 2 == 0 else scale_y)) for i, v in enumerate(region["bbox"])]
    rw, rh = x2 - x1, y2 - y1
    f = fit_font(FONTS[region["font"]], word, int(rw * 0.92), int(rh * 0.82))
    d = ImageDraw.Draw(img)
    bb = f.getbbox(word)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    cx = x1 + (rw - tw) // 2
    cy = y1 + (rh - th) // 2 - bb[1]
    d.text((cx, cy), word, font=f, fill=region["fill"],
           stroke_width=region["stroke_w"], stroke_fill=region["stroke"])
    img.convert("RGB").save(out_path, "PNG")
    return out_path


# ---------- pipeline ----------
def preclean_image(fname, cfg, ts, seed0):
    """对原图所有文字区做预清理, 返回清理后 1mp 图片在 ComfyUI/input 里的文件名。"""
    orig = SRC / fname
    regions = cfg.get("regions", [])
    if not regions:
        return fname
    base_img = Image.open(orig).convert("RGB")
    img_1mp, _ = scale_to_mp(base_img, 1.0)
    bboxes = [r["bbox"] for r in regions]
    mask = make_mask(img_1mp.size, bboxes, dilate=16)
    mask_name = f"ai_{ts}_{Path(fname).stem}_clean_mask.png"
    mask.save(INPUT / mask_name)
    prefix = f"ai_{ts}_{Path(fname).stem}_clean"
    g = build_preclean(fname, mask_name, seed0, prefix)
    pid = submit(g, f"ai_{ts}_{Path(fname).stem}_preclean")
    if not pid:
        print(f"  preclean submit failed for {fname}", flush=True); return fname
    out = wait_outputs(pid, prefix, timeout=240)
    if not out:
        print(f"  preclean failed for {fname}, use original", flush=True); return fname
    # 取 1mp 版本作为后续风格裂变输入
    clean_1mp = out.get("1mp") or out.get("up")
    clean_name = f"ai_{ts}_{Path(fname).stem}_clean.png"
    (INPUT / clean_name).write_bytes(clean_1mp)
    print(f"  preclean OK ({len(clean_1mp)//1024}KB)", flush=True)
    return clean_name


def run_combo(fname, cfg, ci, cleaned_name, out_dir, ts, seed0):
    orig = SRC / fname
    base_img = Image.open(orig).convert("RGB")
    style = cfg["styles"][ci % len(cfg["styles"])]
    tag = f"{Path(fname).stem}_c{ci}"
    print(f"\n[combo] {tag}  style='{style[:40]}...'", flush=True)

    INPUT.mkdir(parents=True, exist_ok=True)
    restyle_prefix = f"ai_{ts}_{tag}_restyle"
    g = build_restyle(cleaned_name, style, seed0, restyle_prefix)
    pid = submit(g, f"ai_{ts}_{tag}_a")
    if not pid:
        print("    restyle submit failed", flush=True); return None
    out = wait_outputs(pid, restyle_prefix, timeout=260)
    if not out:
        print("    restyle failed", flush=True); return None
    base_up = out.get("up") or out.get("1mp")
    base_1mp = out.get("1mp") or out.get("up")
    base_name = f"ai_{ts}_{tag}_base.png"
    (INPUT / base_name).write_bytes(base_up)
    (out_dir / f"{tag}_restyle.png").write_bytes(base_up)
    print(f"    restyle OK ({len(base_up)//1024}KB)", flush=True)

    b1 = Image.open(io.BytesIO(base_1mp))
    sx = b1.size[0] / base_img.size[0]
    sy = b1.size[1] / base_img.size[1]

    regions = cfg.get("regions", [])
    if not regions:
        final = out_dir / f"{tag}_final.png"
        final.write_bytes(base_up)
        return {"tag": tag, "final": final, "words": [], "ocr_pass": True, "style": style}

    # 逐区域文字重绘 (AI)
    cur_base_name = base_name
    final_words = []
    ocr_all_pass = True
    for ri, r in enumerate(regions):
        word = r["banks"][ci % len(r["banks"])]
        final_words.append(word)
        ref = crop_region(base_img, r["bbox"], pad=16)
        ref_name = f"ai_{ts}_{tag}_r{ri}_ref.png"
        ref.save(INPUT / ref_name)
        # 更紧的 mask, 只包文字本身
        mask = make_mask(b1.size, [r["bbox"]], dilate=12)
        mask_name = f"ai_{ts}_{tag}_r{ri}_mask.png"
        mask.save(INPUT / mask_name)
        redraw_prefix = f"ai_{ts}_{tag}_r{ri}"
        ok = False
        best_up = None
        for attempt in range(3):
            seed = seed0 + 1000 * (ri + 1) + attempt * 7
            g2 = build_text_redraw(cur_base_name, ref_name, mask_name, word, seed, redraw_prefix)
            pid2 = submit(g2, f"ai_{ts}_{tag}_b{ri}_{attempt}")
            if not pid2:
                continue
            o2 = wait_outputs(pid2, redraw_prefix, timeout=200)
            if not o2:
                continue
            up2 = o2.get("up") or o2.get("1mp")
            best_up = up2
            (INPUT / f"ai_{ts}_{tag}_r{ri}_base.png").write_bytes(up2)
            cur_base_name = f"ai_{ts}_{tag}_r{ri}_base.png"
            o1 = o2.get("1mp")
            if o1:
                tmp = out_dir / f"_tmp_{tag}_r{ri}_1mp.png"
                tmp.write_bytes(o1)
                if ocr_check(tmp, r["bbox"], word, sx, sy):
                    ok = True
                    (out_dir / f"{tag}_final.png").write_bytes(up2)
                    break
                tmp.unlink(missing_ok=True)
            else:
                (out_dir / f"{tag}_final.png").write_bytes(up2)
                ok = True
                break
        if not ok:
            print(f"    OCR 全失败, 回退确定性叠字: {word}", flush=True)
            fallback_render(best_up if best_up is not None else base_up, r, word, sx, sy,
                            out_dir / f"{tag}_final.png")
            ocr_all_pass = False
        else:
            print(f"    text '{word}' redraw OK (ocr_pass)", flush=True)

    return {"tag": tag, "final": out_dir / f"{tag}_final.png",
            "words": final_words, "ocr_pass": ocr_all_pass, "style": style}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()
    ts = int(time.time())
    out_dir = ROOT / "jobs" / f"text_fission_ai_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== text_fission_ai -> {out_dir} ===", flush=True)

    names = args.images or list(CONFIG.keys())
    if args.test:
        names = ["pinterest_denim_3.jpg"]
    seed0 = 700301
    results = []
    for fi, fname in enumerate(names):
        cfg = CONFIG[fname]
        # 先预清理(每图一次)
        cleaned_name = preclean_image(fname, cfg, ts, seed0 + fi * 1000)
        n_combos = 1 if args.test else (3 if cfg["regions"] else 2)
        for ci in range(n_combos):
            res = run_combo(fname, cfg, ci, cleaned_name, out_dir, ts, seed0 + fi * 100 + ci * 13)
            if res:
                results.append(res)

    html = ["<html><head><meta charset='utf-8'><style>",
            "body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:20px}",
            ".card{display:inline-block;margin:10px;vertical-align:top;border:1px solid #444;border-radius:8px;overflow:hidden}",
            ".card img{height:380px;display:block}",
            ".cap{padding:6px 10px;font-size:13px}",
            ".ok{color:#6f6}.bad{color:#f66}.sty{color:#9cf;font-size:11px}",
            "h2{color:#fd6}", "</style></head><body>"]
    for r in results:
        badge = "OCR✓" if r["ocr_pass"] else "OCR✗(回退)"
        cls = "ok" if r["ocr_pass"] else "bad"
        words = ",".join(r["words"]) if r["words"] else "(纯风格裂变)"
        rel = Path(r["final"]).name
        html.append(f"<div class='card'><img src='{rel}'>")
        html.append(f"<div class='cap'><span class='{cls}'>[{badge}]</span> {r['tag']}<br>")
        html.append(f"<b>{words}</b><br><span class='sty'>{r['style'][:60]}</span></div></div>")
    html.append("</body></html>")
    (out_dir / "gallery.html").write_text("\n".join(html), encoding="utf-8")
    print(f"\n=== done: {len(results)} combos -> {out_dir}/gallery.html ===", flush=True)


if __name__ == "__main__":
    main()
