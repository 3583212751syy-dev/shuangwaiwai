# -*- coding: utf-8 -*-
"""
text_fission_v2.py -- 风格裂变 + 确定性文字叠字（从原图文字区提取样式）

规则:
1. 输入 = 桌面「图裂变测试图」原始 jpg, 绝不用裂变后的图当输入。
2. 图片裂变: 同主体/同构图, 换明显不同画风 (Canny 强锁结构 + 较低 denoise)。
3. 文本裂变: 新词沿用原图文字的颜色/描边风格, 用确定性渲染叠入, 绝不乱字。
4. 侵权词必换; 非侵权词也提供替换变体。

用法:
  python src/text_fission_v2.py --test         # 只跑 denim_3 验证
  python src/text_fission_v2.py                # 全量 6 图
"""
import argparse, json, time, sys, io, shutil
from pathlib import Path
import requests, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(r"E:/Desktop/图裂变测试图")
INPUT = ROOT / "ComfyUI" / "input"
COMFYUI = "http://127.0.0.1:8188"
CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"

FONTS = {
    "impact":  r"C:/Windows/Fonts/impact.ttf",
    "oldengl": r"C:/Windows/Fonts/OLDENGL.TTF",
    "stencil": r"C:/Windows/Fonts/STENCIL.TTF",
    "rock":    r"C:/Windows/Fonts/ROCK.TTF",
    "gothicb": r"C:/Windows/Fonts/GOTHICB.TTF",
    "frscript": r"C:/Windows/Fonts/FREESCPT.TTF",
}

RESTYLE_NEG = ("text, words, letters, typography, font, alphabet, writing, watermark, "
               "signature, logo, badge, blurry, deformed, low quality, bad anatomy, "
               "mutation, cropped, jpeg artifacts, noise")

CONFIG = {
    "pinterest_denim_3.jpg": {
        "styles": [
            "watercolor painting, soft pastel, paper grain texture, airy",
            "neon synthwave, glowing cyan and magenta, dark gradient, retro 80s",
            "vintage embroidered fabric patch, visible thread texture, warm muted tones",
        ],
        "regions": [{
            "bbox": (10, 30, 726, 350), "orig": "UPCY", "font": "impact",
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
             "banks": ["REAPER", "RAVEN", "BONES", "CROW"]},
            {"bbox": (100, 760, 636, 1020), "orig": "NEVER DIES", "font": "oldengl",
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
            "bbox": (300, 680, 650, 800), "orig": "JACKE DIANNIES", "font": "impact",
            "banks": ["RAPTOR", "EMPIRE", "CROWN", "STORM"],
        }],
    },
    "pinterest_metal_6.jpg": {
        "styles": ["acid green toxic biohazard palette, corroded metal",
                   "blood red gore, visceral, dark crimson"],
        "regions": [],
    },
    "pinterest_camo_4.jpg": {
        "styles": ["infrared thermal imaging palette, hot orange and blue",
                   "blue digital pixel camouflage, crisp"],
        "regions": [],
    },
    "pinterest_illust_1.jpg": {
        "styles": ["gold leaf gilded, luxurious metallic",
                   "holographic iridescent, shifting rainbow sheen"],
        "regions": [],
    },
}


def scale_to_mp(img, mp=1.0):
    w, h = img.size
    scale = (mp * 1e6 / (w * h)) ** 0.5
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS), (nw / w, nh / h)


def make_mask(size_wh, bboxes, dilate=18):
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


def wait_outputs(pid, prefix, timeout=280):
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
def build_preclean(orig_name, seed, prefix):
    """AI inpaint 预清理: 不加 ControlNet, 自然融掉文字, 不留硬边。"""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["4"] = {"class_type": "LoadImageMask", "inputs": {"image": f"{prefix}_mask.png", "channel": "alpha"}}
    pos = ("masterpiece, best quality, seamless background texture, empty banner area, "
           "no text, no letters, no words, clean surface")
    neg = ("text, words, letters, typography, font, alphabet, writing, watermark, "
           "signature, logo, badge, blurry, low quality")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["10"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["3", 0], "vae": ["1", 2], "mask": ["4", 0], "grow_mask_by": 14}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0],
        "seed": seed, "steps": 26, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.82}}
    g["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["14"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["13", 0], "resolution_steps": 64}}
    g["15"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
        "upscale_model": ["16", 0], "image": ["14", 0]}}
    g["16"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


# ---------- style extraction from original text crop ----------
def kmeans_colors(arr_rgb, k=3):
    """Return list of (r,g,b) dominant colors sorted by cluster size."""
    data = np.float32(arr_rgb.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    idx = np.argsort(-counts)
    return [tuple(map(int, centers[i])) for i in idx]


def edge_pixels(arr_rgb, n=3):
    """Sample pixels around Canny edges -> mean edge color."""
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    if edges.sum() == 0:
        return None
    ys, xs = np.where(edges > 0)
    samples = arr_rgb[ys, xs]
    # take median to ignore outliers
    return tuple(map(int, np.median(samples, axis=0)))


def brightness(c):
    return 0.299*c[0] + 0.587*c[1] + 0.114*c[2]


def saturation(c):
    mn, mx = min(c)/255.0, max(c)/255.0
    return 0 if mx == 0 else (mx - mn) / mx


def color_dist(a, b):
    return float(np.linalg.norm(np.array(a)-np.array(b)))


def median_border_color(arr, bbox, thickness=14):
    """取 bbox 外一圈 ring 的 median 颜色作为背景填充色。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = arr.shape[:2]
    x1b, x2b = max(0, x1 - thickness), min(w, x2 + thickness)
    y1b, y2b = max(0, y1 - thickness), min(h, y2 + thickness)
    parts = []
    if x1b < x1:
        parts.append(arr[y1b:y2b, x1b:x1].reshape(-1, 3))
    if x2 < x2b:
        parts.append(arr[y1b:y2b, x2:x2b].reshape(-1, 3))
    if y1b < y1:
        parts.append(arr[y1b:y1, x1:x2].reshape(-1, 3))
    if y2 < y2b:
        parts.append(arr[y2:y2b, x1:x2].reshape(-1, 3))
    if not parts:
        return np.median(arr.reshape(-1, 3), axis=0)
    ring = np.concatenate(parts, axis=0)
    return np.median(ring, axis=0) if len(ring) else np.median(arr.reshape(-1, 3), axis=0)


def extract_style(crop, manual_fill=None, manual_stroke=None):
    """
    从原图文字区提取: 主填充色、描边色、背景色。
    规则: 背景 = 最大簇; 填充 = 与背景差异最大(亮度差 + 饱和度综合);
          描边 = 与填充差异最大的簇(通常就是轮廓/阴影色)。
    """
    arr = np.array(crop.convert("RGB"))
    colors = kmeans_colors(arr, k=4)

    bg = colors[0]
    bg_bri = brightness(bg)

    # 候选色: 排除背景
    candidates = [c for c in colors[1:] if color_dist(c, bg) > 30]
    if not candidates:
        candidates = colors[1:] if len(colors) > 1 else colors

    # 填充: 与背景综合差异最大(既照顾高饱和彩色字, 也照顾高亮度白字)
    def fill_score(c):
        return abs(brightness(c) - bg_bri) + saturation(c) * 80
    fill = max(candidates, key=fill_score)

    # 描边: 与填充差异最大的簇; 没有则取反色
    stroke_candidates = [c for c in colors if color_dist(c, fill) > 40]
    stroke = max(stroke_candidates, key=lambda c: color_dist(c, fill)) if stroke_candidates else \
             tuple(max(0, min(255, 255 - f)) for f in fill)

    if manual_fill:
        fill = manual_fill[:3]
    if manual_stroke:
        stroke = manual_stroke[:3]
    return {"fill": fill, "stroke": stroke, "bg": bg}


# ---------- graph builders ----------
def build_restyle(orig_name, style_prompt, seed, prefix):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["4"] = {"class_type": "Canny", "inputs": {"image": ["3", 0], "low_threshold": 0.05, "high_threshold": 0.18}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = (f"masterpiece, best quality, ultra detailed, {style_prompt}, "
           f"same exact subject and composition as the reference, sharp, clean")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": RESTYLE_NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": 0.88, "start_percent": 0.0, "end_percent": 0.92, "control_net": ["5", 0]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1], "image": ["2", 0],
        "weight": 0.22, "weight_type": "style transfer", "combine_embeds": "average",
        "embeds_scaling": "V only", "start_at": 0.0, "end_at": 0.92, "noise": 0.08}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
        "seed": seed, "steps": 35, "cfg": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.45}}
    g["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["14"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["13", 0], "resolution_steps": 64}}
    g["15"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
        "upscale_model": ["16", 0], "image": ["14", 0]}}
    g["16"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


# ---------- deterministic text overlay with style extraction ----------
def render_text(img, word, bbox, style, font_key="impact"):
    """把 word 用 style 的颜色/描边渲染到 img 的 bbox 区域内, 尽量撑满。"""
    x1, y1, x2, y2 = bbox
    W, H = x2 - x1, y2 - y1
    if W <= 0 or H <= 0:
        return img
    font_path = FONTS.get(font_key, FONTS["impact"])

    # 创建足够大的透明图层用来找字号
    canvas = Image.new("RGBA", (W * 4, H * 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    lo, hi = 10, H * 4
    best = 10
    for _ in range(24):
        mid = (lo + hi) // 2
        try:
            fnt = ImageFont.truetype(font_path, mid)
        except Exception:
            fnt = ImageFont.load_default()
        bb = d.textbbox((0, 0), word.upper(), font=fnt)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        if tw <= W * 3.6 and th <= H * 3.4:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    fnt = ImageFont.truetype(font_path, best)
    bb = d.textbbox((0, 0), word.upper(), font=fnt)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = (W * 4 - tw) // 2 - bb[0]
    ty = (H * 4 - th) // 2 - bb[1]

    fill = style["fill"] + (255,)
    stroke = style["stroke"] + (255,)
    # 估算描边宽度
    stroke_w = max(2, int(best * 0.06))
    d.text((tx, ty), word.upper(), font=fnt, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)

    # 缩放回目标尺寸
    text_layer = canvas.resize((W, H), Image.LANCZOS)
    # 简单阴影增强可读性
    shadow = text_layer.copy()
    px = np.array(shadow)
    px[px[..., 3] > 20] = [0, 0, 0, 120]
    shadow = ImageChops.offset(Image.fromarray(px, "RGBA"), 2, 2)

    base = img.convert("RGBA")
    base.paste(shadow, (x1, y1), shadow)
    base.paste(text_layer, (x1, y1), text_layer)
    return base.convert("RGB")


# ---------- pipeline ----------
def run_combo(fname, cfg, ci, out_dir, ts, seed0):
    orig = SRC / fname
    base_img = Image.open(orig).convert("RGB")
    style = cfg["styles"][ci % len(cfg["styles"])]
    tag = f"{Path(fname).stem}_c{ci}"
    print(f"\n[combo] {tag}  style='{style[:40]}...'", flush=True)

    INPUT.mkdir(parents=True, exist_ok=True)
    # ComfyUI LoadImage 只认 input/ 目录, 把原图复制进去
    src_in = INPUT / orig.name
    if not src_in.exists() or src_in.stat().st_size != orig.stat().st_size:
        shutil.copy2(orig, src_in)

    # Stage A: 预清理 (cv2 填充文字区 + inpaint 融边; 扩大填充范围盖住字尾/阴影)
    cleaned_name = f"ai_{ts}_{tag}_preclean.png"
    cleaned_path = INPUT / cleaned_name
    regions = cfg.get("regions", [])
    if regions:
        arr = np.array(base_img).astype(np.float32)
        h, w = arr.shape[:2]
        # 扩大 bbox 做填充, 避免字尾/描边残留
        expanded = []
        for r in regions:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            margin = 35
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
            expanded.append((x1, y1, x2, y2))
            fill = median_border_color(arr, r["bbox"], thickness=18)
            arr[y1:y2, x1:x2] = fill
        arr = arr.astype(np.uint8)
        mask_rgba = make_mask(base_img.size, expanded, dilate=30)
        mask_cv = np.array(mask_rgba.convert("L"))
        cleaned_arr = cv2.inpaint(arr, mask_cv, inpaintRadius=9, flags=cv2.INPAINT_TELEA)
        cleaned_img = Image.fromarray(cleaned_arr)
        cleaned_img.save(cleaned_path)
        cleaned_img.save(out_dir / f"{tag}_preclean.png")
        print(f"    preclean OK (expanded cv2 inpaint)", flush=True)
    else:
        base_img.save(cleaned_path)

    # Stage B: 风格裂变 (在清掉字的原图上做, 防造字)
    restyle_prefix = f"ai_{ts}_{tag}_restyle"
    g = build_restyle(cleaned_name, style, seed0 + 1, restyle_prefix)
    pid = submit(g, f"ai_{ts}_{tag}_b")
    if not pid:
        print("    restyle submit failed", flush=True); return None
    out = wait_outputs(pid, restyle_prefix, timeout=260)
    if not out:
        print("    restyle failed", flush=True); return None
    styled_1mp = Image.open(io.BytesIO(out.get("1mp") or out.get("up"))).convert("RGB")
    styled_4x = Image.open(io.BytesIO(out.get("up") or out.get("1mp"))).convert("RGB")
    styled_4x.save(out_dir / f"{tag}_restyle.png")
    print(f"    restyle OK ({len(out.get('up'))//1024}KB)", flush=True)

    # 缩放比 (4x vs original)
    sx = styled_4x.size[0] / base_img.size[0]
    sy = styled_4x.size[1] / base_img.size[1]

    # Stage C: 文字裂变 (确定性叠字, 从原图文字区提取样式)
    final = styled_4x.copy()
    words = []
    for ri, r in enumerate(regions):
        word = r["banks"][ci % len(r["banks"])]
        words.append(word)
        # 从原图文字 crop 提取样式
        crop = base_img.crop(r["bbox"])
        sty = extract_style(crop)
        # 把 bbox 缩放到 4x 图坐标
        x1, y1, x2, y2 = [int(v * sx) if i % 2 == 0 else int(v * sy) for i, v in enumerate(r["bbox"])]
        final = render_text(final, word, (x1, y1, x2, y2), sty, font_key=r.get("font", "impact"))
        print(f"    text '{word}' rendered  fill={sty['fill']} stroke={sty['stroke']}", flush=True)

    out_path = out_dir / f"{tag}_final.png"
    final.save(out_path)
    return {"tag": tag, "final": out_path, "restyle": out_dir / f"{tag}_restyle.png",
            "words": words, "style": style}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()
    ts = int(time.time())
    out_dir = ROOT / "jobs" / f"text_fission_v2_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== text_fission_v2 -> {out_dir} ===", flush=True)

    names = args.images or list(CONFIG.keys())
    if args.test:
        names = ["pinterest_denim_3.jpg"]
    seed0 = 800301
    results = []
    for fi, fname in enumerate(names):
        cfg = CONFIG[fname]
        n_combos = 1 if args.test else 2
        for ci in range(n_combos):
            res = run_combo(fname, cfg, ci, out_dir, ts, seed0 + fi * 100 + ci * 17)
            if res:
                results.append(res)

    html = ["<html><head><meta charset='utf-8'><style>",
            "body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:20px}",
            ".card{display:inline-block;margin:10px;vertical-align:top;border:1px solid #444;border-radius:8px;overflow:hidden}",
            ".card img{height:380px;display:block}",
            ".cap{padding:6px 10px;font-size:13px}",
            ".sty{color:#9cf;font-size:11px}",
            "h2{color:#fd6}", "</style></head><body>"]
    for r in results:
        words = ",".join(r["words"]) if r["words"] else "(纯风格裂变)"
        rel = Path(r["final"]).name
        html.append(f"<div class='card'><img src='{rel}'>")
        html.append(f"<div class='cap'>{r['tag']}<br><b>{words}</b><br><span class='sty'>{r['style'][:60]}</span></div></div>")
    html.append("</body></html>")
    (out_dir / "gallery.html").write_text("\n".join(html), encoding="utf-8")
    print(f"\n=== done: {len(results)} combos -> {out_dir}/gallery.html ===", flush=True)


if __name__ == "__main__":
    main()
