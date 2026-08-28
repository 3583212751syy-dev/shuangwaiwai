# -*- coding: utf-8 -*-
"""
text_fission_v6.py -- 图裂变正解:
  Stage A 图裂变: 复用 v115-v118 正解参数 (denoise 0.72 + IPAdapter 0.45 + Canny 0.55 + 内部细节重绘 prompt)
  Stage B 改字: SDXL inpaint 自然抹旧字 -> PIL 按原图同字体/同颜色/同位置/同尺寸渲新词
  每张出图后自检: Read 产物确认不是糊的才交付

输入: 桌面「图裂变测试图」原始 jpg.
"""
import argparse, json, time, io, shutil, os
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

# Windows 系统字体, 按原图风格近似
FONTS = {
    "impact":   r"C:/Windows/Fonts/impact.ttf",
    "oldengl":  r"C:/Windows/Fonts/OLDENGL.TTF",
    "stencil":  r"C:/Windows/Fonts/STENCIL.TTF",
    "rock":     r"C:/Windows/Fonts/ROCK.TTF",
    "gothicb":  r"C:/Windows/Fonts/GOTHICB.TTF",
    "frscript": r"C:/Windows/Fonts/FREESCPT.TTF",
    "arialb":   r"C:/Windows/Fonts/ARIALBD.TTF",
}

# style prompt 改成"内部细节微裂变"(不再"换画风"), 保留原图色系和主体身份
CONFIG = {
    "pinterest_denim_3.jpg": {
        "styles": [
            # denim 1: 保留牛仔色, 蝴蝶刺绣细节重绘
            "same denim patch, butterfly embroidery pattern redesigned with new stitching detail, preserve original denim color and layout, sharp embroidered thread texture",
            # denim 2: 同样牛仔色, 不同刺绣细节
            "same denim fabric with butterfly motif, different embroidery thread pattern variation, preserve cool blue denim palette, new ornamental stitching",
        ],
        "regions": [{
            "bbox": (10, 30, 726, 350), "orig": "UPCY",
            "font_key": "impact",  # bold condensed sans, 与原图 UPCY 接近
            "banks": ["JEANS", "PATCH", "MOTH", "INDIGO"],
        }],
    },
    "pinterest_skull_5.jpg": {
        "styles": [
            "same skull illustration, same composition, internal line work and shading details redesigned variation, preserve dark gothic palette and layout",
            "same skull artwork, same layout, different ornamental shading pattern variation, preserve monochrome aesthetic",
        ],
        "regions": [
            {"bbox": (175, 55, 555, 240), "orig": "TRUE",
             "font_key": "oldengl", "banks": ["REAPER", "RAVEN", "BONES", "CROW"]},
            {"bbox": (100, 750, 636, 940), "orig": "NEVER",
             "font_key": "oldengl", "banks": ["STILL", "FOREVER", "ALWAYS", "ENDLESS"]},
            {"bbox": (140, 940, 596, 1220), "orig": "DIES",
             "font_key": "oldengl", "banks": ["BREATHES", "RISES", "LIVES", "ENDURES"]},
        ],
    },
    "pinterest_eagle_2.jpg": {
        "styles": [
            "same eagle illustration, same composition, internal feather and detail pattern variation, preserve original color palette and layout",
            "same eagle artwork, same layout, different feather texture and shading variation, preserve overall identity",
        ],
        "regions": [{
            "bbox": (300, 680, 650, 800), "orig": "JACKE DIANNIES",
            "font_key": "impact", "banks": ["RAPTOR", "EMPIRE", "CROWN", "STORM"],
        }],
    },
    "pinterest_metal_6.jpg": {
        "styles": [
            "same metallic texture surface, same composition, internal pattern and corrosion detail variation, preserve original palette",
            "same metal artwork, same layout, different rust and texture detail variation",
        ],
        "regions": [],
    },
    "pinterest_camo_4.jpg": {
        "styles": [
            "same camouflage pattern, same composition, internal pattern color and shape variation, preserve overall identity",
            "same camo artwork, same layout, different camouflage pattern variation",
        ],
        "regions": [],
    },
    "pinterest_illust_1.jpg": {
        "styles": [
            "same illustration, same composition, internal detail and color variation, preserve original art style",
            "same artwork, same layout, different ornamental detail variation",
        ],
        "regions": [],
    },
}

RESTYLE_NEG = ("text, words, letters, typography, font, alphabet, writing, watermark, "
               "signature, logo, badge, blurry, deformed, low quality, bad anatomy, "
               "mutation, cropped, jpeg artifacts, noise")


# ---------- helpers ----------
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


def wait_outputs(pid, prefix, timeout=320):
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
        for msg in st.get("messages", []):
            if isinstance(msg, (list, tuple)) and msg and msg[0] == "execution_error":
                print(f"    exec error: {str(msg[1])[:500]}", flush=True); return None
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


def kmeans_colors(arr_rgb, k=4):
    data = np.float32(arr_rgb.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    idx = np.argsort(-counts)
    return [tuple(map(int, centers[i])) for i in idx], counts[idx]


def color_dist(a, b): return float(np.linalg.norm(np.array(a) - np.array(b)))


def extract_text_style(orig_img, bbox):
    """从原图文字区中心提取 fill 颜色: 取与背景对比度最大的色簇, 而不是用整 bbox(可能含大量背景)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = y2-y1, x2-x1
    # 取中心 60% 区域, 减少背景干扰
    cx1, cy1 = int(x1 + W*0.2), int(y1 + H*0.2)
    cx2, cy2 = int(x2 - W*0.2), int(y2 - H*0.2)
    crop = np.array(orig_img.crop((cx1, cy1, cx2, cy2)).convert("RGB"))
    colors, counts = kmeans_colors(crop, k=4)
    bg = colors[0]  # 最大簇当背景
    # 候选: 与 bg 距离足够大
    cand = [c for c, n in zip(colors[1:], counts[1:]) if color_dist(c, bg) > 30]
    if not cand:
        cand = colors[1:]
    # fill = 与 bg 距离最大(对比度最强, 最可能是文字)
    fill = max(cand, key=lambda c: color_dist(c, bg))
    return fill, bg


def make_text_mask(size_wh, bboxes, dilate=20):
    w, h = size_wh
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        x1, y1, x2, y2 = [max(0, int(v)) for v in (x1, y1, x2, y2)]
        mask[y1:y2, x1:x2] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate*2+1, dilate*2+1))
    mask = cv2.dilate(mask, k, iterations=1)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask
    return Image.fromarray(rgba, "RGBA")


def render_text(img, word, bbox, font_key, fill):
    """按 bbox 同位置同尺寸用 PIL 渲新词, 加描边阴影."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    W, H = x2 - x1, y2 - y1
    if W <= 0 or H <= 0: return img
    font_path = FONTS.get(font_key, FONTS["impact"])
    base = img.convert("RGBA")

    canvas = Image.new("RGBA", (W*4, H*4), (0,0,0,0))
    d = ImageDraw.Draw(canvas)
    lo, hi, best = 10, H*4, 10
    word_up = word.upper()
    for _ in range(24):
        mid = (lo + hi) // 2
        try:
            fnt = ImageFont.truetype(font_path, mid)
        except Exception:
            fnt = ImageFont.load_default()
        bb = d.textbbox((0,0), word_up, font=fnt)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        if tw <= W*3.7 and th <= H*3.5:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    fnt = ImageFont.truetype(font_path, best)
    bb = d.textbbox((0,0), word_up, font=fnt)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    tx = (W*4 - tw)//2 - bb[0]
    ty = (H*4 - th)//2 - bb[1]
    fill_a = fill + (255,)
    stroke = (0, 0, 0, 255)
    sw = max(3, int(best*0.08))
    d.text((tx, ty), word_up, font=fnt, fill=fill_a, stroke_width=sw, stroke_fill=stroke)
    text_layer = canvas.resize((W, H), Image.LANCZOS)

    # 阴影
    sh = text_layer.copy()
    arr = np.array(sh)
    arr[arr[..., 3] > 20] = [0, 0, 0, 140]
    sh = Image.fromarray(arr, "RGBA")
    base.paste(sh, (x1, y1), sh)
    base.paste(text_layer, (x1, y1), text_layer)
    return base.convert("RGB")


# ---------- graph builders ----------
def build_restyle(orig_name, style_prompt, seed, prefix):
    """图裂变 (v115-v118 正解): Canny 0.55 锁主体 + IPAdapter 0.45 锁风格 + denoise 0.72 细节重绘."""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["4"] = {"class_type": "Canny", "inputs": {"image": ["3", 0], "low_threshold": 0.05, "high_threshold": 0.18}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = (f"masterpiece, best quality, ultra detailed, {style_prompt}, "
           f"redesigned internal details, same composition, same color palette, sharp, clean")
    neg = RESTYLE_NEG
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": 0.55, "start_percent": 0.0, "end_percent": 0.9, "control_net": ["5", 0]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1], "image": ["2", 0],
        "weight": 0.45, "weight_type": "style transfer", "combine_embeds": "average",
        "embeds_scaling": "V only", "start_at": 0.0, "end_at": 0.9, "noise": 0.08}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
        "seed": seed, "steps": 35, "cfg": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.72}}
    g["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["14"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["13", 0], "resolution_steps": 64}}
    g["16"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["15"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
        "upscale_model": ["16", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


def build_inpaint(img_name, mask_name, seed, prefix):
    """SDXL inpaint 自然抹掉旧字, 不留色块."""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
    g["3"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "alpha"}}
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 0], "grow_mask_by": 6}}
    pos = "seamless background texture, empty clean area, no text, no letters, no words"
    neg = "text, words, letters, typography, watermark, signature, logo, blurry, low quality"
    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["4", 0],
        "seed": seed, "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.85}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return g


def build_upscale(img_name, prefix):
    g = {}
    g["U"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["I"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
    g["UP"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["U", 0], "image": ["I", 0]}}
    g["S"] = {"class_type": "SaveImage", "inputs": {"images": ["UP", 0], "filename_prefix": prefix}}
    return g


def self_check_image(path, min_size_kb=50):
    """自检: 产物文件存在且不过小(避免空图/单色块); 返回 (ok, reason)."""
    if not Path(path).exists():
        return False, "missing file"
    size_kb = Path(path).stat().st_size // 1024
    if size_kb < min_size_kb:
        return False, f"file too small ({size_kb}KB)"
    img = np.array(Image.open(path).convert("RGB"))
    # 检查是否为近单色图(色块糊)
    flat = img.reshape(-1, 3)
    sample = flat[::500]
    if len(sample) < 10:
        return False, "image too small to check"
    std = np.std(sample, axis=0).mean()
    if std < 5:
        return False, f"image appears nearly monochrome (std={std:.1f})"
    return True, f"OK ({size_kb}KB, std={std:.1f})"


# ---------- pipeline ----------
def run_combo(fname, cfg, ci, out_dir, ts, seed0):
    orig = SRC / fname
    base_img = Image.open(orig).convert("RGB")
    style = cfg["styles"][ci % len(cfg["styles"])]
    tag = f"{Path(fname).stem}_c{ci}"
    print(f"\n[combo] {tag}  style='{style[:50]}...'", flush=True)

    INPUT.mkdir(parents=True, exist_ok=True)
    src_in = INPUT / orig.name
    if not src_in.exists() or src_in.stat().st_size != orig.stat().st_size:
        shutil.copy2(orig, src_in)

    regions = cfg.get("regions", [])

    # Stage A: 图裂变 (v115-v118 正解参数)
    fission_prefix = f"ai_{ts}_{tag}_fission"
    g = build_restyle(orig.name, style, seed0 + 1, fission_prefix)
    pid = submit(g, f"ai_{ts}_{tag}_f")
    if not pid: print("    fission submit failed", flush=True); return None
    out = wait_outputs(pid, fission_prefix, timeout=300)
    if not out: print("    fission failed", flush=True); return None
    styled_1mp = Image.open(io.BytesIO(out["1mp"])).convert("RGB")
    print(f"    fission OK ({len(out['1mp'])//1024}KB 1mp)", flush=True)

    # 自检 fission 产物
    fission_1mp_path = INPUT / f"ai_{ts}_{tag}_fission_1mp.png"  # may not exist; check via download
    # 直接对内存中的图自检
    arr = np.array(styled_1mp)
    std = np.std(arr.reshape(-1, 3)[::500], axis=0).mean()
    size_kb = len(out["1mp"]) // 1024
    if std < 8 or size_kb < 50:
        print(f"    [SELF-CHECK FAIL] fission 1mp 异常 (std={std:.1f}, {size_kb}KB), 重试...", flush=True)
        return None  # 简单起见直接丢弃; 更复杂可换 seed 重试

    sx, sy = styled_1mp.size[0] / base_img.size[0], styled_1mp.size[1] / base_img.size[1]

    words = []
    if regions:
        cur_img = styled_1mp
        cur_name = f"{tag}_stage_a.png"
        cur_img.save(INPUT / cur_name)
        for ri, r in enumerate(regions):
            word = r["banks"][ci % len(r["banks"])]
            words.append(word)
            bx1, by1 = int(r["bbox"][0] * sx), int(r["bbox"][1] * sy)
            bx2, by2 = int(r["bbox"][2] * sx), int(r["bbox"][3] * sy)
            fill, _bg = extract_text_style(base_img, r["bbox"])
            print(f"    region {ri} '{word}' fill={fill}", flush=True)

            # 1) inpaint 抹旧字
            mask = make_text_mask(cur_img.size, [(bx1, by1, bx2, by2)], dilate=20)
            mask_name = f"{tag}_r{ri}_mask.png"
            mask.save(INPUT / mask_name)
            inpaint_prefix = f"ai_{ts}_{tag}_r{ri}_inpaint"
            g2 = build_inpaint(cur_name, mask_name, seed0 + 7 + ri, inpaint_prefix)
            pid2 = submit(g2, f"ai_{ts}_{tag}_i{ri}")
            if not pid2: print("    inpaint submit failed", flush=True); return None
            out2 = wait_outputs(pid2, inpaint_prefix, timeout=300)
            if not out2: print("    inpaint failed", flush=True); return None
            cur_img = Image.open(io.BytesIO(out2["up"] or out2["1mp"])).convert("RGB")
            cur_name = f"{tag}_r{ri}_inpainted.png"
            cur_img.save(INPUT / cur_name)

            # 2) PIL 按原图字体/颜色/位置渲新词
            cur_img = render_text(cur_img, word, (bx1, by1, bx2, by2), r.get("font_key","impact"), fill)
            cur_name = f"{tag}_r{ri}_texted.png"
            cur_img.save(INPUT / cur_name)
        edited = cur_img
        up_in = cur_name
    else:
        edited = styled_1mp
        up_in = f"{tag}_stage_a.png"

    # Stage C: 4x 放大终稿
    up_prefix = f"ai_{ts}_{tag}_up"
    g3 = build_upscale(up_in, up_prefix)
    pid3 = submit(g3, f"ai_{ts}_{tag}_u")
    if not pid3: print("    upscale submit failed", flush=True); return None
    out3 = wait_outputs(pid3, up_prefix, timeout=300)
    if not out3: print("    upscale failed", flush=True); return None
    final = Image.open(io.BytesIO(out3["up"] or out3["1mp"])).convert("RGB")
    final_path = out_dir / f"{tag}_final.png"
    final.save(final_path)

    # 终稿自检
    ok, reason = self_check_image(final_path)
    print(f"    [SELF-CHECK] final: {reason}", flush=True)
    if not ok:
        print(f"    终稿异常, 跳过 (请人工检查)", flush=True)
        return None

    # 保存中间产物 + 拼图对照
    styled_1mp.save(out_dir / f"{tag}_1mp.png")
    edited.save(out_dir / f"{tag}_edited.png")
    # 拼图对照 (原图 | 终稿)
    H = 900
    o = base_img.resize((int(base_img.width * H / base_img.height), H), Image.LANCZOS)
    f2 = final.resize((int(final.width * H / final.height), H), Image.LANCZOS)
    gap = 20
    canvas = Image.new("RGB", (o.width + gap + f2.width, H + 40), (25, 25, 25))
    canvas.paste(o, (0, 20)); canvas.paste(f2, (o.width + gap, 20))
    cd = ImageDraw.Draw(canvas)
    cd.text((10, 5), "ORIGINAL", fill=(255, 200, 100))
    cd.text((o.width + gap + 10, 5), "FISSIONED", fill=(255, 200, 100))
    cd.line([(o.width + gap // 2, 0), (o.width + gap // 2, H + 40)], fill=(255, 255, 255), width=2)
    cmp_path = out_dir / f"{tag}_compare.png"
    canvas.save(cmp_path)
    print(f"    final saved -> {final_path}", flush=True)
    print(f"    compare   -> {cmp_path}", flush=True)
    return {"tag": tag, "final": final_path, "compare": cmp_path,
            "words": words, "style": style}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()
    ts = int(time.time())
    out_dir = ROOT / "jobs" / f"text_fission_v6_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== text_fission_v6 -> {out_dir} ===", flush=True)

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

    # Gallery 展示拼图对照
    html = ["<html><head><meta charset='utf-8'><style>",
            "body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:20px}",
            ".card{display:inline-block;margin:10px;vertical-align:top;border:1px solid #444;border-radius:8px;overflow:hidden;max-width:100%}",
            ".card img{height:460px;display:block}",
            ".cap{padding:6px 10px;font-size:13px}",
            ".sty{color:#9cf;font-size:11px}",
            "h2{color:#fd6}", "</style></head><body><h1>v6 图裂变 (内部细节重绘 + inpaint 改字 + 拼图对照)</h1>"]
    for r in results:
        words = ",".join(r["words"]) if r["words"] else "(纯图裂变)"
        cmp_rel = Path(r["compare"]).name
        html.append(f"<div class='card'><img src='{cmp_rel}'>")
        html.append(f"<div class='cap'>{r['tag']}<br><b>{words}</b><br><span class='sty'>{r['style'][:60]}</span></div></div>")
    html.append("</body></html>")
    (out_dir / "gallery.html").write_text("\n".join(html), encoding="utf-8")
    print(f"\n=== done: {len(results)} combos -> {out_dir}/gallery.html ===", flush=True)


if __name__ == "__main__":
    main()