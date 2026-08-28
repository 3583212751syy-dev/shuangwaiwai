# -*- coding: utf-8 -*-
"""
text_fission_v4.py -- 真正的图裂变:
  1. 图: img2img 高 denoise, 让构图/背景/元素明显变化(不是微调画风)
  2. 文本: inpaint 自然抹掉旧字 -> PIL 按原图文字风格(同字体/同颜色/同位置/同尺寸)渲新词

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
    "impact":  r"C:/Windows/Fonts/impact.ttf",
    "oldengl": r"C:/Windows/Fonts/OLDENGL.TTF",
    "stencil": r"C:/Windows/Fonts/STENCIL.TTF",
    "rock":    r"C:/Windows/Fonts/ROCK.TTF",
    "gothicb": r"C:/Windows/Fonts/GOTHICB.TTF",
    "frscript":r"C:/Windows/Fonts/FREESCPT.TTF",
    "arialb":   r"C:/Windows/Fonts/ARIALBD.TTF",
}

CONFIG = {
    "pinterest_denim_3.jpg": {
        "styles": [
            "vintage embroidered fabric patch, visible thread texture, warm muted tones",
            "neon synthwave glowing cyan magenta dark gradient retro 80s",
            "grungy streetwear worn denim patch faded paint splattered",
        ],
        "regions": [{
            "bbox": (10, 30, 726, 350), "orig": "UPCY",
            "font_key": "impact",  # bold condensed sans, 与原图 UPCY 接近
            "banks": ["JEANS", "PATCH", "MOTH", "INDIGO"],
        }],
    },
    "pinterest_skull_5.jpg": {
        "styles": [
            "stippled ink engraving monochrome black and white fine hatching",
            "vivid comic pop art bold halftone dots saturated colors",
            "dark gothic oil painting jewel tones dramatic chiaroscuro",
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
            "art deco gold foil geometric luxurious",
            "high contrast woodcut linocut bold black and cream",
            "cyberpunk neon holographic electric blue pink",
        ],
        "regions": [{
            "bbox": (300, 680, 650, 800), "orig": "JACKE DIANNIES",
            "font_key": "impact", "banks": ["RAPTOR", "EMPIRE", "CROWN", "STORM"],
        }],
    },
    "pinterest_metal_6.jpg": {
        "styles": [
            "acid green toxic biohazard palette corroded metal",
            "blood red gore visceral dark crimson",
            "oil slick rainbow iridescent patina",
        ],
        "regions": [],
    },
    "pinterest_camo_4.jpg": {
        "styles": [
            "infrared thermal imaging palette hot orange blue",
            "blue digital pixel camouflage crisp",
            "dazzle camouflage high contrast geometric ship paint",
        ],
        "regions": [],
    },
    "pinterest_illust_1.jpg": {
        "styles": [
            "gold leaf gilded luxurious metallic",
            "holographic iridescent shifting rainbow sheen",
            "matisse cutout bold flat colors paper texture",
        ],
        "regions": [],
    },
}


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


# ---------- 风格裂变: 高 denoise, 明显变化 ----------
def build_fission(orig_name, style_prompt, seed, prefix):
    """真正的图裂变: denoise 0.8 让构图/元素/背景明显变; Canny 0.5 保住主体大致轮廓."""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["4"] = {"class_type": "Canny", "inputs": {"image": ["3", 0], "low_threshold": 0.05, "high_threshold": 0.18}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = (f"masterpiece, best quality, ultra detailed, {style_prompt}, "
           f"same main subject, varied composition and background, sharp, clean")
    neg = ("text, words, letters, typography, font, alphabet, writing, watermark, "
           "signature, logo, badge, blurry, deformed, low quality, bad anatomy, "
           "mutation, cropped, jpeg artifacts, noise")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": 0.5, "start_percent": 0.0, "end_percent": 0.85, "control_net": ["5", 0]}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
        "seed": seed, "steps": 30, "cfg": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.8}}  # ← 关键: 0.8, 让图像明显裂变
    g["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["14"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["13", 0], "resolution_steps": 64}}
    g["16"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["15"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
        "upscale_model": ["16", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


# ---------- inpaint: 自然抹掉旧字(不用色块) ----------
def build_inpaint(img_name, mask_name, seed, prefix, w, h):
    """用 SDXL inpaint 在 mask 区填充周围纹理, 自然抹掉旧字, 不留色块."""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
    g["3"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "alpha"}}
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 0], "grow_mask_by": 8}}
    pos = "seamless background texture, empty clean area, no text, no letters, no words"
    neg = "text, words, letters, typography, watermark, signature, logo, blurry, low quality"
    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["4", 0],
        "seed": seed, "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.85}}  # 高 denoise 彻底抹旧字
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return g


# ---------- PIL 按原图风格渲新字 ----------
def kmeans_colors(arr_rgb, k=3):
    data = np.float32(arr_rgb.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    idx = np.argsort(-counts)
    return [tuple(map(int, centers[i])) for i in idx]


def brightness(c): return 0.299*c[0] + 0.587*c[1] + 0.114*c[2]
def saturation(c):
    mn, mx = min(c)/255.0, max(c)/255.0
    return 0 if mx == 0 else (mx - mn) / mx
def color_dist(a, b): return float(np.linalg.norm(np.array(a) - np.array(b)))


def extract_text_style(orig_img, bbox):
    """从原图文字区提取: 主填充色 + 背景色(用于和填充色区分)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = np.array(orig_img.crop((x1, y1, x2, y2)).convert("RGB"))
    colors = kmeans_colors(crop, k=3)
    bg = colors[0]  # 最大簇当背景
    cand = [c for c in colors[1:] if color_dist(c, bg) > 30]
    if not cand: cand = colors[1:]
    fill = max(cand, key=lambda c: abs(brightness(c) - brightness(bg)) + saturation(c) * 80)
    return fill, bg


def make_text_mask(size_wh, bboxes, dilate=18):
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
    """按 bbox 同位置同尺寸用 PIL 渲新词, 加描边/阴影增加可读性."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    W, H = x2 - x1, y2 - y1
    if W <= 0 or H <= 0: return img
    font_path = FONTS.get(font_key, FONTS["impact"])
    base = img.convert("RGBA")

    # 4x 画布找最大字号
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

    # Stage A: 真正的图裂变 (高 denoise)
    fission_prefix = f"ai_{ts}_{tag}_fission"
    g = build_fission(orig.name, style, seed0 + 1, fission_prefix)
    pid = submit(g, f"ai_{ts}_{tag}_f")
    if not pid: print("    fission submit failed", flush=True); return None
    out = wait_outputs(pid, fission_prefix, timeout=300)
    if not out: print("    fission failed", flush=True); return None
    styled_1mp = Image.open(io.BytesIO(out["1mp"])).convert("RGB")
    print(f"    fission OK ({len(out['1mp'])//1024}KB 1mp)", flush=True)

    # 缩放比 (1mp vs original)
    s1 = styled_1mp.size[0] / base_img.size[0]
    sx, sy = styled_1mp.size[0] / base_img.size[0], styled_1mp.size[1] / base_img.size[1]

    words = []
    if regions:
        # 对每个文字区: inpaint 抹旧字 -> PIL 按原图风格渲新字
        cur_img = styled_1mp
        cur_name = f"{tag}_stage_a.png"
        cur_img.save(INPUT / cur_name)
        for ri, r in enumerate(regions):
            word = r["banks"][ci % len(r["banks"])]
            words.append(word)
            # 从原图提取文字颜色 (按 1mp 坐标)
            orig_bbox_1mp = [int(v * s1) for v in r["bbox"][:2]] + \
                             [int(v * sy) if i % 2 else int(v * sx) for i, v in enumerate(r["bbox"][2:], start=0)]
            # bbox (x1,y1,x2,y2) 统一用 sx, sy
            bx1 = int(r["bbox"][0] * sx); by1 = int(r["bbox"][1] * sy)
            bx2 = int(r["bbox"][2] * sx); by2 = int(r["bbox"][3] * sy)
            fill, _bg = extract_text_style(base_img, r["bbox"])
            print(f"    region {ri} '{word}' fill={fill}", flush=True)

            # 1) inpaint 抹旧字
            mask = make_text_mask(cur_img.size, [(bx1, by1, bx2, by2)], dilate=18)
            mask_name = f"{tag}_r{ri}_mask.png"
            mask.save(INPUT / mask_name)
            inpaint_prefix = f"ai_{ts}_{tag}_r{ri}_inpaint"
            g2 = build_inpaint(cur_name, mask_name, seed0 + 7 + ri, inpaint_prefix,
                                cur_img.size[0], cur_img.size[1])
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
    g3 = build_inpaint  # placeholder; use build_fission's upscaler part
    # reuse 4x upscale via a dedicated graph
    up_prefix = f"ai_{ts}_{tag}_up"
    g3 = {"U": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}},
          "I": {"class_type": "LoadImage", "inputs": {"image": up_in}},
          "UP": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["U", 0], "image": ["I", 0]}},
          "S": {"class_type": "SaveImage", "inputs": {"images": ["UP", 0], "filename_prefix": up_prefix}}}
    pid3 = submit(g3, f"ai_{ts}_{tag}_u")
    if not pid3: print("    upscale submit failed", flush=True); return None
    out3 = wait_outputs(pid3, up_prefix, timeout=300)
    if not out3: print("    upscale failed", flush=True); return None
    final = Image.open(io.BytesIO(out3["up"] or out3["1mp"])).convert("RGB")
    final_path = out_dir / f"{tag}_final.png"
    final.save(final_path)
    edited.save(out_dir / f"{tag}_1mp.png")
    print(f"    final saved -> {final_path}", flush=True)
    return {"tag": tag, "final": final_path, "words": words, "style": style}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()
    ts = int(time.time())
    out_dir = ROOT / "jobs" / f"text_fission_v4_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== text_fission_v4 -> {out_dir} ===", flush=True)

    names = args.images or list(CONFIG.keys())
    if args.test:
        names = ["pinterest_denim_3.jpg"]
    seed0 = 800301
    results = []
    for fi, fname in enumerate(names):
        cfg = CONFIG[fname]
        n_combos = 1 if args.test else 3  # v4 默认 3 个变体(更明显裂变)
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
        words = ",".join(r["words"]) if r["words"] else "(纯图裂变)"
        rel = Path(r["final"]).name
        html.append(f"<div class='card'><img src='{rel}'>")
        html.append(f"<div class='cap'>{r['tag']}<br><b>{words}</b><br><span class='sty'>{r['style'][:60]}</span></div></div>")
    html.append("</body></html>")
    (out_dir / "gallery.html").write_text("\n".join(html), encoding="utf-8")
    print(f"\n=== done: {len(results)} combos -> {out_dir}/gallery.html ===", flush=True)


if __name__ == "__main__":
    main()