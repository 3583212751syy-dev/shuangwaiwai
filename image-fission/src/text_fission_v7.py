# -*- coding: utf-8 -*-
"""
text_fission_v7.py -- 最终方案:
  Stage A 图裂变: 高 IPAdapter(0.8) 锁色系 + Canny 0.75 锁位置 + denoise 0.5 内部细节自由重画
  Stage B 文字: AnyText2 v2.0 提取原图字体生成新词(首选), 失败回退 SDXL inpaint + PIL
  每张出图后自检色系和位置, 过了才交付拼图对照
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
ANYTEXT_CKPT = "anytext_v2.0.ckpt"  # v2.0: 从原图提取字体+颜色
UPSCALE = "4x_NMKD-Siax_200k.pth"

FONTS = {
    "impact":   r"C:/Windows/Fonts/impact.ttf",
    "oldengl":  r"C:/Windows/Fonts/OLDENGL.TTF",
    "stencil":  r"C:/Windows/Fonts/STENCIL.TTF",
    "rock":     r"C:/Windows/Fonts/ROCK.TTF",
    "gothicb":  r"C:/Windows/Fonts/GOTHICB.TTF",
}

CONFIG = {
    "pinterest_denim_3.jpg": {
        "styles": [
            "same color palette, same composition, free internal detail redraw variation",
        ],
        "regions": [{
            "bbox": (10, 30, 726, 350), "orig": "UPCY",
            "font_key": "impact",
            "banks": ["JEANS", "PATCH", "MOTH", "INDIGO"],
        }],
    },
    "pinterest_skull_5.jpg": {
        "styles": [
            "same color palette, same composition, free internal detail redraw variation",
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
            "same color palette, same composition, free internal detail redraw variation",
        ],
        "regions": [{
            "bbox": (300, 680, 650, 800), "orig": "JACKE DIANNIES",
            "font_key": "impact", "banks": ["RAPTOR", "EMPIRE", "CROWN", "STORM"],
        }],
    },
    "pinterest_metal_6.jpg": {
        "styles": ["same color palette, same composition, free internal detail redraw variation"],
        "regions": [],
    },
    "pinterest_camo_4.jpg": {
        "styles": ["same color palette, same composition, free internal detail redraw variation"],
        "regions": [],
    },
    "pinterest_illust_1.jpg": {
        "styles": ["same color palette, same composition, free internal detail redraw variation"],
        "regions": [],
    },
}

RESTYLE_NEG = ("text, words, letters, typography, font, alphabet, writing, watermark, "
               "signature, logo, badge, blurry, deformed, low quality, bad anatomy, "
               "mutation, cropped, jpeg artifacts, noise")


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
    return [tuple(map(int, centers[i])) for i in idx]


def color_dist(a, b): return float(np.linalg.norm(np.array(a) - np.array(b)))


def extract_text_color(orig_img, bbox):
    """中心 60% 区域, 找与背景对比度最大的色."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = y2-y1, x2-x1
    cx1, cy1 = int(x1 + W*0.2), int(y1 + H*0.2)
    cx2, cy2 = int(x2 - W*0.2), int(y2 - H*0.2)
    crop = np.array(orig_img.crop((cx1, cy1, cx2, cy2)).convert("RGB"))
    colors = kmeans_colors(crop, k=4)
    bg = colors[0]
    cand = [c for c in colors[1:] if color_dist(c, bg) > 30]
    if not cand: cand = colors[1:]
    return max(cand, key=lambda c: color_dist(c, bg))


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
    stroke = (0,0,0,255)
    sw = max(3, int(best*0.08))
    d.text((tx, ty), word_up, font=fnt, fill=fill_a, stroke_width=sw, stroke_fill=stroke)
    text_layer = canvas.resize((W, H), Image.LANCZOS)
    sh = text_layer.copy()
    arr = np.array(sh); arr[arr[..., 3] > 20] = [0,0,0,140]
    sh = Image.fromarray(arr, "RGBA")
    base.paste(sh, (x1, y1), sh)
    base.paste(text_layer, (x1, y1), text_layer)
    return base.convert("RGB")


def color_close(c1, c2, tol=30):
    return all(abs(a-b) <= tol for a, b in zip(c1, c2))


def self_check_color(img, orig_img, bbox_1mp):
    """自检: fission 图色系不能漂 (牛仔布不能变铜绿等). 中心区域主色应接近原图主色."""
    arr = np.array(img.convert("RGB"))
    arr_o = np.array(orig_img.convert("RGB"))
    # 取图像中心块采样主色
    h, w = arr.shape[:2]
    cx, cy = w//2, h//2
    sz = min(w, h)//4
    center = arr[cy-sz:cy+sz, cx-sz:cx+sz]
    center_o = arr_o[cy-sz:cy+sz, cx-sz:cx+sz]
    mc = tuple(np.array(center.reshape(-1, 3).mean(axis=0)).astype(int))
    mc_o = tuple(np.array(center_o.reshape(-1, 3).mean(axis=0)).astype(int))
    return color_close(mc, mc_o, tol=40), f"fission_center={mc} orig_center={mc_o}"


# ---------- graph builders ----------
def build_restyle(orig_name, style_prompt, seed, prefix):
    """色系+构图锁定, 内部细节自由重画: Canny 0.75 + IPAdapter 0.8 + denoise 0.5."""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["4"] = {"class_type": "Canny", "inputs": {"image": ["3", 0], "low_threshold": 0.05, "high_threshold": 0.18}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = (f"masterpiece, best quality, ultra detailed, {style_prompt}, "
           f"preserve exact color palette and composition, redesigned details, sharp, clean")
    neg = RESTYLE_NEG
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": 0.75, "start_percent": 0.0, "end_percent": 0.9, "control_net": ["5", 0]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1], "image": ["2", 0],
        "weight": 0.8, "weight_type": "style transfer", "combine_embeds": "average",
        "embeds_scaling": "V only", "start_at": 0.0, "end_at": 0.9, "noise": 0.05}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
        "seed": seed, "steps": 30, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.5}}  # ← 0.5, 内部细节重画但色系不漂
    g["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["14"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["13", 0], "resolution_steps": 64}}
    g["16"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["15"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
        "upscale_model": ["16", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


def build_anytext_edit(img_name, mask_name, words, font_name, seed, prefix, w, h):
    """AnyText v2.0 edit: 从原图提取字体+颜色, 生成 words."""
    g = {}
    g["L"] = {"class_type": "UL_AnyTextLoader", "inputs": {
        "ckpt_name": ANYTEXT_CKPT, "control_net_name": "None",
        "miaobi_clip": "None", "weight_dtype": "fp32", "init_device": "auto"}}  # v2.0 fp32 稳定
    g["IMG"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
    g["MSK"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "alpha"}}
    g["EL"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    a_prompt = ("best quality, extremely detailed, 4k, HD, super legible text, clear text edges, "
                "clear strokes, neat writing, seamless background texture, no watermarks")
    n_prompt = ("low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, "
                "watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture")
    qwords = " ".join(f'"{w}"' for w in words)
    g["FMT"] = {"class_type": "UL_AnyTextFormatter", "inputs": {
        "prompt": (f"a high quality product image, keep the background and every unmasked area "
                   f"completely unchanged, only rewrite the masked text region with the text {qwords}")}}
    g["ENC"] = {"class_type": "UL_AnyTextEncoder", "inputs": {
        "model": ["L", 0], "mask": ["MSK", 0], "prompt": ["FMT", 0], "texts": ["FMT", 1],
        "latent": ["EL", 0], "image": ["IMG", 0],
        "font_name": font_name, "mode": False, "sort_radio": True,
        "a_prompt": a_prompt, "n_prompt": n_prompt, "random_mask": False, "revise_pos": False}}
    g["SMP"] = {"class_type": "UL_AnyTextSampler", "inputs": {
        "model": ["L", 0], "positive": ["ENC", 0], "negative": ["ENC", 1],
        "seed": seed, "steps": 20, "cfg": 9.0, "strength": 1.0,
        "attnx_scale": 1.0, "eta": 0.0, "keep_load": True, "keep_device": True}}
    g["DEC"] = {"class_type": "VAEDecode", "inputs": {"samples": ["SMP", 0], "vae": ["L", 1]}}
    g["SAV"] = {"class_type": "SaveImage", "inputs": {"images": ["DEC", 0], "filename_prefix": prefix}}
    return g


def build_inpaint(img_name, mask_name, seed, prefix):
    """SDXL inpaint 抹旧字(回退方案)."""
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

    # Stage A: 图裂变 (色系+构图锁定)
    fission_prefix = f"ai_{ts}_{tag}_fission"
    g = build_restyle(orig.name, style, seed0 + 1, fission_prefix)
    pid = submit(g, f"ai_{ts}_{tag}_f")
    if not pid: print("    fission submit failed", flush=True); return None
    out = wait_outputs(pid, fission_prefix, timeout=300)
    if not out: print("    fission failed", flush=True); return None
    styled_1mp = Image.open(io.BytesIO(out["1mp"])).convert("RGB")
    print(f"    fission OK ({len(out['1mp'])//1024}KB 1mp)", flush=True)

    # 自检色系
    color_ok, color_info = self_check_color(styled_1mp, base_img, None)
    print(f"    [SELF-CHECK color] {'OK' if color_ok else 'WARN'}: {color_info}", flush=True)
    if not color_ok:
        print(f"    色系偏离, 丢弃 (后续换 seed/降 denoise)", flush=True)
        return None

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
            fill = extract_text_color(base_img, r["bbox"])
            print(f"    region {ri} '{word}' fill={fill}", flush=True)

            mask = make_text_mask(cur_img.size, [(bx1, by1, bx2, by2)], dilate=20)
            mask_name = f"{tag}_r{ri}_mask.png"
            mask.save(INPUT / mask_name)

            # Stage B1: 试 AnyText v2.0 (字体提取)
            edit_prefix = f"ai_{ts}_{tag}_r{ri}_v2edit"
            g2 = build_anytext_edit(cur_name, mask_name, [word],
                                    r.get("anytext_font","Anton-Regular.ttf"),
                                    seed0 + 7 + ri, edit_prefix,
                                    cur_img.size[0], cur_img.size[1])
            pid2 = submit(g2, f"ai_{ts}_{tag}_b{ri}")
            edited_v2_ok = False
            if pid2:
                out2 = wait_outputs(pid2, edit_prefix, timeout=300)
                if out2:
                    try:
                        tmp_edited = Image.open(io.BytesIO(out2["up"] or out2["1mp"])).convert("RGB")
                        tmp_edited.save(INPUT / f"{tag}_r{ri}_v2try.png")
                        edited_v2_ok = True
                        cur_img = tmp_edited
                        cur_name = f"{tag}_r{ri}_v2try.png"
                        print(f"    v2.0 edit OK, 用 v2.0 结果", flush=True)
                    except Exception:
                        pass
            if not edited_v2_ok:
                print(f"    v2.0 edit 失败, 回退到 inpaint+PIL", flush=True)
                # 回退: inpaint 抹旧字
                inpaint_prefix = f"ai_{ts}_{tag}_r{ri}_inpaint"
                g3 = build_inpaint(cur_name, mask_name, seed0 + 13 + ri, inpaint_prefix)
                pid3 = submit(g3, f"ai_{ts}_{tag}_f{ri}")
                if not pid3: print("    inpaint submit failed", flush=True); return None
                out3 = wait_outputs(pid3, inpaint_prefix, timeout=300)
                if not out3: print("    inpaint failed", flush=True); return None
                cur_img = Image.open(io.BytesIO(out3["up"] or out3["1mp"])).convert("RGB")
                cur_name = f"{tag}_r{ri}_inpainted.png"
                cur_img.save(INPUT / cur_name)
                # PIL 渲新字
                cur_img = render_text(cur_img, word, (bx1, by1, bx2, by2), r.get("font_key","impact"), fill)
                cur_name = f"{tag}_r{ri}_texted.png"
                cur_img.save(INPUT / cur_name)

        edited = cur_img
        up_in = cur_name
    else:
        edited = styled_1mp
        up_in = f"{tag}_stage_a.png"

    # Stage C: 4x 放大
    up_prefix = f"ai_{ts}_{tag}_up"
    g4 = build_upscale(up_in, up_prefix)
    pid4 = submit(g4, f"ai_{ts}_{tag}_u")
    if not pid4: print("    upscale submit failed", flush=True); return None
    out4 = wait_outputs(pid4, up_prefix, timeout=300)
    if not out4: print("    upscale failed", flush=True); return None
    final = Image.open(io.BytesIO(out4["up"] or out4["1mp"])).convert("RGB")
    final_path = out_dir / f"{tag}_final.png"
    final.save(final_path)

    styled_1mp.save(out_dir / f"{tag}_1mp.png")

    # 拼图对照
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
    out_dir = ROOT / "jobs" / f"text_fission_v7_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== text_fission_v7 -> {out_dir} ===", flush=True)

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
            ".card img{height:460px;display:block}",
            ".cap{padding:6px 10px;font-size:13px}",
            ".sty{color:#9cf;font-size:11px}",
            "h2{color:#fd6}", "</style></head><body><h1>v7 图裂变 (色系+构图锁定 / 内部细节自由重画 / 字体原样换词)</h1>"]
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