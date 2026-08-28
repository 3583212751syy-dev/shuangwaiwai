# -*- coding: utf-8 -*-
"""
text_fission_v8.py -- 最终方案 (denim 单图验证)
  Stage A 图裂变: Canny 0.75 锁位置 + IPAdapter 0.85 锁色系 + denoise 0.42 (防背景漂)
                   prompt 显式锁 "light gray/white clean background"
  Stage B 去旧字: inpaint denoise 0.25 (只抹文字 bbox, 不扩散到蝴蝶/背景)
  Stage C 新词: 从原图 UPCY 区域"偷"材质带, 迁移到新词每个字母 (复刻复杂拼贴材质)
  Stage D 自检: 程序化指标 (背景四角浅色 / 中心色系 / 文字区有深色覆盖 / 清晰度)
  说明: 当前模型不能读图, 自我检查全靠量化指标, 最终由用户肉眼验收
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

FONTS = {
    "impact":   r"C:/Windows/Fonts/impact.ttf",
    "oldengl":  r"C:/Windows/Fonts/OLDENGL.TTF",
    "stencil":  r"C:/Windows/Fonts/STENCIL.TTF",
    "rock":     r"C:/Windows/Fonts/ROCK.TTF",
    "gothicb":  r"C:/Windows/Fonts/GOTHICB.TTF",
}

# denim 测试: 3 个候选, 词义全来自图内容 (牛仔布/蝴蝶/拼贴)
CONFIG = {
    "pinterest_denim_3.jpg": {
        "styles": [
            "same color palette, same composition, free internal detail redraw variation",
        ],
        "material_bank": True,  # 用 UPCY 区域做材质带迁移
        "regions": [{
            "bbox": (10, 30, 726, 350), "orig": "UPCY",
            "font_key": "impact",
            "banks": ["BUTTERFLY", "DENIM", "FLUTTER"],  # 词义相关候选
        }],
    },
}

RESTYLE_NEG = ("text, words, letters, typography, font, alphabet, writing, watermark, "
               "signature, logo, badge, blurry, deformed, low quality, bad anatomy, "
               "mutation, cropped, jpeg artifacts, noise, color shift, different background")


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


def color_close(c1, c2, tol=40):
    return all(abs(a-b) <= tol for a, b in zip(c1, c2))


def make_text_mask(size_wh, bboxes, dilate=14):
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


def render_text_material(fission_img, word, bbox, font_key, mat_src):
    """从原图 UPCY 区域提取材质带, 迁移到新词每个字母 (复刻复杂拼贴材质)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    W, H = x2 - x1, y2 - y1
    if W <= 0 or H <= 0:
        return fission_img
    font_path = FONTS.get(font_key, FONTS["impact"])
    canvas = Image.new("RGBA", (W*4, H*4), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    lo, hi, best = 10, H*4, 10
    word_up = word.upper()
    for _ in range(24):
        mid = (lo + hi) // 2
        try:
            fnt = ImageFont.truetype(font_path, mid)
        except Exception:
            fnt = ImageFont.load_default()
        bb = d.textbbox((0, 0), word_up, font=fnt)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        if tw <= W*3.7 and th <= H*3.5:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    fnt = ImageFont.truetype(font_path, best)
    bb = d.textbbox((0, 0), word_up, font=fnt)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    tx = (W*4 - tw)//2 - bb[0]; ty = (H*4 - th)//2 - bb[1]
    d.text((tx, ty), word_up, font=fnt, fill=(255, 255, 255, 255))
    mask_layer = canvas.resize((W, H), Image.LANCZOS)
    mask_alpha = mask_layer.split()[3]
    # 材质底: 原 UPCY 区域缩放成 (W,H) -> 材质带横向分布 (铆钉蓝/黑/牛仔蓝/米白)
    mat = mat_src.resize((W, H), Image.LANCZOS).convert("RGBA")
    mat.putalpha(mask_alpha)
    base = fission_img.convert("RGBA")
    base.paste(mat, (x1, y1), mat)
    # 毛边/缝线描边 (模拟 frayed edges)
    m = np.array(mask_alpha)
    kk = np.ones((3, 3), np.uint8)
    md = cv2.dilate(m, kk, iterations=2)
    edge = np.clip(md - m, 0, 255).astype(np.uint8)
    edge_rgba = np.zeros((H, W, 4), np.uint8)
    edge_rgba[..., 0:3] = 18; edge_rgba[..., 3] = (edge * 0.85).astype(np.uint8)
    ei = Image.fromarray(edge_rgba, "RGBA")
    base.paste(ei, (x1, y1), ei)
    return base.convert("RGB")


def self_check_bg(img):
    """背景四角必须浅色, 不能漂成绿灰/橙棕."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    pad = min(60, h//6, w//6)
    corners = [arr[pad:pad*2, pad:pad*2],
               arr[pad:pad*2, -pad*2:-pad],
               arr[-pad*2:-pad, pad:pad*2],
               arr[-pad*2:-pad, -pad*2:-pad]]
    vals = [np.array(c.reshape(-1, 3).mean(0)) for c in corners]
    mean_val = np.mean(vals, axis=0)
    bright = float(mean_val.mean())
    green_bias = float(mean_val[1] - mean_val[0])   # 浅灰应 ~0; 绿灰偏正
    blue_bias = float(mean_val[2] - mean_val[0])
    ok = (bright > 150) and (abs(green_bias) < 45) and (abs(blue_bias) < 60)
    return ok, f"bg_corner_mean={tuple(map(int, mean_val))} bright={bright:.0f} gbias={green_bias:+.0f}"


def self_check_color(img, orig_img):
    arr = np.array(img.convert("RGB"))
    arr_o = np.array(orig_img.convert("RGB"))
    h, w = arr.shape[:2]
    cx, cy = w//2, h//2
    sz = min(w, h)//4
    center = arr[cy-sz:cy+sz, cx-sz:cx+sz]
    center_o = arr_o[cy-sz:cy+sz, cx-sz:cx+sz]
    mc = tuple(np.array(center.reshape(-1, 3).mean(0)).astype(int))
    mc_o = tuple(np.array(center_o.reshape(-1, 3).mean(0)).astype(int))
    return color_close(mc, mc_o, tol=45), f"fission_center={mc} orig_center={mc_o}"


def self_check_text(img, bbox):
    """文字区应有深色字母覆盖 (材质迁移成功)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return False, "bbox OOB"
    crop = np.array(img.convert("RGB"))
    reg = crop[y1:y2, x1:x2]
    gray = reg.mean(2)
    dark_ratio = float((gray < 140).mean())
    ok = dark_ratio > 0.04
    return ok, f"dark_ratio={dark_ratio:.3f}"


def self_check_sharp(img):
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap > 30, f"laplacian_var={lap:.1f}"


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
           f"denim patchwork butterfly collage, frayed fabric edges, stitched wing detail, "
           f"light gray white clean FLAT background, cool indigo and pale denim palette, "
           f"preserve exact composition and background, redesigned internal detail variation, "
           f"sharp, clean, KEEP BACKGROUND LIGHT AND NEUTRAL, no color shift")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": RESTYLE_NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": 0.78, "start_percent": 0.0, "end_percent": 0.9, "control_net": ["5", 0]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1], "image": ["2", 0],
        "weight": 0.85, "weight_type": "style transfer", "combine_embeds": "average",
        "embeds_scaling": "V only", "start_at": 0.0, "end_at": 0.9, "noise": 0.05}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
        "seed": seed, "steps": 30, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.42}}
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
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
    g["3"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "alpha"}}
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 0], "grow_mask_by": 2}}
    pos = "seamless light gray clean background texture, empty clean area, no text"
    neg = "text, words, letters, typography, watermark, signature, logo, blurry, low quality, color shift"
    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["4", 0],
        "seed": seed, "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.25}}   # ← 0.25, 只抹文字, 不扩散到背景/蝴蝶
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
    if not orig.exists():
        # fallback: ComfyUI input 里找
        orig = INPUT / fname
    base_img = Image.open(orig).convert("RGB")
    style = cfg["styles"][ci % len(cfg["styles"])]
    tag = f"{Path(fname).stem}_c{ci}"
    print(f"\n[combo] {tag}  word='{cfg['regions'][0]['banks'][ci % len(cfg['regions'][0]['banks'])]}'", flush=True)

    INPUT.mkdir(parents=True, exist_ok=True)
    src_in = INPUT / orig.name
    if not src_in.exists() or src_in.stat().st_size != orig.stat().st_size:
        shutil.copy2(orig, src_in)

    regions = cfg.get("regions", [])

    # Stage A: 图裂变
    fission_prefix = f"ai_{ts}_{tag}_fission"
    g = build_restyle(orig.name, style, seed0 + 1, fission_prefix)
    pid = submit(g, f"ai_{ts}_{tag}_f")
    if not pid:
        print("    fission submit failed", flush=True); return None
    out = wait_outputs(pid, fission_prefix, timeout=300)
    if not out:
        print("    fission failed", flush=True); return None
    styled_1mp = Image.open(io.BytesIO(out["1mp"])).convert("RGB")
    print(f"    fission OK ({len(out['1mp'])//1024}KB 1mp)", flush=True)

    # 自检: 色系 + 背景
    color_ok, color_info = self_check_color(styled_1mp, base_img)
    bg_ok, bg_info = self_check_bg(styled_1mp)
    print(f"    [SELF-CHECK color] {'OK' if color_ok else 'WARN'}: {color_info}", flush=True)
    print(f"    [SELF-CHECK bg   ] {'OK' if bg_ok else 'WARN'}: {bg_info}", flush=True)
    if not bg_ok:
        print("    !! 背景漂色, 此张丢弃 (换 seed 重跑)", flush=True)
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
            print(f"    region {ri} '{word}' bbox=({bx1},{by1},{bx2},{by2})", flush=True)

            mask = make_text_mask(cur_img.size, [(bx1, by1, bx2, by2)], dilate=14)
            mask_name = f"{tag}_r{ri}_mask.png"
            mask.save(INPUT / mask_name)

            # Stage B: inpaint 抹旧字 (低强度, 只文字区)
            inpaint_prefix = f"ai_{ts}_{tag}_r{ri}_inpaint"
            g3 = build_inpaint(cur_name, mask_name, seed0 + 13 + ri, inpaint_prefix)
            pid3 = submit(g3, f"ai_{ts}_{tag}_f{ri}")
            if not pid3:
                print("    inpaint submit failed", flush=True); return None
            out3 = wait_outputs(pid3, inpaint_prefix, timeout=300)
            if not out3:
                print("    inpaint failed", flush=True); return None
            cur_img = Image.open(io.BytesIO(out3["up"] or out3["1mp"])).convert("RGB")
            cur_name = f"{tag}_r{ri}_inpainted.png"
            cur_img.save(INPUT / cur_name)

            # Stage C: 材质迁移渲染新词
            mat_src = base_img.crop(tuple(int(v) for v in r["bbox"]))
            cur_img = render_text_material(cur_img, word, (bx1, by1, bx2, by2),
                                            r.get("font_key", "impact"), mat_src)
            cur_name = f"{tag}_r{ri}_texted.png"
            cur_img.save(INPUT / cur_name)

        edited = cur_img
        up_in = cur_name
    else:
        edited = styled_1mp
        up_in = f"{tag}_stage_a.png"

    # Stage D: 程序化自检 (文字 + 清晰度)  — 当前模型不能读图, 靠指标
    tb = regions[0]["bbox"] if regions else None
    if tb:
        tbx = (int(tb[0]*sx), int(tb[1]*sy), int(tb[2]*sx), int(tb[3]*sy))
        t_ok, t_info = self_check_text(edited, tbx)
        print(f"    [SELF-CHECK text ] {'OK' if t_ok else 'WARN'}: {t_info}", flush=True)
    s_ok, s_info = self_check_sharp(edited)
    print(f"    [SELF-CHECK sharp] {'OK' if s_ok else 'WARN'}: {s_info}", flush=True)

    # Stage E: 4x 放大
    up_prefix = f"ai_{ts}_{tag}_up"
    g4 = build_upscale(up_in, up_prefix)
    pid4 = submit(g4, f"ai_{ts}_{tag}_u")
    if not pid4:
        print("    upscale submit failed", flush=True); return None
    out4 = wait_outputs(pid4, up_prefix, timeout=300)
    if not out4:
        print("    upscale failed", flush=True); return None
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
    cd.text((o.width + gap + 10, 5), f"FISSIONED {words[0] if words else ''}", fill=(255, 200, 100))
    cd.line([(o.width + gap // 2, 0), (o.width + gap // 2, H + 40)], fill=(255, 255, 255), width=2)
    cmp_path = out_dir / f"{tag}_compare.png"
    canvas.save(cmp_path)
    print(f"    final -> {final_path}", flush=True)
    print(f"    compare-> {cmp_path}", flush=True)
    return {"tag": tag, "final": final_path, "compare": cmp_path, "words": words, "style": style}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()
    ts = int(time.time())
    out_dir = ROOT / "jobs" / f"text_fission_v8_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== text_fission_v8 -> {out_dir} ===", flush=True)

    names = args.images or list(CONFIG.keys())
    if args.test:
        names = ["pinterest_denim_3.jpg"]
    seed0 = 800301
    results = []
    for fi, fname in enumerate(names):
        cfg = CONFIG[fname]
        n_combos = 3 if args.test else 2   # denim 测试出 3 候选
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
            "h2{color:#fd6}", "</style></head><body><h1>v8 图裂变 (denim 验证: 锁背景/材质迁移/词义相关词)</h1>"]
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
