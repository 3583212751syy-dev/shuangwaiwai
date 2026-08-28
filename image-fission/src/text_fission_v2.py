# -*- coding: utf-8 -*-
"""
text_fission_v2.py -- 强风格裂变 + AnyText 原地改字(无遮盖)

规则:
1. 输入 = 桌面「图裂变测试图」原始 jpg, 绝不用裂变后的图当输入。
2. 图片裂变: 同主体/同构图, 换明显不同画风 (Canny 中强度锁结构 + 较高 denoise, 去掉 IPAdapter 不再锁回原图)。
3. 文本裂变: 用 AnyText(通义) 在原图(裂变后)文字区 mask 内, 直接把侵权词改成新词并融合进原纹理 —— 不再用色块遮盖。
   字体挑近似原图的 (denim->Anton 粗黑, skull->UnifrakturMaguntia 哥特, eagle->Cinzel 衬线)。

用法:
  python src/text_fission_v2.py --test         # 只跑 denim_3 验证
  python src/text_fission_v2.py                # 全量 6 图
"""
import argparse, json, time, io, shutil
from pathlib import Path
import requests, numpy as np
from PIL import Image, ImageDraw
import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(r"E:/Desktop/图裂变测试图")
INPUT = ROOT / "ComfyUI" / "input"
COMFYUI = "http://127.0.0.1:8188"
CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
ANYTEXT_CKPT = "anytext_v1.1_fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"

# AnyText 编辑时输入给 SD1.5 的最长边(8 的倍数)
AT_MAX = 768

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
            "bbox": (10, 30, 726, 350), "orig": "UPCY",
            "anytext_font": "Anton-Regular.ttf",
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
            {"bbox": (175, 55, 555, 240), "orig": "TRUE",
             "anytext_font": "UnifrakturMaguntia-Book.ttf",
             "banks": ["REAPER", "RAVEN", "BONES", "CROW"]},
            {"bbox": (100, 750, 636, 940), "orig": "NEVER",
             "anytext_font": "UnifrakturMaguntia-Book.ttf",
             "banks": ["STILL", "FOREVER", "NEVER", "ENDLESS"]},
            {"bbox": (140, 940, 596, 1220), "orig": "DIES",
             "anytext_font": "UnifrakturMaguntia-Book.ttf",
             "banks": ["BREATHES", "MORE", "FADES", "LIVES"]},
        ],
    },
    "pinterest_eagle_2.jpg": {
        "styles": [
            "art deco, gold foil, geometric, luxurious",
            "high contrast woodcut linocut, bold black and cream",
            "cyberpunk neon, holographic, electric blue and pink",
        ],
        "regions": [{
            "bbox": (300, 680, 650, 800), "orig": "JACKE DIANNIES",
            "anytext_font": "AbrilFatface-Regular.ttf",
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


def fit_size(size_wh, target):
    """缩放使最长边 <= target 且为 8 的倍数; 若已更小则原样返回。"""
    w, h = size_wh
    if max(w, h) <= target:
        return w, h
    if w >= h:
        nw, nh = target, int(round(h * target / w))
    else:
        nh, nw = target, int(round(w * target / h))
    return (nw // 8) * 8, (nh // 8) * 8


def make_mask(size_wh, bboxes, dilate=22):
    """文字区 alpha=255 的遮罩 (AnyText 编辑区域)。"""
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
                try:
                    detail = str(msg[1])[:500]
                except Exception:
                    detail = str(msg)[:500]
                print(f"    exec error: {detail}", flush=True); return None
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
def build_restyle(orig_name, style_prompt, seed, prefix):
    """强风格裂变: Canny 中强度锁结构 + 高 denoise, 去掉 IPAdapter 让画风明显变化。"""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0], "resolution_steps": 64}}
    g["4"] = {"class_type": "Canny", "inputs": {"image": ["3", 0], "low_threshold": 0.05, "high_threshold": 0.18}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = (f"masterpiece, best quality, ultra detailed, {style_prompt}, "
           f"same subject and composition as the reference, sharp, clean")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": RESTYLE_NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": 0.6, "start_percent": 0.0, "end_percent": 0.92, "control_net": ["5", 0]}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
        "seed": seed, "steps": 35, "cfg": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": 0.7}}
    g["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["14"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["13", 0], "resolution_steps": 64}}
    g["16"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["15"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
        "upscale_model": ["16", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}}
    g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_1mp"}}
    return g


def build_anytext_edit(styled_name, mask_name, words, font_name, seed, prefix, w, h):
    """AnyText 文字编辑: 在裂变后的图上, mask 区域内直接把文字改成新词并融合, 无遮盖。"""
    g = {}
    g["L"] = {"class_type": "UL_AnyTextLoader", "inputs": {
        "ckpt_name": ANYTEXT_CKPT, "control_net_name": "None",
        "miaobi_clip": "None", "weight_dtype": "fp16", "init_device": "auto"}}
    g["IMG"] = {"class_type": "LoadImage", "inputs": {"image": styled_name}}
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
    print(f"\n[combo] {tag}  style='{style[:40]}...'", flush=True)

    INPUT.mkdir(parents=True, exist_ok=True)
    src_in = INPUT / orig.name
    if not src_in.exists() or src_in.stat().st_size != orig.stat().st_size:
        shutil.copy2(orig, src_in)

    regions = cfg.get("regions", [])

    # Stage A: 强风格裂变
    restyle_prefix = f"ai_{ts}_{tag}_restyle"
    g = build_restyle(orig.name, style, seed0 + 1, restyle_prefix)
    pid = submit(g, f"ai_{ts}_{tag}_a")
    if not pid:
        print("    restyle submit failed", flush=True); return None
    out = wait_outputs(pid, restyle_prefix, timeout=320)
    if not out:
        print("    restyle failed", flush=True); return None
    styled_1mp = Image.open(io.BytesIO(out["1mp"])).convert("RGB")
    print(f"    restyle OK ({len(out['1mp'])//1024}KB 1mp)", flush=True)

    # 适配 SD1.5: 缩放系数 + 最长边 768
    s1 = styled_1mp.size[0] / base_img.size[0]
    rw, rh = fit_size(styled_1mp.size, AT_MAX)
    styled_rs = styled_1mp.resize((rw, rh), Image.LANCZOS)
    styled_name = f"{tag}_styled.png"
    styled_rs.save(INPUT / styled_name)

    words = []
    if regions:
        # 逐区域单独跑 AnyText 编辑: 每区域 1 词 + 1 mask, 避免多区域 mask 连通域合并导致位置数不匹配
        scale = s1 * (rw / styled_1mp.size[0])
        font_name = regions[0].get("anytext_font", "Arial_Unicode.ttf")
        edited = styled_rs
        cur_name = styled_name
        for ri, r in enumerate(regions):
            word = r["banks"][ci % len(r["banks"])]
            words.append(word)
            x1, y1, x2, y2 = [int(v * scale) for v in r["bbox"]]
            mask = make_mask((rw, rh), [(x1, y1, x2, y2)], dilate=20)
            mask_name = f"{tag}_r{ri}_mask.png"
            mask.save(INPUT / mask_name)
            if ri > 0:
                cur_name = f"{tag}_r{ri - 1}_edit.png"
            edit_prefix = f"ai_{ts}_{tag}_r{ri}"
            g2 = build_anytext_edit(cur_name, mask_name, [word], font_name,
                                    seed0 + 7 + ri, edit_prefix, rw, rh)
            pid2 = submit(g2, f"ai_{ts}_{tag}_b{ri}")
            if not pid2:
                print("    edit submit failed", flush=True); return None
            out2 = wait_outputs(pid2, edit_prefix, timeout=320)
            if not out2:
                print("    edit failed", flush=True); return None
            edited = Image.open(io.BytesIO(out2["up"] or out2["1mp"])).convert("RGB")
            edited_name = f"{tag}_r{ri}_edit.png"
            edited.save(INPUT / edited_name)
            print(f"    region {ri} '{word}' edited OK", flush=True)
        up_in, up_prefix = edited_name, f"ai_{ts}_{tag}_up"
    else:
        up_in, up_prefix = styled_name, f"ai_{ts}_{tag}_up"

    # Stage C: 4x 放大成终稿
    g3 = build_upscale(up_in, up_prefix)
    pid3 = submit(g3, f"ai_{ts}_{tag}_c")
    if not pid3:
        print("    upscale submit failed", flush=True); return None
    out3 = wait_outputs(pid3, up_prefix, timeout=320)
    if not out3:
        print("    upscale failed", flush=True); return None
    final = Image.open(io.BytesIO(out3["up"] or out3["1mp"])).convert("RGB")
    final_path = out_dir / f"{tag}_final.png"
    final.save(final_path)
    styled_1mp.save(out_dir / f"{tag}_restyle.png")
    print(f"    final saved -> {final_path}", flush=True)
    return {"tag": tag, "final": final_path, "words": words, "style": style}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()
    ts = int(time.time())
    out_dir = ROOT / "jobs" / f"text_fission_v3_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== text_fission_v3 -> {out_dir} ===", flush=True)

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
