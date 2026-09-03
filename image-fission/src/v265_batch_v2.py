"""
v265_batch_v2 — 改进版批量裂变 (更强裂变 + 强 no-text)

相比 v265_batch_simple.py 的改动:
  - IPAdapter 权重 0.65 -> 0.50 (降权, 让裂变更明显, 不再锁死原图)
  - denoise 0.55 -> 0.78 (主体角度/姿态/细节变化更大, 利于防关联)
  - POS 强化 "changed pose / rotated subject / altered details" + 强 no-text
  - NEG 强化 no-text / no-brand / no-logo / no-watermark
  - 跳过 _SKIP 侵权素材
  - 支持 --picks (按文件名片段匹配) / --limit (只跑前 N 张) 便于小批量验证
  - 输出到 jobs/v265_batch2/ 避免覆盖 v1 结果

用法:
  python src/v265_batch_v2.py                 # 全量 25 张
  python src/v265_batch_v2.py --limit 4       # 前 4 张验证
  python src/v265_batch_v2.py --picks 6978,Pinterest  # 只跑含这些片段的图
"""
import argparse
import json
import sys
import time
import uuid
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC_DIR = Path("E:/Desktop/图裂变测试图")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "v265_batch2"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
LORA = "add-detail-xl.safetensors"
LORA_DETAIL = 0.25

POS = ("high quality vector graphic, t-shirt print design, keep the same overall style "
       "and color palette, but change the subject pose and angle, rotate the composition "
       "slightly, alter minor details, flat illustration, clean edges, solid colors, "
       "centered composition, NO text, NO letters, NO words, NO brand name, NO logo, "
       "NO watermark, NO signature")
NEG = ("text, letters, words, readable text, brand name, logo, watermark, signature, "
       "3d render, photorealistic, realistic, blurry, low quality, noise, grain, "
       "multiple subjects, duplicated subject, deformed, mutated, extra limbs")


def submit(wf):
    data = json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
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
    for nid, node in entry.get("outputs", {}).items():
        if "images" in node:
            for im in node["images"]:
                return Path(COMFY_OUTPUT) / im["filename"]
    return None


def build_workflow(img_name, seed, denoise=0.78, ipa=0.50):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
    g["3"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}}
    g["4"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["5"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["4", 1], "image": ["2", 0],
        "weight": ipa, "weight_type": "style transfer", "combine_embeds": "average",
        "start_at": 0.0, "end_at": 0.9, "noise": 0.08, "embeds_scaling": "V only"}}
    g["6"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["5", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["6", 1], "text": POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["6", 1], "text": NEG}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["6", 0], "positive": ["pg", 0], "negative": ["ng", 0],
        "latent_image": ["3", 0], "seed": seed, "steps": 30, "cfg": 7.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": denoise}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    prefix = Path(img_name).stem[:18]
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v265b2_{prefix}"}}
    return g


def copy_to_input(src_path):
    short = f"v265b2_{src_path.stem[:18]}{src_path.suffix.lower()}"
    dst = COMFY_INPUT / short
    if not dst.exists():
        img = Image.open(src_path).convert("RGB")
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024))
        img.save(dst, quality=95)
    return short, dst


def process_one(src_path, seed):
    if src_path.name.startswith("_SKIP"):
        print(f"[skip] {src_path.name}")
        return None
    print(f"\n[v265b2] processing {src_path.name} ...")
    try:
        img_name, dst = copy_to_input(src_path)
        wf = build_workflow(img_name, seed)
        r = submit(wf)
        if "error" in r:
            print(f"  submit error: {r['error']}")
            return None
        entry = poll(r["prompt_id"], timeout_s=600)
        raw = collect_out(entry)
        if not raw:
            print("  no output")
            return None
        out = JOB / f"{src_path.stem}_fission_v2.png"
        Image.open(raw).convert("RGB").save(out, quality=95)
        print(f"  -> {out}")
        return out
    except Exception as e:
        print(f"  FAIL: {e}")
        return None


def make_contact_sheet(paths, cols=5):
    paths = [p for p in paths if p and Path(p).exists()]
    if not paths:
        return None
    thumb = (240, 300)
    rows = (len(paths) + cols - 1) // cols
    grid_w = cols * thumb[0] + (cols + 1) * 10
    grid_h = rows * thumb[1] + (rows + 1) * 10 + 40
    canvas = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, p in enumerate(paths):
        c, r = i % cols, i // cols
        x = 10 + c * thumb[0]
        y = 10 + r * thumb[1]
        im = Image.open(p).convert("RGB").resize(thumb, Image.LANCZOS)
        canvas.paste(im, (x, y))
        name = p.stem[:18]
        draw.text((x, y + thumb[1] + 4), name, fill=(30, 30, 30), font=font)
    out = JOB / "_contact_sheet_v2.png"
    canvas.save(out, quality=92)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only run first N images")
    ap.add_argument("--picks", type=str, default="", help="comma-separated filename fragments to match")
    args = ap.parse_args()

    picks = [s.strip() for s in args.picks.split(",") if s.strip()] if args.picks else None
    imgs = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    if picks:
        imgs = [p for p in imgs if any(frag.lower() in p.name.lower() for frag in picks)]
    if args.limit:
        imgs = imgs[:args.limit]
    print(f"[v265b2] selected {len(imgs)} images")
    outputs = []
    for i, p in enumerate(imgs):
        out = process_one(p, 426500 + i * 137)
        if out:
            outputs.append(out)
    sheet = make_contact_sheet(outputs)
    print(f"\n[OK] total generated: {len(outputs)}/{len(imgs)}")
    if sheet:
        print(f"[OK] contact sheet -> {sheet}")


if __name__ == "__main__":
    main()
