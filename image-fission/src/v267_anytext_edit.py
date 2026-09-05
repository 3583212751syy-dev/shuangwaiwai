#!/usr/bin/env python3
"""
v267: AnyText v1.1 文字编辑模式 —— 单次调用多行编辑
- 手动/自动检测文字行 bbox
- 生成单通道灰度 mask（每行独立白块，连通域清晰分离）
- 直接传入 texts LIST，不经过 Formatter
- 单次 ComfyUI AnyText 编辑：擦旧字 + 按位置/字体重绘新字
- 输出 resize 回原尺寸并只替换 mask 区域
"""
import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

COMFY = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("E:/Desktop/双接口/image-fission/ComfyUI/output")
OUT = Path("E:/Desktop/双接口/image-fission/jobs/v267")
OUT.mkdir(parents=True, exist_ok=True)


def comfy_upload_image(pil_img, name):
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    r = requests.post(f"{COMFY}/upload/image", files={"image": (name, buf, "image/png")}, data={"overwrite": "true"})
    r.raise_for_status()
    return r.json()["name"]


def queue_and_get(wf):
    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf}, timeout=60)
    r.raise_for_status()
    pid = r.json()["prompt_id"]
    print(f"  prompt queued: {pid}")
    while True:
        time.sleep(2)
        hist = requests.get(f"{COMFY}/history/{pid}", timeout=60).json()
        if pid in hist:
            item = hist[pid]
            status = item.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"ComfyUI execution error: {msgs}")
            outputs = item.get("outputs", {})
            for nid, node_out in outputs.items():
                if "images" in node_out:
                    return node_out["images"]
            return []


def detect_text(pil_img):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    arr = np.array(pil_img.convert("RGB"))
    res = reader.readtext(arr, paragraph=False, detail=1, low_text=0.25, link_threshold=0.25)
    boxes = []
    for poly, text, conf in res:
        if conf < 0.2:
            continue
        poly = np.array(poly, dtype=np.int32)
        xs, ys = poly[:, 0], poly[:, 1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if (x2 - x1) < 20 or (y2 - y1) < 12:
            continue
        boxes.append({"poly": poly, "bbox": (x1, y1, x2, y2), "text": text, "conf": conf, "cy": (y1 + y2) // 2, "cx": (x1 + x2) // 2})
    boxes.sort(key=lambda b: b["cy"])
    return boxes


def merge_text_lines(boxes, y_gap=120, x_gap=200):
    """合并同一行/邻近的小框."""
    if not boxes:
        return []
    rows = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        merged = False
        for r in rows:
            if abs(r["cy"] - b["cy"]) < y_gap and abs(r["cx"] - b["cx"]) < (r["w"] + b["w"]) // 2 + x_gap:
                r["x1"] = min(r["x1"], x1)
                r["y1"] = min(r["y1"], y1)
                r["x2"] = max(r["x2"], x2)
                r["y2"] = max(r["y2"], y2)
                r["texts"].append(b["text"])
                r["cy"] = (r["y1"] + r["y2"]) // 2
                r["cx"] = (r["x1"] + r["x2"]) // 2
                r["w"] = r["x2"] - r["x1"]
                r["h"] = r["y2"] - r["y1"]
                merged = True
                break
        if not merged:
            rows.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
                         "w": x2 - x1, "h": y2 - y1, "texts": [b["text"]]})
    lines = []
    for r in rows:
        lines.append({"bbox": (r["x1"], r["y1"], r["x2"], r["y2"]),
                      "cy": r["cy"], "cx": r["cx"],
                      "orig_text": " ".join(r["texts"])})
    return lines


def manual_bat_rows():
    """BACARDÍ 蝙蝠图手动标定文字行（bbox 之间保留足够间隙，确保连通域分离）."""
    return [
        {"name": "arc",   "bbox": (250, 300, 1300, 460),  "cy": 380,  "cx": 775},
        {"name": "est",   "bbox": (350, 780, 580, 900),   "cy": 840,  "cx": 465},
        {"name": "1862",  "bbox": (970, 780, 1200, 900),  "cy": 840,  "cx": 1085},
        {"name": "main",  "bbox": (220, 940, 1330, 1120), "cy": 1030, "cx": 775},
        {"name": "sub",   "bbox": (380, 1180, 1170, 1380),"cy": 1280, "cx": 775},
    ]


def sort_lines_for_anytext(lines):
    """按 AnyText 的 sort_radio=False(垂直)排序：先 y，相近再 x."""
    gap = 102
    return sorted(lines, key=lambda l: (l["cy"] // gap, l["cx"] // gap))


def build_mask(img_size, lines, pad=6):
    """生成单通道灰度 mask：每行文字区为独立白块，行之间保留黑缝隙."""
    W, H = img_size
    mask = np.zeros((H, W), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line["bbox"]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad)
        y2 = min(H, y2 + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        line["mask_bbox"] = (x1, y1, x2, y2)
    return mask


def verify_mask_components(mask_np, expected):
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    print(f"  mask 连通域数量: {n_labels - 1} (预期 {expected})")
    if n_labels - 1 < expected:
        print("  警告：连通域不足，可能行间距太近被合并")
    return n_labels - 1


def build_prompt_with_texts(texts):
    """构造含引号文本的 prompt，供 AnyText Formatter 解析出 texts LIST."""
    quoted = [f'"{t}"' for t in texts]
    return "a logo badge design with purple bat emblem and elegant typography, with text: " + ", ".join(quoted) + "."


def build_workflow(src_name, mask_name, texts, font_name="AbrilFatface-Regular.ttf"):
    prompt = build_prompt_with_texts(texts)
    wf = {
        "3": {"class_type": "LoadImage", "inputs": {"image": src_name}},
        "4": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "5": {"class_type": "UL_AnyTextLoader", "inputs": {
            "ckpt_name": "anytext_v1.1_fp16.safetensors",
            "control_net_name": "None",
            "miaobi_clip": "None",
            "weight_dtype": "fp16",
            "init_device": "auto",
        }},
        "11": {"class_type": "UL_AnyTextFormatter", "inputs": {"prompt": prompt}},
        "12": {"class_type": "ImageToMask", "inputs": {"image": ["4", 0], "channel": "red"}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["5", 1]}},
        "7": {"class_type": "UL_AnyTextEncoder", "inputs": {
            "model": ["5", 0],
            "mask": ["12", 0],
            "prompt": ["11", 0],
            "texts": ["11", 1],
            "latent": ["6", 0],
            "font_name": font_name,
            "mode": False,
            "sort_radio": False,
            "a_prompt": "best quality, sharp text, clear edges, high contrast, legible typography",
            "n_prompt": "low quality, blurry, messy text, distorted letters, watermark, unreadable",
            "random_mask": False,
            "revise_pos": False,
            "image": ["3", 0],
            "font_apply": True,
            "show_glyph": False,
        }},
        "8": {"class_type": "UL_AnyTextSampler", "inputs": {
            "model": ["5", 0],
            "positive": ["7", 0],
            "negative": ["7", 1],
            "seed": random.randint(1, 2147483647),
            "steps": 25,
            "cfg": 7.5,
            "strength": 1.0,
            "attnx_scale": 1.0,
            "eta": 0.0,
            "keep_load": True,
            "keep_device": True,
        }},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["5", 1]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "v267_anytext"}},
    }
    return wf


def pick_new_words(n_lines, variant):
    pools = {
        "nightbat": ["NIGHTBAT", "EST", "1862", "DUSKHEART", "WINGS OF DARK"],
        "shadow":   ["SHADOWBAT", "EST", "1862", "MOONHEART", "SONAR OATH"],
        "moonbat":  ["MOONBAT", "EST", "1862", "ECHOHEART", "LA CASA DEL ALA"],
    }
    defaults = pools.get(variant, pools["nightbat"])
    if n_lines <= len(defaults):
        return defaults[:n_lines]
    return defaults + ["TEXT"] * (n_lines - len(defaults))


def composite_back(src_pil, edited_pil, mask_np, out_path):
    src = src_pil.convert("RGB")
    W, H = src.size
    edited = edited_pil.resize((W, H), Image.LANCZOS)
    mask = Image.fromarray((mask_np > 0).astype(np.uint8) * 255).resize((W, H), Image.LANCZOS)
    out = Image.composite(edited, src, mask)
    out.save(out_path, "PNG")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--variant", default="nightbat")
    parser.add_argument("--font", default="AbrilFatface-Regular.ttf")
    parser.add_argument("--manual", action="store_true", help="使用手动标定的 BACARDÍ 文字行")
    args = parser.parse_args()

    src_path = Path(args.src)
    print(f"[1/4] 读取原图: {src_path.name}")
    src_pil = Image.open(src_path).convert("RGB")

    if args.manual:
        lines = manual_bat_rows()
        print(f"  使用手动标定 {len(lines)} 行")
    else:
        print("  EasyOCR 检测文字...")
        boxes = detect_text(src_pil)
        print(f"  原始框: {len(boxes)} 个")
        for b in boxes:
            print(f"    {b['bbox']} -> '{b['text']}'")
        lines = merge_text_lines(boxes, y_gap=120, x_gap=250)
        print(f"  合并后: {len(lines)} 行")
    for line in lines:
        print(f"    {line['bbox']} -> '{line.get('orig_text', line.get('name', ''))}'")

    lines = sort_lines_for_anytext(lines)
    print("  按 AnyText 垂直排序后:")
    for line in lines:
        print(f"    {line['bbox']}")

    print("[2/4] 生成灰度 mask")
    mask_np = build_mask(src_pil.size, lines, pad=6)
    verify_mask_components(mask_np, len(lines))
    # 保存可视化
    Image.fromarray(mask_np).save(OUT / f"{src_path.stem}_mask_debug.png")

    # 转 RGB 再上传，避免 ComfyUI LoadImage 对灰度 mask 输出异常
    mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=2)
    mask_pil = Image.fromarray(mask_rgb)
    mask_name = f"v267_mask_{src_path.stem}.png"
    comfy_upload_image(mask_pil, mask_name)
    src_name = f"v267_src_{src_path.stem}.png"
    comfy_upload_image(src_pil, src_name)

    words = pick_new_words(len(lines), args.variant)
    print(f"[3/4] 新文本: {words}")

    print("[4/4] 提交 ComfyUI AnyText 编辑")
    wf = build_workflow(src_name, mask_name, words, font_name=args.font)
    imgs = queue_and_get(wf)
    if not imgs:
        raise RuntimeError("ComfyUI 没有返回图片")
    img_name = imgs[0]["filename"]
    at_path = OUTPUT_DIR / img_name
    print(f"  AnyText 输出: {at_path}")

    final_path = OUT / f"{src_path.stem}_{args.variant}_final.png"
    compare_path = OUT / f"{src_path.stem}_{args.variant}_compare.png"
    at_pil = Image.open(at_path).convert("RGB")
    final = composite_back(src_pil, at_pil, mask_np, final_path)

    cw = src_pil.width * 2
    ch = src_pil.height
    compare = Image.new("RGB", (cw, ch), (245, 245, 245))
    compare.paste(src_pil, (0, 0))
    compare.paste(final, (src_pil.width, 0))
    compare.save(compare_path)
    print(f"  完成 -> {final_path}")
    print(f"  对照 -> {compare_path}")


if __name__ == "__main__":
    main()
