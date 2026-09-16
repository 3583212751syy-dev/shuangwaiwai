#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v268 — 本地去字 backend（LaMa + 扩散 inpaint）

硬规则：
  1) 永远禁止用矩形色块/背景色填充去盖住原文本。
  2) 必须先把旧字彻底擦除（背景由模型自然重建），再在同一位置渲染新词。
  3) 扩散 prompt 必须描述真实背景纹理，禁止用 "plain clean background" 压平画面。

流程：mask 白=去除区 -> inpaint 模型重建背景纹理 -> 返回干净 RGB PIL。
"""
import argparse
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import arc_text

PROJECT = Path("E:/Desktop/双接口/image-fission")
OUT = PROJECT / "jobs" / "v268"
OUT.mkdir(parents=True, exist_ok=True)
SRC_ROOT = Path("E:/Desktop/图裂变测试图")
FONT_PATH = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")
MODEL_PATH = PROJECT / "ComfyUI" / "models" / "RMBG" / "Lama" / "big-lama.pt"

_LAMA = None


def load_lama():
    global _LAMA
    if _LAMA is not None:
        return _LAMA
    src = str(MODEL_PATH)
    # torch.jit.load 在 Windows 上 C++ fopen 不支持中文路径 -> 拷到 ASCII 临时目录
    if any(ord(c) > 127 for c in src):
        tmp = os.path.join(tempfile.gettempdir(), "big-lama.pt")
        if not os.path.exists(tmp) or os.path.getsize(tmp) != os.path.getsize(src):
            shutil.copyfile(src, tmp)
        src = tmp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(src, map_location=dev)
    model.eval()
    model.to(dev)
    _LAMA = (model, dev)
    return _LAMA


def pad_image(image, is_mask=False):
    w, h = image.size
    if w % 8 != 0:
        w = w + (8 - w % 8)
    if h % 8 != 0:
        h = h + (8 - h % 8)
    fill = 0 if is_mask else None
    padded = Image.new(image.mode, (w, h), color=fill)
    padded.paste(image, (0, 0))
    return padded


def lama_inpaint(src_pil, mask_pil, removal_strength=230, edge_smoothness=8):
    """src_pil: RGB; mask_pil: 'L', 白=去除区。返回去字后的 RGB PIL。"""
    model, dev = load_lama()
    w, h = src_pil.size
    p_img = pad_image(src_pil)
    p_mask = pad_image(mask_pil, is_mask=True)
    if p_mask.size != p_img.size:
        p_mask = p_mask.resize(p_img.size, Image.LANCZOS)
    # 节点逻辑: invert -> blur -> threshold
    p_mask = ImageOps.invert(p_mask)
    p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=edge_smoothness))
    gray = p_mask.point(lambda x: 0 if x > removal_strength else 255)
    img_t = torch.from_numpy(np.array(p_img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    mask_t = torch.from_numpy(np.array(gray).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.inference_mode():
        res = model(img_t, mask_t)
    res_img = Image.fromarray((res[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    if res_img.width > w or res_img.height > h:
        res_img = res_img.crop((0, 0, w, h))
    return res_img.convert("RGB")


# ----------------------------------------------------------------------------
# 扩散 inpaint（升级版去字 backend，替换 Big-LaMa）
# 用本地 SD1.5-inpaint 扩散模型彻底擦除旧字并以周围纹理重建背景。
# 与 LaMa 一样：mask 白=去除区；不使用任何矩形色块填充。
# ----------------------------------------------------------------------------
_DIFF_PIPE = None
_DIFF_MODEL_DIR = None
_DEFAULT_DIFF_DIR = "E:/Desktop/双接口/image-fission/models/sd15_inpaint"


def set_diffusion_model_dir(d):
    global _DIFF_MODEL_DIR, _DIFF_PIPE
    _DIFF_MODEL_DIR = d
    _DIFF_PIPE = None


def _load_diff_pipe():
    global _DIFF_PIPE
    if _DIFF_PIPE is not None:
        return _DIFF_PIPE
    from diffusers import (StableDiffusionInpaintPipeline, DDIMScheduler,
                           UNet2DConditionModel, AutoencoderKL)
    from transformers import CLIPTextModel, CLIPTokenizer
    md = _DIFF_MODEL_DIR or os.environ.get("DIFFUSION_INPAINT_MODEL") or _DEFAULT_DIFF_DIR
    if not os.path.isdir(md):
        raise RuntimeError(f"diffusion inpaint 模型目录不存在: {md}；请先下载 SD1.5-inpaint")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionInpaintPipeline(
        vae=AutoencoderKL.from_pretrained(md, subfolder="vae", torch_dtype=dtype),
        text_encoder=CLIPTextModel.from_pretrained(md, subfolder="text_encoder", torch_dtype=dtype),
        tokenizer=CLIPTokenizer.from_pretrained(md, subfolder="tokenizer"),
        unet=UNet2DConditionModel.from_pretrained(md, subfolder="unet", torch_dtype=dtype),
        scheduler=DDIMScheduler.from_pretrained(md, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    _DIFF_PIPE = pipe
    return pipe


def diffusion_inpaint(src_pil, mask_pil, *, prompt=None,
                        negative_prompt="text, letters, words, watermark, logo, sign, typography, caption",
                        strength=1.0, num_inference_steps=50, guidance_scale=7.5, seed=8888):
    """src_pil: RGB; mask_pil: 'L'，白=去除区。返回去字后 RGB PIL（背景由扩散重建）。

    参数：
      prompt: 描述背景纹理/风格的 prompt。必须具体，禁止泛泛 "plain background"。
              例如 "urban gray camouflage fabric, no text"。
      negative_prompt: 默认排除 text/typography。
      strength=1.0: 对 masked 区域做完全重绘。
      num_inference_steps=50: 比默认 25 更彻底，减少旧字 ghost。
      guidance_scale=7.5: 默认即可。
      seed: 固定随机性。
    """
    pipe = _load_diff_pipe()
    img = src_pil.convert("RGB")
    mask = mask_pil.convert("L").resize(img.size)
    w, h = img.size
    p_img = pad_image(img)
    p_mask = pad_image(mask, is_mask=True)
    if p_mask.size != p_img.size:
        p_mask = p_mask.resize(p_img.size)
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=prompt or "clean background texture, no text",
        negative_prompt=negative_prompt,
        image=p_img, mask_image=p_mask,
        height=p_img.height, width=p_img.width,
        strength=strength, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, generator=g,
    ).images[0]
    if out.width > w or out.height > h:
        out = out.crop((0, 0, w, h))
    return out.convert("RGB")


# ----------------------------------------------------------------------------
# 遗留工具（OCR / arc / polygon 等）保留，供旧脚本兼容，但 base.py 不再用色块填充
# ----------------------------------------------------------------------------
def load_ocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)


def detect_text_polygons(src_pil, min_conf=0.25):
    reader = load_ocr()
    arr = np.array(src_pil)
    res = reader.readtext(arr, paragraph=False, detail=1)
    out = []
    for poly, text, conf in res:
        if conf < min_conf:
            continue
        poly = np.array(poly, dtype=np.int32)
        out.append({"poly": poly, "text": text, "conf": conf})
    return out


def build_polygon_mask(size, polys, dilate=10):
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    for p in polys:
        pts = [(int(x), int(y)) for x, y in p["poly"]]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    if dilate > 0:
        mask = mask.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    return mask


def draw_word(canvas, word, bbox, font_path=FONT_PATH, color=(255, 255, 255), max_size=None):
    d = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    size = int(bh * 0.80)
    if max_size:
        size = min(size, max_size)
    font = ImageFont.truetype(font_path, size)
    tb = d.textbbox((0, 0), word, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    if tw > bw * 0.96:
        size = int(size * (bw / tw) * 0.94)
        if max_size:
            size = min(size, max_size)
        font = ImageFont.truetype(font_path, size)
        tb = d.textbbox((0, 0), word, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    d.text((cx - tw // 2, cy - th // 2), word, font=font, fill=color)
    return canvas


def draw_arc_word(canvas, word, bbox, font_path=FONT_PATH, color=(40, 40, 40)):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    radius = 520
    cy = 820
    font_size = 86
    while font_size > 24:
        arc_len = arc_text.fit_arc_text_width(word, font_path, font_size, radius)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 108:
            break
        font_size = int(font_size * 0.92)
    return arc_text.draw_arc_text(canvas, word, font_path, font_size, color,
                                  (cx, cy), radius, 225, 315, char_spacing_px=1)


def main():
    ap = argparse.ArgumentParser().parse_args()
    src_path = SRC_ROOT / "6978fabda2cc99629fa9e81f802762d3.jpg"
    src = Image.open(src_path).convert("RGB")
    polys = detect_text_polygons(src)
    mask = build_polygon_mask(src.size, polys, dilate=14)
    cleaned = diffusion_inpaint(src, mask,
                                prompt="purple vintage badge background, distressed texture, dark violet and black, no text")
    cleaned.save(OUT / f"{src_path.stem}_cleaned.png")
    print(f"cleaned -> {OUT / f'{src_path.stem}_cleaned.png'}")


if __name__ == "__main__":
    main()
