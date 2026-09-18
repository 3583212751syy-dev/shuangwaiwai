"""v319h: SDXL inpaint (Proteus v0.4) for ARMED text removal.

LaMa failed because its Fourier conv is designed for natural scenes (faces/landscapes),
not 'hard-edge ultra-bold text on periodic camo'. The conv always leaves ~2px haze
or anti-alias ghost at mask boundary.

SDXL inpaint uses diffusion, which can generate natural camo texture in the mask
without haze bands. v317c failed because denoise=1.0 made SDXL paint over the
entire image with new colors.

v319h settings:
  - Model: Proteus v0.4 (ComfyUI/models/checkpoints)
  - VAE: same
  - Inpaint model: Fooocus SDXL inpaint unet (already downloaded)
  - Denoise: 0.65 (preserves color palette, fills text with natural camo)
  - Prompt: "high-resolution dark military camouflage pattern, gray and black
             triangular shapes, no text, no symbols, no letters"
  - Negative prompt: "text, letters, words, writing, logo, brand, smooth, gradient"
  - Mask: letter CC & gray<160 (same as v319g)
  - Grow mask: 8 pixels (to ensure anti-alias is covered)
"""
import cv2
import numpy as np
import shutil
import tempfile
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import json
import urllib.request
import time

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = str(PROJECT / "jobs" / "v315" / "source_armed.jpg")
OUT_DIR = str(PROJECT / "jobs" / "v319h")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"
COMFY_URL = "http://127.0.0.1:8188"


def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def find_fsize_for_capH(target_capH, font_path):
    for fs in range(60, 250):
        font = ImageFont.truetype(font_path, fs)
        h_bbox = font.getbbox('H')
        capH = h_bbox[3] - h_bbox[1]
        if capH >= target_capH:
            return fs, font
    return 200, ImageFont.truetype(font_path, 200)


BANDS_DEF = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'WE HONOR', 'orig_cx': 783, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE',   'orig_cx': 829, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION',  'orig_cx': 863, 'orig_w': 917},
]

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Step 1: text vs chain classification (v316e/v319g proven)
chain_mask = np.zeros((H, W), dtype=np.uint8)
letter_mask = np.zeros((H, W), dtype=bool)
for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        is_letter = (h >= band_h * 0.35)
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_letter:
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Step 1b: dog tag detection
dog_tag_mask = np.zeros((H, W), dtype=np.uint8)
for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_gray = gray[y0:y1, :]
    med_dark = ((band_gray >= 80) & (band_gray < 180)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(med_dark, connectivity=8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if 8000 < a < 30000 and h > 80 and 0.5 < w/h < 1.5:
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y_loc = stats[i, cv2.CC_STAT_TOP]
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            dog_tag_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255
            print(f'[v319h] {band["name"]}: dog-tag CC area={a} w={w} h={h}')

# Step 2: text mask
text_mask = letter_mask & (gray < 160) & (chain_mask == 0) & (dog_tag_mask == 0)
print(f'[v319h] text mask px: {int(text_mask.sum())}')

# Grow mask by 8px (SDXL inpaint needs margin)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
text_mask_grown = cv2.dilate(text_mask.astype(np.uint8) * 255, k, iterations=1) > 0
print(f'[v319h] text mask grown px: {int(text_mask_grown.sum())}')

# ComfyUI reads images from its input/ folder only
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_INPUT.mkdir(parents=True, exist_ok=True)
SRC_NAME = "v319h_source.png"
MASK_NAME = "v319h_mask.png"
Image.fromarray((text_mask_grown.astype(np.uint8) * 255)).save(str(COMFY_INPUT / MASK_NAME))
Image.fromarray(img_rgb).save(str(COMFY_INPUT / SRC_NAME))

# Also keep local copies for reference
Image.fromarray((text_mask_grown.astype(np.uint8) * 255)).save(str(Path(OUT_DIR) / 'mask.png'))
src_pil = Image.fromarray(img_rgb)
src_pil.save(str(Path(OUT_DIR) / 'source.png'))

# Find inpaint checkpoint
import urllib.request as ur
ckpts_data = ur.urlopen(f'{COMFY_URL}/object_info/CheckpointLoaderSimple', timeout=10).read()
ckpts = json.loads(ckpts_data)['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0]
print(f'[v319h] Available checkpoints: {ckpts[:5]}')
ckpt = 'Proteus_v0.4.safetensors' if 'Proteus_v0.4.safetensors' in ckpts else ckpts[0]

# Inpaint model
ip_data = ur.urlopen(f'{COMFY_URL}/object_info/INPAINT_LoadInpaintModel', timeout=10).read()
ip_models = json.loads(ip_data)['INPAINT_LoadInpaintModel']['input']['required']['model_name'][0]
print(f'[v319h] Available inpaint models: {ip_models}')
inpaint_model = ip_models[0] if ip_models else None

# Build workflow — FORCE SDXL KSampler inpaint branch (diffusion, not LaMa/Fooocus).
# LaMa Fourier conv leaves haze bands on hard-edge text + camo; SDXL diffusion
# generates natural camo in the masked region without that artifact.
prompt_text = "high-resolution dark military camouflage pattern, gray black triangular shapes, no text no letters no symbols"
neg_text = "text, letters, words, writing, logo, brand, smooth, gradient, watermark, blurry"

# SDXL KSampler inpaint
workflow = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
    "2": {"class_type": "LoadImage", "inputs": {"image": SRC_NAME}},
    "3": {"class_type": "LoadImageMask", "inputs": {"image": MASK_NAME, "channel": "red"}},
    "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
    "5": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["4", 0], "mask": ["3", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["1", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_text, "clip": ["1", 1]}},
    "8": {"class_type": "KSampler",
          "inputs": {
              "model": ["1", 0],
              "positive": ["6", 0],
              "negative": ["7", 0],
              "latent_image": ["5", 0],
              "seed": 42,
              "steps": 30,
              "cfg": 7,
              "sampler_name": "euler_ancestral",
              "scheduler": "normal",
              "denoise": 0.65,
          }},
    "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
    "10": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
}

# Submit workflow
req = ur.Request(f'{COMFY_URL}/prompt', data=json.dumps({"prompt": workflow}).encode(),
                 headers={'Content-Type': 'application/json'})
try:
    resp = ur.urlopen(req, timeout=10).read().decode()
    prompt_id = json.loads(resp)['prompt_id']
    print(f'[v319h] Submitted prompt_id={prompt_id}')
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f'[v319h] HTTP {e.code}: {body}')
    raise

# Poll history
for _ in range(120):
    time.sleep(2)
    hist = json.loads(ur.urlopen(f'{COMFY_URL}/history/{prompt_id}', timeout=10).read().decode())
    if prompt_id in hist and hist[prompt_id].get('outputs'):
        outputs = hist[prompt_id]['outputs']
        # Find any preview/save output
        for node_id, out in outputs.items():
            if 'images' in out:
                for img in out['images']:
                    filename = img['filename']
                    subfolder = img.get('subfolder', '')
                    url = f'{COMFY_URL}/view?filename={filename}&subfolder={subfolder}'
                    data = ur.urlopen(url, timeout=30).read()
                    out_path = Path(OUT_DIR) / filename
                    out_path.write_bytes(data)
                    print(f'[v319h] Got {filename} -> {out_path}')
                break
        break
else:
    print(f'[v319h] Timed out waiting for prompt {prompt_id}')

# Compose final: copy-back chain/dog tag from source
cleaned_path = next(Path(OUT_DIR).glob('*.png'), None)
if cleaned_path is None:
    cleaned_path = next(Path(OUT_DIR).glob('ComfyUI_*.png'), None)
if cleaned_path and cleaned_path.exists():
    cleaned_rgb = np.array(Image.open(cleaned_path).convert('RGB'))
    if cleaned_rgb.shape != img_rgb.shape:
        cleaned_rgb = np.array(Image.open(cleaned_path).convert('RGB').resize((W, H)))
    copy_back = (chain_mask > 0) | (dog_tag_mask > 0)
    cleaned_rgb[copy_back] = img_rgb[copy_back]
    Image.fromarray(cleaned_rgb).save(str(Path(OUT_DIR) / 'armed_sdxl_cleaned.png'))
    cleaned_pil = Image.fromarray(cleaned_rgb)
else:
    print('[v319h] No cleaned image produced')
    cleaned_pil = Image.fromarray(img_rgb)

# Step 4: write new text
out_pil = cleaned_pil.copy()
draw = ImageDraw.Draw(out_pil)
BANDS = []
for b in BANDS_DEF:
    band_h = b['y1'] - b['y0']
    fsize, font = find_fsize_for_capH(band_h, ARIAL_BLACK)
    h_bbox = font.getbbox('H')
    capH = h_bbox[3] - h_bbox[1]
    text = b['new_text']
    bbox = font.getbbox(text)
    nat_w = bbox[2] - bbox[0]
    desired_w = b['orig_w']
    spacing = max(0, (desired_w - nat_w) // max(len(text) - 1, 1))
    total_w = nat_w + spacing * (len(text) - 1)
    paste_x = b['orig_cx'] - total_w // 2 - bbox[0]
    paste_y = b['y1'] - bbox[3]
    BANDS.append({**b, 'fsize': fsize, 'font': font, 'bbox': bbox, 'nat_w': nat_w,
                  'spacing': spacing, 'total_w': total_w,
                  'paste_x': paste_x, 'paste_y': paste_y, 'capH': capH})
    print(f'[v319h] {b["name"]}: capH={capH} fsize={fsize} "{text}" paste=({paste_x},{paste_y})')

for band in BANDS:
    draw.text((band['paste_x'], band['paste_y']),
              band['new_text'], font=band['font'], anchor='lt',
              fill=(20, 20, 20), spacing=band['spacing'])

OUT = Path(OUT_DIR) / 'armed_text_only_v319h.jpg'
out_pil.save(str(OUT), quality=95)

src_compare = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_compare.width * 2 + 30, src_compare.height), (50, 50, 50))
combined.paste(src_compare, (0, 0))
combined.paste(out_pil, (src_compare.width + 30, 0))
combined.save(str(Path(OUT_DIR) / 'armed_compare_v319h.jpg'), quality=90)

src_crop = src_compare.crop((0, 440, 1556, 780))
res_crop = out_pil.crop((0, 440, 1556, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(str(Path(OUT_DIR) / 'armed_text_band_compare_v319h.jpg'), quality=95)
print(f'[v319h] Saved {OUT}')

# Clean up temp files in ComfyUI input/
for tmp in [SRC_NAME, MASK_NAME]:
    p = COMFY_INPUT / tmp
    if p.exists():
        p.unlink()
        print(f'[v319h] Removed temp {tmp}')