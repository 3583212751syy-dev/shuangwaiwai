"""
v317b: ARMED text-only fission via Fooocus Inpaint + SDXL KSampler.

Pipeline:
  1. Detect text bands in source (v316e_lib)
  2. Upload source_clean.png + text_mask.png to ComfyUI
  3. Submit workflow: FooocusInpaint patch + JuggernautXL SDXL + KSampler(denoise=1.0)
     + VAEDecode + PreviewImage
  4. Download inpainted image
  5. Composite onto source (chain/dog tag untouched)
  6. Write new text with Anton font
"""
import os
import sys
import json
import time
import urllib.request
import urllib.parse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r'E:/Desktop/双接口/image-fission/src')
from v316e_lib import prepare_text_mask, BANDS, load_jpg

SRC = r'E:/Desktop/双接口/image-fission/jobs/v315/source_armed.jpg'
OUT_DIR = r'E:/Desktop/双接口/image-fission/jobs/v317b'
os.makedirs(OUT_DIR, exist_ok=True)
ANTON_PATH = r'E:/Desktop/双接口/image-fission/ComfyUI/models/fonts/Anton-Regular.ttf'

img_rgb, fill_mask, H, W = prepare_text_mask(SRC)
print(f'[v317b] Source: {W}x{H}, fill_mask coverage: {fill_mask.sum()/fill_mask.size*100:.2f}%')

img_pil = Image.fromarray(img_rgb)
img_pil.save(os.path.join(OUT_DIR, 'source_clean.png'))
mask_uint8 = (fill_mask * 255).astype(np.uint8)
mask_pil = Image.fromarray(mask_uint8, mode='L')
mask_pil.save(os.path.join(OUT_DIR, 'text_mask.png'))

def upload_to_comfyui(filepath):
    boundary = '----FormBoundary' + ''.join([str(np.random.randint(0, 10)) for _ in range(16)])
    with open(filepath, 'rb') as f:
        data = f.read()
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="image"; filename="{os.path.basename(filepath)}"\r\n'
        f'Content-Type: image/png\r\n\r\n'
    ).encode() + data + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(
        'http://127.0.0.1:8188/upload/image',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())['name']

print('[v317b] Uploading to ComfyUI...')
img_name = upload_to_comfyui(os.path.join(OUT_DIR, 'source_clean.png'))
mask_name = upload_to_comfyui(os.path.join(OUT_DIR, 'text_mask.png'))
print(f'[v317b] Uploaded: img={img_name}, mask={mask_name}')

# Fooocus Inpaint + SDXL + KSampler(denoise=1.0) workflow
prompt = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"},
    },
    "2": {
        "class_type": "INPAINT_LoadFooocusInpaint",
        "inputs": {
            "head": "diffusion_pytorch_model.safetensors",
            "patch": "diffusion_pytorch_model.safetensors",
        },
    },
    "3": {
        "class_type": "LoadImage",
        "inputs": {"image": img_name},
    },
    "4": {
        "class_type": "LoadImage",
        "inputs": {"image": mask_name},
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["1", 1],
            "text": "army camouflage pattern, desert tan brown, military texture, no text, no letters, no numbers, no watermark, no logo",
        },
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["1", 1],
            "text": "text, watermark, blurry, low quality, distorted, ugly, deformed",
        },
    },
    "7": {
        "class_type": "VAEEncodeForInpaint",
        "inputs": {
            "pixels": ["3", 0],
            "vae": ["1", 2],
            "mask": ["4", 1],
            "grow_mask_by": 6,
        },
    },
    "8": {
        "class_type": "INPAINT_ApplyFooocusInpaint",
        "inputs": {
            "model": ["1", 0],
            "patch": ["2", 0],
            "latent": ["7", 0],
        },
    },
    "9": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["8", 0],
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["7", 0],
            "seed": 42,
            "steps": 24,
            "cfg": 6.0,
            "sampler_name": "dpmpp_2m_sde_gpu",
            "scheduler": "karras",
            "denoise": 1.0,
        },
    },
    "10": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["9", 0],
            "vae": ["1", 2],
        },
    },
    "11": {
        "class_type": "PreviewImage",
        "inputs": {"images": ["10", 0]},
    },
}

print('[v317b] Submitting workflow...')
prompt_req = urllib.request.Request(
    'http://127.0.0.1:8188/prompt',
    data=json.dumps({'prompt': prompt}).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(prompt_req, timeout=30) as resp:
    result = json.loads(resp.read())
    if 'error' in result:
        print(f'[v317b] ERROR: {json.dumps(result, indent=2)}')
        sys.exit(1)
    prompt_id = result['prompt_id']
print(f'[v317b] prompt_id={prompt_id}')

print('[v317b] Waiting for completion...')
history = None
for attempt in range(180):
    time.sleep(3)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:8188/history/{prompt_id}', timeout=10) as resp:
            history = json.loads(resp.read())
        if prompt_id in history:
            break
    except Exception:
        pass
else:
    print('[v317b] TIMEOUT after 9 min')
    sys.exit(1)

# Check status
h = history[prompt_id]
status_msgs = h.get('status', {}).get('messages', []) if isinstance(h.get('status'), dict) else []
has_error = any('execution_error' in str(m) for m in status_msgs)
print(f'[v317b] Has error: {has_error}')

outputs = h.get('outputs', {})
inpainted_filename = None
subfolder = ''
img_type = 'output'
for node_id, node_out in outputs.items():
    if 'images' in node_out:
        for img_info in node_out['images']:
            inpainted_filename = img_info['filename']
            subfolder = img_info.get('subfolder', '')
            img_type = img_info.get('type', 'output')
            break
    if inpainted_filename:
        break

if not inpainted_filename:
    print(f'[v317b] No output. Status: {json.dumps(h.get("status", {}), indent=2)[:500]}')
    sys.exit(1)

params = urllib.parse.urlencode({'filename': inpainted_filename, 'subfolder': subfolder, 'type': img_type})
url = f'http://127.0.0.1:8188/view?{params}'
with urllib.request.urlopen(url, timeout=30) as resp:
    inpaint_bytes = resp.read()
inpainted_path = os.path.join(OUT_DIR, 'inpainted_only.png')
with open(inpainted_path, 'wb') as f:
    f.write(inpaint_bytes)
print(f'[v317b] Downloaded: {inpainted_path}')

inpainted = np.array(Image.open(inpainted_path).convert('RGB'))
source_arr = np.array(img_pil)
if inpainted.shape != source_arr.shape:
    inpainted_pil = Image.open(inpainted_path).convert('RGB').resize((source_arr.shape[1], source_arr.shape[0]), Image.LANCZOS)
    inpainted = np.array(inpainted_pil)
    print(f'[v317b] Resized to {source_arr.shape[1]}x{source_arr.shape[0]}')

result = source_arr.copy()
result[fill_mask > 0] = inpainted[fill_mask > 0]
after_path = os.path.join(OUT_DIR, 'after_inpaint.png')
Image.fromarray(result).save(after_path)

# Write new text
out_pil = Image.open(after_path).convert('RGB')
draw = ImageDraw.Draw(out_pil)
tmp_im = Image.new('RGB', (50, 50))
tmp_draw = ImageDraw.Draw(tmp_im)

for band in BANDS:
    text = band['new_text']
    target_h = band['y1'] - band['y0']
    fsize = 50
    for fs in range(50, 250):
        font = ImageFont.truetype(ANTON_PATH, fs)
        bbox = tmp_draw.textbbox((0, 0), text, font=font, anchor='lt')
        if (bbox[3] - bbox[1]) >= target_h * 0.95:
            fsize = fs
            break
    font = ImageFont.truetype(ANTON_PATH, fsize)
    bbox = tmp_draw.textbbox((0, 0), text, font=font, anchor='lt')
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    y_center = (band['y0'] + band['y1']) // 2
    x_center = W // 2
    x = x_center - tw // 2 - bbox[0]
    y = y_center - th // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(20, 20, 20))
    print(f'[v317b] {band["name"]}: text="{text}" font={fsize}px pos=({x},{y})')

final_path = os.path.join(OUT_DIR, 'armed_text_only_v317b.jpg')
out_pil.save(final_path, quality=95)
print(f'[v317b] FINAL: {final_path}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v317b.jpg'), quality=90)
print('[v317b] Done')