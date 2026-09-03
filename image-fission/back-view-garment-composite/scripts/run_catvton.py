"""
CatVTON git-model try-on: produce a back-view image whose garment shows the
FRONT print, by repainting the back person's clothing region with the front
garment (LIMITLESS print). Uses the official zhengchong/CatVTON pipeline.

Run from the image-fission workspace root:
  catvton_venv/Scripts/python.exe run_catvton.py
"""
import os, sys, cv2, numpy as np
from PIL import Image

# ---- paths (raw Windows strings; never /d/ style) ----
ROOT      = r'D:\.workbuddy\2026-08-16-00-13-40\image-fission'
CATVTON   = os.path.join(ROOT, 'CatVTON')
HF_HOME   = r'E:\AI\Cache\huggingface'
os.environ['HF_HOME'] = HF_HOME
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HUB_ENABLE_HF_XET'] = '1'

BACK_IMG  = os.path.join(ROOT, 'ComfyUI', 'input', 'back_reference_real.png')
FRONT_IMG = os.path.join(ROOT, 'ComfyUI', 'input', 'front_model.jpg')

SCHP_LIP  = r'D:\claude-tools\cache\huggingface\models--zhengchong--CatVTON\snapshots\main\SCHP\exp-schp-201908261155-lip.pth'
SCHP_ATR  = r'D:\claude-tools\cache\huggingface\models--zhengchong--CatVTON\snapshots\main\SCHP\exp-schp-201908301523-atr.pth'
ATTN_CKPT = r'D:\claude-tools\cache\huggingface\models--zhengchong--CatVTON-MaskFree\snapshots\main'  # has mix-48k-1024/attention/model.safetensors

# Locally-downloaded SD1.5-inpainting base (ModelScope mirror) -> snapshot dir
import glob as _glob
_cands = _glob.glob(os.path.join(ROOT, 'CatVTON', 'models', 'sd_inpainting', 'models', 'AI-ModelScope--stable-diffusion-inpainting', 'snapshots', '*'))
BASE_CKPT = _cands[0] if _cands else 'booksforcharlie/stable-diffusion-inpainting'
os.environ['CATVTON_VAE'] = os.path.join(BASE_CKPT, 'vae')
print('BASE_CKPT =', BASE_CKPT)

# ---- make CatVTON importable ----
os.chdir(CATVTON)
sys.path.insert(0, CATVTON)

import torch
from model.SCHP import SCHP
from model.pipeline import CatVTONPipeline

# ---- SCHP parsing maps (copied from CatVTON cloth_masker) ----
LIP_MAPPING = {'Background':0,'Hat':1,'Hair':2,'Glove':3,'Sunglasses':4,'Upper-clothes':5,'Dress':6,'Coat':7,'Socks':8,'Pants':9,'Jumpsuits':10,'Scarf':11,'Skirt':12,'Face':13,'Left-arm':14,'Right-arm':15,'Left-leg':16,'Right-leg':17,'Left-shoe':18,'Right-shoe':19}
ATR_MAPPING = {'Background':0,'Hat':1,'Hair':2,'Sunglasses':3,'Upper-clothes':4,'Skirt':5,'Pants':6,'Dress':7,'Belt':8,'Left-shoe':9,'Right-shoe':10,'Face':11,'Left-leg':12,'Right-leg':13,'Left-arm':14,'Right-arm':15,'Bag':16,'Scarf':17}
# Note: ATR has no 'Coat'/'Jumpsuits' — only look up keys that exist in each map.
CLOTH = ['Upper-clothes','Coat','Dress','Jumpsuits','Pants','Skirt']

def _cloth_union(pl, pa):
    m = np.zeros(pl.shape[:2], np.uint8)
    for c in CLOTH:
        if c in LIP_MAPPING:
            m |= (pl == LIP_MAPPING[c])
        if c in ATR_MAPPING:
            m |= (pa == ATR_MAPPING[c])
    return m

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device, '| cuda', torch.cuda.is_available())

# ---------------------------------------------------------------- mask
def clothing_mask(schp_lip, schp_atr, pil_img):
    pl = np.array(schp_lip(pil_img))
    pa = np.array(schp_atr(pil_img))
    m = _cloth_union(pl, pa)
    k = max(3, int(max(m.shape) // 250) | 1)
    m = cv2.dilate(m, np.ones((k, k), np.uint8), 1)
    gk = k * 3 + 1 if (k * 3) % 2 == 0 else k * 3
    m = cv2.GaussianBlur((m * 255).astype(np.uint8), (gk, gk), 0)
    m = (m > 25).astype(np.uint8) * 255
    return Image.fromarray(m)

# ---------------------------------------------------------------- garment condition (front crop)
def garment_condition(schp_lip, schp_atr, front_pil):
    arr = np.array(front_pil.convert('RGB'))
    pl = np.array(schp_lip(front_pil))
    pa = np.array(schp_atr(front_pil))
    m = _cloth_union(pl, pa)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return front_pil
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 20
    x0, x1 = max(0, x0 - pad), min(arr.shape[1], x1 + pad)
    y0, y1 = max(0, y0 - pad), min(arr.shape[0], y1 + pad)
    crop = arr[y0:y1, x0:x1]
    return Image.fromarray(crop).convert('RGB')

# ---------------------------------------------------------------- pad back to 3:4 portrait
def pad_to_portrait(pil_img, fill=(142, 168, 188)):
    w, h = pil_img.size
    target_w = int(round(h * 0.75))
    if target_w <= w:
        return pil_img
    left = (target_w - w) // 2
    right = target_w - w - left
    return Image.fromarray(cv2.copyMakeBorder(np.array(pil_img), 0, 0, left, right, cv2.BORDER_CONSTANT, value=fill))

# ---------------------------------------------------------------- figure composite (GrabCut) onto beige
def grabcut_figure(bgr):
    H, W = bgr.shape[:2]
    rx0, ry0 = int(W * 0.12), int(H * 0.02)
    rw, rh = int(W * 0.76), int(H * 0.96)
    mask = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (rx0, ry0, rw, rh), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = 5
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8), iterations=1)
    return fg

def composite_on_beige(pil_result, out_path, canvas=(1350, 1800), target_bg=(142, 168, 188)):
    bgr = cv2.cvtColor(np.array(pil_result.convert('RGB')), cv2.COLOR_RGB2BGR)
    fg = grabcut_figure(bgr)
    ys, xs = np.where(fg > 0)
    if len(xs) == 0:
        cv2.imwrite(out_path, bgr); return
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = bgr[y0:y1, x0:x1]
    cffg = fg[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = min(canvas[0] / cw, canvas[1] / ch)
    nw, nh = int(cw * scale), int(ch * scale)
    crop = cv2.resize(crop, (nw, nh), cv2.INTER_LANCZOS4)
    cffg = cv2.resize(cffg, (nw, nh), cv2.INTER_NEAREST)
    canvas_img = np.full((canvas[1], canvas[0], 3), target_bg, np.uint8)
    ox = (canvas[0] - nw) // 2
    oy = (canvas[1] - nh) // 2
    soft = cv2.GaussianBlur(cffg, (5, 5), 0).astype(np.float32) / 255.0
    soft = np.stack([soft] * 3, -1)
    roi = canvas_img[oy:oy+nh, ox:ox+nw].astype(np.float32)
    roi = roi * (1 - soft) + crop.astype(np.float32) * soft
    canvas_img[oy:oy+nh, ox:ox+nw] = roi.astype(np.uint8)
    cv2.imwrite(out_path, canvas_img)

# ================================================================ main
def main():
    schp_lip = SCHP(ckpt_path=SCHP_LIP, device=device)
    schp_atr = SCHP(ckpt_path=SCHP_ATR, device=device)
    print('SCHP loaded')

    back = Image.open(BACK_IMG).convert('RGB')
    front = Image.open(FRONT_IMG).convert('RGB')

    back_pad = pad_to_portrait(back)
    mask = clothing_mask(schp_lip, schp_atr, back_pad)
    cond = garment_condition(schp_lip, schp_atr, front)

    back_pad.save(os.path.join(ROOT, 'catvton_person_pad.png'))
    mask.save(os.path.join(ROOT, 'catvton_mask.png'))
    cond.save(os.path.join(ROOT, 'catvton_garment.png'))
    print('person/mask/garment saved')

    print('loading CatVTONPipeline (base=%s, attn=%s) ...' % (BASE_CKPT, ATTN_CKPT))
    pipe = CatVTONPipeline(
        base_ckpt=BASE_CKPT, attn_ckpt=ATTN_CKPT, attn_ckpt_version='mix',
        weight_dtype=torch.float16, device=device, skip_safety_check=True, use_tf32=True,
    )
    print('pipeline ready, running try-on ...')
    out = pipe(
        image=back_pad, condition_image=cond, mask=mask,
        num_inference_steps=50, guidance_scale=2.5, height=1024, width=768,
    )
    raw = out[0]
    raw.save(os.path.join(ROOT, 'catvton_raw.png'))
    print('raw try-on saved', raw.size)

    composite_on_beige(raw, os.path.join(ROOT, 'catvton_back_1350x1800_beige.png'))
    print('DONE_CATVTON')

if __name__ == '__main__':
    main()
