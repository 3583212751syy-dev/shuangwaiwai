"""测试: 用 LaMa 改进 v253.clean_base 去字效果."""
import sys
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location("v253", str(Path(__file__).parent / "v253_bat_logo_inpaint.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

PROJECT = Path("E:/Desktop/双接口/image-fission")
MODEL_PATH = PROJECT / "ComfyUI" / "models" / "RMBG" / "Lama" / "big-lama.pt"
OUT = PROJECT / "jobs" / "v270"
OUT.mkdir(parents=True, exist_ok=True)

_LAMA = None

def load_lama():
    global _LAMA
    if _LAMA is not None:
        return _LAMA
    src = str(MODEL_PATH)
    if any(ord(c) > 127 for c in src):
        tmp = Path(tempfile.gettempdir()) / "big-lama.pt"
        if not tmp.exists() or tmp.stat().st_size != Path(src).stat().st_size:
            shutil.copyfile(src, tmp)
        src = str(tmp)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(src, map_location=dev)
    model.eval().to(dev)
    _LAMA = (model, dev)
    return _LAMA

def pad_image(image, is_mask=False):
    w, h = image.size
    if w % 8 != 0: w += 8 - w % 8
    if h % 8 != 0: h += 8 - h % 8
    fill = 0 if is_mask else None
    padded = Image.new(image.mode, (w, h), color=fill)
    padded.paste(image, (0, 0))
    return padded

def lama_inpaint(src_pil, mask_pil, removal_strength=230, edge_smoothness=4):
    model, dev = load_lama()
    w, h = src_pil.size
    p_img = pad_image(src_pil)
    p_mask = pad_image(mask_pil, is_mask=True)
    if p_mask.size != p_img.size:
        p_mask = p_mask.resize(p_img.size, ImageFilter.LANCZOS)
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

def clean_base_lama(orig_rgb, bgr):
    """v253 clean_base 但把 cv2.inpaint 换成 LaMa."""
    h, w = bgr.shape[:2]
    bat_mask = m.extract_bat(bgr)
    BG = m.sample_bg(bgr)
    # 先擦蝙蝠和字区域 (用原图背景色预填)
    base_np = np.array(orig_rgb).copy()
    base_np[bat_mask] = BG
    # 文字 mask: 暗像素 + 不在蝙蝠区 + 不在圆环区
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.ogrid[:h, :w]
    dd = ((xs - m.RING['cx']) ** 2 + (ys - m.RING['cy']) ** 2) ** 0.5
    ring = (dd >= m.RING['inner_r'] - 6) & (dd <= m.RING['outer_r'] + 6)
    dark_w = gray < 140
    cand = dark_w & (~bat_mask) & (~ring)
    bands = m.find_text_bands(cand, min_h=18, min_px=800, gap=45)
    erase = np.zeros((h, w), bool)
    for a, b in bands:
        erase[a:b + 1, :] |= cand[a:b + 1, :]
    if erase.any():
        # 膨胀 mask 确保覆盖笔画边缘
        erase_u8 = erase.astype(np.uint8) * 255
        kernel = np.ones((25, 25), np.uint8)
        erase_u8 = cv2.dilate(erase_u8, kernel, iterations=1)
        mask = Image.fromarray(erase_u8, "L")
        base = Image.fromarray(base_np)
        base = lama_inpaint(base, mask, removal_strength=235, edge_smoothness=4)
        base_np = np.array(base)
    return Image.fromarray(base_np), bat_mask

def main():
    src = Image.open(PROJECT / "ComfyUI" / "input" / "6978fabda2cc99629fa9e81f802762d3.jpg").convert("RGB")
    bgr = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
    base, bat_mask = clean_base_lama(src, bgr)
    base.save(OUT / "clean_base_lama.png", quality=98)
    print(f"saved {OUT / 'clean_base_lama.png'}")

if __name__ == "__main__":
    main()
