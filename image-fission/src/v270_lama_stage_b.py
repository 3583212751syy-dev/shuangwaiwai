"""v270: 本地 Big-LaMa 真正去字（不走 ComfyUI 中文路径节点），作为 Stage B。"""
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# 复用 v268 的 arc_text 如果要用，但 Stage B 只做去字
sys.path.insert(0, str(Path(__file__).parent))

PROJECT = Path("E:/Desktop/双接口/image-fission")
MODEL_PATH = PROJECT / "ComfyUI" / "models" / "RMBG" / "Lama" / "big-lama.pt"
SRC = PROJECT / "ComfyUI" / "input" / "6978fabda2cc99629fa9e81f802762d3.jpg"
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
    if w % 8 != 0:
        w += 8 - w % 8
    if h % 8 != 0:
        h += 8 - h % 8
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
        p_mask = p_mask.resize(p_img.size, Image.LANCZOS)
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


def make_full_mask(size):
    w, h = size
    scale_x = w / 1552
    scale_y = h / 2000
    def sc(pts): return [(int(x * scale_x), int(y * scale_y)) for x, y in pts]
    def s(x, y): return (int(x * scale_x), int(y * scale_y))

    outer = [
        (120, 380), (120, 450), (140, 600), (180, 750), (250, 880),
        (450, 950), (700, 980), (850, 980), (1100, 950), (1300, 880),
        (1370, 750), (1410, 600), (1430, 450), (1430, 380),
        (1330, 310), (1100, 270), (776, 255), (450, 270), (220, 310)
    ]
    inner = [
        (600, 400), (776, 400), (950, 400),
        (1100, 520), (1100, 720), (1000, 800),
        (900, 850), (776, 900), (650, 850),
        (550, 800), (400, 720), (400, 520)
    ]
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.polygon(sc(outer) + sc(inner)[::-1], fill=255)
    d.rectangle([s(160, 980), s(1390, 1170)], fill=255)
    d.rectangle([s(360, 1180), s(1195, 1340)], fill=255)
    d.rectangle([s(350, 740), s(560, 860)], fill=255)
    d.rectangle([s(1000, 740), s(1220, 860)], fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(11))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=4))
    return mask


def main():
    print("[v270] Stage B: Big-LaMa 去字")
    src = Image.open(SRC).convert("RGB")
    print(f"  source {src.size}")
    mask = make_full_mask(src.size)
    mask.save(OUT / "mask.png")
    print("  mask saved")
    cleaned = lama_inpaint(src, mask, removal_strength=235, edge_smoothness=4)
    cleaned.save(OUT / "cleaned.png", quality=98)
    print(f"  cleaned -> {OUT / 'cleaned.png'}")

    # 1024x1280 版给后续 ComfyUI Stage A
    clean_small = cleaned.resize((1024, 1280), Image.LANCZOS)
    clean_small.save(OUT / "cleaned_1024.png", quality=98)
    clean_small.save(PROJECT / "ComfyUI" / "input" / "v270_clean_1024.png", quality=98)
    print("  cleaned_1024 -> ComfyUI/input/v270_clean_1024.png")


if __name__ == "__main__":
    main()
