"""
burn_v188_infringing.py — v188 输出侵权文字烧字（bat_logo / camo_armed）

v188 已让 SDXL "no readable text"，所以结果里本应无字；但 SDXL 偶尔仍画乱码。
策略：
1. 用原图 OCR 的文字位置（来源：v185 阶段 OCR）按 v188 输出尺寸缩放得到目标 box
2. 先 inpaint 擦除该 box 内任何残留像素（清掉可能的乱码）
3. 烧入不侵权替换词（1:1 近似位置，字体匹配原图风格）
4. OCR 复核烧字区是否读到新词

替换方案：
  bat_logo:  BACARDÍ -> BATANO (arc_top),  HEART -> SOUL (bottom)
  camo_armed: WE SUPPORT THE / ARMED / FORCES -> WE HONOR / THE / BRAVE
"""

import os, sys, numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

JOB = r"E:/Desktop/双接口/image-fission/jobs/smoke_v188"
SRC = r"E:/Desktop/双接口/image-fission/ComfyUI/input"
FONTS = r"E:/Desktop/双接口/image-fission/fonts"

# 原图文件名
SRC_FILE = {
    "bat_logo": "test_6978fabda2cc99629fa9e81f802762d3.jpg",
    "camo_armed": "test_b78e60de8dfdf44acda99395326a7298.jpg",
}

# 来源：v185 阶段对原图 OCR 得到的文字位置（src 原图像素坐标）
# bat_logo 原图 1552x2000；camo_armed 原图 1556x2000
SRC_BOXES = {
    "bat_logo": [
        # BACARDÍ 弧形顶部
        {"src_box": (281, 946, 1284, 1187), "word": "BATANO",
         "font": "MetalMania-Regular.ttf", "color": (235, 230, 220, 255)},
        # HEART 底部
        {"src_box": (463, 1172, 1142, 1345), "word": "SOUL",
         "font": "MetalMania-Regular.ttf", "color": (235, 230, 220, 255)},
    ],
    "camo_armed": [
        {"src_box": (520, 463, 1053, 538), "word": "WE HONOR",
         "font": "Rye-Regular.ttf", "color": (30, 28, 24, 255)},
        {"src_box": (464, 527, 1110, 649), "word": "THE",
         "font": "PirataOne-Regular.ttf", "color": (30, 28, 24, 255)},
        {"src_box": (419, 642, 1153, 774), "word": "BRAVE",
         "font": "PirataOne-Regular.ttf", "color": (30, 28, 24, 255)},
    ],
}


def inpaint_region(img_rgb, box, radius=6, pad=10):
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    x2, y2 = min(x2, w), min(y2, h)
    mask = np.zeros((h, w), np.uint8)
    mask[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 255
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def fit_font(text, font_path, max_w, max_h):
    lo, hi = 20, 700
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        if f.getlength(text) <= max_w and f.size * 1.05 <= max_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ImageFont.truetype(font_path, best)


def burn_text(img_rgb, box, text, font_name, color):
    img = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in box]
    max_w, max_h = x2 - x1, y2 - y1
    fp = os.path.join(FONTS, font_name)
    f = fit_font(text, fp, max_w * 0.95, max_h * 0.88)
    lw = f.getlength(text)
    lh = f.size * 1.0
    lx = x1 + (max_w - lw) / 2
    ly = y1 + (max_h - lh) / 2
    draw.text((lx, ly), text, font=f, fill=color)
    return np.array(img.convert("RGB"))


def burn_one(ref_id):
    gen_path = f"{JOB}/v188_{ref_id}.jpg"
    if not os.path.exists(gen_path):
        print(f"[skip] {ref_id}: 未找到 {gen_path}")
        return None
    out_path = f"{JOB}/v188_{ref_id}_burned.jpg"

    img_rgb = np.array(Image.open(gen_path).convert("RGB"))
    gh, gw = img_rgb.shape[:2]

    # 原图尺寸（用于缩放）
    src_img = Image.open(os.path.join(SRC, SRC_FILE[ref_id]))
    sh, sw = src_img.height, src_img.width
    sx, sy = gw / sw, gh / sh

    print(f"=== {ref_id}  ({gw}x{gh}, 原图 {sw}x{sh}, 缩放 {sx:.2f}/{sy:.2f}) ===")
    for item in SRC_BOXES[ref_id]:
        sb = item["src_box"]
        box = [int(sb[0] * sx), int(sb[1] * sy), int(sb[2] * sx), int(sb[3] * sy)]
        print(f"  [burn] '{item['word']}' box={box}")
        img_rgb = inpaint_region(img_rgb, box, radius=6, pad=12)
        img_rgb = burn_text(img_rgb, box, item["word"], item["font"], item["color"])

    Image.fromarray(img_rgb).save(out_path, quality=95)
    print(f"  saved -> {out_path} ({os.path.getsize(out_path)//1024} KB)\n")
    return out_path


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["bat_logo", "camo_armed"]
    for t in targets:
        if t in SRC_BOXES:
            burn_one(t)
        else:
            print(f"[skip] {t}: 不在烧字列表")
