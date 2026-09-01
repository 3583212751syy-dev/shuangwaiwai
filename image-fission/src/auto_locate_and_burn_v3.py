"""
auto_locate_and_burn_v3.py — v186 设计稿专用管线的烧字后处理
对 v186 重做的 bat_logo 和 camo_armed 跑 OCR → 自动定位文字 → inpaint 擦除 → 烧字

参数：
- src: v186 输出（已无侵权词，但 SDXL 仍可能画出 BACARDI 残字）
- 词替换字典：侵权词 → 不侵权词
- 字体库：根据原图风格选字体
"""

import os, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import easyocr

# ====== 全局配置 ======
JOB = r"E:/Desktop/双接口/image-fission/jobs/smoke_v186"
FONTS = r"E:/Desktop/双接口/image-fission/fonts"

# 词替换字典（原词 → 不侵权新词）
# 注意：烧字前 OCR 找到的文字很可能就是原词（SDXL 几乎照搬了原图文字）
WORD_REPLACE = {
    "bat_logo": {
        "BACARDI": ("BATANO", "MetalMania-Regular.ttf", (235, 230, 220, 255), "arc_top"),
        "BACARDÍ": ("BATANO", "MetalMania-Regular.ttf", (235, 230, 220, 255), "arc_top"),
        "HEART":  ("SOUL",   "MetalMania-Regular.ttf", (235, 230, 220, 255), "bottom"),
        "MHEART": ("MSOUL",  "MetalMania-Regular.ttf", (235, 230, 220, 255), "bottom"),
        # 任何含 BACAR/BACARDI 字母的 OCR 误读
    },
    "camo_armed": {
        "WE SUPPORT THE": ("WE HONOR",   "Rye-Regular.ttf",        (30, 28, 24, 255), "row1"),
        "ARMED":          ("THE",        "PirataOne-Regular.ttf",  (30, 28, 24, 255), "row2"),
        "FORCES":         ("BRAVE",      "PirataOne-Regular.ttf",  (30, 28, 24, 255), "row3"),
        # 整行 OCR 可能合并，需要按 y 位置分行处理
    },
}

OCR = easyocr.Reader(["en"], gpu=True, verbose=False)


def load_rgb(p): return np.array(Image.open(p).convert("RGB"))


def inpaint_region(img_rgb, box, radius=5):
    """OpenCV Telea inpaint 擦除 box 内文字"""
    import cv2
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    x2, y2 = min(x2, w), min(y2, h)
    mask = np.zeros((h, w), np.uint8)
    # 扩展 8px 边界让 inpaint 羽化
    pad = 8
    mask[max(0,y1-pad):y2+pad, max(0,x1-pad):x2+pad] = 255
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2)
    out = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def fit_font(text, font_path, max_w, max_h):
    lo, hi = 50, 600
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        lw = f.getlength(text)
        lh = f.size * 1.05
        if lw <= max_w and lh <= max_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ImageFont.truetype(font_path, best)


def burn_text(img_rgb, box, text, font_name, color):
    """在指定 box 内居中烧字"""
    img = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in box]
    max_w, max_h = x2 - x1, y2 - y1
    fp = os.path.join(FONTS, font_name)
    f = fit_font(text, fp, max_w * 0.95, max_h * 0.90)
    lw = f.getlength(text)
    lh = f.size * 1.0
    lx = x1 + (max_w - lw) / 2
    ly = y1 + (max_h - lh) / 2 + max_h * 0.04
    draw.text((lx, ly), text, font=f, fill=color)
    out = Image.alpha_composite(img, Image.new("RGBA", img.size, (0,0,0,0))).convert("RGB")
    return np.array(out)


def burn_one(ref_id, debug=False):
    """对 v186 输出自动 OCR + 烧字"""
    src_path = f"{JOB}/v186_{ref_id}.jpg"
    if not os.path.exists(src_path):
        print(f"[skip] {ref_id}: v186 输出未找到 {src_path}")
        return None
    out_path = f"{JOB}/v186_{ref_id}_burned.jpg"

    img_rgb = load_rgb(src_path)
    h, w = img_rgb.shape[:2]
    print(f"=== {ref_id} ({w}x{h}) ===")

    # OCR 找所有文字
    res = OCR.readtext(img_rgb, detail=1, paragraph=False)
    # 按 y 排序
    res.sort(key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))

    if debug:
        print("OCR 文字区:")
        for (bbox, text, conf) in res:
            xs=[p[0] for p in bbox]; ys=[p[1] for p in bbox]
            print(f"  '{text}' conf={conf:.2f} box=({min(xs):.0f},{min(ys):.0f},{max(xs):.0f},{max(ys):.0f})")

    # 按 ref_id 选替换字典
    rep = WORD_REPLACE[ref_id]

    # 处理每条 OCR 文字
    for (bbox, text, conf) in res:
        xs=[p[0] for p in bbox]; ys=[p[1] for p in bbox]
        box = [min(xs), min(ys), max(xs), max(ys)]
        text_up = text.upper().strip()
        if text_up not in rep:
            # 容错：包含关键字也算
            matched = None
            for k in rep:
                if k in text_up or text_up in k:
                    matched = k; break
            if not matched:
                if debug: print(f"  [skip] '{text_up}' 不在替换字典")
                continue
            text_up = matched

        new_word, font_name, color, _ = rep[text_up]
        print(f"  [burn] '{text_up}' → '{new_word}' box={box}")

        # 1) inpaint 擦除
        img_rgb = inpaint_region(img_rgb, box, radius=6)
        # 2) 烧字
        img_rgb = burn_text(img_rgb, box, new_word, font_name, color)

    # 保存
    Image.fromarray(img_rgb).save(out_path, quality=95)
    sz = os.path.getsize(out_path) // 1024
    print(f"  saved -> {out_path} ({sz} KB)\n")
    return out_path


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["bat_logo", "camo_armed"]
    for t in targets:
        if t in WORD_REPLACE:
            burn_one(t, debug=True)
        else:
            print(f"[skip] {t}: 不在烧字字典中")