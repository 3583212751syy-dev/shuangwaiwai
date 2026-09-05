#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v268 — 本地 Big-LaMa 干净去字 + PIL 精确重绘新字
不依赖 ComfyUI 重启：直接在 venv 里加载 big-lama.pt（先拷到 ASCII 临时路径避开
Windows 中文路径 fopen bug），复刻 ComfyUI-RMBG AILab_LamaRemover 的去字逻辑。
流程: 手动 bbox 文字区域 -> LaMa 去除旧字(生成匹配背景的纹理) -> PIL 用高对比
      衬线字体(AbrilFatface) 按原 bbox 重绘新字。
治本: 旧字清不干净(LaMa 学过的修复) + 新字糊(正确字体+精确 bbox)。
"""
import argparse
import io
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import arc_text

PROJECT = Path("E:/Desktop/双接口/image-fission")
OUT = PROJECT / "jobs" / "v268"
OUT.mkdir(parents=True, exist_ok=True)
SRC_ROOT = Path("E:/Desktop/图裂变测试图")
FONT_PATH = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")
MODEL_PATH = PROJECT / "ComfyUI" / "models" / "RMBG" / "Lama" / "big-lama.pt"

_LAMA = None
_OCR = None


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
    # 节点逻辑: invert -> blur -> threshold(>strength 变0/保留, 否则255/去除)
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
# BACARDÍ 蝙蝠图手动文字行 / 弧带多边形
# ----------------------------------------------------------------------------
def bat_rows():
    # 坐标由 crop + 暗像素分析共同校准；蝙蝠约 y=358-946，文字必须避开蝙蝠
    return [
        ("arc",  (0, 260, 1552, 370), "NIGHTBAT"),
        ("main", (180, 980, 1370, 1170), "DUSKHEART"),
        ("sub",  (380, 1180, 1175, 1330), "WINGS OF DARK"),
    ]


def arc_ribbon_polygon():
    """顶部弧带 ribbon 真实位置 y=260-370；覆盖整条带，不进入蝙蝠区."""
    outer = [
        (120, 380), (220, 310), (450, 270),
        (776, 255), (1100, 270), (1330, 310), (1430, 380)
    ]
    inner = [
        (1380, 340), (1200, 295), (1000, 275),
        (776, 270), (550, 275), (350, 295), (170, 340)
    ]
    return outer + inner[::-1]


def full_ribbon_polygon():
    """返回 (outer, inner) 多边形：覆盖整条旧字带，内边界绕开蝙蝠尾巴."""
    outer = [
        (120, 380), (120, 450), (140, 600), (180, 750), (250, 880),
        (450, 950), (700, 980), (850, 980), (1100, 950), (1300, 880),
        (1370, 750), (1410, 600), (1430, 450), (1430, 380),
        (1330, 310), (1100, 270), (776, 255), (450, 270), (220, 310),
    ]
    inner = [
        (600, 400), (776, 400), (950, 400),
        (1100, 520), (1100, 720), (1000, 800),
        (900, 850), (776, 900), (650, 850),
        (550, 800), (400, 720), (400, 520),
    ]
    return outer, inner


def rebuild_ribbon(src_pil, polys=None):
    """用手绘 ribbon 多边形 + 采样 ribbon 色平填覆盖整条旧字带，边缘羽化保留蝙蝠主体."""
    arr = np.array(src_pil)
    h, w = arr.shape[:2]
    light, _ = sample_ribbon_colors(arr, polys)
    outer, inner = full_ribbon_polygon()
    # 显式画 outer 白 + inner 黑，得到可靠 annulus mask
    poly_mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(poly_mask)
    d.polygon(outer, fill=255)
    d.polygon(inner, fill=0)
    # 羽化边缘，避免硬边
    poly_mask = poly_mask.filter(ImageFilter.GaussianBlur(radius=6))
    alpha = np.array(poly_mask, dtype=np.float32) / 255.0
    fill_layer = np.full_like(arr, light)
    out = (arr * (1 - alpha[..., None]) + fill_layer * alpha[..., None]).astype(np.uint8)
    return Image.fromarray(out)


def load_ocr():
    global _OCR
    if _OCR is not None:
        return _OCR
    import easyocr
    _OCR = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _OCR


def detect_text_polygons(src_pil, min_conf=0.25):
    """返回 EasyOCR 检测到的文字多边形 [(poly, text, conf)]."""
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
    """把多边形填充成 mask，并膨胀覆盖笔画边缘."""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    for p in polys:
        pts = [(int(x), int(y)) for x, y in p["poly"]]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    # 膨胀
    if dilate > 0:
        mask = mask.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    return mask


def sample_bg_color(src_arr, bbox, border=8):
    """取 bbox 四边边缘的众数/中位 RGB 作为背景色."""
    x1, y1, x2, y2 = bbox
    h, w = src_arr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = src_arr[y1:y2, x1:x2]
    if roi.size == 0:
        return (128, 128, 128)
    # 取边缘像素
    top = roi[:border, :, :].reshape(-1, 3)
    bot = roi[-border:, :, :].reshape(-1, 3)
    left = roi[:, :border, :].reshape(-1, 3)
    right = roi[:, -border:, :].reshape(-1, 3)
    edge = np.concatenate([top, bot, left, right], axis=0)
    # 中位数颜色
    bg = np.median(edge, axis=0).astype(np.uint8)
    return tuple(bg)


def fill_text_backgrounds(cleaned, src, rows):
    """对 main/sub 平面大字块用原图边缘背景色填充，避免 LaMa 残留 ghost.
    est/1862 靠近圆环，不填；弧带单独重绘."""
    out = np.array(cleaned)
    src_arr = np.array(src)
    for name, bbox, word in rows:
        if name not in ("main", "sub"):
            continue
        bg = sample_bg_color(src_arr, bbox, border=12)
        x1, y1, x2, y2 = bbox
        pad = 12
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(src.width, x2 + pad), min(src.height, y2 + pad)
        h = y2 - y1
        w = x2 - x1
        # 渐变：用上下邻域背景色
        top_c = np.array(sample_bg_color(src_arr, (x1, max(0, y1 - 40), x2, y1 + 20), border=6))
        bot_c = np.array(sample_bg_color(src_arr, (x1, y2 - 20, x2, min(src.height, y2 + 40)), border=6))
        grad = np.linspace(top_c, bot_c, max(1, h)).astype(np.uint8)
        grad = np.tile(grad[:, np.newaxis, :], (1, w, 1))
        out[y1:y2, x1:x2] = grad
    return Image.fromarray(out)


def ring_annulus(size, cx=776, cy=500, rx_out=620, ry_out=280, rx_in=310, ry_in=260, n=256):
    """生成徽章外环 annulus 多边形，覆盖整条 ribbon."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append((cx + rx_out * math.cos(a), cy + ry_out * math.sin(a)))
    for i in range(n - 1, -1, -1):
        a = 2 * math.pi * i / n
        pts.append((cx + rx_in * math.cos(a), cy + ry_in * math.sin(a)))
    return pts


def in_ring_annulus(x, y, cx=776, cy=500, rx_out=620, ry_out=280, rx_in=310, ry_in=260):
    """判断点是否在徽章环形带内."""
    out = ((x - cx) ** 2) / (rx_out ** 2) + ((y - cy) ** 2) / (ry_out ** 2) <= 1.0
    inn = ((x - cx) ** 2) / (rx_in ** 2) + ((y - cy) ** 2) / (ry_in ** 2) >= 1.0
    return out and inn


def ribbon_annulus_tight(size, cx=776, cy=500, rx_out=560, ry_out=250, rx_in=220, ry_in=180, n=256):
    """更贴近真实 ribbon 宽度的 annulus，仅用于 mask 裁剪."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append((cx + rx_out * math.cos(a), cy + ry_out * math.sin(a)))
    for i in range(n - 1, -1, -1):
        a = 2 * math.pi * i / n
        pts.append((cx + rx_in * math.cos(a), cy + ry_in * math.sin(a)))
    return pts


def side_arc_mask(size):
    """LA / MURCIELAGO / Est.1862 的兜底块，需再裁剪到 annulus."""
    side = Image.new("L", size, 0)
    ds = ImageDraw.Draw(side)
    ds.polygon([(120, 300), (450, 270), (450, 570), (120, 600)], fill=255)        # LA
    ds.polygon([(1050, 270), (1400, 300), (1400, 600), (1050, 570)], fill=255)    # MURCIELAGO
    ds.polygon([(450, 620), (700, 700), (850, 700), (1100, 620),
                (1100, 800), (850, 810), (700, 810), (450, 800)], fill=255)      # Est. 1862
    return side


def build_arc_mask(size, polys, dilate=12):
    """构建弧形带旧字 mask：EasyOCR 弧字 + 顶/左/右/底兜底 polygon，交给 cv2.inpaint."""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    # EasyOCR 检测到的弧形带文字（y < 900）
    for p in polys:
        c = p["poly"].mean(axis=0)
        if c[1] >= 900:
            continue
        pts = [(int(x), int(y)) for x, y in p["poly"]]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    # 兜底 polygon：覆盖 LA / MURCIELAGO / Est.1862 / 顶部弧线
    d.polygon(arc_ribbon_polygon(), fill=255)
    d.polygon([(120, 320), (450, 270), (450, 580), (120, 620)], fill=255)          # LA
    d.polygon([(1050, 270), (1400, 320), (1400, 620), (1050, 580)], fill=255)     # MURCIELAGO
    d.polygon([(450, 650), (700, 720), (850, 720), (1100, 650),
               (1100, 820), (850, 830), (700, 830), (450, 820)], fill=255)       # Est. 1862
    # 膨胀覆盖笔画边缘
    if dilate > 0:
        mask = mask.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    return mask


def build_combined_mask(size, polys, rows, dilate=14):
    """只 mask 主/副平面大字；弧形带已交给 cv2.inpaint."""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    # 只保留主/副字多边形（y >= 900）
    for p in polys:
        c = p["poly"].mean(axis=0)
        if c[1] < 900:
            continue
        pts = [(int(x), int(y)) for x, y in p["poly"]]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    # 膨胀覆盖笔画边缘
    mask = mask.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    # 手动 bbox 兜底：主/副
    d = ImageDraw.Draw(mask)
    for name, bbox, word in rows:
        if name == "arc":
            continue
        x1, y1, x2, y2 = bbox
        pad = 30 if name == "main" else 20
        d.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], fill=255)
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


def sample_ribbon_colors(src_arr, polys=None):
    """从顶部弧带干净中心采样 ribbon 本色；避免两侧旧字污染中位数."""
    h, w = src_arr.shape[:2]
    # 只取顶部中心几块干净 ribbon
    patches = [
        src_arr[270:320, 680:870].reshape(-1, 3),
        src_arr[275:325, 620:930].reshape(-1, 3),
    ]
    collected = [p for p in patches if p.size > 0]
    if not collected:
        return (225, 210, 225), (160, 150, 160)
    ring_px = np.concatenate(collected, axis=0)
    # 取最亮的 40% 像素（排掉阴影/旧字）
    means = ring_px.mean(axis=1)
    thr = np.percentile(means, 60) if len(means) > 50 else 180
    bright = ring_px[means >= thr]
    if len(bright) < 30:
        bright = ring_px[ring_px.mean(axis=1) > 170]
    if len(bright) < 30:
        bright = ring_px
    light = np.median(bright, axis=0).astype(np.uint8)
    shadow = (light.astype(np.float32) * 0.70).astype(np.uint8)
    return tuple(light.tolist()), tuple(shadow.tolist())


def draw_arc_ribbon(canvas, src, polys=None):
    """用准确采样的 ribbon 色填充整条 annulus，覆盖弧形带所有旧字."""
    arr = np.array(src)
    light, _ = sample_ribbon_colors(arr, polys)
    draw = ImageDraw.Draw(canvas)
    draw.polygon(ring_annulus(canvas.size), fill=light)
    return canvas


def draw_arc_word(canvas, word, bbox, font_path=FONT_PATH, color=(40, 40, 40)):
    """顶部弧线文字，位于 y=260-370 的 ribbon 内."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    radius = 520
    cy = 820  # 顶点 y = cy - radius ≈ 300，落在 ribbon 中央
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
    rows = bat_rows()
    src = Image.open(src_path).convert("RGB")
    print(f"[1/4] EasyOCR 检测文字...")
    polys = detect_text_polygons(src)
    print(f"  检测到 {len(polys)} 个文字区域")
    for p in polys:
        print(f"    '{p['text']}' conf={p['conf']:.2f}")

    print(f"[2/4] 手绘 ribbon 覆盖旧弧字 + LaMa 去主/副字")
    rebuilt = rebuild_ribbon(src, polys)
    rebuilt.save(OUT / f"{src_path.stem}_rebuilt.png")
    mask = build_combined_mask(src.size, polys, rows, dilate=14)
    mask.save(OUT / f"{src_path.stem}_mask.png")
    cleaned = lama_inpaint(rebuilt, mask, removal_strength=235, edge_smoothness=4)
    cleaned = fill_text_backgrounds(cleaned, src, rows)
    cleaned.save(OUT / f"{src_path.stem}_cleaned.png")
    print(f"  去字底图 -> {OUT / f'{src_path.stem}_cleaned.png'}")

    print("[3/4] PIL 重绘新字 (AbrilFatface)")
    out = cleaned.copy()
    for name, bbox, word in rows:
        if name == "arc":
            out = draw_arc_word(out, word, bbox, color=(40, 40, 40))
        elif name == "main":
            draw_word(out, word, bbox, max_size=170)
        elif name == "sub":
            draw_word(out, word, bbox, max_size=95)
        else:
            draw_word(out, word, bbox, max_size=60)
    out.save(OUT / f"{src_path.stem}_final.png")
    print(f"  成品 -> {OUT / f'{src_path.stem}_final.png'}")

    print("[4/4] 对照图")
    cmp = Image.new("RGB", (src.width * 2, src.height), (245, 245, 245))
    cmp.paste(src, (0, 0))
    cmp.paste(out, (src.width, 0))
    cmp.save(OUT / f"{src_path.stem}_compare.png")
    print(f"  对照 -> {OUT / f'{src_path.stem}_compare.png'}")


if __name__ == "__main__":
    main()
