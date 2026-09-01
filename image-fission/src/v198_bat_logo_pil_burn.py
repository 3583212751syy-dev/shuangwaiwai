"""
v198 — v197 迷彩底色底子上 PIL 后期烧清晰字

v197 解决了迷彩底色（橄榄绿/卡其/深棕/黑），但 SDXL 没拼出真英文单词，
徽章元素也跑偏成星徽章。v198 在 v197 底子上 PIL 烧：
  - 顶部飘带：EST. MMXXVI（金色 Rye）
  - 徽章上方：MOONCREST（沿弧排，金色 PirataOne）
  - 徽章中央：CURSE（横字，深紫 PirataOne）
  - 底部小字：（不烧，避免与底部飘带冲突）

诚实告知徽章元素（SDXL 画成星徽章非蝙蝠）是裂变自由度（小元素可换）。
"""
import os, sys, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT = Path(__file__).resolve().parents[1]
FONTS = PROJECT / "fonts"
JOB = PROJECT / "jobs" / "smoke_v198"
JOB.mkdir(parents=True, exist_ok=True)

V197 = PROJECT / "jobs" / "smoke_v197" / "v197_bat_logo.jpg"
OUT = JOB / "v198_bat_logo_burned.jpg"

# ============== 颜色匹配徽章色系 ==============
GOLD = (190, 145, 60)       # 星徽章金色
DARK_PURPLE = (38, 20, 50)  # 深紫（与 BACARDÍ 旧徽章呼应）


def inpaint_box(img_rgb, box, radius=18, pad=18):
    """单 box cv2 Telea inpaint, pad 小不擦到周围"""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return img_rgb
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = 255
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def inpaint_mask(img_rgb, mask, radius=18):
    """自定义 mask 形状（环带等）"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def find_ocr_text(img_rgb):
    """读残留字位置（用 PIL 加载 numpy 绕过 cv2 的 JPG 读图 bug）"""
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    out = reader.readtext(img_rgb, detail=1)
    boxes = []
    for box, text, conf in out:
        if conf < 0.20:
            continue
        x1 = min(p[0] for p in box); y1 = min(p[1] for p in box)
        x2 = max(p[0] for p in box); y2 = max(p[1] for p in box)
        boxes.append((x1, y1, x2, y2, text, conf))
    return boxes


def draw_arc_text(img_pil, text, font_path, font_size, color, center, radius,
                  start_angle_deg, end_angle_deg, char_spacing_px=2, flip_180=False):
    """沿极轴圆弧排字"""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    deg_span = end_angle_deg - start_angle_deg
    arc_len_px = abs(math.radians(deg_span) * radius)
    char_w_estimate = font_size * 0.62
    total_w = char_w_estimate * len(text) + char_spacing_px * (len(text) - 1)
    if total_w > arc_len_px:
        scale = arc_len_px / total_w
        font_size = int(font_size * scale)
        font = ImageFont.truetype(font_path, font_size)
        char_w_estimate = font_size * 0.62
        total_w = char_w_estimate * len(text) + char_spacing_px * (len(text) - 1)

    cx, cy = center
    # 沿弧均匀分布字符角度
    n = len(text)
    angles = np.linspace(start_angle_deg, end_angle_deg, n)
    for ch, deg in zip(text, angles):
        rad = math.radians(deg)
        # 字中心位置
        tx = cx + radius * math.cos(rad)
        ty = cy + radius * math.sin(rad)
        # 字朝向 = 字底部切线方向
        # PIL 旋转：正角度逆时针
        rot_angle = -deg + (180 if flip_180 else 0)
        # 用 ImageDraw.text + Image.rotate 画
        bbox = draw.textbbox((0, 0), ch, font=font)
        cw = bbox[2] - bbox[0]
        ch_h = bbox[3] - bbox[1]
        # 单独画一张字贴图
        txt_img = Image.new("RGBA", (cw + 8, ch_h + 8), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(txt_img)
        d2.text((4, 4 - bbox[1]), ch, font=font, fill=color + (255,))
        txt_img = txt_img.rotate(rot_angle, resample=Image.BICUBIC, expand=True)
        # 贴到主图
        paste_x = int(tx - txt_img.width / 2)
        paste_y = int(ty - txt_img.height / 2)
        # 半透明 alpha_composite
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img.alpha_composite(txt_img, (paste_x, paste_y))
    return img


def main():
    if not V197.exists():
        raise SystemExit(f"missing v197 image: {V197}")

    print(f"[load] {V197}")
    img_pil = Image.open(V197).convert("RGB")
    W, H = img_pil.size
    print(f"[size] {W}x{H}")
    img_rgb = np.array(img_pil)

    # === 1) OCR 找残留字 ===
    boxes = find_ocr_text(img_rgb)
    print(f"[ocr] found {len(boxes)} text boxes")
    for b in boxes:
        print(f"  - '{b[4]}' conf={b[5]:.2f} box=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")

    # === 2) 分区 ===
    # 按 y 位置分：
    #   顶部飘带 (y < 0.18) → EST. MMXXVI
    #   徽章上方 (0.18 < y < 0.40) → MOONCREST 沿弧
    #   徽章中央 (0.40 < y < 0.65) → CURSE
    #   徽章下方 (y > 0.65) → 不烧字
    top_boxes = [b for b in boxes if b[1] < H * 0.18]
    above_badge_boxes = [b for b in boxes if H * 0.18 <= b[1] < H * 0.40]
    center_boxes = [b for b in boxes if H * 0.40 <= b[1] < H * 0.65]

    print(f"[zones] top={len(top_boxes)} above={len(above_badge_boxes)} center={len(center_boxes)}")

    # === 3) 徽章中心 + 半径（人眼看 v197 后定坐标）===
    # v197 徽章 = SDXL 画的金色星徽章，约在图中央偏上
    # 图 4096x4096，徽章圆心实测约 (2050, 1850)，半径约 950
    cb_cx = 2050
    cb_cy = 1850
    cb_w = 1900   # 徽章外径
    cb_h = 1900
    print(f"[badge hardcoded] center=({cb_cx},{cb_cy}) r≈{cb_w//2}")

    # === 4) inpaint 残留字 ===
    for b in top_boxes + above_badge_boxes + center_boxes:
        box_xy = (b[0], b[1], b[2], b[3])
        img_rgb = inpaint_box(img_rgb, box_xy, radius=16, pad=14)
    print("[inpaint] cleared residual text")

    # 备份当前状态
    Image.fromarray(img_rgb).save(JOB / "_01_after_inpaint.jpg", quality=92)

    # === 5) PIL 烧字 ===
    img_pil = Image.fromarray(img_rgb).convert("RGB")

    # 5a) 顶部飘带：EST. MMXXVI（横排，金色 Rye）
    top_y = int(H * 0.10)
    top_font_size = int(H * 0.038)
    img_pil_draw = ImageDraw.Draw(img_pil)
    font_rye = ImageFont.truetype(str(FONTS / "Rye-Regular.ttf"), top_font_size)
    top_text = "EST. MMXXVI"
    bb = img_pil_draw.textbbox((0, 0), top_text, font=font_rye)
    tw = bb[2] - bb[0]
    img_pil_draw.text(((W - tw) // 2, top_y - (bb[1] - 0)), top_text, font=font_rye, fill=GOLD)

    # 5b) 徽章上方弧字 MOONCREST（沿徽章外圈走，flip=False 字正向朝上）
    arc_cx = cb_cx
    arc_cy = cb_cy + 80
    arc_radius = int(cb_w * 0.50)  # 走徽章外圈
    arc_font_size = int(arc_radius * math.radians(140) / (len("MOONCREST") * 1.5))
    arc_font_size = max(120, min(arc_font_size, 250))
    print(f"[arc text] MOONCREST font={arc_font_size} r={arc_radius} center=({arc_cx},{arc_cy})")
    img_pil = draw_arc_text(
        img_pil, text="MOONCREST",
        font_path=str(FONTS / "PirataOne-Regular.ttf"),
        font_size=arc_font_size,
        color=GOLD,
        center=(arc_cx, arc_cy),
        radius=arc_radius,
        start_angle_deg=200,  # 左下 → 沿上半圆周
        end_angle_deg=340,
        char_spacing_px=-int(arc_font_size * 0.02),  # 字符略贴紧
        flip_180=False,  # 字正向朝上（标准徽章字）
    )

    # 5c) 徽章中央横字 CURSE
    curse_font_size = int(cb_w * 0.16)
    curse_font_size = max(140, min(curse_font_size, 360))
    font_pir = ImageFont.truetype(str(FONTS / "PirataOne-Regular.ttf"), curse_font_size)
    img_pil_draw = ImageDraw.Draw(img_pil)
    curse_text = "CURSE"
    bb = img_pil_draw.textbbox((0, 0), curse_text, font=font_pir)
    tw = bb[2] - bb[0]
    curse_y = cb_cy + 100  # 徽章中央偏下
    img_pil_draw.text(((W - tw) // 2, curse_y - (bb[1] - 0)), curse_text, font=font_pir, fill=DARK_PURPLE)

    # === 6) USM 锐化让字清晰 ===
    img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))

    # === 7) 保存 ===
    img_pil.convert("RGB").save(OUT, quality=95)
    print(f"\n[done] {OUT}")
    print(f"size: {img_pil.size}")


if __name__ == "__main__":
    main()