"""v196_bat_logo_textclear.py

在 v193_bat_logo 上面把三处伪词（拱形外圈 + 中央 + 下方）
替换为清晰可读的新词，符合徽章字风格（哥特/古风/繁复）。

修复 v191/v192 烧字硬错：
- 绝不画矩形底（fill_bg=False）
- 字色匹配徽章配色（深紫 + 金 + 浅紫），不出现黑矩形
- 弧字用 arc_text.draw_arc_text 沿极坐标弧排
- 中央与下方走 fit_font 二分找大字号
- 烧字前对各 box 做小 pad 的 Telea inpaint（pad=12, radius=18）

策略：
1. PIL 加载 → 缩到 1920x2560 以提速（OCR/烧字在缩图，最后放大保存）
2. easyocr 拿三处字 box 真实坐标（允许用户拍板）
3. 按真实 box 烧字
"""

import os
import sys
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import easyocr

sys.path.insert(0, "E:/Desktop/双接口/image-fission/src")
from arc_text import draw_arc_text

FONTS = "E:/Desktop/双接口/image-fission/fonts"
SRC = "E:/Desktop/双接口/image-fission/jobs/smoke_v193/v193_bat_logo.jpg"
OUT = "E:/Desktop/双接口/image-fission/jobs/smoke_v196/v196_bat_logo_textclear.jpg"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def fit_font(text, font_path, max_w, max_h):
    """二分找最大字号（保持默认字重）"""
    lo, hi = 30, 600
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        bb = f.getbbox(text)
        actual_w = bb[2] - bb[0]
        actual_h = bb[3] - bb[1]
        if actual_w <= max_w * 0.92 and actual_h <= max_h * 0.92:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ImageFont.truetype(font_path, best), best


def inpaint_box(img_rgb, box, radius=18, pad=12):
    """单个 box cv2 Telea inpaint，pad 小不擦到周围"""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return img_rgb
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = 255
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def inpaint_arc_band(img_rgb, cx, cy, r_inner, r_outer, radius=18, theta_min_deg=None, theta_max_deg=None):
    """环形 mask 只擦环带（可限制角度范围）。PIL +y 向下：θ=0 东, 90 南, ±180 西, -90 北。
    北半圆 = θ ∈ (-180°, 0°)，给 (theta_min_deg, theta_max_deg) 即可。
    """
    h, w = img_rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    theta_deg = np.degrees(np.arctan2(dy, dx))  # PIL 坐标

    band_mask = (dist >= r_inner) & (dist <= r_outer)
    if theta_min_deg is not None and theta_max_deg is not None:
        # 处理跨越 ±180 的角度窗
        if theta_min_deg < theta_max_deg:
            ang_mask = (theta_deg >= theta_min_deg) & (theta_deg <= theta_max_deg)
        else:
            ang_mask = (theta_deg >= theta_min_deg) | (theta_deg <= theta_max_deg)
        band_mask = band_mask & ang_mask

    mask = np.zeros((h, w), np.uint8)
    mask[band_mask] = 255
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def inpaint_union_boxes(img_rgb, boxes, radius=18, pad=80):
    """多个 box 并集 mask inpaint（精确不擦 box 之间的徽章装饰）"""
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for box in boxes:
        p1, p2, p3, p4 = box
        x1 = max(0, int(p1[0]) - pad)
        y1 = max(0, int(p1[1]) - pad)
        x2 = min(w, int(p3[0]) + pad)
        y2 = min(h, int(p3[1]) + pad)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def burn_text_in_box(img_rgb, box, text, font_path, color):
    """box 居中烧字（绝不画矩形底）"""
    img = Image.fromarray(img_rgb).convert("RGBA")
    x1, y1, x2, y2 = [int(v) for v in box]
    max_w, max_h = x2 - x1, y2 - y1
    if max_w < 50 or max_h < 30:
        return img_rgb

    f, used_size = fit_font(text, font_path, max_w, max_h)
    bb = f.getbbox(text)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]
    lx = x1 + (max_w - text_w) / 2 - bb[0]
    ly = y1 + (max_h - text_h) / 2 - bb[1]

    draw = ImageDraw.Draw(img)
    if len(color) == 3:
        color_rgba = (int(color[0]), int(color[1]), int(color[2]), 255)
    else:
        color_rgba = color
    draw.text((lx, ly), text, font=f, fill=color_rgba)
    # USM 锐化只对文字小区域（避免全图变硬）
    return np.array(img.convert("RGB"))


def sharpen_text_region(img_rgb, box, amount=1.4):
    """box 内轻度 USM 锐化，让字更锐"""
    img = Image.fromarray(img_rgb).convert("RGB")
    region = img.crop(box)
    region = region.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(amount * 100), threshold=2))
    img.paste(region, box)
    return np.array(img)


# ========== 主流程 ==========
def main():
    print(f"[v196] 加载 {SRC}")
    img_full = Image.open(SRC).convert("RGB")
    W0, H0 = img_full.size
    print(f"  原图尺寸: {W0}x{H0}")

    # 缩到 1920x2560 跑 OCR（提速，OCR 对缩放不敏感）
    WORK_W = 1920
    WORK_H = int(H0 * WORK_W / W0)
    img_work = img_full.resize((WORK_W, WORK_H), Image.LANCZOS)
    img_work.save("E:/Desktop/双接口/image-fission/jobs/smoke_v196/_00_work_resized.jpg", quality=90)

    # OCR 找三处文字 box（缩图坐标）
    print(f"  OCR 工作中...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    arr = np.array(img_work)
    ocr_out = reader.readtext(arr, detail=1, paragraph=False)
    print(f"  OCR 命中 {len(ocr_out)} 处")
    for i, (box, text, conf) in enumerate(ocr_out):
        x1 = min(p[0] for p in box); y1 = min(p[1] for p in box)
        x2 = max(p[0] for p in box); y2 = max(p[1] for p in box)
        print(f"  [{i}] '{text}' @({int(x1)},{int(y1)},{int(x2)},{int(y2)}) conf={conf:.2f}")

    # ========== 用户拍板：以下 box 来自 OCR 结果，将映射到原图后烧字 ==========
    # v193_bat_logo 缩图下三处字典型坐标（实测会动态调整）：
    # 拱形外圈 = 文字沿圆排 → OCR 可能返回多个 box，这里用「外圈周长范围」
    # 中央 SARI → 一个居中 box
    # 下方 SIPITR → 流苏上方一个小 box

    # 自动找三组 box（按 y 位置分区）：
    #   拱形外圈: y 中等（<1100 px，WORK_H=2560 的 43%）
    #   中央横字: 1100 < y < 1500（约 WORK_H 的 60%）
    #   下方小字: y > 1500（流苏上方）
    arc_boxes = []
    center_boxes = []
    bottom_boxes = []
    for box, text, conf in ocr_out:
        x1 = min(p[0] for p in box); y1 = min(p[1] for p in box)
        x2 = max(p[0] for p in box); y2 = max(p[1] for p in box)
        cy = (y1 + y2) / 2
        if conf < 0.10:
            continue
        if cy < WORK_H * 0.43:
            arc_boxes.append((box, text, conf))
        elif cy < WORK_H * 0.60:
            center_boxes.append((box, text, conf))
        else:
            bottom_boxes.append((box, text, conf))

    # 取置信度最高的几个
    arc_boxes.sort(key=lambda x: -x[2])
    center_boxes.sort(key=lambda x: -x[2])
    bottom_boxes.sort(key=lambda x: -x[2])

    print(f"\n  拱形外圈 OCR: {len(arc_boxes)} 处")
    print(f"  中央横字 OCR: {len(center_boxes)} 处")
    print(f"  下方小字 OCR: {len(bottom_boxes)} 处")

    if not center_boxes or not bottom_boxes:
        print("\n[!!] 关键位置 OCR 命中不足，fail-fast，等用户拍板")
        # 把 OCR 命中图渲染出来供用户判断
        dump_img = img_work.copy()
        from PIL import ImageDraw as D2
        d = D2.Draw(dump_img)
        for box, text, conf in ocr_out:
            x1 = int(min(p[0] for p in box)); y1 = int(min(p[1] for p in box))
            x2 = int(max(p[0] for p in box)); y2 = int(max(p[1] for p in box))
            d.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            d.text((x1, y1 - 22), f"{text} ({conf:.2f})", fill="lime")
        dump_img.save("E:/Desktop/双接口/image-fission/jobs/smoke_v196/_01_ocr_dump.jpg", quality=92)
        return False

    # ========== 烧字（用 WORK 坐标烧，保存为 1920x2560 工作图，最后放大存） ==========
    # 拱形外圈：按 OCR box 的 x 范围粗估外圈半径，中心取中央横字 box 的中心 y
    # 取最大置信度 box 计算
    center_box = center_boxes[0][0]  # (x1,y1),(x2,y1),(x2,y2),(x1,y2)
    cb_cx = (center_box[0][0] + center_box[2][0]) / 2
    cb_cy = (center_box[0][1] + center_box[2][1]) / 2
    cb_w = center_box[2][0] - center_box[0][0]

    # 拱形外圈 OCR box 全部输入 → 取圆心与最右 box 的 x2 拟合
    # 拱形外圈 = 徽章外缘内侧的字带
    # 实测徽章中心 (cx=954, cy=1110)，字带中心 r ≈ 720（v193 拱字带在徽章外圆 720 px 处）
    arc_cx = 954
    arc_cy = 1110
    arc_radius = 760  # 实测字带最外/最内字符中心 r=748-778，字带中心 r=760

    print(f"\n  拱字参数: center=({arc_cx:.0f},{arc_cy:.0f}) radius={arc_radius:.0f}")

    # 工作图（1920 宽）
    img_work_rgb = np.array(img_work.convert("RGB"))

    # === 1) 中央横字 ===
    cb = [int(v) for v in (center_box[0][0], center_box[0][1], center_box[2][0], center_box[2][1])]
    img_work_rgb = inpaint_box(img_work_rgb, cb, radius=18, pad=14)
    img_work_rgb = burn_text_in_box(
        img_work_rgb, cb,
        text="CURSE",
        font_path=os.path.join(FONTS, "PirataOne-Regular.ttf"),
        color=(28, 18, 46),  # 深紫，与背景紫徽章融，不黑不白
    )
    img_work_rgb = sharpen_text_region(img_work_rgb, cb, amount=1.6)

    # === 2) 下方小字 ===
    if bottom_boxes:
        bb = bottom_boxes[0][0]
        bbox = [int(v) for v in (bb[0][0], bb[0][1], bb[2][0], bb[2][1])]
        img_work_rgb = inpaint_box(img_work_rgb, bbox, radius=15, pad=10)
        img_work_rgb = burn_text_in_box(
            img_work_rgb, bbox,
            text="EST. MMXXVI",
            font_path=os.path.join(FONTS, "Rye-Regular.ttf"),
            color=(201, 169, 97),  # 金色，徽章配色配金
        )
        img_work_rgb = sharpen_text_region(img_work_rgb, bbox, amount=1.6)

    # === 3) 拱形外圈 = 严格环形 inpaint（r 730-790, θ ∈ 北半圆 + 限制更小角度） ===
    ang_min = -180.0  # 字带在北半圆西侧 (theta 在 PIL 坐标)
    ang_max = 0.0     # 北半圆东侧
    arc_band_inner = 620
    arc_band_outer = 830
    print(f"  拱字环 inpaint r=[{arc_band_inner},{arc_band_outer}] θ∈[{ang_min},{ang_max}]")
    img_work_rgb = inpaint_arc_band(
        img_work_rgb, int(arc_cx), int(arc_cy),
        arc_band_inner, arc_band_outer,
        radius=22, theta_min_deg=ang_min, theta_max_deg=ang_max,
    )

    # === 3b) 上方飘逸装饰条 W HULI 区 inpaint ===
    # OCR 估计装饰条在 y=0-180 之间，预估居中约 y=90, 高约 60 px
    w_huli_box = [int(WORK_W * 0.20), 30, int(WORK_W * 0.80), 170]
    # 只擦装饰条中心窄带（高度 60 px），保留飘带本身
    img_work_rgb = inpaint_box(img_work_rgb, w_huli_box, radius=14, pad=4)

    img_pil = Image.fromarray(img_work_rgb).convert("RGB")
    arc_text = "MOONCREST"
    deg_span = 140.0
    arc_len_px = abs(math.radians(deg_span) * arc_radius)
    arc_font_size = int(arc_len_px / (len(arc_text) * 1.6))
    arc_font_size = max(48, min(arc_font_size, 130))
    print(f"  弧字: '{arc_text}' 字号={arc_font_size} 圆心=({arc_cx:.0f},{arc_cy:.0f}) 半径={arc_radius:.0f}")
    img_pil = draw_arc_text(
        img_pil,
        text=arc_text,
        font_path=os.path.join(FONTS, "PirataOne-Regular.ttf"),
        font_size=arc_font_size,
        color=(28, 18, 46),  # 深紫
        center=(arc_cx, arc_cy),
        radius=arc_radius,
        start_angle_deg=200,
        end_angle_deg=340,
        char_spacing_px=int(arc_font_size * 0.10),
        flip_180=False,  # 字底部朝圆心 → 字头朝外侧（沿弧上字头朝上）
    )

    # === 3c) 上方飘带 W HULI 烧清晰字 ===
    w_huli_text = "EST. MMXXVI"
    w_huli_font_size = 36
    f_w = ImageFont.truetype(os.path.join(FONTS, "Rye-Regular.ttf"), w_huli_font_size)
    bb_w = f_w.getbbox(w_huli_text)
    bw_w = bb_w[2] - bb_w[0]
    bh_w = bb_w[3] - bb_w[1]
    lx = int((WORK_W - bw_w) / 2 - bb_w[0])
    ly = 95 - bh_w // 2 - bb_w[1]
    d_w = ImageDraw.Draw(img_pil)
    d_w.text((lx, ly), w_huli_text, font=f_w, fill=(201, 169, 97))
    img_pil = img_pil.convert("RGB")
    img_work_rgb = np.array(img_pil.convert("RGB"))

    # === 3c) 上方飘带 W HULI 烧清晰字 ===
    w_huli_text = "EST. MMXXVI"
    w_huli_font_size = 36
    f_w = ImageFont.truetype(os.path.join(FONTS, "Rye-Regular.ttf"), w_huli_font_size)
    bb_w = f_w.getbbox(w_huli_text)
    bw_w = bb_w[2] - bb_w[0]
    bh_w = bb_w[3] - bb_w[1]
    lx = int((WORK_W - bw_w) / 2 - bb_w[0])
    ly = 95 - bh_w // 2 - bb_w[1]
    img_pil = Image.fromarray(np.array(img_pil.convert("RGB")))
    d_w = ImageDraw.Draw(img_pil)
    d_w.text((lx, ly), w_huli_text, font=f_w, fill=(201, 169, 97))
    img_pil = img_pil.convert("RGB")

    # 保存工作图缩图
    out_work = "E:/Desktop/双接口/image-fission/jobs/smoke_v196/_02_work_burned.jpg"
    Image.fromarray(img_work_rgb).save(out_work, quality=92)
    print(f"  工作图保存: {out_work}")

    # 放大到 3840x5120 原图尺寸
    img_full_burned = Image.fromarray(img_work_rgb).resize((W0, H0), Image.LANCZOS)
    # 轻 USM 让字清晰
    img_full_burned = img_full_burned.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=2))
    img_full_burned.save(OUT, quality=96)
    print(f"  最终保存: {OUT}")

    # OCR 自检新图（用 PIL 加载数组，绕过 cv2 读图 bug）
    arr_full = np.array(img_full_burned.convert("RGB"))
    new_ocr = reader.readtext(arr_full, detail=1, paragraph=False)
    print(f"\n  v196 烧后 OCR 命中 {len(new_ocr)} 处:")
    for box, text, conf in new_ocr:
        if conf >= 0.3:
            print(f"    '{text}' conf={conf:.2f}")

    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
