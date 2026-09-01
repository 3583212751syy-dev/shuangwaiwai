"""
v191_burn.py — bat_logo + camo_armed 烧字修复版

关键修复（v190 烧糊的根因）：
  1. inpaint 半径从 8 提到 25（v190 输出 3840x5120 大图，radius=8 擦不干净 SDXL 残留字）
  2. 相邻 box 自动合并 inpaint（bat_logo BACARDI+HEART 间距仅 7px）
  3. 强制 weight_axis="Bold"（Oswald/Lora VF 用默认 Regular → 笔画细）
  4. 用 v190 输出图的 OCR 真实残留字 bbox（不用原图坐标再缩放）
  5. fill_bg=True 在烧字区域填底色，覆盖任何擦不掉的阴影

源图（原图坐标 1552x2000）：
  bat_logo:
    弧字 "LA CASA DEL MURCIÉLAGO"   (420,256,1229,698)   -> "WINGS OF NIGHT"
    BACARDÍ 大字                    (281,946,1284,1187)  -> "BATANO"
    HEART  小字                      (459,1172,1138,1340) -> "SOUL"
  camo_armed:
    顶部小字 "WE SUPPORT THE"        (522,462,1056,534)   -> "WE HONOR"
    ARMED 中间大字                   (467,529,1112,652)   -> 单字 "BRAVE"
    FORCES 下面大字                  (446,640,1157,779)
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

FONTS = "E:/Desktop/双接口/image-fission/fonts"
JOBS_BASE = "E:/Desktop/双接口/image-fission/jobs"

sys.path.insert(0, "E:/Desktop/双接口/image-fission/src")
from arc_text import draw_arc_text


# ====================== 工具 ======================
def fit_font(text, font_path, max_w, max_h, weight_axis="Bold"):
    """二分找最大字号（限制宽高，强制粗体）"""
    lo, hi = 30, 800
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        if weight_axis is not None:
            try:
                f.set_variation_by_name(weight_axis)
            except Exception:
                pass
        actual_w = f.getbbox(text)[2] - f.getbbox(text)[0]
        actual_h = f.getbbox(text)[3] - f.getbbox(text)[1]
        if actual_w <= max_w * 0.96 and actual_h <= max_h * 0.95:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    f = ImageFont.truetype(font_path, best)
    if weight_axis is not None:
        try:
            f.set_variation_by_name(weight_axis)
        except Exception:
            pass
    return f, best


def inpaint_box(img_rgb, box, radius=22, pad=30):
    """box 区域 cv2 Telea inpaint 清空（半径足够大, pad 足够覆盖边缘）"""
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


def merge_boxes(*boxes):
    """合并多个 box 成一个外接矩形"""
    xs = []
    ys = []
    for b in boxes:
        xs.extend([b[0], b[2]])
        ys.extend([b[1], b[3]])
    return [min(xs), min(ys), max(xs), max(ys)]


def sample_bg_color(img_rgb, box):
    """采样 box 内（边缘或顶部）的背景颜色（去掉字后的纸面色）"""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    # 取 box 顶部 8 行作为底色（字都在中下部）
    y_top = max(0, y1)
    y_btm = min(h, y1 + 12)
    x_l = max(0, x1)
    x_r = min(w, x2)
    if y_btm <= y_top or x_r <= x_l:
        return (245, 235, 240)
    region = img_rgb[y_top:y_btm, x_l:x_r]
    if region.size == 0:
        return (245, 235, 240)
    med = np.median(region.reshape(-1, 3), axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def burn_text_in_box(img_rgb, box, text, font_path, color, weight_axis="Bold",
                     fill_bg=True, bg_color=None):
    """在 box 区域烧入文字（水平）"""
    img = Image.fromarray(img_rgb).convert("RGBA")
    x1, y1, x2, y2 = [int(v) for v in box]
    max_w, max_h = x2 - x1, y2 - y1
    if max_w < 50 or max_h < 30:
        return img_rgb
    if fill_bg:
        if bg_color is None:
            bg_color = sample_bg_color(img_rgb, box)
        # 转 RGBA 用
        if len(bg_color) == 3:
            bg_rgba = (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]), 255)
        else:
            bg_rgba = bg_color
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], fill=bg_rgba)
    f, used_size = fit_font(text, font_path, max_w, max_h, weight_axis=weight_axis)
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
    return np.array(img.convert("RGB"))


# ====================== bat_logo 烧字 ======================
def burn_bat_logo(in_path, out_path):
    """源：原图 1552x2000 / v190 输出 3840x5120
       本函数直接对 v190 输出操作，使用 OCR v190 的真实残留字坐标（不需要缩放）
    """
    img_rgb = np.array(Image.open(in_path).convert("RGB"))
    gh, gw = img_rgb.shape[:2]

    # === v190 bat_logo OCR 真实残留字坐标（直接用 v190 输出图的物理像素）===
    # BACARDI: (698,2518,3174,2982)   — 占原图 (281-1284, 946-1187) ≈ 像素位置
    # WHLART:  (1152,2989,2827,3435)  — HEART 残留
    # 弧字残留位于顶部：实际 v190 输出图弧字位置需重新根据比例计算
    # 原图弧字区 (420,256,1229,698)，输出 3840x5120 → 比例 2.47/2.56
    sx, sy = gw / 1552, gh / 2000
    print(f"[bat_logo] 输出图 {gw}x{gh}, scale=({sx:.2f},{sy:.2f})")

    arc_box_src = (420, 256, 1229, 698)
    arc_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
               for i, v in enumerate(arc_box_src)]
    bac_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
               for i, v in enumerate([281, 946, 1284, 1187])]
    heart_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
                 for i, v in enumerate([459, 1172, 1138, 1340])]

    # 合并 BACARDI + HEART 一个大区域做彻底 inpaint（间距仅 ~7px，单独擦会留缝）
    merged_text_box = merge_boxes(bac_box, heart_box)
    # 上下各扩 40 px，左右各扩 40 px
    img_rgb = inpaint_box(img_rgb, merged_text_box, radius=25, pad=40)
    print(f"  [bat_logo] inpaint 合并区域 {merged_text_box}")

    # 画 BATANO（serif serif Bold + 底色填充，严盖 SDXL 残留）
    img_rgb = burn_text_in_box(
        img_rgb, bac_box,
        text="BATANO",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),
        color=(20, 12, 32),
        weight_axis="Bold",
        fill_bg=True,  # 填底色
    )
    # 画 SOUL
    img_rgb = burn_text_in_box(
        img_rgb, heart_box,
        text="SOUL",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),
        color=(20, 12, 32),
        weight_axis="Regular",
        fill_bg=True,
    )

    # === 弧字：先大力度 inpaint 整个弧字区，再画新弧字 ===
    img_rgb = inpaint_box(img_rgb, arc_box, radius=25, pad=35)
    print(f"  [bat_logo] inpaint 弧字区 {arc_box}")

    img_pil = Image.fromarray(img_rgb).convert("RGB")
    cx = (arc_box[0] + arc_box[2]) // 2 + 50  # 中心略微右移补偿左侧空
    cy = arc_box[3] + 50  # 弧心在弧字下方一点
    radius = (arc_box[2] - arc_box[0]) // 2 + 100
    arc_font_size = int((arc_box[2] - arc_box[0]) / 14)
    img_pil = draw_arc_text(
        img_pil,
        text="WINGS OF NIGHT",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),
        font_size=arc_font_size,
        color=(20, 12, 32),
        center=(cx, cy),
        radius=radius,
        start_angle_deg=200,
        end_angle_deg=340,
        char_spacing_px=int(arc_font_size * 0.06),
        flip_180=False,
    )
    img_rgb = np.array(img_pil.convert("RGB"))

    Image.fromarray(img_rgb).save(out_path, quality=95)
    print(f"[bat_logo] burned → {out_path}")
    return out_path


# ====================== camo_armed 烧字 ======================
def burn_camo_armed(in_path, out_path):
    img_rgb = np.array(Image.open(in_path).convert("RGB"))
    gh, gw = img_rgb.shape[:2]

    sx, sy = gw / 1556, gh / 2000
    print(f"[camo_armed] 输出图 {gw}x{gh}, scale=({sx:.2f},{sy:.2f})")

    # v190 camo_armed OCR 残留字坐标直接用（v190 输出图物理像素）
    # 顶小字: (1276,1176,2636,1376)
    # ARMED:  (1164,1357,2735,1673)
    # FORCES: (1047,1620,2856,1984)
    # 原图坐标：
    top_box_src = (522, 462, 1056, 534)
    armed_box_src = (467, 529, 1112, 652)
    forces_box_src = (446, 640, 1157, 779)

    top_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
               for i, v in enumerate(top_box_src)]
    armed_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
                 for i, v in enumerate(armed_box_src)]
    forces_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
                  for i, v in enumerate(forces_box_src)]

    # 合并三个 box（间距都只 ~20px，必须一起擦）
    merged_box = merge_boxes(top_box, armed_box, forces_box)
    print(f"  [camo_armed] 合并区域 {merged_box}")
    img_rgb = inpaint_box(img_rgb, merged_box, radius=28, pad=40)

    # 画顶部小字 WE HONOR（Oswald Bold，相对小字号）
    img_rgb = burn_text_in_box(
        img_rgb, top_box,
        text="WE HONOR",
        font_path=os.path.join(FONTS, "Oswald-VF.ttf"),
        color=(15, 12, 10),
        weight_axis="Bold",
        fill_bg=True,
    )

    # 画中部 BRAVE 大字（Anton Regular 单字，超粗黑）
    # 用 armed_box 的高度（单字占 ARMED 行高）
    img_rgb = burn_text_in_box(
        img_rgb, armed_box,
        text="BRAVE",
        font_path=os.path.join(FONTS, "Anton-Regular.ttf"),
        color=(15, 12, 10),
        weight_axis=None,
        fill_bg=True,
    )

    # forces_box 下面是 FORCES 原位置（未烧，但已擦空，让 SDXL 残留消失）
    # 如果想保留位置可再用细字加小语

    Image.fromarray(img_rgb).save(out_path, quality=95)
    print(f"[camo_armed] burned → {out_path}")
    return out_path


if __name__ == "__main__":
    args = sys.argv[1:]
    job = args[0] if args else "smoke_v190"

    in_job = f"{JOBS_BASE}/{job}"
    if "v190" in job:
        prefix = "v190"
    elif "v189" in job:
        prefix = "v189"
    else:
        prefix = "v188"

    bat_src = f"{in_job}/{prefix}_bat_logo.jpg"
    bat_dst = f"{in_job}/{prefix}_bat_logo_burned_v191.jpg"
    if os.path.exists(bat_src):
        burn_bat_logo(bat_src, bat_dst)
    else:
        print(f"[skip] bat_logo not found: {bat_src}")

    camo_src = f"{in_job}/{prefix}_camo_armed.jpg"
    camo_dst = f"{in_job}/{prefix}_camo_armed_burned_v191.jpg"
    if os.path.exists(camo_src):
        burn_camo_armed(camo_src, camo_dst)
    else:
        print(f"[skip] camo_armed not found: {camo_src}")
