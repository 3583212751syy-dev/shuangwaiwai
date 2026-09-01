"""
v192_burn.py — bat_logo + camo_armed 烧字（回退重做）

承认 v191 三个硬错（不再狡辩）：
  1. fill_bg=True 制造矩形底色，违反铁律"禁止背景色矩形遮字"
  2. weight_axis="Bold" 错误（BACARDI 原图是细字 Regular）
  3. merge_boxes 合并 inpaint 区域过大，擦到周围阴影

v192 修复：
  - fill_bg=False 严格（绝不画矩形）
  - weight_axis="Regular" 字体匹配原图细字
  - 每个 OCR box 单独 inpaint（pad=12 不擦到阴影）
  - inpaint radius=18（足够擦字但不擦周围）
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


def fit_font(text, font_path, max_w, max_h, weight_axis="Regular"):
    """二分找最大字号（强制细字 Regular）"""
    lo, hi = 30, 600
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        if weight_axis is not None:
            try:
                f.set_variation_by_name(weight_axis)
            except Exception:
                pass
        bb = f.getbbox(text)
        actual_w = bb[2] - bb[0]
        actual_h = bb[3] - bb[1]
        if actual_w <= max_w * 0.92 and actual_h <= max_h * 0.92:
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


def inpaint_box(img_rgb, box, radius=18, pad=12):
    """box 区域 cv2 Telea inpaint（pad 小，避免擦到周围阴影）"""
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


def burn_text_in_box(img_rgb, box, text, font_path, color, weight_axis="Regular"):
    """在 box 区域烧入文字（绝不画矩形底色）"""
    img = Image.fromarray(img_rgb).convert("RGBA")
    x1, y1, x2, y2 = [int(v) for v in box]
    max_w, max_h = x2 - x1, y2 - y1
    if max_w < 50 or max_h < 30:
        return img_rgb
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
    img_rgb = np.array(Image.open(in_path).convert("RGB"))
    gh, gw = img_rgb.shape[:2]
    sx, sy = gw / 1552, gh / 2000
    print(f"[bat_logo] 输出图 {gw}x{gh}, scale=({sx:.2f},{sy:.2f})")

    arc_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
               for i, v in enumerate((420, 256, 1229, 698))]
    bac_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
               for i, v in enumerate((281, 946, 1284, 1187))]
    heart_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
                 for i, v in enumerate((459, 1172, 1138, 1340))]

    # 每个 box 单独 inpaint（pad=12 小幅，不擦到周围阴影）
    img_rgb = inpaint_box(img_rgb, bac_box, radius=18, pad=10)
    img_rgb = burn_text_in_box(
        img_rgb, bac_box,
        text="BATANO",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),
        color=(20, 12, 32),
        weight_axis="Regular",  # 改 Regular 匹配原图细字
    )

    img_rgb = inpaint_box(img_rgb, heart_box, radius=15, pad=10)
    img_rgb = burn_text_in_box(
        img_rgb, heart_box,
        text="SOUL",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),
        color=(20, 12, 32),
        weight_axis="Regular",
    )

    # 弧字单独 inpaint
    img_rgb = inpaint_box(img_rgb, arc_box, radius=18, pad=15)
    print(f"  [bat_logo] inpaint 弧字区 {arc_box}")

    img_pil = Image.fromarray(img_rgb).convert("RGB")
    cx = (arc_box[0] + arc_box[2]) // 2 + 50
    cy = arc_box[3] + 50
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

    top_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
               for i, v in enumerate((522, 462, 1056, 534))]
    armed_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
                 for i, v in enumerate((467, 529, 1112, 652))]
    forces_box = [int(v * sx) if i % 2 == 0 else int(v * sy)
                  for i, v in enumerate((446, 640, 1157, 779))]

    # 每个 box 单独 inpaint（小 pad）
    img_rgb = inpaint_box(img_rgb, top_box, radius=15, pad=8)
    img_rgb = burn_text_in_box(
        img_rgb, top_box,
        text="WE HONOR",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),  # 改 serif（Oswald 没 Regular 用 Lora）
        color=(15, 12, 10),
        weight_axis="Regular",
    )

    img_rgb = inpaint_box(img_rgb, armed_box, radius=18, pad=10)
    img_rgb = burn_text_in_box(
        img_rgb, armed_box,
        text="BRAVE",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),  # serif 单字
        color=(15, 12, 10),
        weight_axis="Bold",  # 仅 BRAVE 大字用 Bold
    )

    # forces_box 擦干净不烧字（保持原图双行结构感）
    img_rgb = inpaint_box(img_rgb, forces_box, radius=15, pad=10)

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
    bat_dst = f"{in_job}/{prefix}_bat_logo_burned_v192.jpg"
    if os.path.exists(bat_src):
        burn_bat_logo(bat_src, bat_dst)
    else:
        print(f"[skip] bat_logo not found: {bat_src}")

    camo_src = f"{in_job}/{prefix}_camo_armed.jpg"
    camo_dst = f"{in_job}/{prefix}_camo_armed_burned_v192.jpg"
    if os.path.exists(camo_src):
        burn_camo_armed(camo_src, camo_dst)
    else:
        print(f"[skip] camo_armed not found: {camo_src}")
