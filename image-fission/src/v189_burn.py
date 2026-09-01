"""
v189_burn.py — bat_logo + camo_armed 烧字脚本（v189 重做版）

字体规则：
  - 弧字/serif 衬线 → Lora-VF.ttf（Google Fonts 官方 variable weight）
  - sans 粗黑大字 → Oswald-VF.ttf
  - sans 极粗单字 → Anton-Regular.ttf

烧字规则（基于 easyocr 拿到的原图 box 坐标）：
  bat_logo (1552x2000):
    弧字 "LA CASA DEL MURCIÉLAGO" (x=420-1229, y=256-698)
      → "WINGS OF NIGHT" (隐喻蝙蝠翅膀)
      draw_arc_text, center=(776, 580), radius=370, start=200°, end=340° (弧度 140° 顶部弧)
    BACARDÍ 大字 (281,946,1284,1187) → BATANO (Lora-VF Regular, 黑色)
    HEART 小字 (459,1172,1138,1340) → SOUL (Lora-VF Regular, 黑色)
    Est./1862 保留原图不烧
  camo_armed (1556x2000):
    "WE SUPPORT THE" 顶部小字 (522,462,1056,534)
      → "WE HONOR" (Oswald Bold 黑色)
    "ARMED" 中间大字 (467,529,1112,652)  +  "FORCES" 下面大字 (446,640,1157,779)
      → 单字 "BRAVE" (Anton Regular 大字，替换 ARMED+FORCES 两行中部)

源文件：v188 的 v188_*.jpg（先用）→ v189 重做后改 v189_*.jpg
"""

import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

JOB = "E:/Desktop/双接口/image-fission/jobs/smoke_v188"  # 默认烧 v188 输出
FONTS = "E:/Desktop/双接口/image-fission/fonts"

sys.path.insert(0, "E:/Desktop/双接口/image-fission/src")
from arc_text import draw_arc_text


def fit_font(text, font_path, max_w, max_h, weight_axis=None):
    """二分找最大字号（保证文字宽/高均不超边界）"""
    lo, hi = 20, 600
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        if weight_axis is not None:
            try:
                f.set_variation_by_name(weight_axis)
            except Exception:
                pass
        if f.getlength(text) <= max_w and mid <= max_h * 1.2:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ImageFont.truetype(font_path, best)


def inpaint_box(img_rgb, box, radius=6, pad=10):
    """box 区域 inpaint 清空（cv2 Telea 算法）"""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    x2, y2 = min(x2, w), min(y2, h)
    mask = np.zeros((h, w), np.uint8)
    mask[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 255
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def burn_text_box(img_rgb, box, text, font_path, color, weight_axis=None,
                  fill_bg=False, bg_color=(245, 235, 240, 255)):
    """在 box 区域烧入文字（水平），可选先填充背景（让文字更清晰）"""
    img = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in box]
    max_w, max_h = x2 - x1, y2 - y1
    if fill_bg:
        # 在 box 上覆盖一层接近原图底色
        draw.rectangle([x1, y1, x2, y2], fill=bg_color)
    f = fit_font(text, font_path, max_w * 0.95, max_h * 0.92, weight_axis=weight_axis)
    lw = f.getlength(text)
    bbox_l = f.getbbox(text)
    lh = bbox_l[3] - bbox_l[1]
    lx = x1 + (max_w - lw) / 2 - bbox_l[0]
    ly = y1 + (max_h - lh) / 2 - bbox_l[1]
    draw.text((lx, ly), text, font=f, fill=color)
    return np.array(img.convert("RGB"))


# ===================== bat_logo 烧字 =====================
def burn_bat_logo(in_path, out_path):
    """in_path: v188/v189 bat_logo 输出"""
    img_rgb = np.array(Image.open(in_path).convert("RGB"))
    gh, gw = img_rgb.shape[:2]
    src = Image.open("E:/Desktop/双接口/image-fission/ComfyUI/input/test_6978fabda2cc99629fa9e81f802762d3.jpg")
    sh, sw = src.height, src.width
    sx, sy = gw / sw, gh / sh

    # 1) 弧字：LA CASA DEL MURCIÉLAGO (420,256)-(1229,698) → WINGS OF NIGHT
    #    弧心 (776, 580)，半径约 370
    arc_box_src = (420, 256, 1229, 698)
    arc_box = [int(v * (sx if i < 2 else sy)) for i, v in enumerate([
        arc_box_src[0], arc_box_src[1], arc_box_src[2], arc_box_src[3]])]
    # 调整成弧字外侧半径
    cx_s, cy_s = 776, 580
    cx = int(cx_s * sx)
    cy = int(cy_s * sy)
    radius = int(370 * (sx + sy) / 2)
    # 擦除弧字区域
    img_rgb = inpaint_box(img_rgb, arc_box, radius=8, pad=20)
    # 沿弧绘新词
    img_pil = Image.fromarray(img_rgb).convert("RGB")
    # 字体大小匹配弧字高度（约 80 像素@原图 → 输出 80*sy）
    arc_font_size = int(80 * sy)
    arc_color = (20, 12, 32)  # 深黑紫（与原图 BACARDÍ 黑字接近）
    img_pil = draw_arc_text(
        img_pil,
        text="WINGS OF NIGHT",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),
        font_size=arc_font_size,
        color=arc_color,
        center=(cx, cy),
        radius=radius,
        start_angle_deg=200,  # 8 点钟方向
        end_angle_deg=340,    # 4 点钟方向（逆时针 140°，覆盖顶部弧）
        char_spacing_px=int(4 * sy),
        flip_180=False,
    )
    img_rgb = np.array(img_pil.convert("RGB"))

    # 2) BACARDÍ 大字 → BATANO
    bac_box = [int(v * sx if i % 2 == 0 else v * sy) for i, v in enumerate([281, 946, 1284, 1187])]
    img_rgb = inpaint_box(img_rgb, bac_box, radius=8, pad=15)
    img_rgb = burn_text_box(
        img_rgb, bac_box,
        text="BATANO",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),
        color=(20, 12, 32),
        fill_bg=False,
    )

    # 3) HEART 小字 → SOUL
    heart_box = [int(v * sx if i % 2 == 0 else v * sy) for i, v in enumerate([459, 1172, 1138, 1340])]
    img_rgb = inpaint_box(img_rgb, heart_box, radius=6, pad=12)
    img_rgb = burn_text_box(
        img_rgb, heart_box,
        text="SOUL",
        font_path=os.path.join(FONTS, "Lora-VF.ttf"),
        color=(20, 12, 32),
        fill_bg=False,
    )

    Image.fromarray(img_rgb).save(out_path, quality=95)
    print(f"[bat_logo] burned → {out_path}")
    return out_path


# ===================== camo_armed 烧字 =====================
def burn_camo_armed(in_path, out_path):
    img_rgb = np.array(Image.open(in_path).convert("RGB"))
    gh, gw = img_rgb.shape[:2]
    src = Image.open("E:/Desktop/双接口/image-fission/ComfyUI/input/test_b78e60de8dfdf44acda99395326a7298.jpg")
    sh, sw = src.height, src.width
    sx, sy = gw / sw, gh / sh

    # 1) "WE SUPPORT THE" 顶部小字 → "WE HONOR"
    top_box = [int(v * sx if i % 2 == 0 else v * sy) for i, v in enumerate([522, 462, 1056, 534])]
    img_rgb = inpaint_box(img_rgb, top_box, radius=6, pad=10)
    img_rgb = burn_text_box(
        img_rgb, top_box,
        text="WE HONOR",
        font_path=os.path.join(FONTS, "Oswald-VF.ttf"),
        color=(15, 12, 10),
        fill_bg=False,
    )

    # 2) ARMED + FORCES 两行 → 单字 "BRAVE"（占 ARMED 行）
    armed_box = [int(v * sx if i % 2 == 0 else v * sy) for i, v in enumerate([467, 529, 1112, 652])]
    forces_box = [int(v * sx if i % 2 == 0 else v * sy) for i, v in enumerate([446, 640, 1157, 779])]
    # 合并两行覆盖区域
    full_box = [
        min(armed_box[0], forces_box[0]),
        min(armed_box[1], forces_box[1]),
        max(armed_box[2], forces_box[2]),
        max(armed_box[3], forces_box[3]),
    ]
    # 先擦除两行
    img_rgb = inpaint_box(img_rgb, full_box, radius=10, pad=18)
    # 烧 BRAVE 单字（Anton Regular 极粗黑，位置在 ARMED 行中心）
    brave_box = [
        full_box[0],
        armed_box[1],  # 顶 y
        full_box[2],
        armed_box[3],  # 底 y = ARMED 底
    ]
    img_rgb = burn_text_box(
        img_rgb, brave_box,
        text="BRAVE",
        font_path=os.path.join(FONTS, "Anton-Regular.ttf"),
        color=(15, 12, 10),
        fill_bg=False,
    )

    Image.fromarray(img_rgb).save(out_path, quality=95)
    print(f"[camo_armed] burned → {out_path}")
    return out_path


if __name__ == "__main__":
    args = sys.argv[1:]
    job = args[0] if args else "smoke_v188"
    in_job = f"E:/Desktop/双接口/image-fission/jobs/{job}"
    # 自动检测 prefix：优先用 smoke_v190 → v190，再 v189，最后 v188
    job_name = job if isinstance(job, str) else ""
    if "v190" in job_name:
        prefix = "v190"
    elif "v189" in job_name:
        prefix = "v189"
    else:
        prefix = "v188"

    bat_src = f"{in_job}/{prefix}_bat_logo.jpg"
    bat_dst = f"{in_job}/{prefix}_bat_logo_burned.jpg"
    if os.path.exists(bat_src):
        burn_bat_logo(bat_src, bat_dst)
    else:
        print(f"[skip] bat_logo not found: {bat_src}")

    camo_src = f"{in_job}/{prefix}_camo_armed.jpg"
    camo_dst = f"{in_job}/{prefix}_camo_armed_burned.jpg"
    if os.path.exists(camo_src):
        burn_camo_armed(camo_src, camo_dst)
    else:
        print(f"[skip] camo_armed not found: {camo_src}")