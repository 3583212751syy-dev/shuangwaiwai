"""
v265_bat_fix — 纯 PIL 矢量重绘底图 + 原蝙蝠几何裂变 + Bodoni 精确排版

回应用户三点批评:
  1. 旧字没清干净 -> 底图完全由几何图形重绘,不依赖 inpaint,零 ghost
  2. 主体裂变感没拉开 -> 用原蝙蝠剪影做透视/仿射形变(up/spread/fold),差异可见
  3. 新文字排版丑 -> 改用 Bodoni MT Bold(类原图 BACARDÍ 衬线)并按原文字位置/大小精确摆放

输出: jobs/v265/v265_{up,spread,fold}_final.png + _grid_v265.png + _compare_v265.png
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path("E:/Desktop/双接口/image-fission/src")))
from arc_text import draw_arc_text, fit_arc_text_width

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
JOB = PROJECT / "jobs" / "v265"
JOB.mkdir(parents=True, exist_ok=True)

SRC = COMFY_INPUT / "test_6978fabda2cc99629fa9e81f802762d3.jpg"

# 原图已验证的关键几何/排版常量(来自 layout.json / v253)
W, H = 1552, 2000
CX, CY = 776, 745
OUTER_R, INNER_R = 421, 407
BAT_BBOX = (522, 518, 508, 456)  # x,y,w,h
BAT_CX, BAT_CY = 776, 746
BIG_CENTER = (776, 1070)
SMALL_CENTER = (776, 1285)

BODONI_B = "C:/Windows/Fonts/BOD_B.TTF"
BODONI_R = "C:/Windows/Fonts/BOD_R.TTF"
INK = (26, 10, 31)  # 接近黑色的深紫


def sample_colors(bgr):
    """采样原图关键颜色."""
    h, w = bgr.shape[:2]
    ys, xs = np.mgrid[:h, :w]
    d = np.sqrt((xs - CX) ** 2 + (ys - CY) ** 2)

    # 背景: 远离圆环和蝙蝠的区域
    bg_mask = (d > OUTER_R + 60) & (d < OUTER_R + 200)
    # 限制到图像中心附近,避免边缘
    bg_mask &= (xs > w * 0.2) & (xs < w * 0.8) & (ys > h * 0.15) & (ys < h * 0.85)
    bg = np.median(bgr[bg_mask], axis=0).astype(int) if bg_mask.any() else np.array([183, 126, 171])

    # 内盘(浅色紫): 圆环内部、上半部分
    disc_mask = (d < INNER_R - 30) & (ys < CY + 50)
    disc = np.median(bgr[disc_mask], axis=0).astype(int) if disc_mask.any() else bg

    # 底部月牙(深色紫): 圆环内部、底部
    moon_mask = (d < INNER_R - 20) & (d > INNER_R - 130) & (ys > CY + 20)
    moon = np.median(bgr[moon_mask], axis=0).astype(int) if moon_mask.any() else bg * 0.8

    # 圆环线: 圆环本体
    ring_mask = (d >= INNER_R - 6) & (d <= OUTER_R + 6)
    ring = np.median(bgr[ring_mask], axis=0).astype(int) if ring_mask.any() else np.array([30, 10, 35])

    return tuple(int(x) for x in bg[::-1]), tuple(int(x) for x in disc[::-1]), \
           tuple(int(x) for x in moon[::-1]), tuple(int(x) for x in ring[::-1])


def extract_bat_silhouette(bgr):
    """提取原图蝙蝠剪影: 阈值选最暗区域,再用最大连通域分离蝙蝠."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 蝙蝠/文字/圆环都是深色;蝙蝠尤其黑,用低阈值区分内盘
    _, dark = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
    # 排除圆环外部(只保留内盘区域)
    ys, xs = np.mgrid[:h, :w]
    inside = (((xs - BAT_CX) ** 2 + (ys - BAT_CY) ** 2) < (INNER_R - 6) ** 2).astype(np.uint8) * 255
    dark = cv2.bitwise_and(dark, inside)
    # 闭合小孔(蝙蝠内部若有反光)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # 选最大连通域(蝙蝠)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if num <= 1:
        return dark, 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + np.argmax(areas)
    bat = (labels == largest).astype(np.uint8) * 255
    # 填充蝙蝠内部空洞(原图高光/反光)
    contours, _ = cv2.findContours(bat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bat = np.zeros_like(bat)
    cv2.drawContours(bat, contours, -1, 255, -1)
    # 轻微平滑边缘
    bat = cv2.morphologyEx(bat, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bat = cv2.morphologyEx(bat, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # 裁剪
    ys, xs = np.where(bat > 128)
    if len(xs) == 0:
        return bat, 0, 0
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    pad = 12
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    return bat[y1:y2 + 1, x1:x2 + 1], x1, y1


def make_clean_base(bgr, bg, disc, moon, ring_color):
    """从零绘制干净底图:背景+内盘+月牙+圆环."""
    h, w = bgr.shape[:2]
    base = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(base)

    # 内盘浅色圆
    draw.ellipse([CX - INNER_R, CY - INNER_R, CX + INNER_R, CY + INNER_R], fill=disc)

    # 底部月牙: 用一个偏移的深色椭圆与内盘做差效果,再用椭圆模拟
    # 月牙形状: 底部深色椭圆,上半部分被内盘色覆盖
    moon_r = INNER_R - 15
    my = CY + 25
    # 底部深色圆
    draw.ellipse([CX - moon_r, my - moon_r, CX + moon_r, my + moon_r], fill=moon)
    # 用内盘色切掉上半部分,形成月牙
    cut_r = INNER_R + 10
    cut_y = CY - 55
    draw.ellipse([CX - cut_r, cut_y - cut_r, CX + cut_r, cut_y + cut_r], fill=disc)

    # 重新画内盘边缘让过渡自然
    draw.ellipse([CX - INNER_R, CY - INNER_R, CX + INNER_R, CY + INNER_R], fill=disc)

    # 圆环: 外圆 + 内圆做差
    draw.ellipse([CX - OUTER_R, CY - OUTER_R, CX + OUTER_R, CY + OUTER_R], fill=ring_color)
    draw.ellipse([CX - INNER_R, CY - INNER_R, CX + INNER_R, CY + INNER_R], fill=disc)

    # 底部小三角装饰(原图有)
    tri_y = 1418
    draw.polygon([(CX - 70, tri_y), (CX + 70, tri_y), (CX, tri_y + 45)], fill=INK)

    return base


def warp_bat(crop_mask, tag):
    """对紧凑蝙蝠 mask 做轻微仿射形变(角度/翼展/收翼),保留蝙蝠识别性."""
    h, w = crop_mask.shape
    cx, cy = w / 2.0, h / 2.0

    if tag == "up":
        # 整体轻微上扬 + 纵向拉高
        angle, sx, sy = -8.0, 1.03, 1.14
    elif tag == "spread":
        # 翼展横向拉宽 + 略压扁
        angle, sx, sy = 0.0, 1.20, 0.90
    else:  # fold
        # 整体微右转 + 纵向压缩收翼
        angle, sx, sy = 8.0, 0.95, 1.06

    a = math.radians(angle)
    cos_a, sin_a = math.cos(a), math.sin(a)

    # 旋转 + 非均匀缩放,以中心为锚点
    m00, m01 = sx * cos_a, -sy * sin_a
    m10, m11 = sx * sin_a,  sy * cos_a
    tx = cx - m00 * cx - m01 * cy
    ty = cy - m10 * cx - m11 * cy
    M = np.float32([[m00, m01, tx], [m10, m11, ty]])

    # 输出画布加大 20% 避免翼尖被切
    out_w, out_h = int(w * 1.25), int(h * 1.25)
    ox, oy = (out_w - w) // 2, (out_h - h) // 2
    M2 = M.copy()
    M2[:, 2] += np.array([ox, oy])

    warped = cv2.warpAffine(crop_mask, M2, (out_w, out_h), borderValue=0,
                            flags=cv2.INTER_CUBIC)

    # 二值化并清理
    _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
    warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return warped


def silhouette_to_rgba(mask, color=INK):
    """把单通道 mask 转成 RGBA 剪影."""
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = mask
    return Image.fromarray(rgba, "RGBA")


def calibrate_font(text, target_w, font_path, lo=8, hi=350):
    """二分查找让文字宽度接近 target_w."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        bb = ImageDraw.Draw(Image.new("RGB", (10, 10))).textbbox((0, 0), text, font=f)
        if (bb[2] - bb[0]) < target_w:
            lo = mid
        else:
            hi = mid
    return lo


def burn_top_arc(img, text, font_path, font_size, color=INK):
    """顶部弧线文字,位于圆环上方,文字底部朝圆心."""
    radius = 460
    arc_len = fit_arc_text_width(text, font_path, font_size, radius, char_spacing_px=5)
    total_deg = math.degrees(arc_len / radius)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    cy = CY + 120
    return draw_arc_text(img, text, font_path, font_size, color,
                         (CX, cy), radius, start, end,
                         char_spacing_px=5, flip_180=False)


def burn_text_layout(img, big, arc, sub):
    """按原图文字位置大小绘制新文字."""
    img = img.convert("RGBA")

    # 1) 顶部弧线(跨度约 110°,半径 460,弧长约 880 px)
    fs_arc = calibrate_font(arc, 880, BODONI_R, lo=20, hi=110)
    img = burn_top_arc(img, arc, BODONI_R, fs_arc, INK)

    # 2) 中心大词 (原 BACARDÍ 位置/大小)
    fs_big = calibrate_font(big, 940, BODONI_B, lo=60, hi=260)
    draw = ImageDraw.Draw(img)
    draw.text(BIG_CENTER, big, font=ImageFont.truetype(BODONI_B, fs_big), fill=INK, anchor="mm")

    # 3) 底部副词 (原 MHEART 位置/大小)
    fs_sub = calibrate_font(sub, 680, BODONI_B, lo=40, hi=180)
    draw.text(SMALL_CENTER, sub, font=ImageFont.truetype(BODONI_B, fs_sub), fill=INK, anchor="mm")

    # 4) 两侧 EST. / 1862 (原图左右侧位置)
    fs_est = calibrate_font("1862", 110, BODONI_R, lo=12, hi=60)
    f_est = ImageFont.truetype(BODONI_R, fs_est)
    # 原图 Est. 在左侧约 x=0.155, y=0.40; 1862 在右侧约 x=0.845
    draw.text((int(W * 0.16), int(H * 0.405)), "Est.", font=f_est, fill=INK, anchor="mm")
    draw.text((int(W * 0.84), int(H * 0.405)), "1862", font=f_est, fill=INK, anchor="mm")

    return img.convert("RGB")


def make_variant(tag, big, arc, sub):
    """生成一个蝙蝠裂变变体."""
    orig = Image.open(SRC).convert("RGB")
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    bg, disc, moon, ring_color = sample_colors(bgr)

    # 干净底图
    base = make_clean_base(bgr, bg, disc, moon, ring_color)

    # 提取紧凑蝙蝠并形变
    bat_crop, _, _ = extract_bat_silhouette(bgr)
    warped = warp_bat(bat_crop, tag)

    # 转成 RGBA 并居中贴到底图
    bat_rgba = silhouette_to_rgba(warped, INK)
    paste_x = BAT_CX - bat_rgba.width // 2
    paste_y = BAT_CY - bat_rgba.height // 2
    base.paste(bat_rgba, (paste_x, paste_y), bat_rgba)

    # 加文字
    final = burn_text_layout(base, big, arc, sub)
    return final


SUBJECT_VARIANTS = [
    ("up", "NIGHTBAT", "SHADOW OF THE WING", "ECHO HUNT"),
    ("spread", "DUSKBAT", "WINGS OF TWILIGHT", "SILENT FLIGHT"),
    ("fold", "MOONBAT", "GUARDIAN OF THE DARK", "SONAR OATH"),
]


def main():
    finals = []
    for tag, big, arc, sub in SUBJECT_VARIANTS:
        print(f"[v265] building {tag} ...")
        img = make_variant(tag, big, arc, sub)
        out = JOB / f"v265_{tag}_final.png"
        img.save(str(out), quality=95)
        finals.append(out)
        print(f"  saved {out}")

    # 三变体并排
    imgs = [Image.open(p).convert("RGB") for p in finals]
    gap = 14
    grid = Image.new("RGB", (W * 3 + gap * 4, H), "white")
    for i, im in enumerate(imgs):
        grid.paste(im, (gap + i * (W + gap), 0))
    grid.save(str(JOB / "_grid_v265.png"), quality=92)

    # 原图 vs 三变体 2x2
    orig = Image.open(SRC).convert("RGB").resize((int(W * 0.55), int(H * 0.55)), Image.LANCZOS)
    thumbs = [im.resize((int(W * 0.55), int(H * 0.55)), Image.LANCZOS) for im in imgs]
    cell_w, cell_h = thumbs[0].size
    margin = 16
    cw = cell_w * 2 + margin * 3
    ch = cell_h * 2 + margin * 3 + 50
    comp = Image.new("RGB", (cw, ch), (245, 245, 245))
    draw = ImageDraw.Draw(comp)
    try:
        font = ImageFont.truetype(BODONI_R, 28)
    except Exception:
        font = ImageFont.load_default()
    titles = ["ORIGINAL", "NIGHTBAT (up)", "DUSKBAT (spread)", "MOONBAT (fold)"]
    panels = [orig] + thumbs
    pos = [(margin, margin),
           (cell_w + margin * 2, margin),
           (margin, cell_h + margin * 2),
           (cell_w + margin * 2, cell_h + margin * 2)]
    for title, im, (x, y) in zip(titles, panels, pos):
        comp.paste(im, (x, y))
        bb = draw.textbbox((0, 0), title, font=font)
        tw = bb[2] - bb[0]
        draw.text((x + (cell_w - tw) // 2, y + cell_h + 8), title, fill=(30, 30, 30), font=font)
    comp.save(str(JOB / "_compare_v265.png"), quality=92)

    print(f"[OK] grid -> {JOB / '_grid_v265.png'}")
    print(f"[OK] compare -> {JOB / '_compare_v265.png'}")


if __name__ == "__main__":
    main()
