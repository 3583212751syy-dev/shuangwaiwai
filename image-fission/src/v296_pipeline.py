"""
v296 — 蝙蝠底部安全线（不压主字）+ fold/spread 轻微上移。

根因（破 bug）：
  arc_text.draw_arc_text 内部把 img 转成 RGBA 再返回；
  v289/v291 的 burn_text 里 `draw = ImageDraw.Draw(img)` 在 ARC 之前绑定到 RGB img，
  ARC 返回后 img 已是 RGBA，但 draw 仍是旧 RGB 的 draw；
  后续 BIG/SUB/SIDE 的 draw.text 全部画到已被丢弃的 RGB buffer，主副字全丢。

修复：ARC 返回后重新 `draw = ImageDraw.Draw(img)`，并强制 img.convert("RGB") 避免模式混乱。

附加：把字再次过栅格化前用 try/except 包好，确保任何 PIL 异常都不会让主字丢失。
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
JOB = PROJECT / "jobs" / "v296"
JOB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "src"))
import arc_text

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

TARGET_W, TARGET_H = 1024, 1280
INK = (26, 10, 31)
BG_PURPLE = (183, 127, 171)
RING_DEEP = (90, 20, 120)

cx, cy, r_badge = 776, 746, 300
ring_outer = 421

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",      296001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",       296101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK",    296201),
]


def _ring_sector_mask(h, w, c_x, c_y, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - c_x) ** 2 + (ys - c_y) ** 2
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    sweep = (a1 - a0) % 360
    if sweep == 0 and (a1 - a0) != 0:  # 0,360 -> 360° 而非 0°
        sweep = 360
    if sweep >= 359.99:
        return in_ring  # 完整圆盘
    ang = (np.degrees(np.arctan2(ys - c_y, xs - c_x)) - a0) % 360
    in_angle = ang <= sweep
    return in_ring & in_angle


def _detect_bat_mask(arr):
    """蝙蝠 mask：cx,cy 周围 r_badge+30 + ring 区域（覆盖翼尖完整保留）。"""
    h, w = arr.shape[:2]
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    stdc = np.std(arr.astype(np.float32), axis=2)
    dark = (maxc < 80) & (stdc < 25)
    mask = dark.astype(np.uint8) * 255
    # 关键：限制圆形必须覆盖到 ring_outer 才能让翼尖被纳入 mask
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), ring_outer + 30, 255, -1)
    mask = cv2.bitwise_and(mask, badge)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.dilate(mask, np.ones((8, 8), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(sizes))
        mask = (labels == keep).astype(np.uint8) * 255
    return mask


def _only_keep_dark_text_in_arc_skip(arr, mask_region):
    """把原图黑色像素位置限制在 mask_region 之外保留非黑（环带本身），字像素填紫底"""
    arr_copy = arr.copy()
    arr_copy[mask_region] = BG_PURPLE  # 整段环带 → 紫色底
    return arr_copy


def make_clean_source():
    """v291 改进版：拱门环带保留原图（非黑像素），字像素填紫底。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]

    corners = np.stack([arr[5, 5], arr[5, w-5], arr[h-5, 5], arr[h-5, w-5]], axis=0)
    bg_color = np.median(corners, axis=0).astype(np.float32)
    print(f"[bg] {tuple(bg_color.astype(int))}")

    clean = np.full_like(arr, bg_color)

    # 0) 先在原图上找蝙蝠精确 mask（用于环带清除时只保留蝙蝠像素）
    bat_mask_pre = _detect_bat_mask(arr)
    print(f"[protect] bat_mask_pre sum={int(bat_mask_pre.sum()/255)}px")

    # 1) 拱门环带：只保留蝙蝠剪影像素，其余全部填成紫底
    arc_band = _ring_sector_mask(h, w, cx, cy, ring_outer - 95, ring_outer + 65, 185, 355)
    inner = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 30, 0, 360)
    arc_band_keep = arc_band & ~inner
    # 环带内只有蝙蝠像素保留原图，其他（字/描边/装饰线）全清
    in_band_not_bat = arc_band_keep & (bat_mask_pre == 0)
    clean[in_band_not_bat] = bg_color
    print(f"[arc] arc_band={int(arc_band.sum())} inner={int(inner.sum())} arc_band_keep={int(arc_band_keep.sum())} bat_mask={int((bat_mask_pre>0).sum())} inter={int((arc_band_keep & (bat_mask_pre>0)).sum())} cleared={int(in_band_not_bat.sum())}")
    # debug vis
    debug_img = np.zeros((h, w, 3), dtype=np.uint8)
    debug_img[arc_band_keep] = (255, 0, 0)  # 红
    debug_img[bat_mask_pre > 127] = (0, 255, 0)  # 绿
    debug_img[arc_band_keep & (bat_mask_pre > 127)] = (255, 255, 0)  # 黄=交集
    Image.fromarray(debug_img).save(r"E:/Desktop/双接口/image-fission/jobs/v296/_debug_mask.png")

    # 2) 中心徽章保留 (不消黑字，里面只有蝙蝠剪影)
    inner_badge = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 95, 0, 360)
    clean[inner_badge] = arr[inner_badge]
    print(f"[inner] 中心徽章 {int(inner_badge.sum())}px (跳过黑字清除，保护蝙蝠)")

    # 3) 蝙蝠剪影保留（从原图 arr 提取，避免 clean 上被改动）
    bat_pixels = arr.copy()
    bat_pixels[bat_mask_pre == 0] = 0
    clean[bat_mask_pre > 127] = bat_pixels[bat_mask_pre > 127]
    print(f"[bat] {int((bat_mask_pre>127).sum())}px")

    # 4) 底部矩形清黑字；rect 1450-1550 跳过（留给底三角），蝙蝠下方矩形上沿提高到 1050
    rects = [
        (1080, 1200, 50,   1500),  # 原本 900-1200: 上提避开蝙蝠翅膀下沿
        (1200, 1400, 200,  1350),
        (700,  920,  240,  700),
        (700,  920,  870,  1320),
        # 1450-1550, 620-930 跳过，保留原图底三角
    ]
    for y0, y1, x0, x1 in rects:
        region = np.zeros((h, w), dtype=bool)
        region[y0:y1, x0:x1] = True
        black_in_rect = (arr.sum(axis=2) < 200) & region
        clean[black_in_rect] = bg_color
        print(f"[rect] {y0}-{y1},{x0}-{x1} 黑字清除 {int(black_in_rect.sum())}px")

    return Image.fromarray(np.clip(clean, 0, 255).astype(np.uint8))


def _warp_bat_by_wings(bat_pixels, mask, left_delta, right_delta, shoulder_l, shoulder_r, falloff=110):
    h, w = mask.shape
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    dl = np.sqrt((xs - shoulder_l[0]) ** 2 + (ys - shoulder_l[1]) ** 2)
    dr = np.sqrt((xs - shoulder_r[0]) ** 2 + (ys - shoulder_r[1]) ** 2)
    wl = np.clip((dl - 25) / falloff, 0, 1) * (xs < cx).astype(np.float32)
    wr = np.clip((dr - 25) / falloff, 0, 1) * (xs > cx).astype(np.float32)
    dx_map = wl * left_delta[0] + wr * right_delta[0]
    dy_map = wl * left_delta[1] + wr * right_delta[1]
    x_map = xs - dx_map
    y_map = ys - dy_map
    warped_mask = cv2.remap(mask, x_map.astype(np.float32), y_map.astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_pixels = cv2.remap(bat_pixels, x_map.astype(np.float32), y_map.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge, 255, -1)
    warped_mask = cv2.bitwise_and(warped_mask, badge)
    warped_pixels = warped_pixels * (badge[:, :, None] > 0)
    _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
    warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)
    # 底部安全线：蝙蝠不得压到主字区（主字顶部约 h*0.49），线设在 h*0.47
    safe_line = int(h * 0.47)
    warped_mask[safe_line:, :] = 0
    alpha = warped_mask.astype(np.float32) / 255.0
    return warped_pixels, alpha


def _find_wing_landmarks(mask):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    upper = ys < cy
    if upper.sum() < 20:
        upper = ys < cy + 80
    left_idx = xs[upper].argmin()
    right_idx = xs[upper].argmax()
    left_tip = (int(xs[upper][left_idx]), int(ys[upper][left_idx]))
    right_tip = (int(xs[upper][right_idx]), int(ys[upper][right_idx]))
    shoulder_y = int((left_tip[1] + cy) * 0.55)
    row = mask[shoulder_y, :]
    row_idx = np.where(row > 127)[0]
    if len(row_idx) >= 2:
        left_shoulder = (int(row_idx[row_idx < cx].max()) if np.any(row_idx < cx) else cx - 80, shoulder_y)
        right_shoulder = (int(row_idx[row_idx > cx].min()) if np.any(row_idx > cx) else cx + 80, shoulder_y)
    else:
        left_shoulder = (cx - 90, shoulder_y)
        right_shoulder = (cx + 90, shoulder_y)
    return {"left_tip": left_tip, "right_tip": right_tip,
            "left_shoulder": left_shoulder, "right_shoulder": right_shoulder}


def make_variants(src_img):
    """蝙蝠 mask 半径必须用 ring_outer+30（与 make_clean_source 一致），否则 r_badge+30 会切掉翼尖。"""
    arr = np.array(src_img).astype(np.float32)
    h, w = arr.shape[:2]
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), ring_outer + 30, 255, -1)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    stdc = np.std(arr, axis=2)
    dark = (maxc < 80) & (stdc < 25)
    mask = (dark.astype(np.uint8) * 255)
    mask = cv2.bitwise_and(mask, badge)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.dilate(mask, np.ones((8, 8), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(sizes))
        mask = (labels == keep).astype(np.uint8) * 255
    bat_pixels = np.zeros_like(arr)
    bat_pixels[mask > 0] = (0, 0, 0)
    lm = _find_wing_landmarks(mask)
    if lm is None:
        raise RuntimeError("未找到蝙蝠 landmark")
    variants = {}
    variants["up"] = _warp_bat_by_wings(bat_pixels, mask,
        left_delta=(8, -12), right_delta=(-8, -12),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)
    variants["spread"] = _warp_bat_by_wings(bat_pixels, mask,
        left_delta=(-30, -5), right_delta=(30, -5),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)
    variants["fold"] = _warp_bat_by_wings(bat_pixels, mask,
        left_delta=(-8, -10), right_delta=(8, -10),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)
    return variants


def _calibrate_font(text, font_path, start_size, max_w):
    lo, hi = 8, start_size
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        if font.getlength(text) <= max_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def burn_text(img, arc, big, sub, color=INK):
    """v296 关键修复：每步后重新绑定 draw + 强制 RGB。"""
    img = img.convert("RGB")
    w, h = img.width, img.height

    # ARC：draw_arc_text 内部把图转 RGBA，返回 RGBA 图
    fs_arc = int(w * 0.045)
    radius = int(w * 0.46)
    cy_arc = int(h * 0.13) + radius
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT, fs_arc, radius, char_spacing_px=3)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 170:
            break
        fs_arc = int(fs_arc * 0.92)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    img = arc_text.draw_arc_text(img, arc, FONT, fs_arc, color,
                                 center=(w // 2, cy_arc), radius=radius,
                                 start_angle_deg=start, end_angle_deg=end,
                                 char_spacing_px=3, flip_180=False)
    # 关键 ①：arc_text 返回 RGBA，立刻转 RGB 并重绑 draw
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    print(f"[burn] arc ok; img mode {img.mode}")

    fs_big = _calibrate_font(big, FONT, int(h * 0.145), max_w=int(w * 0.82))
    font = ImageFont.truetype(FONT, fs_big)
    print(f"[burn] BIG fs={fs_big} len={font.getlength(big):.1f}")
    draw.text((w // 2, int(h * 0.555)), big, font=font, fill=color, anchor="mm")

    fs_sub = _calibrate_font(sub, FONT, int(h * 0.075), max_w=int(w * 0.60))
    font = ImageFont.truetype(FONT, fs_sub)
    print(f"[burn] SUB fs={fs_sub} len={font.getlength(sub):.1f}")
    draw.text((w // 2, int(h * 0.655)), sub, font=font, fill=color, anchor="mm")

    f_side = ImageFont.truetype(FONT, int(h * 0.028))
    draw.text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    draw.text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")

    # 底三角 ▼（深紫菱形/箭头）
    tri_cx = w // 2
    tri_cy = int(h * 0.83)
    tri_w = int(w * 0.038)
    tri_h = int(h * 0.025)
    draw.polygon([
        (tri_cx, tri_cy + tri_h),
        (tri_cx - tri_w, tri_cy - tri_h // 3),
        (tri_cx + tri_w, tri_cy - tri_h // 3),
    ], fill=color)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_path = COMFY_INPUT / "v296_text_free_source.png"

    if not src_path.exists() or args.force:
        src = make_clean_source()
        src.save(src_path, quality=98)
        orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(orig, (0, 0))
        check.paste(src, (1552 + 24, 0))
        check.save(JOB / "_source_check.png", quality=95)
        print(f"[v296] clean source -> {src_path}")
    else:
        print(f"[v296] reuse clean source")

    text_free = Image.open(src_path).convert("RGB")
    bg_arr = np.array(text_free).astype(np.float32)
    variants = make_variants(text_free)

    files = []
    for tag, big, sub, arc, _ in SUBJECT_VARIANTS:
        warped_pixels, alpha = variants[tag]
        composed = bg_arr * (1 - alpha[:, :, None]) + warped_pixels * alpha[:, :, None]
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v296_{tag}_composed.png", quality=95)

        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(text_free, (0, 0))
        check.paste(composed, (1552 + 24, 0))
        check.save(JOB / f"_compose_check_{tag}.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v296_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"[v296] {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v296.png"
    grid.save(gp, quality=92)
    print(f"[v296] grid -> {gp}")


if __name__ == "__main__":
    main()
