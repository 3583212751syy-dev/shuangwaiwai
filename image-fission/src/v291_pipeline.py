"""
v291 — 彻底回退：整张图填干净紫底（不 inpaint 也不色块矩形），保留蝙蝠+拱门 mask。

核心策略：
  1. 取原图四角中值 -> 干净紫底 (BG_PURPLE)
  2. 取蝙蝠 mask（深紫剪影、连体三件套）像素，保持原样
  3. 取拱门圆环像素（深紫环带），保持原样
  4. 其余区域全填紫底 ⇒ 没有字、没有色块横痕、没有 inpaint 涂抹
  5. 三种仿射形变蝙蝠（在原蝙蝠轮廓基础上 up/spread/fold）
  6. 烧新字 → 干净紫底上 Didone 大字，无色块遮盖

用户硬规则全部满足：
  - 主体不许缺失：蝙蝠剪影完整保留
  - 新文本不许色块遮挡：紫底干净，新字直接覆盖
  - 配色锁原图色相：BG_PURPLE (183,127,171) 与原图一致
  - 无侵权 logo：变体用 NOCTAVEN/DUSKBAT/MOONBAT 原创品牌名
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
JOB = PROJECT / "jobs" / "v291"
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
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",      291001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",       291101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK",    291201),
]


def _ring_sector_mask(h, w, c_x, c_y, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - c_x) ** 2 + (ys - c_y) ** 2
    ang = (np.degrees(np.arctan2(ys - c_y, xs - c_x)) - a0) % 360
    sweep = (a1 - a0) % 360
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    in_angle = ang <= sweep
    return in_ring & in_angle


def _detect_bat_mask(arr):
    """蝙蝠 mask：拱门圆环内 + 深紫剪影连通域。"""
    h, w = arr.shape[:2]
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    stdc = np.std(arr.astype(np.float32), axis=2)
    dark = (maxc < 80) & (stdc < 25)
    mask = dark.astype(np.uint8) * 255
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge + 30, 255, -1)
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


def make_clean_source():
    """整张图填紫底（取原图四角中值），蝙蝠+拱门保留原图。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]

    # 1. 背景：原图四角取中值 = 纯紫底
    corners = np.stack([arr[5, 5], arr[5, w-5], arr[h-5, 5], arr[h-5, w-5]], axis=0)
    bg_color = np.median(corners, axis=0).astype(np.float32)
    print(f"[bg] {tuple(bg_color.astype(int))}")

    clean = np.full_like(arr, bg_color)

    # 2. 拱门圆环带 (ring band) 像素保留（深紫色环带）
    arc_band = _ring_sector_mask(h, w, cx, cy, ring_outer - 95, ring_outer + 65, 185, 355)
    inner = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 30, 0, 360)
    arc_band_keep = (arc_band & ~inner) & ~(arr.sum(axis=2) < 100)  # 排除已经是黑字的位置
    clean[arc_band_keep] = arr[arc_band_keep]
    print(f"[arc] 保留 {int(arc_band_keep.sum())}px 拱门环带")

    # 3. 拱门中心圆徽章（紫色圆形区域）
    inner_badge = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 95, 0, 360)
    clean[inner_badge] = arr[inner_badge]
    print(f"[inner] 保留 {int(inner_badge.sum())}px 中心徽章")

    # 4. 蝙蝠剪影：完整保留
    bat_mask = _detect_bat_mask(arr)
    bat_pixels = arr.copy()
    bat_pixels[bat_mask == 0] = 0
    clean[bat_mask > 127] = bat_pixels[bat_mask > 127]
    print(f"[bat] 保留 {int((bat_mask>127).sum())}px 蝙蝠剪影")

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
    """从干净源提取蝙蝠，三种姿态形变。"""
    arr = np.array(src_img).astype(np.float32)
    h, w = arr.shape[:2]
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge + 30, 255, -1)
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
        left_delta=(10, -55), right_delta=(-10, -55),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)
    variants["spread"] = _warp_bat_by_wings(bat_pixels, mask,
        left_delta=(-55, -15), right_delta=(55, -15),
        shoulder_l=lm["left_shoulder"], shoulder_r=lm["right_shoulder"], falloff=110)
    variants["fold"] = _warp_bat_by_wings(bat_pixels, mask,
        left_delta=(-15, 55), right_delta=(15, 55),
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
    img = img.convert("RGB")
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img)

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

    fs_big = _calibrate_font(big, FONT, int(h * 0.145), max_w=int(w * 0.82))
    font = ImageFont.truetype(FONT, fs_big)
    draw.text((w // 2, int(h * 0.555)), big, font=font, fill=color, anchor="mm")

    fs_sub = _calibrate_font(sub, FONT, int(h * 0.075), max_w=int(w * 0.60))
    font = ImageFont.truetype(FONT, fs_sub)
    draw.text((w // 2, int(h * 0.655)), sub, font=font, fill=color, anchor="mm")

    f_side = ImageFont.truetype(FONT, int(h * 0.028))
    draw.text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    draw.text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_path = COMFY_INPUT / "v291_text_free_source.png"

    if not src_path.exists() or args.force:
        src = make_clean_source()
        src.save(src_path, quality=98)
        orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(orig, (0, 0))
        check.paste(src, (1552 + 24, 0))
        check.save(JOB / "_source_check.png", quality=95)
        print(f"[v291] clean source -> {src_path}")
    else:
        print(f"[v291] reuse clean source")

    text_free = Image.open(src_path).convert("RGB")
    bg_arr = np.array(text_free).astype(np.float32)
    variants = make_variants(text_free)

    files = []
    for tag, big, sub, arc, _ in SUBJECT_VARIANTS:
        warped_pixels, alpha = variants[tag]
        composed = bg_arr * (1 - alpha[:, :, None]) + warped_pixels * alpha[:, :, None]
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v291_{tag}_composed.png", quality=95)

        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(text_free, (0, 0))
        check.paste(composed, (1552 + 24, 0))
        check.save(JOB / f"_compose_check_{tag}.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v291_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"[v291] {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v291.png"
    grid.save(gp, quality=92)
    print(f"[v291] grid -> {gp}")


if __name__ == "__main__":
    main()
