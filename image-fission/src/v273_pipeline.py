"""v273 (PIL 裂变版) — 可靠三阶段：PIL 蝙蝠变换 + cv2 清字 + AbrilFatface 烧字

修正 v269~v273 反复失败的 SDXL 生蝙蝠路线：
  - Stage A 裂变：对原蝙蝠做 PIL 变换（镜像/旋转/缩放），3 变体姿态明显不同（满足"裂变"），
            且 100% 保留用户认可的蝙蝠主体（v147/v253-fallback 同款可靠路线）
  - Stage B 清字：cv2.inpaint(INPAINT_NS) 从周围像素修补，绝不重建旧字（模型重绘会重建 BACARDI）
  - Stage C 烧字：AbrilFatface 矢量原位烧新字，顶弧用 v253 大半径(0.45w)+180°对称+char_spacing=8（修"糊"）

流程：
  1. v253.clean_base 生成 style_ref（紫底圆环，蝙蝠擦除、无字）
  2. 提取原蝙蝠（带 alpha），按变体做 PIL 变换，复合回 style_ref -> stage_a 输出(1552x2000)
  3. cv2.inpaint 清字（整圈弧带+主副字+Est/年份）
  4. 缩 1024x1280，PIL AbrilFatface 原位烧新字
"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
JOB = PROJECT / "jobs" / "v273"
JOB.mkdir(parents=True, exist_ok=True)

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
TARGET_W, TARGET_H = 1024, 1280

FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

# 原图 1552x2000 上的蝙蝠 bbox（来自 v253）
BAT_BBOX = (522, 518, 508, 430)
BAT_CENTER = (776, 746)

# 3 个变体：PIL 变换 + 新文案(顶弧品牌 / 中心强调词 / 底部标语)
SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",  "rotate", 8, 1.00),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",   "mirror", 0, 1.04),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK", "rotate", -8, 1.00),
]

INK = (28, 18, 30)

sys.path.insert(0, str(PROJECT / "src"))
import arc_text
spec = importlib.util.spec_from_file_location("v253", str(PROJECT / "src" / "v253_bat_logo_inpaint.py"))
v253 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v253)


def _sample_pure_bg(arr):
    """从四个角取背景色（避开文字和徽章），返回 RGB 均值。"""
    h, w = arr.shape[:2]
    corners = [
        arr[0:h//5, 0:w//5],
        arr[0:h//5, w*4//5:w],
        arr[h*4//5:h, 0:w//5],
        arr[h*4//5:h, w*4//5:w],
    ]
    return np.vstack([c.reshape(-1, 3) for c in corners]).mean(axis=0)


def _bat_badge_color(arr, bat_mask, cx, cy, r_inner=80, r_outer=180):
    """取蝙蝠周围环形区域的平均色（深色徽章色），用于填蝙蝠剪影。"""
    h, w = arr.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ring = (d2 >= r_inner ** 2) & (d2 <= r_outer ** 2) & (~bat_mask)
    if ring.sum() == 0:
        return arr.mean(axis=(0, 1))
    return arr[ring].mean(axis=0)


def make_clean_base_ref():
    """生成 style_ref：
    1) 弧带（含两侧小耳朵）/主副字/Est 区用纯正背景色填平 -> 灭 ghost
    2) 蝙蝠剪影用周围徽章色填平（不用 cv2.inpaint，避免条纹）-> 保留深色徽章
    """
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]
    bg = _sample_pure_bg(arr)

    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    bat_mask = v253.extract_bat(bgr)

    def sector_mask(cx, cy, r_in, r_out, a0, a1):
        ys, xs = np.ogrid[:h, :w]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        an = (np.degrees(np.arctan2(ys - cy, xs - cx)) - a0) % 360
        sweep = (a1 - a0) % 360
        in_angle = an <= sweep
        in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
        return in_ring & in_angle

    # 弧带扇形 + 两侧小耳朵（ ribbon 的左右下垂块）
    arc_mask = sector_mask(776, 746, 250, 520, 185, 355)
    side_left = np.zeros((h, w), bool)
    side_left[360:470, 90:220] = True
    side_right = np.zeros((h, w), bool)
    side_right[360:470, 1330:1460] = True
    arc_mask = arc_mask | side_left | side_right

    # 弧带/主副字在浅紫背景 -> 填 bg；Est/1862 在深色徽章侧翼 -> 填 badge_color
    badge_color = _bat_badge_color(arr, bat_mask, 776, 746, r_inner=100, r_outer=220)
    arr[arc_mask] = bg
    arr[975:1175, 155:1395] = bg
    arr[1175:1345, 355:1200] = bg
    arr[725:875, 335:575] = badge_color
    arr[725:875, 985:1235] = badge_color

    # 矩形区小模糊去硬边
    for y0, y1, x0, x1 in [(975, 1175, 155, 1395), (1175, 1345, 355, 1200),
                           (725, 875, 335, 575), (725, 875, 985, 1235)]:
        arr[y0:y1, x0:x1] = cv2.GaussianBlur(arr[y0:y1, x0:x1], (7, 7), 0)

    # 蝙蝠剪影用徽章色填平
    arr[bat_mask] = badge_color

    return Image.fromarray(arr.astype(np.uint8))


def fission_bat_pil(variant):
    """PIL 蝙蝠变换：提取原蝙蝠(透明背景) -> 变换 -> 硬 alpha -> 盖回 style_ref。返回 1552x2000。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    style_ref = make_clean_base_ref()
    style_ref.save(COMFY_INPUT / "v273_style_ref.png", quality=98)
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    bat_mask = v253.extract_bat(bgr)

    bx, by, bw, bh = BAT_BBOX
    cmask = bat_mask[by:by + bh, bx:bx + bw]
    crop = orig.crop((bx, by, bx + bw, by + bh)).convert("RGBA")
    pa = np.array(crop)
    pa[..., 3] = (cmask.astype(np.uint8) * 255)
    bat = Image.fromarray(pa)  # 透明背景蝙蝠

    _, _, _, _, mode, angle, scale = [v for v in SUBJECT_VARIANTS if v[0] == variant][0]
    if mode == "mirror":
        bat = bat.transpose(Image.FLIP_LEFT_RIGHT)
    # 旋转前先给透明背景填上蝙蝠色，避免插值产生灰/紫 fringe
    ba = np.array(bat).astype(np.float32)
    fg0 = ba[..., 3:4] > 30
    ba[..., :3] = np.where(fg0, ba[..., :3], INK)  # 透明区填 INK
    bat = Image.fromarray(ba.astype(np.uint8))

    if angle != 0:
        bat = bat.rotate(angle, expand=True, resample=Image.BICUBIC)
    if scale != 1.0:
        bat = bat.resize((int(bat.width * scale), int(bat.height * scale)), Image.LANCZOS)

    # 硬 alpha：旋转后只保留真正蝙蝠区，其余透明
    ba = np.array(bat)
    alpha = ba[..., 3]
    fg = alpha > 80
    ba[..., :3] = np.where(fg[..., None], INK, ba[..., :3])
    ba[..., 3] = np.where(fg, 255, 0)
    bat = Image.fromarray(ba)

    base = style_ref.convert("RGBA")
    base.paste(bat, (BAT_CENTER[0] - bat.width // 2, BAT_CENTER[1] - bat.height // 2), bat)
    return base.convert("RGB")


def make_text_mask_1552():
    """1552x2000 文字区 mask：整圈弧带 + 主副字 + Est/年份。"""
    mask = Image.new("L", (1552, 2000), 0)
    d = ImageDraw.Draw(mask)
    outer = [(120, 380), (120, 450), (140, 600), (180, 750), (250, 880),
             (450, 950), (700, 980), (850, 980), (1100, 950), (1300, 880),
             (1370, 750), (1410, 600), (1430, 450), (1430, 380),
             (1330, 310), (1100, 270), (776, 255), (450, 270), (220, 310)]
    inner = [(600, 400), (776, 400), (950, 400),
             (1100, 520), (1100, 720), (1000, 800),
             (900, 850), (776, 900), (650, 850),
             (550, 800), (400, 720), (400, 520)]
    d.polygon(outer + inner[::-1], fill=255)
    d.rectangle([(160, 980), (1390, 1170)], fill=255)
    d.rectangle([(360, 1180), (1195, 1340)], fill=255)
    d.rectangle([(350, 740), (560, 860)], fill=255)
    d.rectangle([(1000, 740), (1220, 860)], fill=255)
    # 轻度膨胀+羽化；过度膨胀会让 cv2.inpaint 把边界旧字拉回来
    mask = mask.filter(ImageFilter.MaxFilter(13))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=6))
    return mask


def stage_b_cv2(src_pil):
    """本地 cv2.inpaint 清字：从周围像素修补，绝不重建旧字。输入/输出 1552x2000。"""
    orig = np.array(src_pil.convert("RGB").resize((1552, 2000), Image.LANCZOS))
    mask = np.array(make_text_mask_1552().convert("L"))
    mask_bin = (mask > 127).astype(np.uint8) * 255
    clean = cv2.inpaint(orig, mask_bin, 9, cv2.INPAINT_NS)
    return Image.fromarray(clean)


def calibrate_font(text, font_path, start_size, max_w):
    lo, hi, best = 8, start_size, start_size
    while lo <= hi:
        mid = (lo + hi) // 2
        w = ImageFont.truetype(font_path, mid).getlength(text)
        if w <= max_w:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def burn_text(img, big, arc, sub, color=INK):
    """顶弧用 v253 大 radius + 180° 对称排布 + char_spacing=8（修'糊'）。"""
    img = img.convert("RGB")
    w, h = img.width, img.height

    fs_arc = int(w * 0.045)
    radius = int(w * 0.45)
    cy = int(h * 0.13) + radius
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT, fs_arc, radius, char_spacing_px=3)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 170:
            break
        fs_arc = int(fs_arc * 0.92)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    img = arc_text.draw_arc_text(img, arc, FONT, fs_arc, color,
                                 center=(w // 2, cy), radius=radius,
                                 start_angle_deg=start, end_angle_deg=end,
                                 char_spacing_px=3, flip_180=False)

    fs_big = calibrate_font(big, FONT, int(h * 0.16), max_w=int(w * 0.80))
    ImageDraw.Draw(img).text((w // 2, int(h * 0.555)), big,
                             font=ImageFont.truetype(FONT, fs_big), fill=color, anchor="mm")

    fs_sub = calibrate_font(sub, FONT, int(h * 0.085), max_w=int(w * 0.62))
    ImageDraw.Draw(img).text((w // 2, int(h * 0.655)), sub,
                             font=ImageFont.truetype(FONT, fs_sub), fill=color, anchor="mm")

    f_side = ImageFont.truetype(FONT, int(h * 0.032))
    ImageDraw.Draw(img).text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    ImageDraw.Draw(img).text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")
    return img


def qc_ocr(path):
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        res = reader.readtext(np.array(Image.open(path).convert("RGB")))
        return [(t, round(c, 2)) for _, t, c in res]
    except Exception as e:
        return [("OCR_FAIL", str(e))]


def qc_sharpness(path):
    im = np.array(Image.open(path).convert("RGB").resize((1024, 1280)))
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    roi = gray[int(0.48 * 1280):int(0.70 * 1280), int(0.15 * 1024):int(0.85 * 1024)]
    return round(float(cv2.Laplacian(roi, cv2.CV_64F).var()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="跑完整流程")
    args = ap.parse_args()

    if not args.full:
        print("用法: --full")
        return

    print("[v273 PIL] 完整流程：Stage A(PIL蝙蝠变换) -> Stage B 清字 -> Stage C 烧字 ...")
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H), "white")
    grid.paste(Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H)), (0, 0))

    for tag, arc, big, sub, _, _, _ in SUBJECT_VARIANTS:
        # Stage A
        fission = fission_bat_pil(tag)
        fission.save(JOB / f"v273_{tag}_fission.png", quality=95)
        # Stage B 清字
        clean = stage_b_cv2(fission)
        clean.save(JOB / f"v273_{tag}_clean.png", quality=95)
        # Stage C 烧字（缩到 1024x1280）
        small = clean.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        final = burn_text(small, big, arc, sub)
        final_path = JOB / f"v273_{tag}_final.png"
        final.save(final_path, quality=95)
        print(f"  {tag} final -> {final_path.name}")

        old_res = qc_ocr(str(JOB / f"v273_{tag}_clean.png"))
        new_res = qc_ocr(str(final_path))
        print(f"    clean 旧字: {[t for t,c in old_res if c>0.35]}")
        print(f"    final 可读: {[t for t,c in new_res if c>0.35]}")
        print(f"    text sharpness: {qc_sharpness(str(final_path))}")

        if tag == "up":
            grid.paste(final, (TARGET_W + 24, 0))

    gp = JOB / "_grid_v273.png"
    grid.save(gp, quality=92)
    print(f"  grid -> {gp}")

    # 三变体对照
    var_grid = Image.new("RGB", (TARGET_W * 3 + 24, TARGET_H), "white")
    for i, tag in enumerate(["up", "spread", "fold"]):
        var_grid.paste(Image.open(JOB / f"v273_{tag}_final.png").convert("RGB"),
                       (i * (TARGET_W + 12), 0))
    vgp = JOB / "_variants_v273.png"
    var_grid.save(vgp, quality=92)
    print(f"  variants -> {vgp}")


if __name__ == "__main__":
    main()
