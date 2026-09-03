"""v259 — 按用户最终指令: 结构/构图不动, 主体(bat)微调, 其余(文字)改内容/大小/数量,
文字位置/大小跟原图, 颜色/色块跟原图 1:1 紫锁死。

关键修复 (vs v258 把外环吃成六边形 blob):
  - 擦字 mask 只覆盖「字形暗像素」, 保护外环带 (dist 397~431) 与 bat
  - 双层紫圆徽章 (外环带 + 内圈) 100% 保留原图像素
  - bat 用原图 RGB 像素, 只做仿射微调 (旋/镜像/缩放) 贴回内圈中心
  - 新文字严格按原图排版位置/大小烧 (顶弧/大主字/副字)
  - 风格=2D flat printed, 颜色=原图紫, 无新色
"""
import numpy as np, cv2, importlib.util, sys, json, subprocess, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from arc_text import draw_arc_text, fit_arc_text_width

spec = importlib.util.spec_from_file_location('v253', 'src/v253_bat_logo_inpaint.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

JOB = Path('jobs/smoke_v259'); JOB.mkdir(parents=True, exist_ok=True)
QC = Path('E:/AI/2026-09-02-09-43-13/qc_split_image.py')
INK = m.INK  # (26,10,31)


def build_protect_mask(bgr):
    """保护: 外环带 + bat + 叶饰(浅紫, 非暗, 自然不擦). 返回 uint8 0/255 保护图。"""
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - m.RING['cx'])**2 + (yy - m.RING['cy'])**2)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    prot = np.zeros((h, w), np.uint8)
    # 1) 外环带 (inner_r-10 ~ outer_r+10) 保护
    ring_band = (dist > (m.RING['inner_r'] - 10)) & (dist < (m.RING['outer_r'] + 10))
    prot[ring_band] = 255
    # 2) bat 保护: gray<50 (原图蝙蝠近墨)
    bx, by, bw, bh = m.BAT_BBOX
    bat_region = (gray < 50)
    prot[bat_region] = 255
    return prot


def erase_text_tight(bgr):
    """只擦字形暗像素, 保护外环带与 bat。返回擦净后的 bgr。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = bgr.shape[:2]
    # 暗像素 = 候选字形 (外环带与 bat 已被保护, 这里只取内圈/徽章下方文字)
    dark = (gray < 120)
    prot = build_protect_mask(bgr)
    # 候选文字区 (宽松包围, 但靠 protect 兜底)
    cand = np.zeros((h, w), np.uint8)
    cand[330:500, 380:1180] = 255    # 顶弧 LA CASA
    cand[770:1150, 280:1300] = 255   # BACARDÍ (大主字, 在徽章下方)
    cand[1180:1340, 450:1180] = 255  # WHEART (副字)
    cand[1340:1440, 600:960] = 255   # 小三角装饰
    # Est./1862 在徽章底部两侧: 也擦
    cand[1080:1180, 180:430] = 255   # 左侧 Est.
    cand[1080:1180, 1130:1380] = 255 # 右侧 1862
    text_px = dark & (cand > 0) & (prot == 0)
    mask = (text_px.astype(np.uint8)) * 255
    mask = cv2.dilate(mask, np.ones((4, 4), np.uint8), iterations=2)
    erased = cv2.inpaint(bgr, mask, 9, cv2.INPAINT_TELEA)
    return erased, mask


def extract_bat_strict(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bx, by = 522, 518
    bw, bh = 508, 430
    region_mask = np.zeros_like(gray, dtype=bool)
    region_mask[by:by + bh, bx:bx + bw] = True
    bat = (gray < 50) & region_mask
    bat = cv2.morphologyEx((bat.astype(np.uint8)) * 255, cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8)).astype(bool)
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - m.RING['cx'])**2 + (yy - m.RING['cy'])**2)
    ring_band = (dist > (m.RING['inner_r'] - 4)) & (dist < (m.RING['outer_r'] + 4))
    bat = bat & ~ring_band
    bat = cv2.morphologyEx((bat.astype(np.uint8)) * 255, cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8)).astype(bool)
    return bat


def paste_bat(clean_bg, bat_mask_strict, angle_deg, flip, scale, orig_bgr):
    bx, by, bw, bh = m.BAT_BBOX
    cmask = bat_mask_strict[by:by + bh, bx:bx + bw]
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    crop = orig_rgb[by:by + bh, bx:bx + bw]
    crop_rgba = np.zeros((bh, bw, 4), np.uint8)
    crop_rgba[..., :3] = crop
    crop_rgba[..., 3] = (cmask * 255).astype(np.uint8)
    bat_img = Image.fromarray(crop_rgba, 'RGBA')
    ys, xs = np.where(cmask); cxc, cyc = int(xs.mean()), int(ys.mean())
    if flip:
        bat_img = bat_img.transpose(Image.FLIP_LEFT_RIGHT)
        cxc = bw - cxc
    if abs(scale - 1.0) > 0.001:
        bat_img = bat_img.resize((int(bat_img.width * scale), int(bat_img.height * scale)), Image.LANCZOS)
    if abs(angle_deg) > 0.01:
        bat_img = bat_img.rotate(angle_deg, expand=True, center=(cxc, cyc))
    rw, rh = bat_img.size
    cx_img, cy_img = m.BAT_CENTER
    px, py = cx_img - rw // 2, cy_img - rh // 2
    bg = Image.fromarray(cv2.cvtColor(clean_bg, cv2.COLOR_BGR2RGB)).convert('RGBA')
    bg.paste(bat_img, (px, py), bat_img)
    return Image.fromarray(np.array(bg.convert('RGB')))


def burn_top_arc_orig(img, text, font_size):
    """顶弧严格贴合原图 LA CASA 位置: 圆心=徽章中心(776,745), radius=371 (中位 dist)。"""
    radius = 371
    arc_len = fit_arc_text_width(text, m.FONT_PATH, font_size, radius, char_spacing_px=8)
    total_deg = math.degrees(arc_len / radius)
    start = 270 - total_deg / 2; end = 270 + total_deg / 2
    return draw_arc_text(img, text, m.FONT_PATH, font_size, INK,
                         (m.RING['cx'], m.RING['cy']), radius, start, end,
                         char_spacing_px=8, flip_180=False)


def burn_new_text(img):
    """新文字严格按原图真实排版位置/大小 (逐行直方图实测, 排除bat/ring):
      顶弧 LA CASA 位 (圆心776/745 radius371) /
      大主字 BACARDÍ 实测内下中心 (776,800) 宽~370 /
      副字 WHEART 实测外下中心 (776,1261) 宽~193。无 Est./1862/小三角。
    """
    img = img.convert('RGB')
    h, w = img.height, img.width
    # 顶弧: 原 LA CASA 在 dist~371 弧上, 宽约 344px
    img = burn_top_arc_orig(img, "NOCTIS ALATA DOMVS", int(h * 0.038))
    # 大主字 BACARDÍ 实测中心 (776,800), 原字宽~370 → 校准 NOCTWING 宽 370
    fs_main = m.calibrate("NOCTWING", 370)
    m.burn_centered(img, "NOCTWING", fs_main, 776, 800)
    # 副字 WHEART 实测中心 (776,1261) 徽章外下方, 原字宽~193 → 校准 MORS VINI 宽 193
    fs_small = m.calibrate("MORS VINI", 193)
    m.burn_centered(img, "MORS VINI", fs_small, 776, 1261)
    return img


def make_one(tag, angle, flip, scale):
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert('RGB')
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    bat_mask_strict = extract_bat_strict(bgr)
    erased, tmask = erase_text_tight(bgr)
    composed = paste_bat(erased, bat_mask_strict, angle, flip, scale, bgr)
    final = burn_new_text(composed)
    outp = JOB / f"v259_{tag}_bat_logo.jpg"
    if final.mode != 'RGB':
        final = final.convert('RGB')
    final.save(str(outp), quality=95)
    # 诊断: 外环带是否还在 (dist 397~431 区暗像素应≈0, 因为环带是亮紫)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h2, w2 = bgr.shape[:2]
    yy, xx = np.mgrid[:h2, :w2]
    dist = np.sqrt((xx - m.RING['cx'])**2 + (yy - m.RING['cy'])**2)
    ring_zone = (dist > (m.RING['inner_r'] - 4)) & (dist < (m.RING['outer_r'] + 4))
    ring_dark = (gray[ring_zone] < 120).mean() * 100
    print(f"[{tag}] saved {outp.name} ({outp.stat().st_size//1024}KB)  外环带暗像素残留={ring_dark:.2f}% (越高=环被吃越多)")
    return outp, tmask


VARIANTS = [
    ("A_neutral",   0,  False, 1.00),  # 平展原朝向
    ("B_tiltL",   -22,  False, 1.06),  # 左倾展翼 (差异明显)
    ("C_tiltR",    22,  False, 1.06),  # 右倾展翼 (差异明显)
]


def make_grid(paths, out_path):
    imgs = [Image.open(p).convert('RGB') for p in paths]
    w, h = imgs[0].size
    gap = 12
    grid = Image.new('RGB', (w * len(imgs) + gap * (len(imgs) + 1), h), 'white')
    for i, im in enumerate(imgs):
        grid.paste(im, (gap + i * (w + gap), 0))
    grid.save(str(out_path), quality=92)


if __name__ == '__main__':
    outs = []
    for tag, ang, flip, sc in VARIANTS:
        o, _ = make_one(tag, ang, flip, sc)
        outs.append(o)
    make_grid(outs, JOB / '_grid_v259_variants.jpg')
    for p in outs:
        r = subprocess.run([sys.executable, str(QC), str(p)], capture_output=True, text=True, timeout=120)
        try:
            dec = json.JSONDecoder(); obj, _ = dec.raw_decode(r.stdout.strip())
            flags = obj.get('flags', [])
            red = [f for f in flags if '通过' not in f]
            print(f"QC {p.name}: passed={obj.get('passed')} chi2={obj.get('hist_chi2')} 红旗={red if red else '无'}")
        except Exception as e:
            print(f"QC {p.name} parse err: {e}")
