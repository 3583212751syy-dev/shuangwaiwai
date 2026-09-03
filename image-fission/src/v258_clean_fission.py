"""v258 — 一次性修 5 个真实问题
  1) 彻底擦净所有原图文字（按坐标框定所有文字区域做 inpaint）
  2) 保留原图蝙蝠剪影当主体（无内部蛛网紫线）
  3) 三姿态: 平展 / 镜像 / 微倾
  4) 删除 Est./1862 小字（用户要"小字全清空"）
  5) 背景 100% 干净（只擦字、不做 LAB匹配/snap）

不重画蝙蝠骨（=原图），不画内部细节线（=蛛网），
只换姿态 + 换文字 + 净背景。
"""

import numpy as np, cv2, importlib.util, sys, json, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, 'src')
spec = importlib.util.spec_from_file_location('v253', 'src/v253_bat_logo_inpaint.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

JOB = Path('jobs/smoke_v258'); JOB.mkdir(parents=True, exist_ok=True)
QC = Path('E:/AI/2026-09-02-09-43-13/qc_split_image.py')
INK = m.INK  # (26,10,31)


def erase_all_original_text(orig_bgr):
    """按真实坐标框定所有原图文字区域 (1552x2000 原图)。
    实测位置:
      - 顶弧 "LA CASA DEL MURCIÉLAGO" + 外环: y=300-720, x=350-1200
      - "BACARDÍ" 大字: y=770-1150, x=280-1260
      - "WHEART" 小字: y=1180-1330, x=450-1160
      - 小三角装饰: y=1370-1430, x=660-870
    """
    h, w = orig_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    mask[300:720, 350:1200] = 255    # 顶弧 + 外环
    mask[770:1150, 280:1260] = 255   # BACARDÍ
    mask[1180:1330, 450:1160] = 255  # WHEART
    mask[1370:1430, 660:870] = 255   # 小三角

    # 保护蝙蝠
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    _, bat_mask = m.clean_base(Image.fromarray(orig_rgb), orig_bgr)
    bat_u8 = (bat_mask.astype(np.uint8)) * 255
    mask = cv2.subtract(mask, bat_u8)
    # 膨胀保证边缘擦净
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

    erased = cv2.inpaint(orig_bgr, mask, 9, cv2.INPAINT_TELEA)
    return erased, bat_mask


def extract_bat_strict(bgr):
    """严格提取蝙蝠: gray<50 + 减掉内外圆环区(避免黑框包蝙蝠)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bx, by = 522, 518
    bw, bh = 508, 430
    region_mask = np.zeros_like(gray, dtype=bool)
    region_mask[by:by + bh, bx:bx + bw] = True
    bat = (gray < 50) & region_mask
    bat = cv2.morphologyEx((bat.astype(np.uint8)) * 255, cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8)).astype(bool)
    # 减掉圆环区: RING cx=776 cy=745 outer_r=421 inner_r=407
    # 在 bbox 坐标内重算距离
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - m.RING['cx']) ** 2 + (yy - m.RING['cy']) ** 2)
    # 圆环带 (inner_r-2 ~ outer_r+2): 这些像素都去掉
    ring_band = (dist > (m.RING['inner_r'] - 4)) & (dist < (m.RING['outer_r'] + 4))
    bat = bat & ~ring_band
    # 再开运算去掉小碎点
    bat = cv2.morphologyEx((bat.astype(np.uint8)) * 255, cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8)).astype(bool)
    return bat


def paste_bat(clean_bg, bat_mask_strict, angle_deg, flip, scale, orig_bgr):
    """用原始 RGB 像素 (含膜的紫透感), 不填平 INK。
    原图 bat 区=蝙蝠真形, 不含外圈。"""
    bx, by, bw, bh = m.BAT_BBOX  # (522, 518, 508, 430)
    cmask = bat_mask_strict[by:by + bh, bx:bx + bw]
    # 拿原始 bat 区 RGB, 用 cmask 作 alpha
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    crop = orig_rgb[by:by + bh, bx:bx + bw]
    crop_rgba = np.zeros((bh, bw, 4), np.uint8)
    crop_rgba[..., :3] = crop
    crop_rgba[..., 3] = (cmask * 255).astype(np.uint8)
    bat_img = Image.fromarray(crop_rgba, 'RGBA')
    ys, xs = np.where(cmask); cxc, cyc = int(xs.mean()), int(ys.mean())
    if flip:
        bat_img = bat_img.transpose(Image.FLIP_LEFT_RIGHT)
    if abs(scale - 1.0) > 0.001:
        nw = int(bat_img.width * scale)
        bat_img = bat_img.resize((nw, bat_img.height), Image.LANCZOS)
    if abs(angle_deg) > 0.01:
        bat_img = bat_img.rotate(angle_deg, expand=True,
                                 center=(cxc if not flip else bw - cxc, cyc))
    rw, rh = bat_img.size
    cx_img, cy_img = m.BAT_CENTER
    px = cx_img - rw // 2
    py = cy_img - rh // 2
    bg = Image.fromarray(cv2.cvtColor(clean_bg, cv2.COLOR_BGR2RGB)).convert('RGBA')
    bg.paste(bat_img, (px, py), bat_img)
    return Image.fromarray(np.array(bg.convert('RGB')))


def burn_new_text(img):
    """烧新文字: 顶弧 NOCTWING, 大字 NOCTWING, 小字 MORS VINI; 无 Est./1862"""
    img = img.convert('RGB')
    h, w = img.height, img.width
    # 顶弧 (用 v253 burn_top_arc, 返回新 img)
    img = m.burn_top_arc(img, "NOCTWING", int(h * 0.045))
    # 底部大字 (burn_centered 原地修改)
    fs_main = m.calibrate("NOCTWING", 955)
    m.burn_centered(img, "NOCTWING", fs_main, m.BIG_CENTER[0], m.BIG_CENTER[1])
    # 底部小字
    fs_small = m.calibrate("MORS VINI", 690)
    m.burn_centered(img, "MORS VINI", fs_small, m.SMALL_CENTER[0], m.SMALL_CENTER[1])
    # NO Est./1862 (per user "小字全清空")
    return img


def make_one(tag, angle, flip, scale):
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert('RGB')
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    # 严格蝙蝠掩码 (只抓真黑, 排除外圈)
    bat_mask_strict = extract_bat_strict(bgr)
    # 用 m.extract_bat 的结果作为 bat 保护区(防止擦到 bat)
    # 但用 m.clean_base 仍需 bat_mask 给擦字时保护
    _, bat_mask_old = m.clean_base(orig, bgr)
    # 擦字 (用旧 bat_mask 保护)
    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = orig_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[300:720, 350:1200] = 255
    mask[770:1150, 280:1260] = 255
    mask[1180:1330, 450:1160] = 255
    mask[1370:1430, 660:870] = 255
    bat_u8 = (bat_mask_old.astype(np.uint8)) * 255
    mask = cv2.subtract(mask, bat_u8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    erased = cv2.inpaint(bgr, mask, 9, cv2.INPAINT_TELEA)
    composed = paste_bat(erased, bat_mask_strict, angle, flip, scale, bgr)
    final = burn_new_text(composed)
    outp = JOB / f"v258_{tag}_bat_logo.jpg"
    if final.mode != 'RGB':
        final = final.convert('RGB')
    final.save(str(outp), quality=95)
    print(f"[{tag}] saved {outp.name} ({outp.stat().st_size // 1024} KB)")
    return outp


VARIANTS = [
    ("A_neutral",    0, False, 1.05),  # 平展 (镜像掉, 留原图朝向)
    ("B_mirror",     0, True,  1.08),  # 镜像 (左右翻转)
    ("C_tilt",     -14, False, 1.05),  # 微上扬左倾
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
        outs.append(make_one(tag, ang, flip, sc))
    make_grid(outs, JOB / '_grid_v258_variants.jpg')
    # QC
    for p in outs:
        r = subprocess.run([sys.executable, str(QC), str(p)], capture_output=True, text=True, timeout=120)
        try:
            dec = json.JSONDecoder(); obj, _ = dec.raw_decode(r.stdout.strip())
            flags = obj.get('flags', [])
            red = [f for f in flags if '通过' not in f]
            print(f"QC {p.name}: passed={obj.get('passed')} chi2={obj.get('hist_chi2')} 红旗={red if red else '无'}")
        except Exception as e:
            print(f"QC {p.name} parse err: {e}")