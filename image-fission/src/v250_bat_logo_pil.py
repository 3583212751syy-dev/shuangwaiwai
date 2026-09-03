"""
v250 — bat_logo PIL 主导精确绘制 (忠实原图风格 + 蝙蝠仿射裂变)

打破 v244-v249 SDXL 盲推循环: AI 看不到图 + SDXL 不可控地乱加装饰/扭曲蝙蝠姿态,
六版盲推均不到用户目检 8/10。改用程序完全可控方案:

  - 背景 / 圆环 / 弯月装饰: 原图像素 100% 保留 (配色锁死 + 布局忠实原图)
  - 蝙蝠: 原图蝙蝠剪影连通域提取 + 仿射裂变(水平镜像 + 旋转 -22°)
          -> 材质/画风 100% 保留, 姿态明显裂变(符合铁律#1 主体换角度/镜像)
  - 字: PIL 矢量烧写, 严格对齐原图排版坐标
          顶弧 LA CASA... -> LVMEN NOCTIS (夜之光, 蝙蝠隐喻)
          中部 BACARDI  -> NOCTWING        (夜翼)
          中下 MYHEART  -> MORS VINI       (酒之死, 酒标主题)
          底部 Est.1862  -> Est.1862        (保留 vintage 标记)

用户 2026-09-03 硬指令: "原图什么风格什么设计什么排版就去做" + "蝙蝠更像原图"
本方案完全贴合: 背景/环/三角 1:1 原图, 蝙蝠裂变但材质不变, 字非侵权且隐喻一致。
不依赖 ComfyUI, 纯 cv2 + PIL, 秒级出图, 矢量边 edge 可控最低。
"""
import sys, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
REF_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"
FONT_PATH = str(PROJECT / "fonts" / "PirataOne-Regular.ttf")
JOB = PROJECT / "jobs" / "smoke_v250"
JOB.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT / "src"))
from arc_text import draw_arc_text, fit_arc_text_width

# ===== 原图量化参数 (_analyze_orig_info.json) =====
RING = dict(cx=776, cy=745, outer_r=421, inner_r=407)
BAT_BBOX = (522, 518, 508, 456)          # x,y,w,h
BAT_CENTER = (776, 746)
MOON_BBOX = (689, 999, 484, 150)
BIG_CENTER = (776, 1070)                  # BACARDI 大字视觉中心 (原 big_bbox 中心)
SMALL_CENTER = (776, 1285)                # MYHEART 小字视觉中心
INK = (26, 10, 31)                        # 原图字近黑 #1A0A1F


def sample_bg(bgr):
    """采样原图浅紫背景 (非黑/非环/非蝙蝠区域的中位数)"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 90
    h, w = bgr.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    dd = ((xs - RING['cx']) ** 2 + (ys - RING['cy']) ** 2) ** 0.5
    ring = (dd >= RING['inner_r'] - 8) & (dd <= RING['outer_r'] + 8)
    bx, by, bw, bh = BAT_BBOX
    batbox = np.zeros_like(dark); batbox[by:by + bh, bx:bx + bw] = True
    mask = (~dark) & (~ring) & (~batbox)
    vals = bgr[mask]
    med = np.median(vals, axis=0).astype(int)
    return (int(med[2]), int(med[1]), int(med[0]))   # RGB


def extract_bat(bgr):
    """二值掩码提取蝙蝠 (bat bbox 内黑色区域, 避开 BACARDI 大字 y>=951)"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 90
    bx, by = 522, 518
    bw, bh = 508, 430                    # y: 518..948, 避开 BACARDI(顶 951)
    mask = np.zeros_like(dark); mask[by:by + bh, bx:bx + bw] = True
    bat = (dark & mask).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    bat = cv2.morphologyEx(bat, cv2.MORPH_OPEN, kernel)   # 去小噪点
    return bat.astype(bool)


def fission_bat(orig_rgb, bat_mask):
    """原图蝙蝠剪影 -> 水平镜像 + 旋转 -22° (裂变姿态, 材质保留)"""
    bx, by, bw, bh = BAT_BBOX
    crop = orig_rgb.crop((bx, by, bx + bw, by + bh))
    cmask = bat_mask[by:by + bh, bx:bx + bw]
    pa = np.array(crop.convert("RGBA"))
    pa[..., 3] = (cmask * 255).astype(np.uint8)
    crop_rgba = Image.fromarray(pa)
    mirrored = crop_rgba.transpose(Image.FLIP_LEFT_RIGHT)
    ys, xs = np.where(cmask)
    cxc, cyc = int(xs.mean()), int(ys.mean())
    rotated = mirrored.rotate(-22, expand=True, center=(cxc, cyc))
    return rotated


def burn_top_arc(img, text, font_size, color=INK):
    """顶部弧形带 (弧顶落 y≈0.13h, 匹配原图顶弧位置)"""
    w, h = img.width, img.height
    radius = int(w * 0.45)
    arc_len = fit_arc_text_width(text, FONT_PATH, font_size, radius, char_spacing_px=8)
    total_deg = math.degrees(arc_len / radius)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    cx = w // 2
    cy = int(h * 0.13) + radius
    return draw_arc_text(img, text, FONT_PATH, font_size, color,
                         (cx, cy), radius, start, end, char_spacing_px=8, flip_180=False)


def calibrate(text, target_w, lo=4, hi=400):
    """二分找字号使文字 bbox 宽 ≈ target_w (对齐原图排版度量)"""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(FONT_PATH, mid)
        bb = ImageDraw.Draw(Image.new("RGB", (10, 10))).textbbox((0, 0), text, font=f)
        if (bb[2] - bb[0]) < target_w:
            lo = mid
        else:
            hi = mid
    return lo


def burn_centered(img, text, font_size, cx, cy, color=INK):
    ImageDraw.Draw(img).text((cx, cy), text,
                             font=ImageFont.truetype(FONT_PATH, font_size),
                             fill=color, anchor="mm")


def side_by_side(orig_path, gen_path, out_path):
    o = Image.open(orig_path).convert("RGB"); g = Image.open(gen_path).convert("RGB")
    if o.size != g.size:
        g = g.resize(o.size)
    out = Image.new("RGB", (o.width * 2 + 30, o.height), "white")
    out.paste(o, (0, 0)); out.paste(g, (o.width + 30, 0))
    out.save(out_path, quality=95)


def main():
    orig = Image.open(COMFY_INPUT / REF_IMG).convert("RGB")
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    BG = sample_bg(bgr)
    print(f"[v250] BG rgb={BG}  size={w}x{h}")

    # ---- masks ----
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); dark = gray < 90
    bat_mask = extract_bat(bgr)
    ys, xs = np.ogrid[:h, :w]
    dd = ((xs - RING['cx']) ** 2 + (ys - RING['cy']) ** 2) ** 0.5
    ring = (dd >= RING['inner_r'] - 6) & (dd <= RING['outer_r'] + 6)
    mx, my, mw, mh = MOON_BBOX
    moon = np.zeros_like(dark); moon[my:my + mh, mx:mx + mw] = True; moon = moon & dark
    text_mask = dark & (~bat_mask) & (~ring) & (~moon)

    # ---- 底图: 擦原字 + 擦原蝙蝠, 填背景紫 (环/弯月/背景 100% 保留原图) ----
    base_np = np.array(orig).copy()
    base_np[text_mask] = BG
    base_np[bat_mask] = BG
    base = Image.fromarray(base_np)

    # ---- 裂变蝙蝠 paste (中心对齐原蝙蝠中心) ----
    rot = fission_bat(orig, bat_mask)
    rw, rh = rot.size
    base.paste(rot, (BAT_CENTER[0] - rw // 2, BAT_CENTER[1] - rh // 2), rot)
    print(f"[v250] fission bat pasted size={rw}x{rh}")

    # ---- PIL 矢量烧字 (严格对齐原图排版) ----
    base = burn_top_arc(base, "LVMEN NOCTIS", int(h * 0.045))
    fs_main = calibrate("NOCTWING", 955)
    burn_centered(base, "NOCTWING", fs_main, BIG_CENTER[0], BIG_CENTER[1])
    fs_small = calibrate("MORS VINI", 690)
    burn_centered(base, "MORS VINI", fs_small, SMALL_CENTER[0], SMALL_CENTER[1])
    f_est = ImageFont.truetype(FONT_PATH, int(h * 0.022))
    ImageDraw.Draw(base).text((int(w * 0.30), int(h * 0.745)), "Est.",
                              font=f_est, fill=INK, anchor="mm")
    ImageDraw.Draw(base).text((int(w * 0.70), int(h * 0.745)), "1862",
                              font=f_est, fill=INK, anchor="mm")
    print(f"[v250] burn: NOCTWING fs={fs_main}  MORS VINI fs={fs_small}")

    if base.mode != "RGB":
        base = base.convert("RGB")
    out = JOB / "v250_bat_logo.jpg"
    base.save(str(out), quality=95)
    cmp = JOB / "_compare_v250.jpg"
    side_by_side(COMFY_INPUT / REF_IMG, out, cmp)
    print(f"[v250] saved {out} ({out.stat().st_size//1024} KB)")
    print(f"[v250] saved compare {cmp}")


if __name__ == "__main__":
    main()
