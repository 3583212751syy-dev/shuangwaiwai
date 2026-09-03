"""v255: PIL 矢量蝙蝠 + anatomy 细节 (方案 A).
- 黑剪影(IN原图墨色) + 背景紫(153,103,146) 矢量 anatomy 线(耳廓/翼膜扇骨/翼指/缺刻/波状翼缘)
- 零新色(配色铁律), 细节量/位置完全可控, 不依赖 SDXL
- 3 姿态 (A_up 展翼上扬 / B_dive 俯冲 / C_herald 对称纹章) 通过整体 rotate 实现
- 背景 100% clean_base 像素
"""
from pathlib import Path
import json, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("v253", ROOT / "src" / "v253_bat_logo_inpaint.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

JOB = ROOT / "jobs" / "smoke_v255"
JOB.mkdir(parents=True, exist_ok=True)
QC = Path("E:/AI/2026-09-02-09-43-13/qc_split_image.py")

INK = m.INK                      # (26,10,31) 黑
PURPLE = (153, 103, 146)         # 原图背景紫采样均值 = anatomy 线色 (零新色)
BW, BH = m.BAT_BBOX[2], m.BAT_BBOX[3]   # 508, 456
CX, CY = BW // 2, BH // 2               # 254, 228


def draw_bat(d):
    """在 RGBA 画布 d(508x456) 上画对称朝上蝙蝠: 黑剪影(水平大展翼, 饱满) + 紫 anatomy 线"""
    # 身体椭圆 (加宽)
    d.ellipse([CX-60, CY-30, CX+60, CY+132], fill=INK)
    # 头
    d.ellipse([CX-50, CY-134, CX+50, CY-40], fill=INK)
    # 耳廓 (左/右)
    d.polygon([(CX-32, CY-110), (CX-52, CY-176), (CX-8, CY-124)], fill=INK)
    d.polygon([(CX+32, CY-110), (CX+52, CY-176), (CX+8, CY-124)], fill=INK)
    # 左翼: 水平大展翼 (翼尖与肩同高, 翼后缘大幅下斜到腰 -> 大实心三角)
    wl = [(CX-10, CY-22), (CX-238, CY-22), (CX-222, CY+70),
          (CX-158, CY+108), (CX-170, CY+132), (CX-104, CY+146),
          (CX-112, CY+168), (CX-56, CY+150)]
    d.polygon(wl, fill=INK)
    d.polygon([(BW-x, y) for (x, y) in wl], fill=INK)
    # 脚
    d.polygon([(CX-20, CY+128), (CX, CY+162), (CX+20, CY+128)], fill=INK)
    # ---- 紫 anatomy 线 (挖细节) ----
    lw = 3
    for side in (1, -1):
        sh = (CX - side*10, CY - 22)
        tips = [(CX - side*238, CY-22), (CX - side*222, CY+70),
                (CX - side*158, CY+108), (CX - side*104, CY+146), (CX - side*56, CY+150)]
        for t in tips:
            d.line([sh, t], fill=PURPLE, width=lw)
        d.line([sh, (CX - side*238, CY-22)], fill=PURPLE, width=lw)  # 扇骨
    # 耳内线
    d.line([(CX-32, CY-110), (CX-52, CY-176)], fill=PURPLE, width=lw)
    d.line([(CX+32, CY-110), (CX+52, CY-176)], fill=PURPLE, width=lw)
    # 脊线
    d.line([(CX, CY-134), (CX, CY+146)], fill=PURPLE, width=lw)


def make_bat_rgba(angle, scale=1.0):
    """生成旋转/缩放后的蝙蝠 RGBA (508x456)"""
    canvas = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    draw_bat(d)
    if scale != 1.0:
        canvas = canvas.resize((int(BW*scale), int(BH*scale)))
        # 重新居中
        tmp = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
        tmp.paste(canvas, (CX - canvas.width//2, CY - canvas.height//2), canvas)
        canvas = tmp
    if angle != 0:
        canvas = canvas.rotate(angle, expand=False, center=(CX, CY), resample=Image.BICUBIC)
    return canvas


def composite_one(tag, angle, scale=1.0, burn=True):
    print(f"\n=== {tag} (angle={angle}, scale={scale}) ===")
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert("RGB")
    w, h = orig.size
    bgr_ref = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    base, bat_mask = m.clean_base(orig, bgr_ref)
    # 蝙蝠 RGBA -> 贴到 BAT_BBOX
    bat = make_bat_rgba(angle, scale)
    bx, by, bw, bh = m.BAT_BBOX
    composite = base.copy()
    composite.paste(bat, (bx, by), bat)
    # 烧字
    if burn:
        composite = m.burn_top_arc(composite, "LVMEN NOCTIS", int(h * 0.045))
        fs_main = m.calibrate("NOCTWING", 955); m.burn_centered(composite, "NOCTWING", fs_main, m.BIG_CENTER[0], m.BIG_CENTER[1])
        fs_small = m.calibrate("MORS VINI", 690); m.burn_centered(composite, "MORS VINI", fs_small, m.SMALL_CENTER[0], m.SMALL_CENTER[1])
        f_est = ImageFont.truetype(m.FONT_PATH, int(h * 0.022))
        ImageDraw.Draw(composite).text((int(w * 0.30), int(h * 0.745)), "Est.", font=f_est, fill=m.INK, anchor="mm")
        ImageDraw.Draw(composite).text((int(w * 0.70), int(h * 0.745)), "1862", font=f_est, fill=m.INK, anchor="mm")
    if composite.mode != "RGB":
        composite = composite.convert("RGB")
    out = JOB / f"v255_{tag}_bat_logo.jpg"
    composite.save(str(out), quality=95)
    # QC + 蝙蝠区 Laplacian 自审
    r = subprocess.run([sys.executable, str(QC), str(out)], capture_output=True, text=True, timeout=120)
    try:
        obj, _ = json.JSONDecoder().raw_decode(r.stdout.strip())
    except Exception as e:
        print(f"  QC parse fail: {e}\n  stdout:{r.stdout[:200]}"); return out
    im = np.array(composite)
    bb = im[by:by+bh, bx:bx+bw]
    g = cv2.cvtColor(bb, cv2.COLOR_RGB2GRAY); g = cv2.GaussianBlur(g, (3, 3), 0)
    lap = cv2.Laplacian(g, cv2.CV_64F).var(); dark = (g < 90).mean()
    red = [f for f in obj.get("flags", []) if "通过" not in f and "GATE" not in f]
    print(f"  saved {out.name} ({out.stat().st_size//1024} KB)")
    print(f"  QC passed={obj.get('passed')} hist_chi2={obj.get('hist_chi2'):.4f} "
          f"edge(L={obj['left']['edge_density']:.4f}, R={obj['right']['edge_density']:.4f})")
    print(f"  蝙蝠 Laplacian={lap:.1f} (原图基线106) dark%={dark*100:.1f}  红旗={red if red else '无'}")
    return out


VARIANTS = [("A_up", -16, 1.05), ("B_dive", 22, 1.0), ("C_herald", 0, 1.0)]
if __name__ == "__main__":
    outs = []
    for tag, ang, sc in VARIANTS:
        outs.append(composite_one(tag, ang, sc))
    if len(outs) == 3:
        imgs = [Image.open(p).convert("RGB") for p in outs]
        w, h = imgs[0].size; gap = 12
        grid = Image.new("RGB", (w*3 + gap*4, h), "white")
        for i, im in enumerate(imgs):
            grid.paste(im, (gap + i*(w+gap), 0))
        grid.save(str(JOB / "_grid_v255_variants.jpg"), quality=92)
        print(f"\n[grid] {JOB / '_grid_v255_variants.jpg'}")
