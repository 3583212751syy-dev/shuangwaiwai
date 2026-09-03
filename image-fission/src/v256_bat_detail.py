"""v256 — 真·裂变 + 内容细节 (合规解)

路线(用户选定「原图剪影内加细节线」):
  1. 骨 = 原图蝙蝠剪影 (bat_mask, 100% 保留原图造型, 翼展比 1.11)
  2. 裂变: 镜像 + 旋转(角度) + 缩放(大小) -> 换角度/大小/数量, 骨不变
  3. 内容细节: 在剪影内部用原图背景紫 (153,103,146) 画翼膜扇骨/翼指分隔线
     (纯墨色 INK 在 INK 剪影上不可见, 故用背景紫 -> 原图已有色, 零新色, 符合配色铁律)
  4. 背景 = clean_base 100% 原图像素 (v253_variants_fix 教训: 背景严禁动)
  5. 暗像素强制 INK (剪影精确墨色)
  6. 烧字 (NOCTWING / MORS VINI / Est.1862 + 顶弧)
"""

import numpy as np, cv2, importlib.util, sys, json, subprocess, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, 'src')
spec = importlib.util.spec_from_file_location('v253', 'src/v253_bat_logo_inpaint.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

JOB = Path('jobs/smoke_v256'); JOB.mkdir(parents=True, exist_ok=True)
QC = Path('E:/AI/2026-09-02-09-43-13/qc_split_image.py')

INK = m.INK                      # (26,10,31) 原图墨色
PURPLE = (153, 103, 146)        # 原图背景紫采样, 零新色


def fission_detailed(angle_deg, scale, tag):
    bx, by, bw, bh = m.BAT_BBOX
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert('RGB')
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    base, bat_mask = m.clean_base(orig, bgr)          # 背景干净(100%原图像素) + 原图蝙蝠掩码
    cmask = bat_mask[by:by + bh, bx:bx + bw]          # 原图蝙蝠二值掩码 (bbox内) = 骨

    # 1) 标准姿态蝙蝠剪影 (黑填充, RGBA)
    pa = np.zeros((bh, bw, 4), np.uint8)
    pa[..., :3] = INK
    pa[..., 3] = (cmask * 255).astype(np.uint8)
    bat = Image.fromarray(pa)

    # 2) 剪影内部紫扇骨/翼指线 (标准姿态坐标, 落在剪影内)
    d = ImageDraw.Draw(bat, 'RGBA')
    cx_in, cy_in = bw // 2, int(bh * 0.60)
    ptsL = [(int(bw * 0.16), int(bh * 0.28)), (int(bw * 0.02), int(bh * 0.40)),
            (int(bw * 0.20), int(bh * 0.52)), (int(bw * 0.06), int(bh * 0.64)),
            (int(bw * 0.24), int(bh * 0.72))]
    for p in ptsL:
        d.line([(cx_in, cy_in), p], fill=PURPLE + (255,), width=3)
        d.line([(cx_in, cy_in), (bw - p[0], p[1])], fill=PURPLE + (255,), width=3)
    # 脊线 + 翼膜中线
    d.line([(cx_in, int(bh * 0.22)), (cx_in, int(bh * 0.86))], fill=PURPLE + (255,), width=3)
    d.line([(int(bw * 0.16), int(bh * 0.40)), (int(bw * 0.84), int(bh * 0.40))], fill=PURPLE + (255,), width=2)

    # 3) 剪掉剪影外 (紫线只留在蝙蝠内)
    arr = np.array(bat); arr[~cmask] = (0, 0, 0, 0); bat = Image.fromarray(arr)

    # 4) 裂变 (镜像 + 缩放 + 旋转换姿态)
    bat = bat.transpose(Image.FLIP_LEFT_RIGHT)
    bat = bat.resize((int(bat.width * scale), bat.height), Image.LANCZOS)
    ys, xs = np.where(cmask); cxc, cyc = int(xs.mean()), int(ys.mean())
    rot = bat.rotate(angle_deg, expand=True, center=(cxc, cyc))

    # 5) 合成到 clean_base 背景
    out = base.copy()
    out.paste(rot, (m.BAT_CENTER[0] - rot.width // 2, m.BAT_CENTER[1] - rot.height // 2), rot)

    # 6) 暗像素强制 INK (剪影精确墨色)
    out_np = np.array(out)
    g = cv2.cvtColor(out_np, cv2.COLOR_RGB2GRAY)
    out_np[g < 90] = INK
    out = Image.fromarray(out_np).convert('RGB')

    # 7) 烧字
    h, w = out.height, out.width
    out = m.burn_top_arc(out, "NOCTWING", int(h * 0.045))
    fs_main = m.calibrate("NOCTWING", 955); m.burn_centered(out, "NOCTWING", fs_main, m.BIG_CENTER[0], m.BIG_CENTER[1])
    fs_small = m.calibrate("MORS VINI", 690); m.burn_centered(out, "MORS VINI", fs_small, m.SMALL_CENTER[0], m.SMALL_CENTER[1])
    f_est = ImageFont.truetype(m.FONT_PATH, int(h * 0.022))
    ImageDraw.Draw(out).text((int(w * 0.30), int(h * 0.745)), "Est.", font=f_est, fill=INK, anchor="mm")
    ImageDraw.Draw(out).text((int(w * 0.70), int(h * 0.745)), "1862", font=f_est, fill=INK, anchor="mm")

    outp = JOB / f"v256_{tag}_bat_logo.jpg"
    if out.mode != 'RGB':
        out = out.convert('RGB')
    out.save(str(outp), quality=95)
    print(f"[{tag}] saved {outp.name} ({outp.stat().st_size // 1024} KB)")
    return outp


VARIANTS = [
    ("A_up",     -16, 1.12),   # 展翼上扬左倾
    ("B_dive",    22, 1.05),   # 俯冲掠翼
    ("C_herald",   0, 1.12),   # 对称纹章
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
    for tag, ang, sc in VARIANTS:
        outs.append(fission_detailed(ang, sc, tag))
    make_grid(outs, JOB / '_grid_v256_variants.jpg')
    # QC
    for p in outs:
        r = subprocess.run([sys.executable, str(QC), str(p)], capture_output=True, text=True, timeout=120)
        dec = json.JSONDecoder(); obj, _ = dec.raw_decode(r.stdout.strip())
        flags = obj.get('flags', [])
        red = [f for f in flags if '通过' not in f]
        print(f"QC {p.name}: passed={obj.get('passed')} chi2={obj.get('hist_chi2')} 红旗={red if red else '无'}")
