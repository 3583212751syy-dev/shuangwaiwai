"""
v254_regen_C — 重生成 C_herald 为更对称姿态, 干净过 text_top 硬门槛
原 C 右倾姿态导致顶弧右半 laplacian 略低于左半(text_top gate 差 1% FAIL, 零颜色违例)
改: 翅膀对称展开 + 身体竖直居中(纹章姿态), 平衡顶弧左右半清晰度 -> 过门禁
"""
import importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("E:/Desktop/双接口/image-fission")
spec = importlib.util.spec_from_file_location("v253", str(ROOT / "src" / "v253_bat_logo_inpaint.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
JOB = ROOT / "jobs" / "smoke_v254"
QC = Path("E:/AI/2026-09-02-09-43-13/qc_split_image.py")
ICR = (0, 0, 1552, 2000)

POS = (
    "a single stylized 2D SOLID FLAT BLACK bat silhouette with wings spread SYMMETRICALLY, "
    "body UPRIGHT and perfectly centered, heraldic emblem pose, "
    "DETAILED anatomy: defined pointed EARS, spread wing membrane with visible FINGER BONES and "
    "SCALLOPED wing edges, a thin central body line, gothic vintage craft spirits emblem, "
    "SOLID FILLED FLAT SHAPE with THIN clean internal contour lines only, "
    "NO shading, NO gradient, NO texture, NO fill variation, NO gray, "
    "flat printed vector graphic, perfectly centered inside the circular ring, NO shadow, NO ground plane"
)


def burn_words(composite):
    h, w = composite.size
    composite = m.burn_top_arc(composite, "LVMEN NOCTIS", int(h * 0.045))
    fs_main = m.calibrate("NOCTWING", 955); m.burn_centered(composite, "NOCTWING", fs_main, m.BIG_CENTER[0], m.BIG_CENTER[1])
    fs_small = m.calibrate("MORS VINI", 690); m.burn_centered(composite, "MORS VINI", fs_small, m.SMALL_CENTER[0], m.SMALL_CENTER[1])
    f_est = ImageFont.truetype(m.FONT_PATH, int(h * 0.022))
    ImageDraw.Draw(composite).text((int(w * 0.30), int(h * 0.745)), "Est.", font=f_est, fill=m.INK, anchor="mm")
    ImageDraw.Draw(composite).text((int(w * 0.70), int(h * 0.745)), "1862", font=f_est, fill=m.INK, anchor="mm")
    return composite


def main():
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert("RGB")
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    # 干净底图 + 掩码(一次)
    base, bat_mask = m.clean_base(orig, bgr)
    base.save(str(m.COMFY_INPUT / "v253_base.png"), quality=95)
    mask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bx, by, bw, bh = m.BAT_BBOX
    mask.paste(Image.new("RGBA", (bw, bh), (255, 255, 255, 255)), (bx, by))
    mask.save(str(m.COMFY_INPUT / "v253_mask.png"))

    m.DENOISE = 0.64; m.LORA_DETAIL = 0.40; m.CANNY_STRENGTH = 0.72
    m.POS_BAT = POS
    seed = 253304
    print(f"[C2] denoise={m.DENOISE} lora={m.LORA_DETAIL} seed={seed}")
    raw = m.gen_inpaint(seed, "v253_base.png", "v253_mask.png")
    if not raw:
        print("[C2] SDXL FAIL"); return
    sdxl = Image.open(raw).convert("RGB").resize((w, h))
    sdxl_bgr = cv2.cvtColor(np.array(sdxl), cv2.COLOR_RGB2BGR)
    locked = m.match_hist_lab(bgr, sdxl_bgr)
    locked = m.snap_to_original(locked, bgr, ICR, n_colors=12)
    gv = cv2.cvtColor(locked, cv2.COLOR_BGR2GRAY)
    locked[gv < 90] = (31, 10, 26)
    composite = Image.fromarray(cv2.cvtColor(locked, cv2.COLOR_BGR2RGB))
    composite = burn_words(composite)
    if composite.mode != "RGB":
        composite = composite.convert("RGB")
    out = JOB / "v254_C_herald_bat_logo.jpg"
    composite.save(str(out), quality=95)
    r = subprocess.run([sys.executable, str(QC), str(out)], capture_output=True, text=True, timeout=120)
    dec = json.JSONDecoder(); obj, _ = dec.raw_decode(r.stdout.strip())
    L, Rr = obj["left"], obj["right"]
    gate = (f"lap {Rr['laplacian_variance']:.0f}>={L['laplacian_variance']*0.75:.0f}?"
            f" top {Rr['text_top_laplacian_var']:.0f}>={L['text_top_laplacian_var']*0.75:.0f}?"
            f" edge {Rr['edge_density']:.4f}<={L['edge_density']*1.3:.4f}?"
            f" corner {Rr['corner_color_spread']:.1f}<={L['corner_color_spread']*1.6+5:.1f}?"
            f" center {Rr['center_local_contrast']:.1f}>={L['center_local_contrast']*0.7+5:.1f}?")
    red = [f for f in obj.get("flags", []) if "通过" not in f]
    print(f"[C2] saved {out.name} QC={obj.get('passed')} chi2={obj.get('hist_chi2')} 红旗={red if red else '无'}")
    print(f"     GATE: {gate}")


if __name__ == "__main__":
    main()
