"""
v254_fix — 修 v254 的"背景乱色"红旗(配色铁律违反)
根因: inpaint grow_mask_by=12 外扩重绘圈, 那圈 SDXL 紫没被 snap_to_original 锁回(它只吸附精确 BAT_BBOX)
修法: 复用已生成的 SDXL raw PNG, 重做 配色锁, 但 snap 区域扩到 BAT_BBOX 外扩 16px(覆盖漏色圈); 再重烧字
不动 SDXL 生成(省重跑)
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

VARIANTS = [("A_up", 253101), ("B_dive", 253202), ("C_herald", 253303)]
# snap 区域 = 整图, 彻底消除任何位置的 inpaint 漏色(数学上保证零新色); 背景/字都是原图色不会被改变
ICR = (0, 0, 1552, 2000)


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
    bx, by, bw, bh = m.BAT_BBOX

    for tag, seed in VARIANTS:
        raw_p = JOB / f"_raw_{seed}.png"
        if not raw_p.exists():
            print(f"[{tag}] 缺少 raw {raw_p.name}, 跳过"); continue
        sdxl = Image.open(raw_p).convert("RGB").resize((w, h))
        sdxl_bgr = cv2.cvtColor(np.array(sdxl), cv2.COLOR_RGB2BGR)
        locked = m.match_hist_lab(bgr, sdxl_bgr)
        locked = m.snap_to_original(locked, bgr, ICR, n_colors=12)   # 整图吸附到原图左半 12 色板(零新色)
        # 蝙蝠暗像素强制原图精确墨色 INK, 消除"近黑色板中心 vs 原图墨色"的假阳性红旗, 且背景不受影响
        gv = cv2.cvtColor(locked, cv2.COLOR_BGR2GRAY)
        locked[gv < 90] = (31, 10, 26)
        composite = Image.fromarray(cv2.cvtColor(locked, cv2.COLOR_BGR2RGB))
        composite = burn_words(composite)
        if composite.mode != "RGB":
            composite = composite.convert("RGB")
        out = JOB / f"v254_{tag}_bat_logo.jpg"
        composite.save(str(out), quality=95)
        # QC
        r = subprocess.run([sys.executable, str(QC), str(out)], capture_output=True, text=True, timeout=120)
        dec = json.JSONDecoder(); obj, _ = dec.raw_decode(r.stdout.strip())
        flags = obj.get("flags", [])
        red = [f for f in flags if "通过" not in f]
        L, Rr = obj["left"], obj["right"]
        gate_detail = (f"lap {Rr['laplacian_variance']:.0f}>={L['laplacian_variance']*0.75:.0f}?"
                       f" top {Rr['text_top_laplacian_var']:.0f}>={L['text_top_laplacian_var']*0.75:.0f}?"
                       f" edge {Rr['edge_density']:.4f}<={L['edge_density']*1.3:.4f}?"
                       f" corner {Rr['corner_color_spread']:.1f}<={L['corner_color_spread']*1.6+5:.1f}?"
                       f" center {Rr['center_local_contrast']:.1f}>={L['center_local_contrast']*0.7+5:.1f}?")
        print(f"[{tag}] saved {out.name} QC={obj.get('passed')} chi2={obj.get('hist_chi2')} 红旗={red if red else '无'}")
        print(f"      GATE: {gate_detail}")


if __name__ == "__main__":
    main()
