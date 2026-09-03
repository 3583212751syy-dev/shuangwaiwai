"""v254 fix: 复用 SDXL raw PNG (jobs/smoke_v254/_raw_253{101,202,303}.png),
背景用 clean_base, 蝙蝠框内做局部 LAB 匹配 + snap + INK 强制.
不重跑 SDXL. 等价于 v253_variants_fix 的思路套到 v254 (详细蝙蝠)."""
from pathlib import Path
import shutil, json, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("v253", ROOT / "src" / "v253_bat_logo_inpaint.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# 复用 fix 函数
spec2 = importlib.util.spec_from_file_location("fix", ROOT / "src" / "v253_variants_fix.py")
fx = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fx)

JOB = ROOT / "jobs" / "smoke_v254"
QC = Path("E:/AI/2026-09-02-09-43-13/qc_split_image.py")

def regen(tag, seed):
    print(f"\n=== {tag} ===")
    raw = JOB / f"_raw_{seed}.png"
    if not raw.exists():
        print(f"  缺 raw {raw.name}"); return None
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert("RGB")
    w, h = orig.size
    bgr_ref = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    base, bat_mask = m.clean_base(orig, bgr_ref)
    sdxl = Image.open(raw).convert("RGB").resize((w, h))
    sdxl_bgr = cv2.cvtColor(np.array(sdxl), cv2.COLOR_RGB2BGR)
    # 复用 v253_variants_fix 的 lock 函数: 蝙蝠框内 LAB 局部匹配 + snap + INK
    locked_bgr = fx.lock_bat_region_only(sdxl_bgr, bgr_ref, m.BAT_BBOX)
    composite_bgr = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
    bx, by, bw, bh = m.BAT_BBOX
    composite_bgr[by:by+bh, bx:bx+bw] = locked_bgr[by:by+bh, bx:bx+bw]
    composite = Image.fromarray(cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB))
    # 烧字
    composite = m.burn_top_arc(composite, "LVMEN NOCTIS", int(h * 0.045))
    fs_main = m.calibrate("NOCTWING", 955); m.burn_centered(composite, "NOCTWING", fs_main, m.BIG_CENTER[0], m.BIG_CENTER[1])
    fs_small = m.calibrate("MORS VINI", 690); m.burn_centered(composite, "MORS VINI", fs_small, m.SMALL_CENTER[0], m.SMALL_CENTER[1])
    f_est = ImageFont.truetype(m.FONT_PATH, int(h * 0.022))
    ImageDraw.Draw(composite).text((int(w * 0.30), int(h * 0.745)), "Est.", font=f_est, fill=m.INK, anchor="mm")
    ImageDraw.Draw(composite).text((int(w * 0.70), int(h * 0.745)), "1862", font=f_est, fill=m.INK, anchor="mm")
    if composite.mode != "RGB":
        composite = composite.convert("RGB")
    out = JOB / f"v254_{tag}_bat_logo.jpg"
    composite.save(str(out), quality=95)
    # QC
    r = subprocess.run([sys.executable, str(QC), str(out)], capture_output=True, text=True, timeout=120)
    try:
        obj, _ = json.JSONDecoder().raw_decode(r.stdout.strip())
    except Exception as e:
        print(f"  QC parse fail: {e}\n  stdout: {r.stdout[:300]}"); return None
    red = [f for f in obj.get("flags", []) if "通过" not in f and "GATE" not in f]
    bg = np.array(composite)[60:160, 60:160].reshape(-1,3).std(0).round(1)
    print(f"  saved {out.name} ({out.stat().st_size//1024} KB)")
    print(f"  QC passed={obj.get('passed')} hist_chi2={obj.get('hist_chi2'):.4f} "
          f"edge(L={obj['left']['edge_density']:.4f}, R={obj['right']['edge_density']:.4f})")
    print(f"  bg_std={[float(x) for x in bg]}  flags={red if red else '无'}")
    return out


VARIANTS = [("A_up", 253101), ("B_dive", 253202), ("C_herald", 253303)]
if __name__ == "__main__":
    outs = []
    for tag, seed in VARIANTS:
        p = regen(tag, seed)
        if p: outs.append(p)
    if len(outs) == 3:
        imgs = [Image.open(p).convert("RGB") for p in outs]
        w, h = imgs[0].size; gap = 12
        grid = Image.new("RGB", (w*3 + gap*4, h), "white")
        for i, im in enumerate(imgs):
            grid.paste(im, (gap + i*(w+gap), 0))
        grid_path = JOB / "_grid_v254_variants.jpg"
        grid.save(str(grid_path), quality=92)
        print(f"\n[grid] {grid_path}")
