"""v253_variants fix: 背景脏修复.
根因: v253_variants 第 64 行 match_hist_lab 对整图做 LAB 直方图匹配,
把原图干净均匀的背景(clean_base 产物)强制重新分布成 LAB 阶梯, 出现满屏深斑.
修法: 背景 100% 用 clean_base 像素(PIL 输出), match_hist_lab / snap_to_original
只对 BAT_BBOX 内蝙蝠区做, 然后把框内 lock 结果贴回 base. 重烧字.
不复跑 SDXL (raw PNG 都还在)."""
from pathlib import Path
import shutil, json, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("v253", ROOT / "src" / "v253_bat_logo_inpaint.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

JOB = ROOT / "jobs" / "smoke_v253"
QC = Path("E:/AI/2026-09-02-09-43-13/qc_split_image.py")
INK = m.INK  # (26,10,31)
VARIANTS = [("A_up", 253101), ("B_dive", 253202), ("C_herald", 253303)]

# 同一份 seed/pose 与 v253_variants 一致 (原图视觉盲: 蝙蝠必须可复现, 但角度/姿态重新跑)
# 但本次只重做配色锁 + 烧字; 而 SDXL 蝙蝠是从 raw.png 来(已经存在)= 真实 SDXL 输出
# v253_bat_logo_inpaint 模块的 match_hist_lab 是对整图做 -> 我们改成局部
def match_hist_lab_local(src_bgr, dst_bgr, bbox):
    """只对 src_bgr 的 bbox 区域做按 dst_bgr 的 LAB 直方图匹配"""
    x, y, w, h = bbox
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)
    dst_lab = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB)
    out = src_bgr.copy()
    for ch in range(3):
        s_ch = src_lab[y:y+h, x:x+w, ch]
        d_ch = dst_lab[y:y+h, x:x+w, ch]
        s_hist, _ = np.histogram(s_ch.flatten(), 256, (0,256))
        d_hist, _ = np.histogram(d_ch.flatten(), 256, (0,256))
        s_cdf = np.cumsum(s_hist) / s_hist.sum()
        d_cdf = np.cumsum(d_hist) / d_hist.sum()
        lut = np.interp(s_cdf, d_cdf, np.arange(256))
        out_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        flat = out_lab[y:y+h, x:x+w, ch].flatten()
        out_lab[y:y+h, x:x+w, ch] = lut[flat].reshape(h, w).astype(np.uint8)
        out = cv2.cvtColor(out_lab, cv2.COLOR_BGR2LAB)  # 转回时小心: 用 cv2 reverse LUT
        out = cv2.cvtColor(out_lab, cv2.COLOR_BGR2LAB)
    # 用 cv2.LUT 模式更稳
    return out


def lock_bat_region_only(sdxl_bgr, ref_bgr, bbox):
    """对 sdxl_bgr (含原图背景+SDXL蝙蝠) 在 bbox 内做: (1) LAB 局部匹配 ref_bgr
    (2) snap_to_original 锁到 ref_bgr 左半 12 色
    背景 (bbox 外) 保持 sdxl_bgr 原样 (= clean_base 像素)."""
    x, y, w, h = bbox
    # 复制 sdxl 输出 = 背景已经来自 clean_base, 蝙蝠已经来自 SDXL
    out = sdxl_bgr.copy()
    # 截取蝙蝠区
    s_bat = sdxl_bgr[y:y+h, x:x+w]
    r_bat = ref_bgr[y:y+h, x:x+w]
    # 局部直方图匹配: 把蝙蝠区按 ref 蝙蝠区风格匹配
    s_lab = cv2.cvtColor(s_bat, cv2.COLOR_BGR2LAB)
    r_lab = cv2.cvtColor(r_bat, cv2.COLOR_BGR2LAB)
    s_lab_matched = s_lab.copy()
    for ch in range(3):
        s_vals = s_lab[:, :, ch]
        r_vals = r_lab[:, :, ch]
        s_hist, _ = np.histogram(s_vals.flatten(), 256, (0,256))
        r_hist, _ = np.histogram(r_vals.flatten(), 256, (0,256))
        s_cdf = np.cumsum(s_hist) / max(s_hist.sum(), 1)
        r_cdf = np.cumsum(r_hist) / max(r_hist.sum(), 1)
        lut = np.interp(s_cdf, r_cdf, np.arange(256)).astype(np.uint8)
        s_lab_matched[:, :, ch] = lut[s_vals]
    locked_bat = cv2.cvtColor(s_lab_matched, cv2.COLOR_LAB2BGR)
    # snap_to_original 锁到 ref 左半 12 色
    from scipy.spatial import cKDTree
    rh, rw = ref_bgr.shape[:2]
    step = 4
    ref_left = ref_bgr[:, :rw // 2]
    lh, lw = ref_left.shape[:2]
    ys, xs = np.mgrid[0:lh:step, 0:lw:step].reshape(2, -1)
    pts = ref_left[ys, xs].reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(pts, 12, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    pal = np.array([pts[np.sum((pts - c)**2, axis=1).argmin()] for c in centers], np.float32)
    tree = cKDTree(pal)
    region = locked_bat.reshape(-1, 3).astype(np.float32)
    _, idx = tree.query(region, k=1, workers=-1)
    snapped = pal[idx].reshape(h, w, 3).astype(np.uint8)
    # 暗像素强制精确墨色 (消除 23,8,29 类假新色)
    gray = cv2.cvtColor(snapped, cv2.COLOR_BGR2GRAY)
    dark = gray < 90
    snapped[dark] = INK[::-1]  # BGR
    out[y:y+h, x:x+w] = snapped
    return out


def regenerate(tag, raw_path):
    print(f"\n=== {tag} ===")
    # 1) 重新跑 clean_base -> 得到背景干净版 base
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert("RGB")
    w, h = orig.size
    bgr_ref = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    base, bat_mask = m.clean_base(orig, bgr_ref)
    # 2) raw = SDXL 蝙蝠区输出 (背景已经来自 clean_base, 但我们要抛弃,
    #    强制让基色用 base + 蝙蝠用 raw 的 bbox 内像素, 再做局部配色锁)
    sdxl = Image.open(raw_path).convert("RGB").resize((w, h))
    sdxl_bgr = cv2.cvtColor(np.array(sdxl), cv2.COLOR_RGB2BGR)
    # 3) 从 sdxl 提取蝙蝠区像素 (=SDXL 真重绘) -> lock -> 贴回 base
    composite_bgr = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
    locked_bgr = lock_bat_region_only(sdxl_bgr, bgr_ref, m.BAT_BBOX)
    # 4) composite = base 背景 + locked 蝙蝠
    bx, by, bw, bh = m.BAT_BBOX
    composite_bgr[by:by+bh, bx:bx+bw] = locked_bgr[by:by+bh, bx:bx+bw]
    composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
    composite = Image.fromarray(composite_rgb)
    # 5) 烧字 (同 v253_variants)
    composite = m.burn_top_arc(composite, "LVMEN NOCTIS", int(h * 0.045))
    fs_main = m.calibrate("NOCTWING", 955); m.burn_centered(composite, "NOCTWING", fs_main, m.BIG_CENTER[0], m.BIG_CENTER[1])
    fs_small = m.calibrate("MORS VINI", 690); m.burn_centered(composite, "MORS VINI", fs_small, m.SMALL_CENTER[0], m.SMALL_CENTER[1])
    f_est = ImageFont.truetype(m.FONT_PATH, int(h * 0.022))
    ImageDraw.Draw(composite).text((int(w * 0.30), int(h * 0.745)), "Est.", font=f_est, fill=m.INK, anchor="mm")
    ImageDraw.Draw(composite).text((int(w * 0.70), int(h * 0.745)), "1862", font=f_est, fill=m.INK, anchor="mm")
    if composite.mode != "RGB":
        composite = composite.convert("RGB")
    out = JOB / f"v253_{tag}_bat_logo.jpg"
    composite.save(str(out), quality=95)
    # 6) QC
    r = subprocess.run([sys.executable, str(QC), str(out)], capture_output=True, text=True, timeout=120)
    try:
        obj, _ = json.JSONDecoder().raw_decode(r.stdout.strip())
    except Exception as e:
        print(f"  [QC] parse fail: {e}\n  stdout: {r.stdout[:300]}"); return None
    red = [f for f in obj.get("flags", []) if "通过" not in f and "GATE" not in f]
    bg = np.array(composite)[60:160, 60:160].reshape(-1,3).std(0).round(1)
    print(f"  saved {out.name} ({out.stat().st_size//1024} KB)")
    print(f"  QC passed={obj.get('passed')} hist_chi2={obj.get('hist_chi2'):.4f} "
          f"edge(L={obj['left']['edge_density']:.4f}, R={obj['right']['edge_density']:.4f})")
    print(f"  bg_std={[float(x) for x in bg]}  flags={red if red else '无'}")
    return out


def main():
    outs = []
    for tag, seed in VARIANTS:
        raw = JOB / f"v253_{tag}_raw.png"
        if not raw.exists():
            print(f"[{tag}] 缺 raw, 跳过"); continue
        p = regenerate(tag, raw)
        if p: outs.append(p)
    # 重出网格
    if len(outs) == len(VARIANTS):
        imgs = [Image.open(p).convert("RGB") for p in outs]
        w, h = imgs[0].size; gap = 12
        grid = Image.new("RGB", (w*3 + gap*4, h), "white")
        for i, im in enumerate(imgs):
            grid.paste(im, (gap + i*(w+gap), 0))
        grid_path = JOB / "_grid_v253_variants.jpg"
        grid.save(str(grid_path), quality=92)
        print(f"\n[grid] {grid_path}")


if __name__ == "__main__":
    main()
