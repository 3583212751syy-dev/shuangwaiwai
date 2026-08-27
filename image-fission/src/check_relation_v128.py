"""v128 关联度自检 + OCR 验证 + SSIM (原图 vs v128_clean + v128_final)."""
import os, json
import numpy as np
from PIL import Image

LIB = r"E:\迁移\Documents\My Pictures\Saved Pictures\歪歪.library\images"
J128 = r"E:\Desktop\双接口\image-fission\jobs\smoke_v128_1787814169"
OUT_FINAL = r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v128"

# 正确映射 (color profile + visual confirmed)
REFS = {
    "illust_1": (LIB + r"\MT3UHX7QT1WBN.info\Pinterest.jpg", "B/W 卷草线稿"),
    "eagle_2":  (LIB + r"\MT3UHVX87ZVIZ.info\Pinterest.jpg", "黑橙鹰骷髅盾 (用户本次上传)"),
    "denim_3":  (LIB + r"\MT3UGQT7VHWIQ.info\Pinterest.jpg", "牛仔蓝白"),
    "camo_4":   (LIB + r"\MT3UGIMFW69PT.info\Pinterest.jpg", "棕榈迷彩"),
    "skull_5":  (LIB + r"\MT3UGHFWBTPEY.info\Pinterest.jpg", "骷髅 TRUE NEVER DIES (黑/红)"),
    "metal_6":  (LIB + r"\MT3UGEIP1ED3B.info\Pinterest.jpg", "黑白金属"),
}

def load(p, max_side=1024):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1: im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return np.array(im).astype(np.float32)

def load_gray(p, max_side=1024):
    im = Image.open(p).convert("L")
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1: im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return np.array(im).astype(np.float32)

def cf(a):
    r,g,b = a[...,0], a[...,1], a[...,2]
    return {
        "orange": float(((r>140)&(g>50)&(g<150)&(b<110)).mean()*100),
        "black":  float(((r<35)&(g<35)&(b<35)).mean()*100),
        "white":  float(((r>220)&(g>220)&(b>220)).mean()*100),
        "red":    float(((r>150)&(g<80)&(b<80)).mean()*100),
        "blue":   float(((b>100)&(b>r+30)&(g>60)&(g<140)).mean()*100),
    }

def ssim_simple(a, b):
    """快速近似 SSIM (resize b 到 a 的尺寸)."""
    if a.shape != b.shape:
        from PIL import Image
        h, w = a.shape
        b_img = Image.fromarray(b.astype(np.uint8)).resize((w, h), Image.LANCZOS)
        b = np.array(b_img).astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    C1, C2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
    return float(num / den)

def hist_chi2(a, b):
    def H(im):
        s = (im//32).astype(np.int32)
        h = np.zeros((8,8,8), dtype=np.float64)
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    h[i,j,k] = ((s[...,0]==i)&(s[...,1]==j)&(s[...,2]==k)).mean()
        return h.flatten()
    ha, hb = H(a), H(b)
    eps = 1e-9
    return float(((ha-hb)**2/(ha+hb+eps)).sum())

report = {}
for key, (rpath, desc) in REFS.items():
    if not os.path.isfile(rpath): continue
    arr_ref, gray_ref = load(rpath), load_gray(rpath)
    rf = cf(arr_ref); rf["desc"]=desc
    bucket = {"v128_clean":[], "v128_final":[]}

    cpath = os.path.join(J128, f"v128_clean_{key}.png")
    if os.path.isfile(cpath):
        a = load(cpath); g = load_gray(cpath)
        f = cf(a); f["ssim_approx"]=ssim_simple(gray_ref, g); f["chi2"]=hist_chi2(arr_ref, a)
        bucket["v128_clean"].append({"file": os.path.basename(cpath), **f})

    fpath = os.path.join(OUT_FINAL, f"{key}_final.jpg")
    if os.path.isfile(fpath):
        a = load(fpath); g = load_gray(fpath)
        f = cf(a); f["ssim_approx"]=ssim_simple(gray_ref, g); f["chi2"]=hist_chi2(arr_ref, a)
        bucket["v128_final"].append({"file": os.path.basename(fpath), **f})

    report[key] = {"ref": rf, **bucket}

print("="*80)
print("v128 自检报告（v126/127 vs v128 改进对比）")
print("="*80)

# Reuse v126 numbers for direct comparison
v126_metrics = {
    "illust_1": 0.034,
    "eagle_2": 0.274,
    "denim_3": 0.104,
    "camo_4": 1.165,
    "skull_5": 1.442,
    "metal_6": 0.077,
}
print(f"{'key':10} {'原图chi2_v126':16} {'v128_clean_chi2':16} {'v128_clean_ssim':15} 评估")
print("-" * 90)
for key, d in report.items():
    if not d.get("v128_clean"): continue
    item = d["v128_clean"][0]
    chi2_v126 = v126_metrics.get(key, "-")
    chi2_v128 = item.get("chi2", 0)
    ssim_v128 = item.get("ssim_approx", 0)
    # 评估: SSIM 0.30-0.55 = 同源但不同; chi2 < 0.5 = 颜色近; 接近 0.5-1 = 边缘
    eval_str = ""
    if ssim_v128 > 0.7: eval_str = "❌ 太像 (侵权)"
    elif ssim_v128 > 0.55: eval_str = "⚠️ 偏像"
    elif ssim_v128 > 0.30: eval_str = "✅ 同源+不同"
    elif ssim_v128 > 0.15: eval_str = "⚠️ 偏不同"
    else: eval_str = "❌ 太不同 (无关联)"
    print(f"  {key:10} {chi2_v126!s:16} {chi2_v128:16.3f} {ssim_v128:15.3f}  {eval_str}")

print()
for key, d in report.items():
    if "err" in d: continue
    r = d["ref"]
    print(f"\n【{key}】 {r['desc']}")
    print(f"  原图: orange={r['orange']:5.1f}%  black={r['black']:5.1f}%  white={r['white']:5.1f}%  red={r['red']:5.1f}%  blue={r['blue']:5.1f}%")
    for tag in ["v128_clean", "v128_final"]:
        if not d[tag]: continue
        for item in d[tag]:
            print(f"  {tag:11} {item['file']:32} orange={item['orange']:5.1f} black={item['black']:5.1f} white={item['white']:5.1f} red={item['red']:5.1f} blue={item['blue']:5.1f} ssim={item['ssim_approx']:5.3f} chi2={item['chi2']:5.3f}")

with open(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v128_check.json","w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
