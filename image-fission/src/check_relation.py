"""关联度自检: 6 张用户原图 vs v126/v127_clean/v127_final 实际输出的色彩 + 直方图 + 边缘密度对比."""
import os, json
import numpy as np
from PIL import Image

LIB = r"E:\迁移\Documents\My Pictures\Saved Pictures\歪歪.library\images"
J126 = r"E:\Desktop\双接口\image-fission\jobs\smoke_v126_1787808430"
J127 = r"E:\Desktop\双接口\image-fission\jobs\smoke_v127_1787811412"
OUT_FINAL = r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v127"

# 通过色彩 profile + 尺寸匹配验证
# illust_1=黑白卷草 -> MT3UHX7QT1WBN.info (558x960, 41%白+38%黑)
# eagle_2=黑橙烈焰+骷髅 -> MT3UHVX87ZVIZ.info (964x1280, 6.35%橙+5.91%红) ←用户给的Pinterest.jpg
# denim_3=牛仔蓝白 -> MT3UGQT7VHWIQ.info (736x1308, 65%白+13%蓝)
# camo_4=棕榈迷彩 -> MT3UGHFWBTPEY.info (736x1288, 64%黑+5%红)
# skull_5=骷髅黑红 -> MT3UGIMFW69PT.info (1242x1754, 20%黑)
# metal_6=黑白金属 -> MT3UGEIP1ED3B.info (3543x4961, 55%黑+14%白)
REFS = {
    "illust_1": (LIB + r"\MT3UHX7QT1WBN.info\Pinterest.jpg", "B/W 卷草线稿"),
    "eagle_2":  (LIB + r"\MT3UHVX87ZVIZ.info\Pinterest.jpg", "黑橙鹰骷髅盾 (用户本次上传)"),
    "denim_3":  (LIB + r"\MT3UGQT7VHWIQ.info\Pinterest.jpg", "牛仔蓝白"),
    "camo_4":   (LIB + r"\MT3UGHFWBTPEY.info\Pinterest.jpg", "棕榈迷彩"),
    "skull_5":  (LIB + r"\MT3UGIMFW69PT.info\Pinterest.jpg", "骷髅"),
    "metal_6":  (LIB + r"\MT3UGEIP1ED3B.info\Pinterest.jpg", "黑白金属"),
}

def load(p, max_side=768):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1:
        im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return np.array(im).astype(np.float32), im.size

def cf(a):
    r,g,b = a[...,0], a[...,1], a[...,2]
    return {
        "orange": float(((r>140)&(g>50)&(g<150)&(b<110)).mean()*100),
        "black":  float(((r<35)&(g<35)&(b<35)).mean()*100),
        "white":  float(((r>220)&(g>220)&(b>220)).mean()*100),
        "red":    float(((r>150)&(g<80)&(b<80)).mean()*100),
        "blue":   float(((b>100)&(b>r+30)&(g>60)&(g<140)).mean()*100),
        "size":   list(a.shape[:2]),
    }

def edge_density(a):
    g = a.mean(axis=2)
    return float((np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()) / 2)

def chi2(a, b):
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

def scan(d, prefix, key):
    if not os.path.isdir(d): return []
    out = []
    for fn in sorted(os.listdir(d)):
        if prefix == "v126":
            if not fn.startswith(key+"_") or not fn.endswith(".jpg"): continue
        else:
            if not fn.startswith(prefix+"_"+key) and not fn.startswith(prefix+"_"+key): continue
            if not fn.endswith(".jpg"): continue
        out.append(fn)
    return out

report = {}
for key, (rpath, desc) in REFS.items():
    if not os.path.isfile(rpath):
        report[key] = {"err": f"missing ref: {rpath}"}
        continue
    arr_ref, _ = load(rpath)
    rf = cf(arr_ref); rf["edge"]=edge_density(arr_ref); rf["file"]=os.path.basename(os.path.dirname(rpath)); rf["desc"]=desc

    bucket = {"v126":[], "v127_clean":[], "v127_final":[]}

    # v126
    if os.path.isdir(J126):
        for fn in sorted(os.listdir(J126)):
            if not (fn.startswith(key+"_") and fn.endswith(".jpg")): continue
            try:
                a, _ = load(os.path.join(J126, fn))
                f = cf(a); f["edge"]=edge_density(a); f["chi2"]=chi2(arr_ref, a)
                bucket["v126"].append({"file":fn, **f})
            except Exception as e:
                bucket["v126"].append({"file":fn,"err":str(e)})

    # v127 clean
    if os.path.isdir(J127):
        for fn in sorted(os.listdir(J127)):
            if not (fn.startswith(f"v127_clean_{key}") and fn.endswith(".jpg")): continue
            try:
                a, _ = load(os.path.join(J127, fn))
                f = cf(a); f["edge"]=edge_density(a); f["chi2"]=chi2(arr_ref, a)
                bucket["v127_clean"].append({"file":fn, **f})
            except Exception as e:
                bucket["v127_clean"].append({"file":fn,"err":str(e)})

    # v127 final
    if os.path.isdir(OUT_FINAL):
        for fn in sorted(os.listdir(OUT_FINAL)):
            if not (fn.startswith(f"{key}_final") and fn.endswith(".jpg")): continue
            try:
                a, _ = load(os.path.join(OUT_FINAL, fn))
                f = cf(a); f["edge"]=edge_density(a); f["chi2"]=chi2(arr_ref, a)
                bucket["v127_final"].append({"file":fn, **f})
            except Exception as e:
                bucket["v127_final"].append({"file":fn,"err":str(e)})

    report[key] = {"ref": rf, **bucket}

# 人类可读输出
print("="*80)
print("关联度自检报告（原图 vs v126 / v127_clean / v127_final）")
print("="*80)
for key, d in report.items():
    if "err" in d:
        print(f"\n【{key}】  ❌ {d['err']}")
        continue
    r = d["ref"]
    print(f"\n【{key}】 {r['desc']}  原图尺寸={r['size']}")
    print(f"  原图:  orange={r['orange']:5.1f}%  black={r['black']:5.1f}%  white={r['white']:5.1f}%  red={r['red']:5.1f}%  blue={r['blue']:5.1f}%  edge={r['edge']:.1f}")
    for tag in ["v126","v127_clean","v127_final"]:
        if not d[tag]:
            print(f"  {tag:11} -- (no files)")
            continue
        for item in d[tag]:
            if "err" in item:
                print(f"  {tag:11} {item['file']:42} ERR")
                continue
            deltas = {
                "Δorange": item["orange"] - r["orange"],
                "Δblack":  item["black"]  - r["black"],
                "Δwhite":  item["white"]  - r["white"],
                "Δred":    item["red"]    - r["red"],
                "Δblue":   item["blue"]   - r["blue"],
            }
            worst = max(deltas.items(), key=lambda x: abs(x[1]))
            print(f"  {tag:11} {item['file']:42} orange={item['orange']:5.1f} black={item['black']:5.1f} white={item['white']:5.1f} red={item['red']:5.1f} blue={item['blue']:5.1f} edge={item['edge']:5.1f} chi2={item['chi2']:.3f}  最大偏移={worst[0]}={worst[1]:+.1f}")

with open(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\relation_check.json","w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\nFull JSON saved to outputs\\relation_check.json")
