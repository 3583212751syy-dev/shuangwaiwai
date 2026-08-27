"""Identify the 6 user-given original PNG/JPG files by perceptual content match against the v127 outputs (which were generated FROM those originals)."""
import os, numpy as np
from PIL import Image

LIB = r"E:\迁移\Documents\My Pictures\Saved Pictures\歪歪.library\images"
JOBS126 = r"E:\Desktop\双接口\image-fission\jobs\smoke_v126_1787808430"
JOBS127 = r"E:\Desktop\双接口\image-fission\jobs\smoke_v127_1787811412"

def hist(im, bins=8):
    s = (np.array(im.resize((256,256), Image.LANCZOS).convert("RGB")).astype(int) // 32)
    h = np.zeros((bins,bins,bins), dtype=np.float64)
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                h[i,j,k] = ((s[...,0]==i)&(s[...,1]==j)&(s[...,2]==k)).mean()
    return h.flatten()

def color_profile(im):
    a = np.array(im.resize((256,256), Image.LANCZOS).convert("RGB"))
    r,g,b = a[...,0], a[...,1], a[...,2]
    return {
        "orange": float(((r>140)&(g>50)&(g<150)&(b<110)).mean()*100),
        "black":  float(((r<35)&(g<35)&(b<35)).mean()*100),
        "white":  float(((r>220)&(g>220)&(b>220)).mean()*100),
        "red":    float(((r>150)&(g<80)&(b<80)).mean()*100),
        "blue":   float(((b>100)&(b>r+30)&(g>60)&(g<140)).mean()*100),
    }

# Load all 6 originals
origs = []
for d in sorted(os.listdir(LIB)):
    p = os.path.join(LIB, d, "Pinterest.jpg")
    if os.path.isfile(p):
        im = Image.open(p)
        cp = color_profile(im)
        origs.append({"dir": d, "path": p, "size_bytes": os.path.getsize(p), "dim": im.size, "profile": cp})
        print(f"orig {d}: {os.path.getsize(p)}B  {im.size}  {cp}")

# Compare each to each v126 output profile to see which original produces which dominant style
print("\n=== comparing one v126 per key to all 6 originals by color distance ===")
keys = ["eagle_2", "denim_3", "skull_5", "metal_6"]
for k in keys:
    # pick the v126 eagle_2 first as proxy (they all came from the same generator setting)
    out_files = [f for f in os.listdir(JOBS126) if f.startswith(k + "_") and f.endswith(".jpg")]
    if not out_files: continue
    of = os.path.join(JOBS126, sorted(out_files)[0])
    out_profile = color_profile(Image.open(of))
    print(f"\nv126 {k} ({of}) -> {out_profile}")
    scores = []
    for o in origs:
        d = sum(abs(out_profile[k2] - o["profile"][k2]) for k2 in out_profile)
        scores.append((d, o["dir"], o["path"], o["profile"]))
    scores.sort(key=lambda x: x[0])
    for s, d, p, prof in scores[:3]:
        print(f"   dist={s:.1f}  {d}  {p}  prof={prof}")
