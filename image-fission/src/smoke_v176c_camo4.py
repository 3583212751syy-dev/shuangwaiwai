"""v176c camo_4 双目标：LoRA=0(圆润) + IPA=0.45(锁色) + CT 扫最优。推过 0.85。"""
import sys, time, json, requests, shutil, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164
import color_transfer as ct
from PIL import Image, ImageFilter
import numpy as np

OUT = HERE.parent / "outputs" / "v176"
OUT.mkdir(parents=True, exist_ok=True)
COMFY = v164.COMFYUI

try:
    v164.IPA_WEIGHT = 0.45
    v164.LORA_DETAIL = 0.0
    ref = next(r for r in v164.REFS if r["id"] == "camo_4")
    g = v164.build(ref, v164.SEED)
    g["15"]["inputs"]["filename_prefix"] = "v176_camo_4"
    print(f"[v176c] IPA={v164.IPA_WEIGHT} LORA={v164.LORA_DETAIL} DENOISE={v164.DENOISE} 提交 camo_4", flush=True)
    r = requests.post(f"{COMFY}/prompt", json={"prompt": g, "client_id": f"v176c_{int(time.time())}"}, timeout=15)
    j = r.json()
    if r.status_code != 200 or "error" in j:
        print("[ERR] submit", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
    pid = j.get("prompt_id"); print("[v176c] pid=", pid, flush=True)
    raw=None
    for i in range(120):
        time.sleep(5)
        h = requests.get(f"{COMFY}/history/{pid}", timeout=10).json()
        if pid not in h: continue
        rec=h[pid]; st=rec.get("status",{})
        if st.get("completed"):
            imgs=rec.get("outputs",{}).get("15",{}).get("images",[])
            if imgs:
                fn=imgs[0]["filename"]; sub=imgs[0].get("subfolder","")
                data=requests.get(f"{COMFY}/view?filename={fn}&type=output&subfolder={sub}",timeout=60).content
                raw=OUT/"v176_camo_4_raw.png"; raw.write_bytes(data)
                im=Image.open(raw).convert("RGB")
                im.filter(ImageFilter.UnsharpMask(radius=2,percent=150,threshold=3)).save(raw,quality=95)
                print(f"[v176c] raw saved {raw.stat().st_size/1024/1024:.1f}MB", flush=True)
                break
        if "error" in st: print("[ERR] exec", st); break
    if not raw: print("[ERR] timeout"); sys.exit(1)
    # CT 扫
    o=np.array(Image.open(HERE.parent/"web_gallery"/"img"/"orig_camo_4.jpg").convert("RGB"))
    b=np.array(Image.open(raw).convert("RGB"))
    best=None
    for s in [0.5,0.7,0.85,1.0,1.15,1.4]:
        out=ct.reinhard(b,o,s); p=OUT/f"camo4_ct_s{s:.2f}.png"; Image.fromarray(out).save(p)
        ci=ct.color_intersect(str(HERE.parent/"web_gallery"/"img"/"orig_camo_4.jpg"),str(p))
        print(f"  camo s={s:.2f} color∩={ci:.3f}")
        if best is None or ci>best[0]: best=(ci,s,p)
    print(f"[v176c] BEST s={best[1]:.2f} color∩={best[0]:.3f}")
    final=HERE.parent/"outputs"/"final"/"camo_4.jpg"
    final.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(best[2],final)
    shutil.copyfile(best[2], HERE.parent/"web_gallery"/"img"/"fiss_camo_4.jpg")
    print(f"[v176c] ✓ camo_4 final -> {final}")
except Exception:
    print("[FATAL]", traceback.format_exc())
