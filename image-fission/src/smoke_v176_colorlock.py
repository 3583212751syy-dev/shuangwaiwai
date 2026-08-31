"""v176 色彩锁版：单一变量 IPA_WEIGHT 提升，把裂变图色域拉回原图。
- metal_6: IPA 0.18->0.45 (颜色偏最重，CT 天花板仅 0.59，需生成期锁色)
- eagle_2: IPA 0.18 (冻结 v147 基线，补当年跳过的底图)
其余参数全锁死 v147。渲染后自动接 ColorTransfer(target=生成, ref=原图) 扫 strength 取最优色域。
"""
import sys, time, json, requests, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164
import color_transfer as ct

OUT = HERE.parent / "outputs" / "v176"
OUT.mkdir(parents=True, exist_ok=True)
COMFY = v164.COMFYUI

# 任务表: (id, ipa_weight, lora_detail, filename_prefix)
JOBS = [
    ("metal_6", 0.45, 1.0, "v176_metal_6"),
    ("eagle_2", 0.18, 1.0, "v176_eagle_2"),
]

def render(job):
    rid, ipa, lora, prefix = job
    v164.IPA_WEIGHT = ipa
    v164.LORA_DETAIL = lora
    ref = next(r for r in v164.REFS if r["id"] == rid)
    seed = v164.SEED
    g = v164.build(ref, seed)
    g["15"]["inputs"]["filename_prefix"] = prefix
    print(f"[v176] {rid}: IPA={ipa} LORA={lora} DENOISE={v164.DENOISE} TILE={v164.TILE_STRENGTH} CANNY={v164.CANNY_STRENGTH}", flush=True)
    r = requests.post(f"{COMFY}/prompt", json={"prompt": g, "client_id": f"v176_{int(time.time())}"}, timeout=15)
    j = r.json()
    if r.status_code != 200 or "error" in j:
        print("[ERR]", r.status_code, json.dumps(j)[:1500]); return None, None
    pid = j.get("prompt_id"); print(f"[v176] {rid} pid={pid}", flush=True)
    for i in range(120):
        time.sleep(5)
        h = requests.get(f"{COMFY}/history/{pid}", timeout=10).json()
        if pid not in h:
            continue
        rec = h[pid]; st = rec.get("status", {})
        if st.get("completed"):
            imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
            if imgs:
                fn = imgs[0]["filename"]; sub = imgs[0].get("subfolder", "")
                data = requests.get(f"{COMFY}/view?filename={fn}&type=output&subfolder={sub}", timeout=60).content
                raw = OUT / f"{prefix}_raw.png"; raw.write_bytes(data)
                from PIL import Image, ImageFilter
                im = Image.open(raw).convert("RGB")
                im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)).save(raw, quality=95)
                print(f"[v176] {rid} raw saved {raw.stat().st_size/1024/1024:.1f}MB")
                return rid, raw
        if "error" in st:
            print("[ERR]", st); return None, None
    print("[ERR] timeout", rid); return None, None

def post_ct(rid, raw):
    orig = HERE.parent / "web_gallery" / "img" / f"orig_{rid}.jpg"
    if not orig.exists():
        print(f"[v176] {rid} 无原图对照，跳过 CT"); return raw
    import numpy as np
    from PIL import Image
    o = np.array(Image.open(orig).convert("RGB"))
    b = np.array(Image.open(raw).convert("RGB"))
    best=None
    for s in [0.5,0.7,0.85,1.0,1.15]:
        out = ct.reinhard(b,o,s); p = OUT / f"{rid}_ct_s{s:.2f}.png"
        Image.fromarray(out).save(p)
        ci = ct.color_intersect(str(orig), str(p))
        print(f"  {rid} s={s:.2f} color∩={ci:.3f}")
        if best is None or ci>best[0]: best=(ci,s,p)
    print(f"[v176] {rid} BEST s={best[1]:.2f} color∩={best[0]:.3f} -> {best[2]}")
    # 复制到 final
    final = HERE.parent / "outputs" / "final" / f"{rid}.jpg"
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best[2], final)
    return final

results=[]
for job in JOBS:
    rid, raw = render(job)
    if raw:
        f = post_ct(rid, raw)
        results.append((rid, f))

print("[v176] DONE:", [(r, str(f)) for r,f in results])
