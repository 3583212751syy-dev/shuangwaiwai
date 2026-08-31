"""v174 camo_4 修复版：拆 add-detail LoRA（1.0→0），保持 v147 其余参数全锁死
根因：add-detail-xl + 4x_NMKD-Siax 在大平滑色块上叠高频微纹理 = 圆润迷彩变碎片。
修法：仅 LORA_DETAIL=0，其余节点完全复用 v164 模板。
"""
import sys, time, json, requests
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164

# === 唯一修改：拆 add-detail LoRA ===
v164.LORA_DETAIL = 0.0

OUT = HERE.parent / "outputs" / "v174"
OUT.mkdir(parents=True, exist_ok=True)

camo = next(r for r in v164.REFS if r["id"] == "camo_4")
assert camo["regions"], "camo_4 无 regions 配置"

seed = v164.SEED
g = v164.build(camo, seed)
# 改 SaveImage 文件前缀，避免覆盖 v164 产物
g["15"]["inputs"]["filename_prefix"] = "v174_camo_4"

print(f"[v174] LORA_DETAIL={v164.LORA_DETAIL}  DENOISE={v164.DENOISE}  TILE={v164.TILE_STRENGTH}  CANNY={v164.CANNY_STRENGTH}  IPA={v164.IPA_WEIGHT}")
print(f"[v174] upscaler=4x_NMKD-Siax_200k.pth (锁死)")
print(f"[v174] 提交 camo_4, seed={seed} ...", flush=True)

r = requests.post(f"{v164.COMFYUI}/prompt",
                 json={"prompt": g, "client_id": f"v174_{int(time.time())}"},
                 timeout=15)
j = r.json()
if r.status_code != 200 or "error" in j:
    print("[ERR]", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
pid = j.get("prompt_id")
print(f"[v174] pid={pid}", flush=True)

done = False
for i in range(96):
    time.sleep(5)
    h = requests.get(f"{v164.COMFYUI}/history/{pid}", timeout=10).json()
    if pid not in h:
        print(f"  ...{i*5}s", flush=True); continue
    rec = h[pid]
    st = rec.get("status", {})
    if st.get("completed"):
        imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
        if imgs:
            fn = imgs[0]["filename"]
            sub = imgs[0].get("subfolder", "")
            url = f"{v164.COMFYUI}/view?filename={fn}&type=output&subfolder={sub}"
            data = requests.get(url, timeout=60).content
            out = OUT / "v174_camo_4.jpg"
            out.write_bytes(data)
            # USM 锐化（与 v164 一致）
            from PIL import Image, ImageFilter
            im = Image.open(out).convert("RGB")
            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            sharp.save(out, quality=95)
            print(f"[v174] ✓ saved {out} {out.stat().st_size/1024/1024:.1f}MB")
            done = True; break
    if "error" in st:
        print("[ERR]", st); break

if not done:
    print("[ERR] timeout"); sys.exit(1)

# === 三联对照：原图 | v164 | v174 ===
from PIL import Image, ImageDraw, ImageFont
orig = Image.open(HERE.parent / "web_gallery" / "img" / "orig_camo_4.jpg").convert("RGB")
v164_img = Image.open(HERE.parent / "jobs" / "smoke_v164" / "v164_camo_4.jpg").convert("RGB")
v174_img = Image.open(out).convert("RGB")

# 统一高度
H = 720
def fit(im):
    w, h = im.size
    nw = int(w * H / h)
    return im.resize((nw, H), Image.LANCZOS)
orig_r, v164_r, v174_r = fit(orig), fit(v164_img), fit(v174_img)
gap = 14
W = orig_r.width + v164_r.width + v174_r.width + gap * 2
canvas = Image.new("RGB", (W, H + 50), (14, 15, 18))
canvas.paste(orig_r, (0, 50))
canvas.paste(v164_r, (orig_r.width + gap, 50))
canvas.paste(v174_r, (orig_r.width + v164_r.width + gap*2, 50))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font = ImageFont.load_default()
labels = [("ORIGINAL", (0, 12)), ("v147 (LORA=1.0) 碎块", (orig_r.width + gap, 12)),
          ("v174 (LORA=0) 圆润修复", (orig_r.width + v164_r.width + gap*2, 12))]
for txt, (x, y) in labels:
    color = (255, 106, 61) if "v174" in txt else (230, 234, 240) if "ORIGINAL" in txt else (154, 160, 171)
    draw.text((x + 6, y), txt, fill=color, font=font)

cmp_path = OUT / "v174_camo_4_3way.jpg"
canvas.save(cmp_path, quality=92)
print(f"[v174] ✓ 3way saved {cmp_path} {cmp_path.stat().st_size/1024/1024:.2f}MB")

# === 更新网页画廊 fiss_camo_4.jpg 为 v174 ===
import shutil
gallery_fiss = HERE.parent / "web_gallery" / "img" / "fiss_camo_4.jpg"
shutil.copyfile(out, gallery_fiss)
print(f"[v174] ✓ 更新画廊 fiss_camo_4.jpg ← v174_camo_4.jpg")