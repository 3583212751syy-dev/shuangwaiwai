#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v175: 在 v174 干净底图基础上，末尾接 ColorTransfer 后期修色
  - generation 部分复刻 v164/v174 全部参数(0 改动)
  - 在 ImageUpscaleWithModel 之后分两路:
    1) SaveImage "15" = 纯 v175 修色前 (保留 raw 对照)
    2) 新增分支: ColorTransfer(target=v175raw, ref=原图 camo_4, method=reinhard_lab, strength=1.0)
       → SaveImage "18" 前缀 "v175_camo_4"
  - 改 0 行 v164 模板，加 3 个节点 (16=ColorTransfer, 17=LoadImage(ref), 18=SaveImage)
  - 装: 无（ColorTransfer 是 ComfyUI 内置）
"""
import sys, json, time, requests
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164

# === 唯一修改：拆 add-detail LoRA（必须，否则碎块回潮）===
v164.LORA_DETAIL = 0.0

OUT = HERE.parent / "outputs" / "v175"
OUT.mkdir(parents=True, exist_ok=True)

camo = next(r for r in v164.REFS if r["id"] == "camo_4")
assert camo["regions"], "camo_4 无 regions 配置"

# 把原图 camo_4 复制到 ComfyUI/input/ 一份稳定文件名给 LoadImage 用
src_orig = HERE.parent / "ComfyUI" / "input" / "pinterest_camo_4.jpg"
dst_ref = HERE.parent / "ComfyUI" / "input" / "v175_color_ref_camo_4.jpg"
if not dst_ref.exists():
    import shutil
    shutil.copyfile(src_orig, dst_ref)
    print(f"[v175] 准备 ref 图 {dst_ref.name}")
else:
    print(f"[v175] ref 图已就位 {dst_ref.name}")

seed = v164.SEED
g = v164.build(camo, seed)
# 改 SaveImage 文件前缀（保留 raw 备份）
g["15"]["inputs"]["filename_prefix"] = "v175raw_camo_4"

# === 插入 ColorTransfer 分支 ===
# node 16 = ColorTransfer，image_target=ImageUpscale 输出 [14,0]，image_ref=原图 camo_4
g["16"] = {
    "class_type": "ColorTransfer",
    "inputs": {
        "image_target": ["14", 0],
        "image_ref": ["17", 0],
        "method": "reinhard_lab",
        "source_stats": "per_frame",
        "strength": 1.0
    }
}
# node 17 = LoadImage ref
g["17"] = {"class_type": "LoadImage", "inputs": {"image": "v175_color_ref_camo_4.jpg"}}
# node 18 = SaveImage 修色后版本
g["18"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": "v175_camo_4"}}

print(f"[v175] LORA_DETAIL={v164.LORA_DETAIL} DENOISE={v164.DENOISE} TILE={v164.TILE_STRENGTH} CANNY={v164.CANNY_STRENGTH} IPA={v164.IPA_WEIGHT}")
print(f"[v175] upscaler=4x_NMKD-Siax_200k.pth (锁死)")
print(f"[v175] 末尾新增: ColorTransfer(image_target=v175raw, ref=原图 camo_4, method=reinhard_lab, strength=1.0)")
print(f"[v175] 提交 camo_4, seed={seed} ...", flush=True)

r = requests.post(f"{v164.COMFYUI}/prompt",
                 json={"prompt": g, "client_id": f"v175_{int(time.time())}"},
                 timeout=15)
j = r.json()
if r.status_code != 200 or "error" in j:
    print("[ERR]", r.status_code, json.dumps(j)[:2000]); sys.exit(1)
pid = j.get("prompt_id")
print(f"[v175] pid={pid}", flush=True)

done_raw = done_fix = False
for i in range(96):
    time.sleep(5)
    h = requests.get(f"{v164.COMFYUI}/history/{pid}", timeout=10).json()
    if pid not in h:
        if i % 6 == 0: print(f"  ...{i*5}s", flush=True)
        continue
    rec = h[pid]
    st = rec.get("status", {})
    if st.get("completed"):
        outs = rec.get("outputs", {})
        # 15 = raw, 18 = 修色后
        for nid, attr, flag_attr in [("15", "raw", "done_raw"), ("18", "fix", "done_fix")]:
            imgs = outs.get(nid, {}).get("images", [])
            for img in imgs:
                fn = img["filename"]; sub = img.get("subfolder", "")
                url = f"{v164.COMFYUI}/view?filename={fn}&type=output&subfolder={sub}"
                data = requests.get(url, timeout=60).content
                out_path = OUT / f"v175_{attr}_{camo['id']}.jpg"
                out_path.write_bytes(data)
                # USM 锐化
                from PIL import Image, ImageFilter
                im = Image.open(out_path).convert("RGB")
                sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                sharp.save(out_path, quality=95)
                print(f"[v175] ✓ {attr} {out_path.name} {out_path.stat().st_size/1024/1024:.1f}MB")
                if attr == "raw": done_raw = True
                if attr == "fix":  done_fix = True
        break
    if "error" in st:
        print("[ERR]", st); break

if not (done_raw and done_fix):
    print(f"[ERR] timeout/incomplete raw={done_raw} fix={done_fix}"); sys.exit(1)

# === 四联对照：原图 | v147 | v174 | v175 修色后 ===
from PIL import Image, ImageDraw, ImageFont
orig = Image.open(HERE.parent / "web_gallery" / "img" / "orig_camo_4.jpg").convert("RGB")
v164_img = Image.open(HERE.parent / "jobs" / "smoke_v164" / "v164_camo_4.jpg").convert("RGB")
v174_img = Image.open(OUT / "v175_raw_camo_4.jpg").convert("RGB")
v175_img = Image.open(OUT / "v175_fix_camo_4.jpg").convert("RGB")

H = 720
def fit(im):
    w, h = im.size
    nw = int(w * H / h)
    return im.resize((nw, H), Image.LANCZOS)
imgs = [fit(orig), fit(v164_img), fit(v174_img), fit(v175_img)]
gap = 12
W = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
canvas = Image.new("RGB", (W, H + 60), (14, 15, 18))
x = 0
for im in imgs:
    canvas.paste(im, (x, 60))
    x += im.width + gap
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except Exception:
    font = ImageFont.load_default()
labels = [
    ("ORIGINAL",                0,                     (230, 234, 240)),
    ("v147 (LORA=1.0) 碎块",      imgs[0].width + gap,    (154, 160, 171)),
    ("v174 (LORA=0) 圆润但糊",   imgs[0].width + imgs[1].width + gap*2, (255, 168, 76)),
    ("v175 +ColorTransfer 修色", imgs[0].width + imgs[1].width + imgs[2].width + gap*3, (110, 231, 183)),
]
for txt, lx, col in labels:
    draw.text((lx + 6, 18), txt, fill=col, font=font)

cmp_path = OUT / "v175_camo_4_4way.jpg"
canvas.save(cmp_path, quality=92)
print(f"[v175] ✓ 4way saved {cmp_path} {cmp_path.stat().st_size/1024/1024:.2f}MB")

# === 更新网页画廊 fiss_camo_4.jpg 为 v175 修色版 ===
import shutil
gallery_fiss = HERE.parent / "web_gallery" / "img" / "fiss_camo_4.jpg"
shutil.copyfile(OUT / "v175_fix_camo_4.jpg", gallery_fiss)
print(f"[v175] ✓ 画廊 fiss_camo_4.jpg ← v175_fix_camo_4.jpg")

print("\n[v175] ✅ 完工。请刷新 http://127.0.0.1:8777/ 拖 camo_4 滑块看 v175 修色版")
