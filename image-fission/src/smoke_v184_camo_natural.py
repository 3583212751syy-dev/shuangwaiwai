"""v184 camo_4：严格红线版（自然裂变，不硬塞角度/HEX）
基于 v174 的干净基座（LORA_DETAIL=0）+ v164 的自然角度词（"slightly bent by wind"/"straight up"）
只改一个维度：用自然语言加大小/数量裂变（不写度数、不写 HEX 色值）
换种子让角度/位置自然变化。
"""
import sys, time, json, requests, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164

# === 干净基座：拆 LoRA 防碎裂 ===
v164.LORA_DETAIL = 0.0

# === 只改 camo_4 的 global_pos + regions，其他参数全锁死 ===
camo = next(r for r in v164.REFS if r["id"] == "camo_4")

camo["global_pos"] = (
    "bold military camouflage print pattern, vector illustration style, "
    "olive green and tan and dark brown color blocks with soft rounded organic edges, "
    "black palm tree silhouettes with crisp outline, sharp contrast, "
    "no text, no letters, no words anywhere, "
    "fabric print quality, repeatable seamless pattern feel, "
    "trees of VARYING sizes from tall bold foreground to tiny distant background"
)

camo["regions"] = [
    # 主棕榈：明显更大
    {"x": 0.30, "y": 0.00, "w": 0.40, "h": 0.50, "strength": 1.25,
     "prompt": ("a TALL bold royal palm tree centered in the design, "
                "thin curving trunk slightly bent by wind, "
                "TOP CROWN of wide fan-shaped fronds (NOT feather pinnate leaves) "
                "spreading in 8 to 10 distinct plumes, "
                "pure black silhouette with crisp clean outlines, "
                "MUCH LARGER than all other trees, "
                "occasional frond overlapping a neighboring tree to suggest depth, "
                "tropical military style. " + v164.COHESIVE)},
    # 副棕榈：约主树一半高
    {"x": 0.62, "y": 0.20, "w": 0.30, "h": 0.50, "strength": 1.20,
     "prompt": ("a SECONDARY shorter coconut palm tree in the right portion, "
                "sturdier trunk straight up, smaller curved fronds in 5 plumes, "
                "about HALF the height of the main palm, "
                "slightly different shape from the main palm to break uniformity, "
                "pure black silhouette with crisp outline. " + v164.COHESIVE)},
    # 左下小棕榈：明显更小
    {"x": 0.05, "y": 0.45, "w": 0.30, "h": 0.55, "strength": 1.10,
     "prompt": ("a small palm tree in the bottom-left corner, "
                "young sapling style, only 4 drooping fronds, "
                "MUCH SMALLER than the others, "
                "tucked behind a camo color block, "
                "pure black silhouette. " + v164.COHESIVE)},
    # 迷彩色块：纯文字配色，圆润有机，不写 HEX
    {"x": 0.00, "y": 0.20, "w": 1.0, "h": 0.30, "strength": 1.30,
     "prompt": ("bold camouflage color blocks in irregular organic blob shapes, "
                "olive green and tan and dark brown mixed together, "
                "LARGE color field, no gradient, no soft airbrush, "
                "fabric-print-ready camouflage. " + v164.COHESIVE)},
    # 底部队列：数量从 6 → 8
    {"x": 0.00, "y": 0.92, "w": 1.0, "h": 0.08, "strength": 1.05,
     "prompt": ("a thin horizontal band of EIGHT tiny palm tree silhouettes in a row at the very bottom, "
                "each slightly smaller than the previous, marching right to left, "
                "pure black silhouettes with no detail, " + v164.COHESIVE)},
]

OUT = HERE.parent / "outputs" / "v184"
OUT.mkdir(parents=True, exist_ok=True)

seed = v164.SEED + 7   # 换种子 → 角度/位置自然变化
g = v164.build(camo, seed)
g["15"]["inputs"]["filename_prefix"] = "v184_camo_4"

print(f"[v184] LORA_DETAIL={v164.LORA_DETAIL} DENOISE={v164.DENOISE} TILE={v164.TILE_STRENGTH} CANNY={v164.CANNY_STRENGTH} IPA={v164.IPA_WEIGHT}")
print(f"[v184] 配色=纯文字(olive/tan/dark brown) 无HEX; 角度=自然(slightly bent/straight); 大小/数量=自然语言; seed={seed}")
print(f"[v184] 提交 camo_4 ...", flush=True)

r = requests.post(f"{v164.COMFYUI}/prompt", json={"prompt": g, "client_id": f"v184_{int(time.time())}"}, timeout=15)
j = r.json()
if r.status_code != 200 or "error" in j:
    print("[ERR]", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
pid = j.get("prompt_id")
print(f"[v184] pid={pid}", flush=True)

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
            fn = imgs[0]["filename"]; sub = imgs[0].get("subfolder", "")
            url = f"{v164.COMFYUI}/view?filename={fn}&type=output&subfolder={sub}"
            data = requests.get(url, timeout=60).content
            out = OUT / "v184_camo_4.jpg"
            out.write_bytes(data)
            from PIL import Image, ImageFilter
            im = Image.open(out).convert("RGB")
            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            sharp.save(out, quality=95)
            print(f"[v184] ✓ saved {out} {out.stat().st_size/1024/1024:.1f}MB")
            done = True; break
    if "error" in st:
        print("[ERR]", st); break

if not done:
    print("[ERR] timeout"); sys.exit(1)

# === 定量自检（我无法肉眼看图，用指标校验配色保真）===
import numpy as np
from PIL import Image
sys.path.insert(0, str(HERE))
from selfcheck_metrics import color_intersect, ssim
o = str(HERE.parent / "web_gallery" / "img" / "orig_camo_4.jpg")
f = str(out)
ci = color_intersect(o, f)
sm = ssim(o, f)
print(f"[v184] color_intersect(orig vs v184) = {ci:.3f}  (配色保真, 越近1越好)")
print(f"[v184] ssim(orig vs v184)            = {sm:.3f}  (构图/结构)")

# === 拼图对照：原图 | v174(干净基线) | v184 ===
orig = Image.open(HERE.parent / "web_gallery" / "img" / "orig_camo_4.jpg").convert("RGB")
v174_img = Image.open(HERE.parent / "outputs" / "v174" / "v174_camo_4.jpg").convert("RGB")
v184_img = Image.open(out).convert("RGB")
H = 760
def fit(im):
    w, h = im.size; return im.resize((int(w*H/h), H), Image.LANCZOS)
o_r, v174_r, v184_r = fit(orig), fit(v174_img), fit(v184_img)
gap = 14
W = o_r.width + v174_r.width + v184_r.width + gap*2
canvas = Image.new("RGB", (W, H+50), (14,15,18))
canvas.paste(o_r, (0,50)); canvas.paste(v174_r, (o_r.width+gap,50)); canvas.paste(v184_r, (o_r.width+v174_r.width+gap*2,50))
d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype("arial.ttf", 22)
except Exception: font = ImageFont.load_default()
labels = [("ORIGINAL", (0,12)), ("v174 (LORA=0 干净基线)", (o_r.width+gap,12)), (f"v184 自然裂变 ci={ci:.2f}", (o_r.width+v174_r.width+gap*2,12))]
for txt,(x,y) in labels:
    col = (255,106,61) if "v184" in txt else ((230,234,240) if "ORIGINAL" in txt else (154,160,171))
    d.text((x+6,y), txt, fill=col, font=font)
cmp = OUT / "compare_orig_v174_v184.jpg"
canvas.save(cmp, quality=92)
print(f"[v184] ✓ 对照 saved {cmp}")

# 同步画廊
gallery_fiss = HERE.parent / "web_gallery" / "img" / "fiss_camo_4.jpg"
shutil.copyfile(out, gallery_fiss)
print(f"[v184] ✓ 更新画廊 fiss_camo_4.jpg")
