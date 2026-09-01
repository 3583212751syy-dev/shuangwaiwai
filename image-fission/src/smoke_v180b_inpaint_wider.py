"""v180b ControlNet inpaint 局部重绘（修 v180 mask 范围不够 + inpaint 区偏差 bug）：

v180 bug：
  ① mask x=0.18..0.82 没覆盖到左侧 x<0.18 的 D 字母 → 字母还在
  ② inpaint 区(顶部右上)小蝴蝶被 SDXL 改成金黄色 → 配色漂移

修复（单变量）：
  ① mask 扩展到 x=0.05..0.95, y=0.00..0.25，覆盖整个顶部字母区
  ② prompt 加强：明确"the inpaint area must seamlessly blend into the surrounding white background
    and remain completely separate from the small butterflies in the upper-right"，
    避免 SDXL 改变 inpaint 区以外的元素
  ③ grow_mask_by 从 6 提到 10，mask 边缘过渡更宽

参数锁死：denoise=1.0 / steps=30 / cfg=7.0 / seed=700403 / ProteusV0.4 /
                controlnet-inpaint-dreamer-sdxl fp16 / ImageScale lanczos 3328x5888
"""
import sys, time, json, requests, shutil, copy
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164

OUT = HERE.parent / "outputs" / "v180b"
OUT.mkdir(parents=True, exist_ok=True)
FINAL = HERE.parent / "outputs" / "final"
GALLERY = HERE.parent / "web_gallery" / "img"
COMFY = v164.COMFYUI
COMFY_INPUT = HERE.parent / "ComfyUI" / "input"

CN_INPAINT = "controlnet-inpaint-dreamer-sdxl.fp16.safetensors"
CN_INPAINT_PATH = HERE.parent / "ComfyUI" / "models" / "controlnet" / CN_INPAINT

assert CN_INPAINT_PATH.exists(), f"ControlNet inpaint 模型不存在: {CN_INPAINT_PATH}"

# v179b 输入
SRC_IMG = HERE.parent / "outputs" / "v179b" / "v179b_denim_3.jpg"
SRC_MASK = HERE.parent / "outputs" / "v179b" / "mask_top_letters.png"
assert SRC_IMG.exists() and SRC_MASK.exists()

# 单变量 ①：mask 范围扩大（覆盖整个顶部字母区 + 左边 D）
mask = Image.new("L", Image.open(SRC_IMG).size, 0)
W0, H0 = Image.open(SRC_IMG).size
x0 = int(W0 * 0.05); x1 = int(W0 * 0.95)
y0 = int(H0 * 0.00); y1 = int(H0 * 0.25)
ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(40))  # 加大羽化让边缘过渡自然
mask.save(SRC_MASK)  # 覆盖回原 mask
print(f"[v180b] ① mask 扩大: x=[{x0}..{x1}], y=[{y0}..{y1}] (覆盖顶部 25% 高度 + 左 5% 到右 5%)")

# 降级到 SDXL 工作尺寸
WORK_W, WORK_H = 1024, 1810
small_img = Image.open(SRC_IMG).resize((WORK_W, WORK_H), Image.LANCZOS)
small_img.save(COMFY_INPUT / "v179b_small_for_inpaint.jpg", quality=95)
small_mask = mask.resize((WORK_W, WORK_H), Image.LANCZOS)
small_mask.save(COMFY_INPUT / "mask_top_letters_v180b.png")
print(f"[v180b] 降级到 {WORK_W}x{WORK_H}")

# 单变量 ②：prompt 加强隔离
EXTRA_NEG = (
    ", denim letter, denim letters, fabric letter, fabric letters, stitched letter, stitched letters, "
    "alphabet patch, alphabet shape, character shape, character outline, "
    "fabric typography, denim typography, denim word, fabric word, fabric alphabet, denim alphabet, "
    "embroidered letter, embroidered letters, appliqued letter, appliqued letters, "
    "text made of fabric, text made of denim, letters made of denim, letters made of fabric, "
    "3D fabric letter, 3D denim letter, fabric text, denim text, "
    "yellow color, golden color, mustard color, amber color, orange color, "
    "color shift away from indigo blue, different color palette, warm tones"
)
POS = (
    "a CLEAN FLAT EMPTY WHITE-AND-DENIM BORDER AREA at the top of the image, "
    "ABSOLUTELY NO TEXT, NO LETTERS, NO ALPHABET SHAPES, NO CHARACTER OUTLINES, "
    "no word shapes, no logo, no number, no marks of any kind, "
    "the inpaint region is a smooth flat EMPTY space that seamlessly blends "
    "into the surrounding PURE WHITE BACKGROUND of the original image, "
    "preserving the EXACT same white background color (#FFFFFF) as the surrounding area, "
    "the inpaint region must remain COMPLETELY SEPARATE from the small butterflies in the upper-right "
    "— do not modify the small butterflies, do not change their colors, do not introduce yellow or golden tones, "
    "the small butterflies stay as they are in indigo denim blue, "
    "the only fabric allowed in the inpaint region is a small flat horizontal "
    "indigo DENIM FABRIC BORDER STRIP at the very top edge, "
    "smooth flat rectangular piece of denim fabric with diagonal TWILL WEAVE texture "
    "and pale-yellow DOUBLE-NEEDLE TOP-STITCHED BORDER around the edges, "
    "rounded stitched corners, like a real clothing label edge, "
    "everything ELSE in the inpaint region is simply empty flat pure WHITE BACKGROUND, "
    "natural fabric texture, premium fashion textile print quality"
)
NEG = v164.NEG_BASE + EXTRA_NEG


def build():
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v179b_small_for_inpaint.jpg"}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": "mask_top_letters_v180b.png"}}
    g["4"] = {"class_type": "ImageToMask", "inputs": {"image": ["3", 0], "channel": "red"}}
    g["5"] = {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": CN_INPAINT}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}}
    g["6"] = {"class_type": "ControlNetApply",
              "inputs": {"conditioning": ["pg", 0], "control_net": ["5", 0],
                         "image": ["2", 0], "strength": 1.0}}
    g["7"] = {"class_type": "VAEEncodeForInpaint",
              "inputs": {"pixels": ["2", 0], "vae": ["1", 2], "mask": ["4", 0],
                         "grow_mask_by": 10}}  # 加大 grow_mask_by 让边缘更宽
    g["8"] = {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["ng", 0],
                         "latent_image": ["7", 0], "seed": 700403,
                         "steps": 30, "cfg": 7.0,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 1.0}}
    g["9"] = {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["1", 2]}}
    g["16"] = {"class_type": "ImageScale",
               "inputs": {"image": ["9", 0], "width": 3328, "height": 5888,
                          "upscale_method": "lanczos", "crop": "disabled"}}
    g["15"] = {"class_type": "SaveImage",
               "inputs": {"images": ["16", 0], "filename_prefix": "v180b_denim_3"}}
    return g


print(f"\n[v180b] 单变量 ① mask 扩大 (x=0.05..0.95, y=0.00..0.25)")
print(f"[v180b] 单变量 ② prompt 加强隔离 + 禁黄/金色系")
print(f"[v180b] seed=700403 (避开 v180)")
print(f"[v180b] 提交 ...", flush=True)

g = build()
r = requests.post(f"{COMFY}/prompt",
                 json={"prompt": g, "client_id": f"v180b_{int(time.time())}"},
                 timeout=15)
j = r.json()
if r.status_code != 200 or "error" in j:
    print("[ERR]", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
pid = j.get("prompt_id")
print(f"[v180b] pid={pid}", flush=True)

done = False
for i in range(120):
    time.sleep(5)
    h = requests.get(f"{COMFY}/history/{pid}", timeout=10).json()
    if pid not in h:
        if i % 6 == 0: print(f"  ...{i*5}s", flush=True)
        continue
    rec = h[pid]
    st = rec.get("status", {})
    if st.get("completed"):
        imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
        if imgs:
            fn = imgs[0]["filename"]
            sub = imgs[0].get("subfolder", "")
            data = requests.get(f"{COMFY}/view?filename={fn}&type=output&subfolder={sub}", timeout=60).content
            out = OUT / "v180b_denim_3.jpg"
            out.write_bytes(data)
            im = Image.open(out).convert("RGB")
            im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)).save(out, quality=95)
            print(f"[v180b] ✓ {out} {out.stat().st_size/1024/1024:.1f}MB")
            done = True
            break
    if "error" in st:
        print("[ERR]", st); break

if not done:
    print("[ERR] timeout"); sys.exit(1)

# 复制到 final + gallery（覆盖 v180 的输出）
shutil.copyfile(out, FINAL / "denim_3.jpg")
shutil.copyfile(out, GALLERY / "fiss_denim_3.jpg")
print(f"[v180b] ✓ denim_3 → final + gallery（覆盖 v180）")

# 自检
from selfcheck_metrics import color_intersect, ssim, frag_ratio
orig = GALLERY / "orig_denim_3.jpg"
ci = color_intersect(str(orig), str(out))
sm = ssim(str(orig), str(out))
fr = frag_ratio(str(orig), str(out))
print(f"[v180b] denim_3 metrics: color∩={ci:.3f}  ssim={sm:.3f}  fragR={fr:.3f}")

# camo 仍用 v174
v174_camo = HERE.parent / "outputs" / "v174" / "v174_camo_4.jpg"
shutil.copyfile(v174_camo, FINAL / "camo_4.jpg")
shutil.copyfile(v174_camo, GALLERY / "fiss_camo_4.jpg")
orig_camo = GALLERY / "orig_camo_4.jpg"
ci_c = color_intersect(str(orig_camo), str(v174_camo))
sm_c = ssim(str(orig_camo), str(v174_camo))
fr_c = frag_ratio(str(orig_camo), str(v174_camo))
print(f"[v180b] camo_4  metrics: color∩={ci_c:.3f}  ssim={sm_c:.3f}  fragR={fr_c:.3f}  (← v174)")

print("\n[v180b] DONE")
print(f"  final/camo_4.jpg   ← v174 (圆润+配色)")
print(f"  final/denim_3.jpg  ← v180b (CLEAN 蝴蝶 + 真牛仔布料 + 顶部无字母)")