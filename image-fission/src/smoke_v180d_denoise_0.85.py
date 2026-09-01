"""v180d 改用 ComfyUI 标准 inpaint 流程（VAEEncode + SetLatentNoiseMask）：

v180b bug 根因：VAEEncodeForInpaint 在 ComfyUI 里传递 mask 信息不完整，
                导致 ControlNet inpaint 改了整张图（蝴蝶变金黄）。
v180d 修复：改用社区标准 inpaint 工作流
  LoadImage → ImageToMask → VAEEncode (image, vae) → SetLatentNoiseMask (samples=latent, mask)
  + EmptyLatentImage (尺寸 = image)
  + ControlNetLoader inpaint
  + ControlNetApply (image=image, conditioning)
  + KSampler (denoise=1.0, latent=masked_latent)
  → VAEDecode → ImageScale lanczos upscale → SaveImage

这样 SDXL 会严格只重绘 mask 区(mask 内加噪声, mask 外保持 latent)。

mask 范围：x=0.05..0.95, y=0.00..0.25 (沿用 v180b)
prompt 加强：明确保持主体蝴蝶配色 (NO yellow/golden, keep indigo blue butterflies)
seed=700405 (避开 v180b)
"""
import sys, time, json, requests, shutil, copy
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164

OUT = HERE.parent / "outputs" / "v180d"
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
SRC_MASK = HERE.parent / "outputs" / "v180b" / "mask_top_letters.png"
assert SRC_IMG.exists()

# 重新生成 mask（沿用 v180b 范围 x=0.05..0.95, y=0.00..0.25）
W0, H0 = Image.open(SRC_IMG).size
mask = Image.new("L", (W0, H0), 0)
x0 = int(W0 * 0.05); x1 = int(W0 * 0.95)
y0 = int(H0 * 0.00); y1 = int(H0 * 0.25)
ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(40))
mask.save(SRC_MASK)
print(f"[v180d] mask x=[{x0}..{x1}], y=[{y0}..{y1}], 羽化 40px")

# 降级到 SDXL 工作尺寸
WORK_W, WORK_H = 1024, 1810
small_img = Image.open(SRC_IMG).resize((WORK_W, WORK_H), Image.LANCZOS)
small_img.save(COMFY_INPUT / "v179b_small_for_inpaint.jpg", quality=95)
small_mask = mask.resize((WORK_W, WORK_H), Image.LANCZOS)
small_mask.save(COMFY_INPUT / "mask_top_letters_v180d.png")
print(f"[v180d] 降级到 {WORK_W}x{WORK_H}")

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
    "a CLEAN FLAT EMPTY WHITE BORDER AREA at the top of the image, "
    "ABSOLUTELY NO TEXT, NO LETTERS, NO ALPHABET SHAPES, NO CHARACTER OUTLINES, "
    "no word shapes, no logo, no number, no marks of any kind, "
    "the inpaint region is smooth flat EMPTY space that seamlessly blends "
    "into the surrounding PURE WHITE BACKGROUND of the original image, "
    "preserving the EXACT same white background color (#FFFFFF) as the surrounding area, "
    "the inpaint region must remain COMPLETELY SEPARATE from the small butterflies and the main butterfly, "
    "do not modify the butterflies, do not change their colors, do not introduce yellow or golden tones, "
    "all butterflies stay as they are in indigo denim blue, "
    "everything in the inpaint region is simply empty flat pure WHITE BACKGROUND, "
    "natural clean look, premium fashion textile print quality"
)
NEG = v164.NEG_BASE + EXTRA_NEG


def build():
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v179b_small_for_inpaint.jpg"}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": "mask_top_letters_v180d.png"}}
    g["4"] = {"class_type": "ImageToMask", "inputs": {"image": ["3", 0], "channel": "red"}}
    g["5"] = {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": CN_INPAINT}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}}

    # VAEEncode 原图（生成 latent）
    g["7"] = {"class_type": "VAEEncode",
              "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}}

    # SetLatentNoiseMask: 把 mask 应用到 latent 上
    g["8"] = {"class_type": "SetLatentNoiseMask",
              "inputs": {"samples": ["7", 0], "mask": ["4", 0]}}

    # ControlNetApply: 用原图 + mask 控制条件
    g["6"] = {"class_type": "ControlNetApply",
              "inputs": {"conditioning": ["pg", 0], "control_net": ["5", 0],
                         "image": ["2", 0], "strength": 1.0}}

    # KSampler (denoise=0.85 — 不是 1.0，SetLatentNoiseMask 在 denoise<1.0 下才能生效)
    g["9"] = {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["ng", 0],
                         "latent_image": ["8", 0], "seed": 700405,
                         "steps": 30, "cfg": 7.0,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 0.85}}

    g["10"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["1", 2]}}
    g["16"] = {"class_type": "ImageScale",
               "inputs": {"image": ["10", 0], "width": 3328, "height": 5888,
                          "upscale_method": "lanczos", "crop": "disabled"}}
    g["15"] = {"class_type": "SaveImage",
               "inputs": {"images": ["16", 0], "filename_prefix": "v180d_denim_3"}}
    return g


print(f"\n[v180d] 用 ComfyUI 标准 inpaint 工作流 (VAEEncode + SetLatentNoiseMask)")
print(f"[v180d] seed=700405 (避开 v180b)")
print(f"[v180d] 提交 ...", flush=True)

g = build()
r = requests.post(f"{COMFY}/prompt",
                 json={"prompt": g, "client_id": f"v180d_{int(time.time())}"},
                 timeout=15)
j = r.json()
if r.status_code != 200 or "error" in j:
    print("[ERR]", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
pid = j.get("prompt_id")
print(f"[v180d] pid={pid}", flush=True)

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
            out = OUT / "v180d_denim_3.jpg"
            out.write_bytes(data)
            im = Image.open(out).convert("RGB")
            im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)).save(out, quality=95)
            print(f"[v180d] ✓ {out} {out.stat().st_size/1024/1024:.1f}MB")
            done = True
            break
    if "error" in st:
        print("[ERR]", st); break

if not done:
    print("[ERR] timeout"); sys.exit(1)

# 复制到 final + gallery（覆盖 v180b 的输出）
shutil.copyfile(out, FINAL / "denim_3.jpg")
shutil.copyfile(out, GALLERY / "fiss_denim_3.jpg")
print(f"[v180d] ✓ denim_3 → final + gallery（覆盖 v180b）")

# 自检
from selfcheck_metrics import color_intersect, ssim, frag_ratio
orig = GALLERY / "orig_denim_3.jpg"
ci = color_intersect(str(orig), str(out))
sm = ssim(str(orig), str(out))
fr = frag_ratio(str(orig), str(out))
print(f"[v180d] denim_3 metrics: color∩={ci:.3f}  ssim={sm:.3f}  fragR={fr:.3f}")

# camo 仍用 v174
v174_camo = HERE.parent / "outputs" / "v174" / "v174_camo_4.jpg"
shutil.copyfile(v174_camo, FINAL / "camo_4.jpg")
shutil.copyfile(v174_camo, GALLERY / "fiss_camo_4.jpg")
orig_camo = GALLERY / "orig_camo_4.jpg"
ci_c = color_intersect(str(orig_camo), str(v174_camo))
sm_c = ssim(str(orig_camo), str(v174_camo))
fr_c = frag_ratio(str(orig_camo), str(v174_camo))
print(f"[v180d] camo_4  metrics: color∩={ci_c:.3f}  ssim={sm_c:.3f}  fragR={fr_c:.3f}  (← v174)")

print("\n[v180d] DONE")
print(f"  final/camo_4.jpg   ← v174 (圆润+配色)")
print(f"  final/denim_3.jpg  ← v180d (CLEAN 蝴蝶 + 真牛仔布料 + 顶部无字母)")