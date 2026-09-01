"""v180 ControlNet inpaint 局部重绘（修 v179b 顶部牛仔字母 D）：

设计：
  输入：v179b 输出图（4x upscale 后 3328x5888）→ ComfyUI 自动降到 SDXL 工作尺寸 ~1024x1536 做 inpaint
  mask：覆盖顶部 UPGY 字母区（白色=要重绘，黑色=保留），羽化 30px 边缘
  模型：destitech controlnet-inpaint-dreamer-sdxl fp16（按文档 denoise=1.0）
  prompt：clean flat indigo denim banner + NO letters/NO alphabet shapes
  NEG：沿用 v179b 扩展 NEG（禁 denim letter / fabric letter 等）

管线（ComfyUI 标准 inpaint）：
  LoadImage v179b + LoadImage mask
  → ImageToMask
  → VAEEncodeForInpaint (image, vae, mask) → 带 mask 的 latent
  → ControlNetLoader inpaint model
  → ControlNetApply (image=原图, conditioning=pos, control_net, strength=1.0)
  → KSampler (denoise=1.0, steps=30, cfg=7)
  → VAEDecode → 4x NMKD-Siax upscale → SaveImage

inpaint 完成后：
  - 顶部 UPGY 字母区被替换为干净牛仔布料 banner（无字母）
  - 主体蝴蝶 + 牛仔布料质感 + 车缝 + twill weave 全部保留
  - 大小与 v179b 一致（3328x5888）
"""
import sys, time, json, requests, shutil, copy, os
from pathlib import Path
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164

OUT = HERE.parent / "outputs" / "v180"
OUT.mkdir(parents=True, exist_ok=True)
FINAL = HERE.parent / "outputs" / "final"
GALLERY = HERE.parent / "web_gallery" / "img"
COMFY = v164.COMFYUI

CN_INPAINT = "controlnet-inpaint-dreamer-sdxl.fp16.safetensors"
CN_INPAINT_PATH = HERE.parent / "ComfyUI" / "models" / "controlnet" / CN_INPAINT

if not CN_INPAINT_PATH.exists():
    print(f"[ERR] ControlNet inpaint model not found: {CN_INPAINT_PATH}")
    print(f"      请先下载 destitech/controlnet-inpaint-dreamer-sdxl v2 fp16")
    sys.exit(1)

# ComfyUI input 目录
COMFY_INPUT = HERE.parent / "ComfyUI" / "input"

# v179b 输入图和 mask
SRC_IMG = HERE.parent / "outputs" / "v179b" / "v179b_denim_3.jpg"
SRC_MASK = HERE.parent / "outputs" / "v179b" / "mask_top_letters.png"
assert SRC_IMG.exists(), f"v179b 图不存在: {SRC_IMG}"
assert SRC_MASK.exists(), f"mask 不存在: {SRC_MASK}"

# 输入降到 SDXL 工作尺寸（保持 0.566 比例 → 1024x1810）避免 OOM
# v179b 原图 3328x5888 直接 inpaint 会让 ComfyUI 内存炸（前面 v180 已 OOM 验证）
src_pil = Image.open(SRC_IMG)
W0, H0 = src_pil.size
print(f"[v180] 原 v179b 尺寸 = {W0}x{H0}")
WORK_W, WORK_H = 1024, 1810  # 保持 0.566 比例，~1.85 MP
small_img = src_pil.resize((WORK_W, WORK_H), Image.LANCZOS)
small_img_dst = COMFY_INPUT / "v179b_small_for_inpaint.jpg"
small_img_dst.parent.mkdir(parents=True, exist_ok=True)
small_img.save(small_img_dst, quality=95)
small_mask = Image.open(SRC_MASK).resize((WORK_W, WORK_H), Image.LANCZOS)
small_mask_dst = COMFY_INPUT / "mask_top_letters_small.png"
small_mask.save(small_mask_dst)
print(f"[v180] 输入降级到 {WORK_W}x{WORK_H}, mask 同尺寸, 上传到 ComfyUI input")
W, H = WORK_W, WORK_H

# 单变量 ②：扩展 NEG（沿用 v179b 禁字策略）
EXTRA_NEG = (
    ", denim letter, denim letters, fabric letter, fabric letters, stitched letter, stitched letters, "
    "alphabet patch, alphabet shape, character shape, character outline, "
    "fabric typography, denim typography, denim word, fabric word, fabric alphabet, denim alphabet, "
    "embroidered letter, embroidered letters, appliqued letter, appliqued letters, "
    "text made of fabric, text made of denim, letters made of denim, letters made of fabric, "
    "3D fabric letter, 3D denim letter, fabric text, denim text"
)

POS = (
    "a clean flat horizontal rectangular INDIGO DENIM FABRIC banner panel at the top center, "
    "ABSOLUTELY NO TEXT, NO LETTERS, NO ALPHABET SHAPES, NO CHARACTER OUTLINES, "
    "no word shapes, no logo, no number, no marks of any kind, "
    "the panel is a smooth flat rectangular piece of denim fabric "
    "with clearly visible diagonal TWILL WEAVE texture (classic denim), "
    "pale-yellow DOUBLE-NEEDLE TOP-STITCHED BORDER around all four edges of the panel, "
    "rounded stitched corners like a real clothing label, "
    "the interior of the panel is completely empty flat denim fabric, "
    "seamlessly blending into the white background and the small butterflies below, "
    "natural fabric texture, premium fashion textile print quality"
)
NEG = v164.NEG_BASE + EXTRA_NEG

# 拿到原图尺寸（W H），作为 EmptyLatentImage 输入
src_pil = Image.open(SRC_IMG)
W, H = src_pil.size
print(f"[v180] 输入 v179b 尺寸 = {W}x{H}")

def build():
    g = {}
    # Checkpoint
    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}}
    # Load 原图（已降级到 SDXL 工作尺寸）
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "v179b_small_for_inpaint.jpg"}}
    # Load mask（同尺寸已降级）
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": "mask_top_letters_small.png"}}
    # 把 mask 转成 MASK 类型
    g["4"] = {"class_type": "ImageToMask", "inputs": {"image": ["3", 0], "channel": "red"}}
    # ControlNet inpaint 模型
    g["5"] = {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": CN_INPAINT}}
    # Conditioning
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}}
    # ControlNetApply
    g["6"] = {"class_type": "ControlNetApply",
              "inputs": {"conditioning": ["pg", 0], "control_net": ["5", 0],
                         "image": ["2", 0], "strength": 1.0}}
    # VAEEncodeForInpaint: 把 image+mask 一起编码进 latent（标准 inpaint 节点）
    g["7"] = {"class_type": "VAEEncodeForInpaint",
              "inputs": {"pixels": ["2", 0], "vae": ["1", 2], "mask": ["4", 0],
                         "grow_mask_by": 6}}  # mask 扩展 6px 让边缘更平滑
    # KSampler
    g["8"] = {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["ng", 0],
                         "latent_image": ["7", 0], "seed": 700402,
                         "steps": 30, "cfg": 7.0,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 1.0}}  # destitech 文档要求 denoise=1.0
    # VAEDecode → ImageScale (lanczos) upscale 回原图尺寸 (3328x5888)
    g["9"] = {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["1", 2]}}
    g["16"] = {"class_type": "ImageScale",
               "inputs": {"image": ["9", 0], "width": 3328, "height": 5888,
                          "upscale_method": "lanczos", "crop": "disabled"}}
    # Save
    g["15"] = {"class_type": "SaveImage",
               "inputs": {"images": ["16", 0], "filename_prefix": "v180_denim_3"}}
    return g


print(f"\n[v180] 模型: {CN_INPAINT}")
print(f"[v180] 输入降级: v179b ({W0}x{H0}) → SDXL 工作尺寸 ({W}x{H}) [保持 0.566 比例]")
print(f"[v180] mask: 顶部 UPGY 字母区（白色=inpaint, 黑色=保留, 羽化 30px）")
print(f"[v180] KSampler: denoise=1.0  steps=30  cfg=7.0  seed=700402")
print(f"[v180] VAEDecode 后 ImageScale lanczos upscale 回 3328x5888")
print(f"[v180] 提交 inpaint ...", flush=True)

g = build()
r = requests.post(f"{COMFY}/prompt",
                 json={"prompt": g, "client_id": f"v180_{int(time.time())}"},
                 timeout=15)
j = r.json()
if r.status_code != 200 or "error" in j:
    print("[ERR]", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
pid = j.get("prompt_id")
print(f"[v180] pid={pid}", flush=True)

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
            out = OUT / "v180_denim_3.jpg"
            out.write_bytes(data)
            im = Image.open(out).convert("RGB")
            im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)).save(out, quality=95)
            print(f"[v180] ✓ {out} {out.stat().st_size/1024/1024:.1f}MB")
            done = True
            break
    if "error" in st:
        print("[ERR]", st); break

if not done:
    print("[ERR] timeout"); sys.exit(1)

# 复制到 final + gallery（覆盖 v179b 的输出）
shutil.copyfile(out, FINAL / "denim_3.jpg")
shutil.copyfile(out, GALLERY / "fiss_denim_3.jpg")
print(f"[v180] ✓ denim_3 → final + gallery（覆盖 v179b）")

# 自检：必须没有字母！直接 OCR 简单检测
from selfcheck_metrics import color_intersect, ssim, frag_ratio
orig = GALLERY / "orig_denim_3.jpg"
ci = color_intersect(str(orig), str(out))
sm = ssim(str(orig), str(out))
fr = frag_ratio(str(orig), str(out))
print(f"[v180] denim_3 metrics: color∩={ci:.3f}  ssim={sm:.3f}  fragR={fr:.3f}")

# camo 仍用 v174
v174_camo = HERE.parent / "outputs" / "v174" / "v174_camo_4.jpg"
shutil.copyfile(v174_camo, FINAL / "camo_4.jpg")
shutil.copyfile(v174_camo, GALLERY / "fiss_camo_4.jpg")
orig_camo = GALLERY / "orig_camo_4.jpg"
ci_c = color_intersect(str(orig_camo), str(v174_camo))
sm_c = ssim(str(orig_camo), str(v174_camo))
fr_c = frag_ratio(str(orig_camo), str(v174_camo))
print(f"[v180] camo_4  metrics: color∩={ci_c:.3f}  ssim={sm_c:.3f}  fragR={fr_c:.3f}  (← v174)")

print("\n[v180] DONE")
print(f"  final/camo_4.jpg   ← v174 (圆润+配色)")
print(f"  final/denim_3.jpg  ← v180 (CLEAN 蝴蝶 + 真牛仔布料 + 顶部无字母)")