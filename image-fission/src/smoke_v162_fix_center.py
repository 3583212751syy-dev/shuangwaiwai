"""
v162 - 修 v161 stageA 中央混乱区
问题: v161 stageA 中央被 AI 融合成"鹰头+骷髅+人脸+皮夹克+刀"混乱体
方案: 对中央区域(x=0.22-0.78, y=0.30-0.76)单独 InpaintModelConditioning
      高 denoise(0.85) + 极简 prompt(单骷髅+小铁冠) 强制替换
      不动好的部分(翅膀/火焰/铁链/底部3骷髅/顶部小鹰)
"""
import json, time, sys, shutil
from pathlib import Path
import requests
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageFont
import numpy as np

COMFYUI  = "http://127.0.0.1:8188"
ROOT     = Path("E:/Desktop/双接口/image-fission")
INPUT_DIR = ROOT / "ComfyUI" / "input"
JOBS     = ROOT / "jobs" / "smoke_v162"
JOBS.mkdir(parents=True, exist_ok=True)
OUTPUTS  = Path("C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v162")
OUTPUTS.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"  # 与 v161 一致 (Proteus v0.4 哥特/矢量风强)
USM = dict(radius=2, percent=140, threshold=3)

# --- 中央混乱区 mask 范围（1024 归一化）
# 1024 坐标: x=225-799, y=307-778  -> 归一化 x=0.22-0.78, y=0.30-0.76
CENTER_RECT = dict(x=0.22, y=0.30, w=0.56, h=0.46)

# --- 干净的中心 prompt: 一个哥特骷髅+小铁冠
A_CENTER_POS = (
    "a single clean weathered human skull with deep realistic cracks and fractures on the bone, "
    "wearing a SIMPLE small gothic iron crown with exactly 5 sharp upward spikes, "
    "dark empty eye sockets, detailed sharp teeth, "
    "centered in frame, surrounded by pure black empty background, "
    "no birds, no wings, no feathers, no beak, no faces, no skin, no hair, no beard, "
    "no weapons, no knife, no dagger, no sword, no axe, no scepter, no staff, "
    "no jacket, no leather, no clothing, no fabric, no shirt, "
    "no flames, no fire, no chains, no other skulls, no people, no figures, no bodies, "
    "no extra elements, no decoration, no text, no watermark, "
    "simple clean composition, dark gothic vector illustration style, "
    "high detail, sharp lines, monochromatic light gray skull on pure black background"
)

N_CENTER = (
    "blurry, deformed, distorted, extra fingers, extra limbs, watermark, signature, text, "
    "eagle, bird, raven, hawk, wings, feathers, beak, talons, "
    "face, human face, skin, hair, long hair, beard, mustache, "
    "knife, dagger, sword, weapon, axe, scepter, staff, gun, blade, "
    "leather, jacket, clothing, fabric, shirt, vest, armor, hood, cape, "
    "multiple skulls, two skulls, three skulls, extra skulls, "
    "flames, fire, chain, jewelry, crown with gems, crown with jewels, ornate crown, "
    "colorful, colored, bright, neon, pastel, "
    "people, person, figure, character, body, hands, arms, "
    "background detail, scenery, landscape, architecture, building, "
    "blurry, soft, low quality, lowres, jpeg artifacts, noise"
)

def make_center_mask(in_png: Path, out_png: Path):
    """画中央矩形 mask: 黑色=保留, 白色=重画; 边缘羽化 25px"""
    im = Image.open(in_png).convert("RGB")
    W, H = im.size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    x0 = int(CENTER_RECT["x"] * W)
    y0 = int(CENTER_RECT["y"] * H)
    x1 = int((CENTER_RECT["x"] + CENTER_RECT["w"]) * W)
    y1 = int((CENTER_RECT["y"] + CENTER_RECT["h"]) * H)
    draw.rectangle([x0, y0, x1, y1], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=20))  # 边缘羽化
    mask.save(out_png)
    print(f"  [MASK] {out_png.name} {W}x{H}  rect=({x0},{y0})-({x1},{y1})  {out_png.stat().st_size/1024:.0f}KB")

def build_inpaint_center(input_name, mask_name, prompt, neg, seed, prefix):
    """单步 InpaintModelConditioning(VAEEncode/inpaint/KSampler/VAEDecode/Save)"""
    g = {}
    g["1"]  = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"]  = {"class_type": "LoadImage",      "inputs": {"image": input_name}}
    g["3"]  = {"class_type": "LoadImageMask",   "inputs": {"image": mask_name, "channel": "red"}}
    g["4"]  = {"class_type": "VAEEncode",      "inputs": {"pixels": ["2",0], "vae": ["1",2]}}
    g["5"]  = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1",1], "text": prompt}}
    g["6"]  = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1",1], "text": neg}}
    g["7"]  = {"class_type": "InpaintModelConditioning", "inputs": {
        "positive": ["5",0], "negative": ["6",0], "vae": ["1",2],
        "pixels": ["2",0], "mask": ["3",0], "noise_mask": True}}
    g["8"]  = {"class_type": "KSampler", "inputs": {
        "model": ["1",0], "positive": ["7",0], "negative": ["7",1], "latent_image": ["7",2],
        "seed": seed, "steps": 32, "cfg": 6.5, "sampler_name": "euler",
        "scheduler": "normal", "denoise": 0.85}}  # 高 denoise 完全替换 mask 区
    g["9"]  = {"class_type": "VAEDecode", "inputs": {"samples": ["8",0], "vae": ["1",2]}}
    g["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9",0], "filename_prefix": prefix}}
    return g

def build_upscale(input_name, prefix):
    g = {}
    g["1"] = {"class_type": "LoadImage", "inputs": {"image": input_name}}
    g["2"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["3"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2",0], "image": ["1",0]}}
    g["4"] = {"class_type": "SaveImage", "inputs": {"images": ["3",0], "filename_prefix": prefix}}
    return g

def submit_and_fetch(g, reads, tag, hard_timeout=300):
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v162_{int(time.time()*1000)}"}, timeout=20)
    try: j = r.json()
    except Exception: j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {json.dumps(j)[:800]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid: print(f"[ERR] {tag}: no pid", flush=True); return False
    print(f"  [{tag}] pid={pid}", flush=True)
    t0 = time.time()
    while time.time() - t0 < hard_timeout:
        time.sleep(4)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=8).json()
            if pid in h:
                rec = h[pid]
                if rec.get("status", {}).get("completed"):
                    ok = True
                    for (node_id, out_path, usm) in reads:
                        imgs = rec.get("outputs", {}).get(node_id, {}).get("images", [])
                        if not imgs: ok = False; continue
                        url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                        data = requests.get(url, timeout=90).content
                        out_path.write_bytes(data)
                        if usm:
                            im = Image.open(out_path).convert("RGB")
                            im = im.filter(ImageFilter.UnsharpMask(**USM))
                            im.save(out_path, "JPEG", quality=95, optimize=True)
                        print(f"    {out_path.name} {out_path.stat().st_size/1024/1024:.2f}MB", flush=True)
                    return ok
                if rec.get("status", {}).get("error"):
                    print(f"  [{tag}] COMFY错误", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] poll异常 {e}", flush=True)
    print(f"  [{tag}] TIMEOUT", flush=True); return False

def composite_center(fixed_center_png: Path, stageA_4x_png: Path, out_png: Path, mask_png: Path):
    """把 inpaint 修好的中心,按 mask 贴回 4x stageA"""
    fixed = Image.open(fixed_center_png).convert("RGB")  # 1024
    stageA_4x = Image.open(stageA_4x_png).convert("RGB")  # 4096
    W4, H4 = stageA_4x.size
    # inpaint 输出的 mask 区与原图同位置, 缩放到 4x
    fixed_up = fixed.resize((W4, H4), Image.LANCZOS)
    mask = Image.open(mask_png).convert("L").resize((W4, H4), Image.LANCZOS)
    # 用 mask alpha 合成
    out = Image.composite(fixed_up, stageA_4x, mask)
    out.save(out_png, "JPEG", quality=95, optimize=True)
    print(f"  [COMPOSITE] {out_png.name} {W4}x{H4} {out_png.stat().st_size/1024/1024:.2f}MB", flush=True)

def main():
    # 1. 准备: 复制 v161 stageA 到 v162 jobs 目录和 ComfyUI input
    src_work = ROOT / "jobs" / "smoke_v161" / "stageA_eagle_2_work.png"
    src_4x   = ROOT / "jobs" / "smoke_v161" / "stageA_eagle_2.jpg"
    if not src_work.exists():
        print(f"[ERR] 缺 {src_work}"); return

    work_local = JOBS / "stageA_work.png"
    work_4x    = JOBS / "stageA_4x.jpg"
    if not work_local.exists(): shutil.copy(src_work, work_local)
    if not work_4x.exists():    shutil.copy(src_4x, work_4x)

    # 2. 生成中心 mask
    mask_local = JOBS / "center_mask.png"
    if not mask_local.exists():
        make_center_mask(work_local, mask_local)

    # 3. 复制到 ComfyUI input
    in_work = INPUT_DIR / "v162_stageA_work.png"
    in_mask = INPUT_DIR / "v162_center_mask.png"
    shutil.copy(work_local, in_work)
    shutil.copy(mask_local, in_mask)
    print(f"  [INPUT] {in_work.name} {in_work.stat().st_size/1024:.0f}KB", flush=True)
    print(f"  [INPUT] {in_mask.name} {in_mask.stat().st_size/1024:.0f}KB", flush=True)

    # 4. 提交 inpaint
    fixed_local = JOBS / "center_fixed.png"
    if not fixed_local.exists() or fixed_local.stat().st_size < 10000:
        print(">> Inpaint center", flush=True)
        g = build_inpaint_center(in_work.name, in_mask.name, A_CENTER_POS, N_CENTER, 77889900, "v162_center")
        if not submit_and_fetch(g, [("10", fixed_local, False)], "v162/inpaint"):
            print("Inpaint 失败"); return
    else:
        print(f"  center_fixed 已存在 {fixed_local.stat().st_size/1024:.0f}KB 跳过", flush=True)

    # 5. 把修好的中心按 mask 贴回 4x stageA
    composite_4x = JOBS / "stageA_center_fixed.jpg"
    if not composite_4x.exists():
        composite_center(fixed_local, work_4x, composite_4x, mask_local)

    # 6. USM 锐化
    final_local = JOBS / "final_eagle_2.jpg"
    if not final_local.exists():
        im = Image.open(composite_4x).convert("RGB")
        im = im.filter(ImageFilter.UnsharpMask(**USM))
        im.save(final_local, "JPEG", quality=95, optimize=True)
        print(f"  [FINAL] {final_local.name} {final_local.stat().st_size/1024/1024:.2f}MB", flush=True)

    # 7. 烧字 DOMINION (底部居中)
    burn = JOBS / "eagle_2_burned.jpg"
    if not burn.exists():
        im = Image.open(final_local).convert("RGB")
        W, H = im.size
        # 哥特字体: PirataOne
        font_path = Path("C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/scripts/fonts/PirataOne-Regular.ttf")
        if not font_path.exists():
            # 退到系统字体
            font_path = Path("C:/Windows/Fonts/PirataOne-Regular.ttf")
        if not font_path.exists():
            font_path = Path("C:/Windows/Fonts/segoeui.ttf")  # 兜底
        # 字号: 图宽 8% (4096 -> 330)
        fs = int(W * 0.085)
        try:
            font = ImageFont.truetype(str(font_path), fs)
        except Exception as e:
            print(f"  [FONT] 加载失败用默认: {e}"); font = ImageFont.load_default()
        text = "DOMINION"
        # 居中, y 在 92% 高度
        try:
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(text) * fs // 2, fs
        tx = (W - tw) // 2
        ty = int(H * 0.92)
        # 阴影 + 描边
        shadow = Image.new("RGBA", im.size, (0,0,0,0))
        sd = ImageDraw.Draw(shadow)
        sd.text((tx+4, ty+4), text, font=font, fill=(0,0,0,200))
        im_rgba = im.convert("RGBA")
        im_rgba.alpha_composite(shadow)
        # 描边
        stroke = Image.new("RGBA", im.size, (0,0,0,0))
        sd2 = ImageDraw.Draw(stroke)
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            sd2.text((tx+dx, ty+dy), text, font=font, fill=(0,0,0,255))
        im_rgba.alpha_composite(stroke)
        # 主字: 银白
        main = Image.new("RGBA", im.size, (0,0,0,0))
        md = ImageDraw.Draw(main)
        md.text((tx, ty), text, font=font, fill=(220,220,230,255))
        im_rgba.alpha_composite(main)
        im_out = im_rgba.convert("RGB")
        im_out.save(burn, "JPEG", quality=95, optimize=True)
        print(f"  [BURN] {burn.name} {burn.stat().st_size/1024/1024:.2f}MB", flush=True)

    # 8. 复制到 outputs/ 和桌面
    desktop = Path("E:/Desktop/双接口/image-fission/outputs/image-fission-v162-eagle_2.jpg")
    shutil.copy(burn, OUTPUTS / "eagle_2_final.jpg")
    shutil.copy(burn, desktop)
    print(f"  [OUT] {desktop}", flush=True)

    # 9. 拼图
    IM = Image
    compare = JOBS / "compare_eagle_2_v162.png"
    orig = ROOT / "jobs" / "smoke_v161" / "_original_eagle_2.jpg"
    if not orig.exists():
        # 退到 v127 原始
        orig = Path("E:/Desktop/图裂变测试图/pinterest_eagle_2.jpg")
    parts = [("原图", orig), ("Stage A (失真)", JOBS/"stageA_4x.jpg"), ("v162 修复后", burn)]
    imgs = [IM.open(p).convert("RGB") for _, p in parts if p.exists()]
    if len(imgs) == 3:
        # 统一高度
        H = 800
        resized = []
        for im in imgs:
            w = int(im.width * H / im.height)
            resized.append(im.resize((w, H), IM.LANCZOS))
        gap = 20
        total_w = sum(im.width for im in resized) + gap * (len(resized)-1)
        canvas = IM.new("RGB", (total_w, H+40), (30,30,30))
        x = 0
        d = ImageDraw.Draw(canvas)
        for i, (im, (label, _)) in enumerate(zip(resized, parts)):
            canvas.paste(im, (x, 40))
            d.text((x+10, 8), label, fill=(255,255,255))
            x += im.width + gap
        canvas.save(compare, "JPEG", quality=92, optimize=True)
        print(f"  [COMPARE] {compare.name} {compare.stat().st_size/1024/1024:.2f}MB", flush=True)

    # 10. 指标
    def load(p): return np.asarray(Image.open(p).convert("L")).astype(np.float64)
    def black(p, thr=24):
        im = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
        return float((im.mean(axis=2) < thr).mean())
    def sharp(p):
        im = np.asarray(Image.open(p).convert("L")).astype(np.float64)
        gx = np.gradient(im, axis=1); gy = np.gradient(im, axis=0)
        return float((gx**2+gy**2).mean())
    print(f"\n[指标]  v162 vs v161 stageA:")
    print(f"  原图黑底={black(orig):.3f} 锐度={sharp(orig):.0f}")
    print(f"  v161 stageA: 黑底={black(JOBS/'stageA_4x.jpg'):.3f} 锐度={sharp(JOBS/'stageA_4x.jpg'):.0f}")
    print(f"  v162 修复后: 黑底={black(burn):.3f} 锐度={sharp(burn):.0f}")

if __name__ == "__main__":
    main()
