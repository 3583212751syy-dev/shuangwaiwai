"""
v163 - 主体回归：大秃鹰作 main subject
根因: v161/v162 反复把小元素(骷髅)当主体, v162 终端成了大骷髅
正确: 原图主体是大秃鹰展翅(占画面 60%), 裂变必须保留秃鹰作主体
      小元素(底部3骷髅/铁链/顶部小鹰)可换可改数量
      角度可改(原图正面 -> v163 改 3/4 侧脸)
策略: 复用 v161 stageA 作底, 中央定向 inpaint 改为大秃鹰
"""
import json, time, sys, shutil
from pathlib import Path
import requests
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np

COMFYUI  = "http://127.0.0.1:8188"
ROOT     = Path("E:/Desktop/双接口/image-fission")
INPUT_DIR = ROOT / "ComfyUI" / "input"
JOBS     = ROOT / "jobs" / "smoke_v163"
JOBS.mkdir(parents=True, exist_ok=True)
OUTPUTS  = Path("C:/Users/lenovo/WorkBuddy/2026-08-24-16-39-13/outputs/v163")
OUTPUTS.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
USM = dict(radius=2, percent=140, threshold=3)

# --- 中央 mask 范围（1024 归一化）—— 覆盖主体区
# 主体是大秃鹰: 头在中上 y=0.35-0.50, 身体 y=0.45-0.70, 翅膀根部 x=0.25-0.75
# 留出外侧翅膀尖 (x<0.20, x>0.80) 和顶部小鹰 (y<0.20) 保留
CENTER_RECT = dict(x=0.20, y=0.28, w=0.60, h=0.50)  # 稍大于 v162, 给鹰身体更多空间

# --- 大秃鹰主体 prompt (可改角度: 原图正面 -> v163 改 3/4 侧脸)
A_EAGLE_POS = (
    "a massive majestic bald eagle as the main central subject, "
    "head turned to the LEFT side in a three-quarter view (varied angle from frontal, "
    "showing the left side of the face more), "
    "one sharp intense orange-yellow eye visible, strong curved dark hooked beak in profile, "
    "spread wings extending outward and upward on both sides, "
    "detailed silver-gray and white feathers with precise gothic linework, "
    "powerful chest and shoulders covered in layered feathers, "
    "fierce commanding expression, "
    "dominating and filling the center of the composition, "
    "surrounded by dark black empty background, "
    "no skulls, no human face, no human skin, no hair, no weapons, no jacket, no clothing, no armor, "
    "no flames, no fire, no chains, no people, no figures, no other birds, "
    "no text, no watermark, no signature, no extra elements, "
    "dark gothic vector illustration style, "
    "high contrast, detailed linework, monochromatic silver-gray eagle on pure black background, "
    "print-ready apparel design quality"
)

N_EAGLE = (
    "blurry, deformed, distorted, extra fingers, extra limbs, watermark, signature, text, words, letters, "
    "skull, multiple skulls, human skull, skeleton, bones, death face, grim reaper, "
    "human face, skin, hair, long hair, short hair, beard, mustache, "
    "knife, dagger, sword, weapon, axe, scepter, staff, gun, blade, shield, "
    "leather, jacket, clothing, fabric, shirt, vest, armor, hood, cape, glove, "
    "flames, fire, flame tongue, chain, jewelry, crown, gem, rose, flower, "
    "colorful, colored, bright, neon, pastel, saturated, "
    "people, person, figure, character, body, hands, arms, legs, "
    "bird flock, multiple birds, two birds, three birds, raven, crow, sparrow, songbird, hummingbird, "
    "realistic photo, 3D render, photograph, photorealistic, "
    "small eagle, tiny eagle, distant eagle, "
    "low quality, lowres, jpeg artifacts, noise, grain, "
    "multiple eagles, two eagles, three eagles"
)

def make_center_mask(in_png: Path, out_png: Path):
    """画中央矩形 mask: 黑色=保留, 白色=重画; 边缘羽化"""
    im = Image.open(in_png).convert("RGB")
    W, H = im.size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    x0 = int(CENTER_RECT["x"] * W)
    y0 = int(CENTER_RECT["y"] * H)
    x1 = int((CENTER_RECT["x"] + CENTER_RECT["w"]) * W)
    y1 = int((CENTER_RECT["y"] + CENTER_RECT["h"]) * H)
    draw.rectangle([x0, y0, x1, y1], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=20))
    mask.save(out_png)
    print(f"  [MASK] {out_png.name} {W}x{H}  rect=({x0},{y0})-({x1},{y1})", flush=True)

def build_inpaint_center(input_name, mask_name, prompt, neg, seed, prefix):
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

def submit_and_fetch(g, reads, tag, hard_timeout=300):
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v163_{int(time.time()*1000)}"}, timeout=20)
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

def composite_center(fixed_png, stageA_4x, out_png, mask_png):
    """把 inpaint 修好的中心,按 mask 贴回 4x stageA"""
    fixed = Image.open(fixed_png).convert("RGB")  # 1024
    stageA_4x_im = Image.open(stageA_4x).convert("RGB")
    W4, H4 = stageA_4x_im.size
    fixed_up = fixed.resize((W4, H4), Image.LANCZOS)
    mask = Image.open(mask_png).convert("L").resize((W4, H4), Image.LANCZOS)
    out = Image.composite(fixed_up, stageA_4x_im, mask)
    out.save(out_png, "JPEG", quality=95, optimize=True)
    print(f"  [COMPOSITE] {out_png.name} {W4}x{H4} {out_png.stat().st_size/1024/1024:.2f}MB", flush=True)

def main():
    src_work = ROOT / "jobs" / "smoke_v161" / "stageA_eagle_2_work.png"
    src_4x   = ROOT / "jobs" / "smoke_v161" / "stageA_eagle_2.jpg"
    if not src_work.exists():
        print(f"[ERR] 缺 {src_work}"); return
    work_local = JOBS / "stageA_work.png"
    work_4x    = JOBS / "stageA_4x.jpg"
    if not work_local.exists(): shutil.copy(src_work, work_local)
    if not work_4x.exists():    shutil.copy(src_4x, work_4x)

    mask_local = JOBS / "center_mask.png"
    if not mask_local.exists():
        make_center_mask(work_local, mask_local)

    in_work = INPUT_DIR / "v163_stageA_work.png"
    in_mask = INPUT_DIR / "v163_center_mask.png"
    shutil.copy(work_local, in_work)
    shutil.copy(mask_local, in_mask)

    fixed_local = JOBS / "center_fixed.png"
    if not fixed_local.exists() or fixed_local.stat().st_size < 10000:
        print(">> Inpaint 主体: 大秃鹰 (3/4 侧脸)", flush=True)
        g = build_inpaint_center(in_work.name, in_mask.name, A_EAGLE_POS, N_EAGLE, 33112200, "v163_eagle")
        if not submit_and_fetch(g, [("10", fixed_local, False)], "v163/inpaint"):
            print("Inpaint 失败"); return
    else:
        print(f"  center_fixed 已存在 {fixed_local.stat().st_size/1024:.0f}KB 跳过", flush=True)

    composite_4x = JOBS / "stageA_center_fixed.jpg"
    if not composite_4x.exists():
        composite_center(fixed_local, work_4x, composite_4x, mask_local)

    final_local = JOBS / "final_eagle_2.jpg"
    if not final_local.exists():
        im = Image.open(composite_4x).convert("RGB")
        im = im.filter(ImageFilter.UnsharpMask(**USM))
        im.save(final_local, "JPEG", quality=95, optimize=True)
        print(f"  [FINAL] {final_local.name} {final_local.stat().st_size/1024/1024:.2f}MB", flush=True)

    burn = JOBS / "eagle_2_burned.jpg"
    if not burn.exists():
        im = Image.open(final_local).convert("RGB")
        W, H = im.size
        font_path = Path("C:/Windows/Fonts/PirataOne-Regular.ttf")
        if not font_path.exists():
            font_path = Path("C:/Windows/Fonts/segoeui.ttf")
        fs = int(W * 0.085)
        try:
            font = ImageFont.truetype(str(font_path), fs)
        except Exception as e:
            print(f"  [FONT] 加载失败用默认: {e}"); font = ImageFont.load_default()
        text = "DOMINION"
        try:
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(text) * fs // 2, fs
        tx = (W - tw) // 2
        ty = int(H * 0.92)
        # 阴影
        shadow = Image.new("RGBA", im.size, (0,0,0,0))
        ImageDraw.Draw(shadow).text((tx+4, ty+4), text, font=font, fill=(0,0,0,200))
        im_rgba = im.convert("RGBA")
        im_rgba.alpha_composite(shadow)
        # 描边
        stroke = Image.new("RGBA", im.size, (0,0,0,0))
        sd = ImageDraw.Draw(stroke)
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            sd.text((tx+dx, ty+dy), text, font=font, fill=(0,0,0,255))
        im_rgba.alpha_composite(stroke)
        # 主字
        main = Image.new("RGBA", im.size, (0,0,0,0))
        ImageDraw.Draw(main).text((tx, ty), text, font=font, fill=(220,220,230,255))
        im_rgba.alpha_composite(main)
        im_out = im_rgba.convert("RGB")
        im_out.save(burn, "JPEG", quality=95, optimize=True)
        print(f"  [BURN] {burn.name} {burn.stat().st_size/1024/1024:.2f}MB", flush=True)

    desktop = Path("E:/Desktop/双接口/image-fission/outputs/image-fission-v163-eagle_2.jpg")
    shutil.copy(burn, OUTPUTS / "eagle_2_final.jpg")
    shutil.copy(burn, desktop)
    print(f"  [OUT] {desktop}", flush=True)

    # 拼图
    compare = JOBS / "compare_eagle_2_v163.png"
    orig = Path("E:/Desktop/图裂变测试图/pinterest_eagle_2.jpg")
    parts = [("原图 (秃鹰主体)", orig), ("v162 错版 (骷髅主体)", JOBS/"stageA_4x.jpg"), ("v163 修正 (秃鹰主体)", burn)]
    imgs = []
    for _, p in parts:
        if p.exists():
            imgs.append(Image.open(p).convert("RGB"))
    if len(imgs) == 3:
        H = 800
        resized = []
        for im in imgs:
            w = int(im.width * H / im.height)
            resized.append(im.resize((w, H), Image.LANCZOS))
        gap = 20
        total_w = sum(im.width for im in resized) + gap * (len(resized)-1)
        canvas = Image.new("RGB", (total_w, H+40), (30,30,30))
        x = 0
        d = ImageDraw.Draw(canvas)
        for i, (im, (label, _)) in enumerate(zip(resized, parts)):
            canvas.paste(im, (x, 40))
            d.text((x+10, 8), label, fill=(255,255,255))
            x += im.width + gap
        canvas.save(compare, "JPEG", quality=92, optimize=True)
        print(f"  [COMPARE] {compare.name} {compare.stat().st_size/1024/1024:.2f}MB", flush=True)

    # 指标
    def load(p): return np.asarray(Image.open(p).convert("L")).astype(np.float64)
    def black(p, thr=24):
        im = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
        return float((im.mean(axis=2) < thr).mean())
    def sharp(p):
        im = np.asarray(Image.open(p).convert("L")).astype(np.float64)
        gx = np.gradient(im, axis=1); gy = np.gradient(im, axis=0)
        return float((gx**2+gy**2).mean())
    print(f"\n[指标]  v163 vs v162:")
    print(f"  原图黑底={black(orig):.3f} 锐度={sharp(orig):.0f}")
    print(f"  v162 stageA: 黑底={black(JOBS/'stageA_4x.jpg'):.3f} 锐度={sharp(JOBS/'stageA_4x.jpg'):.0f}")
    print(f"  v163 修正后: 黑底={black(burn):.3f} 锐度={sharp(burn):.0f}")

if __name__ == "__main__":
    main()
