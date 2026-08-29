"""v150 三路线对比：同一张 eagle_2，用三种不同方法做"裂变"，给用户挑方向。
- V2  PIL 几何裂变：镜像+旋转+放大，零 AI 失真（你的第2条）
- V1  元素级参考重绘：抠出原图老鹰 crop，用它自身当 IP-Adapter 参考 + 改姿态提示重画，合成回原图（你的第1条）
- V3  模型锁元素身份：原图整体当强 IP-Adapter 参考(0.85) + 轻 Canny(0.30) + 裂变提示（你的第3条）

输出：jobs/compare3/{v1_elem,v2_pil,v3_idlock}.jpg + compare3_strip.png
"""
import time, requests, sys
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"
SEED = 700401
CKPT = "ProteusV0.4.safetensors"
LORA_DETAIL = 1.0
REF_IMG = "pinterest_eagle_2.jpg"
SRC = Path(r"E:/Desktop/图裂变测试图") / REF_IMG
IN = PROJECT_ROOT / "ComfyUI" / "input"
OUT = PROJECT_ROOT / "jobs" / "compare3"
OUT.mkdir(parents=True, exist_ok=True)


# ---------- 工具：提交+轮询 ----------
def submit(graph, prefix):
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": graph,
                    "client_id": f"v150_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {prefix}: {r.status_code} {str(j)[:400]}", flush=True)
        return None
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {prefix}: 无 prompt_id {str(j)[:300]}", flush=True)
        return None
    print(f"  [{prefix}] pid={pid}", flush=True)
    for i in range(72):
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                rec = h[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
                    if imgs:
                        url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                        return requests.get(url, timeout=60).content
                elif rec.get("status", {}).get("error"):
                    print(f"  [{prefix}] COMFY错误 {rec['status'].get('error')}", flush=True)
                    return None
        except Exception as e:
            print(f"  [{prefix}] 轮询异常 {e}", flush=True)
        if i % 6 == 0:
            print(f"    [{prefix}] {i*5}s...", flush=True)
    print(f"  [{prefix}] TIMEOUT", flush=True)
    return None


def usm(im):
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))


# ---------- V1 元素级参考重绘 ----------
def build_v1_eagle_redraw(seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": "_eagle_crop.png"}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    # 用 crop 自身当参考：锁住"这只鹰"的身份
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": 0.55, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}
    pos = ("a bald eagle FACING THE CAMERA head-on (NOT side profile), both wings spread symmetrically "
           "outward from a central body, wings and body anatomically connected as one bird, "
           "fierce open beak, sharp white and silver feathers with fine detail, "
           "pure black background, gothic tattoo style, "
           "this is the SAME eagle as the reference image, only the POSE/ANGLE is changed, "
           "KEEP the exact same eagle identity, feather pattern and detailing")
    neg = ("side profile, different eagle, generic eagle, cartoon, mutated, deformed, extra wings, "
           "blurry, low quality, watermark, text, extra colors, background objects")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": neg}}
    # 弱 Canny 允许姿态改变
    g["16"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "controlnet-canny-sdxl-1.0.fp16.safetensors"}}
    g["17"] = {"class_type": "Canny", "inputs": {"image": ["2", 0], "low_threshold": 0.10, "high_threshold": 0.30, "resolution": 1024}}
    g["18"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["8", 0], "control_net": ["16", 0], "image": ["17", 0], "strength": 0.25}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 30, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.60}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 22, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.18}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "v1_elem"}}
    return g


# ---------- V3 模型锁元素身份 ----------
def build_v3_idlock(seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMG}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.5, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    # 原图整体当强参考，锁住整张图的元素身份/配色
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": 0.85, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}
    pos = ("gothic heraldic crest on pure black background, "
           "the SAME bald eagle with spread silver-white wings and sharp yellow beak, "
           "the SAME small eagle above, the SAME three human skulls at the bottom, "
           "the SAME iron chains on both sides, red and orange flames, black heraldic shield, "
           "RECOMPOSE with changed ANGLES and SCALE: eagle now faces the camera, "
           "chains rearranged with metal spikes, flames rise from lower-left diagonally, "
           "skull larger with deeper cracks, KEEP exact same element identity palette and category")
    neg = ("different eagle, different skull, different design, rearranged into new unrelated composition, "
           "mutated, deformed, cartoon, blurry, low quality, watermark, text, new colors, "
           "extra subject, missing element")
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": neg}}
    # 轻 Canny 允许角度/大小改变但不焊死
    g["16"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "controlnet-canny-sdxl-1.0.fp16.safetensors"}}
    g["17"] = {"class_type": "Canny", "inputs": {"image": ["2", 0], "low_threshold": 0.10, "high_threshold": 0.30, "resolution": 1024}}
    g["18"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["8", 0], "control_net": ["16", 0], "image": ["17", 0], "strength": 0.30}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 30, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.62}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 22, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.18}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "v3_idlock"}}
    return g


def main():
    # ---- 抠老鹰 crop 存到 ComfyUI/input ----
    orig = Image.open(SRC).convert("RGB")
    W, H = orig.size
    box = (int(W * 0.06), 0, int(W * 0.94), int(H * 0.48))
    eagle = orig.crop(box)
    IN.mkdir(parents=True, exist_ok=True)
    eagle.save(IN / "_eagle_crop.png")
    print(f"eagle crop saved: {eagle.size}", flush=True)

    # ---- V1 ----
    print("=== V1 元素级参考重绘 ===", flush=True)
    d1 = submit(build_v1_eagle_redraw(SEED), "v1")
    if d1:
        v1raw = OUT / "v1_elem_raw.png"
        v1raw.write_bytes(d1)
        v1 = usm(Image.open(v1raw).convert("RGB"))
        # 合成回原图：把重绘鹰贴回原图鹰区（黑底叠黑底）
        v1fit = v1.resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS)
        composited = orig.copy()
        composited.paste(v1fit, box)
        composited.save(OUT / "v1_elem.jpg", "JPEG", quality=95, optimize=True)
        print(f"V1 合成完成 -> {OUT/'v1_elem.jpg'}", flush=True)
    else:
        print("V1 失败", flush=True)

    # ---- V3 ----
    print("=== V3 模型锁元素身份 ===", flush=True)
    d3 = submit(build_v3_idlock(SEED + 100), "v3")
    if d3:
        v3raw = OUT / "v3_idlock_raw.png"
        v3raw.write_bytes(d3)
        v3 = usm(Image.open(v3raw).convert("RGB"))
        v3.save(OUT / "v3_idlock.jpg", "JPEG", quality=95, optimize=True)
        print(f"V3 完成 -> {OUT/'v3_idlock.jpg'}", flush=True)
    else:
        print("V3 失败", flush=True)

    # ---- V2 PIL 几何裂变 ----
    print("=== V2 PIL 几何裂变 ===", flush=True)
    m = ImageOps.mirror(orig)
    r = m.rotate(18, fillcolor=(0, 0, 0), expand=False)
    zw, zh = int(W * 1.15), int(H * 1.15)
    z = r.resize((zw, zh), Image.LANCZOS)
    left, top = (zw - W) // 2, (zh - H) // 2
    v2 = z.crop((left, top, left + W, top + H))
    v2.save(OUT / "v2_pil.jpg", "JPEG", quality=95, optimize=True)
    print(f"V2 完成 -> {OUT/'v2_pil.jpg'}", flush=True)

    # ---- 拼 4 联对照 ----
    print("=== 拼对照图 ===", flush=True)
    imgs = [("ORIGINAL", orig), ("V1 抠鹰重绘", Image.open(OUT / "v1_elem.jpg")),
            ("V2 几何裂变", v2), ("V3 锁身份裂变", Image.open(OUT / "v3_idlock.jpg"))]
    Hd = 900
    cols = []
    for label, im in imgs:
        w, h = im.size
        nh = Hd
        nw = int(w * nh / h)
        cols.append((label, im.resize((nw, nh), Image.LANCZOS)))
    gap = 14
    Wtot = sum(c[1].width for c in cols) + gap * (len(cols) - 1)
    strip = Image.new("RGB", (Wtot, Hd + 40), (20, 20, 20))
    x = 0
    from PIL import ImageDraw
    d = ImageDraw.Draw(strip)
    for label, im in cols:
        strip.paste(im, (x, 40))
        d.text((x + 6, 10), label, fill="white")
        x += im.width + gap
    strip.save(OUT / "compare3_strip.png", "PNG", optimize=True)
    print(f"对照图 -> {OUT/'compare3_strip.png'}", flush=True)
    print("ALL done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
