"""
v248 — bat_logo strict original style（按原图 BACARDÍ bat logo 风格/设计/排版严格做）

用户 2026-09-03 09:45 硬指令: "原图什么风格什么设计什么排版就去做"

目标: BACARDÍ bat_logo (test_6978fabda2cc99629fa9e81f802762d3.jpg)
原图精确风格/设计/排版:
  - 风格: vintage gothic purple craft spirits label, 2D flat vector, 浅紫背景 + 黑色 bat + 黑色粗体衬线字
  - 设计: 三层布局 — 顶部弧形带 / 中部圆形徽章含 bat + Est. 1862 / 底部 BACARDI 粗体
  - 排版: 上中下竖向堆叠, 居中对称
v248 改造:
  1. v147 锁基线 (denoise 0.80 / ipa 0.18 / lora 1.0 / canny 0.25 / tile 0.60)
  2. 严格按原图风格/设计/排版: 浅紫背景 + 三层布局 + bat 居中展翅
  3. bat 旋转 30 度 + 翅膀倾斜 (v242/v244 被打脸"跟原图没区别", 必须真改)
  4. 顶弧/底部饰带 SDXL 画 vintage fleur-de-lis + scrollwork 装饰花纹 (NEG 强禁字形)
  5. 底部 PIL 后期烧非侵权词 (NOCTWING) + 顶弧弧字 (LVMEN NOCTIS) + Est. 1862 标记
  6. 强拉 2D vector cartoon sticker papercut (修 v245 "flat 2D printed" 联想 3D 浮雕)
  7. Reinhard LAB 锁色 (保持原图浅紫 + 深紫 + 黑)
  8. AI 肉眼看图自检 (v244+ 红线)

v245 翻车根因 (4/10):
  - "flat 2D printed" 让 SDXL 联想真实金属压凸 → 改 "flat vector cartoon sticker papercut" 强拉矢量
  - 顶弧乱字 "CIICIMD VE KHIARARP" → 改 "vintage fleur-de-lis scrollwork pattern ONLY" + 强 NEG 禁字形
  - 底部 "LIRA" 大写可读字 → 改 SDXL 画装饰花纹 + PIL 后期烧非侵权词 (不经过 SDXL 不会糊)
  - bat 切碎 → 强化 bat 中心位置 + 减 bat 周围装饰 + NEG 禁 triangle/eye column
"""
import os, sys, json, time, shutil, uuid, urllib.request, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v248"
JOB.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT / "src"))
from arc_text import draw_arc_text, fit_arc_text_width

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"
FONT_PATH = str(PROJECT / "fonts" / "PirataOne-Regular.ttf")

SEED = 248001  # 与 v246 (246001)/v245 (245001)/v244 (244001)/v213 (213001) 不同

# ==================== v147 锁死基线 ====================
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

# ==================== NEG: 全部禁用(v245 完整 + v248 新增禁字形) ====================
NEG_BASE = (
    # 基础 — 禁可读字+禁原样复制+禁漂色
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "legible text, readable text, recognizable letters, clear word, English word, brand name, logo word, "
    "spelled out words, specific brand word, recognizable brand, BACARDI, Mooncrest, Curse, "
    "calligraphy, scripture, "
    "3d, photographic, painterly, photorealistic bat, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "anatomically incorrect, extra wings, asymmetric error, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "new colors, different color palette, extra colors, color shift, "
    "v196-v212 style word swap, identical layout to source, "
    # v244 — 禁月牙球/3D锥/紫雾/inpaint漂白
    "sphere above emblem, moon sphere, circular moon ball, crescent moon above, "
    "3D pyramid below, 3D cone below, obelisk below, column below, totem pole, "
    "haze, fog, halo, inpaint bleach, smudge band, smudge ring, purple mist, "
    "foggy edges, soft purple haze around silhouette, vapor around bat, "
    "horizontal gradient band across middle, sky reflection, melted halo, "
    "text duplicate, double-pass lettering, over-painted words, "
    "stretched mirror artifacts, inpaint halos, retouch blotches, "
    "subject trapped in fog, fog swallowing the subject"
    # v245 — 禁项链/吊坠/缎带/金属浮雕/3D 珠宝
    "necklace, pendant, jewelry, gemstone drop, teardrop gem, water drop pendant, charm, amulet, locket, chain, "
    "draped cloth, draped fabric, silk ribbon, fabric ribbon, satin ribbon, cloth crossing, textile, "
    "metallic silver surface, 3D silver relief, chrome, metallic luster, polished metal, gold trim, brass, "
    "3D rendering, photorealistic, jewelry aesthetic, pendant necklace, ornamental chain, tiara, brooch"
    # v248 新增 — 禁所有字形 (SDXL 不要画任何字母/字符/字形, 字由 PIL 后期烧)
    "letter shapes, character forms, alphabet forms, readable letter forms, "
    "specific characters, individual letters, capital letters, lowercase letters, "
    "letter-shaped decorative ribbons, letterform patterns, "
    "spelling, inscribed letters, embossed letters, carved letters, "
    "Roman letters, Greek letters, Cyrillic letters, decorative alphabet, "
    "faux text, mock writing, illegible scribbles, alphabet soup, "
    # v248 新增 — 禁罗盘/表盘/指针/时钟类崩坏 (v247 暴露徽章底部"罗盘/表盘")
    "compass, compass rose, dial, clock face, clockwork, watch face, pointer, needle, gauge, meter, sun dial, "
    "compass needle, hour hand, minute hand, clock hands, sundial, astrolabe, navigation tool, instrument panel"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "natural overlap hierarchy, fits the overall composition"
)

REF_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

# ==================== v248 GLOBAL_POS: 严格按原图风格/设计/排版 ====================
GLOBAL_POS = (
    # 风格: 浅紫背景 + vintage gothic + 2D vector + craft spirits label
    "a vintage gothic purple craft spirits product label printed on paper, "
    "soft light purple #B57BC8 background with subtle grunge paper texture, "
    "purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F color blocks only, "
    # 设计: 三层布局 (按原图严格) — 顶弧/中徽章/底饰带
    "a centered symmetric three-tier vertical layout: "
    "a thin ornamental arched band at the top, "
    "a central circular emblem medallion in the middle, "
    "a decorative bottom banner below the emblem, "
    # 排版: 居中对称
    "the entire layout is centered and symmetric like the original, "
    # 强拉矢量 (修 v245 漂 3D 浮雕根因)
    "flat 2D vector cartoon sticker papercut style, "
    "solid color blocks with sharp clean edges, "
    "no 3D rendering, no metallic luster, no shading, no gradient, no relief, no silver chrome, no gold trim, "
    "screen print aesthetic, vintage print label art, "
    # bat 描述 (旋转 + 换角度, 不能再"跟原图没区别")
    "the central circular emblem contains a black bat silhouette with wings spread upward, "
    "the bat is rotated 30 degrees to its right and its wings tilted asymmetrically "
    "so it looks like a clearly different but related bat, same bat species, "
    # 装饰描述 (SDXL 画装饰花纹, 不画字)
    "the top arched band is intentionally clean blank light purple, no decoration added by SDXL, "
    "the actual decorative band and text will be added later via post-processing, do not draw anything in the top band, "
    "the bottom banner contains only abstract ornamental scrollwork pattern, "
    "no letters, no characters, no alphabet, no readable text, no spelled-out words anywhere, "
    "the actual wordmark will be added later via post-processing, do not draw any letters or words, "
    # 修 v245 bat 切碎根因
    "no triangle above the bat, no column eyes, no eye symbols, no extra geometric symbols, "
    "no necklace, no pendant, no ribbon, no draped fabric"
)

REGIONS = [
    # 中心徽章: bat 旋转 + 徽章外环装饰
    {
        "x": 0.10, "y": 0.22, "w": 0.80, "h": 0.50, "strength": 1.35,
        "prompt": (
            "a centered circular purple emblem medallion like the source, "
            "inside is a black bat silhouette with wings spread upward, "
            "the bat is rotated 30 degrees to its right and its wings tilted asymmetrically, "
            "the emblem has a simple thin black outline ring, no heavy decoration on the outside, no compass, no dial, no clock face, no watch face, no pointer, no needle, no gauge, no sun dial, "
            "no triangle above the bat, no column eyes, no eye symbols, no extra geometric symbols, "
            "no letter shapes inside the emblem, no readable text, no spelled-out words, "
            "flat 2D vector cartoon sticker papercut style, sharp clean edges, no 3D rendering, no metallic, "
            "keep purple/black palette. " + COHESIVE
        )
    },
    # 顶部弧形带: v248 改成完全留白 (SDXL 不画任何东西, 装饰花纹 + 烧字全 PIL 后期)
    {
        "x": 0.10, "y": 0.02, "w": 0.80, "h": 0.18, "strength": 0.40,
        "prompt": (
            "a clean smooth solid light purple #B57BC8 area at the top, "
            "NO decoration, NO pattern, NO fleur-de-lis, NO scrollwork, NO ornament, "
            "NO letter shapes, NO character forms, NO alphabet, NO readable text, "
            "NO specific characters, NO individual letters, NO capital letters, NO lowercase letters, "
            "NO letterform patterns, NO inscribed letters, NO embossed letters, NO carved letters, "
            "NO Roman letters, NO Greek letters, NO Cyrillic letters, NO faux text, "
            "the top band is intentionally blank, the actual decorative band and text will be added later, "
            "just a flat solid light purple color, no anything else. " + COHESIVE
        )
    },
    # 底部饰带: 装饰花纹 (无字, 烧字后续 PIL 加)
    {
        "x": 0.15, "y": 0.78, "w": 0.70, "h": 0.18, "strength": 1.15,
        "prompt": (
            "a flat 2D decorative bottom banner with abstract ornamental scrollwork pattern ONLY, "
            "NO letter shapes, NO character forms, NO alphabet, NO readable text, "
            "NO specific characters, NO individual letters, NO capital letters, NO lowercase letters, "
            "NO letterform patterns, NO inscribed letters, NO embossed letters, NO carved letters, "
            "NO wordmark, NO spelled-out words, NO 4-letter word, NO 5-letter word, NO 6-letter word, "
            "the actual brand wordmark will be added later via post-processing, do not draw any letters or words, "
            "flat 2D vector cartoon sticker papercut style, sharp clean edges, no 3D rendering, no metallic. " + COHESIVE
        )
    },
]


# ==================== build workflow ====================
def build(seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMG}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}

    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}

    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": CANNY_STRENGTH}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": TILE_STRENGTH}}

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    region_nodes = []
    for i, r in enumerate(REGIONS):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"] * REGION_STRENGTH_SCALE}}
        region_nodes.append(sk)

    comb_in = {"global_cond": ["pg", 0]}
    for i, sk in enumerate(region_nodes):
        comb_in[f"region{i+1}"] = [sk, 0]
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 24, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "v248_bat_logo"}}
    return g


def color_transfer(src_bgr, dst_bgr, alpha=1.0):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = dst.copy()
    for i in range(3):
        s_mean, s_std = src[:, :, i].mean(), src[:, :, i].std() + 1e-6
        d_mean, d_std = dst[:, :, i].mean(), dst[:, :, i].std() + 1e-6
        out[:, :, i] = (dst[:, :, i] - d_mean) * (s_std / d_std) + s_mean
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if alpha >= 1.0:
        return out
    blended = dst.astype(np.float32) * (1 - alpha) + out.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def hist_intersection(src_bgr, dst_bgr, bins=32):
    inter = 0.0
    for c in range(3):
        hs = cv2.calcHist([src_bgr], [c], None, [bins], [0, 256])
        hd = cv2.calcHist([dst_bgr], [c], None, [bins], [0, 256])
        cv2.normalize(hs, hs); cv2.normalize(hd, hd)
        inter += cv2.compareHist(hs, hd, cv2.HISTCMP_INTERSECT)
    return inter / 3.0


def structural_diff(src_bgr, dst_bgr):
    h, w = min(src_bgr.shape[0], dst_bgr.shape[0]), min(src_bgr.shape[1], dst_bgr.shape[1])
    s = cv2.resize(src_bgr, (w, h)); d = cv2.resize(dst_bgr, (w, h))
    mse = ((s.astype(np.float32) - d.astype(np.float32)) ** 2).mean()
    return float(np.clip(1 - mse / (255.0 ** 2), 0, 1))


# ==================== ComfyUI API ====================
def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def history(pid):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
        return json.loads(r.read())


def gen(seed):
    g = build(seed)
    r = submit(g)
    pid = r["prompt_id"]
    print(f"[v248] submitted pid={pid[:8]} denoise={DENOISE} ipa={IPA_WEIGHT} (strict original style + vector + post-burn)")
    for _ in range(120):
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        print(f"[v248] TIMEOUT")
        return None
    outputs = h[pid].get("outputs", {})
    raw_path = None
    for node_id, info in outputs.items():
        for img in info.get("images", []):
            src = COMFY_OUTPUT / img["filename"]
            if not src.exists():
                src = Path(img.get("abs_path", src))
            if not src.exists():
                continue
            raw_path = str(src)
            break
        if raw_path:
            break
    if not raw_path:
        cands = sorted(COMFY_OUTPUT.glob("v248_bat_logo*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        print(f"[v248] NOT FOUND raw_path")
        return None
    return raw_path


def side_by_side(orig_path, gen_path, out_path):
    orig = Image.open(orig_path).convert("RGB")
    gen = Image.open(gen_path).convert("RGB")
    if orig.size != gen.size:
        gen = gen.resize(orig.size)
    out = Image.new("RGB", (orig.width * 2 + 30, orig.height), "white")
    out.paste(orig, (0, 0))
    out.paste(gen, (orig.width + 30, 0))
    out.save(out_path, quality=95)


# ==================== v248 新增: PIL 后期烧非侵权词 (main 末尾, 不经过 SDXL 不会糊) ====================
def burn_word(img, text, font_size, x, y, color=(26, 10, 31)):
    """烧单个静态词 (底部 logo)"""
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x - tw // 2, y - th // 2), text, font=font, fill=color)


def burn_top_arc(img, text, font_size, color=(26, 10, 31)):
    """烧顶部弧形带 (rainbow arch, 圆心在带下方, 弧顶落在 y≈0.13h 画布内)"""
    w, h = img.width, img.height
    radius = int(w * 0.45)
    arc_len = fit_arc_text_width(text, FONT_PATH, font_size, radius, char_spacing_px=8)
    total_deg = math.degrees(arc_len / radius)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    cx = w // 2
    cy = int(h * 0.13) + radius  # 圆心在带下方 -> 弧顶(270°)落在 y≈0.13h
    return draw_arc_text(img, text, FONT_PATH, font_size, color,
                         (cx, cy), radius, start, end, char_spacing_px=8, flip_180=False)


def burn_est_year(img, font_size, color=(26, 10, 31)):
    """烧 'Est.' 和 '1862' 标记 (vintage 风格)"""
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, font_size)
    y_pos = int(img.height * 0.69)
    for txt, x_frac in [("Est.", 0.27), ("1862", 0.73)]:
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((int(img.width * x_frac) - tw // 2, y_pos - th // 2),
                  txt, font=font, fill=color)


def main():
    out_final = JOB / "v248_bat_logo.jpg"
    if out_final.exists() and out_final.stat().st_size > 100000:
        print(f"[skip] already exists {out_final}")
    else:
        raw = gen(SEED)
        if not raw:
            print("[v248] FAIL gen")
            return
        src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF_IMG).convert("RGB")), cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(np.array(Image.open(raw).convert("RGB")), cv2.COLOR_RGB2BGR)
        matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
        out_raw = JOB / "v248_bat_logo_raw.jpg"
        Image.fromarray(cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)).save(str(out_raw), quality=95)
        out_pre_burn = JOB / "v248_bat_logo_pre_burn.jpg"
        Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)).save(str(out_pre_burn), quality=95)

        # v248 新增: PIL 后期烧非侵权词 (main 末尾, 不经过 SDXL 不会糊)
        final_img = Image.open(out_pre_burn).convert("RGB")
        try:
            final_img = burn_top_arc(final_img, "LVMEN NOCTIS", int(final_img.height * 0.045))
            print("  [v248] top arc burn OK: LVMEN NOCTIS")
        except Exception as e:
            print(f"  [v248] top arc burn FAIL: {e}")
        try:
            burn_word(final_img, "NOCTWING", int(final_img.height * 0.075),
                      int(final_img.width * 0.5), int(final_img.height * 0.86))
            burn_word(final_img, "MORS VINI", int(final_img.height * 0.045),
                      int(final_img.width * 0.5), int(final_img.height * 0.93))
            print("  [v248] bottom word burn OK: NOCTWING + MORS VINI")
        except Exception as e:
            print(f"  [v248] bottom word burn FAIL: {e}")
        try:
            burn_est_year(final_img, int(final_img.height * 0.022))
            print("  [v248] est year burn OK: Est. 1862")
        except Exception as e:
            print(f"  [v248] est year burn FAIL: {e}")
        if final_img.mode != "RGB":
            final_img = final_img.convert("RGB")
        final_img.save(str(out_final), quality=95)

        hi_before = hist_intersection(src_bgr, dst_bgr)
        pre_burn_bgr = cv2.cvtColor(np.array(Image.open(out_pre_burn).convert("RGB")), cv2.COLOR_RGB2BGR)
        hi_after = hist_intersection(src_bgr, pre_burn_bgr)
        sd = structural_diff(src_bgr, pre_burn_bgr)
        print(f"  saved v248_bat_logo.jpg ({out_final.stat().st_size//1024} KB)")
        print(f"    配色: 迁移前={hi_before:.3f} → 迁移后={hi_after:.3f} | 结构差异={sd:.3f}")

    # 对照拼图
    out_cmp = JOB / "_compare_v248.jpg"
    side_by_side(COMFY_INPUT / REF_IMG, out_final, out_cmp)
    print(f"  saved compare: {out_cmp}")


if __name__ == "__main__":
    main()
