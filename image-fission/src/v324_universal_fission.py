# -*- coding: utf-8 -*-
"""
v324_universal_fission.py -- 通用图裂变管线 (元素层重生 + 文字带按位置大小重画新词 + LAB 颜色锁)
  Stage A 元素层重生:  Canny ControlNet + RegionalPrompting + LoRA
                      按 CONFIG['regions'] 把每个 region 用独立 CLIP prompt 约束
                      KSampler 双段(denoise1 + denoise2) 元素姿态+细节重生
  Stage B 抹旧字:      本地 Big-LaMa 按 text_regions  mask 抹除旧字, 保周围纹理
  Stage C 新词渲染:    PIL 按 bbox / arc 位置大小重画新词, 字体按原图 capH 匹配
  Stage D 颜色锁:      Reinhard LAB 色彩迁移 把裂变图色系拉回 (硬锁原图色族)
  Stage E 4x 上采样:   NMKD-Siax (预留, 当前未启用)

v324g 调优 (2026-09-11):
  - 移除 Tile ControlNet 与 IPAdapter, 仅保留 Canny 锁构图 (适配 12GB VRAM)
  - Stage A 分辨率降至 0.5 MP 避免 lowvram 补丁导致 25-40 s/it
  - 修复 Canny 条件未接入 KSampler 的 bug (之前 dangling)
"""
import argparse, json, time, io, os, shutil, uuid, urllib.request, urllib.error, sys, tempfile, math
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageChops
sys.path.insert(0, str(Path(__file__).parent))
from v268_lama_clean import load_lama, lama_inpaint   # 本地 LaMa inpaint (v318c 定稿)
from arc_text import draw_arc_text, fit_arc_text_width  # 弧形文字 (替换 logo 顶弧/横幅)

ROOT = Path(__file__).resolve().parent.parent
SRC_DEFAULT = Path(r"E:/Desktop/图裂变测试图")
COMFY_INPUT = ROOT / "ComfyUI" / "input"
COMFY_OUTPUT = ROOT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"

# 预加载 LaMa
_LAMA_MODEL = load_lama()

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.fp16.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

# 字体 (含 v319t 用户硬规则定稿: BlackOpsOne (big) + Arial Narrow Bold (small))
FONTS = {
    # 军标风 (大写 A/D 尖三角, horizontal 横杠切割) -- ARMED FORCES / STEEL / SOLDIER 类
    "blackopsone":     str(ROOT / "fonts" / "BlackOpsOne-Regular.ttf"),
    "arialnarrow_b":   r"C:/Windows/Fonts/ARIALNB.TTF",
    "impact":          r"C:/Windows/Fonts/impact.ttf",
    "oldengl":         r"C:/Windows/Fonts/OLDENGL.TTF",
    "stencil":         r"C:/Windows/Fonts/STENCIL.TTF",
    "rock":            r"C:/Windows/Fonts/ROCK.TTF",
    "gothicb":         r"C:/Windows/Fonts/GOTHICB.TTF",
    "georgiab":        r"C:/Windows/Fonts/georgiab.ttf",
    # Didone / Playfair Display (BACARDÍ 风) -- 宽衬线粗体大字
    "playfair_bold":   str(ROOT / "fonts" / "PlayfairDisplay-Bold.ttf"),
    # 万圣节/吸血鬼/蝙蝠/植物恐怖风
    "creepster":       str(ROOT / "fonts" / "Creepster-Regular.ttf"),
    "metalmania":      str(ROOT / "fonts" / "MetalMania-Regular.ttf"),
    "vampiroone":      str(ROOT / "fonts" / "VampiroOne-Regular.ttf"),
    "nosifer":         str(ROOT / "fonts" / "Nosifer-Regular.ttf"),
    "unifraktur":      str(ROOT / "fonts" / "UnifrakturMaguntia-Regular.ttf"),  # 字体检测 broken, 备用
    # 西部/装饰
    "rye":             str(ROOT / "fonts" / "Rye-Regular.ttf"),
    "pirataone":       str(ROOT / "fonts" / "PirataOne-Regular.ttf"),
    "frijole":         str(ROOT / "fonts" / "Frijole-Regular.ttf"),
    "sairastencil":    str(ROOT / "fonts" / "SairaStencilOne-Regular.ttf"),
}


# ===================== CONFIG =====================
# 每图独立: regions(元素层 + 文字带坐标) + banks(新词候选) + font_key + stage_a_prompt
# bbox 全部用绝对像素坐标 (x1, y1, x2, y2)
# text_regions 支持两种形式:
#   - 直线 bbox: {"bbox": (...), "capH": int, "banks": [...], "font_key": ...}
#   - 弧形文字:   {"arc": {"cx": int, "cy": int, "radius": int, "start": deg, "end": deg,
#                          "capH": int, "flip_180": bool},
#                 "banks": [...], "font_key": ..., "dilate": int}
#
# 用户硬规则复盘 (v324 定稿):
#   - 文字带按位置大小重画新词 (直线/弧形都支持)
#   - 主体元素: 同种主体, 同位置/同大小, 但允许角度/细节变化 (element_regen=True + Canny 锁构图)
#   - 配色严格锁原图色族 (Reinhard LAB 后处理)
#   - BACARDÍ 等注册商标绝不出现, 必须原位改写

CONFIG = {
    # b78e60 军徽 ARMED FORCES (1556x2000)
    # 元素: 迷彩底 + 居中狗牌/链条 -> 裂变: 狗牌形状/链条细节变化, 迷彩纹理变化
    "b78e60de8dfdf44acda99395326a7298.jpg": {
        "src": SRC_DEFAULT / "b78e60de8dfdf44acda99395326a7298.jpg",
        "element_regen": True,
        "lab_alpha": 0.85,
        "stage_a_prompt": (
            "monochrome gray military combat camouflage background, "
            "tactical dog tags and ball chain in dead center, dog tag shape slightly varied, "
            "chain detail varied, same centered composition, matte gray monochrome palette, "
            "strict 2D flat printed military graphic style, NO 3D, NO photorealistic, "
            "gray camo background pattern, dark gray and light gray ONLY, "
            "NO color, NO purple, NO red, NO blue"
        ),
        "neg_extra": "color, color shift, purple, red, blue, green, yellow, orange, brown, gradient",
        "stage_a_params": {
            "canny_res": 512, "canny_strength": 0.60, "canny_lo": 0.10, "canny_hi": 0.25,
            "tile_strength": 0.65, "ipa_weight": 0.45, "ipa_end": 0.85, "ipa_noise": 0.05,
            "lora": 0.40, "denoise1": 0.45, "denoise2": 0.15, "steps1": 18, "steps2": 10,
            "cfg": 7.5,
        },
        "regions": [
            {"kind": "element", "bbox": (0.05, 0.05, 0.95, 0.95), "label": "camo+dogtag+chain"},
        ],
        "text_regions": [
            {"bbox": (517, 452, 1051, 531), "capH": 79, "dilate": 4, "banks": [
                "WE HONOR HEROES", "WE SALUTE THE BRAVE", "WE BACK OUR TROOPS",
            ], "font_key": "blackopsone"},
            {"bbox": (459, 535, 1109, 647), "capH": 112, "dilate": 4, "banks": [
                "STEEL", "VIGIL", "HONOR",
            ], "font_key": "blackopsone"},
            {"bbox": (459, 650, 1109, 750), "capH": 105, "dilate": 4, "banks": [
                "TROOPS", "GUARDS", "VALORS",
            ], "font_key": "blackopsone"},
        ],
    },

    # Pinterest (2).jpg 鹰/骷髅/火焰徽章 (964x1280)
    # 元素: 展翅鹰 + 3 骷髅 + 火焰 + 链条 -> 裂变: 鹰姿态/骷髅裂纹/火焰形态变化
        # 文字: 顶部横幅 "JACKE DIANNIES" + 盾顶弧 "JACKE DIANNIES" + 底部横幅 "TALKDAN LANDUK OHI"
        "Pinterest (2).jpg": {
        "src": SRC_DEFAULT / "Pinterest (2).jpg",
        "element_regen": True,
        "lab_alpha": 0.85,
        "stage_a_prompt": (
            "bold vector illustration, eagle with spread wings perched on skulls, "
            "chains and flames, eagle pose slightly varied, skull crack patterns varied, "
            "flame shapes varied, same symmetrical composition, "
            "red orange black and white palette ONLY, NO 3D, NO photorealistic"
        ),
        "neg_extra": "purple, blue, green, gray, brown, beige, washed out, desaturated, "
                     "3D, photorealistic, glossy gradient",
        "stage_a_params": {
            "canny_res": 512, "canny_strength": 0.60, "canny_lo": 0.12, "canny_hi": 0.30,
            "tile_strength": 0.60, "ipa_weight": 0.45, "ipa_end": 0.85, "ipa_noise": 0.05,
            "lora": 0.45, "denoise1": 0.45, "denoise2": 0.15, "steps1": 18, "steps2": 10,
            "cfg": 7.0,
        },
        "regions": [
            {"kind": "element", "bbox": (0.0, 0.0, 1.0, 1.0), "label": "eagle+skulls+flames"},
        ],
        "text_regions": [
            # 顶部横幅 (近似弧形, 用直线 bbox + western 字体)
            {"bbox": (300, 380, 664, 430), "capH": 42, "dilate": 4, "banks": [
                "RAVEN CLAW", "IRON WING", "STORM EYE",
            ], "font_key": "rye", "color": (235, 230, 220, 255)},
            # 盾顶弧 (light text on black shield; 直线 bbox 近似)
            {"bbox": (340, 590, 630, 660), "capH": 38, "dilate": 8, "banks": [
                "IRON EAGLE", "SKULL BORN", "FIRE CLAN",
            ], "font_key": "rye", "color": (235, 230, 220, 255)},
            # 底部横幅 (原文字位置 y≈820-900, 扩到 910 覆盖完整)
            {"bbox": (150, 810, 830, 910), "capH": 55, "dilate": 4, "banks": [
                "BLOOD OATH", "DARK HARBOR", "ASH & IRON",
            ], "font_key": "rye", "color": (235, 230, 220, 255)},
        ],
    },

    # Pinterest (3).jpg 牛仔蝴蝶 (736x1308)
    # 元素: 牛仔布蝴蝶 + 小蝴蝶/虚线轨迹 -> 裂变: 蝴蝶翅膀纹理/小蝴蝶姿态变化
    # 文字: 顶部 "UPGY" 大字母 (denim  varsity 风格)
    "Pinterest (3).jpg": {
        "src": SRC_DEFAULT / "Pinterest (3).jpg",
        "element_regen": True,
        "lab_alpha": 0.85,
        "stage_a_prompt": (
            "denim fabric patch butterfly with frayed edges, butterfly wing pattern varied, "
            "small butterflies and dotted flight trails, light blue denim texture, "
            "same centered composition, white background, "
            "NO 3D, NO photorealistic"
        ),
        "neg_extra": "purple, red, green, gray, brown, black background, "
                     "3D, photorealistic, glossy gradient",
        "stage_a_params": {
            "canny_res": 512, "canny_strength": 0.65, "canny_lo": 0.10, "canny_hi": 0.25,
            "tile_strength": 0.60, "ipa_weight": 0.45, "ipa_end": 0.85, "ipa_noise": 0.05,
            "lora": 0.40, "denoise1": 0.40, "denoise2": 0.12, "steps1": 18, "steps2": 10,
            "cfg": 7.5,
        },
        "regions": [
            {"kind": "element", "bbox": (0.0, 0.0, 1.0, 1.0), "label": "denim butterfly+letters"},
        ],
        "text_regions": [
            # 顶部大字母: 原为牛仔布贴 varsity 风格, 现用 rock 粗衬线 + 牛仔蓝, 均匀白底直接 fill
            {"bbox": (0, 0, 736, 500), "capH": 230, "dilate": 12, "banks": [
                "DENIM", "PATCH", "STYLE",
            ], "font_key": "rock", "color": (60, 95, 165, 255), "fill": (245, 245, 245)},
        ],
    },

    # 6978fab BACARDÍ 蝙蝠徽章 (1552x2000)
    # 元素: 蝙蝠 + 圆形徽章 -> 裂变: 蝙蝠翅膀姿态变化, 徽章保持
    # 文字: 顶弧 LA CASA DEL MURCIELAGO + Est. + 1862 + BACARDÍ + MCKHEART
    # v324h 调优: 采用 v268 经验 —— 对均匀紫色背景文字带用 fill 采样色平填,
    #  彻底消除 LaMa 残留 ghost; 扩大 arc 扇区覆盖原弧字; lab_alpha=0 避免全局洗色.
    "6978fabda2cc99629fa9e81f802762d3.jpg": {
        "src": SRC_DEFAULT / "6978fabda2cc99629fa9e81f802762d3.jpg",
        "element_regen": False,
        "lab_alpha": 0.0,
        "stage_a_prompt": (
            "purple vintage craft spirits label with stylized 2D bat silhouette, "
            "bat wing pose slightly varied inside perfectly centered circular badge, "
            "thin clean ring outline, "
            "saturated purple #6B2C8C and deep violet #2A0A3F and black ONLY, "
            "STRICTLY 2D flat printed vintage craft spirits label print style, "
            "preserve exact composition: top arc line + circular badge with bat + bottom triangle"
        ),
        "neg_extra": (
            "BACARDÍ, BACARDI, BACARDO, BATMAN, "
            "trophy, cup, vase, 3D, photorealistic, glossy, gradient, "
            "gray, beige, brown, green, blue, yellow, desaturated"
        ),
        "stage_a_params": {
            "canny_res": 320, "canny_strength": 0.85, "canny_lo": 0.10, "canny_hi": 0.25,
            "tile_strength": 0.30, "ipa_weight": 0.30, "ipa_end": 0.85, "ipa_noise": 0.00,
            "lora": 0.45, "denoise1": 0.35, "denoise2": 0.10, "steps1": 18, "steps2": 10,
            "cfg": 7.5,
        },
        "regions": [
            {"kind": "element", "bbox": (0.05, 0.10, 0.95, 0.95), "label": "bat badge"},
        ],
        "text_regions": [
            # 顶弧弧形文字: 大扇区覆盖原 LA CASA DEL MURCIELAGO, fill 采样 ribbon 色
            {"arc": {"cx": 776, "cy": 950, "radius": 520, "start": 198, "end": 342,
                     "capH": 85, "flip_180": False}, "dilate": 28, "fill": (185, 126, 174),
             "banks": [
                "LA TUMBA DEL VAMPIRO", "CASA DE LAS SOMBRAS", "EL REINO NOCTURNO",
            ], "font_key": "playfair_bold"},
            # Est. / 1862 小字 (badge 左右)
            {"bbox": (410, 840, 590, 990), "capH": 70, "dilate": 24, "fill": (185, 125, 175),
             "banks": ["Set.", "Est.", "Sir."], "font_key": "playfair_bold"},
            {"bbox": (960, 840, 1140, 990), "capH": 70, "dilate": 24, "fill": (185, 125, 175),
             "banks": ["1877", "1888", "1899"], "font_key": "playfair_bold"},
            # BACARDÍ / MCKHEART: 宽 bbox 覆盖 accent/superscript, fill 采样背景色
            {"bbox": (180, 980, 1370, 1170), "capH": 135, "dilate": 24, "fill": (184, 125, 175),
             "banks": ["NOCTAVEN", "DUSKBAT", "MOONBAT"], "font_key": "playfair_bold"},
            {"bbox": (380, 1180, 1175, 1330), "capH": 125, "dilate": 20, "fill": (185, 125, 175),
             "banks": ["MOONHEART", "DARKHEART", "STARLING"], "font_key": "playfair_bold"},
        ],
    },
}


# ===================== ComfyUI helpers =====================
def submit(graph, client_id):
    data = json.dumps({"prompt": graph, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            print(f"  submit HTTP {e.code}: {body[:500]}", flush=True)
            return None
        except Exception as e:
            print(f"  submit conn err: {e}", flush=True); time.sleep(3)
    return None


def wait_outputs(pid, prefix, timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(4)
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=10) as r:
                h = json.loads(r.read())
        except Exception:
            continue
        if pid not in h:
            continue
        rec = h[pid]
        st = rec.get("status", {})
        for msg in st.get("messages", []):
            if isinstance(msg, (list, tuple)) and msg and msg[0] == "execution_error":
                print(f"  exec error: {str(msg[1])[:500]}", flush=True)
                return None
        if st.get("completed"):
            for node, data in rec.get("outputs", {}).items():
                for im in data.get("images", []):
                    fn = im["filename"]
                    if not fn.startswith(prefix):
                        continue
                    sub = im.get("subfolder", "")
                    url = (f"{COMFY_URL}/view?filename={fn}&subfolder={sub}"
                           f"&type={im.get('type','output')}")
                    try:
                        with urllib.request.urlopen(url, timeout=180) as r:
                            return r.read()
                    except Exception as e:
                        print(f"  download err {fn}: {e}", flush=True)
    print(f"  TIMEOUT {int(time.time()-t0)}s", flush=True)
    return None


# ===================== Stage A: 元素层重生 (RegionalPrompting + Canny + Tile) =====================
def build_stage_a(orig_name, stage_a_prompt, neg_extra, regions_prompts, seed, prefix, params=None):
    """regions_prompts: [(bbox_norm, prompt, strength), ...] 覆盖 CONFIG['regions']
       params: 每图可调 Stage A 超参 (裂变强度/锁结构强度), 默认偏"保元素"."""
    p = dict(
        ipa_weight=0.50, ipa_end=0.85, ipa_noise=0.05,
        lora=0.40,
        canny_strength=0.55, canny_res=1024, canny_lo=0.10, canny_hi=0.25,
        tile_strength=0.85,
        denoise1=0.65, denoise2=0.18, steps1=24, steps2=18, cfg=7.5,
    )
    if params:
        p.update(params)
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 0.5, "image": ["2", 0],
        "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["1", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": p["lora"], "strength_clip": p["lora"]}}
    # Canny 边缘 (锁宏观构图, 允许元素姿态/细节变化); Tile 已移除以节省 VRAM
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": p["canny_lo"], "high_threshold": p["canny_hi"],
        "resolution": p["canny_res"]}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["comb", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": p["canny_strength"]}}

    # Global prompt
    neg = ("text, words, letters, typography, watermark, signature, logo, "
           "blurry, low quality, color shift, " + neg_extra)
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": stage_a_prompt}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": neg}}

    # Regional conditioning: 把每个 region 局部化
    region_nodes = []
    for i, (bn, rp, rs) in enumerate(regions_prompts):
        rk = f"rp{i}"; sk = f"sa{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": rp}}
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": bn[2], "height": bn[3],
            "x": bn[0], "y": bn[1], "strength": rs}}
        region_nodes.append(sk)

    comb_in = {"global_cond": ["pg", 0]}
    for i, sk in enumerate(region_nodes):
        comb_in[f"region{i+1}"] = [sk, 0]
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["22", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": p["steps1"], "cfg": p["cfg"],
        "sampler_name": "euler", "scheduler": "normal", "denoise": p["denoise1"]}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": p["steps2"], "cfg": p["cfg"],
        "sampler_name": "euler", "scheduler": "normal", "denoise": p["denoise2"]}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}}
    return g


# ===================== Stage B: bbox mask inpaint 抹旧字 =====================
def fill_region(img, mask_rgba, color):
    """用纯色填充 mask 区域（mask 的 alpha 通道控制填充范围），用于均匀底色文字带清理."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    mask = np.array(mask_rgba.split()[3]).astype(np.float32) / 255.0
    color_np = np.array(color[:3]).astype(np.float32)
    for c in range(3):
        arr[..., c] = arr[..., c] * (1 - mask) + color_np[c] * mask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def build_inpaint(img_name, mask_name, seed, prefix, denoise=0.40):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
    g["3"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "alpha"}}
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 0], "grow_mask_by": 6}}
    pos = "seamless continuation of the surrounding pattern texture, empty clean area, no text, no letters, no words"
    neg = "text, words, letters, typography, watermark, signature, logo, blurry, low quality, color shift"
    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["4", 0],
        "seed": seed, "steps": 22, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "denoise": denoise}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return g


# ===================== Stage C: PIL 按 bbox 位置大小渲染新词 =====================
def make_text_mask(size_wh, bbox, dilate=12):
    w, h = size_wh
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x1, y1 = min(x1, w-1), min(y1, h-1)
    x2, y2 = max(x1+1, min(x2, w)), max(y1+1, min(y2, h))
    mask[y1:y2, x1:x2] = 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate*2+1, dilate*2+1))
        mask = cv2.dilate(mask, k, iterations=1)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask
    return Image.fromarray(rgba, "RGBA")


def make_arc_text_mask(size_wh, arc, dilate=4):
    """生成弧形文字带的环形扇区 mask."""
    w, h = size_wh
    cx, cy, radius = arc["cx"], arc["cy"], arc["radius"]
    capH = arc.get("capH", 48)
    start, end = arc["start"], arc["end"]
    inner_r = max(0, radius - capH//2 - 8)
    outer_r = radius + capH//2 + 8
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    # 角度约定与 arc_text.draw_arc_text 保持一致: 0=east, 顺时针增加.
    d.pieslice([cx-outer_r, cy-outer_r, cx+outer_r, cy+outer_r], start=start, end=end, fill=255)
    # 挖掉内圆
    inner = Image.new("L", (w, h), 0)
    di = ImageDraw.Draw(inner)
    if inner_r > 0:
        di.ellipse([cx-inner_r, cy-inner_r, cx+inner_r, cy+inner_r], fill=255)
    img = ImageChops.subtract(img, inner)
    if dilate > 0:
        mask = np.array(img)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate*2+1, dilate*2+1))
        mask = cv2.dilate(mask, k, iterations=1)
        img = Image.fromarray(mask, "L")
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = np.array(img)
    return Image.fromarray(rgba, "RGBA")


def render_arc_text_at_bbox(img, word, arc, font_key, color=(0, 0, 0, 255), all_caps=True):
    """用 arc_text.draw_arc_text 在弧形区域渲染新词."""
    cx, cy, radius = arc["cx"], arc["cy"], arc["radius"]
    start, end = arc["start"], arc["end"]
    capH = arc.get("capH", 48)
    flip = arc.get("flip_180", False)
    fp = FONTS.get(font_key, FONTS["impact"])
    text = word.upper() if all_caps else word
    # 估算字号: 用 capH 与字体 capH 比率
    try:
        probe = ImageFont.truetype(fp, 200)
        probe_capH = probe.getbbox("A")[3] - probe.getbbox("A")[1]
        ratio = probe_capH / 200
    except Exception:
        ratio = 0.70
    fsize = max(8, int(capH / max(0.30, ratio) * 0.95))
    # 让文字不超过弧长: 量一下在指定 radius 上占用像素
    arc_len_px = fit_arc_text_width(text, fp, fsize, radius, char_spacing_px=2)
    available_px = math.radians((end - start) % 360) * radius
    if arc_len_px > available_px * 0.95 and fsize > 8:
        fsize = max(8, int(fsize * (available_px * 0.95 / arc_len_px)))
    out = draw_arc_text(img, text, fp, fsize, color,
                        center=(cx, cy), radius=radius,
                        start_angle_deg=start, end_angle_deg=end,
                        char_spacing_px=2, flip_180=flip)
    return out.convert("RGB") if out.mode != "RGB" else out


def render_text_at_bbox(img, word, bbox, font_key, color=(0, 0, 0, 255),
                         letter_pad=0.04, all_caps=True, target_capH=None):
    """PIL 渲染 word 到 bbox 区域, 字号约束:
       - 如有 target_capH, 按 (target_capH / 原 capH) 算字号 (精确匹配原字高度)
       - 否则二分找最大字号 (填满 bbox)"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    W, H = x2 - x1, y2 - y1
    if W <= 4 or H <= 4:
        return img
    fp = FONTS.get(font_key, FONTS["impact"])
    text = word.upper() if all_caps else word
    canvas = Image.new("RGBA", (W*4, H*4), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    if target_capH:
        # 按 capH 算字号 (font-agnostic: 先量字体自身 capH/fontsize 比率)
        try:
            probe_fnt = ImageFont.truetype(fp, 200)
            probe_capH = probe_fnt.getbbox("A")[3] - probe_fnt.getbbox("A")[1]
            capH_ratio = probe_capH / 200
        except Exception:
            capH_ratio = 0.70
        fsize = int(target_capH / max(0.30, capH_ratio) * 4 * 0.95)
        try:
            fnt = ImageFont.truetype(fp, fsize)
        except Exception:
            fnt = ImageFont.load_default()
        bb = d.textbbox((0, 0), text, font=fnt)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        # 如果太宽, 等比缩到 W*3.7 内
        if tw > W*3.7:
            scale = (W*3.7) / tw
            fsize = max(8, int(fsize * scale))
            fnt = ImageFont.truetype(fp, fsize)
            bb = d.textbbox((0, 0), text, font=fnt)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
        # 如果太高 (超 H*3.5), 也等比缩
        if th > H*3.5:
            scale = (H*3.5) / th
            fsize = max(8, int(fsize * scale))
            fnt = ImageFont.truetype(fp, fsize)
            bb = d.textbbox((0, 0), text, font=fnt)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
        best = fsize
    else:
        # 二分找最大字号
        lo, hi, best = 8, H*4, 8
        for _ in range(28):
            mid = (lo + hi) // 2
            try:
                fnt = ImageFont.truetype(fp, mid)
            except Exception:
                fnt = ImageFont.load_default()
            bb = d.textbbox((0, 0), text, font=fnt)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
            if tw <= W*3.7 and th <= H*3.5:
                best = mid; lo = mid + 1
            else:
                hi = mid - 1
        fnt = ImageFont.truetype(fp, best)
        bb = d.textbbox((0, 0), text, font=fnt)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
    tx = (W*4 - tw)//2 - bb[0]; ty = (H*4 - th)//2 - bb[1]
    d.text((tx, ty), text, font=fnt, fill=color)
    # 缩放回 bbox
    rendered = canvas.resize((W, H), Image.LANCZOS)
    base = img.convert("RGBA")
    base.alpha_composite(rendered, (x1, y1))
    return base.convert("RGB")


# ===================== Stage D: Reinhard LAB 颜色锁 =====================
def lab_color_transfer(src_rgb, dst_rgb, alpha=1.0):
    """src -> 色系参考 (原图); dst -> 被锁图 (裂变图).  返回 dst 锁色版."""
    src = cv2.cvtColor(np.array(src_rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    dst = cv2.cvtColor(np.array(dst_rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    out = dst.copy()
    for i in range(3):
        s_mean, s_std = src[:, :, i].mean(), src[:, :, i].std() + 1e-6
        d_mean, d_std = dst[:, :, i].mean(), dst[:, :, i].std() + 1e-6
        out[:, :, i] = (dst[:, :, i] - d_mean) * (s_std / d_std) + s_mean
    out_rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    if alpha >= 1.0:
        return Image.fromarray(out_rgb)
    out_arr = (np.array(dst_rgb).astype(np.float32) * (1 - alpha) +
               out_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(out_arr)


# ===================== Pipeline runner =====================
def run_one(fname, cfg, ci, out_dir, ts, seed0):
    orig_path = cfg["src"]
    if not orig_path.exists():
        print(f"  [SKIP] {fname} 原图不存在: {orig_path}", flush=True)
        return None
    base_img = Image.open(orig_path).convert("RGB")
    W0, H0 = base_img.size
    print(f"\n=== [{ci+1}/3] {fname}  size=({W0}x{H0})  ===", flush=True)

    # 把原图拷进 ComfyUI input
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    src_in = COMFY_INPUT / orig_path.name
    if not src_in.exists() or src_in.stat().st_size != orig_path.stat().st_size:
        shutil.copy2(orig_path, src_in)

    # 前置清理: 先把所有文字带用 LaMa 抹掉, 生成 clean_plate
    # 这样 Stage A 元素裂变不会把旧字/乱码写进背景, 后续渲染更干净
    clean_plate = base_img.copy()
    if cfg["text_regions"]:
        for ri, tr in enumerate(cfg["text_regions"]):
            dilate = tr.get("dilate", 6)
            if "arc" in tr:
                arc = tr["arc"]
                ca = {
                    "cx": arc["cx"], "cy": arc["cy"],
                    "radius": arc["radius"], "start": arc["start"], "end": arc["end"],
                    "capH": arc.get("capH", 48), "flip_180": arc.get("flip_180", False),
                }
                mask_img = make_arc_text_mask(clean_plate.size, ca, dilate=dilate)
            else:
                mask_img = make_text_mask(clean_plate.size, tr["bbox"], dilate=dilate)
            print(f"  [pre-clean] region{ri} dilate={dilate}", flush=True)
            mask_l = mask_img.split()[3]
            if tr.get("fill"):
                clean_plate = fill_region(clean_plate, mask_img, tr["fill"])
            else:
                clean_plate = lama_inpaint(clean_plate, mask_l, removal_strength=230, edge_smoothness=6)
        clean_name = f"v324_{ts}_{ci}_clean.png"
        clean_plate.save(COMFY_INPUT / clean_name)

    # Stage A: 元素层处理 (CFG['element_regen'] 开关, 默认 False=原图直传保 100% 元素)
    element_regen = cfg.get("element_regen", False)
    if element_regen:
        orig_name_for_stage_a = f"v324_{ts}_{ci}_clean.png" if cfg["text_regions"] else orig_path.name
        # SDXL 元素层轻重生 (Canny 锁位置 + 轻重生)
        regions_prompts = []
        for r in cfg["regions"]:
            bn = r["bbox"]
            if max(bn) > 1.5:
                x1, y1, x2, y2 = bn
                bn_norm = (x1/W0, y1/H0, (x2-x1)/W0, (y2-y1)/H0)
            else:
                bn_norm = bn
            regions_prompts.append((bn_norm, cfg["stage_a_prompt"], 1.0))

        a_prefix = f"v324_{ts}_{ci}_a"
        g_a = build_stage_a(orig_name=orig_name_for_stage_a, stage_a_prompt=cfg["stage_a_prompt"],
                            neg_extra=cfg["neg_extra"], regions_prompts=regions_prompts,
                            seed=seed0, prefix=a_prefix, params=cfg.get("stage_a_params"))
        r = submit(g_a, f"v324_{ts}_{ci}_a")
        if not r:
            print(f"  Stage A submit failed, fallback to 原图", flush=True)
            cur = base_img.copy()
            cur_name = orig_path.name
        else:
            pid = r["prompt_id"]
            print(f"  Stage A pid={pid[:8]}", flush=True)
            blob = wait_outputs(pid, a_prefix, timeout=1500)
            if not blob:
                print(f"  Stage A failed, fallback to 原图", flush=True)
                cur = base_img.copy()
                cur_name = orig_path.name
            else:
                cur = Image.open(io.BytesIO(blob)).convert("RGB")
                cur.save(out_dir / f"{fname.replace('.jpg','')}_c{ci}_stageA.png")
                cur_name = f"v324_{ts}_{ci}_stageA.png"
                cur.save(COMFY_INPUT / cur_name)
    else:
        # 默认: 跳过 SDXL 元素层重生, 直接用原图保 100% 元素
        cur = base_img.copy()
        cur_name = orig_path.name  # 直接用 ComfyUI/input 里的原图
        print(f"  [A] 元素层=原图 (保 100% 元素, element_regen=False)", flush=True)

    words_used = []
    for ri, tr in enumerate(cfg["text_regions"]):
        word = tr["banks"][ci % len(tr["banks"])]
        words_used.append(word)
        dilate = tr.get("dilate", 6)
        capH = tr.get("capH")

        if "arc" in tr:
            # ---- 弧形文字带 ----
            arc = tr["arc"]
            sx, sy = cur.size[0] / W0, cur.size[1] / H0
            ca = {
                "cx": int(arc["cx"] * sx), "cy": int(arc["cy"] * sy),
                "radius": int(arc["radius"] * (sx+sy)/2),
                "start": arc["start"], "end": arc["end"],
                "capH": int(arc.get("capH", capH or 48) * (sx+sy)/2),
                "flip_180": arc.get("flip_180", False),
            }
            print(f"  region{ri} '{word}' arc={ca} (dilate={dilate})", flush=True)
            mask_img = make_arc_text_mask(cur.size, ca, dilate=dilate)
            print(f"  [B] LaMa inpaint arc  capH={ca['capH']}  mask_area={int(np.array(mask_img)[...,3].mean()/255*100)}%", flush=True)
            mask_l = mask_img.split()[3]
            if tr.get("fill"):
                cur = fill_region(cur, mask_img, tr["fill"])
            else:
                cur = lama_inpaint(cur, mask_l, removal_strength=230, edge_smoothness=6)
            cur_name = f"v324_{ts}_{ci}_r{ri}_lama.png"
            cur.save(COMFY_INPUT / cur_name)
            cur.save(out_dir / f"{fname.replace('.jpg','')}_c{ci}_r{ri}_lama.png")

            cur = render_arc_text_at_bbox(cur, word, ca, tr["font_key"], color=tr.get("color", (0, 0, 0, 255)))
            cur_name = f"v324_{ts}_{ci}_r{ri}_texted.png"
            cur.save(COMFY_INPUT / cur_name)

        else:
            # ---- 直线 bbox 文字带 ----
            bx1, by1, bx2, by2 = tr["bbox"]
            sx, sy = cur.size[0] / W0, cur.size[1] / H0
            cb = (int(bx1*sx), int(by1*sy), int(bx2*sx), int(by2*sy))
            print(f"  region{ri} '{word}' bbox_norm={tr['bbox']} -> cur={cb} (dilate={dilate})", flush=True)

            # Stage B 用 LaMa (本地) 替代 SDXL inpaint -- v318c 定稿: LaMa 只填 mask 区, 保周围纹理
            mask_img = make_text_mask(cur.size, cb, dilate=dilate)
            print(f"  [B] LaMa inpaint  bbox={cb}  capH={capH}  mask_area={int(np.array(mask_img)[...,3].mean()/255*100)}%", flush=True)
            mask_l = mask_img.split()[3]   # alpha channel
            if tr.get("fill"):
                cur = fill_region(cur, mask_img, tr["fill"])
            else:
                cur = lama_inpaint(cur, mask_l, removal_strength=230, edge_smoothness=6)
            cur_name = f"v324_{ts}_{ci}_r{ri}_lama.png"
            cur.save(COMFY_INPUT / cur_name)
            cur.save(out_dir / f"{fname.replace('.jpg','')}_c{ci}_r{ri}_lama.png")

            # Stage C: PIL 渲染新词 (按 capH 算字号, 严格匹配原字高度)
            cur = render_text_at_bbox(cur, word, cb, tr["font_key"], target_capH=capH, color=tr.get("color", (0, 0, 0, 255)))
            cur_name = f"v324_{ts}_{ci}_r{ri}_texted.png"
            cur.save(COMFY_INPUT / cur_name)

    LAB_ALPHA = cfg.get("lab_alpha", 0.85)   # 每图可调: 1.0=完全锁, 0.85=部分锁保细节, 低=允许色族内偏差

    # Stage D: LAB 颜色锁 (alpha<1.0 保留纹理细节, 但仍硬锁色族)
    final = lab_color_transfer(base_img, cur, alpha=LAB_ALPHA)
    final_path = out_dir / f"{fname.replace('.jpg','')}_c{ci}_final.png"
    final.save(final_path)
    print(f"  -> {final_path}", flush=True)

    # 拼图对照
    H = 900
    o = base_img.resize((int(base_img.width * H / base_img.height), H), Image.LANCZOS)
    f2 = final.resize((int(final.width * H / final.height), H), Image.LANCZOS)
    gap = 20
    canvas = Image.new("RGB", (o.width + gap + f2.width, H + 40), (25, 25, 25))
    canvas.paste(o, (0, 20)); canvas.paste(f2, (o.width + gap, 20))
    cd = ImageDraw.Draw(canvas)
    cd.text((10, 5), "ORIGINAL", fill=(255, 200, 100))
    cd.text((o.width + gap + 10, 5), "V324 " + (" ".join(words_used) if words_used else "(无文字)"),
            fill=(255, 200, 100))
    cmp_path = out_dir / f"{fname.replace('.jpg','')}_c{ci}_compare.png"
    canvas.save(cmp_path)
    print(f"  compare-> {cmp_path}", flush=True)
    return {"final": final_path, "compare": cmp_path, "words": words_used}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--ncands", type=int, default=3, help="每个图像生成几个候选 (默认 3)")
    args = ap.parse_args()
    ts = int(time.time())
    out_dir = ROOT / "jobs" / f"v324_universal_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== v324_universal_fission -> {out_dir} ===", flush=True)

    names = args.images or list(CONFIG.keys())
    seed0 = 324001
    results = []
    for fi, fname in enumerate(names):
        cfg = CONFIG[fname]
        for ci in range(args.ncands):  # 每图 ncands 候选
            res = run_one(fname, cfg, ci, out_dir, ts, seed0 + fi * 100 + ci * 17)
            if res:
                results.append(res)

    # 拼 HTML
    html = ["<html><head><meta charset='utf-8'><style>",
            "body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:20px}",
            ".card{display:inline-block;margin:10px;vertical-align:top;border:1px solid #444;"
            "border-radius:8px;overflow:hidden}",
            ".card img{height:460px;display:block}",
            ".cap{padding:6px 10px;font-size:13px}",
            "h1{color:#fd6}", "</style></head><body>",
            "<h1>v324 通用图裂变 (元素重生 + 文字带按位置大小重画 + LAB 颜色锁)</h1>"]
    for r in results:
        words = " ".join(r["words"]) if r["words"] else "(无文字)"
        cmp_rel = Path(r["compare"]).name
        html.append(f"<div class='card'><img src='{cmp_rel}'>"
                    f"<div class='cap'><b>{words}</b></div></div>")
    html.append("</body></html>")
    (out_dir / "gallery.html").write_text("\n".join(html), encoding="utf-8")
    print(f"\n=== done: {len(results)} combos -> {out_dir}/gallery.html ===", flush=True)


if __name__ == "__main__":
    main()