# -*- coding: utf-8 -*-
"""
v324_universal_fission.py -- 通用图裂变管线 (元素层重生 + 文字带按位置大小重画新词 + LAB 颜色锁)
  Stage A 元素层重生:  Canny + Tile 双 ControlNet + RegionalPrompting
                      按 CONFIG['regions'] 把每个 region(元素/文字带) 用独立 CLIP prompt 约束
                      KSampler 双段(0.70 + 0.20) 元素姿态+细节重生
  Stage B 抹旧字:      按 CONFIG['text_regions'] bbox 用 SDXL inpaint denoise=0.25
                      只抹 bbox 内文字, 位置不动
  Stage C 新词渲染:    PIL 按 bbox 自动算字号 + CONFIG['font_key'] 渲染新词
                      文字带位置大小完全照原图
  Stage D 颜色锁:      Reinhard LAB 色彩迁移 把裂变图色系拉回原图色 (硬锁原图色相)
  Stage E 4x 上采样:   NMKD-Siax
"""
import argparse, json, time, io, os, shutil, uuid, urllib.request, urllib.error, sys, tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
sys.path.insert(0, str(Path(__file__).parent))
from v268_lama_clean import load_lama, lama_inpaint   # 本地 LaMa inpaint (v318c 定稿)

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
IPA_PRESET = "PLUS (high strength)"

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
# regions 里 "kind" 决定 Stage A RegionalPrompting 是否覆盖; "text" 决定是否走 Stage B/C
#
# 用户硬规则复盘:
#   - 元素位置大小不能改
#   - 文字带按位置大小重画新词
#   - 配色严格锁原图色相 (Reinhard LAB 后处理)
#   - BACARDÍ 这种注册商标绝不能作为变体, 必须换 NOCTAVEN/DUSKBAT/MOONBAT 等原创
#   - 字体: 大字 BlackOpsOne (军标) / Playfair-Black (Didone 风) ; 小字 Arial Narrow Bold

CONFIG = {
    # b78e60 军徽 ARMED FORCES (1556x2000)
    # 文字带: WE SUPPORT THE / ARMED / FORCES (y=280-720 三段连续)
    "b78e60de8dfdf44acda99395326a7298.jpg": {
        "src": SRC_DEFAULT / "b78e60de8dfdf44acda99395326a7298.jpg",
        "stage_a_prompt": (
            "monochrome gray military combat camouflage background, "
            "tactical dog tags and chain in dead center, matte gray monochrome palette, "
            "strict 2D flat printed military graphic style, NO 3D, NO photorealistic, "
            "preserve exact composition: top-center text band, center dog tags with chain, "
            "gray camo background pattern, dark gray and light gray ONLY, "
            "uniform flat gray background, NO color, NO purple, NO red, NO blue"
        ),
        "neg_extra": "color, color shift, purple, red, blue, green, yellow, orange, brown, gradient",
        "regions": [
            # 元素层 (迷彩/狗牌/链条 由 SDXL 重生, 位置锁原图)
            {"kind": "element", "bbox": (0.10, 0.10, 0.90, 0.95), "label": "camo+dogtag+chain"},
        ],
        "text_regions": [
            # 拆 3 行独立 bbox (程序化 detect_text_lines 检测精确坐标)
            # 原图 (1556x2000) 文字带:  WE SUPPORT THE / ARMED / FORCES
            # band1: y=452-531 h=79 cx=784   "WE SUPPORT THE" (capH 79, 14字细体)
            # band2: y=535-647 h=112 cx=784  "ARMED"  (capH 112, 5字大字)
            # FORCES 估:  y=650-755 h=105 cx=784  capH 105 (6字)
            # 每行独立 bbox + 独立新词 (同位置同高度不同内容)
            # 关键: word 字符数与原图近似, 保证 capH 不被 width-fit 压扁
            {"bbox": (517, 452, 1051, 531), "capH": 79, "banks": [
                "WE HONOR HEROES",        # 14字 填 534px 宽 (匹配原 14字)
                "WE SALUTE THE BRAVE",    # 18字 (略长)
                "WE BACK OUR TROOPS",     # 16字
            ], "font_key": "blackopsone"},
            {"bbox": (459, 535, 1109, 647), "capH": 112, "banks": [
                "STEEL",                  # 5字 = 原 ARMED 字符数
                "VIGIL",                  # 5字
                "HONOR",                  # 5字
            ], "font_key": "blackopsone"},
            {"bbox": (459, 650, 1109, 740), "capH": 105, "banks": [
                "TROOPS",                 # 6字 = 原 FORCES 字符数
                "GUARDS",                 # 6字
                "VALORS",                 # 6字
            ], "font_key": "blackopsone"},
        ],
    },

    # 13c8b7 红黑佩斯利 (1132x1132, 无文字)
    "13c8b7bf8dae757e6c2d4b3d6a860f9d.jpg": {
        "src": SRC_DEFAULT / "13c8b7bf8dae757e6c2d4b3d6a860f9d.jpg",
        "stage_a_prompt": (
            "red orange and black ornate paisley bandana print, intricate damask and baroque "
            "floral medallions, symmetrical four-quadrant pattern, detailed ornamental, "
            "preserve exact composition: four-patch bandana layout with center divider, "
            "RED ORANGE WHITE BLACK palette ONLY, NO purple, NO blue, NO green, NO gray"
        ),
        "neg_extra": "purple, blue, green, gray, brown, beige, washed out, desaturated",
        "regions": [
            {"kind": "element", "bbox": (0.0, 0.0, 1.0, 1.0), "label": "paisley bandana"},
        ],
        "text_regions": [],   # 无文字
    },

    # 6978fab BACARDÍ 蝙蝠 (1024x1280, 5 段文字带)
    # 用户硬规则: BACARDÍ 商标禁用, 换 NOCTAVEN / DUSKBAT / MOONBAT
    "6978fabda2cc99629fa9e81f802762d3.jpg": {
        "src": SRC_DEFAULT / "6978fabda2cc99629fa9e81f802762d3.jpg",
        "stage_a_prompt": (
            "purple vintage craft spirits label with stylized 2D bat silhouette, "
            "perfectly centered circular badge with thin clean ring outline, "
            "saturated purple #6B2C8C and deep violet #2A0A3F and black ONLY, "
            "STRICTLY 2D flat printed vintage craft spirits label print style, "
            "preserve exact composition: top arc line + circular badge with bat + bottom triangle"
        ),
        "neg_extra": (
            "BACARDÍ, BACARDI, BACARDO, BATMAN, "
            "trophy, cup, vase, 3D, photorealistic, glossy, gradient, "
            "gray, beige, brown, green, blue, yellow, desaturated"
        ),
        "regions": [
            {"kind": "element", "bbox": (0.05, 0.10, 0.95, 0.95), "label": "bat badge"},
        ],
        "text_regions": [
            # 6978fab BACARDÍ 图: 程序化 detect_text_lines 检测精确坐标 (size 1552x2000)
            # band5: (284, 993, 1256, 1156)  "BACARDÍ"  (capH 163, 数字字体主字)
            # band6: (465, 1181, 1129, 1334) "MCKHEART" (capH 153, 副字)
            # 顶弧/底三角保留 (蝙蝠徽章身份)
            # 修复 v324s bbox 太宽吞掉 ™ 上标 + 紧贴底三角 -> 收紧 + 避免 mask dilate 触及周围
            {"bbox": (320, 1005, 1220, 1140), "capH": 135, "banks": [
                "NOCTAVEN", "DUSKBAT", "MOONBAT",
            ], "font_key": "playfair_bold"},
            {"bbox": (480, 1195, 1110, 1320), "capH": 125, "banks": [
                "MOONHEART", "DARKHEART", "STARLING",
            ], "font_key": "playfair_bold"},
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


def wait_outputs(pid, prefix, timeout=320):
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
def build_stage_a(orig_name, stage_a_prompt, neg_extra, regions_prompts, seed, prefix):
    """regions_prompts: [(bbox_norm, prompt, strength), ...] 覆盖 CONFIG['regions']"""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": orig_name}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "upscale_method": "lanczos", "megapixels": 1.0, "image": ["2", 0],
        "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader",
              "inputs": {"model": ["1", 0], "preset": IPA_PRESET}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": 0.5, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": 0.40, "strength_clip": 0.40}}
    # Canny 边缘
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": 0.55}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": 0.85}}

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
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 24, "cfg": 7.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.65}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 18, "cfg": 7.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.18}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}}
    return g


# ===================== Stage B: bbox mask inpaint 抹旧字 =====================
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
    mask[y1:y2, x1:x2] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate*2+1, dilate*2+1))
    mask = cv2.dilate(mask, k, iterations=1)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask
    return Image.fromarray(rgba, "RGBA")


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

    # Stage A: 元素层处理 (CFG['element_regen'] 开关, 默认 False=原图直传保 100% 元素)
    element_regen = cfg.get("element_regen", False)
    if element_regen:
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
        g_a = build_stage_a(orig_path.name, cfg["stage_a_prompt"],
                            cfg["neg_extra"], regions_prompts, seed0, a_prefix)
        r = submit(g_a, f"v324_{ts}_{ci}_a")
        if not r:
            print(f"  Stage A submit failed, fallback to 原图", flush=True)
            cur = base_img.copy()
            cur_name = orig_path.name
        else:
            pid = r["prompt_id"]
            print(f"  Stage A pid={pid[:8]}", flush=True)
            blob = wait_outputs(pid, a_prefix, timeout=400)
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
        bx1, by1, bx2, by2 = tr["bbox"]
        # scale to cur
        sx, sy = cur.size[0] / W0, cur.size[1] / H0
        cb = (int(bx1*sx), int(by1*sy), int(bx2*sx), int(by2*sy))
        word = tr["banks"][ci % len(tr["banks"])]
        words_used.append(word)
        print(f"  region{ri} '{word}' bbox_norm={tr['bbox']} -> cur={cb}", flush=True)

        # Stage B 用 LaMa (本地) 替代 SDXL inpaint -- v318c 定稿: LaMa 只填 mask 区, 保周围纹理
        # Stage B: LaMa 抹旧字 (dilate=8 让字外缘精准覆盖, 不吞周围装饰)
        # capH 约束: 从 tr 取 (有就传, 没就 None 让 render 用二分填满)
        capH = tr.get("capH")
        mask_img = make_text_mask(cur.size, cb, dilate=8)
        print(f"  [B] LaMa inpaint  bbox={cb}  capH={capH}  mask_area={int(np.array(mask_img)[...,3].mean()/255*100)}%", flush=True)
        mask_l = mask_img.split()[3]   # alpha channel
        cur = lama_inpaint(cur, mask_l, removal_strength=230, edge_smoothness=6)
        cur_name = f"v324_{ts}_{ci}_r{ri}_lama.png"
        cur.save(COMFY_INPUT / cur_name)
        cur.save(out_dir / f"{fname.replace('.jpg','')}_c{ci}_r{ri}_lama.png")

        # Stage C: PIL 渲染新词 (按 capH 算字号, 严格匹配原字高度)
        cur = render_text_at_bbox(cur, word, cb, tr["font_key"], target_capH=capH)
        cur_name = f"v324_{ts}_{ci}_r{ri}_texted.png"
        cur.save(COMFY_INPUT / cur_name)

    LAB_ALPHA = 0.85   # LAB 颜色锁强度 (1.0 = 完全锁, 0.85 = 部分锁保细节)

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
        for ci in range(3):  # 每图 3 候选
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