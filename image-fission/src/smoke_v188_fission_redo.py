"""
v188 — 真裂变 + 硬锁色（修正 v186 的两大错误）

v186 错误复盘：
  1. IPA weight 0.05-0.12 太弱 → 锁不住配色
  2. 提示词写 "PRESERVE EXACTLY / DO NOT move" → 反裂变（只重描/擦字）

v188 修正：
  - IPA style transfer weight = 0.70（强锁色；style transfer 对内容影响小，可拉高）
  - denoise = 0.50（足够做裂变，又不崩结构）
  - Canny 0.50 / Tile 0.60（降到中档，给裂变留空间，不再 0.75 锁死）
  - LORA 0.40（降噪保文字清晰）
  - 后置 Reinhard LAB ColorTransfer：把结果色彩分布强制对齐原图
    → 无论 SDXL 漂成什么样，最后按原图配色收口（硬保 配色铁律）
  - 真裂变提示词：主体改角度/大小/数量 + 小元素换内容 + 数量元素改数量
    （画风/配色/材质不变，只动结构）

每张 REFS 含 fission 描述（明确改什么），不再写 "PRESERVE EXACTLY"。
"""

import os, sys, json, time, shutil, uuid
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# ==================== 配置 ====================
PROJECT = Path(__file__).resolve().parents[1]
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v188"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 700508  # 与 v185/v186 都不同

# ==================== NEG_BASE ====================
NEG_BASE = (
    "frame, border, white border, letterbox, "
    "garbled text, illegible letters, mangled typography, distorted characters, broken text, "
    "wrong letters, scrambled letters, partially missing letters, "
    "3d, photographic, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, watercolor, pastel, "
    "noise, grain, pixelated, jagged edges, aliasing, "
    "mutated, malformed, deformed anatomy, extra limbs, "
    "melted, fused, smudged, bleeding, "
    "new colors not in original palette, different color palette, extra colors, color shift, "
    "duplicate, watermark"
)

COHESIVE = (
    "cohesive with the rest of the design, natural overlap hierarchy, "
    "fits the overall composition, same art style as the original"
)

# ==================== 5 张 REFS（真裂变参数）====================
REFS = [

    # === 1. 橙色佩斯利 4 格面板 — 裂变：每格花纹旋转不同角度 + 大小不一 + 增减数量 ===
    {
        "id": "orange_paisley",
        "ref_img": "test_13c8b7bf8dae757e6c2d4b3d6a860f9d.jpg",
        "denoise": 0.50,
        "canny_strength": 0.50,
        "tile_strength": 0.60,
        "ipa_weight": 0.70,
        "lora_detail": 0.40,
        "global_pos": (
            "ornate orange paisley bandana print pattern, "
            "warm orange and white and dark brown and deep red color blocks, "
            "traditional persian paisley teardrop motifs, "
            "fabric bandana quality, seamless pattern feel, "
            "no readable text, sharp clean edges, intricate detail"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 0.5, "h": 0.5, "strength": 1.0,
             "prompt": (
                 "top-left cell: rotate the central paisley teardrop MOTIF 35 degrees clockwise, "
                 "enlarge it to 1.3x its original size, keep orange+white+brown+red palette and paisley art style, "
                 "add 2 small paisley bud accents in the empty corners. " + COHESIVE)},
            {"x": 0.50, "y": 0.00, "w": 0.5, "h": 0.5, "strength": 1.0,
             "prompt": (
                 "top-right cell: rotate the central paisley teardrop MOTIF 20 degrees counter-clockwise, "
                 "shrink it to 0.7x, replace the small corner flourish with a different ornamental curl shape, "
                 "keep same palette and paisley style. " + COHESIVE)},
            {"x": 0.00, "y": 0.50, "w": 0.5, "h": 0.5, "strength": 1.0,
             "prompt": (
                 "bottom-left cell: keep the central motif upright but DUPLICATE it into a pair of smaller "
                 "paisley motifs side by side (quantity changed from 1 to 2), "
                 "keep same palette and style. " + COHESIVE)},
            {"x": 0.50, "y": 0.50, "w": 0.5, "h": 0.5, "strength": 1.0,
             "prompt": (
                 "bottom-right cell: rotate the central motif 90 degrees, enlarge 1.15x, "
                 "change the small leaf fillings to a different leaf shape, same palette and style. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 0.6,
             "prompt": (
                 "KEEP the 2x2 grid divider lines visible and the original four-panel layout structure, "
                 "keep the small size tag in each bottom-right corner, "
                 "only the inner motifs change angle/size/quantity. " + COHESIVE)},
        ],
    },

    # === 2. RACING T恤 — 裂变：红带斜切角度 + 数字换 + 旗/徽章换位 + 数量变 ===
    {
        "id": "racing",
        "ref_img": "test_184432b34a4787fbed628b3b986b37a2.jpg",
        "denoise": 0.50,
        "canny_strength": 0.50,
        "tile_strength": 0.60,
        "ipa_weight": 0.70,
        "lora_detail": 0.40,
        "global_pos": (
            "bold racing motorsport t-shirt design layout, "
            "white and red and black color blocks, "
            "horizontal red bands, racing typography, flag motifs, speedometer numerals, "
            "keep the existing RACING wordmark and other text legible, sharp clean edges"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 0.35, "strength": 1.0,
             "prompt": (
                 "top section: tilt the horizontal red band to a 12-degree diagonal slant, "
                 "enlarge the big RACING wordmark by 1.2x and shift it slightly left, "
                 "move the checkered flag icon from left to the upper-right corner, "
                 "keep white/red/black racing style. " + COHESIVE)},
            {"x": 0.00, "y": 0.35, "w": 1.0, "h": 0.35, "strength": 1.0,
             "prompt": (
                 "middle section: change the emblem numeral from 23 to 47, "
                 "rotate the circular emblem 25 degrees, add one extra small red star accent, "
                 "keep same palette and racing style. " + COHESIVE)},
            {"x": 0.00, "y": 0.70, "w": 1.0, "h": 0.30, "strength": 1.0,
             "prompt": (
                 "bottom section: tilt the lower red band to a -10-degree diagonal, "
                 "replace the small side flame motif with a different speed streak shape, "
                 "keep white/red/black. " + COHESIVE)},
        ],
    },

    # === 3. 黑暗骑士 — 裂变：骑士转体 + 剑角度 + 小装饰换内容 ===
    {
        "id": "dark_knight",
        "ref_img": "test_3a300c32794aeea08f8abb2517f3afe1.jpg",
        "denoise": 0.50,
        "canny_strength": 0.50,
        "tile_strength": 0.60,
        "ipa_weight": 0.70,
        "lora_detail": 0.40,
        "global_pos": (
            "dark gothic knight kneeling with sword, black and charcoal monochrome, "
            "gothic typography background text, dramatic lighting, "
            "keep the existing background gothic text legible, sharp clean edges"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 0.6, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "left/center: turn the kneeling knight's torso 25 degrees to the left, "
                 "raise the sword from angled to fully vertical, enlarge the sword 1.15x, "
                 "keep black gothic monochrome style. " + COHESIVE)},
            {"x": 0.60, "y": 0.00, "w": 0.4, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "right side: replace the small glowing ember decorations with different gothic "
                 "ornamental filigree curls, keep black palette and gothic style. " + COHESIVE)},
        ],
    },

    # === 4. BACARDÍ 蝙蝠徽章 — 裂变：蝙蝠转角 + 翼展放大 + 小星换内容（文字后烧 BATANO）===
    {
        "id": "bat_logo",
        "ref_img": "test_6978fabda2cc99629fa9e81f802762d3.jpg",
        "denoise": 0.50,
        "canny_strength": 0.50,
        "tile_strength": 0.60,
        "ipa_weight": 0.70,
        "lora_detail": 0.40,
        "global_pos": (
            "purple circular emblem badge with a central bat silhouette, "
            "purple and black and white color blocks, bold emblem style, "
            "no readable text, sharp clean edges"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "rotate the central BAT silhouette 25 degrees clockwise and enlarge its wingspan to 1.25x, "
                 "replace the small star accents around the badge ring with different geometric triangle accents, "
                 "keep purple/black/white palette and emblem badge art style. " + COHESIVE)},
        ],
    },

    # === 5. ARMED FORCES 军牌 — 裂变：狗牌转角 + 链节数量变 + 星换鹰（文字后烧）===
    {
        "id": "camo_armed",
        "ref_img": "test_b78e60de8dfdf44acda99395326a7298.jpg",
        "denoise": 0.50,
        "canny_strength": 0.50,
        "tile_strength": 0.60,
        "ipa_weight": 0.70,
        "lora_detail": 0.40,
        "global_pos": (
            "white background military support athletic t-shirt design, "
            "pure white and dark charcoal and black color blocks, "
            "military dog tag pendant, clean product print mockup style, "
            "no readable text, sharp clean edges"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "rotate the military DOG TAG pendant 18 degrees and enlarge it 1.2x, "
                 "increase the number of chain links from the original count to a different number, "
                 "replace the small star accents with small eagle silhouettes, "
                 "keep pure white background and black military stencil style. " + COHESIVE)},
        ],
    },
]


# ==================== build workflow ====================
def build(ref, seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.2, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {
        "model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": ref["ipa_weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": ref["lora_detail"], "strength_clip": ref["lora_detail"]}}
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": ref["canny_strength"]}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": ref["tile_strength"]}}

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": ref["global_pos"]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    region_nodes = []
    for i, r in enumerate(ref["regions"]):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"]}}
        region_nodes.append(sk)

    comb_in = {"global_cond": ["pg", 0]}
    for i, sk in enumerate(region_nodes):
        comb_in[f"region{i+1}"] = [sk, 0]
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0],
        "seed": seed, "steps": 24, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": ref["denoise"]}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0],
        "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
        "upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {
        "images": ["14", 0], "filename_prefix": f"v188_{ref['id']}"}}
    return g


# ==================== 后置：Reinhard LAB 色彩迁移（硬锁配色）====================
def color_transfer(src_bgr, dst_bgr, alpha=1.0):
    """把 dst 的色彩分布对齐 src（让结果配色严格按原图）。alpha=1.0 完全对齐。"""
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
    """RGB 直方图交集（越高=配色越接近）。返回 0-1。"""
    inter = 0.0
    for c in range(3):
        hs = cv2.calcHist([src_bgr], [c], None, [bins], [0, 256])
        hd = cv2.calcHist([dst_bgr], [c], None, [bins], [0, 256])
        cv2.normalize(hs, hs); cv2.normalize(hd, hd)
        inter += cv2.compareHist(hs, hd, cv2.HISTCMP_INTERSECT)
    return inter / 3.0


def structural_diff(src_bgr, dst_bgr):
    """结构差异（越大=裂变越明显；太小=没裂变，太大=崩了）。返回 0-1。"""
    h, w = min(src_bgr.shape[0], dst_bgr.shape[0]), min(src_bgr.shape[1], dst_bgr.shape[1])
    s = cv2.resize(src_bgr, (w, h)); d = cv2.resize(dst_bgr, (w, h))
    mse = ((s.astype(np.float32) - d.astype(np.float32)) ** 2).mean()
    return float(np.clip(1 - mse / (255.0 ** 2), 0, 1))


# ==================== API 调用 ====================
import urllib.request
def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def history(pid):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
        return json.loads(r.read())

def gen(ref, seed):
    import cv2
    g = build(ref, seed)
    r = submit(g)
    pid = r["prompt_id"]
    print(f"  submitted {ref['id']} pid={pid[:8]} denoise={ref['denoise']} ipa={ref['ipa_weight']}")
    for _ in range(72):
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        print(f"  TIMEOUT {ref['id']}")
        return None
    outputs = h[pid].get("outputs", {})
    raw_path = None
    for node_id, info in outputs.items():
        for img in info.get("images", []):
            src = Path(COMFY_INPUT.parent) / "ComfyUI" / "output" / img["filename"]
            if not src.exists():
                src = Path(img.get("abs_path", src))
            if not src.exists():
                continue
            raw_path = str(src)
            break
        if raw_path:
            break
    # 兜底：直接 glob ComfyUI/output 里 v188_{id} 前缀的最新 png
    if not raw_path:
        out_dir = COMFY_INPUT.parent / "ComfyUI" / "output"
        cands = sorted(out_dir.glob(f"v188_{ref['id']}*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        return None
    # 读原图 + 结果（cv2.imread 对本项目 JPEG 偶发失败，统一走 PIL）
    src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / ref["ref_img"]).convert("RGB")), cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(np.array(Image.open(raw_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    # 后置色彩迁移
    matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
    # 保存（cv2.imwrite 在本环境静默失败，统一走 PIL）
    out_raw = JOB / f"v188_{ref['id']}_raw.jpg"
    out_final = JOB / f"v188_{ref['id']}.jpg"
    Image.fromarray(cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)).save(str(out_raw), quality=95)
    Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)).save(str(out_final), quality=95)
    # 量化自查
    hi_before = hist_intersection(src_bgr, dst_bgr)
    hi_after = hist_intersection(src_bgr, matched)
    sd = structural_diff(src_bgr, matched)
    print(f"  saved v188_{ref['id']}.jpg  ({out_final.stat().st_size//1024} KB)")
    print(f"    配色交集: 迁移前={hi_before:.3f} → 迁移后={hi_after:.3f}  | 结构差异(裂变度)={sd:.3f}")
    if hi_after < 0.80:
        print(f"    ⚠ 配色交集仍偏低({hi_after:.3f})，可能引入原图没有的色")
    if sd < 0.45:
        print(f"    ⚠ 结构差异过低({sd:.3f})，可能没真正裂变（接近原图）")
    if sd > 0.88:
        print(f"    ⚠ 结构差异过高({sd:.3f})，可能崩了")
    return str(out_final)


def main():
    import cv2
    targets = sys.argv[1:] if len(sys.argv) > 1 else [r["id"] for r in REFS]
    for r in REFS:
        if r["id"] not in targets:
            continue
        out = JOB / f"v188_{r['id']}.jpg"
        if out.exists() and out.stat().st_size > 100000:
            print(f"[skip] {r['id']} already done")
            continue
        gen(r, SEED)


if __name__ == "__main__":
    main()
