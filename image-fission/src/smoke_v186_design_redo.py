"""
v186 设计稿专用管线 — 重做 v185 失败的 5 张（orange_paisley / racing / dark_knight / bat_logo / camo_armed）
v147 (denoise 0.80 / Canny 0.25 / Tile 0.60 / IPA 0.18 / LORA 1.0) 对设计稿/排版/几何花纹/logo稿完全失效，
元素被重绘力打飞。新参数：
- denoise 0.30-0.40（极低重绘，保留原图所有元素位置）
- Canny 0.65-0.75（高锁轮廓，锁住所有文字/图标边缘）
- Tile 0.85+（高锁构图，保留版面/分格/网格布局）
- IPA 0.05-0.12（低权重锁配色，不锁内容）
- LORA 0.4-0.5（降噪，避免高频细节破坏文字）

提示词铁律：写什么 AI 就画什么。本批每张提示词都明确「保留 X 元素位置」+「只调整 Y 细节」，
不重绘主体。配色铁律（v147 同样的 IPA 锁色）+ 主体保留铁律。

5 张单独指定参数；其他参数（PROTEUS 基底 / 4x_NMKD-Siax / 双 KSampler 24+20）保留。
"""

import os, sys, json, time, shutil, uuid
from pathlib import Path
from PIL import Image, ImageFilter

# ==================== 配置 ====================
PROJECT = Path(__file__).resolve().parents[1]
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v186"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 700502  # 与 v185 不同，避免雷同

# ==================== NEG_BASE（v185 基础上加强设计稿禁词）====================
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "garbled text, illegible letters, mangled typography, distorted characters, broken text, misspelled text, "
    "wrong letters, scrambled letters, partially missing letters, "
    "banner, banner inscription, engraved lettering, runic text, readable text, glyphs, calligraphy, "
    "3d, photographic, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "melted, fused, smudged, bleeding, water damaged, anatomically incorrect, "
    "extra wings, asymmetric error, garbled forms, nonsense, AI artifact, tiling, repeating pattern, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "elements touching each other, elements touching neighboring elements, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "crowded center, cluttered middle area, "
    "low contrast between adjacent elements, no clear black separating outline, "
    "new colors, different color palette, extra colors, color shift"
)

COHESIVE = (
    "cohesive with the rest of the design, all elements in their exact original positions, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

# ==================== 5 张 REFS（设计稿专用参数）====================
REFS = [

    # === 1. 橙色佩斯利 4 格面板 — denoise 0.35 / Canny 0.70 / Tile 0.85 / IPA 0.10 ===
    {
        "id": "orange_paisley",
        "ref_img": "test_13c8b7bf8dae757e6c2d4b3d6a860f9d.jpg",
        "denoise": 0.35,
        "canny_strength": 0.70,
        "tile_strength": 0.85,
        "ipa_weight": 0.10,
        "lora_detail": 0.5,
        "global_pos": (
            "ornate orange paisley bandana print pattern, "
            "warm orange and white and dark brown and deep red color blocks, "
            "traditional persian paisley teardrop motifs, "
            "fabric bandana quality, repeatable seamless pattern feel, "
            "no text, no letters, no words anywhere, "
            "sharp clean edges, intricate detail"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "PRESERVE EXACTLY: four equal square panels in 2x2 grid layout, "
                 "top-left top-right bottom-left bottom-right quadrants each with size tag in bottom-right corner (s/m/l/xl), "
                 "KEEP the 2x2 grid divider lines visible, "
                 "KEEP every existing paisley teardrop motif position unchanged, "
                 "KEEP warm orange + white + dark brown + deep red palette, "
                 "only ENRICH each teardrop with MORE intricate baroque inner curlwork and tiny leaf fillings, "
                 "add fine dark-brown vine connectors between teardrops, "
                 "make pattern more ornate and detailed but DO NOT move any motif or change layout. " + COHESIVE)},
        ],
    },

    # === 2. RACING T恤设计图 — denoise 0.30 / Canny 0.75 / Tile 0.85 / IPA 0.08 ===
    {
        "id": "racing",
        "ref_img": "test_184432b34a4787fbed628b3b986b37a2.jpg",
        "denoise": 0.30,
        "canny_strength": 0.75,
        "tile_strength": 0.85,
        "ipa_weight": 0.08,
        "lora_detail": 0.4,
        "global_pos": (
            "bold racing motorsport t-shirt design layout, "
            "white and red and black color blocks, "
            "horizontal red bands between text rows, "
            "racing typography, flag motifs, speedometer numerals, "
            "no text, no letters, no words anywhere, "
            "sharp clean edges"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "PRESERVE EXACTLY: the existing 5-row text layout, "
                 "top row big italic RACING (with checkered flag icon to its left and circular emblem to its right), "
                 "second row small italic LIFE BECOMES LIMITLESS inside a horizontal bar, "
                 "third row CHICAGO with DREAM MAKER subtitle, "
                 "fourth row FOURTY-SIX, "
                 "fifth row big italic numerals 23, "
                 "KEEP the 4 horizontal red bands separating each row, "
                 "KEEP the white background between rows, "
                 "KEEP all text positions and flag/emblem positions unchanged, "
                 "only ENHANCE the typography to be slightly more italic-angled motorsport sans-serif with sharper edges, "
                 "make the red bands slightly more saturated. " + COHESIVE)},
        ],
    },

    # === 3. 黑暗骑士跪地插画 — denoise 0.40 / Canny 0.65 / Tile 0.85 / IPA 0.12 ===
    {
        "id": "dark_knight",
        "ref_img": "test_3a300c32794aeea08f8abb2517f3afe1.jpg",
        "denoise": 0.40,
        "canny_strength": 0.65,
        "tile_strength": 0.85,
        "ipa_weight": 0.12,
        "lora_detail": 0.5,
        "global_pos": (
            "dark gothic knight illustration, "
            "pure black and dark charcoal and silver and deep red color blocks, "
            "kneeling knight with sword, "
            "multiple lines of gothic blackletter text on the left and bottom, "
            "no text, no letters, no words anywhere, "
            "dramatic dark atmosphere"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "PRESERVE EXACTLY: kneeling knight in plate armor at center-right, "
                 "helmet visor down, sword planted point-down in ground before him, "
                 "left half of canvas filled with 14 stacked horizontal lines of gothic blackletter text "
                 "(THE DEVIL SAW ME WITH MY HEAD DOWN AND I THOUGHT I'D WON UNTIL I SAID I COULDN'T DO IT), "
                 "bottom band has ONE more line of text, "
                 "KEEP the knight pose exactly, sword position, helmet, armor plates, "
                 "KEEP every text line position and gothic blackletter style unchanged, "
                 "KEEP pure black background, "
                 "only ENHANCE the armor with more intricate gothic engravings and subtle dark-red heraldic trim, "
                 "add faint dark-red glowing runes along the sword blade. " + COHESIVE)},
        ],
    },

    # === 4. 紫底圆形蝙蝠徽章 logo — denoise 0.30 / Canny 0.75 / Tile 0.85 / IPA 0.08 ===
    {
        "id": "bat_logo",
        "ref_img": "test_6978fabda2cc99629fa9e81f802762d3.jpg",
        "denoise": 0.30,
        "canny_strength": 0.75,
        "tile_strength": 0.85,
        "ipa_weight": 0.08,
        "lora_detail": 0.4,
        "global_pos": (
            "ornate purple gothic emblem logo design, "
            "deep purple and white and silver color blocks, "
            "circular medallion with bat silhouette inside, "
            "curved text arched above and straight text below, "
            "decorative baroque scrollwork flourishes, "
            "no text, no letters, no words anywhere, "
            "sharp clean vector edges"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "PRESERVE EXACTLY: the central CIRCULAR MEDALLION (concentric ring design), "
                 "a BAT SILHOUETTE inside the medallion with spread wings, "
                 "curved/arched uppercase text wrapping the TOP of the medallion (curved banner style), "
                 "straight uppercase text below the medallion (currently reads MHEART), "
                 "ornate baroque scrollwork flourishes extending left and right from the medallion, "
                 "tiny decorative dots and a small crown ornament at top-center, "
                 "KEEP deep purple background, "
                 "KEEP medallion circular ring exactly, "
                 "KEEP bat silhouette position inside medallion, "
                 "KEEP every text position and curved-arch layout unchanged, "
                 "only REFINE the curved text to read a non-infringing placeholder word (will be replaced by PIL burn), "
                 "make scrollwork slightly more intricate. " + COHESIVE)},
        ],
    },

    # === 5. 白底运动衫印花「WE SUPPORT THE ARMED FORCES」 — denoise 0.30 / Canny 0.75 / Tile 0.85 / IPA 0.05 ===
    {
        "id": "camo_armed",
        "ref_img": "test_b78e60de8dfdf44acda99395326a7298.jpg",
        "denoise": 0.30,
        "canny_strength": 0.75,
        "tile_strength": 0.85,
        "ipa_weight": 0.05,
        "lora_detail": 0.4,
        "global_pos": (
            "white background military support athletic t-shirt design, "
            "pure white and dark charcoal and black color blocks, "
            "bold black sans-serif text in three stacked rows, "
            "military dog tag pendant, "
            "clean product print mockup style, "
            "no text, no letters, no words anywhere, "
            "sharp clean edges"
        ),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": (
                 "PRESERVE EXACTLY: pure white background, "
                 "three rows of bold black sans-serif uppercase text stacked center: "
                 "top row WE SUPPORT THE, middle row ARMED (largest), bottom row FORCES, "
                 "a military dog tag pendant with chain hanging from the upper area, "
                 "KEEP every text row position and character exactly as-is, "
                 "KEEP dog tag pendant and chain position unchanged, "
                 "KEEP pure white background pure, no camo no pattern no decoration, "
                 "only REFINE the typography to a slightly more modern military stencil-style sans-serif, "
                 "add subtle black outline on each text character for sharper print-mockup feel, "
                 "DO NOT add any camo, no military pattern, no chain, no background color. " + COHESIVE)},
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
        "images": ["14", 0], "filename_prefix": f"v186_{ref['id']}"}}
    return g


# ==================== API 调用 ====================
import urllib.request, urllib.parse

def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def history(pid):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
        return json.loads(r.read())

def gen(ref, seed, out_base):
    g = build(ref, seed)
    r = submit(g)
    pid = r["prompt_id"]
    print(f"  submitted {ref['id']} pid={pid[:8]} denoise={ref['denoise']} canny={ref['canny_strength']} tile={ref['tile_strength']} ipa={ref['ipa_weight']}")
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
    for node_id, info in outputs.items():
        for img in info.get("images", []):
            src = Path(COMFY_INPUT.parent) / "ComfyUI" / "output" / img["filename"]
            if not src.exists():
                # 备用：直接读绝对路径
                src = Path(img.get("abs_path", src))
            if not src.exists():
                continue
            dst = out_base.parent / out_base.name
            shutil.copy(src, dst)
            print(f"  saved -> {dst}  ({dst.stat().st_size//1024} KB)")
            return str(dst)
    return None


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else [r["id"] for r in REFS]
    for r in REFS:
        if r["id"] not in targets:
            continue
        out = JOB / f"v186_{r['id']}.jpg"
        if out.exists() and out.stat().st_size > 100000:
            print(f"[skip] {r['id']} already done")
            continue
        gen(r, SEED, out)
    print(f"\n[done] {JOB}")

if __name__ == "__main__":
    main()