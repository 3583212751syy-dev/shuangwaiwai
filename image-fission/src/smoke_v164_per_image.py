"""v164 多图 v147 模板裂变（5 区域提示 + Tile 0.60 + Canny 0.25 + 双 KSampler）

每张图独立 5 区域 prompt（保留主体+可改角度/数量/小元素），共用 v147 技术管线。
- 主体保留铁律（用户 2026-08-29 立）：每张原图的主体类型不替换，可改角度/姿态/数量
- 小元素可换（顶饰/侧装饰/角饰按各图真实内容定制）

v147 公共参数：ProteusV0.4 / IPA 0.18 style / add-detail-xl 1.0 /
                CN Tile 0.60 + CN Canny 0.25 / 5 region / KSampler 24+20 / 4x upscale

用法：python smoke_v164_per_image.py  (按 REFS 列表跑全部 6 张；eagle_2 已跑过会自动跳过)
"""
import time, requests, sys, os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.2

CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

# 共享 NEG（每图自定义 NEG 在 REFS 中补充）
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, alphabet, "
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
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

# ===== 6 个 ref 的独立 5 区域配置（主体保留 + 可改角度/数量/小元素）=====
REFS = [
    # === eagle_2（已 v147 跑过，留此跳过）===
    {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "global_pos": ("gothic tattoo illustration, pure black background, "
                       "red and orange flames, white and silver eagle and skull, gray iron, "
                       "bold t-shirt graphic print, high contrast, sharp edges, "
                       "no text, no letters, no words, no banner, no inscription anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [],  # 留空，触发跳过
    },

    # === camo_4 = 棕榈迷彩（黑棕绿）===
    # 主体：棕榈树+军绿/橄榄/棕色/沙色迷彩块。改进：换棕榈种类（扇叶→羽叶）+ 加深度（光影）+ 添动态（摇摆）
    {
        "id": "camo_4", "ref_img": "pinterest_camo_4.jpg",
        "global_pos": ("bold military camouflage print pattern, vector illustration style, "
                       "olive green and tan and dark brown color blocks with sharp edges, "
                       "black palm tree silhouettes with crisp outline, sharp contrast, "
                       "no text, no letters, no words anywhere, "
                       "fabric print quality, repeatable seamless pattern feel"),
        "regions": [
            # 主体：居中一棵大棕榈（保留 palm 主体，改羽叶→扇叶种类，加微风摇动态）
            {"x": 0.30, "y": 0.00, "w": 0.40, "h": 0.50, "strength": 1.25,
             "prompt": ("a TALL bold royal palm tree centered in the design, "
                        "thin curving trunk slightly bent by wind, "
                        "TOP CROWN of wide fan-shaped fronds (NOT feather pinnate leaves) "
                        "spreading in 8 to 10 distinct plumes, "
                        "pure black silhouette with crisp clean outlines, "
                        "occasional frond overlapping a neighboring tree to suggest depth, "
                        "tropical military style. " + COHESIVE)},
            # 主体旁一棵较小的椰子树（数量从原 1 棵→2 棵对称）
            {"x": 0.62, "y": 0.20, "w": 0.30, "h": 0.50, "strength": 1.20,
             "prompt": ("a SECONDARY shorter coconut palm tree in the right portion, "
                        "sturdier trunk straight up, smaller curved fronds in 5 plumes, "
                        "slightly different shape from the main palm to break uniformity, "
                        "pure black silhouette with crisp outline. " + COHESIVE)},
            # 左下小棕榈（数量更多）
            {"x": 0.05, "y": 0.45, "w": 0.30, "h": 0.55, "strength": 1.10,
             "prompt": ("a small palm tree in the bottom-left corner, "
                        "young sapling style, only 4 drooping fronds, "
                        "tucked behind a camo color block, "
                        "pure black silhouette. " + COHESIVE)},
            # 迷彩色块加强：深浅堆叠，增加视觉层次
            {"x": 0.00, "y": 0.20, "w": 1.0, "h": 0.30, "strength": 1.30,
             "prompt": ("bold camouflage color blocks in irregular organic blob shapes, "
                        "olive drab #4B5320 mixed with tan #C2B280 mixed with dark brown #4A2C2A "
                        "and dusty khaki, "
                        "LARGE color field with hard crisp edges between blocks, "
                        "no gradient, no soft airbrush, "
                        "fabric-print-ready camouflage. " + COHESIVE)},
            # 底部一条装饰带：小棕榈头像队列
            {"x": 0.00, "y": 0.92, "w": 1.0, "h": 0.08, "strength": 1.05,
             "prompt": ("a thin horizontal band of SIX tiny palm tree silhouettes in a row at the very bottom, "
                        "each slightly smaller than the previous, marching right to left, "
                        "pure black silhouettes with no detail, "
                        "tactical collar-band decoration. " + COHESIVE)},
        ],
    },

    # === illust_1 = 黑白巴洛克花卉卷草纹（黑底白花纹 / 白底黑花纹）===
    # 主体：黑白花纹卷草，保持对称繁复，去掉具体文字（无文字）
    {
        "id": "illust_1", "ref_img": "pinterest_illust_1.jpg",
        "global_pos": ("ornate black and white baroque floral scrollwork illustration, "
                       "high contrast monochrome, pure black and pure white ONLY, "
                       "victorian damask textile pattern, "
                       "no text, no letters, no words anywhere, "
                       "sharp clean edges, intricate detail, repeatable ornamental"),
        "regions": [
            # 主体：大花朵核心 + 卷须集中在右上
            {"x": 0.35, "y": 0.00, "w": 0.65, "h": 0.55, "strength": 1.30,
             "prompt": ("a LARGE central baroque flower with 6 layered petals at the heart of the upper right, "
                        "surrounded by swirling acanthus leaf scrolls flowing down-right, "
                        "fine dot-work circles and dotted beads between leaves, "
                        "PURE WHITE on pure black, NO gray shading, "
                        "ornamental silhouette with crisp detail. " + COHESIVE)},
            # 卷草主流：从右上延伸到左下
            {"x": 0.10, "y": 0.40, "w": 0.60, "h": 0.60, "strength": 1.20,
             "prompt": ("a flowing baroque acanthus scroll vine sweeping diagonally from upper-right "
                        "to lower-left, with 4 small 6-petal flowers spaced along its length, "
                        "fine leafy tendrils branching outward, "
                        "PURE WHITE on pure black. " + COHESIVE)},
            # 左下方对称卷曲
            {"x": 0.00, "y": 0.55, "w": 0.45, "h": 0.45, "strength": 1.15,
             "prompt": ("a swirling mirror-image baroque scroll in the lower-left corner, "
                        "curling upward in a flourish, "
                        "two small simple 5-petal flowers inside the curl, "
                        "PURE WHITE on pure black, mirror composition. " + COHESIVE)},
            # 右下点状装饰
            {"x": 0.55, "y": 0.78, "w": 0.45, "h": 0.22, "strength": 1.05,
             "prompt": ("a thin curving tendril exiting at the lower-right edge, "
                        "with a row of 8 small white dots fading into the black, "
                        "decorative flourish ending in a tiny spiral. " + COHESIVE)},
            # 顶部小细节
            {"x": 0.40, "y": 0.05, "w": 0.25, "h": 0.10, "strength": 1.00,
             "prompt": ("a small standalone decorative emblem at the top center, "
                        "a circular medallion with a single stylized 8-petal rosette, "
                        "PURE WHITE on pure black, "
                        "tiny ornamental seal. " + COHESIVE)},
        ],
    },

    # === denim_3 = 牛仔布拼贴蝴蝶 + 小蝴蝶 + 蓝白调 ===
    # 主体：牛仔布蝴蝶（保留主体），加叠层+绣线细节
    {
        "id": "denim_3", "ref_img": "pinterest_denim_3.jpg",
        "global_pos": ("denim fabric collage illustration on a clean bright WHITE background, "
                       "denim indigo blue and bleached white tones, "
                       "high contrast, fabric-textured patchwork feel, "
                       "no text, no letters, no words anywhere, "
                       "fashion textile print quality, soft natural shadows"),
        "regions": [
            # 主体：牛仔拼贴蝴蝶（保留，布片层叠+绣线可见）
            {"x": 0.20, "y": 0.40, "w": 0.60, "h": 0.45, "strength": 1.30,
             "prompt": ("ONE large butterfly shape built from layered DENIM PATCHES, "
                        "irregular fabric pieces with visible raw frayed edges along the wing outlines, "
                        "denim panel stitching as cross-hatched wing veins, "
                        "two-tone indigo blue denim with one center seam, "
                        "body and antennae in stitched dark thread, "
                        "occasional frayed thread tufts escaping wing edges. " + COHESIVE)},
            # 右侧第二只蝴蝶（数量 1→2）
            {"x": 0.65, "y": 0.20, "w": 0.20, "h": 0.20, "strength": 1.10,
             "prompt": ("a SECONDARY smaller denim butterfly in the upper-right, "
                        "about one-third the size of the main butterfly, "
                        "diagonally tilted in flight pose, "
                        "lighter bleached denim tone with one small indigo detail, "
                        "same fabric patchwork style. " + COHESIVE)},
            # 左下第三只蝴蝶（数量再+1，对角构图）
            {"x": 0.10, "y": 0.78, "w": 0.20, "h": 0.15, "strength": 1.00,
             "prompt": ("a TINY denim butterfly in the bottom-left, "
                        "smallest of the three, "
                        "flying in opposite direction from the main, "
                        "pure indigo denim with white stitching. " + COHESIVE)},
            # 点状轨迹：连接三只蝴蝶（装饰）
            {"x": 0.30, "y": 0.30, "w": 0.65, "h": 0.50, "strength": 0.95,
             "prompt": ("a trailing arc of 12 small indigo dots forming a sweeping S-curve "
                        "from the small upper-right butterfly DOWN to the main butterfly "
                        "and continuing to the bottom-left tiny butterfly, "
                        "dots evenly spaced, soft fabric-ink feel. " + COHESIVE)},
            # 顶部装饰：牛仔线头流苏
            {"x": 0.30, "y": 0.05, "w": 0.40, "h": 0.12, "strength": 1.00,
             "prompt": ("a small horizontal strip at the top with 6 SHORT denim thread tassels "
                        "hanging downward like fringe, "
                        "each tassel in slightly different blue tone (light wash to dark indigo), "
                        "decorative header element. " + COHESIVE)},
        ],
    },

    # === skull_5 = 黑红骷髅 + 翼 + 蛇 + 玫瑰 + 血 ===
    # 主体：戴黑色眼罩的骷髅 + 缠眼镜蛇（保留），改进：翅膀从顶盖扇形→尖锐鞭状
    {
        "id": "skull_5", "ref_img": "pinterest_skull_5.jpg",
        "global_pos": ("gothic tattoo illustration, pure black background, "
                       "white skull with deep cracks, blood-red wings rose snake, "
                       "bold t-shirt graphic print, high contrast, sharp edges, "
                       "no text, no letters, no words, no banner, no inscription, "
                       "cohesive dark composition"),
        "regions": [
            # 主体：骷髅头（保留眼罩+血污+裂纹）
            {"x": 0.32, "y": 0.30, "w": 0.36, "h": 0.45, "strength": 1.35,
             "prompt": ("ONE large forward-facing human skull with realistic cracks across the forehead, "
                        "a single black leather EYE PATCH covering the right eye socket, "
                        "dried blood splatter across the cheek bone and forehead, "
                        "bleached white bone with realistic gray shadow in eye sockets, "
                        "no hat, no helmet, no crown, just bone and patch. " + COHESIVE)},
            # 红翼（保留双侧，换短粗羽→长尖刺羽）
            {"x": 0.05, "y": 0.10, "w": 0.30, "h": 0.35, "strength": 1.20,
             "prompt": ("a blood-red CEREMONIAL angel wing on the LEFT side of the skull, "
                        "TALL and NARROW with long sharp pointed feathers, "
                        "5 distinct feather rows separated by hard black gaps, "
                        "deep crimson red with darker shading along feather spines, "
                        "wing tip pointing upper-left corner. " + COHESIVE)},
            {"x": 0.65, "y": 0.10, "w": 0.30, "h": 0.35, "strength": 1.20,
             "prompt": ("a MIRROR blood-red ceremonial angel wing on the RIGHT side, "
                        "matching the left in size shape and color, "
                        "wing tip pointing upper-right corner, "
                        "symmetric to the left wing. " + COHESIVE)},
            # 蛇缠绕（保留，改变缠绕方式：从颈部→躯干缠绕）
            {"x": 0.30, "y": 0.72, "w": 0.40, "h": 0.28, "strength": 1.15,
             "prompt": ("a blood-red SCALED SERPENT wrapping around the bottom of the skull, "
                        "coiled in an S-shape, "
                        "head lifted on the right side facing the skull's mouth, "
                        "forked tongue flicking outward, "
                        "red scale texture clearly visible. " + COHESIVE)},
            # 玫瑰（数量从 2 → 1 大朵+ 几片散落花瓣）
            {"x": 0.10, "y": 0.55, "w": 0.20, "h": 0.30, "strength": 1.05,
             "prompt": ("ONE large dark blood-red rose on the left side near the skull cheek, "
                        "fully bloomed with layered petals and visible thorns on the stem, "
                        "a few scattered loose petals falling downward, "
                        "no white rose, no pink rose, only deep crimson. " + COHESIVE)},
        ],
    },

    # === metal_6 = 死亡金属鹰 + 骷髅 + MRCHGSR + 闪电 ===
    # 主体：鹰 + 角骷髅 + 闪电 + 闪电铁刺（保留主要元素，强化锐利度）
    {
        "id": "metal_6", "ref_img": "pinterest_metal_6.jpg",
        "global_pos": ("brutal death metal band illustration, pure black background, "
                       "white eagle, brown feather shading, white skull with brown horns, "
                       "high contrast sharp extreme detail, bold graphic print, "
                       "no text, no letters, no words, no banner, no inscription, "
                       "extreme line weight contrast, harsh jagged edges"),
        "regions": [
            # 主体鹰（保留正面俯视，改羽毛→更具刺突质感的刃羽）
            {"x": 0.30, "y": 0.10, "w": 0.40, "h": 0.40, "strength": 1.30,
             "prompt": ("a HUGE bald eagle FACING THE CAMERA with wings half-spread downward, "
                        "both shoulder blades visible, "
                        "head slightly tilted down with piercing orange-yellow eyes and half-open yellow beak, "
                        "white head feathers with sharp hatch lines, "
                        "brown BODY feathers with very bold geometric block-shading, "
                        "yellow legs with sharp talons gripping forward, "
                        "fierce intense stare. " + COHESIVE)},
            # 角骷髅（保留，改 2 弯角→ 4 长尖角）
            {"x": 0.25, "y": 0.45, "w": 0.50, "h": 0.40, "strength": 1.30,
             "prompt": ("a large white human skull DIRECTLY BELOW the eagle, "
                        "mouth wide open in a roar (no helmet on), "
                        "FOUR long curved horns sprouting from the skull crown, "
                        "two main horns curling outward and up to either side, "
                        "two secondary shorter horns rising straight up between them, "
                        "brown horns with rough texture. " + COHESIVE)},
            # 左侧闪电荆棘（保留，改稀疏→密）
            {"x": 0.00, "y": 0.15, "w": 0.25, "h": 0.75, "strength": 1.15,
             "prompt": ("a burst of 7 sharp white metal lightning spikes fanning out from "
                        "the eagle-skull junction on the left side, "
                        "spikes of varying lengths the longest 1.5x the shortest, "
                        "each spike has 3-4 short perpendicular barbs, "
                        "pure white with crisp outline, no gradient. " + COHESIVE)},
            # 右侧闪电荆棘
            {"x": 0.75, "y": 0.15, "w": 0.25, "h": 0.75, "strength": 1.15,
             "prompt": ("a MIRROR burst of 7 sharp white metal lightning spikes on the right side, "
                        "symmetric to the left, "
                        "matching length variation. " + COHESIVE)},
            # 底部羽毛笔触装饰带
            {"x": 0.10, "y": 0.88, "w": 0.80, "h": 0.12, "strength": 1.00,
             "prompt": ("a thin band of 5 sharp downward white spikes at the very bottom edge, "
                        "like a jagged maw of teeth, "
                        "each spike 3x as tall as wide, "
                        "creates a metallic crown frame for the whole design. " + COHESIVE)},
        ],
    },
]


def scaled_region_strengths(ref):
    return [{**r, "strength": r["strength"] * REGION_STRENGTH_SCALE} for r in ref["regions"]]


def build(ref, seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": MEGA_PIXELS, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}

    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1],
        "lora_name": "add-detail-xl.safetensors",
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

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": ref["global_pos"]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    region_nodes = []
    for i, r in enumerate(scaled_region_strengths(ref)):
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
        "latent_image": ["4", 0], "seed": seed, "steps": 24, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v164_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v164_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在 {out.stat().st_size/1024/1024:.1f}MB，跳过", flush=True); return True
    if not ref.get("regions"):
        print(f"  [{tag}] 无 regions 配置，跳过", flush=True); return True

    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v164_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {json.dumps(j)[:1500]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {tag}: 无 prompt_id {str(j)[:400]}", flush=True); return False
    print(f"  [{tag}] pid={pid} running...", flush=True)
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
                        try:
                            data = requests.get(url, timeout=60).content
                        except Exception as e:
                            print(f"  [{tag}] 取图失败 {e}", flush=True); return False
                        out.write_bytes(data)
                        try:
                            from PIL import Image, ImageFilter
                            im = Image.open(out).convert('RGB')
                            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                            sharp.save(out, 'JPEG', quality=95, optimize=True)
                            print(f"  [{tag}] USM锐化 {out.stat().st_size/1024/1024:.1f}MB", flush=True)
                        except Exception as e:
                            print(f"  [{tag}] USM失败 原图保留 {e}", flush=True)
                        print(f"  [{tag}] OK {out.stat().st_size/1024/1024:.1f}MB", flush=True); return True
                elif rec.get("status", {}).get("error"):
                    err = rec["status"].get("error")
                    print(f"  [{tag}] COMFY错误 {str(err)[:600]}", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] 轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT 跳过重试", flush=True); return False


def main():
    wants = sys.argv[1:] if len(sys.argv) > 1 else [r["id"] for r in REFS]
    out = PROJECT_ROOT / "jobs" / "smoke_v164"
    out.mkdir(parents=True, exist_ok=True)
    for want in wants:
        ref = next((r for r in REFS if r["id"] == want), None)
        if not ref:
            print(f"未知 ref_id={want}，可选: {[r['id'] for r in REFS]}"); continue
        print(f"\n=== {want} ===", flush=True)
        gen(ref, SEED, out)
    print("\nALL done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
