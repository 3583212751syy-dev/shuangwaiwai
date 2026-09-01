"""
v195 — 下一波 6 张桌面原图（ramen_weed 因违规跳过）

跑 6 张：
1. camo_classic (5784eab) — 经典 4 色迷彩无字
2. floral_bw (Pinterest_1) — 黑白花卉卷草
3. denim_patch (Pinterest_3) — 牛仔布剪裁蝴蝶 (UPGY 字 → SDXL NO readable text)
4. palm_camo (Pinterest_4) — 棕榈树 + 4 色迷彩
5. skull_snake_rose (Pinterest_5) — 骷髅红翼蛇玫瑰 (TRUE NEVER DIES → 红色字保留)
6. eagle_skull_metal (Pinterest_6) — 鹰俯视+角骷髅+金属字 (MRCHRGSR 侵权 → 烧字)

每张独立 5 区域提示词，主体保留 + 改角度/大小/数量/换小元素。
v147 锁死基线参数（铁律）：
  DENOISE 0.80 / IPA 0.18 / LORA 1.0 / CANNY 0.25 / TILE 0.60
后置 Reinhard LAB 色彩迁移兜底（v188 已验证）。
NEG_BASE 加 "no readable text" 防文字乱码。
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
JOB = PROJECT / "jobs" / "smoke_v195"
JOB.mkdir(parents=True, exist_ok=True)

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"

SEED = 700550  # 与 v185/v188/v193/v194 都不同

# ==================== v147 锁死基线参数 ====================
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

# ==================== NEG_BASE（v147 标准 + 不可读字）====================
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "legible text, readable text, recognizable letters, clear word, English word, brand name, logo word, "
    "spelled out words, calligraphy, scripture, "
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

# ==================== 6 张 REFS ====================
REFS = [

    # === camo_classic = 经典 4 色迷彩（黑/橄榄绿/棕色/沙）===
    # 主体：圆润有机迷彩色块，加几条粗条纹装备带/弹夹/钢扣符号
    {
        "id": "camo_classic",
        "ref_img": "test_5784eab326634d17573b469e91cdc565.jpg",
        "global_pos": (
            "bold military camouflage print pattern with classic 4-color palette, "
            "olive green #4B5320 and tan #C2B280 and dark brown #4A2C2A and black color blocks, "
            "rounded organic blob shapes with sharp edges between color zones, "
            "fabric print quality, repeatable seamless pattern feel, "
            "no text, no letters, no words anywhere, "
            "tactical military design"
        ),
        "regions": [
            # 加一条粗条纹装备带（保留迷彩主体）
            {"x": 0.00, "y": 0.35, "w": 1.0, "h": 0.10, "strength": 1.15,
             "prompt": (
                 "a single horizontal tactical webbing strip across the middle of the pattern, "
                 "dark olive green belt texture with 6 evenly spaced metallic clip buckles, "
                 "the webbing strap is a NEW element added across the middle of the camo pattern, "
                 "keep the original 4-color camouflage as the base, "
                 "military equipment overlay aesthetic. " + COHESIVE)},
            # 加 3 个钢扣
            {"x": 0.05, "y": 0.10, "w": 0.20, "h": 0.20, "strength": 1.05,
             "prompt": (
                 "a single small circular steel snap button in the upper-left, "
                 "dark gunmetal gray with 4 small center holes, "
                 "tactical gear accent. " + COHESIVE)},
            {"x": 0.75, "y": 0.10, "w": 0.20, "h": 0.20, "strength": 1.05,
             "prompt": (
                 "a SECOND matching steel snap button in the upper-right corner, "
                 "mirror position to the left snap. " + COHESIVE)},
            {"x": 0.45, "y": 0.85, "w": 0.10, "h": 0.10, "strength": 1.00,
             "prompt": (
                 "a small rectangular tactical name-tape patch at the bottom center, "
                 "dark olive color with no text on it (blank woven tape), "
                 "tactical label badge. " + COHESIVE)},
            # 主迷彩保留——加强锐利边缘
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.30,
             "prompt": (
                 "KEEP the classic 4-color camouflage base pattern with rounded organic blob shapes, "
                 "sharp CRISP edges between olive/tan/brown/black color zones, "
                 "NO gradient, NO soft airbrush, hard fabric-print edges. " + COHESIVE)},
        ],
    },

    # === floral_bw = 黑白花卉卷草（左下 → 右上，巴洛克风）===
    # 主体：黑色卷草花纹，白色花卉点缀
    # 裂变：右上转角 30° + 加圆点装饰 + 换一种叶形
    {
        "id": "floral_bw",
        "ref_img": "test_Pinterest_1.jpg",
        "global_pos": (
            "ornate black and white baroque floral scrollwork illustration, "
            "pure black and pure white ONLY, no gray shading, "
            "victorian damask textile pattern, "
            "intricate botanical scrollwork with curling leaves and 6-petal flowers, "
            "no text, no letters, no words anywhere, "
            "sharp clean edges, highly detailed monochrome pattern"
        ),
        "regions": [
            # 主体：右上花卉 + 卷草流
            {"x": 0.35, "y": 0.00, "w": 0.65, "h": 0.50, "strength": 1.30,
             "prompt": (
                 "the upper-right baroque floral cluster: rotate 30 degrees clockwise, "
                 "enlarge the central 8-petal rosette flower by 1.2x, "
                 "replace the side leaves with broader acanthus leaf shapes, "
                 "keep PURE BLACK on pure white only. " + COHESIVE)},
            # 中段卷草主流
            {"x": 0.10, "y": 0.30, "w": 0.65, "h": 0.50, "strength": 1.20,
             "prompt": (
                 "central baroque scroll: the main curling vine sweeps diagonally, "
                 "add 3 NEW small 6-petal flowers along the vine length, "
                 "PURE WHITE on pure black. " + COHESIVE)},
            # 左下：加圆点装饰 + 次卷草
            {"x": 0.00, "y": 0.60, "w": 0.40, "h": 0.40, "strength": 1.10,
             "prompt": (
                 "lower-left mirror scroll curling upward, "
                 "with a row of 8 small white dots fading into the black background, "
                 "decorative flourish ending in a tiny spiral. " + COHESIVE)},
            # 左下角
            {"x": 0.00, "y": 0.85, "w": 0.30, "h": 0.15, "strength": 1.00,
             "prompt": (
                 "a small standalone scroll flourish at the bottom-left corner, "
                 "2 small 5-petal flowers and 4 dots, "
                 "PURE WHITE on pure black. " + COHESIVE)},
            # 顶部：装饰带
            {"x": 0.30, "y": 0.00, "w": 0.40, "h": 0.10, "strength": 0.95,
             "prompt": (
                 "a thin curving tendril exiting at the upper-right edge, "
                 "with a row of 6 small white dots fading, decorative flourish. " + COHESIVE)},
        ],
    },

    # === denim_patch = 牛仔布剪裁蝴蝶 + UPGY + 2 小蝴蝶（类似 denim_3 v178 成功方案）===
    # 主体：牛仔蝴蝶（保留），改角度/加数量/换小元素
    # 注意：UPGY 字按 "no readable text" 走（不烧字，因为不像品牌词）
    {
        "id": "denim_patch",
        "ref_img": "test_Pinterest_3.jpg",
        "global_pos": (
            "denim fabric collage illustration on a clean WHITE background, "
            "denim indigo blue and bleached white tones, "
            "high contrast, fabric-textured patchwork feel, "
            "decorative ornamental stylized lettering integrated as graphical elements "
            "(NOT readable English words, NOT spelled-out words, NOT brand names, NOT logos), "
            "fashion textile print quality, soft natural shadows"
        ),
        "regions": [
            # 主体：牛仔拼贴蝴蝶（保留+补 INTACT/CLEAN）
            {"x": 0.20, "y": 0.40, "w": 0.60, "h": 0.45, "strength": 1.30,
             "prompt": (
                 "ONE large INTACT symmetrical butterfly shape built from layered DENIM PATCHES, "
                 "WHOLE and COMPLETE — NOT torn, NOT frayed, NOT patchwork, NOT ripped, no loose threads, "
                 "denim panel stitching as cross-hatched wing veins, "
                 "two-tone indigo blue denim with one center seam, "
                 "body and antennae in stitched dark thread, "
                 "smooth continuous wing edges with subtle fringed thread tufts. " + COHESIVE)},
            # 顶部：装饰性字区域（替代 UPGY）
            {"x": 0.20, "y": 0.05, "w": 0.60, "h": 0.20, "strength": 1.10,
             "prompt": (
                 "a thin horizontal band at the top with 6 SHORT denim thread tassels hanging downward like fringe, "
                 "each tassel in slightly different blue tone (light wash to dark indigo), "
                 "decorative header element replacing any text in the top area. " + COHESIVE)},
            # 右侧小蝴蝶
            {"x": 0.65, "y": 0.20, "w": 0.25, "h": 0.25, "strength": 1.05,
             "prompt": (
                 "a SECONDARY smaller denim butterfly in the upper-right, "
                 "about one-third the size of the main butterfly, "
                 "diagonally tilted in flight pose, "
                 "lighter bleached denim tone with one small indigo detail, "
                 "same INTACT fabric patchwork style. " + COHESIVE)},
            # 左下第三只蝴蝶
            {"x": 0.05, "y": 0.75, "w": 0.25, "h": 0.20, "strength": 1.00,
             "prompt": (
                 "a TINY denim butterfly in the bottom-left, "
                 "smallest of the three, "
                 "flying in opposite direction from the main, "
                 "pure indigo denim with white stitching. " + COHESIVE)},
            # 点状轨迹
            {"x": 0.30, "y": 0.30, "w": 0.65, "h": 0.50, "strength": 0.95,
             "prompt": (
                 "a trailing arc of 12 small indigo dots forming a sweeping S-curve "
                 "from the small upper-right butterfly DOWN to the main butterfly "
                 "and continuing to the bottom-left tiny butterfly. " + COHESIVE)},
        ],
    },

    # === palm_camo = 棕榈 + 4 色迷彩（类似 camo_4 v174 风格）===
    # 主体：棕榈树 + 迷彩色块；裂变：树转角 + 大小变 + 加沙色装饰带
    {
        "id": "palm_camo",
        "ref_img": "test_Pinterest_4.jpg",
        "global_pos": (
            "tropical military camouflage pattern with palm tree silhouettes, "
            "olive green #4B5320 and tan #C2B280 and dark brown #4A2C2A and black color blocks, "
            "rounded organic blob shapes with sharp edges, "
            "fabric print quality, repeatable seamless feel, "
            "no text, no letters, no words anywhere"
        ),
        "regions": [
            # 中央棕榈（保留 + 改角度）
            {"x": 0.20, "y": 0.00, "w": 0.40, "h": 0.60, "strength": 1.25,
             "prompt": (
                 "a TALL royal palm tree centered, curved trunk slightly bent 20 degrees to the left, "
                 "TOP CROWN of 8-10 wide fan-shaped fronds spreading outward, "
                 "pure black silhouette with crisp outline, "
                 "occasional frond overlapping a neighboring tree to suggest depth. " + COHESIVE)},
            # 右侧较矮椰子树（数量 1→2）
            {"x": 0.55, "y": 0.10, "w": 0.30, "h": 0.55, "strength": 1.15,
             "prompt": (
                 "a SECONDARY shorter coconut palm tree in the right portion, "
                 "sturdier trunk straight up, 5 smaller curved fronds, "
                 "slightly different shape from the main palm, "
                 "pure black silhouette. " + COHESIVE)},
            # 左下小棕榈（数量更多）
            {"x": 0.00, "y": 0.50, "w": 0.30, "h": 0.50, "strength": 1.05,
             "prompt": (
                 "a small palm tree in the bottom-left corner, "
                 "young sapling style, 4 drooping fronds. " + COHESIVE)},
            # 迷彩色块
            {"x": 0.00, "y": 0.20, "w": 1.0, "h": 0.30, "strength": 1.20,
             "prompt": (
                 "bold camouflage color blocks in rounded organic blob shapes, "
                 "olive drab mixed with tan mixed with dark brown, "
                 "LARGE color field with hard crisp edges between blocks, "
                 "no gradient, fabric-print-ready camouflage. " + COHESIVE)},
            # 底部队列装饰
            {"x": 0.00, "y": 0.92, "w": 1.0, "h": 0.08, "strength": 1.00,
             "prompt": (
                 "a thin horizontal band of 6 tiny palm tree silhouettes in a row at the very bottom, "
                 "each slightly smaller than the previous, marching right to left, "
                 "tactical collar-band decoration. " + COHESIVE)},
        ],
    },

    # === skull_snake_rose = 骷髅+红翼+红蛇+红玫瑰+"TRUE NEVER DIES"（骷髅 v169e 风格）===
    # 主体：骷髅+红翼+蛇+玫瑰。裂变：玫瑰数量变 + 蛇转角 + 翅膀尖刺化 + 字作为装饰（不期待拼对）
    {
        "id": "skull_snake_rose",
        "ref_img": "test_Pinterest_5.jpg",
        "global_pos": (
            "gothic tattoo illustration, pure black background, "
            "white skull with red wings and red snake and red roses, "
            "decorative ornamental STYLIZED LETTERING as graphical elements "
            "(NOT readable English words, NOT spelled-out words, NOT brand names), "
            "bold t-shirt graphic print, high contrast, sharp edges"
        ),
        "regions": [
            # 主体：骷髅（保留眼罩+裂纹+血污）
            {"x": 0.30, "y": 0.30, "w": 0.40, "h": 0.45, "strength": 1.30,
             "prompt": (
                 "ONE large forward-facing human skull with realistic cracks across the forehead, "
                 "a single black leather EYE PATCH covering the right eye socket, "
                 "dried blood splatter across the cheek bone and forehead, "
                 "bleached white bone with realistic gray shadow in eye sockets, "
                 "no hat, no helmet, no crown, just bone and patch. " + COHESIVE)},
            # 左侧红翼
            {"x": 0.05, "y": 0.10, "w": 0.30, "h": 0.40, "strength": 1.20,
             "prompt": (
                 "a blood-red TALL NARROW angel wing on the LEFT side of the skull, "
                 "5 distinct feather rows with hard black gaps, "
                 "deep crimson red with darker shading along feather spines. " + COHESIVE)},
            # 右侧红翼
            {"x": 0.65, "y": 0.10, "w": 0.30, "h": 0.40, "strength": 1.20,
             "prompt": (
                 "a MIRROR blood-red ceremonial angel wing on the RIGHT side, "
                 "matching the left in size shape color, "
                 "symmetric to the left wing. " + COHESIVE)},
            # 蛇缠绕
            {"x": 0.30, "y": 0.70, "w": 0.40, "h": 0.30, "strength": 1.15,
             "prompt": (
                 "a blood-red SCALED SERPENT wrapping around the bottom of the skull, "
                 "coiled in an S-shape, head lifted on the right side, "
                 "red scale texture clearly visible. " + COHESIVE)},
            # 顶部装饰性字（替代 TRUE）
            {"x": 0.15, "y": 0.00, "w": 0.70, "h": 0.15, "strength": 1.05,
             "prompt": (
                 "decorative ornamental stylized lettering as graphical elements woven at the very top "
                 "(NOT readable English words, NOT spelled-out words, NOT brand names), "
                 "red letter-shaped ribbons as decorative motifs. " + COHESIVE)},
            # 底部装饰性字（替代 NEVER DIES）+ 玫瑰
            {"x": 0.10, "y": 0.60, "w": 0.25, "h": 0.30, "strength": 1.05,
             "prompt": (
                 "ONE large dark blood-red rose on the left side near the skull cheek, "
                 "fully bloomed with layered petals and visible thorns. " + COHESIVE)},
        ],
    },

    # === eagle_skull_metal = 鹰俯视+角骷髅+金属字 MRCHRGSR（类似 metal_6 v176）===
    # 主体：鹰 + 角骷髅 + 闪电荆棘。裂变：闪电加密 + 鹰转头 + 字烧为 BONE/CROW
    {
        "id": "eagle_skull_metal",
        "ref_img": "test_Pinterest_6.jpg",
        "global_pos": (
            "brutal death metal band illustration, pure black background, "
            "white eagle with brown feather shading, white skull with brown horns, "
            "decorative ornamental STYLIZED LETTERING as graphical elements "
            "(NOT readable English words, NOT spelled-out words, NOT brand names), "
            "extreme line weight contrast, harsh jagged edges, "
            "no legible English words anywhere, "
            "bold t-shirt graphic print quality"
        ),
        "regions": [
            # 鹰主体（保留俯视 + 改转头）
            {"x": 0.30, "y": 0.10, "w": 0.40, "h": 0.40, "strength": 1.30,
             "prompt": (
                 "a HUGE bald eagle FACING THE CAMERA with wings half-spread downward, "
                 "head slightly tilted DOWN with piercing orange-yellow eyes and half-open yellow beak, "
                 "white head feathers with sharp hatch lines, "
                 "brown BODY feathers with very bold geometric block-shading, "
                 "yellow legs with sharp talons gripping forward. " + COHESIVE)},
            # 角骷髅
            {"x": 0.25, "y": 0.45, "w": 0.50, "h": 0.40, "strength": 1.30,
             "prompt": (
                 "a large white human skull DIRECTLY BELOW the eagle, "
                 "mouth wide open in a roar (no helmet on), "
                 "FOUR long curved horns sprouting from the skull crown, "
                 "two main horns curling outward and up to either side, "
                 "brown horns with rough texture. " + COHESIVE)},
            # 左侧闪电
            {"x": 0.00, "y": 0.15, "w": 0.25, "h": 0.75, "strength": 1.15,
             "prompt": (
                 "a burst of 7 sharp white metal lightning spikes fanning out from "
                 "the eagle-skull junction on the left side, "
                 "spikes of varying lengths the longest 1.5x the shortest, "
                 "pure white with crisp outline, no gradient. " + COHESIVE)},
            # 右侧闪电
            {"x": 0.75, "y": 0.15, "w": 0.25, "h": 0.75, "strength": 1.15,
             "prompt": (
                 "a MIRROR burst of 7 sharp white metal lightning spikes on the right side, "
                 "symmetric to the left, matching length variation. " + COHESIVE)},
            # 顶部装饰性字（替代 MRCHRGSR）
            {"x": 0.10, "y": 0.00, "w": 0.80, "h": 0.15, "strength": 1.00,
             "prompt": (
                 "decorative ornamental stylized lettering as graphical elements at the very top "
                 "(NOT readable English words, NOT spelled-out words, NOT brand names), "
                 "white spike-like letterform ribbons as decorative band, "
                 "metal-style ornaments. " + COHESIVE)},
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

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": ref["global_pos"]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    region_nodes = []
    for i, r in enumerate(ref["regions"]):
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
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v195_{ref['id']}"}}
    return g


# ==================== 后置：Reinhard LAB 色彩迁移 ====================
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


# ==================== API ====================
import urllib.request
def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def history(pid):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
        return json.loads(r.read())


def gen(ref, seed):
    g = build(ref, seed)
    r = submit(g)
    pid = r["prompt_id"]
    print(f"  submitted {ref['id']} pid={pid[:8]}", flush=True)
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
    if not raw_path:
        out_dir = COMFY_INPUT.parent / "ComfyUI" / "output"
        cands = sorted(out_dir.glob(f"v195_{ref['id']}*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            raw_path = str(cands[0])
    if not raw_path:
        return None

    src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / ref["ref_img"]).convert("RGB")), cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(np.array(Image.open(raw_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    matched = color_transfer(src_bgr, dst_bgr, alpha=1.0)
    out_final = JOB / f"v195_{ref['id']}.jpg"
    Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)).save(str(out_final), quality=95)
    hi_before = hist_intersection(src_bgr, dst_bgr)
    hi_after = hist_intersection(src_bgr, matched)
    sd = structural_diff(src_bgr, matched)
    print(f"  saved v195_{ref['id']}.jpg  ({out_final.stat().st_size//1024} KB)", flush=True)
    print(f"    配色: 前={hi_before:.3f} → 后={hi_after:.3f} | 结构差异={sd:.3f}", flush=True)
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        res = reader.readtext(str(out_final))
        if res:
            print(f"    ⚠ OCR 读到 {len(res)} 处: " + ", ".join(f'{t}({c:.2f})' for _, t, c in res), flush=True)
        else:
            print(f"    ✅ OCR 未读到任何字符", flush=True)
    except Exception as e:
        print(f"    OCR 失败: {e}", flush=True)
    return str(out_final)


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else [r["id"] for r in REFS]
    for r in REFS:
        if r["id"] not in targets:
            continue
        out = JOB / f"v195_{r['id']}.jpg"
        if out.exists() and out.stat().st_size > 100000:
            print(f"[skip] {r['id']} already done", flush=True)
            continue
        gen(r, SEED)


if __name__ == "__main__":
    main()
