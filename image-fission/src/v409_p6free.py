# -*- coding: utf-8 -*-
"""v409_p6free — p6 鹰骷髅 **掩膜内自由重画**（第 29 轮，"变化要大"的决定性解法）

问题定性（用户第 29 轮，第 3 次说"没拉开"）
--------------------------------------------
第 27 轮：换种子（cn .20）→ 只改笔触，姿态不变。
第 28 轮：v147 区域提示 + hires 两遍（Tile .25→.50）→ **剪影仍与原图同构**
          （实测 IoU 0.733，但展翼角度/骷髅正面朝向肉眼几乎一致）。
根因：整条链路是 **img2img + ControlNet(Tile/Canny)**。ControlNet 的物理作用就是
"把原图的结构透传进 latent"；Tile 更强（逐块纹理对齐）。只要它们还在，
提示词能改的只有"笔触/细节/局部形状"，**改不了姿态骨架**。压 strength 会崩解剖，
不压就只能小改 —— 这条路上无解（第 27/28 轮已把两端都试到）。

本轮换机制
----------
**掩膜内 denoise = 1.0 的自由重画**（`SetLatentNoiseMask` + KSampler denoise 1.0）：
  · 掩膜内 latent **从纯噪声开始** → ControlNet 的"透传原结构"不再有立足点
    （它只能在 early steps 影响一点点 8% 步数），**姿态真正由提示词决定**；
  · 掩膜外像素原封不动 → 黑底 / 标题带 / 细射线天然安全；
  · 提示词仍按 **v147 逐元素区域**（RegionalListCombine + ConditioningSetAreaPercentage）
    组织 —— 每个元素一条独立 prompt，显式写"换成什么姿态"；
  · 解剖仍靠 **守护区**（鹰头 strength 1.60 / 鹰爪 1.45，最后叠加即最优先）；
  · 放大用 hires **两遍**：第二遍的 Tile/Canny 参考图 = **第一遍自己的输出**（自引导），
    只把平涂硬边找回来，不会把姿态拽回原图。

⚠️ 与 🔴2 的关系：🔴38 把**老鹰/骷髅判为档 C 自由型**（装饰主体，无剪影标识功能），
"整只重绘、结构随便变，只要物种/配色/画面位置对上 + 解剖合理"是**明确许可**的。
本脚本不搬移、不仿射、不叠加原图元素（🔴34）—— 像素全部是新生成的。

体位变化方向（3 个 variant，供用户挑）
---------------------------------------
  V1_wingV   双翼上举收成高窄 V（消灭"横向大展翼"这个最抓眼的同构特征）
  V2_asym    不对称动势：左翼高举 / 右翼低垂前伸，鹰头强侧脸；骷髅倾 12°
  V3_lowbrace 鹰低伏收翅（翼贴体、头压低），骷髅放大并裂到下颌

用法
----
  python src/v409_p6free.py --variant V1_wingV --seeds 888,2024,42
  python src/v409_p6free.py --variant V3_lowbrace --seeds 888 --no-canny  # 极端档
产物：jobs/v409_free/p6_<variant>_s<seed>.jpg（含原图标题带的整图，喂 P6_REBIRTH）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v342_rebirth import (CUI_IN, _disk, _fetch, _lum,          # noqa: E402
                          _match_region, comfy_submit, mask_p6)
from v377_p6rb import snap_p6                                   # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
OUT = ROOT / "jobs" / "v409_free"
OUT.mkdir(parents=True, exist_ok=True)

CKPT = "juggernautXL_ragnarokBy.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA_DETAIL = "add-detail-xl.safetensors"

# ── 第一遍：掩膜内自由重画（近原生分辨率）────────────────────────────────
PASS1_MP = 1.10
PASS1_STEPS = 26
CANNY_S, CANNY_E = 0.10, 0.18     # 只锚"鹰在上/骷髅在下"的宏观排布，全程 ≤18% 步数
# 第二遍（自引导细节遍）
TARGET = 2048
HIRES_DN = 0.42
HIRES_STEPS = 18
HC_S, HC_E = 0.15, 0.50           # 参考图 = 第一遍输出
HT_S, HT_E = 0.45, 0.65
REGION_SCALE = 0.80               # 无 ControlNet 竞争 → 区域提示词可以拿更大权重

MASK_GROW = 20                    # 噪声掩膜外扩（第二遍同样加掩膜，故不再需要大幅外扩）

GLOBAL_POS = (
    "gothic tattoo illustration screen print of a bald eagle with a horned demon skull, "
    "flat vector illustration, bold screen print, hard clean edges, crisp linework, "
    "solid flat colors, NO gradients, NO soft shading, "
    "pure black background, white lightning bolt rays, "
    "white and silver skull, dark chocolate brown wing feathers, "
    "tan ribbed horns, bright golden-yellow beak and claws, fierce eyes, "
    "high contrast, centered composition, "
    "no text, no letters, no words, no banner, no signature anywhere, "
    "cohesive composition, all elements connected and spatially consistent"
)

GLOBAL_NEG = (
    # 🔴35 解剖红线
    "yellow patches, yellow stains, yellow spots on skull, yellow feathers on skull, "
    "yellow nose, yellow nasal cavity, yellow teeth, beak on the skull, second beak, "
    "yellow patch on the eagle head, yellow blotch on the forehead, yellow glow behind the eagle, "
    "yellow halo, pale cream background glow, bright pale background behind the bird, "
    "melted beak, deformed beak, missing eye, extra eye, "
    "dirty bone, dark stains, smudges, blotches, mud, grime, "
    "glowing eyes, luminous eyes, eye light, "
    "extra heads, extra skulls, extra birds, extra wings, extra horns, "
    "deformed, anatomically incorrect, broken bones, missing claws, "
    # 抗融合/抗穿模（v147）
    "fused elements, melted edges, elements touching each other, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "floating disconnected parts, crowded center, cluttered middle area, "
    "no clear black separating outline, bleeding borders, soft halo, "
    # 画风/背景
    "photo, realistic, 3d render, airbrush, painterly, soft gradients, smooth shading, "
    "gray haze, grey haze, fog, mist, smoke, glow, halo, gradient background, "
    "gray background, noise, speckle, dirty background, "
    "text, letters, words, watermark, signature, numbers, "
    "blurry, low quality, out of focus, "
    "pale beak, white beak, desaturated, washed out"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

# ── 区域几何（裁块百分比）。依据本图实测：
#    喙 x[0.41,0.51] y[0.09,0.16]；爪 x[0.40,0.62] y[0.27,0.38]；
#    颅骨 y[0.36,1.00]；双角 y[0.30,0.55]。顺序 = 叠加顺序，守护区放最后（最优先）。
GEO = {
    "eagle":  {"x": 0.04, "y": 0.00, "w": 0.92, "h": 0.38, "strength": 1.15},
    "horns":  {"x": 0.00, "y": 0.28, "w": 1.00, "h": 0.32, "strength": 1.10},
    "skull":  {"x": 0.14, "y": 0.34, "w": 0.72, "h": 0.66, "strength": 1.25},
    "rayL":   {"x": 0.00, "y": 0.30, "w": 0.28, "h": 0.70, "strength": 1.05},
    "rayR":   {"x": 0.72, "y": 0.30, "w": 0.28, "h": 0.70, "strength": 1.05},
    "talons": {"x": 0.34, "y": 0.24, "w": 0.36, "h": 0.18, "strength": 1.45},
    "head":   {"x": 0.30, "y": 0.02, "w": 0.34, "h": 0.22, "strength": 1.60},
}

_HEAD_GUARD = (
    "the bald eagle's HEAD only: a FULLY FORMED white feathered eagle head readable as an eagle head, "
    "both fierce eyes clearly visible on the face, "
    "the bright golden-yellow hooked beak ATTACHED TO THE FRONT OF THE FACE with a clean hooked tip, "
    "crisp dark edge lines on the white head feathers, "
    "NO second beak, NO beak on the skull, NO melted beak, NO missing eye, "
    "NO yellow marks on the head. " + COHESIVE
)
_TALON_GUARD = (
    "the eagle's bright golden-yellow clawed talons gripping the top of the skull, "
    "clearly separated sharp talons per foot, talons resting ON the skull surface and "
    "anchored to the eagle's legs, NO floating yellow blobs, NO yellow marks on the bone. "
    + COHESIVE
)

VARIANTS: dict[str, dict[str, str]] = {
    # ① 双翼上举收成高窄 V —— 正面消灭"横向大展翼 + 侧俯冲头"这两个最抓眼的同构特征
    "V1_wingV": {
        "eagle": (
            "the bald eagle at the top: BOTH WINGS SWEPT UP AND INWARD INTO A TALL NARROW UPRIGHT V, "
            "wingtips pointing straight up above the head, "
            "the wide horizontal wingspan of the reference is COMPLETELY GONE, "
            "the wings now fold upward close to the body as two tall narrow columns of stacked feathers, "
            "long separated primary feathers hanging in vertical rows, "
            "the whole bird reads as a TALL NARROW eagle, not a wide spread eagle, "
            "chest and shoulders pushed forward, one continuous bird with no wing-body separation, "
            "dark chocolate brown wing feathers with crisp dark outlines. " + COHESIVE),
        "head": (
            "the eagle's head held HIGH above the body, three-quarter FRONT angle, "
            "both fierce eyes visible, hooked golden beak pointing forward and slightly down. "
            + _HEAD_GUARD),
        "horns": (
            "the two horns of the skull sweeping BACKWARD and UPWARD in a new tighter curve, "
            "noticeably shorter than before with FEWER visible ribbed segments and thicker heavy bases, "
            "tips pointing up and back, bases attached firmly to the sides of the skull, "
            "sep by clear black space from the lightning bolts, no text on the horns. " + COHESIVE),
        "skull": (
            "the human skull in the lower half turned at a strong THREE-QUARTER angle, "
            "jaw WIDE OPEN in a roar showing a completely different row of teeth, "
            "a new fractured skullcap cracked from the crown down to the brow, "
            "pitch-black empty eye sockets and pitch-black nasal cavity, "
            "clean white bone with even fine cross-hatch shading, NO crown, "
            "NO yellow marks anywhere on the bone. " + COHESIVE),
        "ray": ("white lightning bolt rays radiating outward behind the skull, "
                "fewer and much LONGER jagged bolts with new steep angles, "
                "sharp tapered spikes, solid white flat shapes with clean hard edges, "
                "separated by pure black background. " + COHESIVE),
    },
    # ② 不对称动势 + 强侧脸 + 骷髅倾斜
    "V2_asym": {
        "eagle": (
            "the bald eagle in a strongly ASYMMETRIC dynamic pose: "
            "the LEFT wing raised high upward and folded back over the shoulder, "
            "the RIGHT wing swept low and thrust forward and down, "
            "the body tilted with the chest thrust forward, "
            "head turned to the RIGHT in a strong PROFILE, screaming with a wide open hooked beak, "
            "a single fierce eye visible on the profile face, "
            "dark chocolate brown wing feathers, long separated primaries with crisp outlines, "
            "one continuous bird, no wing-body separation, "
            "clear black gap under the lowered right wing. " + COHESIVE),
        "head": ("the eagle's head in a strong SIDE PROFILE facing right, one fierce eye visible, "
                 "open screaming hooked golden beak pointing right and slightly down. "
                 + _HEAD_GUARD),
        "horns": ("one horn curling UP and inward toward the crown, the other sweeping out and DOWN "
                  "in a longer flatter curve, clearly different curvature left vs right, "
                  "fewer ribbed segments, heavy bold bases, separated from the bolts by black space. "
                  + COHESIVE),
        "skull": ("the human skull TILTED about 12 degrees counter-clockwise, "
                  "not level, jaw hanging open with a full new row of teeth, "
                  "one eye socket larger and cracked at the edge, "
                  "deep new fracture lines across the cheekbone and forehead, "
                  "pitch-black hollow sockets, clean white bone, NO yellow marks on the bone. "
                  + COHESIVE),
        "ray": ("white lightning bolt rays radiating outward behind the skull in a new asymmetric "
                "arrangement, long jagged bolts at new angles, solid white flat shapes with "
                "clean hard edges, separated by pure black background. " + COHESIVE),
    },
    # ③ 鹰低伏收翅 + 骷髅放大
    "V3_lowbrace": {
        "eagle": (
            "the bald eagle CROUCHED LOW with its wings HALF-FOLDED and swept BACK along its flanks "
            "like a bird that has just landed and folded its wings, "
            "the wide open spread wings of the reference are GONE — only folded closed wing shapes "
            "lie along the sides of the body as compact dark brown feather masses, "
            "heavy shoulders, head LOWERED and thrust forward close to the top of the skull, "
            "neck bent, hooked golden beak pointing DOWN and forward, both eyes on the face, "
            "compact powerful low silhouette instead of a wide one. " + COHESIVE),
        "head": ("the eagle's head lowered and thrust forward, three-quarter front angle, "
                 "both eyes visible, hooked golden beak pointing down and forward. "
                 + _HEAD_GUARD),
        "horns": ("the two horns of the skull much THICKER and shorter, curving outward and slightly "
                  "downward in a new heavy curve, bold wide ribbed bands, "
                  "firmly rooted in the sides of the skull, separated from the bolts by black space. "
                  + COHESIVE),
        "skull": ("a much LARGER dominant human skull filling the lower half, "
                  "cracked from the crown all the way down to the jaw with new deep fractures, "
                  "the right eye socket partly shattered, jaw WIDE OPEN in a roar with a new tooth row, "
                  "pitch-black hollow sockets, clean white bone with fine even cross-hatch shading, "
                  "NO yellow marks anywhere on the bone. " + COHESIVE),
        "ray": ("white lightning bolt rays radiating outward behind the skull, "
                "fewer shorter sharper bolts hugging the skull edge at new angles, "
                "solid white flat shapes with clean hard edges, separated by black background. "
                + COHESIVE),
    },
}


def _regions(pose: str):
    v = VARIANTS[pose]
    out = []
    for key in ("eagle", "horns", "skull", "rayL", "rayR", "talons", "head"):
        g = dict(GEO[key])
        g["prompt"] = {
            "eagle": v["eagle"], "horns": v["horns"], "skull": v["skull"],
            "talons": _TALON_GUARD, "head": v["head"],
            "rayL": ("the LEFT half of " + v["ray"] + " no skull, no bird in this area."),
            "rayR": ("the RIGHT half of " + v["ray"] + " no skull, no bird in this area."),
        }[key]
        out.append(g)
    return out


def _add_regions(g: dict, base_pos, region_scale: float, pose: str) -> dict:
    comb = {"global_cond": base_pos}
    for i, r in enumerate(_regions(pose)):
        g[f"rp{i}"] = {"class_type": "CLIPTextEncode",
                       "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        g[f"sa{i}"] = {"class_type": "ConditioningSetAreaPercentage",
                       "inputs": {"conditioning": [f"rp{i}", 0], "width": r["w"],
                                  "height": r["h"], "x": r["x"], "y": r["y"],
                                  "strength": r["strength"] * region_scale}}
        comb[f"region{i+1}"] = [f"sa{i}", 0]
    return comb


def build_wf(crop: Image.Image, mask_png: str, seed: int, pose: str, *,
             pass1_mp=PASS1_MP, steps=PASS1_STEPS, canny_s=CANNY_S, canny_e=CANNY_E,
             tile_s=0.0, tile_e=0.0, region_scale=REGION_SCALE, target=TARGET,
             hires_dn=HIRES_DN, steps2=HIRES_STEPS, hc_s=HC_S, hc_e=HC_E,
             ht_s=HT_S, ht_e=HT_E, lora_w=0.5, **ignored) -> dict:
    ts = str(int(time.time() * 1000))
    name = f"v409_src_{ts}.png"
    crop.save(str(CUI_IN / name))
    cw, ch = crop.size
    sc = min(1.0, target / float(max(cw, ch)))
    tw = max(8, (int(round(cw * sc)) // 8) * 8)
    th = max(8, (int(round(ch * sc)) // 8) * 8)

    g: dict = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoadImage", "inputs": {"image": name}},
        "2m": {"class_type": "LoadImage", "inputs": {"image": mask_png}},
        "2mm": {"class_type": "ImageToMask", "inputs": {"image": ["2m", 0], "channel": "red"}},
        "3": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["2", 0], "upscale_method": "lanczos",
                         "megapixels": pass1_mp, "resolution_steps": 64}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}},
        "4m": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["4", 0], "mask": ["2mm", 0]}},
        # 第二遍的自引导参考：第一遍本身（❌ 不能用原图，否则姿态被拽回去）
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "32": {"class_type": "ImageScale", "inputs": {"image": ["31", 0], "upscale_method": "lanczos",
                                                      "width": tw, "height": th, "crop": "disabled"}},
    }
    g["7"] = {"class_type": "LoraLoader",
              "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": LORA_DETAIL,
                         "strength_model": lora_w, "strength_clip": lora_w}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_NEG}}
    # 第一遍：弱 canny 只锚宏观排布（≤18% 步数），**不挂 Tile**（Tile 会透传原姿态）
    g["20"] = {"class_type": "CannyEdgePreprocessor",
               "inputs": {"image": ["3", 0], "low_threshold": 0.10,
                          "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    pos0, neg0 = ["pg", 0], ["ng", 0]
    if canny_s > 0:
        g["22"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": pos0, "negative": neg0, "control_net": ["21", 0],
                              "image": ["20", 0], "strength": canny_s,
                              "start_percent": 0.0, "end_percent": canny_e}}
        pos0, neg0 = ["22", 0], ["22", 1]
    if tile_s > 0:
        g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
        g["24"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": pos0, "negative": neg0, "control_net": ["23", 0],
                              "image": ["3", 0], "strength": tile_s,
                              "start_percent": 0.0, "end_percent": tile_e}}
        pos0, neg0 = ["24", 0], ["24", 1]
    comb = _add_regions(g, pos0, region_scale, pose)
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb}
    # denoise = 1.0 → 掩膜内从纯噪声重画（这才是"姿态真变"的来源）
    g["10"] = {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
                          "latent_image": ["4m", 0], "seed": seed, "steps": steps,
                          "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": 1.0}}
    # ── 第二遍：放大 + 自引导轻 CN 找回平涂硬边（不会改姿态）──
    # ⚠️ 第二遍**同样必须加噪声掩膜**（第 29 轮踩到）：不加的话 denoise 0.42 会把
    #    整幅裁块（含白射线 / 黑底）重画一遍 → 射线变乱絮、黑底糊出黄晕。
    g["30"] = {"class_type": "LatentUpscale",
               "inputs": {"samples": ["10", 0], "upscale_method": "bilinear",
                          "width": tw, "height": th, "crop": "disabled"}}
    g["30m"] = {"class_type": "SetLatentNoiseMask",
                "inputs": {"samples": ["30", 0], "mask": ["2mm", 0]}}
    g["33"] = {"class_type": "CannyEdgePreprocessor",
               "inputs": {"image": ["32", 0], "low_threshold": 0.10,
                          "high_threshold": 0.25, "resolution": 1024}}
    p2, n2 = ["comb", 0], ["ng", 0]
    if hc_s > 0:
        g["34"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": p2, "negative": n2, "control_net": ["21", 0],
                              "image": ["33", 0], "strength": hc_s,
                              "start_percent": 0.0, "end_percent": hc_e}}
        p2, n2 = ["34", 0], ["34", 1]
    if ht_s > 0:
        g["36"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
        g["37"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": p2, "negative": n2, "control_net": ["36", 0],
                              "image": ["32", 0], "strength": ht_s,
                              "start_percent": 0.0, "end_percent": ht_e}}
        p2, n2 = ["37", 0], ["37", 1]
    g["40"] = {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "positive": p2, "negative": n2,
                          "latent_image": ["30m", 0], "seed": seed + 7, "steps": steps2,
                          "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": hires_dn}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "SaveImage",
               "inputs": {"images": ["12", 0], "filename_prefix": f"v409_{pose}_{seed}"}}
    return g


def _bg_protect(gen: np.ndarray, orig: np.ndarray, black_t=10.0,
                keep_bright=100.0, sat_max=55.0, min_area=300) -> np.ndarray:
    """黑底保护：原图近纯黑处，只放行"成片的、低饱和的亮白"（=新画出的射线），
    其余一律还原原图。

    第 29 轮新增 `sat_max`：自由重画会在主体周边糊出一圈**淡黄/米色的假光晕**
    （模型把黑底当成了浅底）。palette 里只有"白射线"该出现在黑底上，
    所以要求 亮度>100 且 (max-min)<55 才放行 —— 黄色光晕 (R-B 通常 >60) 被拦掉。
    """
    ol = _lum(orig); gl = _lum(gen)
    bg = ol < black_t
    if not bg.any():
        return gen
    sat = gen.max(axis=2) - gen.min(axis=2)
    strong = (gl > keep_bright) & (sat < sat_max)
    lab, n = ndi.label(strong)
    if n:
        sz = np.bincount(lab.ravel())
        big = sz >= min_area
        big[0] = False
        strong = big[lab]
    else:
        strong = np.zeros_like(strong)
    out = gen.copy()
    rest = bg & (~strong)
    out[rest] = orig[rest]
    return out


def _crop_box(mask: np.ndarray, W: int, H: int, margin: int = 90):
    ys, xs = np.where(mask)
    x0 = max(0, xs.min() - margin); x1 = min(W, xs.max() + 1 + margin)
    y0 = max(0, ys.min() - margin); y1 = min(H, ys.max() + 1 + margin)
    x0 -= x0 % 8; y0 -= y0 % 8
    x1 = min(W, x1 + ((8 - (x1 - x0) % 8) % 8))
    y1 = min(H, y1 + ((8 - (y1 - y0) % 8) % 8))
    return x0, y0, x1, y1


TITLE_LINE = 1500


def rebirth_free(img: Image.Image, mask: np.ndarray, seed: int, pose: str, *,
                 margin=90, color_alpha=0.85, verbose=True, **kw) -> Image.Image:
    W, H = img.size
    x0, y0, x1, y1 = _crop_box(mask, W, H, margin)
    crop = img.crop((x0, y0, x1, y1))
    mfull = np.zeros((H, W), bool); mfull[mask] = True
    mcut = mfull[y0:y1, x0:x1]
    # 噪声掩膜外扩 → 新姿态有地方长
    mgrow = ndi.binary_dilation(mcut, _disk(MASK_GROW))
    ts = str(int(time.time() * 1000))
    mp = f"v409_mask_{seed}_{ts}.png"
    Image.fromarray((mgrow * 255).astype(np.uint8), "L").convert("RGB").save(str(CUI_IN / mp))

    wf = build_wf(crop, mp, seed, pose, **kw)
    if verbose:
        print(f"[v409] {pose} crop={crop.size} mask={int(mgrow.sum())} "
              f"pass1={kw.get('pass1_mp', PASS1_MP)}MP dn=1.0 → {kw.get('target', TARGET)}")
    t0 = time.time()
    outs = comfy_submit(wf)
    res = None
    for _nid, o in outs.items():
        if "images" in o:
            res = _fetch(o); break
    if res is None:
        raise RuntimeError("no comfy output")
    res = res.resize(crop.size, Image.LANCZOS)
    if verbose:
        print(f"[v409] comfy seed={seed} {time.time() - t0:.0f}s -> {res.size}")

    oc = np.asarray(crop, np.float32)
    rc = np.asarray(res, np.float32)
    if color_alpha and float(color_alpha) > 0:
        sel = ndi.binary_dilation(mcut, iterations=12)
        if int(sel.sum()) > 50:
            rc = _match_region(rc, oc, sel, alpha=float(color_alpha))
    rc = _bg_protect(rc, oc)
    tb = (y0 + np.arange(crop.height)) < TITLE_LINE
    if tb.any():
        rc[tb, :] = oc[tb, :]
    full = np.array(img).copy()
    full[y0:y1, x0:x1] = np.clip(rc, 0, 255).astype(np.uint8)
    return Image.fromarray(full)


def run(seed: int, pose: str, out_dir=None, tag=None, snap=True, **kw) -> Path:
    out_dir = Path(out_dir) if out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"{pose}_s{seed}"
    src = Image.open(SRC).convert("RGB")
    mask = mask_p6(src)
    t = time.time()
    full = rebirth_free(src, mask, seed, pose, **kw)
    dst = out_dir / f"p6_{tag}.jpg"
    (snap_p6(src, full, mask) if snap else full).save(str(dst), quality=94)
    print(f"[v409] {tag} → {dst.name} ({time.time() - t:.0f}s)", flush=True)
    return dst


def main() -> None:
    global MASK_GROW
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="V1_wingV", choices=sorted(VARIANTS))
    ap.add_argument("--seeds", default="888")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--nosnap", action="store_true")
    ap.add_argument("--no-canny", action="store_true")
    ap.add_argument("--canny", type=float, default=CANNY_S)
    ap.add_argument("--canny-end", type=float, default=CANNY_E)
    ap.add_argument("--pass1-mp", type=float, default=PASS1_MP)
    ap.add_argument("--hires-dn", type=float, default=HIRES_DN)
    ap.add_argument("--region-scale", type=float, default=REGION_SCALE)
    ap.add_argument("--mask-grow", type=int, default=MASK_GROW)
    ap.add_argument("--target", type=int, default=TARGET)
    a = ap.parse_args()
    if a.no_canny:
        a.canny = 0.0
    MASK_GROW = a.mask_grow
    for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
        run(s, a.variant, out_dir=a.out_dir,
            tag=(f"{a.tag}_s{s}" if a.tag else None), snap=not a.nosnap,
            pass1_mp=a.pass1_mp, canny_s=a.canny, canny_e=a.canny_end,
            hires_dn=a.hires_dn, region_scale=a.region_scale, target=a.target)


if __name__ == "__main__":
    main()
