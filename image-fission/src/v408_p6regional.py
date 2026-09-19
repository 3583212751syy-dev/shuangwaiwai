# -*- coding: utf-8 -*-
"""v408_p6regional — p6 鹰骷髅 **v147 逐元素区域控制真裂变**（第 28 轮）

用户第 28 轮原话（配 p6 主体对照截图）：
  「这种自由度高的图片就裂变变化大一点，参考本机 v147 的风格」

背景
----
第 27 轮已把文本按原图排版/形状重排（v407），但主体"拉开"只做到笔触/细节层
（canny 弱引导 cn .20 / end .55 / dn .98 —— 构图骨架仍被锁住，鹰/骷髅姿态不变）。
用户现在给的是**方法级指令**：自由度高的图（p6 属于 🔴38 **档 C 自由型**：
老鹰/骷髅是装饰主体，只要物种/配色/画面位置对上 + 解剖合理，形态随便变）应当
**裂变幅度加大**，且**照 v147 的方法做**。

v147 是什么（本机历史基线，commit `8f8b78b` / `8285e76`）
--------------------------------------------------------
`src/smoke_v146_region_tile.py`（v147 微调版）确立的「逐元素区域控制裂变」：
  · **RegionalListCombine + 每元素 ConditioningSetAreaPercentage** —— 每个元素
    一条独立提示词，显式要求**换朝向/换姿态/换结构**（v147 原话：老鹰"正对镜头
    而不是侧俯冲"、骷髅"3/4 朝向而不是正面"、火焰"左下→右上斜升"、铁链"三段沿
    右边沿"）—— 这是"变化大"的来源，不是靠调 denoise 硬冲。
  · **Tile ControlNet 0.60**（锁纹理/细节，防元素本身糊掉 + 防背景被画成灰雾）
  · **Canny ControlNet 0.25**（只锁"大位置/相对关系"，不焊轮廓）
  · **denoise 0.80**（高噪声 = 真换内容）
  · 每条区域 prompt 拼 `COHESIVE`（"与整体连贯/解剖相连/无漂浮碎块"）+
    全局 NEG 拦"elements touching / fused elements / melted edges"
  · 二次 KSampler denoise 0.20 收细节
后续 v232/v242 沿这条线加码：Canny 0.65 + Tile 0.95（那是"锁死"档，用于蝙蝠
徽章这种 🔴38 档 A~B 标识型）；**p6 是档 C 自由型，取 v147 原档的松引导**。

与既有 p6 通道的关系（🔴26/🔴24）
--------------------------------
· 仍是 ComfyUI 本地 SDXL、仍是**原生 2048 主体裁块**（不降采样不放大）。
· 模型沿用 p6 定稿的 `juggernautXL_ragnarokBy.safetensors`（v393 起用）。
· 产物是"含原图标题带的整图"中间件，喂给 `make_v329` 的 `P6_REBIRTH` 环境变量 →
  标题仍由 v407 的新排版字标贴回（🔴37 主体/文本分两步）。
· 擦字/底板逻辑不动（`do_pinterest6` 里 LaMa 按**原图**标题墨迹掩膜擦，与本文件无关）。

🔴35 解剖验收线（一票否决）：鹰头清晰可辨 + 黄喙长在鹰脸上 + 爪在颅骨上 +
颅骨无黄渍/黄斑。第 27 轮把 canny 压到 .15 + dn 1.0 时这三条会崩，所以本轮
**不靠压引导**换姿态，而靠 v147 的「区域提示词显式换姿态」+ 结构不焊。
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
OUT = ROOT / "jobs" / "v408_regional"
OUT.mkdir(parents=True, exist_ok=True)

# p6 定稿模型（v393 起）
CKPT = "juggernautXL_ragnarokBy.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
LORA_DETAIL = "add-detail-xl.safetensors"

# ── v147 原档（松引导 = 换姿态）─────────────────────────────────────────────
CANNY_S, CANNY_E = 0.25, 0.60      # v147(.25) / 收在 60% 步数 → 后段放开改结构
TILE_S, TILE_E = 0.60, 0.80        # v147 .60
DENOISE = 0.80                     # v147 原档
IPA_W, IPA_E = 0.25, 0.50          # 只在前半步锁色族，后段放开换形（🔴20 的教训）
REGION_SCALE = 0.55                # v147 REGION_STRENGTH_SCALE
MS = 2048                          # 原生 2048（= 既有 p6 通道口径）

GLOBAL_POS = (
    "gothic tattoo illustration screen print of a bald eagle perched on top of a horned demon skull, "
    "flat vector illustration, bold screen print, hard clean edges, crisp linework, "
    "solid flat colors, NO gradients, NO soft shading, "
    "pure black background, white lightning bolts, "
    "white and silver skull, dark chocolate brown wing feathers, "
    "tan ribbed horns, bright golden-yellow beak and claws, fierce eyes, "
    "high contrast, centered composition, "
    "no text, no letters, no words, no banner, no signature anywhere, "
    "cohesive composition, all elements connected and spatially consistent"
)

GLOBAL_NEG = (
    # —— 🔴35 解剖红线 ——
    "yellow patches, yellow stains, yellow spots on skull, yellow feathers on skull, "
    "yellow nose, yellow nasal cavity, yellow teeth, beak on the skull, second beak, "
    "dirty bone, dark stains, smudges, blotches, mud, grime, "
    "glowing eyes, luminous eyes, eye light, "
    "extra heads, extra skulls, extra birds, extra wings, extra horns, "
    "deformed, anatomically incorrect, broken bones, missing claws, "
    # —— v147 抗融合/抗穿模 ——
    "fused elements, melted edges, elements touching each other, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "floating disconnected parts, crowded center, cluttered middle area, "
    "no clear black separating outline, bleeding borders, soft halo, "
    # —— 画风/背景 ——
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

# 区域 = 裁块百分比（裁块 = 主体 bbox + margin；见 _crop_box）
# 几何依据：`jobs/_probe/v408_region_grid.jpg`（3x3 网格目检）
# 顺序即叠加顺序；**小区域后置 + 高 strength** 以在局部压过宽区域。
REGIONS = [
    # ① 老鹰整体：换朝向（正对镜头而非侧俯冲）+ 换翼展/羽层
    {"x": 0.08, "y": 0.00, "w": 0.84, "h": 0.44, "strength": 1.10,
     "prompt": ("the bald eagle at the top of the design: FACING THE CAMERA head-on, "
                "NOT a side profile, NOT a diving pose, head raised high above the skull, "
                "beak pointing forward and slightly to one side at a NEW angle, "
                "both wings spread outward from the body with a WIDER wingspan than before, "
                "wings raised much higher with a new swept-back feather angle, "
                "feather rows rearranged into different overlapping layers and new feather shapes, "
                "wings and body anatomically connected as one continuous bird, no wing-body separation, "
                "dark chocolate brown wing feathers, white feathered head, "
                "CLEAR BLACK GAP between this eagle and any lower element, centered dominant element. "
                + COHESIVE)},
    # ② 鹰头（**解剖守护区**：第 28 轮 E 档实测——放宽引导后鹰头/喙最先崩，
    #    故单列一个高 strength 小区域，把"喙必须长在脸上、无多余喙、眼可见"写死）
    {"x": 0.30, "y": 0.00, "w": 0.32, "h": 0.24, "strength": 1.55,
     "prompt": ("the bald eagle's HEAD only, a FULLY FORMED white feathered eagle head seen from a "
                "three-quarter FRONT angle, both fierce eyes visible on the face, "
                "the bright golden-yellow hooked beak attached to the FRONT of the face and "
                "pointing forward and slightly down, beak fully intact with a clean hooked tip, "
                "clean white head feathers with crisp dark edge lines, "
                "NO second beak, NO beak on the skull, NO melted beak, NO missing eye, "
                "NO yellow marks on the head, head clearly readable as an eagle head. "
                + COHESIVE)},
    # ③ 鹰爪（🔴35：爪必须落在颅骨上，不许悬空黄块）
    {"x": 0.30, "y": 0.24, "w": 0.40, "h": 0.14, "strength": 1.45,
     "prompt": ("the eagle's bright golden-yellow clawed talons gripping the top of the skull, "
                "five clearly separated sharp talons per foot, talons resting ON the skull surface, "
                "talons anchored to the eagle's legs, NO floating yellow blobs, "
                "NO yellow marks on the bone. "
                + COHESIVE)},
    # ④ 双角：换弯曲角度/换分节节奏（整只角，别只角尖）
    {"x": 0.00, "y": 0.12, "w": 1.00, "h": 0.46, "strength": 1.05,
     "prompt": ("the two big horns of the skull: the LEFT horn sweeping out to the left and the "
                "RIGHT horn sweeping out to the right, both horns CURVED AT A NEW ANGLE with a "
                "different overall curvature and a DIFFERENT number of visible ribbed band segments, "
                "thicker heavier horns with bold solid tan and bone-white ribbed bands, "
                "tips pointing outward and upward at new directions, "
                "horn bases attached firmly to the sides of the skull, "
                "each horn SEPARATED from the lightning bolts by clear black space, "
                "no decoration, no patterns, no text on the horns. "
                + COHESIVE)},
    # ⑤ 骷髅：换朝向（3/4 而非正面）+ 换牙列/下颌/裂纹
    {"x": 0.20, "y": 0.36, "w": 0.60, "h": 0.60, "strength": 1.30,
     "prompt": ("the human skull in the lower half: turned at a THREE-QUARTER angle, "
                "NOT a straight-on front view, tilted at a NEW angle, "
                "deep realistic cracks and fractures redesigned across the bone surface, "
                "a DIFFERENT jaw shape and a DIFFERENT row of teeth, "
                "different eye socket shapes, empty hollow pitch-black eye sockets and "
                "pitch-black nasal cavity, clean white bone with even fine cross-hatch shading, "
                "NO crown, NO hat, just bone with a cracked skullcap, "
                "NO yellow marks anywhere on the bone, "
                "CLEAR BLACK GAP between this skull and the horns. "
                + COHESIVE)},
    # ⑥ 左侧白色射线：换排列/换角度
    {"x": 0.00, "y": 0.35, "w": 0.30, "h": 0.65, "strength": 1.05,
     "prompt": ("white lightning bolt rays on the LEFT side radiating outward, "
                "a DIFFERENT arrangement of jagged bolts with new angles and new lengths, "
                "sharp tapered spikes, solid white flat shapes with clean hard edges, "
                "separated from each other by pure black background, "
                "no skull, no bird in this area. "
                + COHESIVE)},
    # ⑦ 右侧白色射线
    {"x": 0.70, "y": 0.35, "w": 0.30, "h": 0.65, "strength": 1.05,
     "prompt": ("white lightning bolt rays on the RIGHT side radiating outward, "
                "a DIFFERENT arrangement of jagged bolts with new angles and new lengths, "
                "sharp tapered spikes, solid white flat shapes with clean hard edges, "
                "separated from each other by pure black background, "
                "no skull, no bird in this area. "
                + COHESIVE)},
]

# 标题带保护（原图标题白墨顶 y88；主体翼尖自 y≈1560 起）—— 此线以上禁止改动
TITLE_LINE = 1500


def _crop_box(img: Image.Image, mask: np.ndarray, margin: int = 90):
    ys, xs = np.where(mask)
    W, H = img.size
    x0 = max(0, xs.min() - margin); x1 = min(W, xs.max() + 1 + margin)
    y0 = max(0, ys.min() - margin); y1 = min(H, ys.max() + 1 + margin)
    # 8 的倍数（VAE 友好）
    x0 -= x0 % 8; y0 -= y0 % 8
    x1 = min(W, x1 + ((8 - (x1 - x0) % 8) % 8))
    y1 = min(H, y1 + ((8 - (y1 - y0) % 8) % 8))
    return x0, y0, x1, y1


def _add_regions(g: dict, base_pos, base_neg, region_scale: float) -> dict:
    comb = {"global_cond": base_pos}
    for i, r in enumerate(REGIONS):
        g[f"rp{i}"] = {"class_type": "CLIPTextEncode",
                       "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        g[f"sa{i}"] = {"class_type": "ConditioningSetAreaPercentage",
                       "inputs": {"conditioning": [f"rp{i}", 0], "width": r["w"],
                                  "height": r["h"], "x": r["x"], "y": r["y"],
                                  "strength": r["strength"] * region_scale}}
        comb[f"region{i+1}"] = [f"sa{i}", 0]
    return comb


def build_wf_hires(crop: Image.Image, seed: int, *, canny_s=CANNY_S, canny_e=CANNY_E,
                   tile_s=TILE_S, tile_e=TILE_E, ipa_w=IPA_W, ipa_e=IPA_E,
                   region_scale=REGION_SCALE, steps=26, refine=0.20,
                   pass1_mp=1.10, pass1_dn=1.0, hires_dn=0.45,
                   hc_s=0.12, hc_e=0.50, ht_s=0.45, ht_e=0.65,
                   target=2048, steps2=18, **ignored) -> dict:
    """v147 区域控制 + **hires 两遍**（本轮新增）。

    为什么需要两遍：SDXL 原生约 1MP。把 3312×3561 的裁块直接降到 2048 长边（3.9MP）
    等于让模型在 4 倍于原生分辨率的画布上作画 —— 此时两股 ControlNet（尤其 Tile）
    的"保外观"作用被放大到压倒提示词，实测鹰头姿态/颅骨朝向纹丝不动。
    ① 第一遍在 **~1.1MP 近原生**画布上跑区域提示词 → 模型有真自由度去换朝向/换姿态；
    ② 把 latent 放大到目标分辨率，再用同一套区域 conditioning + 轻 Tile/Canny 走一遍
       **denoise≈0.45** 的细节遍 → 把平涂硬边找回来（🔴22：成品不得比原图糊）。
    """
    ts = str(int(time.time() * 1000))
    name = f"v408h_src_{ts}.png"
    crop.save(str(CUI_IN / name))
    cw, ch = crop.size
    sc = min(1.0, (target / float(max(cw, ch))))
    tw = max(8, (int(round(cw * sc)) // 8) * 8)
    th = max(8, (int(round(ch * sc)) // 8) * 8)
    g: dict = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoadImage", "inputs": {"image": name}},
        "3": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["2", 0], "upscale_method": "lanczos",
                         "megapixels": pass1_mp, "resolution_steps": 64}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}},
                # ⚠️ 第二遍的参考图必须来自**第一遍的输出**（自引导），
                # 不能拿原图（否则 Tile 会把姿态拽回原图，换姿态白做）。
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "32": {"class_type": "ImageScale",
               "inputs": {"image": ["31", 0], "upscale_method": "lanczos",
                          "width": tw, "height": th, "crop": "disabled"}},
    }
    mref = ["1", 0]
    if ipa_w > 0:
        g["5"] = {"class_type": "IPAdapterUnifiedLoader",
                  "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
        g["6"] = {"class_type": "IPAdapterAdvanced",
                  "inputs": {"model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
                             "weight": ipa_w, "weight_type": "style transfer",
                             "combine_embeds": "average", "start_at": 0.0,
                             "end_at": ipa_e, "noise": 0.05, "embeds_scaling": "V only"}}
        mref = ["6", 0]
    g["7"] = {"class_type": "LoraLoader",
              "inputs": {"model": mref, "clip": ["1", 1], "lora_name": LORA_DETAIL,
                         "strength_model": 0.50, "strength_clip": 0.50}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_NEG}}
    # ── 第一遍（近原生分辨率）：引导只做"大位置"锚定，姿态交给区域提示词 ──
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
    comb = _add_regions(g, pos0, neg0, region_scale)
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb}
    g["10"] = {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
                          "latent_image": ["4", 0], "seed": seed, "steps": steps,
                          "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": pass1_dn}}
    # ── 放大 + 第二遍（细节遍）：同一套区域 conditioning + 轻 CN 找回平涂硬边 ──
    g["30"] = {"class_type": "LatentUpscale",
               "inputs": {"samples": ["10", 0], "upscale_method": "bilinear",
                          "width": tw, "height": th, "crop": "disabled"}}
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
                          "latent_image": ["30", 0], "seed": seed + 7, "steps": steps2,
                          "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": hires_dn}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "SaveImage",
               "inputs": {"images": ["12", 0], "filename_prefix": f"v408h_{seed}"}}
    return g


def build_wf(crop: Image.Image, seed: int, *, canny_s=CANNY_S, canny_e=CANNY_E,
             tile_s=TILE_S, tile_e=TILE_E, denoise=DENOISE, ipa_w=IPA_W, ipa_e=IPA_E,
             region_scale=REGION_SCALE, steps=30, refine=0.20, **ignored) -> dict:
    ts = str(int(time.time() * 1000))
    name = f"v408_src_{ts}.png"
    crop.save(str(CUI_IN / name))
    g: dict = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoadImage", "inputs": {"image": name}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
    }
    # 模型链：ckpt → IPAdapter(style 锁色族) → add-detail LoRA
    mref = ["1", 0]
    if ipa_w > 0:
        g["5"] = {"class_type": "IPAdapterUnifiedLoader",
                  "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
        g["6"] = {"class_type": "IPAdapterAdvanced",
                  "inputs": {"model": ["1", 0], "ipadapter": ["5", 1], "image": ["2", 0],
                             "weight": ipa_w, "weight_type": "style transfer",
                             "combine_embeds": "average", "start_at": 0.0,
                             "end_at": ipa_e, "noise": 0.05, "embeds_scaling": "V only"}}
        mref = ["6", 0]
    g["7"] = {"class_type": "LoraLoader",
              "inputs": {"model": mref, "clip": ["1", 1], "lora_name": LORA_DETAIL,
                         "strength_model": 0.50, "strength_clip": 0.50}}
    # 全局正负向
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_NEG}}
    # v147 空间一致性双 ControlNet：Canny(只锁大位置) → Tile(锁纹理/背景)
    # 两者 strength=0 时整段不挂（"松到底"档：只靠区域提示词换姿态 + 解剖靠 prompt 保）
    g["20"] = {"class_type": "CannyEdgePreprocessor",
               "inputs": {"image": ["2", 0], "low_threshold": 0.10,
                          "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    if canny_s > 0:
        g["22"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": ["pg", 0], "negative": ["ng", 0],
                              "control_net": ["21", 0], "image": ["20", 0],
                              "strength": canny_s, "start_percent": 0.0,
                              "end_percent": canny_e}}
        pos0, neg0 = ["22", 0], ["22", 1]
    else:
        pos0, neg0 = ["pg", 0], ["ng", 0]
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    if tile_s > 0:
        g["24"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": pos0, "negative": neg0,
                              "control_net": ["23", 0], "image": ["2", 0],
                              "strength": tile_s, "start_percent": 0.0,
                              "end_percent": tile_e}}
        pos0, neg0 = ["24", 0], ["24", 1]
    # 逐元素区域提示词（v147 核心）
    comb = _add_regions(g, pos0, neg0, region_scale)
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb}
    g["10"] = {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
                          "latent_image": ["4", 0], "seed": seed, "steps": steps,
                          "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": denoise}}
    g["11"] = {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
                          "latent_image": ["10", 0], "seed": seed + 1, "steps": 20,
                          "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": float(refine)}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11" if float(refine) > 0 else "10", 0],
                                                      "vae": ["1", 2]}}
    g["13"] = {"class_type": "SaveImage",
               "inputs": {"images": ["12", 0], "filename_prefix": f"v408_{seed}"}}
    return g


def _bg_protect(gen: np.ndarray, orig: np.ndarray, black_t=10.0,
                keep_bright=110.0, min_area=300) -> np.ndarray:
    """黑底保护：原图为近纯黑的像素，除非新图在此画出**成片强亮**的白射线，
    否则一律还原原图 → 既杀掉"新内容在黑底留灰雾/灰块"（p6 历史顽疾），
    又允许新射线的形态真的变化。"""
    ol = _lum(orig); gl = _lum(gen)
    bg = ol < black_t
    if not bg.any():
        return gen
    strong = gl > keep_bright
    lab, n = ndi.label(strong)
    if n:
        sz = np.bincount(lab.ravel())
        big = sz >= min_area
        big[0] = False
        strong = big[lab]
    else:
        strong = np.zeros_like(strong)
    restore = bg & (~strong)
    out = gen.copy()
    out[restore] = orig[restore]
    return out


def rebirth_regional(img: Image.Image, mask: np.ndarray, seed: int, *,
                     margin=90, max_side=MS, color_alpha=0.85, verbose=True,
                     hires=False, **kw) -> Image.Image:
    """整幅 p6：主体区按 v147 逐元素区域控制重画，其余像素零改动。

    hires=False → 单遍（v147 原档，裁块降到 max_side 直接重画）
    hires=True  → 两遍（近原生分辨率自由换姿态 + 放大后细节遍）
    """
    W, H = img.size
    x0, y0, x1, y1 = _crop_box(img, mask, margin)
    crop = img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    mfull = np.zeros((H, W), bool); mfull[mask] = True
    mcut = mfull[y0:y1, x0:x1]

    if hires:
        # 两遍模式自己管分辨率：喂整块裁块，让第一遍落在近原生 mp 上
        inj = crop
        kw.setdefault("target", max_side)
        wf = build_wf_hires(inj, seed, **kw)
        if verbose:
            print(f"[v408h] crop {cw}x{ch}  第一遍 {kw.get('pass1_mp')}MP"
                  f"  第二遍 target={kw.get('target')} dn={kw.get('hires_dn')}")
    else:
        # 原大小超 max_side → 降采样重绘（原生 2048 口径）
        if max_side and max(cw, ch) > max_side:
            sc = max_side / float(max(cw, ch))
            iw = max(8, (int(round(cw * sc)) // 8) * 8)
            ih = max(8, (int(round(ch * sc)) // 8) * 8)
            inj = crop.resize((iw, ih), Image.LANCZOS)
            if verbose:
                print(f"[v408] inj downscale {cw}x{ch} -> {iw}x{ih}")
        else:
            inj = crop
        wf = build_wf(inj, seed, **kw)
    t0 = time.time()
    outs = comfy_submit(wf)
    res = None
    for _nid, o in outs.items():
        if "images" in o:
            res = _fetch(o); break
    if res is None:
        raise RuntimeError("no comfy output")
    res = res.resize((cw, ch), Image.LANCZOS)
    if verbose:
        print(f"[v408] comfy seed={seed} {(time.time() - t0):.0f}s -> {res.size}")

    oc = np.asarray(crop, np.float32)
    rc = np.asarray(res, np.float32)
    # ① 配色锁（只对主体剪影区，Reinhard LAB）→ 保色族
    if color_alpha and float(color_alpha) > 0:
        sel = ndi.binary_dilation(mcut, iterations=12)
        if int(sel.sum()) > 50:
            rc = _match_region(rc, oc, sel, alpha=float(color_alpha))
    # ② 黑底保护（杀灰雾；放行成片强亮的新射线）
    rc = _bg_protect(rc, oc)
    # ③ 标题带保护：TITLE_LINE 以上一律还原原图（新字标由 v407 通道贴回）
    tb = (y0 + np.arange(ch)) < TITLE_LINE
    if tb.any():
        rc[tb, :] = oc[tb, :]
    comp = Image.fromarray(np.clip(rc, 0, 255).astype(np.uint8), "RGB")
    full = np.array(img).copy()
    full[y0:y1, x0:x1] = np.asarray(comp)
    return Image.fromarray(full)


def run(seed: int, out_dir=None, tag=None, snap=True, **kw) -> Path:
    out_dir = Path(out_dir) if out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"reg_s{seed}"
    src = Image.open(SRC).convert("RGB")
    mask = mask_p6(src)
    t = time.time()
    full = rebirth_regional(src, mask, seed, **kw)
    dst = out_dir / f"p6_{tag}.jpg"
    if snap:
        snap_p6(src, full, mask).save(str(dst), quality=94)
    else:
        full.save(str(dst), quality=94)
    print(f"[v408] seed={seed} → {dst.name} ({time.time() - t:.0f}s)", flush=True)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="888")
    ap.add_argument("--ms", type=int, default=MS)
    ap.add_argument("--denoise", type=float, default=DENOISE)
    ap.add_argument("--canny", type=float, default=CANNY_S)
    ap.add_argument("--canny-end", type=float, default=CANNY_E)
    ap.add_argument("--tile", type=float, default=TILE_S)
    ap.add_argument("--ipa", type=float, default=IPA_W)
    ap.add_argument("--region-scale", type=float, default=REGION_SCALE)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--refine", type=float, default=0.20)
    ap.add_argument("--tile-end", type=float, default=TILE_E)
    ap.add_argument("--hires", action="store_true", help="两遍：近原生换姿态 + 放大细节遍")
    ap.add_argument("--pass1-mp", type=float, default=1.10)
    ap.add_argument("--pass1-dn", type=float, default=1.0)
    ap.add_argument("--hires-dn", type=float, default=0.45)
    ap.add_argument("--hc", type=float, default=0.12, help="细节遍 canny 强度")
    ap.add_argument("--ht", type=float, default=0.45, help="细节遍 tile 强度")
    ap.add_argument("--steps2", type=int, default=18)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--nosnap", action="store_true")
    a = ap.parse_args()
    for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
        run(s, out_dir=a.out_dir, tag=a.tag, snap=not a.nosnap, max_side=a.ms,
            denoise=a.denoise, canny_s=a.canny, canny_e=a.canny_end, tile_s=a.tile,
            tile_e=a.tile_end, ipa_w=a.ipa, region_scale=a.region_scale,
            steps=a.steps, refine=a.refine, hires=a.hires, pass1_mp=a.pass1_mp,
            pass1_dn=a.pass1_dn, hires_dn=a.hires_dn, hc_s=a.hc, ht_s=a.ht,
            steps2=a.steps2)


if __name__ == "__main__":
    main()
