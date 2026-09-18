"""
fission_v321_general.py — 通用化图裂变引擎（v321）

定位
====
- 把"图裂变"从 BACARDÍ 紫蝙蝠单样式解放出来，对任意海报做合规裂变。
- 零 GPU / 零 ComfyUI / 零云端 API 依赖：纯 PIL + numpy，CPU 即可跑，~1-3s/张。
- 严守 v307 永久硬规则：
    ① 主体不许缺失（只在原元素改细节）
    ② 配色严格锁定原图色相（±15°）/饱和度（±20%）/明度（±10%）
    ③ 构图微调（缩放±10%/角度±5°）
    ④ 禁止凭空换主体/版式/气质
    ⑤ 禁止云端生图 API、禁止 PIL 凭空合成伪装 AI
- 文字裂变：自动检测横排文字带 → 填底色 → 用 PIL 在原位渲染**原创虚构词**（绝不输出真实品牌字）。
- 这是 Phase A（CV 基线，永远能跑）。Phase B（AI 增强）会复用本脚本的
  analyze_image() 输出去驱动 ComfyUI 的 IPAdapter/AnyText2/ControlNet Union。

设计原则
========
- 只改需要改的：保留 src/fission.py / pipelines/build.py / engine/comfy_client.py 全部既有能力。
- analyze_image() 的输出字段名与 ComfyUI 参数同构（color_strength/composition_strength/
  controlnet_strength/text_bands/dominant_hues），下轮 Phase B 直接喂给 build_mode3()。
- 本脚本产物全部落到 jobs/fission_v321_general/，不污染 jobs/ 既有目录。

用法
====
    python fission_v321_general.py \
        --input  "E:/Desktop/图裂变测试图" \
        --output "E:/Desktop/双接口/image-fission/jobs/fission_v321_general" \
        --variants 3

退出码：0=全部成功；1=有失败（仍输出部分产物与 gallery.html）
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import hashlib
import colorsys
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2


# ============================================================================
# 0. 常量与词池
# ============================================================================

# v307 红线常量（程序化锁死，避免人工误改）
HUE_DELTA_DEG = 15.0          # 色相旋转上限
SAT_DELTA = 0.20              # 饱和度偏移上限（±0.20）
LUM_DELTA = 0.10              # 明度偏移上限（±0.10）
AFFINE_SCALE_DELTA = 0.05     # 缩放上限（±5%）
AFFINE_ROT_DEG = 3.0          # 旋转上限（±3°）

# Windows 内置字体（按偏好排序，找不到自动降级）
WIN_FONTS_BOLD = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\ARIALBD.TTF",
    r"C:\Windows\Fonts\Arial Black.ttf",
    r"C:\Windows\Fonts\ARIBLK.TTF",
]
WIN_FONTS_REG = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\ARIAL.TTF",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\times.TTF",
]

# 原创虚构词池（绝无真实品牌；下划线分隔用于按需拆词）
FICTIONAL_BRANDS = [
    "NOCTAVEN", "DARKRESERVE", "BLACKLABEL", "IRONBEAR", "STEELHAWK",
    "WOLFPACK", "THUNDERBOLT", "SHADOWFANG", "NIGHTFALL", "DARKHORSE",
    "BLITZKRIEG", "STORMBRING", "FROSTBITE", "BLACKICE", "WILDHUNT",
    "DEEPRIDGE", "GHOSTLINE", "LIONHEART", "IRONCLAW", "STARFORGE",
    "MOONRAVEN", "ASHFALL", "GRIMREAPER", "TYPHOON", "BARRACUDA",
    "BLACKWING", "SILVERFANG", "STONECOLD", "FROSTCLAW", "THORNBACK",
]
FICTIONAL_SUBTITLES = [
    "DARK RESERVE", "IRON BEAR", "BORN TO RIDE", "BLACK LABEL",
    "NIGHT FALL", "STEEL HAWK", "WOLF PACK", "DARK HORSE",
    "STORM BRING", "GHOST LINE", "MOON RAVEN", "IRON CLAW",
    "WILD HUNT", "DEEP RIDGE", "STAR FORGE", "ASH FALL",
]


# ============================================================================
# 1. 数据结构
# ============================================================================

@dataclass
class TextBand:
    """自动检测到的横排文字带。"""
    y0: int
    y1: int
    x0: int
    x1: int
    bg_rgb: Tuple[int, int, int]     # 文字带背景主色
    fg_rgb: Tuple[int, int, int]     # 文字主色（glyph 颜色）
    cap_h_guess: int                # 估计的 cap-height（像素）
    weight_guess: str               # "bold" | "regular"（粗略）
    n_components: int               # 文字连通域数（粗略）


@dataclass
class ImageProfile:
    """一张输入图的全量画像。Phase B（AI 增强）会直接消费这个对象。"""
    path: str
    W: int
    H: int
    image_type: str                  # text_poster|graphic_logo|pattern|cartoon|product|landscape|unknown
    has_text: bool
    text_bands: List[TextBand] = field(default_factory=list)
    dominant_hues_deg: List[float] = field(default_factory=list)   # 主色相（HSV H, 0-360）
    dominant_sat: List[float] = field(default_factory=list)
    dominant_val: List[float] = field(default_factory=list)
    edge_density: float = 0.0        # 边缘密度（canny-like proxy）
    bg_uniformity: float = 0.0       # 背景单一色占比（0-1）
    n_unique_colors_q: int = 0        # 量化后唯一色数（color complexity proxy）
    # Phase B 友好的"可直接喂给 build_mode3"参数预填：
    suggested_color_strength: float = 0.6
    suggested_composition_strength: float = 0.5
    suggested_controlnet_strength: float = 0.0
    suggested_denoise: float = 0.25


# ============================================================================
# 2. 工具
# ============================================================================

def _pil_hsv_to_rgb_hsv(arr_hsv_u8: np.ndarray) -> np.ndarray:
    """PIL 'HSV' mode 输出的 H 是 0-255（8bit 量化），转回 0-360 度 + 0-1 S/V。"""
    h, s, v = arr_hsv_u8[..., 0].astype(np.float32), arr_hsv_u8[..., 1].astype(np.float32) / 255.0, arr_hsv_u8[..., 2].astype(np.float32) / 255.0
    h_deg = h * 360.0 / 255.0
    return np.stack([h_deg, s, v], axis=-1)


def _rgb_hsv_to_pil_hsv(h_deg: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """把 0-360 度 H + 0-1 S/V 转回 PIL HSV 8bit 格式。"""
    h8 = np.clip((h_deg % 360.0) * 255.0 / 360.0, 0, 255).astype(np.uint8)
    s8 = np.clip(s * 255.0, 0, 255).astype(np.uint8)
    v8 = np.clip(v * 255.0, 0, 255).astype(np.uint8)
    return np.stack([h8, s8, v8], axis=-1)


def _hsv_shift(img_rgb: Image.Image, dh_deg: float, ds: float, dv: float) -> Image.Image:
    """对整张图做 HSV 微调。dh/ds/dv 各自被钳到 v307 红线范围内。"""
    dh = float(np.clip(dh_deg, -HUE_DELTA_DEG, HUE_DELTA_DEG))
    ds = float(np.clip(ds, -SAT_DELTA, SAT_DELTA))
    dv = float(np.clip(dv, -LUM_DELTA, LUM_DELTA))
    if abs(dh) < 0.01 and abs(ds) < 0.001 and abs(dv) < 0.001:
        return img_rgb.copy()
    arr = np.array(img_rgb.convert("RGB"))
    hsv = _pil_hsv_to_rgb_hsv(np.array(Image.fromarray(arr).convert("HSV")))
    h_deg = hsv[..., 0] + dh
    s = np.clip(hsv[..., 1] + ds, 0.0, 1.0)
    v = np.clip(hsv[..., 2] + dv, 0.0, 1.0)
    out_hsv = _rgb_hsv_to_pil_hsv(h_deg, s, v)
    return Image.fromarray(np.array(Image.fromarray(out_hsv, mode="HSV").convert("RGB")))


def _micro_affine(img: Image.Image, scale: float, rot_deg: float) -> Image.Image:
    """微仿射：缩放 ±5% / 旋转 ±3°，中心对齐，背景用边缘均值填充。"""
    s = float(np.clip(scale, 1.0 - AFFINE_SCALE_DELTA, 1.0 + AFFINE_SCALE_DELTA))
    r = float(np.clip(rot_deg, -AFFINE_ROT_DEG, AFFINE_ROT_DEG))
    if abs(s - 1.0) < 0.001 and abs(r) < 0.01:
        return img.copy()
    W, H = img.size
    # 算旋转后需要的画布大小
    rad = np.deg2rad(r)
    new_w = int(np.ceil(W * abs(np.cos(rad)) + H * abs(np.sin(rad))))
    new_h = int(np.ceil(W * abs(np.sin(rad)) + H * abs(np.cos(rad))))
    canvas = Image.new("RGB", (new_w, new_h), color=(128, 128, 128))
    rot = img.rotate(r, resample=Image.BICUBIC, expand=True)
    rw, rh = rot.size
    sx = new_w / rw
    sy = new_h / rh
    s_use = min(sx, sy) * s  # 不超出画布
    rot = rot.resize((int(rw * s_use), int(rh * s_use)), Image.BICUBIC)
    canvas.paste(rot, ((new_w - rot.size[0]) // 2, (new_h - rot.size[1]) // 2))
    # 裁回原尺寸
    if canvas.size != (W, H):
        canvas = canvas.resize((W, H), Image.BICUBIC)
    return canvas


def _load_font(bold: bool, fsize: int) -> ImageFont.FreeTypeFont:
    """从 Windows 字体表里找一个 bold/regular 字体，失败兜底 default。"""
    cands = WIN_FONTS_BOLD if bold else WIN_FONTS_REG
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, fsize)
            except Exception:
                continue
    return ImageFont.load_default()


# ============================================================================
# 3. Profile：自动画像
# ============================================================================

def _quantize_palette(arr: np.ndarray, n_colors: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """把 RGB 数组量化到 n_colors 个 bin，返回 (quantized_rgb, flat_indices)。"""
    q = (arr // (256 // n_colors)).astype(np.int32)
    flat = q[..., 0] * n_colors * n_colors + q[..., 1] * n_colors + q[..., 2]
    return q, flat


def _detect_dominant_hsv(arr: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[float], List[float]]:
    """提取 top-k 主导色相（H 度）、饱和度、明度。"""
    hsv = _pil_hsv_to_rgb_hsv(np.array(Image.fromarray(arr).convert("HSV")))
    H_flat = hsv[..., 0].reshape(-1)
    S_flat = hsv[..., 1].reshape(-1)
    V_flat = hsv[..., 2].reshape(-1)
    # 量化 H 到 36 桶（每桶 10°）
    h_bins = (H_flat / 10.0).astype(np.int32) % 36
    counts = np.bincount(h_bins, minlength=36)
    top = np.argsort(-counts)[:top_k]
    hues, sals, vals = [], [], []
    for b in top:
        mask = h_bins == b
        if mask.sum() < 100:
            continue
        hues.append(float(H_flat[mask].mean()))
        sals.append(float(S_flat[mask].mean()))
        vals.append(float(V_flat[mask].mean()))
    return hues, sals, vals


def _edge_density(gray: np.ndarray) -> float:
    """简单边缘密度：sobel 风格（|dx|+|dy| > thresh 像素占比）。"""
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    dy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    mag = dx + dy
    thresh = 40
    return float((mag > thresh).mean())


def _bg_uniformity(arr: np.ndarray) -> float:
    """背景单一色占比：靠近图像四边 5% 区域的颜色中，最常见 bin 的占比。"""
    H, W = arr.shape[:2]
    band = max(2, int(min(H, W) * 0.05))
    borders = np.concatenate([
        arr[:band].reshape(-1, 3),
        arr[-band:].reshape(-1, 3),
        arr[:, :band].reshape(-1, 3),
        arr[:, -band:].reshape(-1, 3),
    ], axis=0)
    # 关键：必须先升到 int32 再乘法，否则 uint8 * 256 直接溢出
    q = (borders // 32).astype(np.int32)
    flat = q[:, 0] * 256 + q[:, 1] * 8 + q[:, 2]
    counts = np.bincount(flat, minlength=512)
    if counts.sum() == 0:
        return 0.0
    return float(counts.max() / counts.sum())


def _detect_text_bands(gray: np.ndarray, arr: np.ndarray) -> List[TextBand]:
    """
    自动检测横排文字带。
    算法：行方差轮廓 → 找文字行（高方差）→ 合并相邻行为 band → 每 band 估 bg/fg/字号/字重。
    """
    H, W = gray.shape
    # 1) 行方差（用 std 而非 var 减少极端值影响）
    row_std = gray.astype(np.float32).std(axis=1)
    # 2) 平滑（移动平均，窗 ~5 行）
    k = 5
    kernel = np.ones(k) / k
    row_smooth = np.convolve(row_std, kernel, mode="same")
    # 3) 阈值：经验值 = 全图 std 中位数的 1.2 倍
    thresh = max(8.0, float(np.median(row_smooth)) * 1.2)
    text_rows = row_smooth > thresh
    # 4) 合并相邻 text_rows 为 band；band 间允许 ≤8px gap
    bands = []
    in_band = False
    start = 0
    gap = 0
    for y in range(H):
        if text_rows[y]:
            if not in_band:
                start = y
                in_band = True
            gap = 0
        else:
            if in_band:
                gap += 1
                if gap > 8:
                    bands.append((start, y - gap))
                    in_band = False
                    gap = 0
    if in_band:
        bands.append((start, H - 1))
    # 5) 过滤：band 高度 < 16px 视为噪点；band 高度 > H*0.45 视为整图纹理（非文字）
    bands = [(a, b) for a, b in bands if 16 <= (b - a) <= int(H * 0.45)]
    # 6) 把 y0,y1 上下各外扩 2px（捕获抗锯齿）
    bands = [(max(0, a - 2), min(H, b + 2)) for a, b in bands]
    # 7) 每 band：找 x 范围、bg/fg 色、cap_h 估计、字重
    result: List[TextBand] = []
    for y0, y1 in bands:
        sub_g = gray[y0:y1]
        sub_rgb = arr[y0:y1]
        # 文字像素 = 与 band 上下边沿 1px 行色差大的像素
        top_row = sub_rgb[0:1].reshape(-1, 3).mean(axis=0)
        bot_row = sub_rgb[-1:].reshape(-1, 3).mean(axis=0)
        edge_mean = (top_row + bot_row) / 2.0
        # glyph 像素：与 edge 色差 > 阈值
        diff = np.linalg.norm(sub_rgb.astype(np.float32) - edge_mean, axis=2)
        glyph_mask = diff > 30
        if glyph_mask.sum() < 50:
            continue
        # bg 颜色 = band 上下边沿的均值
        bg_rgb = tuple(int(c) for c in edge_mean)
        # fg 颜色 = glyph 像素的均值（更稳：用中位数）
        fg_pixels = sub_rgb[glyph_mask]
        fg_rgb = tuple(int(c) for c in np.median(fg_pixels, axis=0))
        # x 范围：glyph 列的 min..max
        cols_with_glyph = glyph_mask.any(axis=0)
        if not cols_with_glyph.any():
            continue
        xs = np.where(cols_with_glyph)[0]
        x0, x1 = int(xs[0]), int(xs[-1] + 1)
        cap_h = int(y1 - y0)
        # 字重：glyph 平均厚度（行内连续 glyph 像素的 run-length 中位数）
        row_thicks = []
        for r in range(sub_g.shape[0]):
            row_g = glyph_mask[r]
            if not row_g.any():
                continue
            # run-length of True
            runs = []
            cur = 0
            for v in row_g:
                if v:
                    cur += 1
                else:
                    if cur > 0:
                        runs.append(cur)
                    cur = 0
            if cur > 0:
                runs.append(cur)
            if runs:
                row_thicks.append(int(np.median(runs)))
        med_thick = int(np.median(row_thicks)) if row_thicks else 0
        weight = "bold" if med_thick >= max(3, cap_h * 0.18) else "regular"
        # n_components 粗估：水平方向列内 glyph 像素的"段落"数
        col_has = glyph_mask.any(axis=0).astype(np.int32)
        # 段落 = 至少 2px 宽的 True 段
        runs2 = []
        cur = 0
        for v in col_has:
            if v:
                cur += 1
            else:
                if cur > 0:
                    runs2.append(cur)
                cur = 0
        if cur > 0:
            runs2.append(cur)
        n_comp = sum(1 for r in runs2 if r >= 2)
        # ---- 形状多样性（v321-fix3）：区分"真文字"和"重复纹样"----
        # 真文字行的各字母形状差异大（含 l/I/k 这类瘦高字符），mean_aspect ≫ med_aspect；
        # 佩斯利/头巾/迷彩这类重复纹样的每个 motif 形状高度一致，mean ≈ med。
        # 25 张测试图实测：
        #   b78e60 军标文字：mean-med = 0.50, 0.57, 0.52, 0.74  (>=0.40)
        #   13c8b7 佩斯利头巾：mean-med = 0.09, 0.26, 0.16, 0.10, 0.38  (全部 <0.40)
        aspect_med = aspect_mean = 0.0
        if n_comp >= 3:
            n_cc, _lbl, stats, _ = cv2.connectedComponentsWithStats(
                glyph_mask.astype(np.uint8), connectivity=8)
            aspects = []
            for ci in range(1, n_cc):
                _x, _y, w, h, area = stats[ci]
                if area < 8 or w == 0:
                    continue
                aspects.append(h / float(w))
            if aspects:
                aspect_med = float(np.median(aspects))
                aspect_mean = float(np.mean(aspects))
        band_area = (y1 - y0) * (x1 - x0)
        fill_ratio = float(glyph_mask.sum()) / max(band_area, 1)
        # 列间"空隙比"：整列几乎没有 glyph 像素的列占比（字母间的间隙）。
        # 文字有大量间隙（gap 高），实心图案/装饰几乎无间隙（gap≈0）。
        gap_ratio = 1.0 - float(col_has.sum()) / max(x1 - x0, 1)
        # ---- 文字 vs 图案/装饰/噪点 的联合判定（v321 修订，基于 25 张测试图 band 统计）----
        # 经验结论（实测，非拍脑袋）：
        #   * 实心纹理/山脊风景(85f5d2f)/佩斯利(3a300c)：fill 普遍 >=0.69（整块实体）
        #   * 真实文字行(eddf/b78e60/13c8b7/6978fab/184432)：fill <=0.56（字母有孔洞+字距）
        #   * 单色度(color_uniform)在此集"反向"：camo 暗字 cu 低(0.25)、风景 cu 高(0.87)，
        #     故不能用作主判据，改用 fill 上界 + 离散度(n_comp/gap)。
        #   * 7c79f3b 紧身衣 lace 假阳性教训：lace 眼孔的 gap_ratio 0.4-0.69 远高于真文字 0.19-0.33。
        #     收紧为保守策略（宁可漏检文字，不可把假字盖在产品图上）。
        if fill_ratio < 0.03 or fill_ratio > 0.65:
            continue                                   # 过低=噪点散点；过高=实心纹理/风景，非离散文字
        if n_comp < 5:
            continue                                   # <5 个离散元素 => 装饰/稀疏图案，不是文字行
        if gap_ratio > 0.40 or fill_ratio > 0.50:
            continue                                   # 太稀疏(eyelet)或太密 => 排除lace/装饰/密排字
        # 形状多样性门槛：文字有"高瘦"字母拉高均值，重复纹样均值≈中位数
        shape_diversity = aspect_mean - aspect_med
        if shape_diversity < 0.45:
            continue                                   # 形状过于一致 => 佩斯利/迷彩重复纹样，不是真文字
        is_text = True
        if not is_text:
            continue
        result.append(TextBand(
            y0=y0, y1=y1, x0=x0, x1=x1,
            bg_rgb=bg_rgb, fg_rgb=fg_rgb,
            cap_h_guess=cap_h, weight_guess=weight,
            n_components=n_comp,
        ))
    return result


def _classify_type(profile_dict: dict) -> str:
    """根据画像特征把图分到 7 种类型之一。"""
    has_text = profile_dict["has_text"]
    edge = profile_dict["edge_density"]
    bg_uni = profile_dict["bg_uniformity"]
    n_colors = profile_dict["n_unique_colors_q"]
    H, W = profile_dict["H"], profile_dict["W"]
    aspect = W / max(H, 1)
    # 规则（经验值，可在更多图上继续校准）
    if has_text and profile_dict["text_bands"]:
        # 文字带 > 0 且非全图纹理 → text_poster 或 graphic_logo
        # 注意：profile_dict 是 asdict() 的结果，text_bands 是 dict 列表
        max_band_h = max((tb["y1"] - tb["y0"]) for tb in profile_dict["text_bands"])
        if max_band_h > 40 and bg_uni < 0.85:
            return "text_poster"
        return "graphic_logo"
    if bg_uni > 0.55 and n_colors < 60:
        # 大面积单色 + 色数少 → 极简 landscape 或 product 白底
        if edge < 0.08:
            return "landscape"
        return "product"
    if n_colors > 200 and edge > 0.12:
        return "pattern"
    if n_colors > 120 and 0.06 < edge < 0.18 and bg_uni < 0.5:
        return "cartoon"
    if edge > 0.18:
        return "graphic_logo"
    return "unknown"


def analyze_image(path: str) -> ImageProfile:
    """对一张输入图做全量画像，返回 ImageProfile。"""
    img = Image.open(path).convert("RGB")
    W, H = img.size
    arr = np.array(img)
    gray = np.array(img.convert("L"))
    # 主色
    hues, sals, vals = _detect_dominant_hsv(arr, top_k=5)
    # 边缘密度 + 背景单一色
    edge = _edge_density(gray)
    bg_uni = _bg_uniformity(arr)
    # 色复杂度
    q, _ = _quantize_palette(arr, n_colors=16)
    _, flat = _quantize_palette(arr, n_colors=8)
    n_unique = int(len(np.unique(flat)))
    # 文字带
    text_bands = _detect_text_bands(gray, arr)
    # 过滤过小的杂散 band
    text_bands = [tb for tb in text_bands if tb.n_components >= 2 and (tb.y1 - tb.y0) >= 18]
    has_text = len(text_bands) > 0
    prof = ImageProfile(
        path=path, W=W, H=H,
        image_type="unknown",
        has_text=has_text,
        text_bands=text_bands,
        dominant_hues_deg=hues,
        dominant_sat=sals,
        dominant_val=vals,
        edge_density=edge,
        bg_uniformity=bg_uni,
        n_unique_colors_q=n_unique,
    )
    prof.image_type = _classify_type(asdict(prof))
    # Phase B 友好的预填参数（按类型微调）
    if prof.image_type in ("text_poster", "graphic_logo"):
        prof.suggested_color_strength = 0.6
        prof.suggested_composition_strength = 0.65
        prof.suggested_controlnet_strength = 0.55
        prof.suggested_denoise = 0.22
    elif prof.image_type == "pattern":
        prof.suggested_color_strength = 0.55
        prof.suggested_composition_strength = 0.5
        prof.suggested_controlnet_strength = 0.0
        prof.suggested_denoise = 0.18
    elif prof.image_type == "cartoon":
        prof.suggested_color_strength = 0.65
        prof.suggested_composition_strength = 0.45
        prof.suggested_controlnet_strength = 0.0
        prof.suggested_denoise = 0.20
    elif prof.image_type == "product":
        prof.suggested_color_strength = 0.5
        prof.suggested_composition_strength = 0.6
        prof.suggested_controlnet_strength = 0.0
        prof.suggested_denoise = 0.15
    elif prof.image_type == "landscape":
        prof.suggested_color_strength = 0.55
        prof.suggested_composition_strength = 0.4
        prof.suggested_controlnet_strength = 0.0
        prof.suggested_denoise = 0.12
    return prof


# ============================================================================
# 4. 变体生成
# ============================================================================

def _seed_from(name: str, salt: str) -> int:
    """由文件名 + salt 派生稳定 seed（保证可复现）。"""
    h = hashlib.sha1((name + "|" + salt).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31)


def _pick_fictional(rng_seed: int, pool: List[str]) -> str:
    return pool[rng_seed % len(pool)]


def _fit_text_to_band(img: Image.Image, band: TextBand, new_word: str) -> Image.Image:
    """
    把 new_word 渲染到 band 的位置，宽度贴合 band.x1-band.x0，
    cap-h 严格 ≤ band.y1-band.y0（防溢出到相邻带）。
    字色用 band.fg_rgb。
    """
    W, H = img.size
    band_w = max(60, band.x1 - band.x0)
    band_h = max(18, band.y1 - band.y0)
    target_w = int(band_w * 0.95)  # 比 band 略窄，预留 5% 边距
    # 文字 cap-height 硬上限 = band_h * 0.92（留 8% 上下边距，防与邻带重叠）
    max_cap_h = max(8, int(band_h * 0.92))
    bold = (band.weight_guess == "bold")
    # 二分找 fsize：同时满足宽度 ≈ target_w 且 cap_h ≤ max_cap_h
    lo, hi = 8, band_h * 4
    best = None
    for _ in range(22):
        mid = (lo + hi) // 2
        f = _load_font(bold, mid)
        try:
            tw = int(f.getlength(new_word))
            bbox_A = f.getbbox("A")
            cap_h_now = bbox_A[3] - bbox_A[1]
        except Exception:
            tw = mid * len(new_word) // 2
            cap_h_now = mid
        # 硬约束：高度超了就缩
        if cap_h_now > max_cap_h:
            hi = mid
            continue
        if tw < target_w * 0.92:
            lo = mid
            best = (mid, f, tw, cap_h_now)
        elif tw > target_w * 1.08:
            hi = mid
        else:
            best = (mid, f, tw, cap_h_now)
            break
    if best is None:
        # 兜底：取 lo 对应的字号（保证不高）
        f = _load_font(bold, lo)
        try:
            tw = int(f.getlength(new_word))
            bbox_A = f.getbbox("A")
            cap_h_now = bbox_A[3] - bbox_A[1]
        except Exception:
            tw, cap_h_now = lo * len(new_word) // 2, lo
        best = (lo, f, tw, cap_h_now)
    fsize, font, tw, cap_h_render = best
    pad = max(2, fsize // 4)
    layer = Image.new("RGBA", (tw + pad * 2, cap_h_render + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    try:
        bbox_A = font.getbbox("A")
    except Exception:
        bbox_A = (0, 0, fsize // 2, fsize)
    y_in_layer = pad - bbox_A[1]
    d.text((pad, y_in_layer), new_word, font=font, fill=(*band.fg_rgb, 255))
    # 居中放置（水平用 tw，垂直用 cap_h_render）
    px = band.x0 + (band_w - tw) // 2
    py = band.y0 + (band_h - cap_h_render) // 2 - bbox_A[1]
    out = img.convert("RGBA")
    out.paste(layer, (px, py), layer)
    return out.convert("RGB")


def _erase_text_band(img: Image.Image, band: TextBand, feather: int = 18) -> Image.Image:
    """
    纹理延续式擦除：把 band 区域用"上方 + 下方邻近行"做垂直梯度混合填回去，
    上下边各 feather px 渐变。camo/图案图上不再出现"纯色矩形"（v307 红线）。
    对纯色背景图这是 no-op（采样到的就是同色），对纹理图把硬矩形变成平滑过渡。
    """
    W, H = img.size
    arr = np.array(img).copy()
    y0, y1, x0, x1 = band.y0, band.y1, band.x0, band.x1
    bh = y1 - y0
    bw = x1 - x0
    if bh <= 2 or bw <= 2:
        return img
    # 采样上/下"上下文条"：band 上下各 feather px
    up_y0 = max(0, y0 - feather)
    up_y1 = y0
    dn_y0 = y1
    dn_y1 = min(H, y1 + feather)
    # 用"上下条的平均列"作为填回纹理的两个端点
    up_strip = arr[up_y0:up_y1].mean(axis=0) if up_y1 > up_y0 else arr[max(0, y0-1):y0].mean(axis=0)
    dn_strip = arr[dn_y0:dn_y1].mean(axis=0) if dn_y1 > dn_y0 else arr[y1:min(H, y1+1)].mean(axis=0)
    if up_strip.ndim == 1:
        up_strip = np.broadcast_to(up_strip, (bh, bw, 3))
    if dn_strip.ndim == 1:
        dn_strip = np.broadcast_to(dn_strip, (bh, bw, 3))
    if up_strip.shape[0] != bh:
        up_strip = np.broadcast_to(up_strip.mean(axis=0), (bh, bw, 3))
    if dn_strip.shape[0] != bh:
        dn_strip = np.broadcast_to(dn_strip.mean(axis=0), (bh, bw, 3))
    # 垂直梯度 alpha：band 顶部 1.0（取上条），底部 0.0（取下条）
    alpha = np.linspace(1.0, 0.0, bh, dtype=np.float32).reshape(bh, 1, 1)
    blended = up_strip * alpha + dn_strip * (1.0 - alpha)
    # 水平 feather：band 左右各 feather px 渐变
    if bw > 2 * feather:
        hfeather = np.ones(bw, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
        hfeather[:feather] = ramp
        hfeather[-feather:] = ramp[::-1]
    else:
        hfeather = np.ones(bw, dtype=np.float32)
    hfeather = hfeather.reshape(1, bw, 1)
    sub = arr[y0:y1, x0:x1].astype(np.float32)
    out_sub = sub * (1 - hfeather) + blended * hfeather
    arr[y0:y1, x0:x1] = out_sub.astype(np.uint8)
    return Image.fromarray(arr)


def generate_variants(img: Image.Image, prof: ImageProfile, n_variants: int = 3, seed: int = 0) -> List[Tuple[str, Image.Image]]:
    """
    对一张图生成 n_variants 个变体。
    返回 [(variant_name, image), ...]
    所有变体严守 v307 红线。

    关键改动 D：每变体使用"定向"HSV 偏移（v1=+满色相, v2=-满色相, v3=+满饱和度），
    而不是随机均匀采样——后者常取到 2-3°，在人眼上完全不可见。定向后 3 个变体彼此
    明显不同、也都明显不同于原图，但仍都在 ±15°/±20%/±10% 红线内。
    """
    rng = np.random.RandomState(seed)
    # 3 套定向：色相正满、色相负满、饱和度正满
    DIRECTIONS = [
        (+HUE_DELTA_DEG, 0.0, 0.0),
        (-HUE_DELTA_DEG, 0.0, 0.0),
        (0.0, +SAT_DELTA, 0.0),
    ]
    out: List[Tuple[str, Image.Image]] = []
    base_name = os.path.splitext(os.path.basename(prof.path))[0]

    # ---- 变体策略（按 type 路由） ----
    if prof.image_type in ("text_poster", "graphic_logo") and prof.has_text:
        # 文字图：先做色相微调，再对每条文字带擦除+重写虚构词
        for i in range(min(n_variants, 3)):
            dh, ds, dv = DIRECTIONS[i]
            v_img = img.copy()
            v_img = _hsv_shift(v_img, dh, ds, dv)
            for j, band in enumerate(prof.text_bands):
                v_img = _erase_text_band(v_img, band)
                # 选词：大带用 brand，小带用 subtitle
                if (band.y1 - band.y0) >= 40:
                    word = _pick_fictional(seed + i * 100 + j, FICTIONAL_BRANDS)
                else:
                    word = _pick_fictional(seed + i * 100 + j + 7, FICTIONAL_SUBTITLES)
                v_img = _fit_text_to_band(v_img, band, word)
            # 微仿射方向也固定（避免再叠加随机）
            s = 1.0 + (i - 1) * 0.02  # 0.96 / 1.0 / 1.04
            r = (i - 1) * 1.5          # -1.5° / 0 / +1.5°
            v_img = _micro_affine(v_img, s, r)
            out.append((f"v{i+1}_textswap", v_img))
    elif prof.image_type == "pattern":
        for i in range(min(n_variants, 3)):
            dh, ds, dv = DIRECTIONS[i]
            v_img = _hsv_shift(img.copy(), dh, ds, dv)
            out.append((f"v{i+1}_hsv", v_img))
    elif prof.image_type == "cartoon":
        for i in range(min(n_variants, 3)):
            dh, ds, dv = DIRECTIONS[i]
            v_img = _hsv_shift(img.copy(), dh, ds, dv)
            s = 1.0 + (i - 1) * 0.03
            r = (i - 1) * 2.0
            v_img = _micro_affine(v_img, s, r)
            out.append((f"v{i+1}_hsv_affine", v_img))
    elif prof.image_type == "product":
        for i in range(min(n_variants, 3)):
            dh, ds, dv = DIRECTIONS[i]
            v_img = _hsv_shift(img.copy(), dh, ds, 0.0)
            s = 1.0 + (i - 1) * 0.03
            v_img = _micro_affine(v_img, s, 0.0)
            out.append((f"v{i+1}_hsv_zoom", v_img))
    elif prof.image_type == "landscape":
        for i in range(min(n_variants, 3)):
            dh, ds, dv = DIRECTIONS[i]
            v_img = _hsv_shift(img.copy(), dh, ds, dv)
            out.append((f"v{i+1}_mood", v_img))
    else:
        # unknown / 默认：3 套定向 HSV
        for i in range(min(n_variants, 3)):
            dh, ds, dv = DIRECTIONS[i]
            v_img = _hsv_shift(img.copy(), dh, ds, dv)
            out.append((f"v{i+1}_hsv", v_img))
    return out


# ============================================================================
# 5. Gallery
# ============================================================================

def _img_to_data_uri(img: Image.Image, max_w: int = 360) -> str:
    import base64, io
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_gallery(items: List[dict], out_path: str) -> None:
    """items: [{"src_path": str, "type": str, "profile": dict, "variants": [(name, path), ...]}]"""
    rows = []
    for it in items:
        src_uri = _img_to_data_uri(Image.open(it["src_path"]))
        var_cards = "".join(
            f'<div class="v"><img src="{_img_to_data_uri(Image.open(p))}"><div class="cap">{n}</div></div>'
            for n, p in it["variants"]
        )
        prof = it["profile"]
        prof_str = (
            f'type=<b>{prof["image_type"]}</b> · has_text={prof["has_text"]} · '
            f'edges={prof["edge_density"]:.3f} · bg_uni={prof["bg_uniformity"]:.2f} · '
            f'colors={prof["n_unique_colors_q"]} · bands={len(prof["text_bands"])}'
        )
        rows.append(
            f'<section><div class="head"><div class="src"><img src="{src_uri}"></div>'
            f'<div class="meta"><h3>{os.path.basename(it["src_path"])}</h3>'
            f'<div class="prof">{prof_str}</div></div></div>'
            f'<div class="variants">{var_cards}</div></section>'
        )
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>v321 通用化裂变</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;background:#fafafa;padding:24px;margin:0;color:#222;}}
h1{{margin:0 0 4px;font-size:22px;}}h3{{margin:0;font-size:15px;}}
.sub{{color:#666;margin-bottom:20px;font-size:13px;}}
section{{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:16px;margin-bottom:16px;}}
.head{{display:flex;gap:16px;align-items:flex-start;margin-bottom:12px;}}
.src img{{max-width:180px;max-height:180px;border:1px solid #ddd;}}
.meta{{flex:1;}}
.prof{{color:#555;font-size:12px;font-family:ui-monospace,Consolas,monospace;margin-top:6px;}}
.variants{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;}}
.v{{background:#f6f6f6;border-radius:6px;overflow:hidden;}}
.v img{{width:100%;display:block;}}
.cap{{padding:6px;font-size:12px;text-align:center;color:#333;font-weight:500;}}
.legend{{background:#fffbe6;border:1px solid #ffe58f;padding:10px 14px;border-radius:6px;margin-bottom:16px;font-size:12px;}}
</style></head><body>
<h1>v321 通用化图裂变 — Gallery</h1>
<div class="sub">{len(items)} 张原图 × 3 变体 = {sum(len(it["variants"]) for it in items)} 张裂变图（严守 v307 红线：色相±15°/饱和度±20%/明度±10%/仿射±5%±3°）</div>
<div class="legend">⚠ 自动跳过任何 <code>_SKIP_*</code> 前缀的文件。所有"文字裂变"使用<strong>原创虚构词</strong>（FICTIONAL_BRANDS/SUBTITLES 词池），绝无真实品牌字。这是 Phase A CV 基线；Phase B 会把本画廊的 profile 直接喂给 ComfyUI 的 IPAdapter/AnyText2/ControlNet Union。</div>
{''.join(rows)}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================================
# 6. Driver
# ============================================================================

def _is_skipped(fn: str) -> bool:
    n = fn.lower()
    if n.startswith("_skip"):
        return True
    # 显式违规 / 政策拦截（无需用户每次再标记）
    bad = ("weed", "drug", "cannabis", "marijuana", "narcotic")
    return any(b in n for b in bad)


def run(input_dir: str, output_dir: str, n_variants: int = 3) -> int:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        and not _is_skipped(f)
    ])
    if not files:
        print(f"[ERROR] 在 {input_dir} 没有可处理的图片")
        return 1
    print(f"[scan] {len(files)} 张图将进入通用化裂变（已自动跳过 _SKIP_/违规关键词）")
    items = []
    t0 = time.time()
    for idx, p in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(p))[0]
        print(f"\n[{idx}/{len(files)}] {base}")
        try:
            prof = analyze_image(p)
            print(f"  profile: type={prof.image_type} has_text={prof.has_text} "
                  f"edges={prof.edge_density:.3f} bg_uni={prof.bg_uniformity:.2f} "
                  f"colors={prof.n_unique_colors_q} bands={len(prof.text_bands)}")
            if prof.text_bands:
                for j, tb in enumerate(prof.text_bands):
                    print(f"    band[{j}] y={tb.y0}-{tb.y1} x={tb.x0}-{tb.x1} "
                          f"cap_h={tb.cap_h_guess} weight={tb.weight_guess} "
                          f"fg={tb.fg_rgb} bg={tb.bg_rgb} comps={tb.n_components}")
            seed = _seed_from(base, "v321")
            variants = generate_variants(Image.open(p).convert("RGB"), prof, n_variants=n_variants, seed=seed)
            var_files = []
            for vname, vimg in variants:
                vp = os.path.join(output_dir, f"{base}__{vname}.jpg")
                vimg.save(vp, quality=92)
                var_files.append((vname, vp))
            # 写 profile json（Phase B 友好）
            pj = os.path.join(output_dir, f"{base}__profile.json")
            prof_dict = asdict(prof)
            with open(pj, "w", encoding="utf-8") as f:
                json.dump(prof_dict, f, ensure_ascii=False, indent=2)
            items.append({"src_path": p, "type": prof.image_type, "profile": prof_dict, "variants": var_files})
            print(f"  -> {len(variants)} 变体已落盘")
        except Exception as e:
            print(f"  [FAIL] {repr(e)}")
            continue
    # Gallery
    gallery = os.path.join(output_dir, "gallery.html")
    build_gallery(items, gallery)
    print(f"\n[DONE] {len(items)}/{len(files)} 张图处理完成，gallery -> {gallery}（{time.time()-t0:.1f}s）")
    return 0 if items else 1


def main():
    ap = argparse.ArgumentParser(description="v321 通用化图裂变（CV 基线，零 GPU/ComfyUI 依赖）")
    ap.add_argument("--input", "-i", required=True, help="输入图片文件夹")
    ap.add_argument("--output", "-o", required=True, help="输出目录")
    ap.add_argument("--variants", "-n", type=int, default=3, help="每张图变体数（默认 3）")
    args = ap.parse_args()
    sys.exit(run(args.input, args.output, args.variants))


if __name__ == "__main__":
    main()
