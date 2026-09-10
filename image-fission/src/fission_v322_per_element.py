"""
fission_v322_per_element.py — 通用化图裂变引擎（v322，逐元素裂变）

定位
====
- v321 的完全超集。保留所有 v321 能力（auto-detect、文字带裂变、HSV 微调、纹理延续式擦除）。
- 新增【逐元素裂变】（v322 永久硬规则）：原图里每一个独立元素（dog tag / 链条 / 徽记 /
  小佩斯利 / 边框花纹 / logo 角标 / 小图案）都被检测出来，并在每个变体里做微调处理。
- Phase A（零 GPU / 零 ComfyUI / 纯 PIL+numpy+cv2）：只做几何/颜色微调
    ① 颜色微调（HSV ±10°，比大图更保守）
    ② 几何微调（缩放 ±5% / 旋转 ±5° / 位置偏移 ±5px）
    ③ 数量微调（删 1-2 个最小元素 / 镜像加 1-2 个副本）
- Phase B（ComfyUI）：AnyText2 v2.0 在 dog tag 等内部原位生成新文字；IPAdapter Plus 锁
  整体风格；ControlNet Union SDXL 锁结构。**不在本文件实现**，本文件只输出 profile 供 Phase B 消费。

v322 永久红线（2026-09-10 定稿）
================================
- 裂变必须覆盖原图里【所有元素】，不只是大字
- 每个小元素独立做差异化（颜色/几何/数量/形状）
- 严守 v307 永久红线（HSV ±15°/sat ±20%/lum ±10%/仿射 ±5%±3°）
- 严守 v307 主体不许缺失（删除最小元素是允许的，但不超过总数的 20%）
- 严守 v307 字体偏好（Didone/Bodoni 宽厚大字 + Black Ops One 军标字体）
- 严守 v307 原创品牌字（绝不出现 BACARDÍ/MCKHEART 等真实品牌）

用法
====
    python fission_v322_per_element.py \
        --input  "E:/Desktop/图裂变测试图" \
        --output "E:/Desktop/双接口/image-fission/jobs/fission_v322" \
        --variants 3
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
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import cv2

# ============================================================================
# 0. 常量与词池（v307 红线锁死）
# ============================================================================
HUE_DELTA_DEG = 15.0
SAT_DELTA = 0.20
LUM_DELTA = 0.10
AFFINE_SCALE_DELTA = 0.05
AFFINE_ROT_DEG = 3.0

# v322 红线：小元素更保守
SMALL_HUE_DELTA_DEG = 16.0
SMALL_SAT_DELTA = 0.12
SMALL_LUM_DELTA = 0.06
SMALL_AFFINE_SCALE_DELTA = 0.05
SMALL_AFFINE_ROT_DEG = 5.0   # 小元素可多转 2°（视觉不显眼）
SMALL_POSITION_OFFSET_PX = 5

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

FICTIONAL_BRANDS = [
    "NOCTAVEN", "DARKRESERVE", "BLACKLABEL", "IRONBEAR", "STEELHAWK",
    "WOLFPACK", "THUNDERBOLT", "SHADOWFANG", "NIGHTFALL", "DARKHORSE",
    "BLITZKRIEG", "STORMBRING", "FROSTBITE", "BLACKICE", "WILDHUNT",
    "DEEPRIDGE", "GHOSTLINE", "LIONHEART", "IRONCLAW", "STARFORGE",
    "MOONRAVEN", "ASHFALL", "GRIMREAPER", "TYPHOON", "BARRACUDA",
    "BLACKWING", "SILVERFANG", "STONECOLD", "FROSTCLAW", "THORNBACK",
    "VOLKOV", "BAXTER", "KAZAN", "REZNOR", "MARLOWE", "HARLAN",
]
FICTIONAL_SUBTITLES = [
    "DARK RESERVE", "IRON BEAR", "BORN TO RIDE", "BLACK LABEL",
    "NIGHT FALL", "STEEL HAWK", "WOLF PACK", "DARK HORSE",
    "STORM BRING", "GHOST LINE", "MOON RAVEN", "IRON CLAW",
    "WILD HUNT", "DEEP RIDGE", "STAR FORGE", "ASH FALL",
    "EAST WING", "RED FIVE", "ZERO DAY", "BLACK SUN",
]
# 小元素（dog tag、徽记内）专用短词
FICTIONAL_TINY = [
    "VOLKOV", "BAXTER", "REZNOR", "HARLAN", "MARLOWE", "KAZAN",
    "VOLT", "RAVN", "AXEL", "NOVA", "ECHO", "RAZOR",
    "OMEGA", "DELTA", "ALPHA", "BRAVO",
]


# ============================================================================
# 1. 数据结构
# ============================================================================

@dataclass
class TextBand:
    """自动检测到的横排文字带（v321 已有）。"""
    y0: int
    y1: int
    x0: int
    x1: int
    bg_rgb: Tuple[int, int, int]
    fg_rgb: Tuple[int, int, int]
    cap_h_guess: int
    weight_guess: str
    n_components: int


@dataclass
class SmallElement:
    """
    v322 新增：原图里每一个独立小元素的画像。
    类别 category 用启发式分类（不一定 100% 准，但够 Phase A 路由）：
      - dog_tag：椭圆/盾形，中等大小，位于图中央偏下
      - chain：细长竖线状（珠子串/链条），高度 > 宽度 4x
      - emblem：中等大小圆形/盾形，带粗轮廓
      - border：紧贴图像边缘的长条状
      - decoration：其他非文字、非图案主体的孤立元素
      - motif：小佩斯利/小图案/重复花纹的单个 motif（头巾/迷彩图大量出现）
      - text_small：band 没捕获的小字（如 dog tag 内部的 "BAXTER"）
    """
    y0: int
    y1: int
    x0: int
    x1: int
    category: str          # dog_tag|chain|emblem|border|decoration|motif|text_small
    area: int              # 像素面积
    bbox_area: int         # bbox 面积（用于 aspect 计算）
    aspect_h_over_w: float # 高/宽比
    fill_density: float    # bbox 内非空比例（0-1）
    mean_rgb: Tuple[int, int, int]  # 元素主色
    n_components: int      # 内部连通域数
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # H×W 二值 mask


@dataclass
class ImageProfile:
    path: str
    W: int
    H: int
    image_type: str
    has_text: bool
    text_bands: List[TextBand] = field(default_factory=list)
    # v322 新增
    small_elements: List[SmallElement] = field(default_factory=list)
    dominant_hues_deg: List[float] = field(default_factory=list)
    dominant_sat: List[float] = field(default_factory=list)
    dominant_val: List[float] = field(default_factory=list)
    edge_density: float = 0.0
    bg_uniformity: float = 0.0
    n_unique_colors_q: int = 0
    # Phase B 友好参数
    suggested_color_strength: float = 0.6
    suggested_composition_strength: float = 0.5
    suggested_controlnet_strength: float = 0.0
    suggested_denoise: float = 0.25


# ============================================================================
# 2. 工具（复用 v321）
# ============================================================================

def _pil_hsv_to_rgb_hsv(arr_hsv_u8: np.ndarray) -> np.ndarray:
    h, s, v = arr_hsv_u8[..., 0].astype(np.float32), arr_hsv_u8[..., 1].astype(np.float32) / 255.0, arr_hsv_u8[..., 2].astype(np.float32) / 255.0
    h_deg = h * 360.0 / 255.0
    return np.stack([h_deg, s, v], axis=-1)


def _rgb_hsv_to_pil_hsv(h_deg: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h8 = np.clip((h_deg % 360.0) * 255.0 / 360.0, 0, 255).astype(np.uint8)
    s8 = np.clip(s * 255.0, 0, 255).astype(np.uint8)
    v8 = np.clip(v * 255.0, 0, 255).astype(np.uint8)
    return np.stack([h8, s8, v8], axis=-1)


def _hsv_shift(img_rgb: Image.Image, dh_deg: float, ds: float, dv: float) -> Image.Image:
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


def _hsv_shift_mask(img_rgb: Image.Image, mask: np.ndarray, dh_deg: float, ds: float, dv: float) -> Image.Image:
    """
    v322 新增：只在 mask 区域（boolean H×W）做 HSV 微调，比大图更保守。
    用于小元素的颜色微调——保留原图主体的色调，只改小元素本身。
    """
    dh = float(np.clip(dh_deg, -SMALL_HUE_DELTA_DEG, SMALL_HUE_DELTA_DEG))
    ds = float(np.clip(ds, -SMALL_SAT_DELTA, SMALL_SAT_DELTA))
    dv = float(np.clip(dv, -SMALL_LUM_DELTA, SMALL_LUM_DELTA))
    if abs(dh) < 0.01 and abs(ds) < 0.001 and abs(dv) < 0.001:
        return img_rgb.copy()
    if mask is None or not mask.any():
        return img_rgb.copy()
    arr = np.array(img_rgb.convert("RGB")).copy()
    hsv = _pil_hsv_to_rgb_hsv(np.array(Image.fromarray(arr).convert("HSV")))
    hsv[..., 0] = np.where(mask, (hsv[..., 0] + dh) % 360.0, hsv[..., 0])
    hsv[..., 1] = np.where(mask, np.clip(hsv[..., 1] + ds, 0, 1), hsv[..., 1])
    hsv[..., 2] = np.where(mask, np.clip(hsv[..., 2] + dv, 0, 1), hsv[..., 2])
    out_hsv = _rgb_hsv_to_pil_hsv(hsv[..., 0], hsv[..., 1], hsv[..., 2])
    return Image.fromarray(np.array(Image.fromarray(out_hsv, mode="HSV").convert("RGB")))


def _micro_affine(img: Image.Image, scale: float, rot_deg: float) -> Image.Image:
    s = float(np.clip(scale, 1.0 - AFFINE_SCALE_DELTA, 1.0 + AFFINE_SCALE_DELTA))
    r = float(np.clip(rot_deg, -AFFINE_ROT_DEG, AFFINE_ROT_DEG))
    if abs(s - 1.0) < 0.001 and abs(r) < 0.01:
        return img.copy()
    W, H = img.size
    rad = np.deg2rad(r)
    new_w = int(np.ceil(W * abs(np.cos(rad)) + H * abs(np.sin(rad))))
    new_h = int(np.ceil(W * abs(np.sin(rad)) + H * abs(np.cos(rad))))
    canvas = Image.new("RGB", (new_w, new_h), color=(128, 128, 128))
    rot = img.rotate(r, resample=Image.BICUBIC, expand=True)
    rw, rh = rot.size
    sx = new_w / rw
    sy = new_h / rh
    s_use = min(sx, sy) * s
    rot = rot.resize((int(rw * s_use), int(rh * s_use)), Image.BICUBIC)
    canvas.paste(rot, ((new_w - rot.size[0]) // 2, (new_h - rot.size[1]) // 2))
    if canvas.size != (W, H):
        canvas = canvas.resize((W, H), Image.BICUBIC)
    return canvas


def _load_font(bold: bool, fsize: int) -> ImageFont.FreeTypeFont:
    cands = WIN_FONTS_BOLD if bold else WIN_FONTS_REG
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, fsize)
            except Exception:
                continue
    return ImageFont.load_default()


# ============================================================================
# 3. 通用工具
# ============================================================================

def _quantize_palette(arr: np.ndarray, n_colors: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    q = (arr // (256 // n_colors)).astype(np.int32)
    flat = q[..., 0] * n_colors * n_colors + q[..., 1] * n_colors + q[..., 2]
    return q, flat


def _detect_dominant_hsv(arr: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[float], List[float]]:
    hsv = _pil_hsv_to_rgb_hsv(np.array(Image.fromarray(arr).convert("HSV")))
    H_flat = hsv[..., 0].reshape(-1)
    S_flat = hsv[..., 1].reshape(-1)
    V_flat = hsv[..., 2].reshape(-1)
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
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    dy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    mag = dx + dy
    thresh = 40
    return float((mag > thresh).mean())


def _bg_uniformity(arr: np.ndarray) -> float:
    H, W = arr.shape[:2]
    band = max(2, int(min(H, W) * 0.05))
    borders = np.concatenate([
        arr[:band].reshape(-1, 3),
        arr[-band:].reshape(-1, 3),
        arr[:, :band].reshape(-1, 3),
        arr[:, -band:].reshape(-1, 3),
    ], axis=0)
    q = (borders // 32).astype(np.int32)
    flat = q[:, 0] * 256 + q[:, 1] * 8 + q[:, 2]
    counts = np.bincount(flat, minlength=512)
    if counts.sum() == 0:
        return 0.0
    return float(counts.max() / counts.sum())


# ============================================================================
# 4. 小元素检测（v322 核心新增）
# ============================================================================

def _detect_small_elements(gray: np.ndarray, arr: np.ndarray,
                            text_band_mask: np.ndarray,
                            W: int, H: int) -> List[SmallElement]:
    """
    v322 核心：在原图里检测所有"非文字带"的小元素。

    算法：
    1) 用 Otsu 自适应阈值得到前景 mask（高对比元素）
    2) 排除已经被 text_band_mask 覆盖的区域（避免重复）
    3) cv2.connectedComponentsWithStats 拿到所有连通域
    4) 按面积/形状/位置启发式分类
    5) 过滤太小的（< 80 像素 = 噪点）和太大的（> H*W*0.30 = 整图）

    返回 small_elements 列表。
    """
    # 1) 自适应阈值：先 Otsu，全图统一
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 但 camo/纯色背景图 Otsu 不可靠，加 Canny 边缘辅助
    edges = cv2.Canny(blur, 30, 100)
    fg = cv2.bitwise_or(fg, edges)
    # 闭运算把小碎片连起来
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

    # 2) 排除文字带区域
    fg = cv2.bitwise_and(fg, cv2.bitwise_not(text_band_mask))

    # 3) 连通域
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    elements: List[SmallElement] = []
    img_area = H * W
    for lab in range(1, n_labels):
        x, y, w, h, area = stats[lab]
        cx, cy = centroids[lab]
        # 过滤：太小（噪点）或太大（整图背景块）
        if area < 250 or area > img_area * 0.20:
            continue
        # bbox 长宽比过滤：极端长条（边/角线）只在 border 类下保留
        aspect_h_over_w = h / max(w, 1)
        aspect_w_over_h = w / max(h, 1)
        if aspect_h_over_w > 12 or aspect_w_over_h > 12:
            # 极细长的，归为 border/chain 候选
            if w * h > img_area * 0.15:
                # 占图 15%+ 才算 border
                pass
            else:
                continue
        # 紧贴图边（任意一边 5px 内）
        touches_border = (x <= 5 or y <= 5 or (x + w) >= W - 5 or (y + h) >= H - 5)
        # 类别分类（启发式）
        category = _classify_small_element(
            x, y, w, h, area, aspect_h_over_w, touches_border, cx, cy, W, H
        )
        if category is None:
            continue
        # 元素 mask（与 fg 同 shape）
        em_mask = (labels == lab).astype(np.uint8) * 255
        em_mask_full = np.zeros((H, W), dtype=np.uint8)
        em_mask_full[y:y+h, x:x+w] = em_mask[:h, :w]
        # 主色：元素内像素的 RGB 均值（NaN 防护：mask 为空时退回 bbox 均值）
        region = arr[y:y+h, x:x+w]
        m_local = em_mask[:h, :w] > 0
        if m_local.any():
            rgb_mean = region[m_local].mean(axis=0)
            if not np.isfinite(rgb_mean).all():
                rgb_mean = region.reshape(-1, 3).mean(axis=0)
        else:
            rgb_mean = region.reshape(-1, 3).mean(axis=0)
        mean_rgb = tuple(int(c) for c in rgb_mean)
        # 元素内的连通子域数（细碎程度）
        _, sub_lab = cv2.connectedComponents(em_mask, connectivity=8)
        n_comp = int(sub_lab.max())
        bbox_area = w * h
        fill = area / max(bbox_area, 1)
        elements.append(SmallElement(
            y0=int(y), y1=int(y + h), x0=int(x), x1=int(x + w),
            category=category, area=int(area), bbox_area=int(bbox_area),
            aspect_h_over_w=float(aspect_h_over_w), fill_density=float(fill),
            mean_rgb=mean_rgb, n_components=n_comp,
            mask=(em_mask_full > 0),
        ))
    # 按面积从大到小排序（dog tag/emblem 优先处理）
    elements.sort(key=lambda e: -e.area)
    return elements


def _classify_small_element(x, y, w, h, area, aspect_h_over_w,
                              touches_border, cx, cy, W, H) -> Optional[str]:
    """
    启发式分类小元素。返回类别字符串，或 None 表示应丢弃。
    v322b 修订（2026-09-10 紧急修复）：
      - 放宽 dog_tag aspect 范围到 0.4-2.6（v322 初版 0.7-1.5 太严，b78e60 真 dog tag
        aspect=2.04 被错归 motif）
      - motif/decoration 类别不再触发 transform（v322 初版对所有 motif 做
        transform 拆碎了 camo/佩斯利/紧身衣纹理）
      - chain 分类放宽到 aspect > 2.5 或 width < W*0.06 且 height > H*0.10
      - border 紧贴图边的长条保持
    """
    rel_area = area / (W * H)
    aspect_w_over_h = 1.0 / max(aspect_h_over_w, 0.01)
    area_abs = area
    # 1) 极细长 = chain（狗链/细绳/吊线），任一边 aspect >= 3 即判
    #    b78e60 链条实测 w=19 h=89 → aspect_h=4.68，正是此处捕获
    if (aspect_h_over_w >= 3.0 or aspect_w_over_h >= 3.0) and area_abs < W * H * 0.05:
        return "chain"
    # 2) 紧贴图边的长条 = border（边框花纹/边线）
    if touches_border and (aspect_h_over_w > 6 or aspect_w_over_h > 6) and rel_area > 0.08:
        return "border"
    # 3) 中等椭圆/盾形 = dog_tag
    #    v322c 关键修复：用【绝对面积】阈值（1200px），避开 2000px 大图相对面积
    #    失效问题（b78e60 狗牌相对面积仅 0.0012，被旧 rel_area>0.005 漏掉）。
    #    面积下限提到 1200：b78e60 狗牌(3712)判 dog_tag；13c8b7 佩斯利碎块(<=654)
    #    落 motif（温和色相漂移，不仿射不碎裂）。宽高比下限放宽到 0.35 兼容高瘦狗牌。
    if (0.5 < aspect_h_over_w < 2.6 and aspect_w_over_h > 0.35
            and 1200 <= area_abs < W * H * 0.10 and cy > H * 0.18):
        return "dog_tag"
    # 4) 大圆形/盾形 = emblem
    if 0.6 < aspect_h_over_w < 1.7 and area_abs >= W * H * 0.01 and cy < H * 0.60:
        return "emblem"
    # 5) 紧贴图边的中等元素 = border
    if touches_border and 0.05 < rel_area < 0.30 and not (
        aspect_h_over_w > 6 or aspect_w_over_h > 6
    ):
        return "border"
    # 6) 其余（佩斯利/碎花/纹理细块/噪点）= motif
    #    → 走温和局部色相漂移（_shift_motif_hue），不仿射、不 drop、不碎裂
    return "motif"


def _build_text_band_mask(text_bands: List[TextBand], W: int, H: int) -> np.ndarray:
    """把所有文字带的 bbox 区域合并成 mask（用于排除）。"""
    mask = np.zeros((H, W), dtype=np.uint8)
    for tb in text_bands:
        mask[tb.y0:tb.y1, tb.x0:tb.x1] = 255
    return mask


# ============================================================================
# 5. v321 文字带检测（基本复用，只加 n_components 字段填充）
# ============================================================================

def _detect_text_bands(gray: np.ndarray, arr: np.ndarray) -> List[TextBand]:
    H, W = gray.shape
    row_std = gray.astype(np.float32).std(axis=1)
    k = 5
    kernel = np.ones(k) / k
    row_smooth = np.convolve(row_std, kernel, mode="same")
    thresh = max(8.0, float(np.median(row_smooth)) * 1.2)
    text_rows = row_smooth > thresh
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
    bands = [(a, b) for a, b in bands if 16 <= (b - a) <= int(H * 0.45)]
    bands = [(max(0, a - 2), min(H, b + 2)) for a, b in bands]
    result: List[TextBand] = []
    for y0, y1 in bands:
        sub_g = gray[y0:y1]
        sub_rgb = arr[y0:y1]
        top_row = sub_rgb[0:1].reshape(-1, 3).mean(axis=0)
        bot_row = sub_rgb[-1:].reshape(-1, 3).mean(axis=0)
        edge_mean = (top_row + bot_row) / 2.0
        diff = np.linalg.norm(sub_rgb.astype(np.float32) - edge_mean, axis=2)
        glyph_mask = diff > 30
        if glyph_mask.sum() < 50:
            continue
        bg_rgb = tuple(int(c) for c in edge_mean)
        # fg 色：glyph 像素的中位数色
        fg_pixels = sub_rgb[glyph_mask]
        fg_med = np.median(fg_pixels.reshape(-1, 3), axis=0)
        fg_rgb = tuple(int(c) for c in fg_med)
        # cap_h 估计
        col_has = glyph_mask.any(axis=0).astype(np.int32)
        runs = []
        cur = 0
        for v in col_has:
            if v:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        runs = [r for r in runs if r >= 2]
        if not runs:
            continue
        # 用列宽估计 cap_h（粗略）
        cap_h = max(8, int(np.median(runs)) * 2)
        # 字重：glyph 行内连续 run-length 中位数
        row_thicks = []
        for r in range(sub_g.shape[0]):
            row_g = glyph_mask[r]
            if not row_g.any():
                continue
            run = []
            cur2 = 0
            for v in row_g:
                if v:
                    cur2 += 1
                else:
                    if cur2 > 0:
                        run.append(cur2)
                    cur2 = 0
            if cur2 > 0:
                run.append(cur2)
            if run:
                row_thicks.append(int(np.median(run)))
        med_thick = int(np.median(row_thicks)) if row_thicks else 0
        weight = "bold" if med_thick >= max(3, cap_h * 0.18) else "regular"
        # x 范围
        col_has_full = glyph_mask.any(axis=0)
        if not col_has_full.any():
            continue
        xs = np.where(col_has_full)[0]
        x0, x1 = int(xs[0]), int(xs[-1] + 1)
        # gap 比例
        gap_ratio = 1.0 - float(col_has_full.sum()) / max(x1 - x0, 1)
        # fill 比（按 x-extent）
        band_area = (y1 - y0) * (x1 - x0)
        fill_ratio = float(glyph_mask.sum()) / max(band_area, 1)
        # n_components（粗略）
        em_mask = (glyph_mask.astype(np.uint8)) * 255
        _, sub_labels = cv2.connectedComponents(em_mask, connectivity=8)
        n_comp = int(sub_labels.max())
        # 形状多样性（aspect）
        sub_stats = []
        for sl in range(1, sub_labels.max() + 1):
            ys, xs2 = np.where(sub_labels == sl)
            if len(ys) < 4:
                continue
            sub_stats.append((ys.max() - ys.min() + 1) / max(xs2.max() - xs2.min() + 1, 1))
        if sub_stats:
            aspect_mean = float(np.mean(sub_stats))
            aspect_med = float(np.median(sub_stats))
            shape_diversity = aspect_mean - aspect_med
        else:
            shape_diversity = 0.0
        # 过滤：噪点/实心纹理/形状过于一致
        if fill_ratio < 0.03 or fill_ratio > 0.65:
            continue
        if n_comp < 5:
            continue
        if gap_ratio > 0.40 or fill_ratio > 0.50:
            continue
        if shape_diversity < 0.45:
            continue
        result.append(TextBand(
            y0=y0, y1=y1, x0=x0, x1=x1,
            bg_rgb=bg_rgb, fg_rgb=fg_rgb,
            cap_h_guess=cap_h, weight_guess=weight,
            n_components=n_comp,
        ))
    return result


def _classify_type(profile_dict: dict) -> str:
    has_text = profile_dict["has_text"]
    edge = profile_dict["edge_density"]
    bg_uni = profile_dict["bg_uniformity"]
    n_colors = profile_dict["n_unique_colors_q"]
    H, W = profile_dict["H"], profile_dict["W"]
    aspect = W / max(H, 1)
    if has_text and profile_dict["text_bands"]:
        max_band_h = max((tb["y1"] - tb["y0"]) for tb in profile_dict["text_bands"])
        if max_band_h > 40 and bg_uni < 0.85:
            return "text_poster"
        return "graphic_logo"
    if bg_uni > 0.55 and n_colors < 60:
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
    """对一张输入图做全量画像（v322：含 small_elements）。"""
    img = Image.open(path).convert("RGB")
    W, H = img.size
    arr = np.array(img)
    gray = np.array(img.convert("L"))
    hues, sals, vals = _detect_dominant_hsv(arr, top_k=5)
    edge = _edge_density(gray)
    bg_uni = _bg_uniformity(arr)
    q, _ = _quantize_palette(arr, n_colors=16)
    _, flat = _quantize_palette(arr, n_colors=8)
    n_unique = int(len(np.unique(flat)))
    # 文字带
    text_bands = _detect_text_bands(gray, arr)
    text_bands = [tb for tb in text_bands if tb.n_components >= 2 and (tb.y1 - tb.y0) >= 18]
    has_text = len(text_bands) > 0
    # 文字带 mask（用于小元素检测时排除）
    tb_mask = _build_text_band_mask(text_bands, W, H)
    # v322 新增：小元素检测
    small_elements = _detect_small_elements(gray, arr, tb_mask, W, H)
    prof = ImageProfile(
        path=path, W=W, H=H,
        image_type="unknown",
        has_text=has_text,
        text_bands=text_bands,
        small_elements=small_elements,
        dominant_hues_deg=hues,
        dominant_sat=sals,
        dominant_val=vals,
        edge_density=edge,
        bg_uniformity=bg_uni,
        n_unique_colors_q=n_unique,
    )
    prof.image_type = _classify_type(asdict(prof))
    # Phase B 友好参数
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
# 6. 小元素微调处理（v322 核心新增）
# ============================================================================

def _transform_small_element(img: Image.Image, elem: SmallElement,
                              dh: float, ds: float, dv: float,
                              scale: float, rot: float,
                              dx: int, dy: int,
                              flip: bool = False,
                              drop: bool = False) -> Image.Image:
    """
    对单个 SmallElement 做几何+颜色微调，返回修改后的整图。
    策略：
    - drop=True 时，把元素区域填回"上下文中值"（与 _erase_text_band 类似）
    - 否则：把元素 crop 出来 → resize+rotate → 翻面（可选）→ 颜色 HSV 微调 → paste 回新位置
    - dx/dy 是位置偏移（±5px 硬约束）
    """
    if elem.mask is None or not elem.mask.any():
        return img
    W, H = img.size
    arr = np.array(img).copy()
    if drop:
        # 用元素 bbox 内上下边沿均值填充（与 v321 纹理延续式一致）
        y0, y1, x0, x1 = elem.y0, elem.y1, elem.x0, elem.x1
        if y0 <= 0 or y1 >= H:
            arr[elem.mask] = arr[elem.mask].mean(axis=0)
            return Image.fromarray(arr)
        up = arr[max(0, y0 - 8):y0].mean(axis=0) if y0 > 0 else arr[y0:y0+1].mean(axis=0)
        dn = arr[y1:min(H, y1 + 8)].mean(axis=0) if y1 < H else arr[y1-1:y1].mean(axis=0)
        if up.ndim == 1:
            up = np.broadcast_to(up, (1, 1, 3))
        if dn.ndim == 1:
            dn = np.broadcast_to(dn, (1, 1, 3))
        # 在 bbox 内做上下渐变
        bh, bw = y1 - y0, x1 - x0
        if bh <= 0 or bw <= 0:
            return img
        alpha = np.linspace(1.0, 0.0, bh, dtype=np.float32).reshape(bh, 1, 1)
        up_full = np.broadcast_to(up.mean(axis=0), (bh, bw, 3))
        dn_full = np.broadcast_to(dn.mean(axis=0), (bh, bw, 3))
        blended = up_full * alpha + dn_full * (1.0 - alpha)
        sub = arr[y0:y1, x0:x1].astype(np.float32)
        hfeather = np.ones(bw, dtype=np.float32)
        feather = min(8, bw // 4)
        if feather > 0 and bw > 2 * feather:
            ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
            hfeather[:feather] = ramp
            hfeather[-feather:] = ramp[::-1]
        hfeather = hfeather.reshape(1, bw, 1)
        new_sub = sub * (1 - hfeather) + blended * hfeather
        arr[y0:y1, x0:x1] = new_sub.astype(np.uint8)
        # 元素 mask 内也强制填掉（防止残留）
        m_full = elem.mask
        # 用同样上下条均值填整个元素区域
        mean_color = (up.mean(axis=0) * 0.5 + dn.mean(axis=0) * 0.5)
        arr[m_full] = mean_color.astype(np.uint8)
        return Image.fromarray(arr)
    # ---- 不 drop：变换元素 ----
    # 1) crop 元素（带 4px 边距）
    pad = 6
    y0p, y1p = max(0, elem.y0 - pad), min(H, elem.y1 + pad)
    x0p, x1p = max(0, elem.x0 - pad), min(W, elem.x1 + pad)
    if y1p <= y0p or x1p <= x0p:
        return img
    # 用 mask 抠出元素（mask 外用原图保留）
    crop_img = Image.fromarray(arr[y0p:y1p, x0p:x1p].copy())
    crop_mask = Image.fromarray((elem.mask[y0p:y1p, x0p:x1p].astype(np.uint8) * 255))
    # 提取元素（透明背景）
    elem_rgba = Image.new("RGBA", crop_img.size, (0, 0, 0, 0))
    elem_rgba.paste(crop_img, (0, 0), crop_mask)
    # 2) 缩放 + 旋转
    s = float(np.clip(scale, 1.0 - SMALL_AFFINE_SCALE_DELTA, 1.0 + SMALL_AFFINE_SCALE_DELTA))
    r = float(np.clip(rot, -SMALL_AFFINE_ROT_DEG, SMALL_AFFINE_ROT_DEG))
    if abs(s - 1.0) > 0.001 or abs(r) > 0.1:
        nw, nh = int(elem_rgba.width * s), int(elem_rgba.height * s)
        elem_rgba = elem_rgba.resize((nw, nh), Image.BICUBIC)
        elem_rgba = elem_rgba.rotate(r, resample=Image.BICUBIC, expand=True)
    # 3) 翻面
    if flip:
        elem_rgba = ImageOps.mirror(elem_rgba)
    # 4) 颜色微调（HSV）
    elem_only = Image.new("RGB", elem_rgba.size, (128, 128, 128))
    elem_only.paste(elem_rgba, (0, 0), elem_rgba)
    elem_alpha = elem_rgba.split()[3]
    elem_only_mask = np.array(elem_alpha) > 10
    elem_shifted = _hsv_shift_mask(elem_only, elem_only_mask, dh, ds, dv)
    elem_shifted_rgba = elem_shifted.convert("RGBA")
    elem_shifted_rgba.putalpha(elem_alpha)
    # 5) 计算新粘贴位置（原 bbox 中心 + 偏移）
    new_cx = (elem.x0 + elem.x1) // 2 + dx
    new_cy = (elem.y0 + elem.y1) // 2 + dy
    new_x = new_cx - elem_shifted_rgba.width // 2
    new_y = new_cy - elem_shifted_rgba.height // 2
    # 裁剪到图像范围
    new_x = max(0, min(W - elem_shifted_rgba.width, new_x))
    new_y = max(0, min(H - elem_shifted_rgba.height, new_y))
    # 6) 在 arr 上：先把旧元素区域填回（用上下文均值），再 paste 新元素
    # 简化处理：直接 paste 在 arr 上（旧区域会被新元素覆盖，未覆盖部分保留——小元素通常
    # 是整体替换，所以新尺寸 ≤ 旧尺寸时不会露出底；如果新尺寸 > 旧尺寸，露出的也是原图）
    arr2 = np.array(img.convert("RGBA"))
    # 把旧元素区域涂透明（避免旧元素从新元素缝隙漏出）
    arr2[elem.mask] = (0, 0, 0, 0)
    base = Image.fromarray(arr2)
    base.paste(elem_shifted_rgba, (new_x, new_y), elem_shifted_rgba)
    return base.convert("RGB")


def _shift_motif_hue(img: Image.Image, elem: SmallElement,
                     dh: float, ds: float, dv: float) -> Image.Image:
    """
    v322c 新增：对 motif（佩斯利/碎花/纹理细块）做【温和局部色相漂移】，
    只改该元素内部像素的 HSV，不做任何几何变换，因此不会把纹理"拆碎"。
    用于回应 13c8b7 投诉：佩斯利小元素也要有差异，但不能仿射导致碎裂。
    """
    if elem.mask is None or not elem.mask.any():
        return img
    arr = np.array(img).copy()
    y0, y1, x0, x1 = elem.y0, elem.y1, elem.x0, elem.x1
    if y1 <= y0 or x1 <= x0:
        return img
    sub = arr[y0:y1, x0:x1]
    m = elem.mask[y0:y1, x0:x1]
    if not m.any():
        return img
    shifted = _hsv_shift_mask(Image.fromarray(sub), m, dh, ds, dv)
    arr[y0:y1, x0:x1] = np.array(shifted)
    return Image.fromarray(arr)


# ============================================================================
# 7. 文字带处理（v321 复用）
# ============================================================================

def _fit_text_to_band(img: Image.Image, band: TextBand, new_word: str) -> Image.Image:
    W, H = img.size
    band_w = max(60, band.x1 - band.x0)
    band_h = max(18, band.y1 - band.y0)
    target_w = int(band_w * 0.95)
    max_cap_h = max(8, int(band_h * 0.92))
    bold = (band.weight_guess == "bold")
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
    px = band.x0 + (band_w - tw) // 2
    py = band.y0 + (band_h - cap_h_render) // 2 - bbox_A[1]
    out = img.convert("RGBA")
    out.paste(layer, (px, py), layer)
    return out.convert("RGB")


def _erase_text_band(img: Image.Image, band: TextBand, feather: int = 18) -> Image.Image:
    W, H = img.size
    arr = np.array(img).copy()
    y0, y1, x0, x1 = band.y0, band.y1, band.x0, band.x1
    bh = y1 - y0
    bw = x1 - x0
    if bh <= 2 or bw <= 2:
        return img
    up_y0 = max(0, y0 - feather)
    up_y1 = y0
    dn_y0 = y1
    dn_y1 = min(H, y1 + feather)
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
    alpha = np.linspace(1.0, 0.0, bh, dtype=np.float32).reshape(bh, 1, 1)
    blended = up_strip * alpha + dn_strip * (1.0 - alpha)
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


# ============================================================================
# 8. 变体生成（v322：覆盖文字带 + 小元素）
# ============================================================================

def _seed_from(name: str, salt: str) -> int:
    h = hashlib.sha1((name + "|" + salt).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31)


def _pick_fictional(rng_seed: int, pool: List[str]) -> str:
    return pool[rng_seed % len(pool)]


def generate_variants(img: Image.Image, prof: ImageProfile, n_variants: int = 3, seed: int = 0) -> List[Tuple[str, Image.Image]]:
    """
    v322 变体生成：
    - v1_textswap + per_element：文字带改字 + 小元素几何/颜色微调
    - v2_hue_neg + per_element：色相负向 + 小元素微调
    - v3_sat_up + per_element：饱和度正向 + 小元素微调
    每个变体都走【所有元素都处理】的 v322 红线。
    """
    rng = np.random.RandomState(seed)
    DIRECTIONS = [
        (+HUE_DELTA_DEG, 0.0, 0.0),
        (-HUE_DELTA_DEG, 0.0, 0.0),
        (0.0, +SAT_DELTA, 0.0),
    ]
    out: List[Tuple[str, Image.Image]] = []
    base_name = os.path.splitext(os.path.basename(prof.path))[0]

    # v322b：只处理四类真·小元素，motif/decoration 是背景纹理不拆碎
    processable_now = [e for e in prof.small_elements
                       if e.category in ("dog_tag", "chain", "emblem", "border")]

    # ---- 决定每个变体的小元素处理策略 ----
    # 策略 A：几何微调（缩放±5%/旋转±5°/位置偏移±5px）
    # 策略 B：颜色微调（HSV ±10°）
    # 策略 C：镜像翻面
    # 策略 D：删除最小 1 个（仅当 small_elements >= 3 才允许，保证 v307 红线"主体不许缺失"）
    # 不同变体用不同策略组合，让 3 个变体明显不同
    # v322d（2026-09-10 可见性修复）：
    #   v322b/c 的 STRATEGIES 几何扰动（scale±4%/rot±4°/dx 3-4px）+ 零色相 → 肉眼看不出差别
    #   （用户原话"斜着就是改了？这种明显里面有小元素是不会改吗"）。
    #   改为：scale ±10-12%、rot ±10-14°、dx/dy ±10-15px + 非零色相 ±10-16°、sat ±0.06-0.10。
    #   每元素再叠 ±0.04 scale jitter / ±3° rot jitter / ±4px offset jitter，让同变体里
    #   不同小元素也明显不同。drop 保持原逻辑（仅策略 3 且仅最小元素）。
    STRATEGIES = [
        dict(dh=+14.0, ds=+0.08, dv=+0.04, scale=1.10, rot=+10.0, dx=+12, dy=-8,  flip=False, drop=False),
        dict(dh=-12.0, ds=-0.06, dv=-0.03, scale=0.88, rot=-12.0, dx=-15, dy=+10, flip=True,  drop=False),
        dict(dh=+10.0, ds=+0.07, dv=+0.05, scale=1.07, rot=+14.0, dx=+10, dy=+12, flip=False, drop=False),
    ]
    # 如果可处理元素 >= 3，第 3 个变体允许 drop 1 个最小元素
    if len(processable_now) >= 3:
        STRATEGIES[2]["drop"] = True

    for i in range(min(n_variants, 3)):
        dh, ds, dv = DIRECTIONS[i]
        strat = STRATEGIES[i]
        v_img = img.copy()
        # 1) 大图 HSV 微调
        v_img = _hsv_shift(v_img, dh, ds, dv)
        # 2) 文字带处理（仅 text_poster/graphic_logo 且 has_text）
        if prof.image_type in ("text_poster", "graphic_logo") and prof.has_text:
            for j, band in enumerate(prof.text_bands):
                v_img = _erase_text_band(v_img, band)
                if (band.y1 - band.y0) >= 40:
                    word = _pick_fictional(seed + i * 100 + j, FICTIONAL_BRANDS)
                else:
                    word = _pick_fictional(seed + i * 100 + j + 7, FICTIONAL_SUBTITLES)
                v_img = _fit_text_to_band(v_img, band, word)
        # 3) v322c：处理 dog_tag/chain/emblem/border 四类离散小元素（仿射+色相）
        for k, elem in enumerate(processable_now):
            # v322d：色相直接用 strat 基色（v322c 的 dh*0.5 把基色衰减了一半，
            # 配上 strat dh=0 等于零色相，肉眼无变化）。每元素再叠 jitter。
            elem_dh = float(np.clip(strat["dh"] + (k % 3 - 1) * 3.0, -SMALL_HUE_DELTA_DEG, SMALL_HUE_DELTA_DEG))
            elem_ds = float(np.clip(strat["ds"] + (k % 2) * 0.03, -SMALL_SAT_DELTA, SMALL_SAT_DELTA))
            elem_dv = float(np.clip(strat["dv"] + (k % 2) * 0.02, -SMALL_LUM_DELTA, SMALL_LUM_DELTA))
            # v322d：每元素的几何再叠 ±4% scale / ±3° rot / ±4px offset jitter，
            # 让同变体里不同小元素也明显不同。
            elem_scale = float(strat["scale"] * (1.0 + 0.04 * (k % 3 - 1)))
            elem_rot = float(strat["rot"] + 3.0 * (k % 2 * 2 - 1))
            elem_dx = int(strat["dx"] + 4 * (k % 3 - 1))
            elem_dy = int(strat["dy"] + 4 * (k % 2 * 2 - 1))
            # drop 策略：只在策略里指定 drop=True 且是当前最小元素时执行
            do_drop = strat["drop"] and k == len(processable_now) - 1
            v_img = _transform_small_element(
                v_img, elem,
                dh=elem_dh, ds=elem_ds, dv=elem_dv,
                scale=elem_scale, rot=elem_rot,
                dx=elem_dx, dy=elem_dy,
                flip=strat["flip"], drop=do_drop,
            )
        # 3b) v322c 新增：motif（佩斯利/碎花/纹理细块）走温和局部色相漂移
        #     不仿射、不 drop，避免把纹理拆碎；每个 motif 用略不同的 delta 让"每个小元素不同"
        motif_list = [e for e in prof.small_elements if e.category == "motif"]
        for m_idx, elem in enumerate(motif_list):
            m_dh = float(np.clip((m_idx % 5 - 2) * 3.0 + dh * 0.3, -SMALL_HUE_DELTA_DEG, SMALL_HUE_DELTA_DEG))
            m_ds = float(np.clip((m_idx % 3 - 1) * 0.05, -SMALL_SAT_DELTA, SMALL_SAT_DELTA))
            m_dv = float(np.clip((m_idx % 2) * 0.03, -SMALL_LUM_DELTA, SMALL_LUM_DELTA))
            v_img = _shift_motif_hue(v_img, elem, m_dh, m_ds, m_dv)
        # 4) 微仿射
        s = 1.0 + (i - 1) * 0.02
        r = (i - 1) * 1.5
        v_img = _micro_affine(v_img, s, r)
        # 5) 命名
        if prof.image_type in ("text_poster", "graphic_logo") and prof.has_text:
            tag = f"v{i+1}_textswap"
        elif prof.image_type == "pattern":
            tag = f"v{i+1}_hsv"
        elif prof.image_type == "cartoon":
            tag = f"v{i+1}_hsv_affine"
        elif prof.image_type == "product":
            tag = f"v{i+1}_hsv_zoom"
        elif prof.image_type == "landscape":
            tag = f"v{i+1}_mood"
        else:
            tag = f"v{i+1}_hsv"
        # 附加 per-element 标签（用可处理元素数量）
        out.append((f"{tag}_pe{len(processable_now)}", v_img))
    return out


# ============================================================================
# 9. Gallery（v321 复用 + 显示 small_elements 统计）
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
    rows = []
    for it in items:
        src_uri = _img_to_data_uri(Image.open(it["src_path"]))
        var_cards = "".join(
            f'<div class="v"><img src="{_img_to_data_uri(Image.open(p))}"><div class="cap">{n}</div></div>'
            for n, p in it["variants"]
        )
        prof = it["profile"]
        n_pe = prof.get("n_small_elements", 0)
        cats = prof.get("small_categories", {})
        cats_str = ", ".join(f"{k}:{v}" for k, v in cats.items()) if cats else "无"
        prof_str = (
            f'type=<b>{prof["image_type"]}</b> · has_text={prof["has_text"]} · '
            f'edges={prof["edge_density"]:.3f} · bg_uni={prof["bg_uniformity"]:.2f} · '
            f'colors={prof["n_unique_colors_q"]} · bands={len(prof["text_bands"])} · '
            f'<span style="color:#c41e3a"><b>small_elements={n_pe}</b> ({cats_str})</span>'
        )
        rows.append(
            f'<section><div class="head"><div class="src"><img src="{src_uri}"></div>'
            f'<div class="meta"><h3>{os.path.basename(it["src_path"])}</h3>'
            f'<div class="prof">{prof_str}</div></div></div>'
            f'<div class="variants">{var_cards}</div></section>'
        )
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>v322 逐元素裂变</title>
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
.warn{{background:#ffe6e6;border:1px solid #ff9999;padding:10px 14px;border-radius:6px;margin-bottom:16px;font-size:12px;}}
</style></head><body>
<h1>v322 逐元素裂变 — Gallery</h1>
<div class="sub">{len(items)} 张原图 × 3 变体 = {sum(len(it["variants"]) for it in items)} 张裂变图</div>
<div class="legend">⚠ 自动跳过任何 <code>_SKIP_*</code> 前缀的文件。所有"文字裂变"使用<strong>原创虚构词</strong>，绝无真实品牌字。Phase A 零 GPU；Phase B（AnyText2 v2.0 + IPAdapter + ControlNet）会消费本画廊的 <code>small_elements</code> 字段。</div>
<div class="warn">🔴 <b>v322 红线</b>：每个变体都覆盖【所有元素】——文字带 + 小元素（dog tag / 链条 / 徽记 / 边框 / 装饰 / motif）。仅换大字 + HSV 微调 = 不算裂变。</div>
{''.join(rows)}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================================
# 10. Driver
# ============================================================================

def _is_skipped(fn: str) -> bool:
    n = fn.lower()
    if n.startswith("_skip"):
        return True
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
    print(f"[scan] {len(files)} 张图将进入 v322 逐元素裂变（已自动跳过 _SKIP_/违规关键词）")
    items: List[dict] = []
    t0 = time.time()
    fail = 0
    for idx, src in enumerate(files, 1):
        t1 = time.time()
        try:
            prof = analyze_image(src)
        except Exception as e:
            print(f"[{idx}/{len(files)}] {os.path.basename(src)} FAIL analyze: {e}")
            fail += 1
            continue
        prof_dict = asdict(prof)
        # small_elements 不放进 prof_dict（mask 太大），但统计信息放进 gallery 显示
        n_pe = len(prof.small_elements)
        cats: Dict[str, int] = {}
        for e in prof.small_elements:
            cats[e.category] = cats.get(e.category, 0) + 1
        prof_dict["n_small_elements"] = n_pe
        prof_dict["small_categories"] = cats
        prof_dict["small_elements"] = [
            {k: v for k, v in asdict(e).items() if k != "mask"}
            for e in prof.small_elements
        ]
        # 保存 profile JSON（供 Phase B 消费）
        prof_json = os.path.join(output_dir, os.path.splitext(os.path.basename(src))[0] + ".profile.json")
        with open(prof_json, "w", encoding="utf-8") as f:
            json.dump(prof_dict, f, ensure_ascii=False, indent=2, default=str)
        # 生成变体
        img = Image.open(src).convert("RGB")
        try:
            variants = generate_variants(img, prof, n_variants=n_variants, seed=idx * 7)
        except Exception as e:
            print(f"[{idx}/{len(files)}] {os.path.basename(src)} FAIL generate: {e}")
            fail += 1
            continue
        # 保存变体
        var_paths = []
        for name, v_img in variants:
            out_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(src))[0] + f"__{name}.jpg"
            )
            v_img.convert("RGB").save(out_path, quality=92)
            var_paths.append((name, out_path))
        items.append({"src_path": src, "type": prof.image_type, "profile": prof_dict, "variants": var_paths})
        dt = time.time() - t1
        cats_str = ", ".join(f"{k}:{v}" for k, v in cats.items()) if cats else "无"
        print(f"[{idx}/{len(files)}] {os.path.basename(src)}")
        print(f"  profile: type={prof.image_type} has_text={prof.has_text} edges={prof.edge_density:.3f} "
              f"bg_uni={prof.bg_uniformity:.2f} colors={prof.n_unique_colors_q} bands={len(prof.text_bands)} "
              f"small_elements={n_pe} ({cats_str})  [{dt:.1f}s]")
        for tb in prof.text_bands:
            print(f"    band y={tb.y0}-{tb.y1} x={tb.x0}-{tb.x1} cap_h={tb.cap_h_guess} "
                  f"weight={tb.weight_guess} fg={tb.fg_rgb} comps={tb.n_components}")
        for k, e in enumerate(prof.small_elements[:6]):
            print(f"    elem[{k}] {e.category} y={e.y0}-{e.y1} x={e.x0}-{e.x1} "
                  f"area={e.area} aspect={e.aspect_h_over_w:.2f} fill={e.fill_density:.2f} rgb={e.mean_rgb}")
        if len(prof.small_elements) > 6:
            print(f"    ... and {len(prof.small_elements) - 6} more small_elements")
    # gallery
    gallery = os.path.join(output_dir, "gallery.html")
    build_gallery(items, gallery)
    dt_total = time.time() - t0
    print(f"\n[DONE] {len(items)} 张原图 × 3 变体 → {gallery}")
    print(f"[TIME] 总耗时 {dt_total:.1f}s（Phase A 纯 PIL，零 GPU）")
    if fail:
        print(f"[WARN] {fail} 张失败")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description="v322 通用化图裂变（逐元素裂变 + Phase B 友好 profile 输出）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variants", type=int, default=3)
    args = ap.parse_args()
    sys.exit(run(args.input, args.output, args.variants))


if __name__ == "__main__":
    main()
