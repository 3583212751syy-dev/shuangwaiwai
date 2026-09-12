"""styles/base.py — 所有风格模块共享的工具函数。

这里只放"与具体风格无关"的底层能力：图像读写、ComfyUI 调用封装、LAB 颜色锁、
LaMa 抹字封装、按原排版重绘文字（replace_text_plan）、字体度量。
风格专属逻辑一律放在各自模块，不进 base。
"""
from __future__ import annotations
import math, os, sys, subprocess, json, shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from skimage.color import rgb2lab, lab2rgb

# 项目根（styles/ 的上两级：image-fission/）
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
FISSION_CLI = SRC / "fission.py"
COMFYUI_URL = "http://127.0.0.1:8188"

# 字体路径（Windows 无 Bodoni/Didot，用 OFL 替代；Black 版 Playfair 损坏，用 Bold）
FONTS = {
    "blackopsone": ROOT / "fonts" / "BlackOpsOne-Regular.ttf",   # military / stencil display
    "playfair":    ROOT / "fonts" / "PlayfairDisplay-Bold.ttf",  # Didone serif (BACARDÍ 类)
    "metal":       ROOT / "fonts" / "MetalMania-Regular.ttf",    # spiky gothic metal (ARCHOR 类)
    "denim":       ROOT / "fonts" / "LeagueSpartan-Black.ttf",   # bold blocky (UPCY 类)
}

# 把 src 加入 sys.path 以便复用已有工具
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_fission_cli(args: list[str], cwd: str | None = None) -> int:
    """调用现有 fission.py CLI（本地 ComfyUI 出图）。返回 exit code。"""
    cmd = [str(VENV_PY), str(FISSION_CLI), *args]
    print(f"[base] run_fission_cli: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd)


def comfyui_ready(timeout_s: int = 5) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=timeout_s) as r:
            return r.status == 200
    except Exception:
        return False


def lab_color_lock(img: Image.Image, ref: Image.Image, alpha: float = 0.85) -> Image.Image:
    """真正 Reinhard LAB 颜色锁：把 img 的色族迁移到 ref（保留 ref 色族 + 部分 img 纹理）。

    alpha=1.0 完全锁成 ref 色；alpha<1 允许轻微色族内偏移（让裂变差异肉眼可见）。
    使用 skimage rgb2lab/lab2rgb 在感知均匀空间做均值/方差匹配，避免 RGB 逐通道匹配导致的色相漂移。
    """
    a_rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    b_rgb = np.asarray(ref.convert("RGB"), dtype=np.float64) / 255.0

    def stats(x):
        return x.mean(axis=(0, 1)), x.std(axis=(0, 1)) + 1e-6

    a = rgb2lab(a_rgb)
    b = rgb2lab(b_rgb)
    ma, sa = stats(a)
    mb, sb = stats(b)
    matched = (a - ma) * (sb / sa) + mb
    matched = np.clip(matched, [0, -128, -128], [100, 127, 127])
    blended = alpha * matched + (1 - alpha) * a
    blended = np.clip(blended, [0, -128, -128], [100, 127, 127])
    rgb = (lab2rgb(blended) * 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


# ---------------------------------------------------------------------------
# 文字原位替换：LaMa 抹旧字 + PIL 按原字体/字号/位置重画新词
# ---------------------------------------------------------------------------

def _lama_module():
    """惰性加载 v268_lama_clean（含本地 Big-LaMa）。"""
    import v268_lama_clean as lc
    return lc


def _capH_ratio(fp: Path, probe_size: int = 200) -> float:
    f = ImageFont.truetype(str(fp), probe_size)
    bb = f.getbbox("A")
    return (bb[3] - bb[1]) / probe_size


def _fit_font_size(word: str, fp: Path, target_capH: int, band_w: int, scale: int = 4) -> int:
    """返回在 4x 超采样画布上应使用的字体字号（显示 capH≈target_capH 且宽度不溢出）。"""
    ratio = _capH_ratio(fp)
    S = int(target_capH * scale / ratio * 0.95)
    font = ImageFont.truetype(str(fp), S)
    tw = font.getlength(word)
    disp_w = tw / scale
    if disp_w > band_w * 0.96:
        S = int(S * (band_w * 0.96 * scale) / tw)
    return max(12, S)


def _render_word(img: Image.Image, word: str, bbox, font_key: str, color) -> Image.Image:
    """在 img 的 bbox 内按原排版（同位置/同高度）渲染新词（4x 超采样后缩回，整词居中）。

    对超大图（如 pinterest6 3543×4961）做超采样上限保护，避免一次性构造过大画布吃掉内存。
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    fp = FONTS.get(font_key, FONTS["blackopsone"])
    target_capH = int(bh * 0.78)
    scale = 4
    # 超采样保护：宽度超采样后超过 ~14000px 时降档，控制瞬时内存
    if img.width * scale > 14000:
        scale = max(2, 14000 // img.width)
    S = _fit_font_size(word, fp, target_capH, bw, scale)
    font = ImageFont.truetype(str(fp), S)
    W, H = img.width, img.height
    canvas = Image.new("RGBA", (W * scale, H * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    tw = d.textlength(word, font=font)
    cx = (x1 + x2) / 2 * scale
    cy = (y1 + y2) / 2 * scale
    capH = S * _capH_ratio(fp)
    start_x = cx - tw / 2
    top_y = cy - capH / 2
    d.text((start_x, top_y), word, font=font, fill=tuple(color) + (255,))
    canvas = canvas.resize((W, H), Image.LANCZOS)
    out = img.convert("RGBA")
    out.alpha_composite(canvas)
    return out.convert("RGB")


def replace_text_plan(img: Image.Image, plan: list[dict], dilate: int = 12) -> Image.Image:
    """按 text_plan 逐条原位改写文字：LaMa 抹旧字 → PIL 重画新词。

    plan 每条: {"bbox":[x1,y1,x2,y2], "word":str, "font":"blackopsone|playfair|metal|denim",
                "color":[r,g,b], "dilate":int(可选)}。坐标为原图像素坐标。
    禁止矩形色块遮盖——LaMa 只抹文字区、保周围纹理，PIL 直接贴新字。

    内存安全：对超大图（如 pinterest6 3543×4961）只裁剪文字 bbox 周边区域做 LaMa，
    不在整张大图上跑 inpaint，避免内存峰值把进程打挂。
    """
    if not plan:
        return img
    lc = _lama_module()
    out = img.convert("RGB")
    W, H = out.size
    for item in plan:
        bbox = item.get("bbox")
        word = (item.get("word") or "").strip()
        if not bbox or not word:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        item_dilate = int(item.get("dilate", dilate))
        # 裁剪框：以文字 bbox 为中心，上下各留 margin（取文字高度的 ~0.8 或至少 60px）
        bh = y2 - y1
        margin = max(int(bh * 0.8), 60)
        cx1 = max(0, x1 - margin)
        cy1 = max(0, y1 - margin)
        cx2 = min(W, x2 + margin)
        cy2 = min(H, y2 + margin)
        out_crop = out.crop((cx1, cy1, cx2, cy2))
        # 文字 bbox 相对裁剪框的坐标
        bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
        # 1) 构建文字区 mask（白=去除），膨胀覆盖笔画边缘
        mask = Image.new("L", out_crop.size, 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rectangle([bx1, by1, bx2, by2], fill=255)
        mask = mask.filter(ImageFilter.MaxFilter(item_dilate * 2 + 1))
        # 2) 扩散 inpaint 只抹裁剪框内文字区（背景由扩散重建，不用色块盖字）
        try:
            cleaned_crop = lc.diffusion_inpaint(out_crop, mask)
        except Exception as e:
            print(f"[base] diffusion_inpaint failed for {bbox}: {e}, 回退 LaMa")
            try:
                cleaned_crop = lc.lama_inpaint(out_crop, mask, removal_strength=235, edge_smoothness=6)
            except Exception as e2:
                print(f"[base] lama fallback also failed: {e2}")
                cleaned_crop = out_crop
        # 3) 按原排版重画新词（坐标相对裁剪框）
        cleaned_crop = _render_word(cleaned_crop, word, [bx1, by1, bx2, by2],
                                    item.get("font", "blackopsone"),
                                    item.get("color", [0, 0, 0]))
        # 4) 贴回原图（裁剪框周边留足 buffer，LaMa 边缘过渡无缝）
        out.paste(cleaned_crop, (cx1, cy1))
    return out


# ---------------------------------------------------------------------------
# 文字原位替换 v2：笔画级 mask + 弧形文字 + 原字材质迁移
# ---------------------------------------------------------------------------

def _detect_text_mask(crop_arr: np.ndarray, bbox) -> np.ndarray:
    """在 crop 的 bbox 区域内检测旧字笔画（返回 0-255 mask）。

    综合三种线索：
      1) Sobel 梯度（强边缘）
      2) Otsu 亮度分割（黑字/白字/彩字与背景分离）
      3) 全局亮度异常值兜底（极端黑/白/彩色字）
    bbox 会外扩 5%，确保略有外溢的旧字笔画也被包含。
    """
    from scipy import ndimage
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = crop_arr.shape[:2]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad = max(8, int(min(bw, bh) * 0.05))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((H, W), dtype=np.uint8)
    roi = crop_arr[y1:y2, x1:x2].astype(np.float32)
    lum = 0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]
    # 1) Sobel 梯度
    gx = ndimage.sobel(lum, axis=1)
    gy = ndimage.sobel(lum, axis=0)
    grad = np.hypot(gx, gy)
    thr_grad = np.percentile(grad, 75)
    edge = grad > thr_grad
    # 2) Otsu 分割（要求双峰，用双峰性过滤避免纯背景误检）
    from skimage.filters import threshold_otsu
    from skimage.exposure import histogram
    otsu_mask = np.zeros_like(lum, dtype=bool)
    try:
        # 双峰性：计算灰度直方图显著峰值数；少于 2 个峰值时放弃 Otsu
        hist, bin_centers = histogram(lum.astype(np.uint8), nbins=64)
        peaks = (hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])
        if peaks.sum() >= 2:
            thr = threshold_otsu(lum)
            # 文字通常只占 bbox 的一小部分，Otsu 分割后取面积较小的那类作为文字。
            # 若两类面积过于接近，再按梯度投票：与边缘 mask 重合度高的为文字。
            dark = lum < thr
            light = lum > thr
            if dark.sum() == 0:
                otsu_mask = light
            elif light.sum() == 0:
                otsu_mask = dark
            else:
                ratio_dark = dark.sum() / (dark.sum() + light.sum())
                if 0.35 < ratio_dark < 0.65:
                    # 接近 1:1，按与梯度边缘的重合度判断
                    edge_small = ndimage.sobel(lum) > np.percentile(ndimage.sobel(lum), 80)
                    ed = edge_small & dark
                    el = edge_small & light
                    score_dark = ed.sum() / max(1, dark.sum())
                    score_light = el.sum() / max(1, light.sum())
                    otsu_mask = dark if score_dark > score_light else light
                else:
                    otsu_mask = dark if dark.sum() < light.sum() else light
    except Exception:
        pass
    # 3) 全局亮度异常值兜底
    bg = np.percentile(lum, 50)
    diff = np.abs(lum - bg)
    mad = np.median(np.abs(diff - np.median(diff))) + 1e-6
    outlier = diff > max(mad * 2.5, 12.0)
    m = (edge | otsu_mask | outlier).astype(np.uint8) * 255
    # closing：多次膨胀把笔画填实，再轻微腐蚀
    m = ndimage.binary_dilation(m > 0, iterations=6).astype(np.uint8) * 255
    m = ndimage.binary_erosion(m > 0, iterations=1).astype(np.uint8) * 255
    full = np.zeros((H, W), dtype=np.uint8)
    full[y1:y2, x1:x2] = m
    return full


def _letter_stats(crop_arr: np.ndarray, mask: np.ndarray):
    """返回 (letter_mean_rgb, bg_mean_rgb)。

    对 mask 大幅膨胀后再取 mean，确保把旧字实心笔画内部也纳入采样，
    避免只采到边缘像素导致材质色偏暗/偏灰。
    """
    from scipy import ndimage
    mask_b = mask > 127
    if mask_b.sum() < 8:
        return None, None
    # 膨胀填充字内部，再腐蚀回笔画轮廓附近，尽量只保留字像素
    solid = ndimage.binary_dilation(mask_b, iterations=10)
    solid = ndimage.binary_erosion(solid, iterations=2)
    if solid.sum() < 8:
        solid = mask_b
    letter_mean = crop_arr[solid].mean(axis=0)
    # 背景：取 crop 四边 10px 边带的中位数
    H, W = crop_arr.shape[:2]
    edge = np.zeros((H, W), bool)
    edge[:10, :] = True
    edge[-10:, :] = True
    edge[:, :10] = True
    edge[:, -10:] = True
    bg_mean = np.median(crop_arr[edge].reshape(-1, 3), axis=0)
    return letter_mean, bg_mean


def _emboss_alpha(alpha: np.ndarray, strength: float = 0.12) -> np.ndarray:
    """根据 alpha 边缘生成简单浮雕（高光/阴影），让新字有原字的立体材质感。"""
    hi = np.roll(alpha, -1, axis=0)
    hi = np.roll(hi, -1, axis=1)
    sh = np.roll(alpha, 1, axis=0)
    sh = np.roll(sh, 1, axis=1)
    emboss = (hi - sh) * strength
    return emboss


def _render_word_alpha(img_size: tuple[int, int], word: str, bbox, font_key: str) -> np.ndarray:
    """渲染新词 alpha 蒙版（0-1 float），尺寸与 img_size 一致，按 bbox 居中。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    fp = FONTS.get(font_key, FONTS["blackopsone"])
    target_capH = int(bh * 0.78)
    W, H = img_size
    scale = 4
    if W * scale > 14000:
        scale = max(2, 14000 // W)
    S = _fit_font_size(word, fp, target_capH, bw, scale)
    font = ImageFont.truetype(str(fp), S)
    canvas = Image.new("L", (W * scale, H * scale), 0)
    d = ImageDraw.Draw(canvas)
    tw = d.textlength(word, font=font)
    cx = (x1 + x2) / 2 * scale
    cy = (y1 + y2) / 2 * scale
    capH = S * _capH_ratio(fp)
    start_x = cx - tw / 2
    top_y = cy - capH / 2
    d.text((start_x, top_y), word, font=font, fill=255)
    canvas = canvas.resize((W, H), Image.LANCZOS)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _render_arc_word_alpha(img_size: tuple[int, int], word: str, bbox, font_key: str,
                           arc: str = "up", arc_center=None, arc_radius=None,
                           start_angle: float = 225, end_angle: float = 315) -> np.ndarray:
    """用 src/arc_text.py 绘制向上弧线文字，输出 alpha 蒙版（0-1），用于 6978 顶弧。"""
    import arc_text
    W, H = img_size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    fp = FONTS.get(font_key, FONTS["blackopsone"])

    # 圆心/半径：允许 item 显式传入（原图坐标），否则按 bbox 推导
    if arc_center is not None:
        cx, cy = float(arc_center[0]), float(arc_center[1])
    else:
        cx = (x1 + x2) / 2.0
        # 让弧顶落在 bbox 上半，侧边落在 bbox 底边附近
        r = max(bh * 0.8 / (1.0 - math.cos(math.radians(45))), bw / 1.6)
        cy = y1 + bh * 0.35 + r
    if arc_radius is not None:
        r = float(arc_radius)
    elif arc_center is None:
        r = max(bh * 0.8 / (1.0 - math.cos(math.radians(45))), bw / 1.6)

    # 字号：先按 bbox 高度取，再用 fit_arc_text_width 在弧范围内回退
    font_size = int(bh * 0.30)
    available_len = r * math.radians((end_angle - start_angle) % 360)
    while font_size >= 12:
        arc_len = arc_text.fit_arc_text_width(word, str(fp), font_size, r)
        if arc_len <= available_len * 0.85:
            break
        font_size = int(font_size * 0.92)

    tmp = Image.new("RGB", (W, H), (0, 0, 0))
    arc_text.draw_arc_text(tmp, word, str(fp), font_size, (255, 255, 255),
                           (cx, cy), r, start_angle, end_angle, char_spacing_px=2)
    gray = np.asarray(tmp.convert("L"), dtype=np.float32) / 255.0
    return gray


def _highpass_texture(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """提取高频纹理（减去局部低频平均），保留材质细节但去掉背景/字母的绝对颜色。"""
    from scipy.ndimage import gaussian_filter
    low = gaussian_filter(arr, sigma=(sigma, sigma, 0))
    return arr.astype(np.float32) - low


def _compose_material(cleaned_arr: np.ndarray, texture_src: np.ndarray,
                      alpha: np.ndarray, letter_mean: np.ndarray,
                      material_weight: float = 0.70, emboss_strength: float = 0.0) -> np.ndarray:
    """把已清洗干净背景的高频材质纹理迁移到 alpha 形状上，再与背景合成。

    用 cleaned_arr（LaMa 后）作为 texture_src，避免把旧字形状/笔画带入新字材质。
    新字底色为 letter_mean，叠加高频纹理营造原字材质感。
    """
    texture = _highpass_texture(texture_src, sigma=2.0)
    letter = letter_mean + texture * material_weight
    alpha3 = alpha[..., None]
    out = cleaned_arr * (1 - alpha3) + letter * alpha3
    if emboss_strength > 0:
        emboss = _emboss_alpha(alpha, emboss_strength)
        out += emboss[..., None] * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def replace_text_plan_v2(img: Image.Image, plan: list[dict], dilate: int = 8,
                         use_material: bool = True) -> Image.Image:
    """文字替换 v2：笔画级 mask 精准擦除 + 弧形文字 + 原字材质迁移。

    plan 每条可额外包含：
      - "arc": "up"  表示沿向上弧线排版（用于 6978 顶弧）
      - "dilate": int  笔画 mask 膨胀量
      - "color": [r,g,b]  仅当材质采样失败时回退用
    背景 100% 不动，只替换文字像素。
    """
    if not plan:
        return img
    lc = _lama_module()
    out = img.convert("RGB")
    W, H = out.size
    for item in plan:
        bbox = item.get("bbox")
        word = (item.get("word") or "").strip()
        if not bbox or not word:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        item_dilate = int(item.get("dilate", dilate))
        bh = y2 - y1
        # margin：默认给 LaMa 足够上下文重建背景纹理；对超大字号/高对比度字
        # 可显式传小 margin 避免模型把原字“联想”回来。
        if "margin" in item:
            margin = int(item["margin"])
        else:
            margin = max(int(bh * 0.7), 40)
        cx1 = max(0, x1 - margin)
        cy1 = max(0, y1 - margin)
        cx2 = min(W, x2 + margin)
        cy2 = min(H, y2 + margin)
        crop = out.crop((cx1, cy1, cx2, cy2))
        original_crop = np.asarray(crop, dtype=np.float32)
        bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
        # 1) mask = 笔画级检测（含小范围外扩）后膨胀 + bbox 矩形兜底。
        #    bbox 兜底确保旧字带完整被 LaMa 擦除；它不是色块，LaMa 会用周围纹理
        #    自然重建。只当检测失败或旧字外溢时才真正依赖 bbox 兜底。
        detected = _detect_text_mask(original_crop, [bx1, by1, bx2, by2])
        from scipy import ndimage
        # 基础膨胀覆盖笔画边缘；dilate 参数再追加
        detected = ndimage.binary_dilation(detected > 0, iterations=6).astype(np.uint8) * 255
        if item_dilate > 6:
            detected = ndimage.binary_dilation(detected > 0, iterations=item_dilate - 6).astype(np.uint8) * 255
        bbox_mask = np.zeros_like(detected)
        bbox_mask[by1:by2, bx1:bx2] = 255
        # 优先只擦检测到的旧字笔画，最大程度保留文字带之间的原背景纹理。
        # 仅当检测面积极小（< 2% bbox）时才用 bbox 兜底，防止漏擦。
        bbox_area = (bbox_mask > 127).sum()
        detected_area = (detected > 127).sum()
        detected_ratio = detected_area / max(1, bbox_area)
        if detected_ratio >= 0.02:
            stroke_mask = detected
        else:
            stroke_mask = np.maximum(detected, bbox_mask)
        mask_pil = Image.fromarray(stroke_mask, mode="L")
        # 预判断材质权重，决定后续回填策略
        mat_w = float(item.get("material_weight", 0.70 if use_material else 0.0))
        emb = float(item.get("emboss", 0.0))
        # 2) 扩散 inpaint 擦除：先彻底擦除原文字（背景由扩散重建），绝不用色块盖字。
        removal_strength = float(item.get("removal_strength", 240))  # 保留字段，扩散 backend 暂未使用
        edge_smoothness = int(item.get("edge_smoothness", 4))
        try:
            cleaned_crop = lc.diffusion_inpaint(crop, mask_pil)
        except Exception as e:
            print(f"[base] diffusion_inpaint failed for {bbox}: {e}, 回退 LaMa")
            try:
                cleaned_crop = lc.lama_inpaint(crop, mask_pil, removal_strength=removal_strength, edge_smoothness=edge_smoothness)
            except Exception as e2:
                print(f"[base] lama fallback also failed: {e2}")
                cleaned_crop = crop
        cleaned_arr = np.asarray(cleaned_crop, dtype=np.float32)
        # 3) 渲染新词 alpha
        arc = item.get("arc")
        if arc == "up":
            # 弧线圆心常在 bbox 下方很远（如 6978 cy=820），必须在整图上渲染再裁剪，
            # 否则 crop 装不下圆心会导致弧字偏移/被裁。
            arc_center_full = item.get("arc_center")
            full_alpha = _render_arc_word_alpha((W, H), word, [x1, y1, x2, y2],
                                               item.get("font", "blackopsone"), arc="up",
                                               arc_center=arc_center_full,
                                               arc_radius=item.get("arc_radius"),
                                               start_angle=float(item.get("arc_start", 225)),
                                               end_angle=float(item.get("arc_end", 315)))
            alpha = full_alpha[cy1:cy2, cx1:cx2]
        else:
            alpha = _render_word_alpha(crop.size, word, [bx1, by1, bx2, by2],
                                       item.get("font", "blackopsone"))
        # 4) 材质迁移 or 平涂 fallback
        if use_material and mat_w > 0:
            # 材质采样只取笔画级 detected，避免 bbox 兜底区把背景色混进 letter_mean
            letter_mean, _ = _letter_stats(original_crop, detected)
            if letter_mean is None:
                letter_mean = np.array(item.get("color", [0, 0, 0]), dtype=np.float32)
            # 自动判断文字是否“平涂”：若旧字高频纹理很弱，降低材质权重
            texture = _highpass_texture(original_crop, sigma=2.0)
            mask_b = stroke_mask > 127
            if mask_b.sum() > 20:
                tex_std = texture[mask_b].reshape(-1, 3).std(axis=0).mean()
                if tex_std < 12.0:
                    mat_w = min(mat_w, 0.25)
            composed = _compose_material(cleaned_arr, original_crop, alpha, letter_mean,
                                         material_weight=mat_w, emboss_strength=emb)
        else:
            color = np.array(item.get("color", [0, 0, 0]), dtype=np.float32)
            composed = cleaned_arr * (1 - alpha[..., None]) + color * alpha[..., None]
            composed = np.clip(composed, 0, 255).astype(np.uint8)
        out.paste(Image.fromarray(composed, "RGB"), (cx1, cy1))
    return out


def get_image_cfg(cfg: dict, image_path: str) -> dict:
    """从 set.json 找到当前图对应的完整配置条目（按 path/filename 匹配）。"""
    target = str(image_path)
    for img in cfg.get("images", []):
        if img.get("path") == target or img.get("filename") in target:
            return img
    return {}


def get_text_plan_for(cfg: dict, image_path: str) -> list[dict]:
    """从 set.json 找到当前图对应的 text_plan（按 path/filename 匹配）。"""
    return get_image_cfg(cfg, image_path).get("text_plan") or []


def save_variant(img: Image.Image, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    img.save(p, "JPEG", quality=92)
    print(f"[base] saved variant -> {p}")
    return p
