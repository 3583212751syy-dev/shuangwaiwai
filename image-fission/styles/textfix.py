"""styles/textfix.py — 原位文字裂变引擎 (v328)

用户硬规则（2026-09-14，反复强调，不可破坏）：
  🔴 A. **严禁任何遮挡方式改字**：不许矩形色块盖字、不许色块填充、不许矩形 bbox 掩膜。
        擦除掩膜必须是**笔画级**（跟着字形轮廓走），否则 LaMa 会把整块填平 → 肉眼就是"拿色块遮挡"。
  🔴 B. 新词必须沿用**原字体 / 原字号(cap 高) / 原排版位置**（同 box 居中，cap 高对齐），不许突兀。
  🔴 C. 颜色必须保留原图（背景、主体、配色一律不动；新字颜色直接采自旧字像素中位数）。

流程：笔画掩膜 -> LaMa 只擦笔画(背景纹理自然重建) -> 同位置同高重画新词
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from . import base

FONTS = base.FONTS
_FP_CACHE: dict[str, float] = {}


# ---------------------------------------------------------------- 笔画掩膜
def stroke_mask(img: Image.Image, box, thr: float | None = None, mode: str = "lum",
                size: int = 53, contrast: float | None = None,
                direction: str = "dark", dilate: int = 6,
                exclude: np.ndarray | None = None, min_area: int = 60,
                pad: int = 40, fill: bool = True) -> np.ndarray:
    """在 box（外扩 pad）内提取**笔画级**掩膜（全图 bool）。

    mode='lum'  （默认，最稳）：亮度阈值分割。thr=None 时用 Otsu 自动找谷底。
        b78e60 实测：文字全在 lum<40，迷彩最暗 >50 → 干净可分。
    mode='shape'：黑帽/白帽形态学，只有「比周围暗/亮且厚度 < size」的目标中选。
        适用于文字与背景灰度重叠、但厚度差别大的场景。

    禁矩形掩膜 —— 掩膜必须跟着字形走，否则 LaMa 填平就是"色块遮挡"。
    """
    arr = np.asarray(img.convert("RGB"), np.float32)
    lum = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    H, W = lum.shape
    x1, y1, x2, y2 = [int(v) for v in box]
    sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
    sx2, sy2 = min(W, x2 + pad), min(H, y2 + pad)
    sub = lum[sy1:sy2, sx1:sx2]
    if sub.size == 0:
        return np.zeros((H, W), bool)
    if mode == "shape":
        if contrast is None:
            contrast = 0.32 * (np.percentile(sub, 78) - np.percentile(sub, 4))
        m = np.zeros_like(sub, bool)
        if direction in ("dark", "both"):
            m |= (ndi.grey_closing(sub, size=size) - sub) > contrast
        if direction in ("light", "both"):
            m |= (sub - ndi.grey_opening(sub, size=size)) > contrast
    else:
        t = float(thr) if thr is not None else _otsu(sub)
        m = (sub < t) if direction == "dark" else (sub > t)
    if fill:
        m = ndi.binary_fill_holes(m)
    if min_area > 0 and m.any():
        lab, n = ndi.label(m)
        if n:
            sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sizes >= min_area
            m = keep[lab]
    full = np.zeros((H, W), bool)
    full[sy1:sy2, sx1:sx2] = m
    if dilate > 0:
        full = ndi.binary_dilation(full, structure=_disk(dilate))
    if exclude is not None:
        full &= ~exclude
    return full


def _otsu(x: np.ndarray) -> float:
    from skimage.filters import threshold_otsu
    try:
        return float(threshold_otsu(x))
    except Exception:
        return float(np.median(x))


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def detect_lines(img: Image.Image, box, thr: float | None = None, mode: str = "lum",
                 size: int = 53, contrast: float | None = None,
                 direction: str = "dark", pad: int = 40, min_area: int = 60,
                 min_row_px: int = 4, gap: int = 12, min_h: int = 16,
                 exclude: np.ndarray | None = None):
    """宽松框内自动测行：返回 [(line_box, line_mask), ...]。

    line_box 由笔画掩膜的真实外接框给出（自校正手量误差），line_mask 是该行掩膜。
    exclude 用来剔除同区域里的主体轮廓（如狗牌链子），避免把行框撑高。
    """
    m = stroke_mask(img, box, thr=thr, mode=mode, size=size, contrast=contrast,
                    direction=direction, dilate=0, pad=pad, min_area=min_area,
                    fill=True, exclude=exclude)
    ys = np.where(m.any(axis=1))[0]
    if len(ys) == 0:
        return []
    rows = m.sum(axis=1)
    spans, cur = [], None
    for y in range(ys.min(), ys.max() + 1):
        if rows[y] >= min_row_px:
            cur = [y, y] if cur is None else [cur[0], y]
        elif cur is not None and y - cur[1] > gap:
            spans.append(cur); cur = None
    if cur is not None:
        spans.append(cur)
    out = []
    for y1, y2 in spans:
        sub = m[y1:y2 + 1]
        cols = np.where(sub.any(axis=0))[0]
        if len(cols) == 0:
            continue
        lx1, lx2 = int(cols.min()), int(cols.max()) + 1
        lm = np.zeros_like(m)
        lm[y1:y2 + 1, lx1:lx2] = True
        lm &= m
        if (y2 - y1 + 1) < min_h:
            continue
        out.append(((lx1, int(y1), lx2, int(y2) + 1), lm))
    return out


def ring_mask(w: int, h: int, center, radius: float, band: float) -> np.ndarray:
    """弧线文字用的圆环带掩膜（全图 bool）：半径 radius±band。"""
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - float(center[0]), yy - float(center[1]))
    return np.abs(d - radius) <= band


# ---------------------------------------------------------------- 擦除（LaMa，只擦笔画）
def _push_mean(arr: np.ndarray, mask: np.ndarray,
               src_allow: np.ndarray | None = None, win: int = 7,
               max_iter: int = 3000):
    """逐层向内推进的均值填充：每层取局部窗口内**已知**像素的均值。

    不做任何"猜测"（对比 LaMa 会凭空长出云块/色斑）：填充值全部来自真实边界像素，
    因此纯色/渐变/柔光背景能得到干净的延续。返回 (填充后图像, 未填到的残留mask)。
    """
    known = ~mask
    if src_allow is not None:
        known = known & src_allow
    k = int(win) | 1
    val = np.where(known[..., None], arr, 0.0).astype(np.float32)
    cnt = known.astype(np.float32)
    out = arr.copy()
    todo = mask.copy()
    for _ in range(int(max_iter)):
        c2 = ndi.uniform_filter(cnt, size=k, mode="constant")
        cand = todo & (c2 > 1e-6)
        if not cand.any():
            break
        s = np.stack([ndi.uniform_filter(val[..., c], size=k, mode="constant")
                      for c in range(3)], -1)
        avg = s / np.maximum(c2, 1e-6)[..., None]
        m = cand[..., None]
        out = np.where(m, avg, out)
        val = np.where(m, out, val)
        cnt = np.where(cand, 1.0, cnt)
        todo = todo & ~cand
    return out, todo


def diffuse_fill(img: Image.Image, mask: np.ndarray,
                 src_allow: np.ndarray | None = None, down: int = 4,
                 win: int = 7) -> Image.Image:
    """扩散填充（大区域首选）：降采样 -> 逐层均值推进 -> 升采样 -> 只回填 mask 内。

    对「大面积擦除 + 纯色/渐变背景」（金属海报黑底、单色渐变）最稳：
    不会像 LaMa 凭空长出蓝雾/云块，也不会像 nn 单点取样灌入亮点。
    """
    a = np.asarray(img.convert("RGB"), np.float32)
    H, W = mask.shape
    s = max(1, int(down))
    if s > 1:
        tw, th = max(16, W // s), max(16, H // s)
        small = img.resize((tw, th), Image.LANCZOS)
        sm = np.asarray(Image.fromarray((mask * 255).astype(np.uint8), "L")
                        .resize((tw, th), Image.NEAREST)) > 127
        sa = None
        if src_allow is not None:
            sa = np.asarray(Image.fromarray((src_allow * 255).astype(np.uint8), "L")
                            .resize((tw, th), Image.NEAREST)) > 127
    else:
        small, sm, sa = img, mask, src_allow
    sa_small = np.asarray(small, np.float32)
    filled_s, _ = _push_mean(sa_small, sm, src_allow=sa, win=win)
    filled_big = np.asarray(Image.fromarray(np.clip(filled_s, 0, 255).astype(np.uint8))
                            .resize((W, H), Image.BILINEAR), np.float32)
    out = a.copy()
    out[mask] = filled_big[mask]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def constant_fill(img: Image.Image, mask: np.ndarray, color=(0, 0, 0),
                  feather: int = 20) -> Image.Image:
    """定值填充 + 边缘羽化：整块用过 `color` 覆盖，边界 feather px 内平滑过渡。

    适用「背景本来就是纯色/近纯色（金属海报黑底）」的大面积擦除：没有任何模型猜测，
    边缘羽化让定值与周围渐变自然咬合，不会出现硬边或云块。
    """
    a = np.asarray(img.convert("RGB"), np.float32)
    d_in = ndi.distance_transform_edt(mask)
    d_out = ndi.distance_transform_edt(~mask)
    f = max(1.0, float(feather))
    alpha = np.clip((d_in - d_out) / f + 0.5, 0.0, 1.0)[..., None]
    col = np.array(color, np.float32)
    out = a * (1 - alpha) + col * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def nn_fill(img: Image.Image, mask: np.ndarray, smooth_px: float = 0.0,
            median_px: int = 0, src_allow: np.ndarray | None = None) -> Image.Image:
    """最近邻填充：mask 内每点取**最近的未遮蔽像素**的颜色（边界色向内延伸）。

    对「大facets 平坦块」类背景（迷彩碎片/色块/纯色渐变）比 LaMa 更自然：
    LaMa 会抹成一片平滑的雾（肉眼=遮挡块），nn 则让周围色块自然长出、纹理延续。
    - median_px：对填充结果取局部中值（>1 时启用）。**关键**：nn 单点取样会把边界上
      一个亮点/暗点整片灌进笔画内部（实测=白雾/灰雾 ghost），局部中值能压掉这种灌入。
    - src_allow：允许作为取样源的区域（None=全图）。例如弧字只能从**缎带带内**取样，
      否则会把带外的背景色拖进带内，缎带的白色被染成背景色。
    """
    a = np.asarray(img.convert("RGB"), np.float32)
    valid = ~mask
    if src_allow is not None:
        valid = valid & src_allow
    if not valid.any():                       # 兜底：无处取样时退回原样
        return img.convert("RGB")
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = a[tuple(idx)]
    if median_px and median_px > 1:
        k = int(median_px) | 1
        filled = np.stack([ndi.median_filter(filled[..., c], size=k) for c in range(3)], -1)
    out = a.copy()
    out[mask] = filled[mask]
    if smooth_px and smooth_px > 0:
        g = ndi.gaussian_filter(out, sigma=(smooth_px, smooth_px, 0))
        out[mask] = g[mask]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def biharm_fill(img: Image.Image, mask: np.ndarray, max_side: int = 900) -> Image.Image:
    """双调和扩散填充（解 Laplace）：用**边界值**平滑延拓，不含任何高频猜测。

    适合平滑渐变背景（纯色/渐变/柔光）。边界必须取**干净背景**（mask 需已吃掉原字的光晕），
    否则会把光晕的亮色往内扩散（nn 的典型失败）。
    """
    from skimage.restoration import inpaint_biharmonic
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    big = max(img.size)
    f = (max_side / big) if big > max_side else 1.0
    if f < 1.0:
        small = img.resize((max(16, int(img.width * f)), max(16, int(img.height * f))), Image.LANCZOS)
        sm = np.asarray(Image.fromarray((mask * 255).astype(np.uint8), "L").resize(small.size, Image.NEAREST)) > 127
        b = inpaint_biharmonic(np.asarray(small, np.float32) / 255.0, sm, channel_axis=-1)
        b = np.asarray(Image.fromarray((b * 255).clip(0, 255).astype(np.uint8)).resize(img.size, Image.LANCZOS), np.float32) / 255.0
    else:
        b = inpaint_biharmonic(a, mask, channel_axis=-1)
    out = a.copy()
    m3 = mask[..., None]
    out = out * (1 - m3) + b * m3
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGB")


def patch_fill(img: Image.Image, mask: np.ndarray, dy: int, dx: int,
               feather: int = 12) -> Image.Image:
    """从**同图内的干净区域**平移取样填补 mask（内容感知式复制）。

    适合随机纹理背景（迷彩碎片/牛仔/纸张）：`img.shift(dy,dx)` 取来的纹理与原图同族，
    边缘用羽化权重过渡。dy/dx 必须指向**无文字无主体**的干净区域。
    """
    a = np.asarray(img.convert("RGB"), np.float32)
    src = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    h, w = mask.shape
    # 羽化：mask 内距边界越远权重越高
    dist = ndi.distance_transform_edt(mask)
    wgt = np.clip(dist / max(1.0, float(feather)), 0.0, 1.0)[..., None]
    m3 = mask[..., None].astype(np.float32)
    out = a * (1 - m3) + (a * (1 - wgt) + src * wgt) * m3
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def erase(img: Image.Image, mask: np.ndarray, margin: int = 56,
          removal_strength: int = 232, edge_smoothness: int = 4,
          retexture: float = 0.0, tex_sigma: float = 6.0,
          shift: tuple[int, int] = (0, 0), max_side: int = 1100,
          method: str = "lama", nn_smooth: float = 0.0,
          nn_median: int = 0, patch_dy: int = 0, patch_dx: int = 0,
          patch_feather: int = 12, src_allow: np.ndarray | None = None,
          diffuse_down: int = 4, const_color=(0, 0, 0),
          const_feather: int = 20) -> Image.Image:
    """只重建 mask 内部（笔画处），外部像素原样保留。

    method='lama'（默认）：Big-LaMa 结构重建 —— 适合有明确结构的背景（牛仔布/链条）；
        但在**平坦色块**背景上会抹成一片平滑的雾（肉眼看就是"被挡住的一块"）。
    method='nn'  ：最近邻边界色延伸 —— 适合大 facet / 迷彩碎片 / 纯色渐变，纹理自然延续。
    method='hybrid'：先 nn 打底（保住色块边界），再叠 LaMa 低频（结构更连贯）。

    - 先按 mask 外接框裁剪再 inpaint，避免超大图整图跑爆显存；擦完整块 crop 贴回（不羽化，
      羽化会把旧字透回来）。
    - max_side：LaMa 在超大掩膜上会糊成一片（pinterest6 标题 3400x1100 实测失败）。
      超过该边长时先降采样 inpaint，再把**掩膜内**结果按原分辨率贴回（掩膜外仍用原像素，不损失清晰度）。
    - retexture>0：把原图自身的高频纹理（错位 shift 采样）叠加回填充区（慎用：会把旧字高频带回来）。
    """
    if not mask.any():
        return img.convert("RGB")
    W, H = img.size
    ys, xs = np.where(mask)
    x1, y1 = max(0, xs.min() - margin), max(0, ys.min() - margin)
    x2, y2 = min(W, xs.max() + 1 + margin), min(H, ys.max() + 1 + margin)
    base_img = img.convert("RGB")
    crop = base_img.crop((x1, y1, x2, y2))
    sub = mask[y1:y2, x1:x2]

    def _lama_fill(img_crop, m):
        import v268_lama_clean as lc
        big = max(img_crop.size)
        f = (max_side / big) if (max_side and big > max_side) else 1.0
        if f < 1.0:
            small = img_crop.resize((max(8, int(img_crop.width * f)),
                                     max(8, int(img_crop.height * f))), Image.LANCZOS)
            smask = Image.fromarray((m * 255).astype(np.uint8), "L").resize(small.size, Image.NEAREST)
            cs = lc.lama_inpaint(small, smask, removal_strength=removal_strength,
                                 edge_smoothness=max(1, int(edge_smoothness * f)))
            c = cs.resize(img_crop.size, Image.LANCZOS)
            a = np.asarray(img_crop, np.float32)
            b = np.asarray(c, np.float32)
            m3 = m[..., None]
            arr = a * (1 - m3) + b * m3
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        mp = Image.fromarray((m * 255).astype(np.uint8), "L")
        out = lc.lama_inpaint(img_crop, mp, removal_strength=removal_strength,
                              edge_smoothness=edge_smoothness)
        # LaMa 只改 mask 内，但为保险仍按 mask 严格合成
        a = np.asarray(img_crop, np.float32)
        b = np.asarray(out.resize(img_crop.size), np.float32)
        m3 = m[..., None]
        return Image.fromarray(np.clip(a * (1 - m3) + b * m3, 0, 255).astype(np.uint8), "RGB")

    if method == "nn":
        sa = src_allow[y1:y2, x1:x2] if src_allow is not None else None
        cleaned = nn_fill(crop, sub, smooth_px=nn_smooth, median_px=nn_median,
                          src_allow=sa)
    elif method == "biharm":
        cleaned = biharm_fill(crop, sub)
    elif method == "diffuse":
        sa = src_allow[y1:y2, x1:x2] if src_allow is not None else None
        cleaned = diffuse_fill(crop, sub, src_allow=sa, down=diffuse_down)
    elif method == "const":
        cleaned = constant_fill(crop, sub, color=const_color, feather=const_feather)
    elif method == "patch":
        cleaned = patch_fill(crop, sub, dy=patch_dy, dx=patch_dx, feather=patch_feather)
    elif method == "hybrid":
        # nn 保边界色块 + LaMa 补大尺度结构：mask 内 = nn 的高频 + LaMa 的低频
        sa = src_allow[y1:y2, x1:x2] if src_allow is not None else None
        nn_img = nn_fill(crop, sub, smooth_px=nn_smooth, median_px=nn_median, src_allow=sa)
        lm_img = _lama_fill(crop, sub)
        a = np.asarray(nn_img, np.float32)
        b = np.asarray(lm_img, np.float32)
        lo_a = _median_low(a, 61)
        lo_b = _median_low(b, 61)
        c = a - lo_a + lo_b
        cleaned = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8), "RGB")
    else:
        cleaned = _lama_fill(crop, sub)
    if retexture > 0:
        o = np.asarray(crop, np.float32)
        c = np.asarray(cleaned, np.float32)
        low = ndi.gaussian_filter(o, sigma=(tex_sigma, tex_sigma, 0))
        hf = o - low
        if shift != (0, 0):
            hf = np.roll(hf, shift[0], axis=0)
            hf = np.roll(hf, shift[1], axis=1)
        a = sub[..., None].astype(np.float32)
        c = c + hf * a * retexture
        cleaned = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8), "RGB")
    out = base_img.copy()
    out.paste(cleaned, (x1, y1))
    return out


def text_color(img: Image.Image, mask: np.ndarray, fallback=(30, 30, 30)):
    """取旧字像素的颜色中位数 → 新字沿用原字色（颜色保留）。"""
    arr = np.asarray(img.convert("RGB"))
    if mask.sum() < 20:
        return tuple(fallback)
    px = arr[mask]
    # 取最暗的 60% 像素（避开描边/抗锯齿），更接近字的实心色
    lum = px.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
    k = max(1, int(len(px) * 0.6))
    idx = np.argsort(lum)[:k]
    return tuple(int(v) for v in np.median(px[idx], axis=0))


# ---------------------------------------------------------------- 重画（同字体/同高/同位置）
def _cap_ratio(fp: Path) -> float:
    key = str(fp)
    if key not in _FP_CACHE:
        f = ImageFont.truetype(key, 200)
        bb = f.getbbox("A")
        _FP_CACHE[key] = (bb[3] - bb[1]) / 200.0
    return _FP_CACHE[key]


def draw_line(img: Image.Image, word: str, box, font_key: str = "blackopsone",
              color=(0, 0, 0), width_frac: float = 0.98, super: int = 4,
              cap_scale: float = 1.0, fit: str = "shrink") -> Image.Image:
    """在 box 内按**原字号(视觉字高)**居中重画新词。**以墨迹包围盒为准**。

    为什么不用字体 cap 度量：PIL 的 text y 坐标是**行顶(ascender)**，各字体
    (ascent - capHeight) 差异极大（Playfair≈0.18em、MetalMania 更离谱），
    按度量算会整行下沉 100~170px（实测 pinterest6 RAVEN 下坠 133px、6978 下坠 45px）。
    直接量**墨迹**：word 的 ink bbox 高 = 视觉字高，把它缩放/居中到 box 里，
    对任何字体都準，且大写词无降部时 ink 高就等于原字 cap 高。

    - cap_scale < 1 时按比例收窄目标字高（原框含尖刺装饰时用）
    - 宽度超出 box 时：fit="shrink" 缩字号；fit="squeeze" 保字高、横向压缩墨迹
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, int((y2 - y1) * cap_scale)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    fp = FONTS.get(font_key, FONTS["blackopsone"])
    # 探针字号量 ink 比例 → 求使 ink 高 = bh 的字号
    P = 400
    bx0, by0, bx1, by1 = ImageFont.truetype(str(fp), P).getbbox(word)
    ink_w, ink_h = max(1, bx1 - bx0), max(1, by1 - by0)
    S = max(10, int(round(P * bh / ink_h)))
    xscale = 1.0
    w_at_S = ink_w * S / P
    if w_at_S > bw * width_frac:
        if fit == "squeeze":
            xscale = max(0.55, bw * width_frac / w_at_S)
        else:
            S = max(10, int(S * (bw * width_frac) / w_at_S))
    W, H = img.size
    sc = super
    if W * sc > 16000:
        sc = max(1, 16000 // W)
    fsc = ImageFont.truetype(str(fp), S * sc)
    bx0, by0, bx1, by1 = fsc.getbbox(word)
    canvas = Image.new("L", (W * sc, H * sc), 0)
    d = ImageDraw.Draw(canvas)
    # ink 盒水平/竖直都**居中**到 (cx, cy)：
    # ⚠️ 竖直必须用 ink 自身中心 (by0+by1)/2，不能用 bh/2 —— 宽度收缩后 ink 高会小于 bh，
    #    用 bh/2 会把字顶死在框顶（pinterest3 DENIM 实测顶上移 32px）。
    d.text((cx * sc - (bx0 + bx1) / 2.0, cy * sc - (by0 + by1) / 2.0), word, font=fsc, fill=255)
    if xscale != 1.0:
        ink = np.asarray(canvas, np.float32) > 8
        xs = np.where(ink.any(0))[0]
        if len(xs):
            sub = canvas.crop((int(xs.min()), 0, int(xs.max()) + 1, H * sc))
            nw = max(1, int(round(sub.width * xscale)))
            sub = sub.resize((nw, H * sc), Image.LANCZOS)
            canvas = Image.new("L", (W * sc, H * sc), 0)
            canvas.paste(sub, (int(round(cx * sc - sub.width / 2.0)), 0))
    a = np.asarray(canvas.resize((W, H), Image.LANCZOS), np.float32) / 255.0
    out = np.asarray(img.convert("RGB"), np.float32)
    col = np.array(color, np.float32)
    out = out * (1 - a[..., None]) + col * a[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def draw_arc(img: Image.Image, word: str, center, radius: float, cap_h: float,
             font_key: str = "playfair", color=(40, 30, 45),
             start_deg: float = 8.0, end_deg: float = 172.0) -> Image.Image:
    """沿原弧线（同圆心/半径/角度范围）重画新词，字号按 cap 高对齐原字。

    角度约定：start_deg/end_deg 用**数学惯例**（0°=右，逆时针为正，y 轴向上）。
    arc_text 内部是 PIL 坐标系（y 向下），所以这里取负号转换。
    ⚠️ arc_text.draw_arc_text 返回新图（不原地改），必须接返回值。
    """
    import arc_text
    fp = FONTS.get(font_key, FONTS["playfair"])
    ratio = _cap_ratio(fp)
    W, H = img.size
    S = max(10, int(cap_h / ratio))
    avail = radius * math.radians((end_deg - start_deg) % 360)
    while S > 10:
        need = arc_text.fit_arc_text_width(word, str(fp), S, radius)
        if need <= avail * 0.97:
            break
        S = int(S * 0.94)
    tmp = Image.new("RGB", (W, H), (0, 0, 0))
    tmp = arc_text.draw_arc_text(tmp, word, str(fp), S, (255, 255, 255),
                                 (float(center[0]), float(center[1])), float(radius),
                                 -float(end_deg), -float(start_deg), char_spacing_px=1)
    a = np.asarray(tmp.convert("L"), np.float32) / 255.0
    if a.max() <= 0:
        raise RuntimeError(f"draw_arc 渲染为空: word={word!r} S={S} r={radius}")
    out = np.asarray(img.convert("RGB"), np.float32)
    col = np.array(color, np.float32)
    out = out * (1 - a[..., None]) + col * a[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------- 保色工具
def _median_low(arr: np.ndarray, size: int, down: int = 4) -> np.ndarray:
    """降采样后做中值滤波再升采样，得到低频色场（中值可忽略细笔画，不把文字带回来）。"""
    h, w = arr.shape[:2]
    s = max(1, int(down))
    small = arr[::s, ::s]
    k = max(3, int(round(size / s)) | 1)
    low = np.stack([ndi.median_filter(small[..., c], size=k) for c in range(arr.shape[2])], -1)
    zy = h / low.shape[0]
    zx = w / low.shape[1]
    out = np.stack([ndi.zoom(low[..., c], (zy, zx), order=1)[:h, :w] for c in range(arr.shape[2])], -1)
    if out.shape[0] < h or out.shape[1] < w:
        pad = np.zeros((h, w, arr.shape[2]), np.float32)
        pad[:out.shape[0], :out.shape[1]] = out
        out = pad
    return out


def median_color_lock(gen: Image.Image, ref: Image.Image, size: int = 161,
                      alpha: float = 1.0) -> Image.Image:
    """中频颜色锁：把 gen 的"大尺度色场"换成 ref 的，保留 gen 的高频细节。

    out = gen + alpha * (median_low(ref) - median_low(gen))
    - 背景从"羊皮色"拉回原图黑底；主体内部 ref/gen 分布一致 -> 几乎不变
    - 用中值而非高斯：原图里细笔画(白色标题)在窗口内占比<50% -> 不污染背景估计
    """
    a = np.asarray(gen.convert("RGB"), np.float32)
    b = np.asarray(ref.convert("RGB"), np.float32)
    out = a + alpha * (_median_low(b, size) - _median_low(a, size))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def match_fill_level(img: Image.Image, ref: Image.Image, mask: np.ndarray,
                     size: int = 241, alpha: float = 1.0,
                     max_cover: float = 0.08) -> Image.Image:
    """只把 mask 内的低频亮度/色场拉回 ref（原图）的水平，消除 LaMa 填充"偏亮/偏平"的痕迹。

    ⚠️ 掩膜占比过大（> max_cover）时**禁用**：此时中值窗口被"被擦掉的目标"占据，
    参考电平会被目标本身污染（pinterest6 白标题占窗 >50% → 把黑底反拉成白雾）。
    """
    cover = float(mask.mean())
    if cover <= 0 or cover > max_cover:
        return img
    a = np.asarray(img.convert("RGB"), np.float32)
    b = np.asarray(ref.convert("RGB"), np.float32)
    diff = _median_low(b, size) - _median_low(a, size)
    m = mask[..., None].astype(np.float32)
    out = a + alpha * diff * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def rotate_masked(img: Image.Image, mask: np.ndarray, angle: float,
                  center=None) -> Image.Image:
    """把 mask 内的元素绕 center 旋转 angle 度（元素物种不变，只改角度）。"""
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    if center is None:
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
    sub = img.crop((x1, y1, x2, y2))
    sub_m = Image.fromarray((mask[y1:y2, x1:x2] * 255).astype(np.uint8), "L")
    rot = sub.rotate(angle, resample=Image.BICUBIC, center=(center[0] - x1, center[1] - y1))
    rot_m = sub_m.rotate(angle, resample=Image.BICUBIC, center=(center[0] - x1, center[1] - y1))
    out = img.copy()
    out.paste(rot, (x1, y1), rot_m)
    return out
