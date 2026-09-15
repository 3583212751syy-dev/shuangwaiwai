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


# ===========================================================================
# 材质忠实文字裂变 (v329) —— 用户硬规则（2026-09-15）
#   🔴 新词必须**沿用原字的材质 / 描边 / 字体风格**，不许平涂色块：
#      原字是牛仔贴布就画成牛仔贴布（含缝线花边），原字是金属浮雕就带金属渐变+描边+光晕。
#   🔴 不许遮挡覆盖原文：擦除仍是笔画级 inpaint；重画是"把原字材质搬到新字形上"。
#   做法：① 按连通域切出**原字逐个字母**；② 新词按同 cap 高/同字距渲染，得到逐字母 alpha；
#        ③ 每个新字母按归一化 bbox 映射去采**原字母的材质像素**（牛仔纹理/铆钉/白色 bouclé 一起搬过来）；
#        ④ 描边/backing 由"原字外环宽度+实测环色"重建 → 缝线花边/金属暗描边自动同款。
# ===========================================================================

def split_letters(mask: np.ndarray, min_area: int = 200, fuse_gap: int = 0
                  ) -> list[tuple[tuple, np.ndarray]]:
    """把笔画掩膜切成**字母**连通域，按 x 排序返回 [(bbox, comp_mask), ...]。

    小于 min_area 的碎块（原字的撇/点/® /重音）按 x 就近并入相邻字母，
    否则它们会变成"第 5 个字母"把材质映射错位。
    fuse_gap>0 时把水平间距小于该值的相邻字母合并（连笔/粘连字母）。
    """
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    objs = ndi.find_objects(lab)
    big, small = [], []
    for i, sl in enumerate(objs, 1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        (big if area >= min_area else small).append((i, sl, area))
    if not big:                                    # 全是碎块 -> 整体当一个字母
        ys, xs = np.where(mask)
        return [((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), mask)]
    # 碎块并入 x 上最近的字母
    for i, sl, _a in small:
        cx = (sl[1].start + sl[1].stop) / 2.0
        best = min(big, key=lambda t: abs((t[1][1].start + t[1][1].stop) / 2.0 - cx))
        best_sl = best[1]
        ys = (min(best_sl[0].start, sl[0].start), max(best_sl[0].stop, sl[0].stop))
        xs = (min(best_sl[1].start, sl[1].start), max(best_sl[1].stop, sl[1].stop))
        big[big.index(best)] = (best[0], (slice(*ys), slice(*xs)), best[2])
    big.sort(key=lambda t: t[1][1].start)
    boxes = []
    for i, sl, _a in big:
        m = (lab[sl] == i)
        boxes.append(((int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)), m))
    if fuse_gap > 0:
        merged = []
        for bb, m in boxes:
            if merged and bb[0] - merged[-1][0][2] <= fuse_gap:
                pb, pm = merged[-1]
                nb = (pb[0], min(pb[1], bb[1]), bb[2], max(pb[3], bb[3]))
                nm = np.zeros_like(m)
                nm[pb[1]:pb[3], pb[0]:pb[2]] |= pm
                nm[bb[1]:bb[3], bb[0]:bb[2]] |= m
                merged[-1] = (nb, nm)
            else:
                merged.append((bb, m))
        boxes = merged
    return boxes


def split_at_waists(mask: np.ndarray, ratio: float = 0.55, win: int = 12,
                    min_w: int = 40, max_parts: int = 14) -> list[np.ndarray]:
    """在**列投影的深腰处**把粘连的字母切开（返回同尺寸 bool 掩膜列表）。

    为什么需要：贴布字母的花边/外描边常在腰处相接（pinterest3 的 P 与 C 之间
    列计数从 75 掉到 32 但没有 0），单纯腐蚀切不开、按连通域分割会把两个字粘成一个
    "字母"，材质映射就会把两个字宽的纹理压进一个字形里。
    """
    parts = [mask]
    changed = True
    while changed and len(parts) < max_parts:
        changed = False
        new = []
        for p in parts:
            ys, xs = np.where(p)
            if len(xs) == 0:
                continue
            w = int(xs.max()) + 1 - int(xs.min())
            if w < 2 * min_w:
                new.append(p)
                continue
            oy, ox = int(ys.min()), int(xs.min())
            sub = p[oy:ys.max() + 1, ox:xs.max() + 1]
            c = sub.sum(0).astype(np.float32)
            best, bi = None, -1
            for x in range(min_w, sub.shape[1] - min_w):
                L = c[max(0, x - win):x].max() if x > 0 else 0.0
                R = c[x + 1:min(sub.shape[1], x + win + 1)].max() if x + 1 < sub.shape[1] else 0.0
                ref = min(L, R)
                if ref <= 0:
                    continue
                r = float(c[x]) / ref
                if best is None or r < best:
                    best, bi = r, x
            if best is not None and best < ratio:
                left = np.zeros_like(p); right = np.zeros_like(p)
                left[oy:oy + sub.shape[0], ox:ox + bi] = sub[:, :bi]
                right[oy:oy + sub.shape[0], ox + bi:ox + sub.shape[1]] = sub[:, bi:]
                new.extend([left, right]); changed = True
            else:
                new.append(p)
        parts = new
    return [p for p in parts if p.any()]


def detect_letter_patches(img: Image.Image, band, bg: float | None = None,
                          loose: float = 16, tight: float = 45, white_thr: float = 238,
                          seed_erode: int = 3, min_seed: int = 200):
    """把一条文字带切成**逐个字母**，并返回「字肉」与「完整贴布(含花边/描边)」两层掩膜。

    为什么要两层：
      - **字肉 ink**：字母主体的实心部分。用它定字母 bbox 做材质映射（牛仔布/金属面）。
      - **贴布 patch**：把 ink 做**测地膨胀**回"与底色差一点就算"的宽掩膜 —— 于是每个字母
        的缝线花边/深色外描边会被完整包进它自己的 patch，不会跨字母串联。
    典型场景：pinterest3 的 UPCY 牛仔贴布，P 与 C 的字肉相距 17px（可分），但花边几乎相接
    （直接按宽阈值分割会把 P+C 粘成一个"字母" → 材质映射错位）。
    返回 (letters, full_patch)：
      letters = [((x1,y1,x2,y2), ink_mask), ...]  按 x 排序
      full_patch = bool[H,W]，所有字母贴布的并集（用于擦除/采样源）
    """
    from skimage.morphology import reconstruction as _recon
    arr = np.asarray(img.convert("RGB"), np.float32)
    lum = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    H, W = lum.shape
    if bg is None:
        bx1, by1, bx2, by2 = [int(v) for v in band]
        pad = max(6, int(0.35 * (by2 - by1)))
        oy1, oy2 = max(0, by1 - pad), min(H, by2 + pad)
        strip = np.concatenate([lum[oy1:oy2, max(0, bx1 - 4):bx1 + 4].ravel(),
                                lum[0:220, :].ravel()])
        bg = float(np.median(strip))
    bandm = np.zeros((H, W), bool)
    bx1, by1, bx2, by2 = [int(v) for v in band]
    bandm[max(0, by1):min(H, by2), max(0, bx1):min(W, bx2)] = True
    d = np.abs(lum - bg)
    ink = ((d > tight) | (lum > white_thr)) & bandm
    ink = ndi.binary_fill_holes(ink)
    lab, n = ndi.label(ink)
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        ink = np.isin(lab, [i + 1 for i in range(n) if sz[i] >= 900])
    # 先按深腰切开粘连字母，再各自腐蚀出种子；最后用水域式「最近种子」把 loose 连通块
    # 分配给各字母 —— P 与 C 的花边几乎相接，靠腐蚀+最近种子分配才能正确在腰处分开。
    seeds = np.zeros((H, W), bool)
    for part in split_at_waists(ink):
        e = (ndi.binary_erosion(part, structure=np.ones((3, 3), bool), iterations=seed_erode)
             if seed_erode > 0 else part)
        seeds |= e
    slab, sn = ndi.label(seeds, structure=np.ones((3, 3), bool))
    if sn == 0:
        return [], np.zeros((H, W), bool)
    ssz = ndi.sum(np.ones_like(slab), slab, range(1, sn + 1))
    loose_m = ((d > loose) | (lum > white_thr)) & bandm      # 必须 ⊇ ink，否则测地重建报错
    ny, nx = ndi.distance_transform_edt(seeds == 0, return_distances=False,
                                        return_indices=True)
    assign = np.where(seeds[ny, nx], slab[ny, nx], 0)
    letters, allp = [], np.zeros((H, W), bool)
    for i in range(sn):
        if ssz[i] < min_seed:
            continue
        seed = (slab == i + 1)
        pat = loose_m & (assign == i + 1)
        pat |= seed
        core = pat & ink
        core |= seed
        ys, xs = np.where(core)
        if len(ys) == 0:
            continue
        # bbox 用**字肉**的（材质映射以字肉对齐，花边靠偏移量采样自然跟着来）
        letters.append(((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), core))
        allp |= pat
    letters.sort(key=lambda t: t[0][0])
    # 同一字母被切成多块（细笔画断裂）时按 x 重叠度合并
    merged = []
    for bb, m in letters:
        if merged:
            pb, pm = merged[-1]
            if bb[0] < pb[2] - 0.35 * (pb[2] - pb[0]):
                nb = (pb[0], min(pb[1], bb[1]), max(pb[2], bb[2]), max(pb[3], bb[3]))
                merged[-1] = (nb, pm | m)
                continue
        merged.append((bb, m))
    return merged, allp


def ink_color_field(img: Image.Image) -> np.ndarray:
    """返回"最近墨迹色场"：每个像素取其**最近的笔画像素**的颜色（笔画外=向内延伸）。

    用途：按任意坐标映射去采样原字材质时，落在字缝/字外的采样点也会拿到字色，
    不会把背景色（浅灰底/黑底）混进新字内部。
    """
    a = np.asarray(img.convert("RGB"), np.float32)
    return a


def extend_to_ink(rgb: np.ndarray, mask: np.ndarray, sigma: float = 26.0) -> np.ndarray:
    """把 mask 外的像素补成**平滑的字肉色场**（归一化高斯外插），mask 内保留原样。

    早期用"最近墨迹像素"（distance_transform 取最近点）会得到 Voronoi 块状色域，
    1:1 采样到新字上就是一块块的花斑（实测 FLUTTER 里出现大片白斑）。
    改成归一化高斯：num=G(rgb·m)/G(m)，得到连续平滑的牛仔色场，再按 mask 取原像素。
    """
    m = mask.astype(np.float32)[..., None]
    sig = (sigma, sigma, 0)
    num = ndi.gaussian_filter(rgb * m, sigma=sig, mode="nearest")
    den = ndi.gaussian_filter(m, sigma=sig, mode="nearest")
    smooth = num / np.maximum(den, 1e-6)
    # 离字肉很远的地方 den→0，结果不可信：用字肉均值兜底
    far = (den[..., 0] < 1e-3)
    if far.any():
        smooth[far] = rgb[mask].mean(axis=0) if mask.any() else 0.0
    return np.where(m > 0.5, rgb, smooth).astype(np.float32)


def _letter_swatch(raw: np.ndarray, sm: np.ndarray, size: int = 72,
                   min_cov: float = 0.985, flat_scale: float = 1.0,
                   erode: int = 6, rep_pick: bool = True, full: bool = False,
                   detail_sigma: float = 0.0):
    """从源字母墨迹里切一块**纯字肉小样**（不含背景、不含描边、不含原字母外形），用于平铺。

    为什么必须"纯字肉"：
      - 1:1 中心对齐采样会把**原字母的内部结构**（U 的铆钉走向、P 的内腔、C 的高光弧）
        一起搬到新字形上，肉眼看就是"新字里浮出旧字轮廓"（实测 FLUTTER 透出 UPCY）。
      - 平铺一块小样则只保留**材质**（牛仔织纹/白绒/缝线密度），彻底丢弃原字母几何。

    先对字肉做 erode 收缩，去掉**描边/花边/抗锯齿边**（否则小样带上一条浅蓝缝线，
    平铺后在每个字母里重复成**竖条纹**，实测肉眼可辨）。

    选窗策略：在收缩后的墨迹 bbox 内按网格扫描 size×size 窗口，取**覆盖率最高**的那个。

    返回 (swatch_rgb, swatch_ink) 或 None。
    """
    if full:
        # **整字材质场**：把整字 bbox 做成连续材质（extend_to_ink 把字外的"洞"平滑外插），
        # 再做高通滤掉大尺度特征（铆钉 24px / 补丁 40px），只留牛仔织纹。
        # ⚠️ 小窗(<=96px)上做高通是无效的：sigma 一超过窗宽就退化成"只减均值"，
        #    大特征原样保留（实测 sigma=24 时新字母上出现规律排列的珍珠/菱格）。
        # 所以高通必须作用在**整字尺度**的材质场上面。
        ys0, xs0 = np.where(sm)
        if len(xs0) == 0:
            return None
        cy1, cy2 = int(ys0.min()), int(ys0.max()) + 1
        cx1, cx2 = int(xs0.min()), int(xs0.max()) + 1
        core = sm[cy1:cy2, cx1:cx2].astype(bool)
        if erode > 0:
            e = ndi.binary_erosion(core, structure=_disk(1), iterations=int(erode))
            if e.sum() >= max(120, 0.10 * core.sum()):
                core = e
        if core.sum() < 50:
            return None
        field = extend_to_ink(raw[cy1:cy2, cx1:cx2], core, sigma=18.0)
        base = np.median(field[core], axis=0)
        dsig = float(detail_sigma) if detail_sigma and detail_sigma > 0 else 20.0
        sw = np.clip(base[None, None, :] +
                     (field - ndi.gaussian_filter(field, sigma=(dsig, dsig, 0), mode="nearest")),
                     0, 255)
        return sw.astype(np.float32), core
    m = sm.astype(bool)
    if erode > 0 and m.any():
        e = ndi.binary_erosion(m, structure=_disk(1), iterations=int(erode))
        if e.sum() >= max(200, 0.12 * m.sum()):
            m = e
    ys, xs = np.where(m)
    if len(xs) == 0:
        ys, xs = np.where(sm)
        m = sm.astype(bool)
    if len(xs) == 0:
        return None
    H, W = sm.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    size = int(min(size, max(8, min(x1 - x0, y1 - y0))))
    best = None          # 覆盖率最高
    best_rep = None      # 最"代表性"（窗口均色最接近整字字肉中位色）
    ink_med = np.median(raw[m], axis=0) if m.any() else np.zeros(3, np.float32)
    step = max(3, size // 6)
    for cov_bar in (min_cov, 0.90, 0.75, 0.55):
        if best_rep is not None:
            break
        ygrid = list(range(max(0, y0), max(1, y1 - size + 1) + 1, step)) or [max(0, y0)]
        xgrid = list(range(max(0, x0), max(1, x1 - size + 1) + 1, step)) or [max(0, x0)]
        for cy in ygrid:
            for cx in xgrid:
                win = m[cy:cy + size, cx:cx + size]
                if win.shape[0] < size or win.shape[1] < size:
                    continue
                cov = float(win.mean())
                if cov < cov_bar:
                    continue
                if best is None or cov > best[0]:
                    best = (cov, cx, cy)
                block = raw[cy:cy + size, cx:cx + size]
                # ⚠️ 不能用"色差最小（最平）"选窗：U 的白色珍珠铆钉内部就是**最平的窗口**，
                #    选中后整个字母被填成一坨近白 (232,232,232)，实测 DENIM 的 D 变成惨白。
                # 改用"最接近整字中位色"（代表色）——铆钉/补丁/高光弧这类离群窗口自动被排除。
                dcol = float(np.linalg.norm(block[win].mean(axis=0) - ink_med)) if win.any() else 1e9
                if best_rep is None or dcol < best_rep[0]:
                    best_rep = (dcol, cx, cy)
    if rep_pick and best_rep is not None:
        _, cx, cy = best_rep
    elif best is not None:
        _, cx, cy = best
    else:
        cy = int(np.clip(int(ys.mean()) - size // 2, 0, max(0, H - size)))
        cx = int(np.clip(int(xs.mean()) - size // 2, 0, max(0, W - size)))
    sw = raw[cy:cy + size, cx:cx + size].copy()
    si = m[cy:cy + size, cx:cx + size].copy()
    # 残留的少量非字肉像素（覆盖率 <100% 时）用**字肉中位色**补掉：
    # 否则平铺时会在每个重复位置出现同一颗浅色斑点（周期性噪点，肉眼可辨）。
    if si.any() and not si.all():
        med = np.median(sw[si], axis=0)
        sw[~si] = med
    if flat_scale < 1.0:
        k = max(4, int(round(size * flat_scale)))
        im = Image.fromarray(np.clip(sw, 0, 255).astype(np.uint8), "RGB").resize((k, k), Image.BOX)
        sw = np.asarray(im.resize((size, size), Image.BILINEAR), np.float32)
    # **材质分解**：base(字肉中位色) + detail(高通细纹)。
    # 为什么：大尺度特征（C 下缘那块深色补丁、U 的内腔阴影、P 的高光弧）尺寸 20~60px，
    # 平铺后会在新字母里排成**孤立的色块/白斑**（实测 DENIM 的 N 里出现白色圆斑）。
    # 牛仔织纹/毛绒只在 2~4px 尺度。高通把"大特征"滤掉，只留**真正的材质**。
    if detail_sigma and detail_sigma > 0 and si.any():
        base = np.median(sw[si], axis=0)
        det = sw - ndi.gaussian_filter(sw, sigma=(detail_sigma, detail_sigma, 0), mode="nearest")
        det[~si] = 0.0
        sw = np.clip(base[None, None, :] + det, 0, 255)
    return sw.astype(np.float32), si


def _mirror_index(n: int, period: int, phase: int = 0) -> np.ndarray:
    """镜像平铺索引：0..n-1 的坐标映射到 0..period-1（三角波），无接缝。"""
    p = 2 * max(1, period)
    t = np.mod(np.arange(n) + phase, p)
    return np.where(t < period, t, p - 1 - t)


def render_word_letters(img_size: tuple[int, int], word: str, font_key: str,
                        box, cap_scale: float = 1.0, super: int = 4,
                        width_frac: float = 0.99, jitter_deg: float = 0.0,
                        seed: int = 0, fit: str = "shrink") -> dict:
    """渲染新词：返回 {alpha, labels, letters, cap_h, box}。

    - 以**逐字母**渲染，得到每个字母自己的墨迹 bbox（后续材质映射按字母对齐）。
    - 字高统一：按 'H' 的墨迹高定字号，使新词 cap 高 ≈ 目标字高。
    - 字距：把富余宽度均分到字间（tracking），而不是拉宽字形。
    - jitter_deg>0：每个字母随机微旋（模仿手工贴布/徽章字母的错落）。
    - 宽度超出时 fit="shrink" 缩字号 / fit="squeeze" 横向压缩。
    """
    W, H = img_size
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = x2 - x1
    bh = max(8, (y2 - y1) * cap_scale)
    fp = FONTS.get(font_key, FONTS["blackopsone"])
    P = 400
    probe = ImageFont.truetype(str(fp), P)
    bbH = probe.getbbox("H")
    ink_h_H = max(1, bbH[3] - bbH[1])                      # 'H' 的墨迹高 = cap 高
    S = max(10, int(round(P * bh / ink_h_H)))              # 整图坐标系下的字号
    sc = super
    if W * sc > 16000:
        sc = max(1, 16000 // W)
    # ── 一律在**超采样坐标系**里排版，落图时再 /sc 换回整图坐标 ──
    avail = bw * width_frac * sc
    xscale = 1.0
    min_xscale = 0.62                                       # squeeze 下限：再窄就不像原字了
    for _ in range(5):
        fsc = ImageFont.truetype(str(fp), max(10, int(S * sc)))
        met = []
        for ch in word:
            b = fsc.getbbox(ch)
            met.append((ch, max(1, b[2] - b[0]), b[0], b[1], b[2], b[3]))
        tot = float(sum(m[1] for m in met))
        if tot <= avail:
            break
        xs = avail / tot
        if fit == "squeeze" and xs >= min_xscale:
            xscale = xs
            break
        # 否则缩字号重试（squeeze 模式下只缩到刚好还能满足 min_xscale）
        S = max(10, int(S * (xs / min_xscale if fit == "squeeze" else xs)) if fit == "squeeze"
                else max(10, int(S * xs)))
    fsc = ImageFont.truetype(str(fp), max(10, int(S * sc)))
    met = []
    for ch in word:
        b = fsc.getbbox(ch)
        met.append((ch, max(1, b[2] - b[0]), b[0], b[1], b[2], b[3]))
    tot = float(sum(m[1] for m in met))
    if xscale == 1.0 and tot > avail:
        xscale = max(0.5, avail / tot)
    placed_w = tot * xscale
    n = max(1, len(word))
    gap = max(0.0, (avail - placed_w) / (n - 1)) if n > 1 else 0.0
    canvas = Image.new("L", (W * sc, H * sc), 0)
    labimg = Image.new("L", (W * sc, H * sc), 0)
    rng = np.random.default_rng(seed)
    cur = x1 * sc + (avail - (tot * xscale + gap * (n - 1))) / 2.0
    ycen = (y1 + y2) / 2.0 * sc
    letters = []
    cap_sc = max(1, fsc.getbbox("H")[3] - fsc.getbbox("H")[1])
    for idx, (ch, iw, bx0, by0, bx1, by1) in enumerate(met):
        ch_w, ch_h = bx1 - bx0, by1 - by0
        lc = Image.new("L", (max(1, ch_w + 20), max(1, ch_h + 20)), 0)
        ld = ImageDraw.Draw(lc)
        ld.text((10 - bx0, 10 - by0), ch, font=fsc, fill=255)
        ang = float(rng.uniform(-jitter_deg, jitter_deg)) if jitter_deg > 0 else 0.0
        if ang:
            lc = lc.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=0)
        if xscale != 1.0:
            lc = lc.resize((max(1, int(round(lc.width * xscale))), lc.height), Image.LANCZOS)
        a = np.asarray(lc, np.float32)
        ys, xs = np.where(a > 127)
        if len(xs) == 0:
            cur += iw * xscale + gap
            continue
        ix0, ix1 = int(xs.min()), int(xs.max()) + 1
        iy0, iy1 = int(ys.min()), int(ys.max()) + 1
        px = int(round(cur - ix0))
        py = int(round(ycen - (iy0 + iy1) / 2.0))
        canvas.paste(lc, (px, py), lc)
        labimg.paste(Image.new("L", lc.size, min(255, idx + 1)), (px, py), lc)
        letters.append(((int(round((px + ix0) / sc)), int(round((py + iy0) / sc)),
                         int(round((px + ix1) / sc)), int(round((py + iy1) / sc))), idx))
        cur += (ix1 - ix0) + gap
    alpha = np.asarray(canvas.resize((W, H), Image.LANCZOS), np.float32) / 255.0
    labels = np.asarray(labimg.resize((W, H), Image.NEAREST))
    return dict(alpha=alpha, labels=labels, letters=letters,
                cap_h=cap_sc / sc, box=(x1, y1, x2, y2), font_size=S)


def measure_outline(img: Image.Image, mask: np.ndarray, k_max: int = 26):
    """量原字**外环**逐像素宽的环色 + 与背景的差异，用于重建"同款描边/贴布花边"。

    返回 [(k, ring_color, dist_to_bg), ...]；k=1 是紧贴笔画的外环。dist_to_bg 明显
    大于 0 且连续若干 k 稳定的区间就是原字的描边/backing 宽度。
    """
    rgb = np.asarray(img.convert("RGB"), np.float32)
    bg = np.median(rgb[~ndi.binary_dilation(mask, structure=_disk(6))].reshape(-1, 3), axis=0) \
        if (~ndi.binary_dilation(mask, structure=_disk(6))).any() else np.zeros(3, np.float32)
    out = []
    prev = mask
    for k in range(1, k_max + 1):
        cur = ndi.binary_dilation(mask, structure=_disk(k))
        ring = cur & ~prev
        prev = cur
        if ring.sum() < 20:
            out.append((k, None, 0.0))
            continue
        col = rgb[ring].mean(axis=0)
        out.append((k, col, float(np.linalg.norm(col - bg))))
    return out


def ring_profile(src_img: Image.Image, src_mask: np.ndarray, k_max: int = 48):
    """量原字**逐像素外环**的平均色（k=1 是紧贴笔画的环）。用于重建同款描边/光晕/贴布花边。"""
    rgb = np.asarray(src_img.convert("RGB"), np.float32)
    prof, prev = [], src_mask.copy()
    for k in range(1, k_max + 1):
        cur = ndi.binary_dilation(src_mask, structure=_disk(k))
        ring = cur & ~prev
        prev = cur
        prof.append(rgb[ring].mean(axis=0) if int(ring.sum()) >= 12 else None)
    # 空缺处用最近的有效环色补上
    valid = [i for i, c in enumerate(prof) if c is not None]
    if not valid:
        return [np.zeros(3, np.float32)] * k_max
    for i, c in enumerate(prof):
        if c is None:
            j = min(valid, key=lambda v: abs(v - i))
            prof[i] = prof[j]
    return prof


def rebuild_ring(out: np.ndarray, ink: np.ndarray, prof: list, scale: float,
                 ring_px: int, dash: int = 0, dash_color=None) -> np.ndarray:
    """按**尺寸比例缩放后的外环剖面**重建描边/光晕：目标第 k 环取原图第 k/scale 环的颜色。

    为什么不能用"轮廓相对偏移采样"：新字比原字小一半时，绝对偏移量对不上，
    采出来是原字花边的**矩形色块**（实测）。改成剖面法后描边严格跟着新字形走。
    """
    if ring_px <= 0:
        return out
    prev = ink.copy()
    for k in range(1, int(ring_px) + 1):
        cur = ndi.binary_dilation(ink, structure=_disk(k))
        ring = cur & ~prev
        prev = cur
        if not ring.any():
            break
        sk = int(round(k / max(1e-6, scale)))
        col = prof[min(len(prof) - 1, max(0, sk - 1))]
        if dash and dash_color is not None:
            ys, xs = np.where(ring)
            sel = (((xs + ys) // int(dash)) % 2) == 0
            out[ys[sel], xs[sel]] = np.array(dash_color, np.float32)
            out[ys[~sel], xs[~sel]] = col
        else:
            out[ring] = col
    return out


def add_patch_shadow(out: np.ndarray, shape: np.ndarray, dx: int = 7, dy: int = 7,
                     blur: float = 7.0, color=(120, 120, 120),
                     strength: float = 0.55) -> np.ndarray:
    """给"贴布字"投一层软阴影（先画阴影、再把字叠上去）。

    为什么必须有：原图 UPCY 的每个字母都是**实体贴布**——右下有一层灰色投影。
    v329 版把字直接平铺到灰底上，没有投影 → 字像"贴纸浮在画面上"，
    用户反馈"文本光影真实感做好"。位移量/模糊量按原图量测（约 7px 位移、7px 模糊）。
    """
    m = shape.astype(np.float32)
    sh = ndi.shift(m, (float(dy), float(dx)), order=1, mode="nearest")
    if blur > 0:
        sh = ndi.gaussian_filter(sh, float(blur))
    sh = np.clip(sh * float(strength), 0.0, 1.0)
    a = sh[..., None]
    return out * (1.0 - a) + np.array(color, np.float32)[None, None, :] * a


def draw_line_material(img: Image.Image, word: str, box, font_key: str,
                       src_img: Image.Image, src_mask: np.ndarray,
                       src_letters=None, cap_scale: float = 1.0,
                       ring_px: int = 0, jitter_deg: float = 0.0, seed: int = 0,
                       fit: str = "shrink", material_mix: float = 1.0,
                       super: int = 4, width_frac: float = 0.99,
                       mat_erode: int = 8, ring_dash: int = 0, mat_mode: str = "center",
                       ring_dash_color=None, outline_layers=None,
                       swatch_size: int = 72, mat_flat: float = 1.0,
                       swatch_gap: int = 3, swatch_rep: bool = True,
                       swatch_erode: int = 8, swatch_min_cov: float = 0.96,
                       detail_sigma: float = 20.0, swatch_full: bool = True,
                       shadow=None, inner_ring: int = 0,
                       inner_ring_col=None, ring_dark: float = 0.62,
                       seam_px: int = 3) -> Image.Image:
    """把**原字的材质**搬到新词字形上（同字体风格/同材质/同描边），返回合成后的图。

    采样规则（两套，避免任何"平涂"）：
      ① **字内**：新字母 i 的材质取自原字母 (i mod n_src)，字母内部做归一化 bbox 线性映射
         → 牛仔纹理/铆钉/缝线/白色 bouclé/金属渐变整套搬过来。
      ② **字外环**（ring_px>0）：按"离最近笔画像素的偏移量"做**轮廓相对采样** ——
         新字某环点的偏移 o = p - nearest_ink(p)，就到原图 (mapped(nearest_ink) + o) 取样。
         于是原字的贴布花边/缝线虚线/深色描边/金属光晕会被整条搬过来，且跟着新字形走。
         比"按宽度重画一圈纯色"忠实得多（那条路做不出缝线的虚线质感）。

    outline_layers：可选 [(k_outer, (r,g,b)), ...]，用于强制某段环用固定色（当轮廓相对采样
    在某种背景上不理想时兜底）。不传则完全靠采样。
    """
    W, H = img.size
    src_mask = src_mask.astype(bool)
    if src_letters is None:
        src_letters = split_letters(src_mask, min_area=max(120, int(src_mask.sum() * 0.02)))
    if not src_letters:
        return img
    maps = render_word_letters((W, H), word, font_key, box, cap_scale=cap_scale,
                               super=super, jitter_deg=jitter_deg, seed=seed, fit=fit,
                               width_frac=width_frac)
    alpha, labels = maps["alpha"], maps["labels"]
    if alpha.max() <= 0:
        raise RuntimeError(f"draw_line_material 渲染为空: {word!r}")
    raw = np.asarray(src_img.convert("RGB"), np.float32)
    n_src = len(src_letters)
    out = np.asarray(img.convert("RGB"), np.float32)

    # 每个源字母的"最近墨迹色场"（只在自己 bbox 周边算，避免全图 ×4 内存爆炸）：
    # ⚠️ 必须**逐字母**做扩展。用全局 ink 扩展会把邻居字母的背景/颜色拖进来，
    #    新字形里就会浮现出**源字母的形状**（实测 FLUTTER 里透出 U/P/C/Y 的轮廓）。
    fields = []
    for sb, sm in src_letters:
        pad = 10
        cx1, cy1 = max(0, sb[0] - pad), max(0, sb[1] - pad)
        cx2, cy2 = min(W, sb[2] + pad), min(H, sb[3] + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            fields.append((raw, 0, 0))
            continue
        # 只取**字肉内部**（腐蚀掉轮廓/描边那一圈）再向四周扩展：
        # 这样材质里只剩牛仔布/铆钉/白色绒线本身，**不带原字母的外形** ——
        # 否则把原字贴到不同字形上会透出原字轮廓（实测 FLUTTER 里浮出 U/P/C/Y）。
        core = sm[cy1:cy2, cx1:cx2]
        if mat_erode > 0:
            # 自适应腐蚀：腐蚀到面积掉太多就停（细笔画字母如 Y 会被腐蚀没，
            # 留下噪声掩膜 → 材质出现大片白斑）
            area0 = int(core.sum())
            k = 0
            while k < int(mat_erode):
                nxt = ndi.binary_erosion(core, structure=_disk(1))
                if nxt.sum() < max(60, 0.35 * area0):
                    break
                core = nxt
                k += 1
        fields.append((extend_to_ink(raw[cy1:cy2, cx1:cx2], core), cx1, cy1))

    # 平铺模式的**纯字肉小样**（每源字母一块）：只保留材质，彻底不带原字几何。
    swatches = []
    for sb, sm in src_letters:
        swatches.append(_letter_swatch(raw, sm, size=swatch_size, flat_scale=mat_flat,
                                       rep_pick=swatch_rep, erode=swatch_erode,
                                       min_cov=swatch_min_cov, detail_sigma=detail_sigma,
                                       full=swatch_full))

    # 逐源字母的**绣边色**（贴布外圈那层）：字肉腐蚀掉 10px 后剩下的环 = 绣边/贴边。
    ring_cols = []
    for _sb, _sm in src_letters:
        _edge = _sm & ~ndi.binary_erosion(_sm, structure=_disk(10))
        ring_cols.append(np.median(raw[_edge], axis=0) if int(_edge.sum()) > 30
                         else np.median(raw[_sm], axis=0))

    def mapper(bb, li):
        sb, _sm = src_letters[li % n_src]
        tw_, th_ = max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
        sw_, sh_ = max(1, sb[2] - sb[0]), max(1, sb[3] - sb[1])
        return sb, sw_ / tw_, sh_ / th_


    # ---- ① 字内材质 ----
    # 采样方式：**中心对齐 1:1**（默认）。新字通常比原字小，"按 bbox 拉伸"会把原字材质
    # 放大 2 倍以上 → 纹理变成花斑（实测 FLUTTER 里出现大片白斑/色块）。
    # 1:1 采样保留原材质真实尺度；超出源字母部分由 extend_to_ink 的"最近字肉色"兜住，
    # 不会漏出背景。（mat_mode='fit' 可切回 bbox 拉伸）
    ink = alpha > 0.02
    # ---- 贴布风格（pinterest3）：字形=整块贴布，外圈 inner_ring px 是**绣边**，
    #      内圈是**字肉材质**；贴布外侧投软阴影（原图量测：位移≈7px、模糊≈7px）。
    band = np.zeros((H, W), bool)
    if inner_ring > 0:
        band = ink & ~ndi.binary_erosion(ink, structure=_disk(int(inner_ring)))
    if shadow:
        shape = (ndi.binary_dilation(ink, structure=_disk(int(ring_px)))
                 if ring_px > 0 else ink)
        out = add_patch_shadow(out, shape, **shadow)
    m_fill = ink & (~band) if inner_ring > 0 else ink
    mat = np.zeros((H, W, 3), np.float32)
    filled = np.zeros((H, W), bool)
    for (bb, li) in maps["letters"]:
        sb, sm = src_letters[li % n_src]
        # ⚠️ 只填**属于本字母**的像素（用 labels 判定）：字距被压缩时相邻字母 bbox 会重叠，
        #    按 bbox 整块覆盖会让后画的字母把前一个的材质改掉 → 一个字里出现竖向色带接缝。
        m_here = (labels[bb[1]:bb[3], bb[0]:bb[2]] == (li + 1)) & \
                 (m_fill[bb[1]:bb[3], bb[0]:bb[2]])
        if not m_here.any():
            continue
        # ---- 平铺模式：材质只来自"纯字肉小样"，与原字母几何完全无关（无鬼影）----
        if mat_mode == "tile":
            sw = swatches[li % n_src]
            if sw is None:
                continue
            swb = sw[0]
            sh_, sw_ = swb.shape[:2]
            bh_, bw_ = bb[3] - bb[1], bb[2] - bb[0]
            ph = (li + 1) * swatch_gap
            ix = _mirror_index(bw_, sw_, phase=ph)
            iy = _mirror_index(bh_, sh_, phase=ph)
            sub = swb[np.ix_(iy, ix)]
            for c in range(3):
                mat[bb[1]:bb[3], bb[0]:bb[2], c] = np.where(
                    m_here, sub[..., c], mat[bb[1]:bb[3], bb[0]:bb[2], c])
            filled[bb[1]:bb[3], bb[0]:bb[2]] |= m_here
            continue
        fld, ox, oy = fields[li % n_src]
        ys_s, xs_s = np.where(sm)
        scx, scy = (xs_s.mean(), ys_s.mean()) if len(xs_s) else ((sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2)
        tcx, tcy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
        yy, xx = np.mgrid[bb[1]:bb[3], bb[0]:bb[2]]
        if mat_mode == "fit":
            fx = (sb[2] - sb[0]) / max(1, bb[2] - bb[0])
            fy = (sb[3] - sb[1]) / max(1, bb[3] - bb[1])
            sx = (sb[0] - ox) + (xx - bb[0]) * fx
            sy = (sb[1] - oy) + (yy - bb[1]) * fy
        else:
            sx = (scx - ox) + (xx - tcx)
            sy = (scy - oy) + (yy - tcy)
        coords = np.stack([sy.ravel(), sx.ravel()])
        for c in range(3):
            vals = ndi.map_coordinates(fld[..., c], coords, order=1,
                                       mode="nearest").reshape(yy.shape)
            mat[bb[1]:bb[3], bb[0]:bb[2], c] = np.where(
                m_here, vals, mat[bb[1]:bb[3], bb[0]:bb[2], c])
        filled[bb[1]:bb[3], bb[0]:bb[2]] |= m_here

    # ---- ①b 贴布绣边：把字形外圈 inner_ring px 涂成**源字母自己的绣边色**（逐字母），
    #      并在字肉与绣边交界压一条暗缝 → 造出原图那种"布片缝在一圈边上的立体感"。
    if inner_ring > 0 and band.any():
        core_all = ndi.binary_erosion(ink, structure=_disk(int(inner_ring)))
        for (bb, li) in maps["letters"]:
            y1b, y2b, x1b, x2b = bb[1], bb[3], bb[0], bb[2]
            m_here = labels[y1b:y2b, x1b:x2b] == (li + 1)
            mb = band[y1b:y2b, x1b:x2b] & m_here
            if not mb.any():
                continue
            col = (np.array(inner_ring_col, np.float32) if inner_ring_col is not None
                   else np.asarray(ring_cols[li % n_src], np.float32))
            view = mat[y1b:y2b, x1b:x2b]
            view[mb] = col
            seam = ndi.binary_dilation(core_all[y1b:y2b, x1b:x2b],
                                       structure=_disk(int(seam_px))) & mb
            view[seam] = col * float(ring_dark)
            filled[y1b:y2b, x1b:x2b] |= mb

    # ---- ② 字外环：按原图**环剖面**重建（颜色/宽度/缝线虚线都跟着原字，且严格贴新字形）----
    if ring_px > 0:
        src_cap = float(np.median([(b[3] - b[1]) for b, _ in src_letters])) or 1.0
        dst_cap = max(1.0, float(maps["cap_h"]))
        scale = dst_cap / src_cap
        prof = ring_profile(src_img, src_mask, k_max=int(min(90, ring_px / max(0.12, scale)) + 8))
        out = rebuild_ring(out, ink, prof, scale, int(ring_px),
                           dash=int(ring_dash), dash_color=ring_dash_color)
    if outline_layers:
        for k_out, col in outline_layers:
            ring = ndi.binary_dilation(ink, structure=_disk(int(k_out))) & (~ink)
            out[ring] = np.array(col, np.float32)

    a3 = alpha[..., None]
    src_px = np.where(filled[..., None], mat, np.zeros(3, np.float32))
    px = src_px * material_mix + out * (1 - material_mix)
    out = out * (1 - a3) + px * a3
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
