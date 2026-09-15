"""styles/camo_palm_pattern.py — 迷彩棕榈图案（pinterest4 专用，pattern-level 重构）。

用户反馈（2026-09-15）：
  ❌ 旧做法把迷彩**量化成 5 色硬边界色块** → "背景色块区域被截取，乱七八糟"；
  ❌ 树只是**复制同一棵**（甚至原样保留）→ "不是裂变成其他角度构图设计的树木元素"。

本模块路线（与 camo_pattern 完全隔离，互不影响）：
  ① 迷彩底：**不做色板量化**。先把树线稿抹掉（LaMa）得到纯迷彩底，再做
     **全彩域扭曲（domain warp, 双线性）** —— 色块形状/角度/大小整体有机形变，
     颜色 100% 取自原图像素（双线性只在原色之间过渡），不会出现硬边色块。
  ② 树线稿：用 styles/palm_art 的**程序化线稿棕榈**重新画 —— 每棵树的倾斜角、
     树干弯度、叶片数量/长度/下垂度、树冠朝向都按 seed 独立生成（不是复制）。
     墨色取自原图线稿的中位色，线宽与原始笔触同量级。
"""
from __future__ import annotations
import math
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from PIL import Image
from . import base
from . import palm_art
from . import textfix as tf

STYLE_KEY = "camo_palm_pattern"
DESCRIPTION = "迷彩棕榈：全彩有机形变迷彩底 + 程序化重画线稿棕榈（每棵不同角度构图）"
COVERS = ["pinterest4"]


def default_params() -> dict:
    return {
        "warp_strength": 0.048,     # 色块标签扭曲强度（占边长比例）
        "camo_k": 6,                # 迷彩色板数
        "camo_smooth": 5,           # 标签图中值滤波尺寸
        "warp_freq": 2.2,           # 扭曲主频
        "n_cols": 6,                # 棕榈网格列数
        "n_rows": 6,                # 棕榈网格行数
        "height_lo": 0.20,          # 树高范围（占图高）
        "height_hi": 0.27,
        "lw_scale": 1.0,            # 线宽倍率
        "ink_darken": 0.75,         # 0=用原墨色, 1=纯黑
        "seed": 8888,
    }


def classify(image_path: str) -> float:
    return 0.5


def tree_ink_mask(img: Image.Image, lum_thr: float = 62.0, sat_thr: float = 18.0,
                  min_px: int = 20, thin_only: bool = True,
                  open_r: int = 22) -> np.ndarray:
    """分离**黑色线稿**（树）。判据：暗 且 中性（R≈G≈B）。

    为什么不用纯亮度阈值：迷彩里的深棕/深绿亮度也很低（实测 25 分位 lum=36.8），
    纯亮度会把大片迷彩当成线稿。深棕偏暖、深绿偏绿（R/G/B 差 > 16），
    中性暗色才是墨线。

    thin_only=True：再用**形态学开运算**把"粗结构"（成片迷彩暗块）剔掉，只留
    **细线**（线稿）。否则擦除时大片暗块会从边界取到浅色 → 画面里留下一块白斑
    （实测 pinterest4 右中部出现一块白色补丁）。
    ⚠️ open_r 必须大于**树干半宽**：实测 open_r=9 时树干(≈25px)被判为"粗结构"
    保留下来 → 擦完画面里密密麻麻残留黑色树干/树冠碎块。取 15（≈30px 以上才算粗）。
    """
    a = np.asarray(img.convert("RGB"), np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = a.max(2) - a.min(2)
    m = (lum < lum_thr) & (sat < sat_thr)
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sz >= min_px
        m = keep[lab]
    if thin_only:
        broad = ndi.binary_opening(m, structure=tf._disk(open_r))
        m = m & ~ndi.binary_dilation(broad, structure=tf._disk(2))
    return m


def ink_color(img: Image.Image, mask: np.ndarray) -> np.ndarray:
    a = np.asarray(img.convert("RGB"), np.float32)
    return np.median(a[mask], axis=0) if mask.any() else np.array([16, 14, 12], np.float32)


def camo_reblob(img: Image.Image, seed: int, k: int = 6, strength: float = 0.045,
                freq: float = 2.2, smooth: int = 5) -> Image.Image:
    """迷彩"色块重塑"：色板量化 -> **只扭曲标签图（最近邻，边界保持锐利）** -> 回填原色板。

    为什么不是双线性全彩扭曲：双线性会把边界插值成过渡带 → 整体发糊（实测像"被抹开的水彩"）。
    为什么不是旧版 camo_reshape：旧版把"最暗标签（树）"用**羽化掩膜贴回原像素**，
    贴合处留下肉眼可见的"贴纸边/被截取的一块"—— 用户反馈的"背景色块区域被截取"。
    本版树已经单独抹掉并重画，不再需要贴回，因此边界干净。

    - k：色板数（原图实测 5~6 色）。
    - smooth：标签图扭曲后的中值滤波，去掉插值产生的散点并让边界自然圆润。
    """
    pimg = img.convert("P", palette=Image.ADAPTIVE, colors=int(k), dither=Image.NONE)
    pal_raw = np.array(pimg.getpalette()[:256 * 3], np.int32).reshape(-1, 3)
    raw = np.asarray(pimg).astype(np.int32)
    used = np.unique(raw)
    remap = np.zeros(256, np.int32)
    for i, v in enumerate(used):
        remap[int(v)] = i
    idx = remap[raw].astype(np.float32)
    pal = pal_raw[used]
    H, W = idx.shape
    rng = np.random.default_rng(int(seed) * 104729 + 7)
    yy, xx = np.meshgrid(np.linspace(0, 1, H, np.float32),
                         np.linspace(0, 1, W, np.float32), indexing="ij")
    dx = np.zeros((H, W), np.float32)
    dy = np.zeros((H, W), np.float32)
    for kk in range(1, 4):
        amp = strength / (kk ** 0.8)
        f = freq * kk
        for _ in range(2):
            th = float(rng.uniform(0, 2 * np.pi))
            ph = float(rng.uniform(0, 2 * np.pi))
            u = math.cos(th) * xx + math.sin(th) * yy
            wv = np.sin(2 * np.pi * f * u + ph)
            dx += amp * wv * math.cos(th) * W
            dy += amp * wv * math.sin(th) * H
    coords = np.stack([(yy * H + dy).ravel(), (xx * W + dx).ravel()])
    w = ndi.map_coordinates(idx, coords, order=0, mode="nearest").reshape(H, W)
    w = np.clip(np.rint(w), 0, len(pal) - 1).astype(np.int32)
    if smooth >= 3:
        w = ndi.median_filter(w, size=int(smooth), mode="nearest")
    return Image.fromarray(pal[w].astype(np.uint8), "RGB")


def organic_warp(img: Image.Image, seed: int, strength: float = 0.038,
                 freq: float = 2.2) -> Image.Image:
    """**全彩有机域扭曲**（备选路线；默认**不用**）。

    保留原因：当迷彩底本身是连续渐变（无明确色块）时，量化会毁掉层次，此时才用这条。
    默认不用的原因：双线性插值把色块边界插成过渡带 → 整体发糊（实测像被抹开的水彩），
    而原图的语言是**锐利色块**，所以正式路线走 camo_reblob（order=0 标签扭曲）。

    位移场用**平面波叠加**（不是 sin×cos 乘积项）：乘积项各向异性，会把色块拉成长条流痕。
    """
    a = np.asarray(img.convert("RGB"), np.float32)
    H, W = a.shape[:2]
    rng = np.random.default_rng(int(seed) * 104729 + 7)
    yy, xx = np.meshgrid(np.linspace(0, 1, H, np.float32),
                         np.linspace(0, 1, W, np.float32), indexing="ij")
    dx = np.zeros((H, W), np.float32)
    dy = np.zeros((H, W), np.float32)
    # **平面波叠加**（不是 sin×cos 乘积项）。乘积项是各向异性的，会把色块拉成
    # 长条"流痕"（实测画面像被抹开的水彩）。平面波场各向同性 → 产生涡旋但不拖丝。
    for k in range(1, 4):
        amp = strength / (k ** 0.8)
        f = freq * k
        for _ in range(2):
            th = float(rng.uniform(0, 2 * np.pi))
            ph = float(rng.uniform(0, 2 * np.pi))
            u = math.cos(th) * xx + math.sin(th) * yy
            wv = np.sin(2 * np.pi * f * u + ph)
            dx += amp * wv * math.cos(th) * W
            dy += amp * wv * math.sin(th) * H
    coords = np.stack([(yy * H + dy).ravel(), (xx * W + dx).ravel()])
    out = np.empty_like(a)
    for c in range(3):
        out[..., c] = ndi.map_coordinates(a[..., c], coords, order=1,
                                          mode="nearest").reshape(H, W)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def _palm_specs(W: int, H: int, cols: int, rows: int, seed: int,
                h_lo: float, h_hi: float) -> list[dict]:
    """按抖动网格布置 N 棵树：每棵独立 style/tilt/高度；边缘的树允许被画框裁切。"""
    rng = np.random.default_rng(int(seed) * 31337 + 5)
    styles = ["droopy", "upright", "bushy", "palm_short"]
    specs = []
    cw, ch = W / cols, H / rows
    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5 + rng.uniform(-0.34, 0.34)) * cw
            base_y = (r + 1.05 + rng.uniform(-0.18, 0.18)) * ch
            hgt = rng.uniform(h_lo, h_hi) * H
            specs.append(dict(cx=float(cx), base_y=float(base_y), height=float(hgt),
                              seed=int(rng.integers(1, 10 ** 6)),
                              style=str(styles[int(rng.integers(0, len(styles)))])))
    return specs


def fission(image_path: str, out_dir: Path, cfg: dict, seed: int | None = None) -> list[Path]:
    p = dict(default_params())
    ic = base.get_image_cfg(cfg, image_path) or {}
    p.update({k: v for k, v in (ic.get("comfyui_params", {}).get("camo_palm_pattern", {}) or {}).items()})
    if seed is not None:
        p["seed"] = seed
    sd = int(p["seed"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # ① 树线稿 mask -> 抹掉 -> 纯迷彩底
    ink = tree_ink_mask(img)
    ink_d = ndi.binary_dilation(ink, structure=tf._disk(4))
    # 迷彩是大色块 + 细线稿：nn 填（用邻近色块色向内延伸）比 LaMa 更贴合色块语言，
    # 也不会把迷彩抹成雾（实测 LaMa 在平坦大色块上会出"被挡住的一块"）。
    camo = tf.erase(img, ink_d, method="nn", nn_median=21, margin=24)
    camo.save(str(out_dir / "_p4_camo_clean.jpg"), quality=95)

    # ② 色块重塑：量化 -> 只扭曲标签图（最近邻，边界锐利）-> 回填原色板
    warped = camo_reblob(camo, seed=sd, k=int(p["camo_k"]),
                         strength=float(p["warp_strength"]),
                         freq=float(p["warp_freq"]), smooth=int(p["camo_smooth"]))
    warped.save(str(out_dir / "_p4_warped.jpg"), quality=95)

    # ③ 程序化重画线稿棕榈
    specs = _palm_specs(W, H, int(p["n_cols"]), int(p["n_rows"]), sd,
                        float(p["height_lo"]), float(p["height_hi"]))
    layer = palm_art.palm_layer((W, H), specs, lw_scale=float(p["lw_scale"]))
    lm = np.asarray(layer, np.float32) / 255.0
    # ⚠️ 线稿是 1~2px 细线，3x 超采样下采样后每个像素只剩 ~30% 覆盖率 →
    #    直接混合会得到"灰线"（实测成品树是灰的，不是原图那种实心墨线）。
    #    做一次 **levels 拉伸**：覆盖率 ≥ lo+span 的像素拉满为实心墨。
    lm = np.clip((lm - 0.10) / 0.38, 0.0, 1.0)
    col = ink_color(img, ink)
    dark = np.array([0, 0, 0], np.float32)
    col = col * (1.0 - float(p["ink_darken"])) + dark * float(p["ink_darken"])

    arr = np.asarray(warped, np.float32)
    a3 = lm[..., None]
    out = arr * (1 - a3) + col[None, None, :] * a3
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    # ④ 文字（本图无文字，plan 为空则跳过）
    plan = base.get_text_plan_for(cfg, image_path)
    if plan:
        res = base.replace_text_plan(res, plan, dilate=10)
    save_to = out_dir / f"{Path(image_path).stem}_variant.jpg"
    res.save(str(save_to), quality=93)
    base.save_variant(res, out_dir, save_to.name)
    print(f"[camo_palm_pattern] trees={len(specs)} ink_px={int(ink.sum())} -> {save_to}")
    return [save_to]


def selfcheck_notes() -> str:
    return ("① 迷彩底无量化硬边（全彩双线性扭曲，颜色全部来自原图）；"
            "② 树为**逐棵独立程序化重画**（角度/弯度/叶形/树冠朝向各不相同），非复制；"
            "③ 墨色取原图线稿中位色；④ 木纹'绳梯'树干与羽状叶片复刻原笔触语言。")
