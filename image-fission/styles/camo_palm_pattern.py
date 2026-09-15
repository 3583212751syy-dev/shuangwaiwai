"""styles/camo_palm_pattern.py — 迷彩棕榈图案（pinterest4 专用，pattern-level 重构）。

用户反馈（2026-09-15 三轮）：
  ❌ v329 程序化画树 → "元素长得好看吗，跟原图有什么关系"；
  ❌ v331 程序化羽状复叶 → "树木剪影小元素根据原图去裂变不是让你去乱做"。

v332 定论：**剪影元素绝不程序化新画**。真裂变 = 提取原图的每一只剪影
（扇形穗 / 棕榈树 / 横线 / 点），逐元素做**结构形变**（绕枢轴的加权旋转：
树=根部枢轴→树冠摆动；穗=质心→涡旋扭；横线=自转微角；点=保持），
再原位同尺寸贴回。结构/墨色/笔触 100% 来自原图，姿态各异 = 同构异姿。
迷彩底：抹掉元素后做 camo_reblob（标签图扭曲，边界锐利、颜色取自原色板）。
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
from . import subject_morph as smod

STYLE_KEY = "camo_palm_pattern"
DESCRIPTION = "迷彩棕榈：迷彩色块重塑 + 原图剪影逐元素结构形变（同构异姿，不新画）"
COVERS = ["pinterest4"]


def default_params() -> dict:
    return {
        # v333（用户："背景裂变太乱了没规律"）：位移场降频+减倍频+加大平滑
        # → 色块变成少量大尺度连贯形变（"有规律的流动"），不再是 6 组平面波叠加的碎乱。
        "warp_strength": 0.055,     # 色块标签扭曲强度（占边长比例）
        "camo_k": 6,                # 迷彩色板数
        "camo_smooth": 9,           # 标签图中值滤波尺寸（大 → 色块边界圆润连贯）
        "warp_freq": 1.35,          # 扭曲主频（低频 = 大块、有规律）
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
    # v333: 只用 2 个倍频（原 3 个）→ 位移场只有大尺度波动，色块连贯有规律；
    # 倍频过多会把色块切碎（用户："背景裂变太乱了没规律"）。
    for kk in range(1, 3):
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
    styles = ["auto", "auto", "auto", "line", "solid"]
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


def _element_kind(area: int, w_: int, h_: int) -> str:
    """按形状把剪影元素分类：dot（点）/ dash（细长横线）/ tree（纵向棕榈树）/ tuft（扇形穗）。"""
    if area < 140:
        return "dot"
    ar = max(w_, h_) / max(1.0, min(w_, h_))
    if ar > 3.6:
        return "dash"
    if h_ > 1.30 * w_:
        return "tree"
    return "tuft"


def _element_disp(rgba: np.ndarray, kind: str, rng) -> tuple[np.ndarray, np.ndarray] | None:
    """层坐标系位移场：绕枢轴的加权旋转（显示端旋转 +θ ⇔ 采样端旋转 -θ）。

    · tree：枢轴在**根部**（元素最低点的中轴）→ 树冠摇摆而树干根部不动；
    · tuft：枢轴在质心 → 整穗涡旋扭转（穗刺重新排布）；
    · dash：绕自身中心微转；dot：不动。
    只改方向，不改位置/大小（硬规则：同位同大）。
    """
    hh, ww = rgba.shape[:2]
    m = rgba[..., 3] > 120
    if not m.any():
        return None
    ys, xs = np.where(m)
    cx, cy = float(xs.mean()), float(ys.mean())
    R = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) + 10.0
    # v334（用户："前面树木乱七八糟，排版有没有点审美"）：废**逐元素随机角**——
    # 91 个元素各自乱转 = 噪声不是设计。本版改**统一设计性倾斜**：同类元素同向同角
    # （树全部向右倾 7°、穗全部同向扭转、横线一致微倾），只留 ±0.02rad 微抖动保手工感
    # → 排列有规律、有韵律，远看是"设计过的图案"而非"被吹乱的树林"。
    if kind == "tree":
        px, py = cx, float(ys.max())
        theta = 0.12 + float(rng.uniform(-0.02, 0.02))
    elif kind == "tuft":
        px, py = cx, cy
        theta = 0.26 + float(rng.uniform(-0.02, 0.02))
    elif kind == "dash":
        px, py = cx, cy
        theta = 0.08 + float(rng.uniform(-0.01, 0.01))
    else:
        return np.zeros((hh, ww), np.float32), np.zeros((hh, ww), np.float32)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    dx0 = xx - px
    dy0 = yy - py
    wgt = smod.smoothstep(np.hypot(dx0, dy0), 0.05 * R, 0.42 * R)
    ang = -theta * wgt
    ca, sa = np.cos(ang), np.sin(ang)
    rx = dx0 * ca - dy0 * sa
    ry = dx0 * sa + dy0 * ca
    return (ry - dy0).astype(np.float32), (rx - dx0).astype(np.float32)


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

    # ① 提取原图全部黑色剪影元素（不开运算——整只穗/树都要）。
    #    ⚠️ 阈值必须 (55,14)：迷彩深棕块 sat≈28、深绿 sat≈30，放宽到 22 会把大片
    #    迷彩暗块吃进掩膜（实测 42% 面积被误抓）；(55,14) 只抓纯黑墨线（22.6%）。
    ink = tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    vis = np.asarray(img).copy(); vis[ink] = [255, 0, 0]
    Image.fromarray(vis).save(str(out_dir / "_p4_ink_vis.jpg"), quality=92)
    ink_d = ndi.binary_dilation(ink, structure=tf._disk(4))

    # ② 擦掉全部元素 → 纯迷彩底（nn 延色，保住色块语言）
    camo = tf.erase(img, ink_d, method="nn", nn_median=21, margin=24)
    camo.save(str(out_dir / "_p4_camo_clean.jpg"), quality=95)

    # ③ 色块重塑：量化 -> 只扭曲标签图（最近邻，边界锐利）-> 回填原色板
    warped = camo_reblob(camo, seed=sd, k=int(p["camo_k"]),
                         strength=float(p["warp_strength"]),
                         freq=float(p["warp_freq"]), smooth=int(p["camo_smooth"]))
    warped.save(str(out_dir / "_p4_warped.jpg"), quality=95)

    # ④ 逐元素结构形变贴回（100% 取自原图元素，同位同大，姿态各异）
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    out = np.asarray(warped, np.float32)
    rng = np.random.default_rng(sd * 977 + 31)
    n_morph = 0
    for i in range(1, n + 1):
        m_i = lab == i
        area = int(m_i.sum())
        ys, xs = np.where(m_i)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        kind = _element_kind(area, x1 - x0, y1 - y0)
        R = 0.5 * max(x1 - x0, y1 - y0)
        pad = int(R * 0.35) + 14
        bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
        bx1, by1 = min(W, x1 + pad), min(H, y1 + pad)
        sub_m = np.zeros((H, W), bool)
        sub_m[by0:by1, bx0:bx1] = m_i[by0:by1, bx0:bx1]
        rgba, box = smod.make_layer(img, sub_m, feather=1.0, box=(bx0, by0, bx1, by1))
        # v334：贴边元素**原样贴回**（位置不变）——旋转会在图界外采样（nearest 复制）
        # 拉出条纹拉丝（实测左缘两处）。中间元素才做统一倾斜。
        edge_touch = (x0 <= 2) or (y0 <= 2) or (x1 >= W - 2) or (y1 >= H - 2)
        if edge_touch:
            hh_, ww_ = rgba.shape[:2]
            disp = (np.zeros((hh_, ww_), np.float32), np.zeros((hh_, ww_), np.float32))
        else:
            disp = _element_disp(rgba, kind, rng)
        if disp is None:
            continue
        wlay = smod.warp_layer(rgba, disp, order=1)
        al = np.clip(wlay[..., 3:4] / 255.0, 0.0, 1.0)
        bh, bw = wlay.shape[:2]
        reg = out[box[1]:box[1] + bh, box[0]:box[0] + bw]
        out[box[1]:box[1] + bh, box[0]:box[0] + bw] = reg * (1 - al) + wlay[..., :3] * al
        n_morph += 1
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    # ⑤ 文字（本图无文字，plan 为空则跳过）
    plan = base.get_text_plan_for(cfg, image_path)
    if plan:
        res = base.replace_text_plan(res, plan, dilate=10)
    save_to = out_dir / f"{Path(image_path).stem}_variant.jpg"
    res.save(str(save_to), quality=93)
    base.save_variant(res, out_dir, save_to.name)
    print(f"[camo_palm_pattern] elements={n} morphed={n_morph} ink_px={int(ink.sum())} -> {save_to}")
    return [save_to]


def selfcheck_notes() -> str:
    return ("① 剪影元素 100% 提取自原图（禁程序化新画），逐元素绕枢轴旋转=同构异姿；"
            "② 位置/大小不变（硬规则）；③ 迷彩底 camo_reblob 标签扭曲，边界锐利颜色取自原色板；"
            "④ 点/横线等小元素同样处理（🔴3 逐元素）。")
