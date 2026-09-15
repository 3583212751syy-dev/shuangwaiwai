"""palm_art.py — 程序化棕榈（v330：按原图 Pinterest(4) 放大 2 倍逐棵量测重写）

原图的棕榈是**两种画法混排**：

  A. **线稿型**（约 2/3）
     · 树干 = 紧贴的**锯齿线**（振幅 ≈1.3 线宽、步距 ≈1.7 线宽，压成弹簧状）——不是"绳梯"。
     · 叶  = 一条**中脉细线** + 两侧**短叶枝**（每根叶枝是独立短线，彼此留空隙），
             叶枝长 ≈ 叶长 16%~28%、与中脉夹角 36°~58°、向叶尖递减；部分整叶画成锯齿折线。
     · 线宽 ≈ 7~8px（1242px 宽原图上 2*mean(EDT)≈4.6~7）。
  B. **剪影型**（约 1/3）
     · 整棵是实心图标：锥形粗树干 + 披针形**实心长矛状**下垂叶。

两型都**没有大片填充** —— v329 那种"实心风车/蒲公英"正是因为把叶画成了带锯齿的实心多边形，
覆盖率超标、与原图笔触语言完全不符（用户 2026-09-15 反馈"元素长得好看吗，跟原图有什么关系"）。

API 与旧版一致：draw_palm(size, cx, base_y, height, seed, lw_scale, style) / palm_layer(...)
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

SS = 3          # 超采样倍率


def _unit(dx: float, dy: float):
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _perp(tx, ty):
    ux, uy = _unit(tx, ty)
    return -uy, ux


def _rope_zigzag(a, b, amp: float, step: float):
    """a→b 的锯齿点列（左右交替偏移 → 弹簧状树干）。"""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L = math.hypot(dx, dy)
    nx, ny = _perp(dx, dy)
    n = max(2, int(L / max(1e-6, step)))
    return [(ax + dx * (i / n) + nx * (amp if i % 2 else -amp),
             ay + dy * (i / n) + ny * (amp if i % 2 else -amp)) for i in range(n + 1)]


def _spine(x0, y0, ang, length, steps, droop, wobble=0.0, rng=None):
    """从 (x0,y0) 沿 ang 出发、逐步加大下垂角度的脊线点列。"""
    pts = [(x0, y0)]
    a = ang
    seg = length / steps
    x, y = x0, y0
    for i in range(1, steps + 1):
        a += droop * (i / steps) ** 1.15
        if wobble and rng is not None:
            a += float(rng.uniform(-wobble, wobble))
        x += math.cos(a) * seg
        y += math.sin(a) * seg
        pts.append((x, y))
    return pts


def _taper_poly(pts, wfun, side_w):
    """脊线 + 宽度函数 → 闭合多边形（实心叶/实心树干用）。"""
    left, right = [], []
    n = max(1, len(pts) - 1)
    for i, (x, y) in enumerate(pts):
        t = i / n
        j, k = min(len(pts) - 1, i + 1), max(0, i - 1)
        nx, ny = _perp(pts[j][0] - pts[k][0], pts[j][1] - pts[k][1])
        w = max(0.6, wfun(t) * side_w)
        left.append((x + nx * w, y + ny * w))
        right.append((x - nx * w, y - ny * w))
    return left + right[::-1]


# ------------------------------------------------------------------ 叶
def _frond_line(d, ox, oy, ang, length, lw, rng, droop):
    """线稿叶：中脉 + 两侧短叶枝（像梳齿，彼此留空隙）；少数整叶为锯齿折线。

    ⚠️ 密度控制是这版的关键：叶枝必须**隔段画**、且叶枝长 < 段长，否则整棵树冠会缠成
    一团黑线（v330 首版 30% 锯齿 + 每段都画 → 树冠糊成一个墨疙瘩）。
    """
    steps = 10
    pts = _spine(ox, oy, ang, length, steps, droop, wobble=0.02, rng=rng)
    d.line(pts, fill=255, width=max(1, int(round(lw))), joint="curve")
    zig = rng.random() < 0.10
    seg = length / steps
    for i in range(2, steps + 1, 2):
        t = i / steps
        px, py = pts[i]
        qx, qy = pts[i - 1]
        pa = math.atan2(py - qy, px - qx)
        if zig:
            ll, lw2 = seg * 1.15 * (1.0 - 0.3 * t), lw * 0.80
        else:
            ll = min(length * float(rng.uniform(0.26, 0.38)) * (1.0 - 0.40 * t), seg * 0.92)
            lw2 = lw * 0.58
        for s in (-1, 1):
            la = pa + s * math.radians(float(rng.uniform(40, 60)))
            d.line([(px, py), (px + math.cos(la) * ll, py + math.sin(la) * ll)],
                   fill=255, width=max(1, int(round(lw2))))


def _frond_solid(d, ox, oy, ang, length, rng, droop, wmax):
    """剪影叶：实心披针形长矛（基部宽、尖端收细）。"""
    pts = _spine(ox, oy, ang, length, 12, droop, wobble=0.015, rng=rng)
    side = wmax * (1.0 + 0.22 * float(rng.uniform(-1, 1)))

    def wf(t):
        return (0.35 + 0.65 * math.sin(math.pi * min(1.0, t * 1.6)) ** 0.6) * (1.0 - t) ** 0.55
    d.polygon(_taper_poly(pts, wf, side), fill=255)


# ------------------------------------------------------------------ 单棵
def draw_palm(size, cx, base_y, height, seed, lw_scale: float = 1.0, style: str = "auto"):
    """画一棵棕榈，返回与原图同尺寸的 L 掩膜。"""
    W, H = size
    rng = np.random.default_rng(int(seed))
    if style in ("auto", None):
        style = "line" if rng.random() < 0.66 else "solid"

    lw = max(2.0, 0.0062 * W * lw_scale)
    canopy_r = height * float(rng.uniform(0.34, 0.46))
    trunk_len = height * float(rng.uniform(0.52, 0.66))
    top_y = base_y - trunk_len
    pad = int(canopy_r + 40 + lw * 4)
    x0, y0 = int(cx - pad), int(top_y - pad)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, int(cx + pad)), min(H, int(base_y + pad))
    if x1c - x0c <= 2 or y1c - y0c <= 2:
        return Image.new("L", size, 0)

    lwS = lw * SS
    im = Image.new("L", ((x1c - x0c) * SS, (y1c - y0c) * SS), 0)
    d = ImageDraw.Draw(im)
    T = lambda p: ((p[0] - x0c) * SS, (p[1] - y0c) * SS)      # noqa: E731

    # ---- 树干 ----
    sway = canopy_r * float(rng.uniform(-0.10, 0.10))
    top = (cx + sway, top_y)
    if style == "line":
        pts = _rope_zigzag((cx, base_y), top, amp=max(1.2, lw * 1.35), step=max(2.0, lw * 1.7))
        d.line([T(p) for p in pts], fill=255, width=max(1, int(round(lwS))), joint="curve")
    else:
        pts = _spine(cx, base_y, -math.pi / 2 + float(rng.uniform(-0.07, 0.07)),
                     trunk_len, 10, 0.10, wobble=0.01, rng=rng)
        pts = [(cx + (p[0] - cx) * 0.35, p[1]) for p in pts]
        d.polygon(_taper_poly(pts, lambda t: 0.55 + 0.95 * (1.0 - t) ** 1.1, lwS * 1.9), fill=255)

    # ---- 叶冠 ----
    n_fr = int(rng.integers(7, 9))
    a_lo = math.radians(-166 + float(rng.uniform(-6, 6)))
    a_hi = math.radians(-14 + float(rng.uniform(-6, 6)))
    for i in range(n_fr):
        t = (i + float(rng.uniform(0.15, 0.85))) / n_fr
        ang = a_lo + (a_hi - a_lo) * t
        flen = canopy_r * float(rng.uniform(0.98, 1.28))
        droop = math.radians(float(rng.uniform(26, 46)))
        # ⚠️ 叶长/叶宽必须一起乘 SS：脊线点会被 T() 放大 SS 倍，长度若仍用原图尺度
        #    就会画出"3 倍缩小"的叶（实测树冠只剩一小坨黑斑）。
        if style == "line":
            _frond_line(d, top[0], top[1], ang, flen * SS, lwS, rng, droop)
        else:
            _frond_solid(d, top[0], top[1], ang, flen * SS, rng, droop, lwS * 1.35)

    if style == "line":                        # 叶柄交汇处补一点墨，避免"空心"
        r = lw * 0.75
        d.ellipse([T((top[0] - r, top[1] - r)), T((top[0] + r, top[1] + r))], fill=255)

    local = np.asarray(im.resize((x1c - x0c, y1c - y0c), Image.LANCZOS), np.float32)
    out = np.zeros((H, W), np.float32)
    out[y0c:y1c, x0c:x1c] = local
    return Image.fromarray(out.astype(np.uint8), "L")


def palm_layer(size, specs, lw_scale: float = 1.0) -> Image.Image:
    """按 specs 叠加多棵棕榈（np.maximum），返回 L 掩膜。"""
    W, H = size
    acc = np.zeros((H, W), np.float32)
    for sp in specs:
        m = draw_palm(size, sp["cx"], sp["base_y"], sp["height"], sp.get("seed", 1),
                      lw_scale=lw_scale, style=sp.get("style", "auto"))
        acc = np.maximum(acc, np.asarray(m, np.float32))
    return Image.fromarray(acc.astype(np.uint8), "L")
