"""palm_art.py — 程序化棕榈（v331：按原图 Pinterest(4).jpg 重写羽状复叶）

原图棕榈特征（v330 用户反馈"树元素裂变为什么非要做成这样"——v330 画的是蕨类非棕榈）：

  A. **线稿型**（约 2/3）
     · 树干 = **水平环纹横线**（间距 ≈ 2~3 线宽，略随树干倾斜），不是 zigzag/弹簧。
     · 叶  = **羽状复叶**（像羽毛/鱼骨）：长弯曲中脉(rachis) + 两侧**密排叶柄(pinnae)**
             每片叶有 ~18~26 对叶柄（非 v330 的 3~4 对）；
             叶柄基端长(≈中脉长的 28%~42%)、尖端短(≈8%~14%)；
             叶柄与中脉夹角 50°~72°（更张开）；
             整片叶呈宽弓形，droop 较大(40°~65°)。
     · 线宽 ≈ 7~8px（1242px 宽原图上）。

  B. **剪影型**（约 1/3）
     · 锥形粗树干 + 披针形实心下垂叶。

API: draw_palm(size, cx, base_y, height, seed, lw_scale, style) / palm_layer(...)
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


def _draw_trunk_rings(d, T, a, b, lw, rng):
    """在 d 上画横线环纹树干：沿 a→b 每隔间距画一条垂直于树干的短横线段。"""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L = math.hypot(dx, dy)
    if L < 1:
        return
    nx, ny = _perp(dx, dy)  # 横线方向（垂直于树干）
    spacing = lw * 2.2
    n = max(4, int(L / max(1e-6, spacing)))
    lw_draw = max(1, int(round(lw * 0.7)))
    for i in range(1, n):
        t = i / n
        cx = ax + dx * t
        cy = ay + dy * t
        rlen = lw * float(rng.uniform(2.2, 3.8))
        d.line([T((cx + nx * rlen, cy + ny * rlen)),
               T((cx - nx * rlen, cy - ny * rlen))],
              fill=255, width=lw_draw)


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
    """线稿羽状复叶（v331：匹配原图 Pinterest(4).jpg 棕榈）。

    结构 = 长弯曲中脉(rachis) + 两侧密排叶柄(pinnae)，像羽毛/鱼骨。
    与 v330 蕨类的区别：
      - 叶柄密度: ~20 对（v330 仅 4~5 对）
      - 叶柄长度: 基端 28%~42% canopy_r，尖端 8%~14%（v330 仅 16%~38%）
      - 张角: 50°~72°（v330 仅 36°~58°）
      - 下垂: 40°~65°（v330 仅 26°~46°）
    """
    steps = 20  # 密集分步 → 多对叶柄
    pts = _spine(ox, oy, ang, length, steps, droop, wobble=0.03, rng=rng)
    # 画中脉
    d.line(pts, fill=255, width=max(1, int(round(lw))), joint="curve")

    # 沿中脉每步画一对叶柄（跳过首尾两点）
    for i in range(2, steps):
        t = i / steps  # 0=基端, 1=尖端
        px, py = pts[i]
        qx, qy = pts[i - 1]
        pa = math.atan2(py - qy, px -qx)  # 当地中脉切向

        # 叶柄长度：基端长、尖端短（t^0.75 衰减，保留尖端一点可见度）
        base_len_frac = float(rng.uniform(0.22, 0.36))
        tip_len_frac = float(rng.uniform(0.06, 0.12))
        ll = length * (base_len_frac + (tip_len_frac - base_len_frac) * (t ** 0.75))

        # 叶柄线宽比中脉细
        lw2 = lw * float(rng.uniform(0.45, 0.65))

        # 张角：略随位置随机
        base_ang = math.radians(float(rng.uniform(50, 72)))
        ang_var = math.radians(float(rng.uniform(-8, 8))) * (1.0 - t)  # 基端变化更大

        for s in (-1, 1):
            la = pa + s * (base_ang + ang_var)
            ex = px + math.cos(la) * ll
            ey = py + math.sin(la) * ll
            d.line([(px, py), (ex, ey)], fill=255, width=max(1, int(round(lw2))))


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

    # ---- 树干（线稿=横线环纹，剪影=实心锥）----
    sway = canopy_r * float(rng.uniform(-0.10, 0.10))
    top = (cx + sway, top_y)
    if style == "line":
        _draw_trunk_rings(d, T, (cx, base_y), top, lwS, rng)
    else:
        pts = _spine(cx, base_y, -math.pi / 2 + float(rng.uniform(-0.07, 0.07)),
                     trunk_len, 10, 0.10, wobble=0.01, rng=rng)
        pts = [(cx + (p[0] - cx) * 0.35, p[1]) for p in pts]
        d.polygon(_taper_poly(pts, lambda t: 0.55 + 0.95 * (1.0 - t) ** 1.1, lwS * 1.9), fill=255)

    # ---- 叶冠（v331：更多叶、更大下垂弧度、更长的羽状复叶）----
    n_fr = int(rng.integers(8, 12))  # v330: 7~9 → 更丰满树冠
    a_lo = math.radians(-172 + float(rng.uniform(-8, 8)))
    a_hi = math.radians(-8 + float(rng.uniform(-8, 8)))
    for i in range(n_fr):
        t = (i + float(rng.uniform(0.10, 0.90))) / n_fr
        ang = a_lo + (a_hi - a_lo) * t
        flen = canopy_r * float(rng.uniform(1.05, 1.35))  # v330: 0.98~1.28 → 更长
        droop = math.radians(float(rng.uniform(40, 65)))   # v330: 26~46 → 更弯曲下垂
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
