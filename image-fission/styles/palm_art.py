"""palm_art.py — 程序化"手绘线稿棕榈树"生成器（pinterest4 专用）。

为什么不用 SDXL 重画：实测 SDXL 出的是**实心黑剪影**，而原图是**手绘线稿**，
风格完全不同。程序化线稿能按原图的"墨线"语言（线宽/笔触/结构）复刻，同时每棵树的
倾斜角、树干弯度、叶片数量/长度/下垂度、树冠朝向都按 seed 独立生成 —— 真正做到
"裂变成不同角度构图的树木元素"，而不是复制同一棵树。

⚠️ 2026-09-15 重新标定（对照原图逐特征量测，v329 首版树干过细过长、叶片过稀过小）：
  原图（1242×1754）典型树总高 330~440px，其中
    树干长 ≈ 0.62×总高，树干宽 ≈ 0.042×总高（两侧描边 + 密排横梯格）
    树冠半径 ≈ 0.45×总高，叶脉宽 ≈ 0.024×总高，叶片为**边缘带锯齿的实心羽状条**
  → 全部线宽按 tree_height 的比例给，与分辨率无关。
"""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw

SS = 3  # 超采样


def _frond(d: ImageDraw.ImageDraw, base, ang: float, length: float, droop: float,
           w0: float, rng, teeth_scale: float = 1.0):
    """一根叶片 = **边缘带锯齿的实心羽状条**（原图笔触语言）。

    中心线沿 ang 起，末端按下垂 droop 弯；宽度包络中间最宽、两端收细；
    两侧交替外凸形成锯齿（sawtooth）——即原图那种"羽状/锯片状"叶形。
    """
    steps = max(12, int(length / (3.0 * SS)))
    jit = float(rng.uniform(-4.0, 4.0))          # 每根叶片只给**一个**角度抖动
    a0 = math.radians(ang + jit)
    pts = []
    for i in range(steps + 1):
        t = i / steps
        a = a0 + math.radians(droop * (t ** 1.9))
        pts.append((base[0] + math.cos(a) * length * t,
                    base[1] + math.sin(a) * length * t))

    def wid(t: float) -> float:
        env = math.sin(math.pi * min(1.0, max(0.03, t))) ** 0.42
        return max(1.0, w0 * env * (1.0 - 0.52 * t))

    left, right = [], []
    for i in range(steps + 1):
        t = i / steps
        j, k = min(steps, i + 1), max(0, i - 1)
        tx, ty = pts[j][0] - pts[k][0], pts[j][1] - pts[k][1]
        nl = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / nl, tx / nl
        w = wid(t)
        # 锯齿：两侧交替外凸（羽状/锯片语言）
        odd = (i % 2 == 1)
        wl = w * (1.38 if odd else 1.0)
        wr = w * (1.0 if odd else 1.38)
        left.append((pts[i][0] + nx * wl * teeth_scale, pts[i][1] + ny * wl * teeth_scale))
        right.append((pts[i][0] - nx * wr * teeth_scale, pts[i][1] - ny * wr * teeth_scale))
    d.polygon(left + right[::-1], fill=255)


def draw_palm(size, cx: float, base_y: float, height: float, seed: int,
              lw_scale: float = 1.0, style: str = "auto",
              tilt: float | None = None) -> Image.Image:
    """画一棵手绘线稿棕榈，返回 L 掩膜（size 像素坐标系，非超采样）。

    坐标：cx 树干根部横坐标，base_y 树根纵坐标，height 为**整树总高**（含树冠）。
    """
    rng = np.random.default_rng(int(seed) * 7919 + 13)
    W, H = size
    S = SS
    img = Image.new("L", (W * S, H * S), 0)
    d = ImageDraw.Draw(img)
    if style == "auto":
        style = ["droopy", "upright", "bushy", "palm_short"][int(rng.integers(0, 4))]

    hs = height * S                                  # 超采样下的总高
    trunk_len = hs * (0.60 + 0.09 * float(rng.random()))      # 0.58~0.68
    trunk_w = max(2.0, hs * 0.040 * lw_scale)                 # 树干宽
    edge_w = max(1.0, hs * 0.015 * lw_scale)                  # 描边线宽

    if tilt is None:
        tilt = float(rng.uniform(-20, 20))
    curve = float(rng.uniform(-0.26, 0.26)) * (1.0 if rng.random() < 0.7 else 1.5)

    bx, by = cx * S, base_y * S
    top_x = bx + math.sin(math.radians(tilt)) * trunk_len
    top_y = by - trunk_len
    steps = 26
    cen = []
    for i in range(steps + 1):
        t = i / steps
        x = bx + (top_x - bx) * t + math.sin(math.pi * t) * curve * trunk_len * 0.55
        y = by + (top_y - by) * t
        cen.append((x, y))

    # ---- 树干：两侧描边 + 密排横"梯格"（原图绳梯语言）----
    nor = []
    for i in range(steps + 1):
        j = min(steps, i + 1)
        tx, ty = cen[j][0] - cen[i][0], cen[j][1] - cen[i][1]
        nl = math.hypot(tx, ty) or 1.0
        t = i / steps
        nor.append((-ty / nl, tx / nl, (trunk_w * (1.0 - 0.45 * t)) / 2.0))
    edge_l = [(cen[i][0] + nor[i][0] * nor[i][2], cen[i][1] + nor[i][1] * nor[i][2])
              for i in range(steps + 1)]
    edge_r = [(cen[i][0] - nor[i][0] * nor[i][2], cen[i][1] - nor[i][1] * nor[i][2])
              for i in range(steps + 1)]
    ew = max(1, int(round(edge_w)))
    d.line(edge_l, fill=255, width=ew)
    d.line(edge_r, fill=255, width=ew)
    n_ring = max(7, int(trunk_len / (trunk_w * 0.95)))
    for j in range(n_ring):
        t = 0.05 + 0.92 * (j / max(1, n_ring - 1))
        i = min(steps, int(t * steps))
        if rng.random() < 0.10:
            continue
        px, py = cen[i]
        nx, ny, w = nor[i]
        span = w * float(rng.uniform(0.82, 1.02))
        d.line([(px - nx * span, py - ny * span), (px + nx * span, py + ny * span)],
               fill=255, width=max(1, int(round(edge_w * 1.15))))
    # 基部墨块
    d.ellipse([bx - trunk_w * 0.55, by - trunk_w * 0.6,
               bx + trunk_w * 0.55, by + trunk_w * 0.4], fill=255)

    # ---- 树冠：羽状叶片 ----
    top = cen[-1]
    if style == "droopy":
        n_fr, flen, droop, tw = int(rng.integers(12, 15)), hs * 0.33, 68.0, 0.027
    elif style == "upright":
        n_fr, flen, droop, tw = int(rng.integers(13, 16)), hs * 0.31, 34.0, 0.026
    elif style == "bushy":
        n_fr, flen, droop, tw = int(rng.integers(15, 18)), hs * 0.33, 50.0, 0.028
    else:  # palm_short
        n_fr, flen, droop, tw = int(rng.integers(11, 14)), hs * 0.35, 76.0, 0.029
    w0 = max(2.0, hs * tw * lw_scale)

    phase = float(rng.uniform(0, 360))
    for k in range(n_fr):
        frac = k / max(1, n_fr)
        ang = phase + frac * 360.0 + float(rng.uniform(-10, 10))
        L = flen * float(rng.uniform(0.80, 1.15))
        _frond(d, top, ang, L, droop * float(rng.uniform(0.8, 1.2)),
               w0 * float(rng.uniform(0.85, 1.2)), rng)
    rr = max(2, int(trunk_w * 0.55))
    d.ellipse([top[0] - rr, top[1] - rr, top[0] + rr, top[1] + rr], fill=255)

    return img.resize((W, H), Image.LANCZOS)


def palm_layer(size, specs, lw_scale: float = 1.0) -> Image.Image:
    """按 specs=[dict(cx,base_y,height,seed,style?,tilt?)...] 合成整层棕榈（L 掩膜）。"""
    acc = np.zeros((size[1], size[0]), np.float32)
    for sp in specs:
        m = np.asarray(draw_palm(size, sp["cx"], sp["base_y"], sp["height"],
                                 sp["seed"], lw_scale=lw_scale,
                                 style=sp.get("style", "auto"),
                                 tilt=sp.get("tilt")), np.float32) / 255.0
        acc = np.maximum(acc, m)
    return Image.fromarray((np.clip(acc, 0, 1) * 255).astype(np.uint8), "L")
