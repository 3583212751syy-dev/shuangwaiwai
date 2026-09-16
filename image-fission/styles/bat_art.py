"""bat_art.py — 程序化"纹章蝙蝠"生成器（6978 主体裂变专用）。

为什么不用 SDXL 重画：实测 SDXL 出的是卡通剪影（带眼睛的小蝙蝠），与原图
**纹章式**蝙蝠（尖翼 + 扇贝形翼缘 + 长尾 + 尖耳 + 亮紫描边）气质完全不同。

为什么必须重画而不是形变原图：原蝙蝠是 BACARDÍ 注册商标图形（硬规则🔴5），
**绝不能作为变体出现**，必须原位改写成"不同角度/设计细节"的另一只蝙蝠。

本模块按 seed 生成：翼展角（上扬/平展/下掠）、扇贝数、翼尖钩形、耳形、尾长，
并输出 (dark_body, light_edge) 两层掩膜以便复刻原图的"深紫主体 + 亮紫描边"。
"""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw

SS = 3


def draw_bat(size, cx: float, cy: float, span: float, seed: int,
             pose: str = "auto", edge_px: float = 5.0):
    """返回 (dark, light) 两个 L 掩膜（size 像素坐标系）。

    cx,cy：徽标中心；span：单侧翼展（像素）。
    pose：wing_up（上扬）/ wing_mid（平展）/ wing_low（下掠）。
    """
    rng = np.random.default_rng(int(seed) * 6151 + 29)
    W, H = size
    S = SS
    dk = Image.new("L", (W * S, H * S), 0)
    lt = Image.new("L", (W * S, H * S), 0)
    d = ImageDraw.Draw(dk)
    dl = ImageDraw.Draw(lt)

    if pose == "auto":
        pose = ["wing_up", "wing_mid", "wing_low"][int(rng.integers(0, 3))]
    # 翼"上扬量"：0=平展(翼尖齐头)，1=高扬(翼尖远高于头顶)
    up = {"wing_up": 1.00, "wing_mid": 0.62, "wing_low": 0.30}[pose]
    span *= float(rng.uniform(0.94, 1.08))
    X, Y = cx * S, cy * S
    SP = span * S                       # 半翼展
    ed = max(2, int(round(edge_px * S / 2.0)))

    def P(u, v):
        """归一化局部坐标 (u=横向/半翼展, v=纵向/半翼展) -> 画布坐标。"""
        return (X + u * SP, Y + v * SP)

    # ---- 双翼：肩部起、翼尖高扬，翼缘带扇贝齿 ----
    n_sc = int(rng.integers(3, 5))
    tip_v = -0.30 - 0.42 * up                      # 翼尖纵向位置（明显高于头顶）
    for s in (-1, 1):
        pts = [P(s * 0.09, -0.25),                 # 肩（贴身体上部）
               P(s * 0.45, -0.36 - 0.20 * up),
               P(s * 0.78, tip_v + 0.12),
               P(s * 1.00, tip_v)]                 # 翼尖
        # 翼缘：从翼尖向下扫回身体，交替内外点形成扇贝
        sc = [(0.88, 0.04), (0.76, -0.05), (0.63, 0.16), (0.50, 0.06),
              (0.37, 0.28), (0.25, 0.16), (0.14, 0.36)]
        for j in range(n_sc * 2 + 1):
            u, v = sc[j % len(sc)]
            v = v * float(rng.uniform(0.88, 1.14))
            pts.append(P(s * u, v))
        pts.append(P(s * 0.075, 0.42))
        d.polygon(pts, fill=255)

    # ---- 身体：颈 -> 尾根 的渐细柱 ----
    body_top = Y - SP * 0.30
    body_bot = Y + SP * 0.46
    nb = 26
    lf, rt = [], []
    for i in range(nb + 1):
        t = i / nb
        yy = body_top + (body_bot - body_top) * t
        wdt = SP * (0.105 * (1.0 - t ** 1.6) + 0.016)
        lf.append((X - wdt, yy))
        rt.append((X + wdt, yy))
    d.polygon(lf + rt[::-1], fill=255)

    # ---- 头 + 双耳 ----
    hr = SP * 0.105
    head_y = Y - SP * 0.40
    d.ellipse([X - hr, head_y - hr, X + hr, head_y + hr * 1.05], fill=255)
    ear_h = SP * float(rng.uniform(0.14, 0.21))
    ear_w = hr * float(rng.uniform(0.75, 1.0))
    for s in (-1, 1):
        d.polygon([(X + s * hr * 0.25, head_y - hr * 0.65),
                   (X + s * (hr * 0.25 + ear_w * 0.55), head_y - hr * 0.75 - ear_h),
                   (X + s * (hr * 0.25 + ear_w), head_y - hr * 0.20)], fill=255)

    # ---- 尾（短而略粗，不越出徽章椭圆）----
    tail_len = SP * float(rng.uniform(0.34, 0.52))
    bend = float(rng.uniform(-0.08, 0.08)) * SP
    tw1, tw2 = SP * 0.034, SP * 0.010
    d.polygon([(X - tw1, body_bot - SP * 0.03), (X + tw1, body_bot - SP * 0.03),
               (X + bend + tw2, body_bot + tail_len), (X + bend - tw2, body_bot + tail_len)],
              fill=255)

    # ---- 亮紫描边：翼上缘细线 + 耳内侧 ----
    for s in (-1, 1):
        dl.line([P(s * 0.13, -0.29), P(s * 0.49, -0.40 - 0.20 * up),
                 P(s * 0.98, tip_v + 0.03)], fill=255, width=ed)
    for s in (-1, 1):
        dl.line([(X + s * hr * 0.35, head_y - hr * 0.60),
                 (X + s * (hr * 0.25 + ear_w * 0.62), head_y - hr * 0.72 - ear_h * 0.88)],
                fill=255, width=max(2, ed - 1))

    dark = dk.resize((W, H), Image.LANCZOS)
    light = lt.resize((W, H), Image.LANCZOS)
    return dark, light


# ------------------------------------------------------------------ v2 参数化
def draw_bat_v2(size, cx: float, cy: float, spec: dict):
    """参数化纹章蝙蝠 v2 —— 主体裂变用：头(抬头/耳/眼/吻)、翼(姿态/展/扇贝/尖钩)、
    身体(顶/底/宽)、尾(长/粗/摆) 全部显式可调，输出 dark/light 两层 L 掩膜。

    spec 键（均可缺省）：
      span 单侧翼展(px) | pose up|mid|low | tip_hook 翼尖额外上扬(×span)
      scallops 扇贝对数 | head_lift 抬头量(×span) | head_r 头半径(×span)
      ear_h/ear_w/ear_spread 耳(×span) | eyes 亮紫眼弧 | snout 吻长(×span)
      body_top/body_bot 身体上/下端(×span, 相对中心, 负=上) | body_w 身体半宽(×span)
      tail_len 尾长(×span) | tail_w 尾根半宽(×span) | bend 尾摆(×span)
      edge_px 描边宽(px) | seed 抖动种子
    返回 (dark, light, meta)；meta['tips'] = 左右翼尖 (x,y)。
    """
    sp = dict(spec)
    span = float(sp.get('span', 250.0))
    up = {'up': 1.00, 'mid': 0.60, 'low': 0.28}[sp.get('pose', 'up')]
    hook = float(sp.get('tip_hook', 0.06))
    n_sc = int(sp.get('scallops', 5))
    head_lift = float(sp.get('head_lift', 0.06))
    head_r = float(sp.get('head_r', 0.115))
    ear_h = float(sp.get('ear_h', 0.21))
    ear_w = float(sp.get('ear_w', 0.98))
    ear_spread = float(sp.get('ear_spread', 0.34))
    eyes = bool(sp.get('eyes', True))
    snout = float(sp.get('snout', 0.055))
    body_top = float(sp.get('body_top', -0.26))
    body_bot = float(sp.get('body_bot', 0.50))
    body_w = float(sp.get('body_w', 0.115))
    tail_len = float(sp.get('tail_len', 0.52))
    tail_w = float(sp.get('tail_w', 0.030))
    bend = float(sp.get('bend', 0.0))
    edge_px = float(sp.get('edge_px', 5.0))
    rng = np.random.default_rng(int(sp.get('seed', 7)) * 7717 + 13)

    W, H = size
    S = SS
    dk = Image.new("L", (W * S, H * S), 0)
    lt = Image.new("L", (W * S, H * S), 0)
    d = ImageDraw.Draw(dk)
    dl = ImageDraw.Draw(lt)
    X, Y, SP = cx * S, cy * S, span * S
    ed = max(2, int(round(edge_px * S / 2.0)))

    def P(u, v):
        return (X + u * SP, Y + v * SP)

    # ---- 双翼：肩 -> 内段 -> 外段 -> 翼尖，翼缘带扇贝齿回到身体下部 ----
    tip_v = -0.30 - 0.42 * up - hook
    Nsc = 2 * n_sc + 1
    for s in (-1, 1):
        pts = [P(s * 0.085, body_top + 0.03),
               P(s * 0.42, body_top - 0.08 - 0.18 * up),
               P(s * 0.74, tip_v + 0.11),
               P(s * 1.00, tip_v)]
        for j in range(Nsc):
            t = j / (Nsc - 1)
            u = 0.86 - 0.72 * t
            vb = 0.04 + 0.34 * t
            jit = float(rng.uniform(-0.03, 0.03))
            v = vb + (0.055 if j % 2 else -0.015) + jit
            pts.append(P(s * u, v))
        pts.append(P(s * 0.070, body_bot - 0.06))
        d.polygon(pts, fill=255)

    # ---- 身体：颈 -> 尾根 渐细柱 ----
    y0_, y1_ = Y + body_top * SP, Y + body_bot * SP
    nb = 26
    lf, rt = [], []
    for i in range(nb + 1):
        t = i / nb
        yy = y0_ + (y1_ - y0_) * t
        wdt = SP * (body_w * (1.0 - t ** 1.6) + 0.016)
        lf.append((X - wdt, yy))
        rt.append((X + wdt, yy))
    d.polygon(lf + rt[::-1], fill=255)

    # ---- 头（正面 + 抬头）+ 双耳 + 吻部 ----
    hr = SP * head_r
    head_y = Y + (body_top - 0.10 - head_lift) * SP
    d.ellipse([X - hr, head_y - hr * 0.98, X + hr, head_y + hr * 1.06], fill=255)
    if snout > 0:
        sw = SP * snout
        d.polygon([(X - sw * 0.62, head_y + hr * 0.72),
                   (X + sw * 0.62, head_y + hr * 0.72),
                   (X, head_y + hr * 0.72 + sw)], fill=255)
    ehh, eww = SP * ear_h, hr * ear_w
    for s in (-1, 1):
        bx = X + s * hr * ear_spread
        d.polygon([(bx - s * eww * 0.34, head_y - hr * 0.52),
                   (bx + s * eww * 0.22, head_y - hr * 0.86 - ehh),
                   (bx + s * eww * 0.66, head_y - hr * 0.10)], fill=255)

    # ---- 尾：细长 + 可带摆动 ----
    tw1, tw2 = SP * tail_w, SP * tail_w * 0.30
    bt = Y + (body_bot - 0.03) * SP
    d.polygon([(X - tw1, bt), (X + tw1, bt),
               (X + bend * SP + tw2, bt + tail_len * SP),
               (X + bend * SP - tw2, bt + tail_len * SP)], fill=255)

    # ---- 亮紫描边：翼上缘线 + 耳内线 + 眼弧 ----
    for s in (-1, 1):
        dl.line([P(s * 0.13, body_top - 0.02),
                 P(s * 0.46, body_top - 0.12 - 0.18 * up),
                 P(s * 0.96, tip_v + 0.05)], fill=255, width=ed)
    for s in (-1, 1):
        bx = X + s * hr * ear_spread
        dl.line([(bx - s * eww * 0.28, head_y - hr * 0.50),
                 (bx + s * eww * 0.20, head_y - hr * 0.84 - ehh * 0.88)],
                fill=255, width=max(2, ed - 1))
    if eyes:
        er = hr * 0.34
        for s in (-1, 1):
            ex, ey = X + s * hr * 0.44, head_y + hr * 0.02
            dl.arc([ex - er, ey - er * 0.78, ex + er, ey + er * 1.14],
                   start=196, end=352, fill=255, width=max(2, int(ed * 0.85)))

    dark = dk.resize((W, H), Image.LANCZOS)
    light = lt.resize((W, H), Image.LANCZOS)
    meta = dict(tips=[(cx - span, cy + tip_v * span), (cx + span, cy + tip_v * span)],
                tip_v=tip_v, span=span)
    return dark, light, meta
