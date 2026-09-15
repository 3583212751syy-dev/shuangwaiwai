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
