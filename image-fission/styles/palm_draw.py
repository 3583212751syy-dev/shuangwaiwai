"""palm_draw.py (v2) —— 按 Pinterest(4) 原图风格「程序化重绘」棕榈。

原图风格（v364/v360 1:1 笔触级研究，jobs/v364_study/elems_*.jpg）：
  原图只有三类元素，且**线很细、很雅**：
  ① solid  —— 实心瘦棕榈：细实心微弯渐细树干 + 窄实心尖叶（放射后大幅下垂）
               + 3~5 片超长低垂叶
  ② scribble —— 涂鸦线棕榈：树干 = 一列**密集水平涂鸦短划**（弹簧/线圈感）；
               叶 = **细锯齿折线**（zigzag 一笔到底，尖梢下垂）
  ③ tuft  —— 只有放射锯齿叶的草丛（无干）
  签名特征 = 「锯齿折线叶」，细线（≈0.4% 图宽），振幅中等。

用法:
    from styles.palm_draw import make_palm
    spr = make_palm(H=420, seed=7, kind='scribble')   # PIL 'L' 图，墨=0 底=255
"""
import math
import random

from PIL import Image, ImageDraw

SS = 3                      # supersample
BLACK, BG = 0, 255


# ------------------------------------------------------------------ 基础几何
def _qbez(p0, p1, p2, t):
    u = 1.0 - t
    return (u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])


def _qtan(p0, p1, p2, t):
    u = 1.0 - t
    dx = 2 * u * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0])
    dy = 2 * u * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
    n = math.hypot(dx, dy) + 1e-9
    return dx / n, dy / n


def _dir(deg):
    """角度(0=正上,顺时针为正) -> 单位向量(y 向下为正)"""
    r = math.radians(deg)
    return math.sin(r), -math.cos(r)


def _tri(u):
    """三角波，值域 [-1,1]，用于锯齿折线"""
    u = u - math.floor(u)
    return 4.0 * abs(u - 0.5) - 1.0


def _blade_poly(p0, p1, p2, hw, taper=0.62, lobes=2, notch=0.16,
                n=64, base_lock=0.08):
    """窄实心叶多边形：沿二次贝塞尔脊线按半宽函数偏移。"""
    left, right = [], []
    for i in range(n + 1):
        t = i / n
        x, y = _qbez(p0, p1, p2, t)
        tx, ty = _qtan(p0, p1, p2, t)
        nx, ny = -ty, tx
        h = hw * ((1.0 - t) ** taper)
        if base_lock > 0:
            h *= min(1.0, t / base_lock) ** 0.55
        if lobes > 0 and t > 0.12:
            ph = ((t - 0.12) / 0.88) * lobes
            fr = ph - math.floor(ph)
            h *= 1.0 - notch * (1.0 - abs(2 * fr - 1.0))
        h = max(h, 0.0)
        left.append((x + nx * h, y + ny * h))
        right.append((x - nx * h, y - ny * h))
    return left + right[::-1]


def _chevron(d, p0, p1, p2, amp, wl, w, taper=0.45, ramp0=0.08, ramp1=0.30,
             spine=0.60, rng=None):
    """涂鸦叶：**粗人字锯齿缎带** + 细脊线 —— 原图 scribble 棕榈的签名冠形。

    v372 4x 实测（jobs/v372_p6measure/p4_R1.png）：原图那一批棕榈的叶不是
    "细波浪线+小羽刺"，而是沿脊线左右交替的大振幅**尖锐人字齿**
    （实测 amp≈0.028H、齿距≈0.031H、笔宽≈0.010H），齿尖逐齿抖动、
    整体向根部后掠，再叠一根细脊线从齿间穿过。
    """
    N = 400
    xs, ys = [], []
    for i in range(N + 1):
        x, y = _qbez(p0, p1, p2, i / N)
        xs.append(x); ys.append(y)
    L = 0.0
    for i in range(1, N + 1):
        L += math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
    L = max(L, 1.0)
    step = max(1e-4, (wl * 0.5) / L)
    pts = []
    k = 0
    while True:
        t = k * step
        if t > 1.0:
            break
        x, y = _qbez(p0, p1, p2, t)
        tx, ty = _qtan(p0, p1, p2, t)
        nx, ny = -ty, tx
        nn = math.hypot(nx, ny) + 1e-9
        nx, ny = nx / nn, ny / nn
        ramp = min(1.0, max(0.0, (t - ramp0) / max(1e-6, ramp1 - ramp0)))
        jit = rng.uniform(0.80, 1.16) if rng else 1.0
        a = amp * ramp * (1.0 - taper * t) * jit
        sd = 1.0 if (k % 2 == 0) else -1.0
        # 后掠：齿尖整体向根部方向偏一点
        sx, sy = _qtan(p0, p1, p2, max(0.0, t - 0.045))
        bx, by = sx - x, sy - y
        bb = math.hypot(bx, by) + 1e-9
        pts.append((x + nx * sd * a + bx / bb * a * 0.16,
                    y + ny * sd * a + by / bb * a * 0.16))
        k += 1
    if len(pts) >= 2:
        d.line(pts, fill=BLACK, width=w, joint='curve')
    if spine > 0:
        sp = [_qbez(p0, p1, p2, i / 60.0) for i in range(61)]
        d.line(sp, fill=BLACK, width=max(1, int(round(w * spine))), joint='curve')


def _zigline(d, p0, p1, p2, amp, wl, w, phase=0.0, taper=0.45,
             barb_len=0.0, slant=0.48, n=240, rng=None,
             ramp0=0.10, ramp1=0.36):
    """羽叶（原图签名笔触）：**近直脊线 + 两列后掠羽刺**。

    v369 4x 实测（jobs/v366_measure/frond_zoom2.jpg）：
      原图叶不是"锯齿带"，而是「一根粗脊线（微起伏）+ 密排短羽刺，羽刺
      向根部后掠 ~30°」，整叶呈羽毛/杉枝状。amp 很小（≈0.005H），
      羽刺长 ≈0.015H 且显著长于脊线起伏 —— 这是辨识度所在。
    """
    pts, s, marks = [], 0.0, []
    prev, last_k = None, None
    for i in range(n + 1):
        t = i / n
        x, y = _qbez(p0, p1, p2, t)
        tx, ty = _qtan(p0, p1, p2, t)
        nx, ny = -ty, tx
        if prev is not None:
            s += math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
        ramp = min(1.0, max(0.0, (t - ramp0) / max(1e-6, ramp1 - ramp0)))
        a = amp * ramp * (1.0 - taper * t)
        u = s / max(1.5, wl) + phase
        off = a * _tri(u)
        pts.append((x + nx * off, y + ny * off))
        k = math.floor(u)
        if last_k is None:
            last_k = k
        elif k != last_k:
            last_k = k
            marks.append(i)
    d.line(pts, fill=BLACK, width=w, joint='curve')
    if barb_len > 0:
        for m, i in enumerate(marks):
            t = i / n
            x, y = pts[i]
            tx, ty = _qtan(p0, p1, p2, t)
            nx, ny = -ty, tx
            ln = barb_len * (1.0 - 0.55 * t) * (rng.uniform(0.70, 1.28)
                                                if rng else 1.0)
            sd = 1.0 if (m % 2 == 0) else -1.0
            bx = nx * sd - tx * slant
            by = ny * sd - ty * slant
            nn = math.hypot(bx, by) + 1e-9
            d.line([(x, y), (x + bx / nn * ln, y + by / nn * ln)],
                   fill=BLACK, width=max(1, int(round(w * 0.92))))


def _tick_trunk(d, base, mid, top, wl, tick, w, rng, ticks_every=0.5):
    """阶梯纹树干：一根细脊线 + 密排短横档（原图最常见的树干语言）。

    v372 实测（jobs/v372_p6measure/p4_solid2.png）：原图多数树干**不是实心黑柱**，
    而是「细脊线 + 一排短横档」，宽 ≈ 0.03H、档长 ≈ 0.020H、档距 ≈ 0.014H。
    旧版画成实心多边形 → 整幅变"重黑柱"，与原文气质不符。
    """
    n = 60
    pts = [_qbez(base, mid, top, i / n) for i in range(n + 1)]
    d.line(pts, fill=BLACK, width=w, joint='curve')
    L = 0.0
    for i in range(1, len(pts)):
        L += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    nk = max(4, int(L / max(2.0, wl)))
    for k in range(nk + 1):
        t = k / nk
        i = min(n, int(round(t * n)))
        x, y = pts[i]
        dx, dy = _qtan(base, mid, top, t)
        nn = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / nn, dx / nn
        ln = tick * rng.uniform(0.72, 1.30)
        wdt = max(1, int(round(w * rng.uniform(0.80, 1.15))))
        both = rng.random() < 0.72
        d.line([(x - nx * ln * (1.0 if both else 0.85), y - ny * ln * (1.0 if both else 0.85)),
                (x + nx * ln, y + ny * ln)], fill=BLACK, width=wdt)


def _solid_trunk(d, base, mid, top, wb, wt, ticks=0, rng=None):
    """细实心渐细微弯树干（可带少量短刺）"""
    n = 48
    pl, pr = [], []
    for i in range(n + 1):
        t = i / n
        x, y = _qbez(base, mid, top, t)
        w = (wb + (wt - wb) * t) * 0.5
        pl.append((x - w, y))
        pr.append((x + w, y))
    d.polygon(pl + pr[::-1], fill=BLACK)
    if ticks > 0 and rng is not None:
        for i in range(ticks):
            t = 0.12 + 0.78 * (i + rng.uniform(0.15, 0.85)) / max(1, ticks)
            x, y = _qbez(base, mid, top, t)
            w = (wb + (wt - wb) * t) * 0.5
            sd = 1 if i % 2 == 0 else -1
            ln = w * rng.uniform(1.0, 1.7)
            d.line([(x + w * sd, y), (x + w * sd + ln * sd, y - ln * 0.35)],
                   fill=BLACK, width=max(1, int(w * 0.42)))


def _scribble_trunk(d, base, top, colw, step, w, rng):
    """涂鸦树干：一列**离散**水平涂鸦短划（每笔带一个尖钩，笔间留缝）。

    v372 实测：原图是「一笔一笔画上去」的水平 squiggle，笔距 ≈ 笔宽的 1.3~1.6 倍，
    中间**能看到背景缝**；旧版 step 太小 + 相干相位 → 读成一根连续弹簧/线圈。
    """
    L = math.hypot(top[0] - base[0], top[1] - base[1])
    n = max(6, int(L / max(2.0, step)))
    for i in range(n + 1):
        t = i / n
        x = base[0] + (top[0] - base[0]) * t
        y = base[1] + (top[1] - base[1]) * t
        half = colw * 0.5 * (1.0 - 0.22 * t) * rng.uniform(0.66, 1.12)
        k = 6
        ph = rng.uniform(0, 6.28)
        rev = 1.0 if rng.random() < 0.5 else -1.0
        pts = []
        for j in range(k + 1):
            u = j / k
            px = x - half + 2 * half * u
            py = y + rev * step * 0.52 * _tri(u * 1.35 + ph) * rng.uniform(0.55, 1.25)
            pts.append((px, py))
        d.line(pts, fill=BLACK, width=w, joint='curve')


# -------------------------------------------------------------------- 单株
def _crown(solid, d, crown, Hs, cs, rng, chevron=False):
    """冠部：放射叶 + 长低垂叶。chevron=True 用粗人字锯齿缎带（涂鸦棕榈）。"""
    if chevron:
        n = rng.randint(10, 14)
        w = max(2, int(round(Hs * rng.uniform(0.0142, 0.0192) * cs)))
        amp = Hs * rng.uniform(0.0175, 0.0255) * cs
        wl = Hs * rng.uniform(0.048, 0.066) * cs
        a_span = rng.uniform(104.0, 138.0)
        us = sorted(rng.uniform(-1.0, 1.0) for _ in range(n))
        for k in range(n):
            a_tip = a_span * math.copysign(abs(us[k]) ** 0.94, us[k])
            s = abs(math.sin(math.radians(a_tip)))
            L = Hs * (0.30 + 0.17 * s) * cs * rng.uniform(0.82, 1.18)
            a_base = a_tip * rng.uniform(0.06, 0.30)
            dx0, dy0 = _dir(a_base)
            dx1, dy1 = _dir(a_tip * rng.uniform(1.02, 1.22))
            p1 = (crown[0] + 0.55 * L * dx0, crown[1] + 0.55 * L * dy0)
            p2 = (crown[0] + L * dx1, crown[1] + L * dy1 + L * 0.07)
            _chevron(d, crown, p1, p2, amp, wl, w,
                     taper=rng.uniform(0.34, 0.58),
                     ramp0=0.07, ramp1=rng.uniform(0.24, 0.36),
                     spine=rng.uniform(0.48, 0.68), rng=rng)
        # 长低垂叶（外圈，齿更疏更大）
        for _ in range(rng.randint(3, 5)):
            a_tip = rng.choice([-1, 1]) * rng.uniform(120, 158)
            L = Hs * rng.uniform(0.34, 0.50) * cs
            dx0, dy0 = _dir(a_tip * rng.uniform(0.10, 0.26))
            dx1, dy1 = _dir(a_tip)
            p1 = (crown[0] + 0.58 * L * dx0, crown[1] + 0.58 * L * dy0)
            p2 = (crown[0] + L * dx1, crown[1] + L * dy1 + L * 0.05)
            _chevron(d, crown, p1, p2, amp * 1.10, wl * 1.22, w,
                     taper=rng.uniform(0.42, 0.62),
                     ramp0=0.08, ramp1=rng.uniform(0.26, 0.38),
                     spine=rng.uniform(0.45, 0.62), rng=rng)
        return
    if solid:
        # v372：叶数加到 16~21、叶宽 ×1.35、下垂加强 —— 原图冠是"喷泉状密排宽叶"，
        # 旧版 11~15 根细叶均匀放射 → 读成蒲公英/刺球。
        n = rng.randint(16, 21)
        a_span = rng.uniform(104.0, 134.0)
        us = sorted(rng.uniform(-1.0, 1.0) for _ in range(n))
        for k in range(n):
            a_tip = a_span * math.copysign(abs(us[k]) ** 0.92, us[k])
            s = abs(math.sin(math.radians(a_tip)))
            L = Hs * (0.235 + 0.155 * s) * cs * rng.uniform(0.78, 1.22)
            hw = Hs * (0.0175 + 0.0118 * s) * cs * rng.uniform(0.82, 1.16)
            a_base = a_tip * rng.uniform(0.10, 0.34)
            dx0, dy0 = _dir(a_base)
            dx1, dy1 = _dir(a_tip * rng.uniform(1.02, 1.24))
            p1 = (crown[0] + 0.58 * L * dx0, crown[1] + 0.58 * L * dy0)
            p2 = (crown[0] + L * dx1, crown[1] + L * dy1 + L * 0.10)
            d.polygon(_blade_poly(crown, p1, p2, hw,
                                  taper=rng.uniform(0.28, 0.42),
                                  lobes=rng.randint(2, 4),
                                  notch=rng.uniform(0.13, 0.24),
                                  base_lock=0.16), fill=BLACK)
        for _ in range(rng.randint(5, 8)):
            a_tip = rng.choice([-1, 1]) * rng.uniform(118, 166)
            L = Hs * rng.uniform(0.30, 0.48) * cs
            hw = Hs * rng.uniform(0.0155, 0.0215) * cs
            dx0, dy0 = _dir(a_tip * rng.uniform(0.10, 0.26))
            dx1, dy1 = _dir(a_tip)
            p1 = (crown[0] + 0.58 * L * dx0, crown[1] + 0.58 * L * dy0)
            p2 = (crown[0] + L * dx1, crown[1] + L * dy1 + L * 0.09)
            d.polygon(_blade_poly(crown, p1, p2, hw,
                                  taper=rng.uniform(0.42, 0.58),
                                  lobes=rng.randint(2, 3),
                                  notch=rng.uniform(0.16, 0.28),
                                  base_lock=0.15), fill=BLACK)
    else:
        n = rng.randint(12, 16)
        w = max(2, int(round(Hs * rng.uniform(0.0078, 0.0105))))
        amp = Hs * rng.uniform(0.0030, 0.0055)
        wl = Hs * rng.uniform(0.0165, 0.0245)
        bl = Hs * rng.uniform(0.0088, 0.0132)
        a_span = rng.uniform(108.0, 146.0)
        us = sorted(rng.uniform(-1.0, 1.0) for _ in range(n))
        for k in range(n):
            a_tip = a_span * math.copysign(abs(us[k]) ** 0.94, us[k])
            s = abs(math.sin(math.radians(a_tip)))
            L = Hs * (0.205 + 0.115 * s) * cs * rng.uniform(0.80, 1.20)
            a_base = a_tip * rng.uniform(0.06, 0.30)
            dx0, dy0 = _dir(a_base)
            dx1, dy1 = _dir(a_tip * rng.uniform(1.04, 1.26))
            p1 = (crown[0] + 0.55 * L * dx0, crown[1] + 0.55 * L * dy0)
            p2 = (crown[0] + L * dx1, crown[1] + L * dy1 + L * 0.06)
            _zigline(d, crown, p1, p2, amp, wl, w, barb_len=bl,
                     slant=rng.uniform(0.30, 0.62),
                     phase=rng.uniform(0, 6.28), taper=rng.uniform(0.30, 0.52),
                     ramp0=0.10, ramp1=0.34, n=260, rng=rng)


def make_palm(H, seed=0, kind=None, crown_scale=1.0):
    """生成一株棕榈 'L' 图（墨=0，底=255）。

    kind: None(随机) / 'solid' / 'scribble' / 'chevron' / 'tuft'
    """
    rng = random.Random(seed)
    cs = crown_scale
    W = int(H * 0.90)
    Ws, Hs = W * SS, H * SS
    img = Image.new('L', (Ws, Hs), BG)
    d = ImageDraw.Draw(img)

    if kind is None:
        kind = rng.choice(['solid', 'solid', 'chevron', 'scribble', 'tuft'])

    cx = Ws * (0.5 + rng.uniform(-0.015, 0.015))
    crown = (cx, Hs * (0.285 - 0.04 * (cs - 1.0)) * rng.uniform(0.94, 1.06))
    base = (cx + rng.uniform(-0.045, 0.045) * Ws, Hs * 0.995)
    bend = rng.uniform(-0.085, 0.085) * Ws
    mid = ((base[0] + crown[0]) / 2 + bend, (base[1] + crown[1]) / 2)

    if kind == 'tuft':
        crown = (cx, Hs * rng.uniform(0.30, 0.46))
    elif kind == 'stout':
        wb = Hs * rng.uniform(0.021, 0.028)
        wt = wb * rng.uniform(0.34, 0.50)
        _solid_trunk(d, base, mid, crown, wb, wt,
                     ticks=rng.randint(5, 9) if rng.random() < 0.18 else 0,
                     rng=rng)
    elif kind in ('solid', 'tuft'):
        # v372：实心/顶簇也用「细脊线 + 密横档」树干（原图主流树干语言）
        wl = Hs * rng.uniform(0.0135, 0.0195) * cs
        tick = Hs * rng.uniform(0.0125, 0.0195) * cs
        w = max(2, int(round(Hs * rng.uniform(0.0060, 0.0090) * cs)))
        _tick_trunk(d, base, mid, crown, wl, tick, w, rng)
    else:
        colw = Hs * rng.uniform(0.055, 0.075) * (1.10 if kind == 'chevron' else 1.0)
        step = Hs * rng.uniform(0.0130, 0.0175) * (1.0 if kind == 'chevron' else 0.70)
        w = max(2, int(round(Hs * rng.uniform(0.0108, 0.0142) * cs)))
        _scribble_trunk(d, base, crown, colw, step, w, rng)

    rr = Hs * 0.013 * cs
    d.ellipse([crown[0] - rr, crown[1] - rr, crown[0] + rr, crown[1] + rr],
              fill=BLACK)
    _crown(kind == 'solid', d, crown, Hs, cs, rng, chevron=(kind == 'chevron'))
    return img.resize((W, H), Image.LANCZOS)


if __name__ == '__main__':
    ORIG = 'E:/Desktop/图裂变测试图/Pinterest (4).jpg'
    src = Image.open(ORIG).convert('RGB')
    boxes = [(32, 170, 232, 715), (534, 81, 734, 490), (966, 344, 1166, 780),
             (28, 1404, 228, 1700), (940, 60, 1140, 400), (1042, 1050, 1242, 1450)]
    H = 430
    tops = []
    for b in boxes:
        c = src.crop(b)
        sc = H / c.height
        tops.append(c.resize((max(1, int(c.width * sc)), H), Image.LANCZOS))
    cfgs = [dict(seed=201, kind='solid', crown_scale=1.0),
            dict(seed=202, kind='scribble', crown_scale=1.05),
            dict(seed=203, kind='tuft', crown_scale=0.95),
            dict(seed=204, kind='solid', crown_scale=1.15),
            dict(seed=205, kind='scribble', crown_scale=0.9),
            dict(seed=206, kind='scribble', crown_scale=1.2)]
    bots = [make_palm(H, **c).convert('RGB') for c in cfgs]
    gap = 16
    Wtot = max(sum(p.width for p in tops), sum(p.width for p in bots)) + gap * 8
    sheet = Image.new('RGB', (Wtot, H * 2 + gap * 3 + 20), (205, 205, 205))
    x = gap
    for p in tops:
        sheet.paste(p, (x, gap)); x += p.width + gap
    x = gap
    for p in bots:
        sheet.paste(p, (x, H + gap * 2), ); x += p.width + gap
    sheet.save('jobs/v364_study/palm_cmp_v6.jpg', quality=95)
    print('saved palm_cmp_v6.jpg', sheet.size)
