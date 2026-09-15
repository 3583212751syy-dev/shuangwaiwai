"""subject_morph.py — 主体"结构保持裂变"（v330）

用户反馈（2026-09-15）：
  · "这主体在原图结构的基础上去裂变跟原图有关联做不了吗"（6978 蝙蝠被画成紫色团块）
  · "主体看起来很不舒服，全是噪点"（pinterest3 蝴蝶由 SDXL 重绘后贴回，噪声/糊边/接缝）

为什么不再用 SDXL 重绘主体：
  SDXL img2img 会把主体的解剖结构、材质、边缘全部重解释 —— 结果要么变成"另一个东西"
  （与原图无关），要么带大量生成噪声/糊边，且合成回原位必然留接缝。

本模块的做法（真·结构裂变）：把**原图主体本身**当作几何源，施加**平滑位移场**改变姿态
（翼展角 / 左右翼上扬量 / 展开率 / 身尾比例），再原样合成回**原位**：
  · 结构/材质/边缘/投影 100% 来自原图 → 0 生成噪声、0 接缝、天然"与原图有关联"；
  · 位移场随位置变化 → 翼尖/翅缘姿态真的变了（同构异形）；
  · 位移场平滑且**只作用在主体像素层**上 → 背景（圆盘环/迷彩/灰底）一个像素都不动。

配套：抹掉旧主体的"干净底板"用 textfix.erase(..., method='diffuse'/'nn') 生成，
新姿态主体层再叠上去 —— 主体移位让出的空隙由底板自然过渡填充，不会出现色块。
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


# ------------------------------------------------------------------ 数学工具
def smoothstep(x, lo=0.0, hi=1.0):
    """标准 smoothstep：x<=lo → 0，x>=hi → 1，中间三次平滑。"""
    t = np.clip((np.asarray(x, np.float32) - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bbox_of(mask, pad=0):
    ys, xs = np.where(mask)
    if not len(ys):
        return None
    H, W = mask.shape
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(W, int(xs.max()) + 1 + pad), min(H, int(ys.max()) + 1 + pad))


# ------------------------------------------------------------------ 主体层
def make_layer(img, mask, feather=2.0, box=None, pad=6):
    """取出主体 RGBA 层：RGB = 原图像素，A = 掩膜（羽化 feather 像素）。

    返回 (rgba[float32 0..255, H,W,4], (x0,y0))，坐标是**原图坐标系**下的左上角。
    """
    a = np.asarray(img, np.float32)
    if box is None:
        box = _bbox_of(mask, pad=pad)
    x0, y0, x1, y1 = box
    m = mask[y0:y1, x0:x1].astype(np.float32)
    if feather > 0:
        m = ndi.gaussian_filter(m, feather)
    m = np.clip(m / max(1e-6, m.max()), 0.0, 1.0)
    rgb = a[y0:y1, x0:x1]
    rgba = np.concatenate([rgb, (m * 255.0)[..., None]], axis=2)
    return rgba, (x0, y0)


def warp_layer(rgba, disp, order=1):
    """按位移场 warp RGBA 层。disp=(dy,dx)，与层同尺寸；语义：out[y,x] = src[y+dy, x+dx]。

    order=1（双线性）→ 边缘保持锐利；层不大，三通道 + alpha 一起插值。
    """
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy, dx = disp
    coords = [yy + dy, xx + dx]
    out = np.empty_like(rgba)
    for c in range(rgba.shape[2]):
        out[..., c] = ndi.map_coordinates(rgba[..., c], coords, order=order,
                                          mode='nearest', prefilter=False)
    return out


def paste_layer(base_img, rgba, box):
    """把 RGBA 层按 alpha 合成回 base（PIL 图）→ 新 PIL 图。"""
    x0, y0 = box
    a = np.asarray(base_img, np.float32).copy()
    h, w = rgba.shape[:2]
    reg = a[y0:y0 + h, x0:x0 + w]
    al = np.clip(rgba[..., 3:4] / 255.0, 0.0, 1.0)
    a[y0:y0 + h, x0:x0 + w] = reg * (1.0 - al) + rgba[..., :3] * al
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB')


# ------------------------------------------------------------------ 位移场
def wing_shear(size, cx, cy, k_up=0.0, ramp=(30.0, 240.0), span=1.0,
               k_down_body=0.0, body_band=34.0, body_y=0.0, body_ramp=70.0,
               tilt=0.0):
    """翅膀式剪切的位移场（返回 (dy, dx)）。

    参数
    ----
    cx, cy   : 主体轴心（身体中轴 x、肩部 y）
    k_up     : 翼尖上扬量（px）。>0 = 翼尖抬高。按 |x-cx| 用 smoothstep(ramp) 加权，
               身体附近权重 0 → 身体不动，翼随距离渐次上扬（= 姿态变化）。
    span     : 翼展缩放（1.06 = 展宽 6%）。
    tilt     : 附加的整层旋转角（弧度，绕 (cx,cy)，按 |x-cx| 加权）—— 用于让翼"前掠/后掠"。
    k_down_body: 身体/尾下延量（px，仅作用于 |x-cx|<body_band 且 y>body_y 的区域）。
    """
    h, w = size[1], size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx0 = xx - float(cx)
    dy0 = yy - float(cy)
    wgt = smoothstep(np.abs(dx0), ramp[0], ramp[1])

    # 竖向：翼尖上扬（out[y] 取 src[y+k*w] → 形状上移）
    dy = k_up * wgt
    # 身体/尾：向下延伸（out[y] 取 src[y-k] → 形状下移）
    if k_down_body:
        bw = (1.0 - smoothstep(np.abs(dx0), body_band * 0.5, body_band)) * \
             smoothstep(dy0, body_y - float(cy), body_y - float(cy) + body_ramp)
        dy = dy - k_down_body * bw
    # 横向：翼展缩放 + 附加倾斜
    dx = dx0 * (1.0 / float(span) - 1.0)
    if tilt:
        ang = tilt * wgt * np.sign(dx0)
        ca, sa = np.cos(ang), np.sin(ang)
        rx = dx0 * ca - dy0 * sa
        ry = dx0 * sa + dy0 * ca
        dx = dx + (rx - dx0) * (1.0 / float(span))
        dy = dy + (ry - dy0)
    return dy.astype(np.float32), dx.astype(np.float32)


def wing_pose(size, cx, cy, theta=0.0, pivot_dx=34.0, ramp=(28.0, 235.0),
              k_up=0.0, span=1.0, tip_flick=0.0, tip_ramp=(180.0, 260.0),
              body_scale=1.0, body_y0=0.0, body_ramp=90.0, tail_stretch=0.0,
              tail_y=0.0):
    """"展翼姿态"位移场：左右翼各绕**肩部枢轴**旋转 theta 弧度 + 可选附加项。

    语义：out[y,x] = src[y+dy, x+dx]（正向位移场）。
    · theta>0 = 双翼上扬收拢；theta<0 = 双翼下压外展。旋转以 (cx±pivot_dx, cy) 为枢轴，
      权重随 |x-cx| 用 smoothstep(ramp) 渐入 → 身体（|dx| 小）纹丝不动，翼随距离逐渐转。
    · tip_flick：仅翼尖（|dx| 超过 tip_ramp[0]）额外上/下甩（px）——让翼尖"勾"起来。
    · body_scale：身体/头（|dx|<pivot_dx*1.4）纵向缩放，改身型比例。
    · tail_stretch：尾部（|dx| 小且 y>tail_y）纵向拉伸 px（>0 拉长、<0 缩短）。
    """
    h, w = size[1], size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx0 = xx - float(cx)
    dy0 = yy - float(cy)
    dx = np.zeros_like(dx0)
    dy = np.zeros_like(dy0)

    # --- 双翼旋转（绕右/左肩枢轴） ---
    if theta:
        wgt = smoothstep(np.abs(dx0), ramp[0], ramp[1])
        sgn = np.where(dx0 >= 0, 1.0, -1.0)
        px = float(cx) + sgn * float(pivot_dx)
        py = float(cy)
        u = xx - px
        v = yy - py
        ang = float(theta) * (-sgn) * wgt
        ca, sa = np.cos(ang), np.sin(ang)
        u2 = u * ca - v * sa
        v2 = u * sa + v * ca
        dx += (u2 - u)
        dy += (v2 - v)

    # --- 竖向剪切（翼尖整体抬高/压低） ---
    if k_up:
        wgt = smoothstep(np.abs(dx0), ramp[0], ramp[1])
        dy += float(k_up) * wgt

    # --- 翼尖额外甩动 ---
    if tip_flick:
        tf_ = smoothstep(np.abs(dx0), tip_ramp[0], tip_ramp[1])
        dy += float(tip_flick) * tf_

    # --- 翼展 ---
    if span != 1.0:
        dx += dx0 * (1.0 / float(span) - 1.0)

    # --- 身体纵向比例 ---
    if body_scale != 1.0:
        bw = 1.0 - smoothstep(np.abs(dx0), 0.0, float(pivot_dx) * 1.5)
        dy += (yy - float(body_y0)) * (1.0 / float(body_scale) - 1.0) * bw

    # --- 尾长 ---
    if tail_stretch:
        tw = (1.0 - smoothstep(np.abs(dx0), 6.0, 22.0)) * \
             smoothstep(yy, float(tail_y), float(tail_y) + float(body_ramp))
        dy -= float(tail_stretch) * tw
    return dy.astype(np.float32), dx.astype(np.float32)


def carve_notch(mask, box, apex, base_l, base_r, depth_frac=1.0):
    """在 mask 的翼缘上"刻"一个 V 形凹口（= 设计细节差异）。

    apex/base_l/base_r 都在**层坐标系**。刻掉的是 apex 与底边两侧点围成的三角形，
    朝 mask 内部方向（apex 指向翼内）。返回新 mask（不改原 mask）。
    """
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ax, ay = apex
    lx, ly = base_l
    rx, ry = base_r
    def _side(px, py, qx, qy):
        return (qx - px) * (yy - py) - (qy - py) * (xx - px)
    s1 = _side(lx, ly, rx, ry)
    s2 = _side(rx, ry, ax, ay)
    s3 = _side(ax, ay, lx, ly)
    inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    return mask & (~inside)
