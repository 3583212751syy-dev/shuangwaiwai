"""v345e: p4 棕榈树「结构级重组」—— 用原图自身的矢量墨迹重组出全新的树。

为什么必须走这条路（本轮 3 个方向全部实测证伪的总结）：
  · SDXL 整图重生（ProteusV0.4，6 组 prompt/denoise/CN/LoRA 扫参）→ 棕榈被画成
    **软气刷水彩**，二值化后成"绒毛毛球"（jobs/v342_rebirth/_p4_p4A_dm_noLora.jpg）。
  · SDXL 逐棵树裁块重生（34 棵各自实心包络 + 上采样到 1024）→ 分辨率够了，但
    渲染风格仍是软的，二值化后是"毛边糊团"。
  · 换 anime/flat 底模 CounterfeitXL 0.90 → 同样毛边糊团。
  · 程序化 palm_art 重画 → 输出是"刺球"（v330 用户已否过："树元素为什么非要做成这样"）。
  ⇒ **扩散模型无法复刻这张图的矢量硬边棕榈线稿**；要保持"不比原图丑"，只能
    在原图自身的墨迹上做**结构级重组**（🔴2 的"结构级重生"精神：异内容同构）。

三个层次的"真变化"（都不是整体旋转那种"位移"）：
  ① **极坐标平滑重组**：以树冠中心为极点，φ' = φ + w(φ)、r' = r·(1+g(φ))，
     w/g 是多阶正弦（低频、连续）——每根叶被**各自**转过不同角度、拉长/缩短不同比例，
     而中心天然连续、无接缝无空洞（tree_wind 的失败点是"整棵一起弯"，这里是逐叶变）。
  ② **叶数增减**：按角度带复制一根叶 / 抹掉一根叶的外段 → 叶冠丰缺关系改变。
  ③ **树干**：倾角 ±7°、长度 ±10%、环纹间距 ±15%（绕树基作用，不动根部位置）。
  ④ **跨树换叶**：从另一棵树借一根叶（旋转缩放后 max 合成）→ 冠型不再是原树。

全程在 **float alpha 空间**用 ×2 超采样 + `map_coordinates` 双线性重采样，
最后 α 混合（camo*(1-a) + black*a）→ 边缘抗锯齿质量与原图同档。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp            # noqa: E402
import v342_rebirth as R                               # noqa: E402
from v345c_p4seg import trunks, partition, _disk, keep_big  # noqa: E402

OUT = ROOT / "jobs" / "v345e_p4crown"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
BLACK = np.array([12.0, 10.0, 9.0], np.float32)
SS = 3


# ------------------------------------------------------------------ 极坐标重组
def _harm(phi, amps, phases, offs=0.0):
    """多阶正弦叠加：Σ a_k sin(kφ + p_k) + off。"""
    out = np.full_like(phi, offs, np.float32)
    for k, (a, p) in enumerate(zip(amps, phases), start=1):
        out += a * np.sin(k * phi + p)
    return out


def polar_remap(alpha, c, R, amps_w, ph_w, amps_g, ph_g, iters=4,
                phi_off=0.0, r_gain=1.0):
    """以 c 为极点做 φ'=φ+w(φ)、r'=r*(1+g(φ)) 的重采样（逆映射不动点迭代）。

    ⚠️ 关键：位移必须**按半径 taper 到 0**（r→R 时位移归零）。否则 r≈R 的过渡
    环带上"变形后的内容"和"原内容"以 50/50 混在一起 → 出现肉眼可见的**重影/糊边**
    （第一版就是这样，读起来像"比原图丑"）。
    """
    h, w = alpha.shape
    up = np.repeat(np.repeat(alpha, SS, 0), SS, 1)
    H2, W2 = up.shape
    yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
    dx = (xx - c[0] * SS) / SS
    dy = (yy - c[1] * SS) / SS
    rp = np.sqrt(dx * dx + dy * dy)

    def tap(r):
        t = np.clip((R - r) / (0.30 * R + 1e-6), 0.0, 1.0)
        return (t * t * (3 - 2 * t)).astype(np.float32)

    phip = np.arctan2(dy, dx)
    phi = phip.copy()
    rr = rp.copy()
    for _ in range(iters):
        tp = tap(rr)
        phi = phip - (phi_off + _harm(phi, amps_w, ph_w)) * tp
        rr = rp / (r_gain * (1.0 + _harm(phi, amps_g, ph_g) * tp))
        np.clip(rr, 0.0, 3.0 * R, out=rr)
    sx = (c[0] + rr * np.cos(phi)) * SS
    sy = (c[1] + rr * np.sin(phi)) * SS
    out = ndi.map_coordinates(up, [sy.ravel(), sx.ravel()], order=1,
                              mode="constant", cval=0.0).reshape(H2, W2)
    t = tap(rp)
    bl = up * (1 - t) + out * t
    bl = bl.reshape(h, SS, w, SS).mean((1, 3))
    return np.clip(bl, 0.0, 1.0).astype(np.float32)


def band(phi, c0_deg, width_deg):
    """角度带（**度**为单位）的平滑指示，周期拼接，返回 [0,1]。"""
    c0 = np.radians(c0_deg)
    width = np.radians(width_deg)
    d = np.angle(np.exp(1j * (phi - c0)))
    t = np.clip(1.0 - np.abs(d) / (width * 0.5), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def rotate_band(alpha, c, R, delta_deg, c0_deg, width_deg):
    """把角度带内的一段叶**整体旋转 delta** 后 max 合成回去（= 多长出一根叶）。"""
    delta = np.radians(delta_deg)
    h, w = alpha.shape
    up = np.repeat(np.repeat(alpha, SS, 0), SS, 1)
    H2, W2 = up.shape
    yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
    dx = (xx - c[0] * SS) / SS
    dy = (yy - c[1] * SS) / SS
    r = np.sqrt(dx * dx + dy * dy)
    phi = np.arctan2(dy, dx)
    m = band(phi, c0_deg + delta_deg, width_deg) * \
        np.clip((R - r) / max(1.0, 0.15 * R), 0, 1) ** 0.5
    sx = (c[0] + r * np.cos(phi - delta)) * SS
    sy = (c[1] + r * np.sin(phi - delta)) * SS
    src = ndi.map_coordinates(up, [sy.ravel(), sx.ravel()], order=1,
                              mode="constant", cval=0.0).reshape(H2, W2)
    # ⚠️ m 已经是超采样网格上的（phi 来自 mgrid[0:H2,0:W2]），不能再 repeat 一次
    got = np.maximum(up, src * m)
    got = got.reshape(h, SS, w, SS).mean((1, 3))
    return np.clip(got, 0.0, 1.0).astype(np.float32)


def patch_sample(alpha, center, half, scale=1.0):
    """以 center 为中心取 (2half)² 补丁；scale 用于把供体树冠径归一化到本树。"""
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float32)
    sx = center[0] + (xx - half) * scale
    sy = center[1] + (yy - half) * scale
    return ndi.map_coordinates(alpha, [sy.ravel(), sx.ravel()], order=1,
                               mode="constant", cval=0.0).reshape(2 * half, 2 * half)


def patch_rot(P, deg):
    half = (P.shape[0] - 1) / 2.0
    yy, xx = np.mgrid[0:P.shape[0], 0:P.shape[1]].astype(np.float32)
    d = np.deg2rad(deg)
    ca, sa = np.cos(d), np.sin(d)
    xs = half + ca * (xx - half) + sa * (yy - half)
    ys = half - sa * (xx - half) + ca * (yy - half)
    return ndi.map_coordinates(P, [ys.ravel(), xs.ravel()], order=1,
                               mode="constant", cval=0.0).reshape(P.shape)


def paste_max(a, P, c):
    """把补丁 P（中心对齐 c）以 max 合成进 a。"""
    h, w = a.shape
    half = (P.shape[0] - 1) // 2
    cx, cy = int(round(c[0])), int(round(c[1]))
    x0, y0 = cx - half, cy - half
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(w, x0 + P.shape[1]), min(h, y0 + P.shape[0])
    if ax1 <= ax0 or ay1 <= ay0:
        return a
    sub = P[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    a[ay0:ay1, ax0:ax1] = np.maximum(a[ay0:ay1, ax0:ax1], sub)
    return a


# ------------------------------------------------------------------ 单棵树
def remix_tree(own_alpha, c, R, rng, donor=None, amp_scale=1.0):
    """一棵树的重组：多阶极坐标重组（逐叶各转各的、各长各的）+ 冠向/冠幅 + 减叶。
    注：不用"复制一根叶/跨树借叶"两种做法 —— 它们会在冠内留下半透明重影
    （带掩膜的软边界),读起来就是"糊"。改由**高阶谐波**自然实现叶的增多/裂分：
    k=4/5 阶谐波把一片宽叶在角度上"撕"成两瓣，等价于叶数变化且全程连续无接缝。"""
    a = own_alpha
    k = R / 200.0
    aw = np.radians(np.array([14.0, 9.0, 7.0, 6.5, 5.5], np.float32) * k)
    pw = rng.uniform(0, 2 * np.pi, 5).astype(np.float32)
    ag = np.array([0.17, 0.12, 0.09, 0.075, 0.06], np.float32) * amp_scale
    pg = rng.uniform(0, 2 * np.pi, 5).astype(np.float32)
    a = polar_remap(a, c, R, aw, pw, ag, pg,
                    phi_off=float(np.radians(rng.uniform(-20, 20))),
                    r_gain=float(rng.uniform(0.87, 1.13)))
    # 减叶：随机 1~2 个角度带的外段抹掉 → 冠的丰缺关系改变（乘法，无重影）
    for _ in range(int(rng.integers(1, 3))):
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
        r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
        phi = np.arctan2(yy - c[1], xx - c[0])
        c1 = float(rng.uniform(-180, 180))
        wd = float(rng.uniform(18, 32))
        radial = float(rng.uniform(0.18, 0.42))
        kk = band(phi, c1, wd) * np.clip((r - radial * R) / (0.30 * R), 0, 1)
        a = a * (1.0 - np.clip(kk, 0, 1))
    # 边缘回锐：重采样会把 7px 宽的叶软化成"糊边"，用中心对比曲线把 [0.33,0.67]
    # 重新拉满 → 恢复与原图同档的硬边（保留 ~1px 抗锯齿带）
    a = np.clip((a - 0.33) / 0.34, 0.0, 1.0)
    a = a * a * (3.0 - 2.0 * a)
    return np.clip(a, 0.0, 1.0)


def trunk_remix(alpha, c, base_y, R, rng):
    """树干绕**树基**做倾角 + 长度变形（根部不动 → 位置不变）。"""
    h, w = alpha.shape
    up = np.repeat(np.repeat(alpha, SS, 0), SS, 1)
    H2, W2 = up.shape
    yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
    Y = yy / SS
    X = xx / SS
    lean = float(np.radians(rng.uniform(-7, 7)))
    ls = float(rng.uniform(0.90, 1.10))
    span = max(1.0, base_y - (base_y - 2.0 * R))
    t = np.clip((base_y - Y) / span, 0.0, 1.0)
    Xs = X - np.tan(lean) * (base_y - Y)
    Ys = base_y - (base_y - Y) / ls
    src = ndi.map_coordinates(up, [(Ys * SS).ravel(), (Xs * SS).ravel()], order=1,
                              mode="nearest").reshape(H2, W2)
    return src.reshape(h, SS, w, SS).mean((1, 3)).astype(np.float32)


# ------------------------------------------------------------------ 主流程
def build(seed=20260917, amp_scale=1.0, edil=9, eclose=14, min_own=400, limit=None,
          out="crown"):
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    o = np.asarray(img, np.float32)
    ink0 = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    ink_a0 = ndi.gaussian_filter(ink0.astype(np.float32), 0.7)
    _, segs = trunks(ink0, H, W, 25, 60)
    segs = [s for s in segs if 0.06 * H < s["h"] < 0.62 * H]
    win = partition(ink0, segs, H, W)
    trees = []
    yyA, xxA = np.mgrid[0:H, 0:W].astype(np.float32)
    for i, s in enumerate(segs):
        own = (win == i) & ink0
        if own.sum() < min_own:
            continue
        ys, xs = np.where(own)
        bh = ys.max() - ys.min() + 1
        c = (s["x"], s["y_top"])
        # 树冠半径 = 冠心上方树冠墨迹的径距 92 分位（比 bbox 估计稳）
        cm = own & (yyA < c[1] + 0.25 * bh)
        if cm.sum() < 200:
            cm = own
        rr = np.sqrt((xxA[cm] - c[0]) ** 2 + (yyA[cm] - c[1]) ** 2)
        R = float(np.clip(np.percentile(rr, 92), 40.0, 0.32 * H))
        trees.append(dict(own=own, x=s["x"], y_top=s["y_top"], y_bot=s["y_bot"],
                          R=R, bh=bh))
    print(f"[p4crown] 可用树={len(trees)} R范围={min(t['R'] for t in trees):.0f}"
          f"~{max(t['R'] for t in trees):.0f}")
    if limit:
        trees = trees[:limit]

    # 输出 alpha 层：先原样，再逐棵替换
    acc = ink_a0.copy()
    rng = np.random.default_rng(int(seed))
    order = np.argsort([-t["own"].sum() for t in trees])
    done = np.zeros((H, W), bool)
    for k, i in enumerate(order):
        t = trees[i]
        own = t["own"]
        # 该树的处理区（own 外扩）+ 已处理区保护
        reg = ndi.binary_dilation(own, structure=_disk(6)) & ~done
        if reg.sum() < 300:
            continue
        ys, xs = np.where(reg)
        pad = 30
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
        sub = (ink_a0[y0:y1, x0:x1] * reg[y0:y1, x0:x1]).astype(np.float32)
        c = (t["x"] - x0, t["y_top"] - y0)
        R = t["R"] * amp_scale
        # 树冠（冠心以上 + 半径内）交给重组；树干另走 trunk_remix
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
        crown_zone = np.clip((R - r) / (0.2 * R + 1e-6), 0, 1)
        crown_a = sub * crown_zone
        trunk_a = sub * (1 - crown_zone)
        # 借叶：从另一棵树（随机挑，排除自己，冠径同量级）
        j = int(rng.integers(0, len(trees)))
        donor = None
        if j != i:
            d = trees[j]
            if 0.55 * d["R"] < R < 1.9 * d["R"]:
                dys, dxs = np.where(d["own"])
                dy0, dy1 = max(0, dys.min() - pad), min(H, dys.max() + 1 + pad)
                dx0, dx1 = max(0, dxs.min() - pad), min(W, dxs.max() + 1 + pad)
                donor = (ink_a0[dy0:dy1, dx0:dx1].astype(np.float32),
                         (d["x"] - dx0, d["y_top"] - dy0), d["R"])
        new_crown = remix_tree(crown_a, c, R, rng, donor, amp_scale=amp_scale)
        new_trunk = trunk_remix(trunk_a, c, t["y_bot"] - y0, R, rng)
        new_sub = np.maximum(new_crown, new_trunk)
        acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1] * (1 - reg[y0:y1, x0:x1]),
                                       new_sub * reg[y0:y1, x0:x1])
        done |= reg
        if (k + 1) % 8 == 0:
            print(f"   [p4crown] {k+1}/{len(order)}")

    # 合成：迷彩底（原墨 nn 回填）⊕ 黑墨 α 混合
    idx = ndi.distance_transform_edt(ink0, return_distances=False, return_indices=True)
    base = o[idx[0], idx[1]]
    a = acc[..., None]
    prev = base * (1 - a) + BLACK[None, None, :] * a
    pv = Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8), "RGB")
    pv.save(str(OUT / f"_{out}.jpg"), quality=94)
    ink_b = acc > 0.5
    Image.fromarray((ink_b.astype(np.uint8) * 255), "L").save(str(OUT / f"_{out}_ink.png"))
    print(f"[p4crown] 新墨={100*ink_b.mean():.1f}% (原 {100*ink0.mean():.1f}%)")
    return OUT / f"_{out}.jpg"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default="crown")
    a = ap.parse_args()
    build(seed=a.seed, amp_scale=a.amp, limit=a.limit, out=a.out)
