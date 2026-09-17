"""v345f: p4 棕榈树「刚体换位 + 逐棵重姿」—— 唯一能 100% 保住原图矢量质量的裂变。

本轮把 p4 树处理的 5 条路线全部实测过，结论（决定本方案）：

  | 路线 | 实测结果 |
  |---|---|
  | SDXL 整图重生（ProteusV0.4，6 组 prompt/denoise/CN/LoRA 扫参） | 棕榈被画成软气刷水彩，二值化后"绒毛毛球" |
  | SDXL 逐棵树裁块重生（实心包络 + 上采样 1024 × 34 棵） | 分辨率够了，渲染仍软 → "毛边糊团" |
  | 换 anime/flat 底模 CounterfeitXL dn .90 | 同样毛边糊团 |
  | 程序化 palm_art 重画 | 输出"刺球"（v330 用户已否） |
  | 原图墨迹极坐标重组（逐叶转/拉/裂） | 无重影了，但 7px 细叶被重采样+高阶谐波撕成"刮痕" |

  ⇒ 这张图的棕榈是**专业矢量线稿**，"重画/逐叶形变"都会掉档，而用户红线是
    "不允许比原图丑"。**唯一 100% 保真的变换 = 整棵树的刚体变换**
    （旋转/镜像/等比缩放不改变矢量质量）。

方案（同构异内容；位置/大小/版式全部保留）：
  ① 34 棵树由树干定位（closing vline25 + opening vline60）+ 墨迹 Voronoi 归属；
  ② **置换**：每个位置换成另一棵树的树形（derangement 错排，位置不重复）；
  ③ 每棵再做刚体变换：50% 水平镜像 + 绕**树基**旋转 ±16° + 沿干长等比缩放
     （缩放系数 = 目标干长 / 供体干长 → 树高与目标一致、根位不变）；
  ④ 仿射**逆**映射直接采样（×3 超采样 + 双线性 + 均值降采样）→ 旋转后边缘
     仍与原图同档；float alpha + α 混合 → 无硬切、无锯齿。

⚠️ 三个已踩过的坑（写进注释防止回退）：
  · 不要用 PIL 的 `Image.resize` 做缩放：它以**图像原点**为基准，会把"树基对齐"
    一起缩放掉 → 墨量只剩 1/3。
  · 输出窗口必须按**供体自身 ink bbox 经 A 变换后的范围**取；按"目标 bbox 外扩"
    取会把供体周边一大片图案（别的树）一起搬过来 → 整图糊成黑块（实测新墨 78%）。
  · 逆映射必须严格是 A⁻¹ = (1/s)·M·R(-θ)（M 为镜像矩阵，自逆）；
    随手写 `(ca*dx - sa*dy)/s` 不是它的逆 → 树被搬到别处去。
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp            # noqa: E402
from v345c_p4seg import trunks, partition, _disk       # noqa: E402

OUT = ROOT / "jobs" / "v345f_p4swap"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
BLACK = np.array([12.0, 10.0, 9.0], np.float32)
SS = 6


def _AA(ang_deg, scale, flip):
    """正向 A = s·R(θ)·M。"""
    th = np.radians(ang_deg)
    ca, sa = np.cos(th), np.sin(th)
    R = np.array([[ca, -sa], [sa, ca]], np.float32)
    if flip:
        M = np.array([[-1.0, 0.0], [0.0, 1.0]], np.float32)
    else:
        M = np.eye(2, dtype=np.float32)
    return (scale * (R @ M)).astype(np.float32)


def _inv(dx, dy, ang_deg, scale, flip):
    """A⁻¹·(dx,dy) = (1/s)·M·R(-θ)·(dx,dy)。"""
    th = np.radians(ang_deg)
    ca, sa = np.cos(th), np.sin(th)
    rx = (ca * dx + sa * dy) / scale
    ry = (-sa * dx + ca * dy) / scale
    if flip:
        rx = -rx
    return rx, ry


def place_tree(alpha, donor, target, ang_deg, scale, flip, box):
    """把 alpha 中 donor 树基/ink 范围内的内容刚体变换后对齐到 target 的树基，
    只在 box 内输出（box 由 transformed bbox 决定）。"""
    x0, y0, x1, y1 = box
    nx, ny = x1 - x0, y1 - y0
    if nx <= 0 or ny <= 0:
        return None
    gx = (np.arange(nx * SS, dtype=np.float32) + 0.5) / SS + x0 - 0.5
    gy = (np.arange(ny * SS, dtype=np.float32) + 0.5) / SS + y0 - 0.5
    XX, YY = np.meshgrid(gx, gy)
    rx, ry = _inv(XX - target[0], YY - target[1], ang_deg, scale, flip)
    sx = donor[0] + rx
    sy = donor[1] + ry
    got = ndi.map_coordinates(alpha, [sy.ravel(), sx.ravel()], order=1,
                              mode="constant", cval=0.0).reshape(ny * SS, nx * SS)
    return got.reshape(ny, SS, nx, SS).mean((1, 3))


def build(seed=20260917, max_ang=16.0, scl_var=0.08, flip_p=0.5, pad=8):
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    o = np.asarray(img, np.float32)
    ink0 = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    ink_a0 = ndi.gaussian_filter(ink0.astype(np.float32), 0.7)

    _, segs = trunks(ink0, H, W, 25, 60)
    segs = [s for s in segs if 0.06 * H < s["h"] < 0.62 * H]
    win = partition(ink0, segs, H, W)
    trees = []
    for i, s in enumerate(segs):
        own = (win == i) & ink0
        if own.sum() < 400:
            continue
        ys, xs = np.where(own)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        # ⚠️ 供体 alpha 必须取**整棵连通树**：直接用 Voronoi 的 own 当支撑，
        #    邻树交界会把细叶沿分界线**整条切断** → 搬过去就是"断点/虚线状叶"
        #    （实测最明显的就是这个）。做法：在 own 的 bbox(+pad) 内做连通域标记，
        #    保留所有与 own 有交集的连通域 → 恢复完整树形，且不越出 bbox 太多。
        px0, px1 = max(0, x0 - 12), min(W, x1 + 13)
        py0, py1 = max(0, y0 - 12), min(H, y1 + 13)
        band_ = ink0[py0:py1, px0:px1]
        lab_, k_ = ndi.label(band_, structure=np.ones((3, 3), bool))
        ids = np.unique(lab_[own[py0:py1, px0:px1] & band_])
        ids = ids[ids > 0]
        if len(ids) == 0:
            continue
        sup = np.isin(lab_, ids)
        full = np.zeros((H, W), np.float32)
        full[py0:py1, px0:px1] = ink_a0[py0:py1, px0:px1] * sup
        _rows = np.where(full.any(1))[0]
        _cols = np.where(full.any(0))[0]
        if len(_rows) == 0 or len(_cols) == 0:
            continue
        trees.append(dict(own=own, cx=float(s["x"]), base=float(s["y_bot"]),
                          x0=int(_cols.min()), x1=int(_cols.max()),
                          y0=int(_rows.min()), y1=int(_rows.max()),
                          trunk=float(max(1.0, s["y_bot"] - s["y_top"])),
                          alpha=full))
    n = len(trees)
    rng = np.random.default_rng(int(seed))
    perm = np.arange(n)
    for _ in range(300):
        rng.shuffle(perm)
        if not (perm == np.arange(n)).any():
            break
    if (perm == np.arange(n)).any():
        perm = (np.arange(n) + 1) % n
    print(f"[p4swap] 树={n}")

    tree_union = np.zeros((H, W), bool)
    for t in trees:
        tree_union |= t["own"]
    acc = (ink_a0 * (~ndi.binary_dilation(tree_union, structure=_disk(3)))).astype(np.float32)

    moved = 0
    for i, t in enumerate(trees):
        d = trees[int(perm[i])]
        scale = float(np.clip(t["trunk"] / d["trunk"], 0.80, 1.30)) * \
            float(rng.uniform(1 - scl_var, 1 + scl_var))
        ang = float(rng.uniform(-max_ang, max_ang))
        flip = bool(rng.random() < flip_p)
        A = _AA(ang, scale, flip)
        pts = []
        for ddx, ddy in product((d["x0"] - d["cx"], d["x1"] - d["cx"]),
                                (d["y0"] - d["base"], d["y1"] - d["base"])):
            uv = A @ np.array([ddx, ddy], np.float32)
            pts.append((t["cx"] + float(uv[0]), t["base"] + float(uv[1])))
        bx0 = int(max(0, min(p[0] for p in pts) - pad))
        bx1 = int(min(W, max(p[0] for p in pts) + 1 + pad))
        by0 = int(max(0, min(p[1] for p in pts) - pad))
        by1 = int(min(H, max(p[1] for p in pts) + 1 + pad))
        if bx1 - bx0 < 8 or by1 - by0 < 8:
            continue
        placed = place_tree(d["alpha"], (d["cx"], d["base"]), (t["cx"], t["base"]),
                            ang, scale, flip, (bx0, by0, bx1, by1))
        if placed is None:
            continue
        acc[by0:by1, bx0:bx1] = np.maximum(acc[by0:by1, bx0:bx1], placed)
        moved += 1
    print(f"[p4swap] 换位 {moved} 棵")

    a = np.clip((acc - 0.28) / 0.44, 0.0, 1.0)
    a = a * a * (3.0 - 2.0 * a)

    # 预览底色：**必须**用管线的干净/重 blob 迷彩底，不能用"最近邻回填"。
    # 原因：树被换掉后，原墨迹范围内未被新树覆盖的像素会露出底色；最近邻回填会把
    # 叶内部空出的像素填成远处迷彩色（浅棕/灰）→ 肉眼就是"叶上有噪点/断成虚线"，
    # 会被误读成"比原图丑"。管线的 _p4_warped.jpg 本身就是干净迷彩，直接复用。
    wr = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
    if wr.exists():
        base = np.asarray(Image.open(wr).convert("RGB").resize((W, H), Image.LANCZOS),
                          np.float32)
    else:
        import styles.texture_fx as tf
        base = np.asarray(cpp.tf.erase(img, ndi.binary_dilation(ink0, structure=_disk(4)),
                                       method="nn", nn_median=21, margin=24), np.float32)
    aa = a[..., None]
    prev = base * (1 - aa) + BLACK[None, None, :] * aa
    pv = Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8), "RGB")
    pv.save(str(OUT / "_swap.jpg"), quality=95)
    Image.fromarray((a > 0.5).astype(np.uint8) * 255, "L").save(str(OUT / "_swap_ink.png"))
    print(f"[p4swap] 新墨={100*(a>0.5).mean():.1f}% (原 {100*ink0.mean():.1f}%)")
    return OUT / "_swap.jpg"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--max-ang", type=float, default=16.0)
    ap.add_argument("--scl-var", type=float, default=0.08)
    ap.add_argument("--flip-p", type=float, default=0.5)
    a = ap.parse_args()
    build(seed=a.seed, max_ang=a.max_ang, scl_var=a.scl_var, flip_p=a.flip_p)
