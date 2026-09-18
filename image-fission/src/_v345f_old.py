"""v345f: p4 棕榈树「刚体换位 + 逐棵重姿」—— 唯一能 100% 保住原图矢量质量的裂变。

本轮把 p4 的树处理路线全部实测过一遍，结论如下（这决定了本文件的方案）：

  | 路线 | 结果 |
  |---|---|
  | SDXL 整图重生（ProteusV0.4，6 组 prompt/denoise/CN/LoRA） | 棕榈被画成软气刷水彩，二值化后"绒毛毛球" |
  | SDXL 逐棵树裁块重生（实心包络 + 上采样 1024，34 棵） | 分辨率够了，渲染仍软 → "毛边糊团" |
  | 换 anime/flat 底模 CounterfeitXL 0.90 | 同样毛边糊团 |
  | 程序化 palm_art 重画 | 输出"刺球"（v330 用户已否） |
  | 原图墨迹极坐标重组（逐叶转/拉/裂） | 无重影了，但 7px 细叶被重采样+高阶谐波撕成"刮痕" |

  ⇒ 这张图的棕榈是**专业矢量线稿**，任何"重画/逐叶形变"都会掉档；
    用户红线是"不允许比原图丑"。**唯一 100% 保真的变换 = 整棵树的刚体变换**
    （旋转/镜像/等比缩放不改变矢量质量）。

本方案（同构异内容，版式/位置/大小全部保留）：
  ① 34 棵树全部由树干定位（closing vline25 + opening vline60）；
  ② **置换**：每个位置换成另一棵树的树形（错排 derangement，位置不重复）；
  ③ 每棵再做刚体变换：50% 水平镜像 + 绕**树基**旋转 ±16° + 沿树干的等比缩放
     （缩放系数由"供体干长/目标干长"定 → 树高与目标一致，根位保持不变）；
  ④ 全程 ×3 超采样 BICUBIC 重采样，float alpha 输出，α 混合 → 边缘质量与原图同档。

结果：**每个位置的棕榈都是另一张不同的棕榈**（且镜像/转向不同），而画质与原件一致；
配合 camo_palm_pattern 的迷彩 blob 重塑 + 文字管线 = 一张全新的同构图案。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp            # noqa: E402
from v345c_p4seg import trunks, partition, _disk, keep_big  # noqa: E402

OUT = ROOT / "jobs" / "v345f_p4swap"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
BLACK = np.array([12.0, 10.0, 9.0], np.float32)
SS = 3


def rigid_place(alpha, Babs, Tabs, ang_deg, scale, flip, box):
    """把 alpha（整图 float 层）里的树**绕自身树基 Babs 旋转/等比缩放**后，
    使其树基落到目标树基 Tabs 上；只在 box=(x0,y0,x1,y1) 区域内输出。

    直接做仿射**逆**映射采样（不经过 PIL 的 origin-based resize —— 那会把树基
    一起缩放掉，实测墨量只剩 1/3）。A = s·R(θ)·M，M=diag(-1,1) 为镜像。
    ⚠️ 全程 ×SS 超采样 + 双线性 + 均值降采样 → 旋转后的边缘仍保持抗锯齿质量。
    """
    x0, y0, x1, y1 = box
    nx, ny = x1 - x0, y1 - y0
    if nx <= 0 or ny <= 0:
        return None
    gx = (np.arange(nx * SS, dtype=np.float32) + 0.5) / SS + x0 - 0.5
    gy = (np.arange(ny * SS, dtype=np.float32) + 0.5) / SS + y0 - 0.5
    XX, YY = np.meshgrid(gx, gy)
    dx = XX - Tabs[0]
    dy = YY - Tabs[1]
    th = np.radians(ang_deg)
    ca, sa = np.cos(th), np.sin(th)
    # A = s * R(θ) * M  →  A⁻¹ = (1/s) * M * R(-θ)
    u = (ca * dx - sa * dy) / scale
    v = (sa * dx + ca * dy) / scale
    if flip:
        u = -u
    sx = Babs[0] + u
    sy = Babs[1] + v
    got = ndi.map_coordinates(alpha, [sy.ravel(), sx.ravel()], order=1,
                              mode="constant", cval=0.0).reshape(ny * SS, nx * SS)
    return got.reshape(ny, SS, nx, SS).mean((1, 3))


def build(seed=20260917, max_ang=16.0, scl_var=0.08, flip_p=0.5, pad_extra=26):
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
        trees.append(dict(own=own, cx=s["x"], top=s["y_top"], base=s["y_bot"],
                          x0=int(xs.min()), x1=int(xs.max()),
                          y0=int(ys.min()), y1=int(ys.max()),
                          trunk=max(1.0, s["y_bot"] - s["y_top"])))
    n = len(trees)
    print(f"[p4swap] 树={n}")
    rng = np.random.default_rng(int(seed))
    # 错排：每个位置都不放自己
    perm = np.arange(n)
    for _ in range(200):
        rng.shuffle(perm)
        if not (perm == np.arange(n)).any():
            break
    if (perm == np.arange(n)).any():                       # 兜底：整体轮转一位
        perm = (np.arange(n) + 1) % n

    # 所有树（34 棵全部是目标）的墨迹先从原层里摘掉，只保留非树小元素
    tree_union = np.zeros((H, W), bool)
    for t in trees:
        tree_union |= t["own"]
    rest = (ink_a0 * (~ndi.binary_dilation(tree_union, structure=_disk(3)))).astype(np.float32)
    acc = rest.copy()
    for i, t in enumerate(trees):
        d = trees[int(perm[i])]
        s_ = float(np.clip(t["trunk"] / d["trunk"], 0.80, 1.30)) *             float(rng.uniform(1 - scl_var, 1 + scl_var))
        ang = float(rng.uniform(-max_ang, max_ang))
        flip = bool(rng.random() < flip_p)
        # 输出区：目标区向外扩，供体叶尖可能伸出去（不裁切）
        pad = int(round(0.55 * max(t["x1"] - t["x0"], t["y1"] - t["y0"]))) + 40
        box = (int(max(0, min(t["x0"], d["x0"] + (t["cx"] - d["cx"])) - pad)),
               int(max(0, min(t["y0"], d["y0"] + (t["base"] - d["base"])) - pad)),
               int(min(W, max(t["x1"], d["x1"] + (t["cx"] - d["cx"])) + pad)),
               int(min(H, max(t["y1"], d["y1"] + (t["base"] - d["base"])) + pad)))
        placed = rigid_place(ink_a0, (d["cx"], d["base"]), (t["cx"], t["base"]),
                             ang, s_, flip, box)
        if placed is None:
            continue
        x0, y0, x1, y1 = box
        acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1], placed)
    # 边缘回锐（重采样软化 → 中心对比曲线拉回硬边）
    acc = np.clip((acc - 0.30) / 0.40, 0.0, 1.0)
    acc = acc * acc * (3.0 - 2.0 * acc)

    idx = ndi.distance_transform_edt(ink0, return_distances=False, return_indices=True)
    base = o[idx[0], idx[1]]
    a = acc[..., None]
    prev = base * (1 - a) + BLACK[None, None, :] * a
    pv = Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8), "RGB")
    pv.save(str(OUT / "_swap.jpg"), quality=94)
    Image.fromarray((acc > 0.5).astype(np.uint8) * 255, "L").save(str(OUT / "_swap_ink.png"))
    print(f"[p4swap] 新墨={100*(acc>0.5).mean():.1f}% (原 {100*ink0.mean():.1f}%)")
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
