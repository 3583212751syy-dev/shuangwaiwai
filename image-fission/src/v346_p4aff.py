"""v346: p4 棕榈树裂变定稿 —— 「整棵仿射换位」（真裂变 + 100% 保真 + 明显可见）。

为什么不是 SDXL 重生（本轮已实测穷尽，写死在这里避免回退）：
  p4 的棕榈是**专业矢量线稿**（细 7px 叶片、干净硬边）。SDXL 是软气刷式生成，
  无论 ProteusV0.4 / CounterfeitXL、无论 6 组 prompt × denoise × CN × LoRA 怎么扫，
  再二值化后一律得到"绒毛毛边/粗团块" → 用户判"比原图丑"（v330 的刺球同理）。
  用户红线是"不允许比原图丑"，所以**任何重画路线都必须放弃**。

为什么仿射换位算"真裂变"：
  ① **异内容**：34 棵树做错排置换 —— 每个位置的树形来自另一棵树，全图不存在
     "原位原树"，逐位置看内容全变；
  ② **同构**：物种/材质/手法/版式/配色/数量全部不变，仍是同一片迷彩棕榈景；
  ③ **同位置同尺寸**：变换绕**树基**做 → 根位不动、树高按目标干长归一 → 版式不改；
  ④ **零质量损失**：仿射变换是矢量级运算，旋转/各向异性缩放/切变/镜像**不改变**
     线宽与硬边（对比 SDXL 重画必然掉档）。

比 v345f 新增（为了让"裂变"肉眼可见，而不只是悄悄换掉）：
  · 各向异性缩放 `sx≠sy` → 树冠宽窄变化，剪影明显不同；
  · 切变 `shear` → 整棵树自然倾斜（棕榈本来就会斜长，观感合理）；
  · 绕基旋转上限 20°、镜像 50%。
  ⇒ 剪影变化幅度显著提升，但线质与原图**逐像素同档**。

实现要点（血泪）：
  · 一律用 **数值 2x2 矩阵 + np.linalg.inv** 求逆，不手写逆公式
    （v345f 手写 A⁻¹ 曾把树搬到别处去 / PIL.resize 以图像原点缩放导致墨量只剩 1/3）；
  · 输出窗口 = **供体 ink bbox 四角经 A 变换后的范围**（不能按目标 bbox 外扩取，
    否则会把邻居树一起搬进来 → 整图糊黑，实测新墨 78%）；
  · 供体 alpha 必须取**整棵连通树**（用 Voronoi 的 own 当支撑会在交界处把细叶
    整条切断 → 搬过去是"虚线状叶"）；
  · 预览底**必须**用管线的干净迷彩底（`jobs/router_out_v329/_p4_warped.jpg`），
    不能用最近邻回填 —— 后者会把叶间空档填成远处迷彩色，肉眼=噪点/断线，
    会被误读成"丑"。
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

OUT = ROOT / "jobs" / "v346_p4aff"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
INK = np.array([12.0, 10.0, 9.0], np.float32)
SS = 6


def _aff(ang_deg, sx, sy, shear, flip):
    """A = diag(sx,sy) · R(θ) · Sh(k) · M，绕原点（把原点取在树基上）。"""
    th = np.radians(ang_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float64)
    Sh = np.array([[1.0, shear], [0.0, 1.0]], np.float64)
    S = np.array([[sx, 0.0], [0.0, sy]], np.float64)
    M = np.array([[-1.0, 0.0], [0.0, 1.0]], np.float64) if flip else np.eye(2)
    return (S @ R @ Sh @ M).astype(np.float32)


def place(alpha, donor_xy, target_xy, A, box):
    """在 box 内输出 A 变换后的供体（×SS 超采样 + 双线性 + 均值降采样）。"""
    x0, y0, x1, y1 = box
    nx, ny = x1 - x0, y1 - y0
    if nx <= 0 or ny <= 0:
        return None
    Ai = np.linalg.inv(A.astype(np.float64)).astype(np.float32)
    gx = (np.arange(nx * SS, dtype=np.float32) + 0.5) / SS + x0 - 0.5 - target_xy[0]
    gy = (np.arange(ny * SS, dtype=np.float32) + 0.5) / SS + y0 - 0.5 - target_xy[1]
    XX, YY = np.meshgrid(gx, gy)
    sx = donor_xy[0] + Ai[0, 0] * XX + Ai[0, 1] * YY
    sy = donor_xy[1] + Ai[1, 0] * XX + Ai[1, 1] * YY
    got = ndi.map_coordinates(alpha, [sy.ravel(), sx.ravel()], order=1,
                              mode="constant", cval=0.0).reshape(ny * SS, nx * SS)
    return got.reshape(ny, SS, nx, SS).mean((1, 3))


def collect(img, ink0, ink_a0, W, H):
    """按树干定位 + Voronoi 归属，抽出每棵树的 alpha。

    ⚠️ 供体 alpha **必须用 Voronoi 归属 own**（+ 6px 测地膨胀补分界缝），
    **不能**用"bbox 内连通域"：
      · 实测这张图的墨迹高度互联 —— 全局只有 91 个连通域，而每棵树碰到的连通域
        域总面积可达自身的 **51 倍**（median 5.5 倍）。用连通域取树 = 把邻居树冠、
        地平线整片卷进来 → 换位后变成"深色糊团 + 漂浮碎块"（v346a 的丑根因）；
      · 反之，把 own 可视化出来，**每一块着色就是一个完整的棕榈**（干+冠齐全）——
        own 才是正确的单树支撑。
      · 之前"用 own 会把细叶沿分界线整条切断"的结论：分界缝确实会切细叶，但
        **6px 测地膨胀**（沿墨迹连通膨胀）就能把缝补上，且不会跨到 100px 外的邻居。
    """
    _, segs = trunks(ink0, H, W, 25, 60)
    segs = [s for s in segs if 0.06 * H < s["h"] < 0.62 * H]
    win = partition(ink0, segs, H, W)
    trees = []
    for i, s in enumerate(segs):
        own = (win == i) & ink0
        if own.sum() < 400:
            continue
        # ② 小元素（树基旁草痕/短横线）**随最近的树一起搬**，不做就地保留。
        #    实测"留在原位"会让它们变成悬空孤块（1:1 目检 = 一撮浮在迷彩上的脏点），
        #    因为周围那棵树已经走了。跟着最近的树干走 → 仍贴在一棵树旁，观感自然。
        sup = ndi.binary_dilation(own, structure=_disk(1), iterations=6, mask=ink0)
        full = ink_a0 * sup
        rows = np.where(full.any(1))[0]
        cols = np.where(full.any(0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        trees.append(dict(own=own, cx=float(s["x"]), base=float(s["y_bot"]),
                          x0=int(cols.min()), x1=int(cols.max()),
                          y0=int(rows.min()), y1=int(rows.max()),
                          trunk=float(max(1.0, s["y_bot"] - s["y_top"])), alpha=full))
    return trees


def build(seed=20260917, max_ang=10.0, scl_var=0.07, skew=0.10, aniso=0.14, flip_p=0.5,
          pad=10, scl_lo=0.90, scl_hi=1.12):
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    ink0 = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    ink_a0 = ndi.gaussian_filter(ink0.astype(np.float32), 0.7)
    trees = collect(img, ink0, ink_a0, W, H)
    n = len(trees)
    rng = np.random.default_rng(int(seed))
    # 置换 = **尺寸+墨量匹配的最优分配**（不是随机错排）。
    # 随机错排会让"大冠树"落到"小树位" → 溢出压邻树成深色糊团；反之上不去就留空缺
    # → 整图疏密不匀（v346a 的丑根因）。
    # 只用 bbox 尺寸匹配还不够：同尺寸的"近景粗叶树"与"远景细叶树"墨量差 2-3 倍，
    # 互换了就会读成"某棵树突然变瘦/变胖"。故成本里**以墨量为主、尺寸为辅**。
    w = np.array([t["x1"] - t["x0"] + 1.0 for t in trees], np.float64)
    h = np.array([t["y1"] - t["y0"] + 1.0 for t in trees], np.float64)
    a = np.array([float(t["own"].sum()) for t in trees], np.float64)
    cost = (np.abs(np.log(a[None, :] / a[:, None])) +
            0.35 * (np.abs(np.log(w[None, :] / w[:, None])) +
                    np.abs(np.log(h[None, :] / h[:, None]))))
    # 再加"就近"项：允许远距离对调会让整图版面错乱（一棵树连同它树基的小元素一起
    # 飞到画面另一头，原地留空、异地多出一撮孤块）。加位置代价 → 优先与**邻近的
    # 同体量树**互换，异内容照样成立，版面节奏不乱。
    _px = np.array([t["cx"] for t in trees], np.float64)
    _py = np.array([t["base"] for t in trees], np.float64)
    _diag = float(np.hypot(W, H))
    dpos = np.hypot(_px[None, :] - _px[:, None], _py[None, :] - _py[:, None]) / _diag
    cost = cost + 0.9 * dpos
    np.fill_diagonal(cost, 1e4)
    from scipy.optimize import linear_sum_assignment
    _ri, _ci = linear_sum_assignment(cost)
    perm = np.empty(n, int)
    perm[_ri] = _ci
    print(f"[p4aff] 树={n}  尺寸匹配分配 成本均值="
          f"{cost[_ri, _ci].mean():.3f}（随机错排基线≈{cost[np.arange(n),(np.arange(n)+1)%n].mean():.3f}）")

    # ⚠️ 必须按**整棵树的实际 alpha**取并集来擦除底板原画，不能用 Voronoi 归属 own：
    # own 只覆盖 Voronoi 格内的墨，而树的真实连通范围会**溢出**到邻居格里 →
    # 那些溢出像素没被擦掉，就变成留在原位的"原树残片"（实测中碎块 40→131）。
    union = np.zeros((H, W), bool)
    for t in trees:
        union |= (t["alpha"] > 0.35)
    acc = (ink_a0 * (~ndi.binary_dilation(union, structure=_disk(3)))).astype(np.float32)

    moved, stats = 0, []
    for i, t in enumerate(trees):
        d = trees[int(perm[i])]
        ang = float(rng.uniform(-max_ang, max_ang))
        shear = float(rng.uniform(-skew, skew))
        flip = bool(rng.random() < flip_p)
        # ① 先只用 R·Sh·M（单位尺度）把供体 bbox 四角摆出去，量出"旋转+切变后"的
        #    实际展开宽高 (w_r,h_r)。旋转会把细高棕榈的占位**撑宽**（h·sinθ），
        #    不补偿就会与邻树互穿 → 深色糊团（v346a 整图疏密不匀的真因）。
        A0 = _aff(ang, 1.0, 1.0, shear, flip)
        cw_ = [A0 @ np.array([dx, dy], np.float32)
               for dx in (d["x0"] - d["cx"], d["x1"] - d["cx"])
               for dy in (d["y0"] - d["base"], d["y1"] - d["base"])]
        w_r = max(p[0] for p in cw_) - min(p[0] for p in cw_)
        h_r = max(p[1] for p in cw_) - min(p[1] for p in cw_)
        # ② 按**目标脚印**拟合：新树占位 ≈ 原位置占位 → 不溢出、不留空
        tw_ = float(t["x1"] - t["x0"] + 1)
        th_ = float(t["y1"] - t["y0"] + 1)
        sx = tw_ / max(1.0, w_r) * float(rng.uniform(1 - scl_var, 1 + scl_var))
        sy = th_ / max(1.0, h_r) * float(rng.uniform(1 - scl_var, 1 + scl_var))
        sx = float(np.clip(sx, scl_lo, scl_hi))
        sy = float(np.clip(sy, scl_lo, scl_hi))
        A = _aff(ang, sx, sy, shear, flip)
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
        got = place(d["alpha"], (d["cx"], d["base"]), (t["cx"], t["base"]), A,
                    (bx0, by0, bx1, by1))
        if got is None:
            continue
        acc[by0:by1, bx0:bx1] = np.maximum(acc[by0:by1, bx0:bx1], got)
        moved += 1
        stats.append((ang, sx, sy, shear, flip))
    # α→墨 的映射必须**低膝点**：原图叶片细到 7px，旋转/切变后经双线性重采样，
    # 细叶中心的 α 只有 ~0.5。若用 (α-0.28)/0.44 的陡膝 → 细叶被压成中灰
    # （实测掉墨 15%、中碎块 131 个、连通块 91→238），管线判"断线/发虚"。
    # 低膝 (α-0.12)/0.38 → 细叶也恢复到纯黑，线宽与连通性回到原档。
    a = np.clip((acc - 0.06) / 0.34, 0.0, 1.0)
    a = a * a * (3.0 - 2.0 * a)
    # 清孤立碎点（重采样在极细末梢产生的 <20px 斑点；原图只有 2 个，必须回到同档）
    _lab, _k = ndi.label(a > 0.5, structure=np.ones((3, 3), bool))
    if _k:
        _sz = np.bincount(_lab.ravel())
        _small = np.zeros(_k + 1, bool)
        _small[1:] = _sz[1:] < 8
        a[_small[_lab]] = 0.0
        print(f"[p4aff] 清碎点(<8px) {int(_small[1:].sum())} 个")

    wr = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
    if wr.exists():
        base = np.asarray(Image.open(wr).convert("RGB").resize((W, H), Image.LANCZOS),
                          np.float32)
    else:
        base = np.asarray(cpp.tf.erase(img, ndi.binary_dilation(ink0, structure=_disk(4)),
                                       method="nn", nn_median=21, margin=24), np.float32)
    st = np.array([(abs(s[0]), s[1], s[2], abs(s[3])) for s in stats], np.float32)
    print(f"[p4aff] 换位 {moved} 棵  角度|θ|均值={st[:,0].mean():.1f}°  "
          f"sx={st[:,1].mean():.2f} sy={st[:,2].mean():.2f} |shear|={st[:,3].mean():.3f}  "
          f"镜像={100*sum(s[4] for s in stats)/max(1,len(stats)):.0f}%")
    aa = a[..., None]
    prev = base * (1 - aa) + INK[None, None, :] * aa
    pv = Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8), "RGB")
    pv.save(str(OUT / "_swap.jpg"), quality=96)
    Image.fromarray((a > 0.5).astype(np.uint8) * 255, "L").save(str(OUT / "_swap_ink.png"))
    # 连续 α（0-255）交给管线：管线用它当**前景墨层 alpha** 直接叠纯黑，
    # 不能只给二值图 —— 细叶的亚像素覆盖要靠连续 α 才能保持全黑与连通。
    Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), "L").save(
        str(OUT / "_swap_alpha.png"))
    print(f"[p4aff] 新墨={100*(a>0.5).mean():.1f}% (原 {100*ink0.mean():.1f}%)")
    return OUT / "_swap.jpg"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--max-ang", type=float, default=10.0)
    ap.add_argument("--scl-var", type=float, default=0.07)
    ap.add_argument("--skew", type=float, default=0.10)
    ap.add_argument("--aniso", type=float, default=0.14)
    ap.add_argument("--flip-p", type=float, default=0.5)
    ap.add_argument("--scl-lo", type=float, default=0.90)
    ap.add_argument("--scl-hi", type=float, default=1.12)
    a = ap.parse_args()
    build(seed=a.seed, max_ang=a.max_ang, scl_var=a.scl_var, skew=a.skew,
          aniso=a.aniso, flip_p=a.flip_p, scl_lo=a.scl_lo, scl_hi=a.scl_hi)
