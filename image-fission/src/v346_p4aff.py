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
from skimage.segmentation import watershed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp            # noqa: E402
from v345c_p4seg import trunks, partition, _disk       # noqa: E402

OUT = ROOT / "jobs" / "v346_p4aff"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
INK = np.array([12.0, 10.0, 9.0], np.float32)
SS = int(__import__('os').environ.get('P4_SS', '6'))


def _aff(ang_deg, sx, sy, shear, flip):
    """A = diag(sx,sy) · R(θ) · Sh(k) · M，绕原点（把原点取在树基上）。"""
    th = np.radians(ang_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float64)
    Sh = np.array([[1.0, shear], [0.0, 1.0]], np.float64)
    S = np.array([[sx, 0.0], [0.0, sy]], np.float64)
    M = np.array([[-1.0, 0.0], [0.0, 1.0]], np.float64) if flip else np.eye(2)
    return (S @ R @ Sh @ M).astype(np.float32)


def place(alpha, donor_xy, target_xy, A, box):
    """在 box 内输出 A 变换后的供体（×SS 超采样 + 双线性 + 均值降采样）。

    `alpha` 可以是 2D（掩膜/灰度）或 3D（RGB）——v387 起同时搬运**供体原色**，
    这是"不许一块块"的关键（见 build() docstring ③）。
    """
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
    co = [sy.ravel(), sx.ravel()]
    if alpha.ndim == 2:
        got = ndi.map_coordinates(alpha, co, order=1, mode="constant", cval=0.0)
        return got.reshape(ny, SS, nx, SS).mean((1, 3))
    # scipy.map_coordinates 要求坐标轴数 == 输入维数 → 多通道逐通道搬（3 次，开销可忽略）
    nch = alpha.shape[2]
    acc = None
    for c in range(nch):
        g = ndi.map_coordinates(alpha[..., c], co, order=1, mode="constant", cval=0.0)
        g = g.reshape(ny, SS, nx, SS).mean((1, 3))
        if acc is None:
            acc = np.empty((ny, nx, nch), np.float32)
        acc[..., c] = g
    return acc


def collect(img, ink0, ink_a0, W, H, min_ink=120):
    """按树干定位 + **测地洪泛**归属，抽出每棵树的 alpha。

    v387⑤（用户第 18 轮否决"乱七八糟一块一块的"）：
      旧版用 **Euclidean Voronoi**（按到树干轴的直线距离）把墨分给最近的干。
      棕榈是**细高的**：树冠（叶片团）离树干轴可达 150px 以上，常比邻居的干更远
      → 树冠被判给邻树 → **干走了而冠留在原地**（或反之）→ 用户看到的
      "乱七八糟"：漂浮的冠 + 孤立的干 + 满地游离碎块（1:1 目检确认）。
      修法：**测地归属** —— 以树干为种子、沿墨迹洪泛（细缝由 3px 膨胀桥接），
      每个墨像素归给**沿墨连通的最近树干**。冠与自己的干在墨上连通，
      因此冠必然随自己的干一起搬；邻树不连通则抢不走。
      （对照旧结论：直接用"全局连通域"取树会把 51 倍面积卷进来——那是**全图**
      连通；测地洪泛是**以每个干为源**的分水岭，天然按连通分割，不会卷邻树。）
    """
    _, segs = trunks(ink0, H, W, 25, 60)
    segs = [s for s in segs if 0.06 * H < s["h"] < 0.62 * H]
    if not segs:
        return []
    # ① 树干种子（沿树干轴的窄带 ∩ 墨）
    mk = np.zeros((H, W), np.int32)
    for i, s in enumerate(segs):
        x = int(np.clip(round(float(s["x"])), 0, W - 1))
        y0 = int(np.clip(round(float(s["y_top"])), 0, H - 1))
        y1 = int(np.clip(round(float(s["y_bot"])), 0, H))
        if y1 <= y0:
            y1 = min(H, y0 + 1)
        band = np.zeros((H, W), bool)
        band[y0:y1, max(0, x - 2):min(W, x + 3)] = True
        band &= ink0
        if int(band.sum()) < 2:
            band[max(0, y1 - 2):y1, x:min(W, x + 1)] = True
        mk[band] = i + 1
    # ② 测地洪泛：扁平场 + mask ⇒ 按 mask 内路径长度展开（≈连通最近邻）
    mask = ndi.binary_dilation(ink0, structure=_disk(3))
    lab = watershed(np.zeros((H, W), np.uint8), mk, mask=mask)
    # ③ 洪泛不到的孤立墨岛 → 就近指派（EDT 最近标签）
    unreach = (lab == 0) & mask
    if unreach.any() and mk.any():
        _, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(unreach, mk[ind[0], ind[1]], lab)
    # ④ 丢弃过小的干域（噪点误检），其墨**就近并入保留树** → 不留原位残片
    area = np.array([int(((lab == i + 1) & ink0).sum()) for i in range(len(segs))])
    keep = [i for i in range(len(segs)) if area[i] >= min_ink]
    if not keep:
        keep = list(range(len(segs)))
    if len(keep) < len(segs):
        cy = np.zeros(len(segs)); cx = np.zeros(len(segs))
        for i in range(len(segs)):
            ys, xs = np.where(lab == i + 1)
            if len(ys):
                cy[i] = ys.mean(); cx[i] = xs.mean()
        remap = np.arange(len(segs) + 1)
        ks = np.array(keep)
        for i in range(len(segs)):
            if i in keep:
                continue
            d = np.hypot(cx[ks] - cx[i], cy[ks] - cy[i])
            remap[i + 1] = int(ks[int(np.argmin(d))]) + 1
        lab = remap[lab]
    a_rgb = np.asarray(img, np.float32)
    trees = []
    for i in keep:
        own = (lab == i + 1) & ink0
        if int(own.sum()) < min_ink:
            continue
        s = segs[i]
        # ② 小元素（树基旁草痕/短横线）**随最近的树一起搬**，不做就地保留。
        #    实测"留在原位"会让它们变成悬空孤块（1:1 目检 = 一撮浮在迷彩上的脏点），
        #    因为周围那棵树已经走了。跟着最近的树干走 → 仍贴在一棵树旁，观感自然。
        sup = ndi.binary_dilation(own, structure=_disk(1), iterations=6, mask=ink0)
        full = ink_a0 * sup
        rows = np.where(full.any(1))[0]
        cols = np.where(full.any(0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        # v387③：**连同供体的原色一起搬**（只搬 α=1 的墨/杆身像素；α=0 的迷彩档
        # 不参与，因为最终合成是 base*(1-α)+rgb*α，α=0 处 RGB 被完全丢弃）。
        # 这是"不许一块块"的关键：原图树干是「棕褐杆身 + 纯黑横档」，只搬黑白掩膜
        # 再统一涂常数黑 → 杆身/抗锯齿过渡像素全变黑 → 档与缝糊成一根实心条。
        rgb = a_rgb * (full > 0.5)[..., None].astype(np.float32)
        trees.append(dict(own=own, cx=float(s["x"]), base=float(s["y_bot"]),
                          x0=int(cols.min()), x1=int(cols.max()),
                          y0=int(rows.min()), y1=int(rows.max()),
                          trunk=float(max(1.0, s["y_bot"] - s["y_top"])), alpha=full,
                          rgb=rgb))
    return trees


def build(seed=20260917, max_ang=12.0, scl_var=0.06, skew=0.0, aniso=0.0, flip_p=0.5,
          pad=10, scl_lo=0.94, scl_hi=1.07, knee_lo=0.15, knee_w=0.20,
          identity=False, outdir=None, gauss=0.0):
    """v387（用户第 18 轮否决"乱七八糟一块一块的…不许一块块碎片化"）：

    量化根因（`_diag_p4.py`，ORIG vs v386 成品，均在 lum<30 墨迹掩膜上）：
      · 连通块数 832 → **193**（原图的 747 个 <12px 小件 = **树干上的细横档**，
        成品只剩 34 个）→ 细横档被**焊成实心竖条** = 用户看到的"一块块"；
      · 平均笔宽 7.80px → 8.30px（+6%）。
    ⇒ 两个真凶都在本文件的**重采样**环节，与"换位"这个思路无关：
      ① `_aff` 带 **切变 shear + 各向异性缩放 sx≠sy** → 线宽非等比变化；
      ② α→墨的映射用**低膝点** `(α-0.06)/0.34` → 两档之间的间隙像素（α≈0.25~0.35，
         6x 超采样均值降采样后必然出现）被判成实心墨 → **横档之间的缝被填死**。
    ⇒ 修法：**只允许相似变换**（等比缩放 + 旋转 + 平移，禁切变/各向异性）+ 把
      关断点抬到 0.5 附近（陡膝），让 ≤50% 覆盖度的像素保持"非墨"，缝就是缝。
       相似变换下"7px 的线转 12° 还是 7px"，线质可逐像素还原（原图矢量线稿的根）。
    """
    _ = aniso  # 保留形参以兼容旧调用；本版禁止各向异性（见 docstring ①）
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    a_rgb_orig = np.asarray(img, np.float32)
    ink0 = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    # v387④：**不再对掩膜做高斯模糊**（旧版 sigma=0.7）。
    # 模糊会在每条笔画外生成 ~1.5px 的 α 晕圈；原图树干是**细密横档**（档间距仅几 px），
    # 相邻两档的晕圈在缝里叠加 → 缝被提亮/变暗成"半墨" → 档与缝连成一体（实测
    # 恒等自检：原图 lum<30 连通块 832 → 仅 257）。去掉模糊后，α 由 6x 超采样
    # 自带的抗锯齿生成（边缘过渡 ≤1/6 px），缝就是缝。
    ink_a0 = (ndi.gaussian_filter(ink0.astype(np.float32), float(gauss))
              if float(gauss) > 0 else ink0.astype(np.float32))
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
    if identity:
        # 自检模式：恒等置换 + 零角/零缩放/不镜像 → 输出应与原图**逐像素同档**。
        # 用它把"重采样/合成链"的质量损失与"换位本身"分离开（MEMORY：先量化验证假设）。
        perm = np.arange(n)
        max_ang = 0.0
        scl_var = 0.0
        flip_p = 0.0
        scl_lo, scl_hi = 1.0, 1.0
        print("[p4aff] *** IDENTITY 自检：理想输出 = 原图（ink%/块数/笔宽 应全部还原）***")
    print(f"[p4aff] 树={n}  尺寸匹配分配 成本均值="
          f"{cost[_ri, _ci].mean():.3f}（随机错排基线≈{cost[np.arange(n),(np.arange(n)+1)%n].mean():.3f}）")

    # ⚠️ 必须按**整棵树的实际 alpha**取并集来擦除底板原画，不能用 Voronoi 归属 own：
    # own 只覆盖 Voronoi 格内的墨，而树的真实连通范围会**溢出**到邻居格里 →
    # 那些溢出像素没被擦掉，就变成留在原位的"原树残片"（实测中碎块 40→131）。
    union = np.zeros((H, W), bool)
    for t in trees:
        union |= (t["alpha"] > 0.35)
    acc = (ink_a0 * (~ndi.binary_dilation(union, structure=_disk(3)))).astype(np.float32)
    # v387③：颜色累加器。初值 = **原图本身**——`acc` 里那部分"不属于任何树而留在原位"
    # 的墨（残件）必须带上它自己的原色，否则会被当成"α>0 但 rgb=0"而涂黑。
    accc = (a_rgb_orig * acc[..., None]).copy()

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
        # v387：**等比相似变换**（用户第 18 轮）——只用一个尺度 s，
        # 取长/宽两个拟合值的几何均值。旧版 sx≠sy（各向异性）会把 7px 的横档
        # 横向拉成 9px、纵向压成 6px → 档与档之间的缝被挤死 = "一块块"。
        # v388：**尺度改由"树干长度比"决定**（旋转不变），不再用 bbox 拟合。
        # 旧版 s=√(tW/w_r · tH/h_r)：旋转会把细高棕榈的 bbox 撑大（h·sinθ），
        # 于是 s 被强行压小 → 整图墨量 22.6%→19.8%（用户读作"少了几棵树/稀"）。
        # 树干长度在旋转下不变 → 用它比值定尺，旋转不再"以缩小为代价"。
        t_tr = float(t.get("trunk", 1.0))
        d_tr = float(d.get("trunk", 1.0))
        s = (t_tr / max(1.0, d_tr)) * float(rng.uniform(1 - scl_var, 1 + scl_var))
        s = float(np.clip(s, scl_lo, scl_hi))
        sx = sy = s
        A = _aff(ang, sx, sy, shear, flip)
        pts = []
        for ddx, ddy in product((d["x0"] - d["cx"], d["x1"] - d["cx"]),
                                (d["y0"] - d["base"], d["y1"] - d["base"])):
            uv = A @ np.array([ddx, ddy], np.float32)
            pts.append((t["cx"] + float(uv[0]), t["base"] + float(uv[1])))
        bx0 = int(min(p[0] for p in pts) - pad)
        bx1 = int(max(p[0] for p in pts) + 1 + pad)
        by0 = int(min(p[1] for p in pts) - pad)
        by1 = int(max(p[1] for p in pts) + 1 + pad)
        if bx1 - bx0 < 8 or by1 - by0 < 8:
            continue
        # v387⑥ **四方连续环绕贴回**：原图是 repeating wallpaper，越界部分绕到对边。
        # 旧版把 bbox 硬裁到 0..W/H → 边缘树被截断 → 墨量 22.63%→19~20%（用户会读成
        # "少了几棵树"）。环绕既保住密度、又符合图案语义。位移编码进 **target**：
        # 贴到 (t.cx+dx, t.base+dy)，box 取"平移后 bbox ∩ 画框"。
        hit = 0
        for dx in (0, -W, W):
            for dy in (0, -H, H):
                ix0 = max(0, bx0 + dx); ix1 = min(W, bx1 + dx)
                iy0 = max(0, by0 + dy); iy1 = min(H, by1 + dy)
                if ix1 - ix0 < 4 or iy1 - iy0 < 4:
                    continue
                tt = (t["cx"] + dx, t["base"] + dy)
                got = place(d["alpha"], (d["cx"], d["base"]), tt, A,
                            (ix0, iy0, ix1, iy1))
                if got is None:
                    continue
                gotc = place(d["rgb"], (d["cx"], d["base"]), tt, A,
                             (ix0, iy0, ix1, iy1))
                subA = acc[iy0:iy1, ix0:ix1]
                bb = got > subA
                subA[bb] = got[bb]
                if gotc is not None:
                    subC = accc[iy0:iy1, ix0:ix1]
                    subC[bb] = gotc[bb]
                hit += 1
        if hit:
            moved += 1
            stats.append((ang, sx, sy, shear, flip))
    # α→墨 的映射：
    # v387③ 起改为**线性 AA 三角**（膝点 0.25 / 宽 0.50，即 0.25→0、0.75→1、0.5→0.5）。
    # 为什么现在敢用"中间膝"：旧版只能靠低膝去补墨量，是因为**颜色被写死成常数黑**，
    # 细叶中心 α≈0.5 处若判 0 就会变半透明灰 → 只能低膝补偿。现在颜色走
    # **供体原色直传**（见 collect/build docstring ③），半覆盖像素带的是供体自己的
    # 颜色，"半透明"不再等于"灰"，于是可以用正确的 50% 关断点：
    # 缝保持缝（缝的覆盖度 ~0.2 < 0.25 → 0），墨量也不再掉档。
    a = np.clip((acc - float(knee_lo)) / max(1e-6, float(knee_w)), 0.0, 1.0)
    # 清孤立碎点：阈值降到 4px（旧版 8px 会把原图就有的细横档当噪点删掉）
    _lab, _k = ndi.label(a > 0.5, structure=np.ones((3, 3), bool))
    if _k:
        _sz = np.bincount(_lab.ravel())
        _small = np.zeros(_k + 1, bool)
        _small[1:] = _sz[1:] < 4
        a[_small[_lab]] = 0.0
        print(f"[p4aff] 清碎点(<4px) {int(_small[1:].sum())} 个")

    wr = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
    if wr.exists():
        base = np.asarray(Image.open(wr).convert("RGB").resize((W, H), Image.LANCZOS),
                          np.float32)
    else:
        base = np.asarray(cpp.tf.erase(img, ndi.binary_dilation(ink0, structure=_disk(4)),
                                       method="nn", nn_median=21, margin=24), np.float32)
    st = np.array([(abs(s[0]), s[1], s[2], abs(s[3])) for s in stats], np.float32)
    od = Path(outdir) if outdir else OUT
    od.mkdir(parents=True, exist_ok=True)
    print(f"[p4aff] 换位 {moved} 棵  角度|θ|均值={st[:,0].mean():.1f}°  "
          f"sx={st[:,1].mean():.2f} sy={st[:,2].mean():.2f} |shear|={st[:,3].mean():.3f}  "
          f"镜像={100*sum(s[4] for s in stats)/max(1,len(stats)):.0f}%")
    # 覆盖诊断：α 覆盖 vs 原墨迹 vs 供体 own 面积之和 —— 定位"掉墨"到底掉在哪
    _cov = float((a > 0.5).mean())
    _own = float(sum(t["own"].sum() for t in trees)) / (W * H)
    print(f"[p4aff] 覆盖诊断：α覆盖={100*_cov:.2f}%  供体own合计={100*_own:.2f}%  "
          f"原ink0={100*ink0.mean():.2f}%")
    aa = a[..., None]
    # **预乘 alpha 的原色层**：交给管线后它就是 `out = base*(1-α) + rgbp`。
    # 预乘（rgbp = 供体原色 × α）→ 管线端只做一次加法，边界 1px 自然抗锯齿。
    rgbp = np.clip(accc * aa, 0, 255)
    prev = base * (1 - aa) + rgbp
    pv = Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8), "RGB")
    pv.save(str(od / "_swap.jpg"), quality=96)
    Image.fromarray((a > 0.5).astype(np.uint8) * 255, "L").save(str(od / "_swap_ink.png"))
    # 连续 α（0-255）交给管线：管线用它当**前景墨层 alpha**。
    Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), "L").save(
        str(od / "_swap_alpha.png"))
    # v387③：**预乘原色层**（管线直接取用；旧版是常数黑，把杆身/缝一起涂黑 = "一块块"）
    Image.fromarray(rgbp.astype(np.uint8), "RGB").save(str(od / "_swap_rgb.jpg"),
                                                       quality=97)
    print(f"[p4aff] 新墨={100*(a>0.5).mean():.1f}% (原 {100*ink0.mean():.1f}%)")
    return od / "_swap.jpg"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--max-ang", type=float, default=12.0)
    ap.add_argument("--scl-var", type=float, default=0.06)
    ap.add_argument("--skew", type=float, default=0.0)      # v387：禁切变
    ap.add_argument("--aniso", type=float, default=0.0)     # v387：禁各向异性
    ap.add_argument("--flip-p", type=float, default=0.5)
    ap.add_argument("--scl-lo", type=float, default=0.94)
    ap.add_argument("--scl-hi", type=float, default=1.07)
    ap.add_argument("--knee-lo", type=float, default=0.15)
    ap.add_argument("--knee-w", type=float, default=0.20)
    ap.add_argument("--identity", action="store_true",
                    help="自检：恒等置换+零角+零缩放 → 输出应逐像素还原原图")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--gauss", type=float, default=0.0)
    a = ap.parse_args()
    build(seed=a.seed, max_ang=a.max_ang, scl_var=a.scl_var, skew=a.skew,
          aniso=a.aniso, flip_p=a.flip_p, scl_lo=a.scl_lo, scl_hi=a.scl_hi,
          knee_lo=a.knee_lo, knee_w=a.knee_w, identity=a.identity, outdir=a.outdir,
          gauss=a.gauss)
