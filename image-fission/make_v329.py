"""make_v329.py — v329 五图裂变（按用户 2026-09-15 反馈重做）

用户反馈与本版对策：
  ① "文本裂变不许靠遮挡覆盖原文本来做" → 文字一律**原位 LaMa 抹除 + 按原材质重画**，
     不出现任何矩形色块/平涂（见 styles/textfix.draw_line_material，route='tile'）。
  ② pinterest3(牛仔贴布字)：新词材质取自原字母**纯字肉小样平铺**（base+高通细纹），
     描边按原图环剖面跟随新字形重建 → 同风格/同材质/同字体语言，不同单词。
  ③ pinterest4(迷彩棕榈)：迷彩底**不做 5 色硬量化**（旧版"色块被截取"的元凶），
     改成"抹树 → 色板标签扭曲（边界锐利）→ 回填原色板"；树木用**程序化线稿棕榈**
     逐棵重画（角度/弯度/叶形/树冠朝向各不相同），不再复制同一棵树。
  ④ 6978(蝙蝠徽章)：主体蝙蝠（BACARDÍ 注册商标图形，必须改写）用 styles/bat_art
     重画成**不同姿态/翼展角/扇贝细节**的纹章蝙蝠；文字原位改写。
  ⑤ pinterest6(金属尖刺标题)：标题用 Harrlogos XL 生成的**黑金属尖刺字**替换，
     不复用旧版 MetalMania 平涂白字。

用法：venv/Scripts/python.exe make_v329.py [iid ...]
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, '.')
sys.path.insert(0, 'src')
from styles import textfix as tf
from styles import base as sbase
from styles import camo_palm_pattern as cpp
from styles import bat_art
import make_v328 as v328

ROOT = Path('.')
OUT = ROOT / 'jobs' / 'router_out_v329'
OUT.mkdir(parents=True, exist_ok=True)
VIS = ROOT / 'jobs' / '_probe'
VIS.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / 'regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}
SRC = Path('E:/Desktop/图裂变测试图')


def _new_word(iid, default):
    return (BY[iid].get('new_text') or {}).get('primary') or default


# ---------------------------------------------------------------- pinterest3
def _p3_subject_swap(base_img, gen_path):
    """把 ComfyUI 裂变出的**新蝴蝶**贴回**原图背景**（只换主体，不换背景/版式）。

    为什么需要这一步：pinterest3 的 mode1 输出虽然蝴蝶物种/材质对了（牛仔蓝刺绣 + 磨边），
    但整张图的背景被 SDXL 改了（顶部出现蓝色花纹、底色偏白）——违反"版式/配色不变"。
    做法：原蝴蝶(含影)按 nn 抹成灰底 → 新蝴蝶按"与底色差"取蒙版贴回原位。
    """
    gen = Image.open(gen_path).convert('RGB')
    if gen.size != base_img.size:
        gen = gen.resize(base_img.size, Image.LANCZOS)
    W, H = base_img.size
    a = np.asarray(base_img, np.float32)
    g = np.asarray(gen, np.float32)
    y0, y1 = 495, 1180
    ab = a[y0:y1]
    gb = g[y0:y1]
    bg_a = np.median(np.concatenate([ab[:24].reshape(-1, 3), ab[-24:].reshape(-1, 3)]), 0)
    bg_g = np.median(np.concatenate([gb[:24].reshape(-1, 3), gb[-24:].reshape(-1, 3)]), 0)
    d_old = np.abs(ab - bg_a).sum(2)
    d_new = np.abs(gb - bg_g).sum(2)
    m_old = tf.ndi.binary_closing(d_old > 34, structure=tf._disk(5))
    m_new = tf.ndi.binary_closing(d_new > 30, structure=tf._disk(4))
    m_new = tf.ndi.binary_fill_holes(m_new)

    def _big(m):
        lab, n = tf.ndi.label(m, structure=np.ones((3, 3), bool))
        if not n:
            return m, None
        sz = tf.ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        i = int(np.argmax(sz)) + 1
        mm = lab == i
        ys, xs = np.where(mm)
        return mm, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    m_old, bb_old = _big(m_old)
    m_new, bb_new = _big(m_new)
    if bb_old is None or bb_new is None:
        return base_img
    # 新蝴蝶按其自身 alpha 裁出，缩放到**原蝴蝶紧框**，贴回原位置 —— 构图/占位不变
    core = tf.ndi.binary_erosion(m_new, structure=tf._disk(2))
    alph = np.where(core, 1.0, np.clip((d_new - 34.0) / 50.0, 0.0, 1.0))
    bx0, by0, bx1, by1 = bb_new
    ax0, ay0, ax1, ay1 = bb_old
    aw, ah = ax1 - ax0, ay1 - ay0
    nw, nh = bx1 - bx0, by1 - by0
    sc_ = min(aw / nw, ah / nh)                      # 等比缩放，避免拉扁蝴蝶
    tw_, th_ = max(1, int(round(nw * sc_))), max(1, int(round(nh * sc_)))
    tx_ = ax0 + (aw - tw_) // 2
    ty_ = ay0 + (ah - th_) // 2
    crop = Image.fromarray(gb[by0:by1, bx0:bx1].astype(np.uint8), 'RGB').resize((tw_, th_), Image.LANCZOS)
    acrop = Image.fromarray((alph[by0:by1, bx0:bx1] * 255).astype(np.uint8), 'L').resize((tw_, th_), Image.LANCZOS)
    g_small = np.asarray(crop, np.float32)
    al = (np.asarray(acrop, np.float32) / 255.0)[..., None]

    full_old = np.zeros((H, W), bool)
    full_old[y0:y1] = m_old
    erased = tf.erase(base_img, tf.ndi.binary_dilation(full_old, structure=tf._disk(3)),
                      method='nn', nn_median=41)
    out = np.asarray(erased, np.float32)
    py, px = ty_ + y0, tx_                      # bb_old/bb_new 都在 y0 偏移坐标系里
    reg = out[py:py + th_, px:px + tw_]
    out[py:py + th_, px:px + tw_] = reg * (1 - al) + g_small * al
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGB')
    res.save(VIS / 'v329_p3_subj_swap.jpg', quality=95)
    print(f'[p3-subj] old_bb={bb_old} new_bb={bb_new} fit=({tw_}x{th_})@({tx_},{ty_})')
    return res


def do_pinterest3():
    """牛仔贴布字母：① 主体蝴蝶按 ComfyUI 裂变结果换新（保留原背景/版式）
    ② 文字材质平铺重画（同材质/同描边，不同单词）。"""
    ic = BY['pinterest3']
    src_img = Image.open(ic['path']).convert('RGB')
    # v330: 不再用 ComfyUI 蝴蝶（_p3_lockA_0.jpg 带大量噪点/糊边），
    # 改用原图干净蝴蝶（零噪点）。后续可在此处接 subject_morph 做姿态裂变。
    img = src_img
    W, H = img.size
    band = (0, 60, W, 348)                       # 只取字母带，排除下方小蝴蝶
    det = src_img                                # 字母检测/取材质一律用**原图**：
    # 主体交换后灰底统计值变了，直接在交换图上检测会只剩 1 个字母（实测）。
    letters, patch = tf.detect_letter_patches(det, band, loose=16, tight=45,
                                              white_thr=238, seed_erode=3, min_seed=200)
    if not letters:
        raise RuntimeError('pinterest3 未检出字母')
    ux1 = min(b[0][0] for b in letters); uy1 = min(b[0][1] for b in letters)
    ux2 = max(b[0][2] for b in letters); uy2 = max(b[0][3] for b in letters)
    md = tf.ndi.binary_dilation(patch, structure=tf._disk(14))
    # v336（用户："通过叠加乱七八糟达到裂变糊弄我"）：v335 用 nn 填充字母带，nn 从
    # 未掩净的牛仔蓝字母取样 → 右缘漏出蓝色斑块（v336 取证 v329_p3_erased.jpg）。
    # 字母带背景是**纯平灰 (233,233,233)**（全图 63 万像素中值实测），改 const 填充：
    # 掩膜 = 检测补丁 ∪ 带内全部非背景像素（原词残边/未知残片一网打尽），填充结果
    # 与背景逐像素同色，零色块、零残片。
    bg3 = (233, 233, 233)
    a3 = np.asarray(img, np.float32)
    band3 = np.zeros((H, W), bool)
    band3[60:348, :] = True
    notbg3 = np.abs(a3 - np.array(bg3, np.float32)[None, None, :]).sum(2) > 30
    md = md | (band3 & notbg3)
    md = tf.ndi.binary_dilation(md, structure=tf._disk(8))
    vis = np.asarray(det).copy(); vis[md] = [255, 0, 0]
    Image.fromarray(vis).save(VIS / 'v329_p3_mask.png')
    out = tf.erase(img, md, method='const', const_color=bg3, const_feather=6)
    out.save(VIS / 'v329_p3_erased.jpg', quality=95)
    word = _new_word('pinterest3', 'DENIM')
    res = tf.draw_line_material(out, word, (ux1, uy1, ux2, uy2), 'rock_cond_b', det, patch,
                                src_letters=letters, cap_scale=1.0, ring_px=11,
                                jitter_deg=1.5, seed=3, fit='squeeze', mat_mode='tile',
                                swatch_size=64, swatch_erode=7, swatch_gap=3,
                                swatch_min_cov=0.96, detail_sigma=12, swatch_full=False,
                                width_frac=1.0,
                                shadow=dict(dx=7, dy=7, blur=7.0, color=(130, 130, 130), strength=0.55),
                                inner_ring=6, inner_ring_col=(200, 205, 215),
                                ring_dark=0.58, seam_px=3)

    # ---- v332: 蝴蝶主体姿态裂变（用户："主体蝴蝶跟原图没看出来区别"）----
    # subject_morph 像素位移 = 材质/边缘零损失；位移场必须用**层内坐标**（v331 之鉴）。
    # 主蝴蝶=双翼绕肩枢轴旋转+翼尖下垂；小蝴蝶=自转微角；虚线点保持原样。
    from styles import subject_morph as smod
    sa_ = np.asarray(src_img, np.float32)
    notbg = sa_.min(2) < 222
    cand = np.zeros((H, W), bool)
    cand[370:, :] = notbg[370:, :]               # 排除字母带（虚线点后续按面积跳过）
    cand = tf.ndi.binary_closing(cand, structure=tf._disk(3))
    clab, cn = tf.ndi.label(cand, structure=np.ones((3, 3), bool))
    rng3 = np.random.default_rng(303)
    n_bf = 0
    small_layers = []                            # v339：(rgba, 层内质心, 全图质心, area)
    trail_ends = []                              # v339：每条轨迹的 (近端, 远端) 全图坐标
    dots = np.zeros((H, W), bool)                # v336: 虚线点收集（轨迹重画用）
    for i in range(1, cn + 1):
        m_i = clab == i
        area = int(m_i.sum())
        if area < 2500:                          # 小组件 → 分拣：圆点 vs 噪声
            if 60 <= area <= 400:
                ys_, xs_ = np.where(m_i)
                if (xs_.max() - xs_.min()) <= 22 and (ys_.max() - ys_.min()) <= 22:
                    dots |= m_i                  # 圆点（实测 20 颗：99-150px、11-20px 宽）
            continue
        ys, xs = np.where(m_i)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        R = 0.5 * max(x1 - x0, y1 - y0)
        # v338：pad 0.30R+16 → 0.45R+20（小元素新增 ±11.5° 斜掠 + 1.3x 体型，
        # 形变后可能超出原 bbox；pad 不够会被裁掉翼尖）
        pad = int(R * 0.45) + 20
        bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
        bx1, by1 = min(W, x1 + pad), min(H, y1 + pad)
        sub_m = np.zeros((H, W), bool)
        sub_m[by0:by1, bx0:bx1] = m_i[by0:by1, bx0:bx1]
        rgba, mbox = smod.make_layer(src_img, sub_m, feather=1.4, box=(bx0, by0, bx1, by1))
        hh, ww = rgba.shape[:2]
        if area > 50000:                         # 主蝴蝶：翼姿态变化（已获用户认可，不动）
            dyy, dxx = smod.wing_pose((ww, hh), cx=float(xs.mean()) - bx0,
                                      cy=float(ys.mean()) - by0,
                                      theta=-0.34, pivot_dx=20.0, ramp=(12.0, 150.0),
                                      k_up=0.0, span=1.10, tip_flick=-26.0,
                                      tip_ramp=(80.0, 170.0))
        else:
            # v336（用户："通过叠加乱七八糟达到裂变糊弄我"）：废 v335 镜像+缩放
            # （朝向/大小一变 = 贴纸感，用户一眼识破）。改**原位翅姿形变**：与主蝶
            # 同机制的 wing_pose 小振幅版——位置/朝向/大小全部不变，只有翼尖剪影
            # 上下摆动（两只蝶正负交替），看得见变化且绝无"贴上去"的痕迹。
            # v337（用户第 8 轮："蝴蝶我说的是蝴蝶的小元素蝴蝶去裂变修改"）：
            # ±0.20rad 在小蝶(~100px)上肉眼看不见 → 幅度按**元素自身尺度**放大：
            # ±0.46rad 翅转角 + 1.20x 翼展 + 翼尖甩 26px（蝶体不动、位置/朝向不变）。
            # v338（用户第 9 轮："蝴蝶这类的有小元素的可以裂变多一点，只对主体要求有相关，
            # 小元素知道跟主体有关就行"）→ **小元素放开幅度**：从 ±10~18% 各向异性缩放 +
            # ±9° 翅角，提到 **±30% 缩放 + ±17° 翅角 + 翼展 0.88/1.12 + 整体 ±11.5° 斜掠**。
            # 依据：小元素只需"看得出还是蝴蝶、跟主体(刺绣蝶)是一家"，不受"同位同大"硬约束；
            # 位置仍**锚在原处**（绕自身质心变化，不搬移/不缩放整体尺寸），所以仍读得出关联。
            # 两条物理通道沿用（见下），互不干扰，细附件不会被拉丝。
            cxl, cyl = float(xs.mean()) - bx0, float(ys.mean()) - by0
            # 幅度定稿（v339，用户第 10 轮："小元素裂变效果可以比主图要大一点，多点创意"）
            # ⚠️ 实测：把"翅角 + 翼展 + 各向异性缩放 + 斜掠 + 翼尖甩"五重叠加在同一只
            # 小蝶上 → 刺绣细节被搅成"糊团蝴蝶"（探针 p3_p1.jpg 第 2 行）。
            # 小元素最干净的"更狠" = **刚体旋转 + 等比放大 + 小翅角**（等比/旋转是线性
            # 场，刺绣针脚零扭曲）。定稿：
            #   偶数只：×1.22 + 右倾 15° + 翼展 0.92（收翅立姿）
            #   奇数只：×1.14 + 左倾 13° + 翼展 1.08（展翅扁姿）
            ev = (n_bf % 2 == 0)
            if ev:
                sx = sy = 1.22
                th, span_, tilt, flick = 0.16, 0.92, 0.26, 0.0
            else:
                sx = sy = 1.14
                th, span_, tilt, flick = -0.13, 1.08, -0.23, 0.0
            dyr, dxr = smod.wing_pose((ww, hh), cx=cxl, cy=cyl, theta=th,
                                      pivot_dx=0.34 * R, ramp=(0.30 * R, 1.00 * R),
                                      k_up=0.0, span=span_, tip_flick=flick,
                                      tip_ramp=(0.62 * R, 1.10 * R))
            _sol = tf.ndi.binary_opening(m_i[by0:by1, bx0:bx1], structure=tf._disk(3))
            if _sol.sum() < 0.25 * max(1, int(m_i[by0:by1, bx0:bx1].sum())):
                _sol = m_i[by0:by1, bx0:bx1]
            _wm = tf.ndi.gaussian_filter(
                tf.ndi.binary_dilation(_sol, structure=tf._disk(4)).astype(np.float32), 3.0)
            _wm = np.clip(_wm, 0.0, 1.0)
            _Yl, _Xl = np.mgrid[0:hh, 0:ww]
            _Yl = _Yl.astype(np.float32)
            _Xl = _Xl.astype(np.float32)
            dyy = dyr * _wm + (1.0 / sy - 1.0) * (_Yl - cyl)
            dxx = dxr * _wm + (1.0 / sx - 1.0) * (_Xl - cxl)
            # 整体斜掠：绕自身质心刚体旋转（线性场 → 细触角/足只会被等比转，不会拉丝）
            if tilt:
                _ca, _sa = math.cos(-tilt), math.sin(-tilt)
                _u = _Xl - cxl
                _v = _Yl - cyl
                dyy = dyy + ((_u * _sa + _v * _ca) - _v)
                dxx = dxx + ((_u * _ca - _v * _sa) - _u)
        wlay = smod.warp_layer(rgba, (dyy.astype(np.float32), dxx.astype(np.float32)), order=1)
        if area <= 50000:
            small_layers.append((rgba.copy(), (cxl, cyl),
                                 float(xs.mean()), float(ys.mean()), area))
        # v337 关键修复：**先擦原蝶再贴回形变层**。此前只有 paste（原图仍带原蝶）→
        # 形变层叠在原蝶上 = 半透明叠影；振幅一小就"看不出变化"（用户历轮反馈），
        # 放大就"糊成一团"。背景是纯平灰，const 擦除零痕迹。
        _em = np.zeros((H, W), bool)
        _em[by0:by1, bx0:bx1] = m_i[by0:by1, bx0:bx1]
        res = tf.erase(res, tf.ndi.binary_dilation(_em, structure=tf._disk(5)),
                       method='const', const_color=bg3, const_feather=3)
        res = smod.paste_layer(res, wlay, mbox)
        n_bf += 1
    print(f'[pinterest3] butterflies morphed={n_bf}')

    # ---- v336: 虚线轨迹裂变 = **原位重画**（新曲线，端点锚定）----
    # v335 把整条轨迹当刚体旋转+平移（用户："叠加乱七八糟糊弄我"）→ 废。
    # 背景是纯平灰 → 原点 const 擦除零痕迹；新点在原位基础上加"弯曲扰动"重画：
    # 同点数/同半径/同颜色，垂直于轨迹的正弦凸起（首末点权重 0 = 锚定不动），
    # 凸起方向逐轨迹交替 → 轨迹弯曲方向可见变化，但绝无搬移/贴纸感。
    n_tr = 0
    if dots.any():
        from PIL import ImageDraw
        rng_tr = np.random.default_rng(907)
        main_m = np.zeros((H, W), bool)
        for i in range(1, cn + 1):
            if int((clab == i).sum()) > 50000:
                main_m |= (clab == i)
        main_guard = tf.ndi.binary_dilation(main_m, structure=tf._disk(3))
        dl = tf.ndi.binary_dilation(dots, structure=tf._disk(18))
        dlab, dn = tf.ndi.label(dl, structure=np.ones((3, 3), bool))
        for k_ in range(1, dn + 1):
            pts = dots & (dlab == k_)
            if int(pts.sum()) < 350:
                continue
            plab, pn = tf.ndi.label(pts, structure=np.ones((3, 3), bool))
            cents, rads, cols = [], [], []
            sa_src = np.asarray(src_img, np.float32)
            for j in range(1, pn + 1):
                mj = plab == j
                aj = int(mj.sum())
                if aj < 40:
                    continue
                ysj, xsj = np.where(mj)
                cents.append((float(xsj.mean()), float(ysj.mean())))
                # v338d：**圆点必须复刻原样**（原图是深藏青实心点 r≈6）。
                # 旧码 radius=sqrt(area/π) 把抗锯齿外沿算进去 → 偏大；color=整块中值
                # 把边缘与米色底混合的浅像素算进去 → 发灰。实测 2x 目检：变体点变成
                # "灰蓝发虚的大点"，与原图完全不是一个东西。
                # 改：用距离变换取**核心盘**（dt > 0.55·rmax）采样 → 真·点色；
                #     半径取核心半径 rmax（外沿 0.6px 交给 AA），尺寸/色相一致。
                dtj = tf.ndi.distance_transform_edt(mj)
                rmax = float(dtj.max())
                core = mj & (dtj > 0.55 * rmax)
                if int(core.sum()) < 6:
                    core = mj & (dtj > 0.40 * rmax)
                cents[-1] = (float(xsj.mean()), float(ysj.mean()))
                rads.append(rmax + 0.6)
                cols.append(np.median(sa_src[core], 0))
            if len(cents) < 5:
                continue
            C = np.array(cents, np.float32)
            # 贪心链排序（C 形轨迹也适用）：从相互距离最远的一端出发
            d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            i0 = int(np.unravel_index(np.argmax(d2), d2.shape)[0])
            order = [i0]
            unv = set(range(len(C))) - {i0}
            while unv:
                last = order[-1]
                nxt = min(unv, key=lambda j: float(((C[j] - C[last]) ** 2).sum()))
                order.append(nxt)
                unv.discard(nxt)
            C = C[order]
            rads = [rads[j] for j in order]
            trail_ends.append((C[0].astype(np.float32).copy(), C[-1].astype(np.float32).copy()))
            seg = np.hypot(np.diff(C[:, 0]), np.diff(C[:, 1]))
            t = np.concatenate([[0.0], np.cumsum(seg)])
            t = t / max(float(t[-1]), 1e-6)
            tan = C[-1] - C[0]
            L = float(np.hypot(tan[0], tan[1])) + 1e-6
            nvec = np.array([-tan[1], tan[0]], np.float32) / L
            sgn = 1.0 if (k_ % 2) else -1.0
            amp = min(20.0, 0.12 * L) * sgn
            NewC = C.copy()
            for j in range(len(C)):
                bump = amp * math.sin(math.pi * t[j]) ** 1.2
                jit = 0.0 if j in (0, len(C) - 1) else float(rng_tr.uniform(-2.5, 2.5))
                NewC[j] = C[j] + nvec * (bump + jit)
            x0_ = int(min(C[:, 0].min(), NewC[:, 0].min()))
            x1_ = int(max(C[:, 0].max(), NewC[:, 0].max()))
            y0_ = int(min(C[:, 1].min(), NewC[:, 1].min()))
            y1_ = int(max(C[:, 1].max(), NewC[:, 1].max()))
            pad = 24
            bx0_, by0_ = max(0, x0_ - pad), max(0, y0_ - pad)
            bx1_, by1_ = min(W, x1_ + pad), min(H, y1_ + pad)
            em = np.zeros((H, W), bool)
            em[by0_:by1_, bx0_:bx1_] = pts[by0_:by1_, bx0_:bx1_]
            em = tf.ndi.binary_dilation(em, structure=tf._disk(4)) & (~main_guard)
            res = tf.erase(res, em, method='const', const_color=bg3, const_feather=4)
            # 4x 超采样画新点（抗锯齿），逐轨迹中值色
            SS = 4
            cw_, ch_ = bx1_ - bx0_, by1_ - by0_
            canvas = Image.new('L', (cw_ * SS, ch_ * SS), 0)
            dr = ImageDraw.Draw(canvas)
            colj = np.mean(np.stack(cols, 0), 0)
            for j in range(len(NewC)):
                rj = rads[j] * float(rng_tr.uniform(0.92, 1.08))
                px_ = ((NewC[j][0] - bx0_) * SS, (NewC[j][1] - by0_) * SS)
                rr_ = max(1.0, rj * SS)
                dr.ellipse([px_[0] - rr_, px_[1] - rr_, px_[0] + rr_, px_[1] + rr_],
                           fill=255)
            amask = np.asarray(canvas.resize((cw_, ch_), Image.LANCZOS),
                               np.float32) / 255.0
            amask *= (~main_guard[by0_:by1_, bx0_:bx1_]).astype(np.float32)
            reg = np.asarray(res, np.float32).copy()
            sub = reg[by0_:by1_, bx0_:bx1_]
            reg[by0_:by1_, bx0_:bx1_] = sub * (1 - amask[..., None]) \
                + colj[None, None, :] * amask[..., None]
            res = Image.fromarray(np.clip(reg, 0, 255).astype(np.uint8), 'RGB')
            n_tr += 1
    print(f'[pinterest3] trails redrawn={n_tr}')

    # ---- v339 创意小元素修饰：**由主体刺绣蝶派生的迷你蝶飞行队列** ----
    # 用户第 10 轮："小元素裂变效果可以比主图要大一点，多点创意效果，根据主体元素
    # 生成相关小元素修饰。" 做法：取已有的小蝶像素层（与原图同材质/同刺绣质感），
    # 等比缩小 0.58 / 0.40 并旋转，沿每条虚线轨迹的**远端外侧**依次排成一列
    # （近大远小 = 飞行动线延伸），形成"蝶群"。
    # 硬约束：① 只用原图元素像素派生（不程序化新画）；② 避开 DENIM 文本带与主蝶；
    # ③ 贴出画布外则跳过。位置在全图坐标系里做边界与碰撞检查。
    n_mini = 0
    if small_layers:
        # 禁区 = 字母带 ∪ 主蝶紧框(外扩 12) ∪ 小蝶紧框(外扩 12) ∪ 轨迹点(外扩 24)。
        # ⚠️ 不能用"主蝶掩膜"当禁区：主蝶四周的**磨边白须** min-channel>222，不算
        # 前景 → 掩膜漏掉，迷你蝶会压到毛边上（实测第一版就压在主蝶左缘）。
        occ = np.zeros((H, W), bool)
        occ[0:360, :] = True
        for i in range(1, cn + 1):
            m_ = clab == i
            a_ = int(m_.sum())
            if a_ < 2500:
                continue
            ys_, xs_ = np.where(m_)
            occ[max(0, ys_.min() - 12):ys_.max() + 13,
                max(0, xs_.min() - 12):xs_.max() + 13] = True
        occ = tf.ndi.binary_dilation(occ, structure=tf._disk(1))
        # 两块**已验证空场**（原图那里是纯灰底）：左上（文字带下、主蝶上、小蝶左）
        # 与 右下（主蝶下、小蝶右）。迷你蝶排成"近小远大的飞行动线"，朝向小蝶。
        src = max(small_layers, key=lambda s: s[4])          # 用面积最大的那只小蝶做源
        rgba_s = src[0]
        queues = [
            # (x, y, scale, rotate_rad)
            (128, 478, 0.36, -0.30), (232, 452, 0.46, -0.18), (336, 424, 0.58, -0.06),
            (630, 1252, 0.36, 0.34), (528, 1224, 0.46, 0.20), (424, 1192, 0.58, 0.08),
        ]
        for (cx_t, cy_t, sc, ang) in queues:
            lay = Image.fromarray(np.clip(rgba_s, 0, 255).astype(np.uint8), 'RGBA')
            lay = lay.resize((max(1, int(round(lay.width * sc))),
                              max(1, int(round(lay.height * sc)))), Image.LANCZOS)
            lay = lay.rotate(math.degrees(ang), resample=Image.BICUBIC,
                             expand=True, fillcolor=(0, 0, 0, 0))
            la = np.asarray(lay, np.float32)
            if la.shape[2] < 4:
                continue
            lh, lw = la.shape[:2]
            px_ = int(round(cx_t - lw / 2.0))
            py_ = int(round(cy_t - lh / 2.0))
            if px_ < 4 or py_ < 4 or px_ + lw > W - 4 or py_ + lh > H - 4:
                print(f'[p3-mini] skip(OOB) @({cx_t},{cy_t}) sc={sc}')
                continue
            al = np.clip(la[..., 3] / 255.0, 0.0, 1.0)
            cov = al > 0.30
            gsel = np.zeros((H, W), bool)
            gsel[py_:py_ + lh, px_:px_ + lw] = cov
            if (gsel & occ).any():
                print(f'[p3-mini] skip(occupy) @({cx_t},{cy_t}) sc={sc}')
                continue
            reg = np.asarray(res, np.float32).copy()
            sub = reg[py_:py_ + lh, px_:px_ + lw]
            reg[py_:py_ + lh, px_:px_ + lw] = sub * (1 - al[..., None]) + la[..., :3] * al[..., None]
            res = Image.fromarray(np.clip(reg, 0, 255).astype(np.uint8), 'RGB')
            n_mini += 1
    print(f'[pinterest3] mini-butterflies added={n_mini}')

    res.save(OUT / 'pinterest3_variant.jpg', quality=93)
    print(f'[pinterest3] letters={len(letters)} word={word!r} -> {OUT}')
    return res


# ---------------------------------------------------------------- b78e60
def do_b78e60():
    """军标迷彩文字：原字本身就是**实心黑粗字**，材质=纯黑，故原位重画黑字即同材质。
    v332: 狗牌+链条小元素裂变（🔴3 逐元素；用户点名"狗牌为什么没裂变"）。"""
    img0 = Image.open(BY['b78e60']['path'])
    excl = v328.region_mask(*img0.size,
                            pred=lambda xx, yy: (xx > 500) & (xx < 850) & (yy > 762) & (yy < 1120))
    # v334（用户："字体排版一个点是什么"）：v329~v333 传的 words 是**原词**——
    # 等于只擦重画、一个词都没换（STEEL/HAWKS 原样），用户一眼看穿。
    # 本版三行全换新词（军旅语境，长度近似，draw 自动适配原行框）。
    out = v328.text_variant(
        'b78e60',
        [dict(box=(360, 430, 1200, 790),
              words=['FORGED IN THE', 'DARK', 'STORM'], min_area=120)],
        'blackopsone', erase_kw=dict(method='nn', nn_median=41), dilate=5, exclude=excl)

    # v334: 清除 "DARK" 左侧历史暗斑（v332 起就有的原字残片，detect_lines 漏检，
    # 实测位于 (507-545, 567-607)。用户三轮截图里都在，本轮一并清掉。）
    bx_ = (490, 548, 562, 618)
    W0_, H0_ = img0.size
    aa_ = np.asarray(out, np.float32)[bx_[1]:bx_[3], bx_[0]:bx_[2]]
    dm_ = aa_.sum(2) < 330
    if dm_.any():
        mfull = np.zeros((H0_, W0_), bool)
        mfull[bx_[1]:bx_[3], bx_[0]:bx_[2]] = tf.ndi.binary_dilation(dm_, structure=tf._disk(5))
        out = tf.erase(out, mfull, method='nn', nn_median=41, margin=12)

    # ---- v335: 狗牌双牌装配体旋转 ----
    # ⚠️ 链条从文字间穿过：自动检测会把重画的黑字当组件 → 必须用**手工框**（牌 bbox+链走廊）。
    # v335 教训（用户："狗牌裂变就是删掉一个狗牌？"）：原图是**双牌叠挂**（前牌+后牌
    # 连成一个组件），v334 逐牌 ±0.42rad 大角度把组件撕散——后牌被 nn 抹除后未贴回，
    # 视觉上等于"删了一个牌"。且后牌较暗，阈值 45 只罩住前牌。
    # 本版：① 阈值 45→32 + closing disk5（后牌完整入组）；② 整组（链+双牌）作为
    # **单一装配体**绕链顶 -0.20rad + 侧摆 8px——双牌相对位置锁死一起动，
    # 两块牌都完整保留、姿态可见变化。
    from styles import subject_morph as smod
    W0, H0 = img0.size
    tag_box = (int(0.425 * W0), int(0.398 * H0), int(0.570 * W0), int(0.522 * H0))
    chain_box = (int(0.440 * W0), 792, int(0.545 * W0), 838)
    a0 = np.asarray(out, np.float32)
    group = np.zeros((H0, W0), bool)
    for (qx0, qy0, qx1, qy1) in (tag_box, chain_box):
        sub = a0[qy0:qy1, qx0:qx1]
        ring = np.concatenate([sub[:8].reshape(-1, 3), sub[-8:].reshape(-1, 3),
                               sub[:, :8].reshape(-1, 3), sub[:, -8:].reshape(-1, 3)])
        bgc = np.median(ring, 0)
        group[qy0:qy1, qx0:qx1] = np.abs(sub - bgc[None, None, :]).sum(2) > 32
    group = tf.ndi.binary_closing(group, structure=tf._disk(5))
    tlab, tn = tf.ndi.label(group, structure=np.ones((3, 3), bool))
    tags = []
    for i in range(1, tn + 1):
        mi = tlab == i
        if int(mi.sum()) < 25:
            group &= ~mi
            continue
        ys_, xs_ = np.where(mi)
        tags.append((float(xs_.mean()), float(ys_.min()),
                     0.5 * max(xs_.max() - xs_.min(), ys_.max() - ys_.min()) + 6))
    if group.any():
        vis = np.asarray(out).copy(); vis[group] = [255, 0, 0]
        Image.fromarray(vis).save(VIS / 'v332_b78_tags_mask.png')
        out = tf.erase(out, tf.ndi.binary_dilation(group, structure=tf._disk(3)),
                       method='nn', nn_median=41, margin=24)
        rgba, tbox = smod.make_layer(img0, group, feather=1.2, pad=14)
        bx0, by0 = tbox
        hh, ww = rgba.shape[:2]
        yyT, xxT = np.mgrid[0:hh, 0:ww].astype(np.float32)
        gxa, gya = xxT + bx0, yyT + by0
        ys_g, xs_g = np.where(group)
        py_top = float(ys_g.min())                 # 链条挂点
        px_top = float(xs_g[np.argmin(ys_g)])
        rngt = np.random.default_rng(77)
        dyy = np.zeros((hh, ww), np.float32)
        dxx = np.zeros((hh, ww), np.float32)

        def _add_rot(px, py, theta, w_lo, w_hi):
            nonlocal dyy, dxx
            dx0 = gxa - px
            dy0 = gya - py
            wgt = smod.smoothstep(np.hypot(dx0, dy0), w_lo, w_hi)
            ang = -theta * wgt
            ca_, sa_ = np.cos(ang), np.sin(ang)
            dyy += (dx0 * sa_ + dy0 * ca_) - dy0
            dxx += (dx0 * ca_ - dy0 * sa_) - dx0

        # v335：**装配体旋转**（禁逐牌独立角度——v334 撕散双牌的根因）。
        _add_rot(px_top, py_top, -0.20, 4.0, 120.0)                # 整组绕链顶摆 -11.5°
        dxx += 8.0 * smod.smoothstep(gya - py_top, 16.0, 150.0)    # 链牌整体侧摆
        wlay = smod.warp_layer(rgba, (dyy, dxx), order=1)
        out = smod.paste_layer(out, wlay, tbox)
        print(f'[b78e60] dogtags: comps={len(tags)} px={int(group.sum())}')

    out.save(OUT / 'b78e60_variant.jpg', quality=93)
    return out


# ---------------------------------------------------------------- pinterest4
def do_pinterest4():
    return cpp.fission(BY['pinterest4']['path'], OUT, cfg, seed=21)


# ---------------------------------------------------------------- 6978
def do_6978():
    """蝙蝠徽章：① 弧字/品牌字原位改写 ② 主体蝙蝠按新姿态重画（商标改写）。"""
    from styles import subject_morph as smod
    ic = BY['6978']
    img = Image.open(ic['path']).convert('RGB')
    W, H = img.size
    emblem = v328.region_mask(W, H, lambda xx, yy: (xx - 813) ** 2 + (yy - 690) ** 2 <= 268 ** 2)
    cx0, cy0 = 777.0, 728.0
    m_arc0, _, _, _, _ = v328.arc_letter_mask(img, (235, 245, 1345, 825), cx0, cy0,
                                              thr=80, emblem=emblem)
    cx, cy, r, cap_h, a0, a1 = v328.arc_metrics(m_arc0, cx0, cy0)
    m_arc, _, _, _, _ = v328.arc_letter_mask(img, (235, 245, 1345, 825), cx0, cy0,
                                             thr=80, emblem=emblem, radial=(r, cap_h))
    print(f'[6978] arc c=({cx:.0f},{cy:.0f}) r={r:.0f} capH={cap_h:.0f}')
    yy, xx = np.mgrid[0:H, 0:W]
    band = np.hypot(xx - cx, yy - cy)
    ribbon = (band > r - cap_h * 0.95) & (band < r + cap_h * 0.95)
    src_ribbon = ribbon & (~tf.ndi.binary_dilation(m_arc, structure=tf._disk(3)))
    specs = [
        dict(box=(285, 800, 640, 935), words=['Est.'], pad=10, min_area=120,
             draw_box=(398, 826, 548, 908)),
        dict(box=(985, 800, 1300, 935), words=['1868'], pad=10, min_area=120,
             draw_box=(1012, 826, 1160, 908)),
        dict(box=(275, 985, 1285, 1165), words=['NOCTAVEN'], pad=8, min_area=300),
        dict(box=(450, 1180, 1140, 1340), words=['MOONHEART'], pad=8, min_area=300),
    ]
    lines, brand = [], np.zeros((H, W), bool)
    excl_brand = emblem | tf.ndi.binary_dilation(m_arc, structure=tf._disk(8))
    for sp in specs:
        ls = tf.detect_lines(img, sp['box'], direction='dark', mode='lum', thr=80,
                             pad=sp['pad'], min_area=sp['min_area'], exclude=excl_brand)
        if ls:
            b, m = ls[0]
            lines.append((sp['words'][0], tuple(sp.get('draw_box') or b), m)); brand |= m
    for bb in [(1150, 930, 1300, 1010), (1120, 1170, 1200, 1235)]:
        brand |= v328.extra_specks(img, bb, exclude=excl_brand)

    # ---- 蝙蝠擦除掩膜（v336：暗阈值 58→75 + 椭圆扩大 252/262→300/305）----
    # v336 取证：左翼外膜 (522,520)-(646,885) 约 1 万像素 mx∈[58,75] 且在旧椭圆外
    # → 旧 mask 漏掉整条翼带（层法会断翼）。扩椭圆+放宽阈值后完整罩住，
    # 同时仍排除碟面紫（mx≥80）、粉底、弧形字（几何上不相交）。
    arr = np.asarray(img, np.float32)
    mx = arr.max(2)
    lum = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    ell = ((xx - 815) / 300.0) ** 2 + ((yy - 742) / 305.0) ** 2 <= 1.0
    core = (mx < 75) & ell
    # 浅色描边（lum>118 且在核心膨胀范围内 = 原图蝙蝠的 lavender 轮廓线）
    # v338e 取证（v338_warped_layer.jpg 目检）：膨胀半径 10 → **4**。
    # 10px 会把"碟缘浅紫亮带/缎带"整片吃进掩膜（凡与翼缘相距 <10px 的亮像素都被纳入），
    # 这些亮块随翼场一起被搬运 → 成品上翼缘外侧出现**淡紫涂抹**（用户会读成"脏/糊"）。
    # 蝙蝠自身轮廓线只有 2-4px 宽，disk(4) 足够罩住，多的全是背景。
    edge_rgn = (lum > 118) & tf.ndi.binary_dilation(core, structure=tf._disk(4))
    bat = (core | edge_rgn)
    lab, n = tf.ndi.label(bat, structure=np.ones((3, 3), bool))
    if n:
        sz = tf.ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        # v338：**剔除文字碎片**。取证 v330_6978_bat_mask.png：mx<75 & ell 会同时抓住
        # 弧字（LA CASA DEL MURCIELAGO，r≈342-396）与品牌字（BACARDÍ/MOONHEART/
        # Est./1868）的暗笔画 —— 这些碎片随蝙蝠层一起被位移场搬运，大振幅下会在粉底上
        # 拖出可见涂抹。判据：连通块与"已检出的文字掩膜(brand|m_arc)"重叠 >15% 面积
        # → 判为文字碎片丢弃；蝙蝠本体（~10 万 px）与文字几乎不重叠，稳过。
        _txt = brand | m_arc
        # v338b：**只保留"触及碟心"的连通块**（= 蝙蝠本体）。
        # 实测（v338 取证）：除蝙蝠(96252px, 质心角度172°, R∈[0.00,1.12])外，还有一块
        # 10027px 的**缎带/横幅残片**（质心 (1016,571)＝右上 330°、R∈[0.87,1.14]、
        # 与文字重叠仅 0.2% → 躲过了文字判据）。小位移时它待在原位看不出来，位移一放大
        # 就被推到粉底上成一条粗黑弧 = 用户会读成"脏痕/涂抹"。
        # 判据：蝙蝠躯干必然覆盖碟心（内含 r<0.5 的像素），缎带/环线/文字碎片都在外围。
        _r_all = np.sqrt(((xx - 813.0) / 272.0) ** 2 + ((yy - 690.0) / 277.0) ** 2)
        keep = np.zeros(n + 1, bool)
        for j in range(1, n + 1):
            aj = int(sz[j - 1])
            if aj < 800:
                continue
            mj = (lab == j)
            if int((mj & _txt).sum()) > 0.15 * aj:
                continue
            if float(_r_all[mj].min()) > 0.55:       # 够不到碟心 → 不是蝙蝠
                continue
            keep[j] = True
        bat = keep[lab]
    bat = tf.ndi.binary_dilation(bat, structure=tf._disk(3))
    vis = np.asarray(img).copy(); vis[bat] = [255, 0, 0]
    Image.fromarray(vis).save(VIS / 'v330_6978_bat_mask.png')

    out = img
    # 碟面内允许取样区（蝙蝠以外的碟面像素）
    yy0, xx0 = yy, xx
    for tag, msk, kw in (('arc', m_arc, dict(method='nn', nn_median=31, src_allow=src_ribbon)),
                         # 品牌字：hybrid = nn 保边界色 + LaMa 补低频结构。
                         # ⚠️ 不用 patch 平移填充：平移 120px 会**复制到同一行文字本身** →
                         #    新字上叠出旧字双影（实测 NOCTAVEN 上透出 BACARDÍ）。
                         ('brand', brand, dict(method='hybrid', nn_median=31))):
        if not msk.any():
            continue
        mdd = tf.ndi.binary_dilation(msk, structure=tf._disk(9))
        out = tf.erase(out, mdd, **kw)
    out.save(VIS / 'v329_6978_erased.jpg', quality=95)

    # ---- 主体蝙蝠裂变 v336：**双层法**（层内形变，背景零拖动）----
    # v336 第一次实现（全图 0.30rad 场）实测：碟面渐变被翼场拖出波状涂抹——
    # 用户第 3 轮就骂过"背景糊在一起"，大振幅下不可接受。改双层：
    #   ① 擦掉蝙蝠 → 碟面/粉底 nn 重建干净底板（碟面是平滑径向渐变，nn 无痕）；
    #   ② 蝙蝠单独 make_layer，在**层内坐标系**施加翼姿位移场（warp 的是带 alpha
    #      的蝙蝠层，层外 alpha=0）→ 碟面一个像素都不动；
    #   ③ 形变层贴回底板。翼尖上扬让出的空隙由底板自然补上 → 无涂抹无 ghost。
    # 姿态：双翼对称上扬 0.30rad（17°，翼尖各抬 ~105px，平展 T 形 → 明显 V 形）、
    # 歪头 -0.12rad；翼场避开躯干（v335b 中轴对拉剪切之鉴）。
    bat_f = tf.ndi.binary_fill_holes(bat)
    # v336b：底板重建**分区取样**——翼尖让空区跨碟缘两侧，单一 nn 会跨区拉色
    # （碟外区拉进碟面紫 → 粉底上出现紫色斜纹）。按碟面几何椭圆把擦除区分成
    # 碟内/碟外两片，各自只允许从同区像素取样（src_allow）。
    emask = tf.ndi.binary_dilation(bat, structure=tf._disk(4))
    # 椭圆取 272/277 ≈ 亮碟真实边缘（emblem 268 + 4px 余量）：305 实测会把 35px 的
    # 环带/粉底划进碟区 → nn 从碟面紫取样 → 碟缘拉出紫色斜纹。
    disc_r = ((xx - 813.0) / 272.0) ** 2 + ((yy - 690.0) / 277.0) ** 2 <= 1.0
    # v337：取样源 = 碟内全部真实像素（含粉缎带、碟面亮环），只排除"蝙蝠+8px 抗锯齿环"，
    # 让被翼形洞切断的缎带/色环能按最近邻同色续上。
    allow_d = disc_r & (~tf.ndi.binary_dilation(bat, structure=tf._disk(8)))
    allow_o = tf.ndi.binary_dilation((~disc_r) & (~bat), structure=tf._disk(2))
    # v338：nn 回填参数（大振幅腾空区一变大，"糊"会被放大成看得见的光晕）：
    #   nn_median 31→1（31 会把取样值抹平 → 腾空区一片发蒙的软斑）
    #   edge_smoothness 4→1（贴合边不再羽化 → 回填区与周围同硬边）
    #   margin 20→8（更贴边取样，色带/环线衔接更准）
    # ⚠️ 试过"镜像预填"（徽章近似左右对称）：实测镜像把**弧形字的暗笔画**搬到碟左上，
    #    反而添出黑糊带 → 已废弃。nn 逐区回填 + 锐化参数是正解。
    plate = tf.erase(out, emask & disc_r, method='nn', nn_median=1, margin=8,
                     edge_smoothness=1, src_allow=allow_d)
    plate = tf.erase(plate, emask & (~disc_r), method='nn', nn_median=1, margin=8,
                     edge_smoothness=1, src_allow=allow_o)
    # v337 取证（三次试错，勿重犯）：
    # ① 不带掩膜的整体高斯扩散 → 把原蝠亮紫描边平均进腾空区（实测 188,122,186）＝幽灵。
    # ② 掩膜内谐波扩散（140×σ3，mask 外钉死）→ 洞跨过"紫碟面/粉缎带"强边界，扩散把
    #    两色混成灰绿（实测 (960,520) 由 192,133,181 变 156,160,161）＝更大破绽。
    # ③ 正解：**分区最近邻 + 取样源含粉缎带**。此前 allow_d 用 lum<105 排除了粉色缎带，
    #    于是"翼形洞"在缎带上被填成紫色 → 头部右上那条直角接缝（v337i 目检）。
    #    改成只排除"蝙蝠本体 + 8px 抗锯齿环"，缎带/亮环/碟面原样可采样 → 洞被同色续上。
    # v338g：**层要留出外扩余量**。旧码 make_layer(out, bat_f) 直接用掩膜 bbox →
    # 整体放大 1.18 时翼尖被推到 bbox 之外、被 mode='constant' 切平（成品上翼尖是
    # "平的"，只能靠半透明渐隐遮丑 → 又软又糊）。实测外扩 ~max(SXL,SYL) → 给 70px padding。
    _ysb, _xsb = np.where(bat_f)
    _padb = 70
    _bx = (max(0, int(_xsb.min()) - _padb), max(0, int(_ysb.min()) - _padb),
           min(W, int(_xsb.max()) + 1 + _padb), min(H, int(_ysb.max()) + 1 + _padb))
    rgba_b, bbox_b = smod.make_layer(out, bat_f, feather=1.4, box=_bx)
    # v338 **假边清除**（大位移下必做）：实测层边框上存在 alpha 高达 0.8 的孤立像素
    # （掩膜边界/羽化的残留），warp 用 mode='nearest' 会把它沿位移方向复制成一条
    # 半透明涂抹（正是"蝙蝠一放大，碟子上方就出现一缕灰紫脏痕"的来源）。
    # 对策：① 把层最外 3px 的 alpha 钉成 0；② warp 用 mode='constant'（越界=透明）。
    _pad3 = 3
    rgba_b[:_pad3, :, 3] = 0
    rgba_b[-_pad3:, :, 3] = 0
    rgba_b[:, :_pad3, 3] = 0
    rgba_b[:, -_pad3:, 3] = 0
    hhb, wwb = rgba_b.shape[:2]
    yyL, xxL = np.mgrid[0:hhb, 0:wwb].astype(np.float32)
    gxL, gyL = xxL + bbox_b[0], yyL + bbox_b[1]

    def _rotB(dyy, dxx, w, px, py, theta):
        ang = -theta * w
        ca, sa = np.cos(ang), np.sin(ang)
        dx0 = gxL - px
        dy0 = gyL - py
        return dyy + (dx0 * sa + dy0 * ca) - dy0, dxx + (dx0 * ca - dy0 * sa) - dx0

    bat_c = bat_f[bbox_b[1]:bbox_b[1] + hhb, bbox_b[0]:bbox_b[0] + wwb]
    # ============ v338a 主体裂变**大幅加大**（用户第 9 轮）============
    # 用户原话："蝙蝠裂变大点，跟原图区别一直分不开吗"。
    # 第 8 轮的 (1.06x / 0.28rad / 收缩0.92 / 抬头19px / 尾1.02) 幅度太小：
    # 剪影 bbox 仅变 ±6px、翼角变化 16° —— 摆在紫色碟上远看仍是"同一只蝠"。
    # v338 四维度全部拉到**一眼可辨**：
    #  ① 整体放大 1.18（身体/头/翼膜/描边等比，中央质量显著变大）
    #  ② 双翼绕肩上扬 0.34rad（19.5°）+ ③ 翼展收缩 0.84（翅收拢贴身，剪影由
    #     "广展 W 形"变"高耸 V 形"）—— 收缩把翼尖往里拉，正好抵消放大外扩，
    #     实测剪影最大径向 1.153→1.20（碟外干净，不碰 r≈342 的弧形字内缘）
    #  ④ 抬头：上移 32px + 头放大 1.12（头颈上引、"正面抬头"）
    #  ⑤ 尾：纵向拉长 1.16 + 侧摆 26px 成弧（原尾短而直）
    # 符号铁律：_rotB(theta) 等价内容旋转 φ=-theta → 左翼 theta<0、右翼 theta>0 才是上扬。
    CXB, CYB = 776.0, 720.0
    # v338h 起改用**仿射场**（横向收拢 + 纵向拉高），比"整体等比放大"强得多：
    #   · 整体等比放大 1.16 会把翼尖沿径向推出碟外（实测最远点 (478,767) R=1.263，
    #     直接插进弧形字带 R≈1.26；且翼尖被 bbox 切平 → 只能靠渐隐遮丑 = 又软又糊）。
    #   · 换成横向收拢 sx<1 + 纵向拉高 sy>1 的**仿射**场：单调、雅可比处处同号 →
    #     **零空洞、零软边**；翼尖往里收（R 1.12→0.95，稳在碟缘内），头往上抬、
    #     尾往下拉 → "身体长高、翅膀收拢上扬" 一眼可辨，且**不碰碟缘与弧形字**。
    # 参数支持环境变量覆盖（调参/回归用）。
    SXL = float(os.environ.get('BAT_SXL', 1.08))       # 横向放大（绕 x=776）
    SYL = float(os.environ.get('BAT_SYL', 1.20))       # 纵向放大（绕 y=790）
    _pyv = float(os.environ.get('BAT_PYV', 790.0))     # 纵向放大枢轴（低枢轴=头抬多尾伸少）
    B_HL = float(os.environ.get('BAT_HL', 12.0))       # 头部额外上移（px）
    B_HSC = float(os.environ.get('BAT_HSC', 1.08))     # 头部额外放大
    B_TSW = float(os.environ.get('BAT_TSW', 14.0))     # 尾侧摆（px）
    B_ASY = float(os.environ.get('BAT_ASY', 1.06))     # 右翼额外径向扩张（不对称）

    def _rotB(dyy, dxx, w, px, py, theta):
        ang = -theta * w
        ca, sa = np.cos(ang), np.sin(ang)
        dx0 = gxL - px
        dy0 = gyL - py
        return dyy + (dx0 * sa + dy0 * ca) - dy0, dxx + (dx0 * ca - dy0 * sa) - dx0

    # ① 纯仿射扩张（绕蝠体内部一点 (776,790) 放大 sxl×syl，两者都 >1）
    #    —— **数学上零空洞**：仿射双射 + 两轴都放大 = 新蝠完全盖住旧蝠。
    #    纵向放大让头向上抬（枢轴取低 y=790 → 头抬得多、尾伸得少，避开 BACARDÍ 字），
    #    横向放大让整体变宽变壮。翼尖在水平直径附近 → 纵向放大对 R 影响很小
    #    （实测 maxR_top 仅 1.12→1.15，稳在弧形字内缘 1.26 之内）。
    _cx, _cy = 813.0, 690.0
    dxx = (1.0 / SXL - 1.0) * (gxL - CXB)
    dyy = (1.0 / SYL - 1.0) * (gyL - _pyv)

    # ②b 右翼外段**额外径向扩张**（唯一允许的"不对称"手段：扩张型形变不腾空；
    #     收缩/旋转都会让出碟缘亮环 → 底板修不回来）。左翼不动、右翼外扩 →
    #     剪影左右不等 = 斜掠/侧倾姿态，与用户的"翅膀张开收缩"呼应。
    _wR = np.clip(tf.ndi.gaussian_filter(
        tf.ndi.binary_dilation(bat_c & (gxL > 860), structure=tf._disk(6)).astype(np.float32),
        16.0), 0.0, 1.0)
    dxx = dxx + _wR * (1.0 / B_ASY - 1.0) * (gxL - _cx)
    dyy = dyy + _wR * (1.0 / B_ASY - 1.0) * (gyL - _cy)

    # ② 双翼绕肩**换姿态**（v339，用户第 10 轮："主体蝙蝠你到底裂不裂变了"）
    # 纯仿射扩张只把蝠"拉高变壮"，轮廓形状没变 → 用户仍读成"没裂变"。
    # 这里加真正的姿态变化：左右翼绕各自肩点旋转（**不等角 = 斜掠**）。
    # 腾空区落在**碟内平滑紫**（见 v338_mask_on_plate 取证：底板蝠位是均匀紫），
    # nn 回填无痕；权重带 (1-ylow) 冻结翼底 y>600 以下，避免下翼被拖成滴痕。
    # 符号铁律：_rotB(theta) 等价内容旋转 φ=-theta → 左翼 theta<0、右翼 theta>0 = 上扬。
    B_ROTL = float(os.environ.get('BAT_ROTL', 0.20))
    B_ROTR = float(os.environ.get('BAT_ROTR', 0.32))
    ylow = np.clip((gyL - 600.0) / 300.0, 0.0, 1.0)
    ylow = ylow * ylow * (3.0 - 2.0 * ylow)
    wWL = np.clip(tf.ndi.gaussian_filter(
        tf.ndi.binary_dilation(bat_c & (gxL < 735), structure=tf._disk(6)).astype(np.float32),
        14.0), 0.0, 1.0) * (1.0 - ylow)
    wWR = np.clip(tf.ndi.gaussian_filter(
        tf.ndi.binary_dilation(bat_c & (gxL > 817), structure=tf._disk(6)).astype(np.float32),
        14.0), 0.0, 1.0) * (1.0 - ylow)
    dyy, dxx = _rotB(dyy, dxx, wWL, 700.0, 600.0, -B_ROTL)
    dyy, dxx = _rotB(dyy, dxx, wWR, 852.0, 600.0, B_ROTR)

    # ③ 抬头 + 头放大（头在碟内上部 r≈0.55-0.75，腾空区是**平滑紫**，nn 回填无痕）
    head_core = bat_c & (np.abs(gxL - 767.0) < 115.0) & (gyL < 620.0)
    wH = np.clip(tf.ndi.gaussian_filter(head_core.astype(np.float32), 18.0), 0.0, 1.0)
    dyy = dyy * (1.0 - wH) + wH * ((1.0 / SYL - 1.0) * (gyL - _pyv) - B_HL
                                   + (1.0 / B_HSC - 1.0) * (gyL - 620.0))
    dxx = dxx * (1.0 - wH) + wH * ((1.0 / SXL - 1.0) * (gxL - CXB)
                                   + (1.0 / B_HSC - 1.0) * (gxL - 767.0))

    # ③ 尾：侧摆成弧（尾在碟外粉底，腾空区是纯粉 → nn 无痕）
    tail_core = bat_c & (np.abs(gxL - 770.0) < 30.0) & (gyL > 865.0)
    wT = np.clip(tf.ndi.gaussian_filter(tail_core.astype(np.float32), 6.0), 0.0, 1.0)
    tcur = np.clip((gyL - 880.0) / 110.0, 0.0, 1.0)
    tcur = tcur * tcur * (3.0 - 2.0 * tcur)
    dxx = dxx + wT * B_TSW * tcur

    # ---- v338e 径向包容（**必须作用在"输出像素"上**）----
    # ⚠️ 踩坑记录：先试过对**位移/目标点**半径做 tanh 压缩 → 完全反效果。位移压缩是
    # "把远处的输出像素映射回蝠内"，于是 R1 以外每个输出像素都取到蝠边界像素、
    # alpha>0.5 → 形变剪影**反而膨胀**（实测 maxR 1.275→1.425，翼尖糊在粉底上）。
    # 正解：位移照常（保持形变的自然形状），只对**形变层的 alpha 在输出坐标上**做
    # 径向收边：r<1.00 全保留，1.00→1.13 平滑降到 0（约 35px 圆润收口）。
    # 收边之外露出的是底板（碟面/碟缘 nn 重建，干净）→ 翼尖"贴着碟缘收进去"，
    # 既不插进弧形字带（内缘 r≈342 → R1.26），也不产生涂抹。
    _RX, _RY = 272.0, 277.0
    _ra = np.sqrt(((gxL - 813.0) / _RX) ** 2 + ((gyL - 690.0) / _RY) ** 2)
    # v338f：收口带 1.13-1.00（35px）太宽 → 整个翼尖（翼厚仅 60-100px）被压成半透明
    # → 成品上蝠外一圈"暗晕"（用户读成脏）。改**窄收口** 1.09→1.15（16px），
    # 且优先靠**降幅**让自然剪影就落在碟缘附近，收口只做最后几像素的兜底。
    # v338i：仿射扩张路线下自然剪影 maxR=1.199 / maxR_top=1.165（均在弧形字内缘 1.26
    # 之内）→ 收口带挪到 1.22-1.30，**平时不触发**，只作"万一参数被调大"的兜底。
    _R_F0 = float(os.environ.get('BAT_RF0', 1.22))
    _R_F1 = float(os.environ.get('BAT_RF1', 1.30))
    _ka = np.clip((_R_F1 - _ra) / max(1e-6, _R_F1 - _R_F0), 0.0, 1.0)
    _ka = _ka * _ka * (3.0 - 2.0 * _ka)

    wlay = smod.warp_layer(rgba_b, (dyy, dxx), order=1, mode='constant')
    # ---- v338f 硬剪影化（**断掉暗晕的唯一正解**）----
    # 根因（v338e 目检 v338_warped_layer.jpg）：平滑位移场在翼缘/翼尖处梯度很大，
    # 把图层原本 1.4px 的软边**拉成 20-40px 的半透明宽带** → 贴回底板后蝠外一圈
    # 灰紫"暗晕"（用户会读成脏/糊）。任何"窄化收口"都治不了它（它是形变自带的）。
    # 做法：alpha 阈值化(>0.5→1) 再补 0.8σ 高斯 → 边缘回到 1-2px 抗锯齿。
    # 内部本来就是不透明（掩膜层），所以内容零损失；拉宽的软边被直接切掉。
    _al = wlay[..., 3].astype(np.float32) / 255.0
    _al = (_al > 0.5).astype(np.float32)
    _al = tf.ndi.gaussian_filter(_al, 0.8)
    # v338e 径向收边（见上）+ v336 徽章外圈保底钳制（半径 400，只管粉底）
    _core_pre = _al > 0.5                     # 收口**前**的自然剪影（调参取证用）
    wlay[..., 3] = (np.clip(_al * _ka, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    # 徽章外圈保底钳制（半径 400，只管粉底）。**不用 v336 的 272/277**：那会把放大
    # 后的翼尖与尾尖在碟缘处切掉（实测尾尖 y960-977 只剩 45% alpha → 成品缺一截）。
    v_badge = np.sqrt(((gxL - 813.0) / 400.0) ** 2 + ((gyL - 690.0) / 408.0) ** 2)
    kb = np.clip((1.0 - v_badge) / 0.03, 0.0, 1.0)
    kb = kb * kb * (3.0 - 2.0 * kb)
    wlay[..., 3] = wlay[..., 3] * kb
    out = smod.paste_layer(plate, wlay, bbox_b)
    # v337：**轮廓重画**。原蝠的亮紫描边是 3-5px 细线，形变会把细线抹成条带
    # （腾空区实测溜进 188,122,186 亮紫 → "幽灵边"）。用形变后 alpha 的轮廓重画
    # 一圈均匀描边（外描边 4px + 1.2σ 羽化），既盖住抹开的旧描边，又让新蝠有
    # 与原图一致的亮紫轮廓。
    _al = (wlay[..., 3] / 255.0)
    _core = _al > 0.5
    _core_full = np.zeros((H, W), bool)
    _core_full[bbox_b[1]:bbox_b[1] + _core.shape[0], bbox_b[0]:bbox_b[0] + _core.shape[1]] = _core

    if os.environ.get('BAT_DBG'):
        plate.save(VIS / 'v338_plate.jpg', quality=95)
        _w = np.asarray(wlay, np.float32)
        _bgw = np.full((_w.shape[0], _w.shape[1], 3), 235.0, np.float32)
        _al = (_w[..., 3:4] / 255.0)
        _comp = _w[..., :3] * _al + _bgw * (1 - _al)
        Image.fromarray(np.clip(_comp, 0, 255).astype(np.uint8), 'RGB').save(
            VIS / 'v338_warped_layer.jpg', quality=95)
        _mv = np.asarray(plate).copy()
        _mv[_core_full] = [255, 0, 0]
        Image.fromarray(_mv).save(VIS / 'v338_mask_on_plate.jpg', quality=95)
        del _w, _bgw, _al, _comp, _mv


    # ---- v338 几何取证：剪影径向范围（碟心 813,690；碟半径 268/272；弧字内缘更大）----
    if os.environ.get('BAT_GEO'):
        _pre_full = np.zeros((H, W), bool)
        _pre_full[bbox_b[1]:bbox_b[1] + _core_pre.shape[0],
                  bbox_b[0]:bbox_b[0] + _core_pre.shape[1]] = _core_pre
        for tag, mb in (('orig', bat_f), ('warp', _core_full), ('warpRaw', _pre_full)):
            ys_, xs_ = np.where(mb)
            if len(ys_) == 0:
                print(f'[bat-geo] {tag} EMPTY'); continue
            vv = np.sqrt(((xs_ - 813.0) / 272.0) ** 2 + ((ys_ - 690.0) / 277.0) ** 2)
            vv_d = np.sqrt(((xs_ - 813.0) / 268.0) ** 2 + ((ys_ - 690.0) / 268.0) ** 2)
            _im = int(vv.argmax())
            print(f'[bat-geo] {tag} bbox=({xs_.min()},{ys_.min()})-({xs_.max()},{ys_.max()}) '
                  f'area={len(ys_)} maxR={vv.max():.3f} outside268={(vv_d>1).mean():.3f} '
                  f'maxR_top(y<690)={vv[ys_ < 690].max() if (ys_ < 690).any() else 0:.3f} '
                  f'n_gt_1.20={int((vv > 1.20).sum())} argmax_at=({xs_[_im]},{ys_[_im]})')
        # **腾空区**：原蝠占据、新蝠不再覆盖的区域 → 底板必须补出来（nn 只能补平滑渐变，
        # 补不了碟缘黑环/缎带）。vac_out 越大越会在碟缘留下光晕/涂抹 → 用它卡死形变上限。
        _vac = bat_f & (~_core_full)
        _vy, _vx = np.where(_vac)
        _vr = np.sqrt(((_vx - 813.0) / 272.0) ** 2 + ((_vy - 690.0) / 277.0) ** 2)
        print(f'[bat-geo] vacated={int(_vac.sum())} '
              f'({100.0 * _vac.sum() / max(1, int(bat_f.sum())):.1f}% of orig)  '
              f'disc里(r<.88)={int((_vr < 0.88).sum())}  碟缘带(.88-1.10)={int(((_vr >= 0.88) & (_vr < 1.10)).sum())}  '
              f'碟外(>1.10)={int((_vr >= 1.10).sum())}')

    # 实测（沿剪影法向 1-3/3-6/6-10/10-16px 的中位色）：
    #   原图  139,72,135 / 132,57,132 /  98,23,128 / 96,23,127
    #   v337a 124,62,126 / 133,62,135 / 132,62,137 / 102,31,128  ← 6-10px 处偏亮 34
    # 说明"外描边 4px"过宽（原图亮边只 1-6px），且与"排除原蝠 4px"的交集让环断续
    # → 头部出现双道辉光轮廓。v337f 收敛为 **2px + 0.9σ + 0.72 不透明度**的连续细边。
    if int(P6978_RING_W) <= 0:
        _ringf = np.zeros((H, W), np.float32)
    else:
        _ring = tf.ndi.binary_dilation(_core_full, structure=tf._disk(int(P6978_RING_W))) & (~_core_full)
        _ringf = np.clip(tf.ndi.gaussian_filter(_ring.astype(np.float32), 0.9), 0.0, 1.0) * 0.72
    _oa = np.asarray(out, np.float32)
    _edge_col = np.array([150.0, 80.0, 148.0], np.float32)
    _oa = _oa * (1 - _ringf[..., None]) + _edge_col[None, None, :] * _ringf[..., None]
    out = Image.fromarray(np.clip(_oa, 0, 255).astype(np.uint8), 'RGB')
    wvis7 = np.clip(np.stack([wH, bat_c * 0.3, wT], -1) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(wvis7, 'RGB').save(VIS / 'v334_6978_weights.jpg', quality=90)

    # ---- 重画：文字（原字为实心黑 Didone serif，材质=纯黑）----
    for (w, b, m) in lines:
        out = tf.draw_line(out, w, b, 'playfair_black', tf.text_color(img, m), fit='squeeze')
    # v336（用户："弧形文本乱飞"）：v335 的 r*1.22 压平弧让新字**压在未擦除的
    # 缎带装饰弧线上**（弧线属装饰不换）→ 字母跨线、翻转乱飞（第 7 轮批评点）。
    # 本版回到**原弧半径/原字高**（字母永远待在带内，零碰撞），排版裂变改为：
    # 字距放宽（2.5px）+ 整体角位偏移 +4° + 新词本身长短差 → 新词在带内的
    # 起止角/占位与原词明显不同，连贯自然。
    import arc_text as _at
    fp_a = tf.FONTS.get('playfair_black', tf.FONTS['playfair'])
    S_a = max(10, int(cap_h / tf._cap_ratio(fp_a)))
    avail_a = r * math.radians(((a1 + 6.0) - (a0 - 6.0)) % 360)
    while S_a > 10:
        if _at.fit_arc_text_width("LA LUNA NELL'OMBRA", str(fp_a), S_a, r,
                                  char_spacing_px=2) <= avail_a * 0.97:
            break
        S_a = int(S_a * 0.94)
    span_new = math.degrees(
        _at.fit_arc_text_width("LA LUNA NELL'OMBRA", str(fp_a), S_a, r,
                               char_spacing_px=2) / r) + 2.0
    start_new = 0.5 * (a0 + a1) - 0.5 * span_new + 4.0
    out = tf.draw_arc(out, "LA LUNA NELL'OMBRA", (cx, cy), r, cap_h * 1.02,
                      'playfair_black', tf.text_color(img, m_arc),
                      start_deg=start_new, end_deg=start_new + span_new,
                      char_spacing_px=2)
    out.save(OUT / '6978_variant.jpg', quality=93)
    print(f'[6978] lines={len(lines)} arc r={r:.0f}(orig) cap*1.02 shift+4deg spacing=2 '
          f'bat=仿射扩张x{SXL:.2f}/y{SYL:.2f}+双翼绕肩换姿L{B_ROTL:.2f}/R{B_ROTR:.2f}(不等=斜掠)'
          f'+head-{B_HL:.0f}x{B_HSC:.2f}+tailSway{B_TSW:.0f}px')
    return out


# ---------------------------------------------------------------- pinterest6
# v340：p6 主体裂变振幅（可用环境变量覆盖，便于探针扫参）
# 第 10 轮用户："主体老鹰和骷髅头我一直说要裂变就是没有变" —— 旧版翼角仅 0.16/0.13rad
# （翼尖位移 ~110px），且**双角完全没动**，在自相似插画上肉眼读不出来。
P6_WL = float(os.environ.get('P6_WL', 0.30))     # 左翼绕肩上扬（rad，>0=顺时针=抬）
P6_WR = float(os.environ.get('P6_WR', 0.22))     # 右翼（不等角 → 斜掠）
P6_SK = float(os.environ.get('P6_SK', 0.13))     # 骷髅倾斜
P6_HORN = float(os.environ.get('P6_HORN', 0.20))  # 双角外掀（绕角根）
P6_JAW = float(os.environ.get('P6_JAW', 85.0))   # 下颌下沉（张嘴）px


def do_pinterest6(WORD_SRC='n5_0.png', pad=55, tgt=1200.0, cy_place=60):
    """黑金属尖刺标题：擦掉旧标题（保留背景蓝烟），换上同字身高的尖刺字标。

    ⚠️ 擦除改法（v329 首版漏擦）：旧版用 detect_lines 只取**最大连通域**，且扫描窗口
    限到 y<1140 —— 原标题"翼状尖刺"左右两端一直伸到画框边缘、末端低于 y=1140，
    于是两端白色残片漏擦，画面留白斑。
    本版：只标**白色墨线**（亮 + 低饱和）→ 闭运算(disk 12)把整排尖刺合成一块
    → 取"触顶"的最大连通域 → 填洞 + 膨胀(disk 10)吃掉描边 → 常数黑填充。
    只擦白墨、不擦蓝烟：闭运算会把稀疏蓝雾排除，背景烟雾得以保留。
    重画：Harrlogos XL 生成的尖刺字标，裁掉过长"发丝"尖刺后按**原图标题字身高度**
    缩放居中 —— 同风格 / 同材质(白墨) / 同字体语言。
    """
    ic = BY['pinterest6']
    img = Image.open(ic['path']).convert('RGB')
    W, H = img.size

    # ---- v336 先行：标题带内**蓝烟全部擦黑**（全分辨率 + const 黑填充）----
    # v335 教训：蓝烟判据带 lum>40（DS4 下采样）→ 暗蓝(lum≈32)大量漏网；nn 填充又
    # 从残留蓝烟取样回填蓝色；最后"色域软压制"把残蓝变灰雾 → 用户看到的就是
    # "标题后面一大块灰色色块"（第 7 轮批评点）。原图黑底实测纯黑 (0,0,0,
    # p90≤8)，蓝烟擦除改**全分辨率**判据 (b>r+6 & b>22) + **const 黑填充**：
    # 填充结果=背景本色，无雾无块。
    a_full = np.asarray(img, np.float32)
    blue = (a_full[..., 2] > a_full[..., 0] + 6.0) & (a_full[..., 2] > 22.0)
    blue[1500:, :] = False                       # 翼顶 ~1560 之下不动
    blue = tf.ndi.binary_dilation(blue, structure=tf._disk(4))
    visb = np.asarray(img).copy(); visb[blue] = [255, 0, 0]
    Image.fromarray(visb).save(VIS / 'v336_p6_blue_mask.png')
    out = tf.erase(img, blue, method='const', const_color=(2, 2, 3), const_feather=10)
    out.save(VIS / 'v336_p6_smoke_erased.jpg', quality=95)

    # ---- 擦除掩膜（v332 重写：笔画级 + 标题带限定 + LaMa 小掩膜重建）----
    # v329~v331 崩坏根因：closing(disk 12)+fill_holes 把"字母间隙背景/白色射线/翼尖"
    # 全吞成一个 3543x1500 巨块 → LaMa 在超大掩膜上幻觉出金字塔梯形；
    # v331 又把 lum>160 的全部白色（骷髅+射线）当主体 warp → 大面积涂抹（用户暴怒点）。
    # v332：① 掩膜只取标题带（y<1500；鹰翼顶 ≈1560）内"触顶"的亮笔画；
    #       ② closing(disk 5) 只连笔画本身，**不做 fill_holes**（保留字母间隙原背景）；
    #       ③ 填充 = Big-LaMa 小掩膜结构重建（背景烟雾/射线全部原样保留）。
    DS = 4
    sm = img.resize((W // DS, H // DS), Image.LANCZOS)
    a = np.asarray(sm, np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = a.max(2) - a.min(2)
    band_bot = int(1500 // DS)                   # 标题带下缘（翼顶 ~1560，全分辨率 px）
    ink = (lum > 170) & (sat < 50)
    ink[band_bot:, :] = False
    cl = tf.ndi.binary_closing(ink, structure=tf._disk(5))
    lab, n = tf.ndi.label(cl)
    best = np.zeros_like(cl)
    for i in range(1, n + 1):
        m = lab == i
        ys_ = np.where(m.any(1))[0]
        # 触顶（标题从图顶 y≈137 开始；翼尖从 y≈1560=390DS 才出现，被 band 挡住）
        if len(ys_) and ys_.min() < 90 and int(m.sum()) >= 40:
            best |= m
    best = tf.ndi.binary_dilation(best, structure=tf._disk(3))
    mask = np.asarray(Image.fromarray((best * 255).astype(np.uint8), 'L')
                      .resize((W, H), Image.NEAREST)) > 128
    vis = np.asarray(img).copy(); vis[mask] = [255, 0, 0]
    Image.fromarray(vis).save(VIS / 'v332_p6_mask.png')
    out = tf.erase(out, mask, method='lama', max_side=1100, margin=40)
    out.save(VIS / 'v332_p6_erased.jpg', quality=95)

    # ---- v335 鹰+骷髅主体裂变：**大振幅连通位移场**（v334 刚体裁层的最终教训）----
    # v334 失败根因：_part() 的 LCC 提取碎片化（翼只罩住上羽扇、骷髅只有颅盖），
    # 块匹配实测右翼尖位移 (0,0)、下颌 (36,0) —— mask 外的部件根本没动。
    # v333 平滑场失败根因则是 70px 振幅在自对称插画上自相似，而非机制本身。
    # 本版：平滑场 + 振幅×3 —— 左翼 +0.16rad / 右翼 -0.13rad（不对称上扬，
    # 翼尖位移 ~230/190px）、骷髅 -0.07rad + 下颌下沉 55px。
    # 无 mask 裁层、无擦除、无贴回接缝；权重=部件区域重闭运算+fill_holes+高斯 36，
    # 烟/射线在 falloff 内随场平滑流动（"背景一起优化"）。
    from styles import subject_morph as smod
    a6 = np.asarray(out, np.float32)
    a6i = np.asarray(img, np.float32)
    sat6 = a6i.max(2) - a6i.min(2)
    lum6 = a6i @ np.array([0.299, 0.587, 0.114], np.float32)
    brown6 = (sat6 > 40) & (a6i[..., 0] > a6i[..., 2] + 20)
    white6 = (lum6 > 150) & (sat6 < 60)
    H6, W6 = a6.shape[:2]
    yy6, xx6 = np.mgrid[0:H6, 0:W6].astype(np.float32)

    def _wregion(m, close_r=21, frac=0.12, biggest_only=False):
        m = tf.ndi.binary_dilation(m, structure=tf._disk(7))
        m = tf.ndi.binary_closing(m, structure=tf._disk(close_r))
        lab, n = tf.ndi.label(m, structure=np.ones((3, 3), bool))
        if n:
            sz = tf.ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            if biggest_only:
                keep[int(np.argmax(sz)) + 1] = True       # v340：只取最大组件（整只角）
            else:
                keep[1:] = sz >= sz.max() * frac          # union 大组件（碎片全收，防 v334 碎片化）
            m = keep[lab]
        m = tf.ndi.binary_fill_holes(m)
        w = tf.ndi.gaussian_filter(m.astype(np.float32), 36.0)
        return w / max(float(w.max()), 1e-6)              # 归一化峰值=1，部件内全场饱和

    wL = _wregion(brown6 & (xx6 < 1450) & (yy6 > 380) & (yy6 < 2380))
    wR = _wregion(brown6 & (xx6 > 2360) & (yy6 > 380) & (yy6 < 2380))
    wSK = _wregion(white6 & (((xx6 - 1850.0) / 620.0) ** 2
                             + ((yy6 - 3050.0) / 800.0) ** 2 <= 1.0), close_r=17, frac=0.20)
    # v340：**整只角**（旧版完全没动 → 用户读成"主体没裂变"）。
    # 角是可分离的大组件（实测左角 sz=306422 bbox x[347,1219] y[2188,3433]；右角 sz=247690
    # x[2313,3165] y[2201,3231]），故用 biggest_only 取整只，绝不用 y 硬切
    # （v340a 用 y>2450 切 → 角尖被截断、角变细，肉眼可见缺陷）。
    wHL = _wregion(brown6 & (xx6 < 1500) & (yy6 > 1950) & (yy6 < 3600),
                   close_r=21, biggest_only=True)
    wHR = _wregion(brown6 & (xx6 > 2200) & (yy6 > 1950) & (yy6 < 3600),
                   close_r=21, biggest_only=True)

    def _rot6(dyy, dxx, w, px, py, theta):
        ang = -theta * w                              # 采样端 -θ ⇔ 显示端 +θ（θ>0 顺时针）
        ca, sa = np.cos(ang), np.sin(ang)
        dx0 = xx6 - px
        dy0 = yy6 - py
        return dyy + (dx0 * sa + dy0 * ca) - dy0, dxx + (dx0 * ca - dy0 * sa) - dx0

    def _rramp(w, px, py, r0, r1):
        """v340：径向斜坡 —— 让位移在**部件与邻接内容（躯干/爪/颅）的交界处 → 0**。
        根因：区域权重只有轮廓处才有 0→1 的过渡带（σ36≈100px），0.3rad 的旋转要在
        这 100px 里吸收 r·θ≈180px 的位移 → 局部 2.8× 拉伸。带的背景是黑/射线时看不出来，
        压在**棕色躯干**上就是“翼根一团黑斑”（第 10 轮实测 y2250 x1250 处）。
        乘一层径向斜坡后：交界处 θ_eff→0（无拉伸），翼尖保持满幅 → 形变仍肉眼可辨。"""
        rr = np.sqrt((xx6 - px) ** 2 + (yy6 - py) ** 2)
        t = np.clip((rr - float(r0)) / max(1e-6, float(r1) - float(r0)), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        return w * t

    dyy = np.zeros((H6, W6), np.float32)
    dxx = np.zeros((H6, W6), np.float32)
    # 翼：r0=480 起才动（翼根与躯干交界 0 位移 → 无黑斑），r1=1350 到满幅（翼尖位移 ~330px）
    dyy, dxx = _rot6(dyy, dxx, _rramp(wL, 1450.0, 1700.0, 480.0, 1350.0), 1450.0, 1700.0, P6_WL)
    dyy, dxx = _rot6(dyy, dxx, _rramp(wR, 2400.0, 1700.0, 480.0, 1350.0), 2400.0, 1700.0, -P6_WR)
    dyy, dxx = _rot6(dyy, dxx, _rramp(wSK, 1850.0, 2950.0, 350.0, 1000.0), 1850.0, 2950.0, -P6_SK)
    dyy, dxx = _rot6(dyy, dxx, _rramp(wHL, 1180.0, 3000.0, 180.0, 1150.0), 1180.0, 3000.0, -P6_HORN)
    dyy, dxx = _rot6(dyy, dxx, _rramp(wHR, 2520.0, 3000.0, 180.0, 1150.0), 2520.0, 3000.0, P6_HORN)
    # 下颌下沉（张嘴）：y 3300..3750 带内、骷髅权重内，平滑过渡
    jw = smod.smoothstep(yy6, 3260.0, 3450.0) * (1.0 - smod.smoothstep(yy6, 3720.0, 3920.0))
    dyy = dyy - P6_JAW * jw * wSK
    _mag = np.sqrt(dyy * dyy + dxx * dxx)
    print(f'[p6-subj] maxdisp={_mag.max():.0f}px p99={np.percentile(_mag, 99):.0f}px '
          f'WL={P6_WL} WR={P6_WR} SK={P6_SK} HORN={P6_HORN} JAW={P6_JAW:.0f} '
          f'hornLpx={int(wHL.sum())} hornRpx={int(wHR.sum())}')
    coords6 = [yy6 + dyy, xx6 + dxx]
    o6 = np.empty_like(a6)
    for c in range(3):
        o6[..., c] = tf.ndi.map_coordinates(a6[..., c], coords6, order=1,
                                            mode='nearest', prefilter=False)
    out = Image.fromarray(np.clip(o6, 0, 255).astype(np.uint8), 'RGB')
    wvis = np.clip(np.stack([wL, wSK, wR], -1) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(wvis, 'RGB').save(VIS / 'v335_p6_weights.jpg', quality=90)

    # ---- 字标贴回（v340：对齐原图标题排版框）----
    # 第 10 轮用户批评："新的文本跟原图文本的排版位置大小不符合"。
    # 旧版只用「字身高」定标（tgt=1200）→ 新标只占全宽 70%、且顶到 y=60，
    # 左右各留一条黑带，视觉上比原标题又小又高。实测原图 MRCHOSR 白墨：
    #   bbox y[88,1800] h=1713、x[12,3524] w=3513（见 jobs/_probe/p6_geo2.py）
    # 字标 slab 的宽高比 2.063 与原图 2.052 几乎相等 → 纯等比：
    # 只对齐**横向满幅**（宽=3513）再让 ink 顶落到 y=88，两个 bbox 即重合。
    TGT_W = 3513.0          # 原图标题白墨全幅宽
    TOP_Y = 88.0            # 原图标题白墨顶
    G = VIS / 'elemgen_mid' / WORD_SRC
    if G.exists():
        g = Image.open(G).convert('L')
        ga = np.asarray(g, np.float32) / 255.0
        ga = np.clip((ga - 0.42) / 0.30, 0.0, 1.0)          # 只取白色笔画
        inkm = ga > 0.35
        ys, xs = np.where(inkm)
        by0, by1 = int(ys.min()), int(ys.max()) + 1
        bx0, bx1 = int(xs.min()), int(xs.max()) + 1
        rws = inkm.sum(1)
        dense = np.where(rws > rws.max() * 0.25)[0]          # 字身带（去掉发丝尖刺）
        y0 = max(by0, int(dense.min()) - pad)
        y1 = min(by1, int(dense.max()) + pad)
        sub = ga[y0:y1, bx0:bx1]
        sh, sw = sub.shape
        sc = float(TGT_W) / float(sw)                        # v340：按原图横向满幅定标
        tw, th = int(round(sw * sc)), int(round(sh * sc))
        gly = Image.fromarray((sub * 255).astype(np.uint8), 'L').resize((tw, th), Image.LANCZOS)
        aa = np.asarray(gly, np.float32) / 255.0
        # ink 顶对齐：sub 内首行有墨的行号（乘 sc 得落位后的偏移）
        _ri = np.where((sub > 0.35).any(1))[0]
        _ink_top = int(_ri.min()) if len(_ri) else 0
        cy_place = int(round(TOP_Y - _ink_top * sc))
        cy_place = max(0, min(cy_place, H - th))
        a = np.zeros((H, W), np.float32)
        cx0 = (W - tw) // 2
        a[cy_place:cy_place + th, cx0:cx0 + tw] = aa
        # v335：删除 v329 的"深蓝雾气 glow"（合成 RGB(32,52,104) 蓝雾）——
        # 用户："蓝色背景部分是原图没删干净吗"。字标直接压在干净黑底上。
        o = np.asarray(out, np.float32)
        o = o * (1 - a[..., None]) + \
            np.array([238, 240, 248], np.float32)[None, None, :] * a[..., None]
        out = Image.fromarray(np.clip(o, 0, 255).astype(np.uint8), 'RGB')
    out.save(OUT / 'pinterest6_variant.jpg', quality=93)
    print(f'[pinterest6] erase_px={int(mask.sum())} title={tw}x{th}@x{cx0} y{cy_place} '
          f'(orig x[12,3524] y[88,1800])')
    return out


# v337f：6978 轮廓重画宽度（0 = 关闭重画，保留形变自带的原描边）
P6978_RING_W = 0

TASKS = {
    'b78e60': do_b78e60,
    'pinterest3': do_pinterest3,
    'pinterest4': do_pinterest4,
    'pinterest6': do_pinterest6,
    '6978': do_6978,
}

if __name__ == '__main__':
    ids = sys.argv[1:] or list(TASKS)
    for i in ids:
        print(f'===== {i} =====')
        TASKS[i]()
    print('[DONE] ->', OUT)
