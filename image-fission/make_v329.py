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
        pad = int(R * 0.30) + 16
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
            cxl, cyl = float(xs.mean()) - bx0, float(ys.mean()) - by0
            ev = (n_bf % 2 == 0)
            # v337c：**分两条物理通道，各用正确的场型**（实测依据见下）。
            #  ① 翅姿 = 绕肩旋转，带 smoothstep 权重 → 权重梯度会拉长细线，故**只作用
            #     在实心骨架**（开运算 disk3 去掉 <6px 的触角/足 → 膨胀 4 羽化 3 的 _wm）。
            #     v337b 实测：不加这条，3px 触角被拖成 30px 长条 = "糊成一团"。
            #  ② 体型 = 各向异性缩放（sx 横向 / sy 纵向），场是无梯度的线性场——任何线
            #     段只会被等比缩放平移、绝不会拉成条带，故**直接作用于全层**（含细附件，
            #     避免触角与头部错位脱落）。这正是用户点名的"身体/翅膀比例大小"维度。
            #  偶数只：纵向拉长 1.18 / 横向收窄 0.90 + 双翅上扬 0.16rad → 修长立翅型
            #  奇数只：纵向压扁 0.90 / 横向展宽 1.12 + 双翅下压 0.14rad → 扁平展翅型
            sx, sy = (0.90, 1.18) if ev else (1.12, 0.90)
            th = 0.16 if ev else -0.14
            dyr, dxr = smod.wing_pose((ww, hh), cx=cxl, cy=cyl, theta=th,
                                      pivot_dx=0.34 * R, ramp=(0.30 * R, 1.00 * R),
                                      k_up=0.0, span=1.0, tip_flick=0.0)
            _sol = tf.ndi.binary_opening(m_i[by0:by1, bx0:bx1], structure=tf._disk(3))
            if _sol.sum() < 0.25 * max(1, int(m_i[by0:by1, bx0:bx1].sum())):
                _sol = m_i[by0:by1, bx0:bx1]
            _wm = tf.ndi.gaussian_filter(
                tf.ndi.binary_dilation(_sol, structure=tf._disk(4)).astype(np.float32), 3.0)
            _wm = np.clip(_wm, 0.0, 1.0)
            dyy = dyr * _wm + (1.0 / sy - 1.0) * (np.mgrid[0:hh, 0:ww][0].astype(np.float32) - cyl)
            dxx = dxr * _wm + (1.0 / sx - 1.0) * (np.mgrid[0:hh, 0:ww][1].astype(np.float32) - cxl)
        wlay = smod.warp_layer(rgba, (dyy.astype(np.float32), dxx.astype(np.float32)), order=1)
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
            for j in range(1, pn + 1):
                mj = plab == j
                aj = int(mj.sum())
                if aj < 40:
                    continue
                ysj, xsj = np.where(mj)
                cents.append((float(xsj.mean()), float(ysj.mean())))
                rads.append(math.sqrt(aj / math.pi))
                cols.append(np.median(np.asarray(src_img, np.float32)[mj], 0))
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
    edge_rgn = (lum > 118) & tf.ndi.binary_dilation(core, structure=tf._disk(10))
    bat = (core | edge_rgn)
    lab, n = tf.ndi.label(bat, structure=np.ones((3, 3), bool))
    if n:
        sz = tf.ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:] = sz >= 800
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
    plate = tf.erase(out, emask & disc_r, method='nn', nn_median=31, margin=20,
                     src_allow=allow_d)
    plate = tf.erase(plate, emask & (~disc_r), method='nn', nn_median=31, margin=20,
                     src_allow=allow_o)
    # v337 取证（三次试错，勿重犯）：
    # ① 不带掩膜的整体高斯扩散 → 把原蝠亮紫描边平均进腾空区（实测 188,122,186）＝幽灵。
    # ② 掩膜内谐波扩散（140×σ3，mask 外钉死）→ 洞跨过"紫碟面/粉缎带"强边界，扩散把
    #    两色混成灰绿（实测 (960,520) 由 192,133,181 变 156,160,161）＝更大破绽。
    # ③ 正解：**分区最近邻 + 取样源含粉缎带**。此前 allow_d 用 lum<105 排除了粉色缎带，
    #    于是"翼形洞"在缎带上被填成紫色 → 头部右上那条直角接缝（v337i 目检）。
    #    改成只排除"蝙蝠本体 + 8px 抗锯齿环"，缎带/亮环/碟面原样可采样 → 洞被同色续上。
    rgba_b, bbox_b = smod.make_layer(out, bat_f, feather=2.0)
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
    # ============ v336g 主体裂变加大（用户第 8 轮）============
    # 原蝠几何实测：主体中心 (776,720)；翼展最宽 506 @ y700-740（翼尖 x522/1030）；
    # 肩/头 x700-820、耳尖 y~531；尾 x762-777 自 y900 延到 y977（短而直）。
    # 用户点名四条裂变维度 → 四段位移场叠加（全部按**层内坐标**算；alpha 与 RGB
    # 一起被 warp，材质随形同步走）：
    #  ① 整体放大 1.13x（身体/翼/紫翼膜/亮紫描边等比）+ ② 双翼绕肩上扬 0.34rad
    #     + ③ 头部正面抬头（上移 26px + 1.06x）+ ④ 尾拉伸 1.07 + 侧摆 8px。
    # 符号铁律（v336 之前搞反了）：内容绕枢轴旋转 φ 时 d=(M(-φ)-I)(p-pivot)；
    # _rotB(theta) 等价 φ=-theta → 左翼 theta<0、右翼 theta>0 才是"上扬"。
    # v336 旧码两翼同号(-0.30/+... 实为双双下掠) → 用户"没看到裂变"。
    CXB, CYB, SCL = 776.0, 720.0, 1.06

    def _rotB(dyy, dxx, w, px, py, theta):
        ang = -theta * w
        ca, sa = np.cos(ang), np.sin(ang)
        dx0 = gxL - px
        dy0 = gyL - py
        return dyy + (dx0 * sa + dy0 * ca) - dy0, dxx + (dx0 * ca - dy0 * sa) - dx0

    # ① 整体放大：内容放大 S 倍 → d = (1/S - 1)(p - c)
    dyy = (1.0 / SCL - 1.0) * (gyL - CYB)
    dxx = (1.0 / SCL - 1.0) * (gxL - CXB)

    # ② 双翼上扬；翼底 y>820 smoothstep 冻结（切向位移 ∝ 力臂 dy0，下翼 dy0~280
    #    不冻结会被拖 80-120px 卷成滴痕 —— v336d 之鉴）
    ylow = np.clip((gyL - 600.0) / 300.0, 0.0, 1.0)
    ylow = ylow * ylow * (3.0 - 2.0 * ylow)
    wWL = tf.ndi.gaussian_filter(
        tf.ndi.binary_dilation(bat_c & (gxL < 735), structure=tf._disk(6)).astype(np.float32), 14.0) * (1.0 - ylow)
    wWR = tf.ndi.gaussian_filter(
        tf.ndi.binary_dilation(bat_c & (gxL > 817), structure=tf._disk(6)).astype(np.float32), 14.0) * (1.0 - ylow)
    dyy, dxx = _rotB(dyy, dxx, wWL, 700.0, 600.0, -0.28)
    dyy, dxx = _rotB(dyy, dxx, wWR, 852.0, 600.0, 0.28)
    # ②b 翼展**收缩** 0.92（用户"翅膀张开收缩"）：整体放大 1.06 后再把翼收回，
    #    净翼展 0.975 → 身体/头/尾变大而翼不外扩（不越出碟缘、不压环带文字）。
    SW = 0.92
    _px_sh = np.where(gxL < CXB, 700.0, 852.0)
    dyy = dyy + (1.0 / SW - 1.0) * (gyL - 600.0) * (wWL + wWR)
    dxx = dxx + (1.0 / SW - 1.0) * (gxL - _px_sh) * (wWL + wWR)

    # ③ 抬头：头场覆盖双耳（半宽 115 罩住 x700-820 的耳/头），覆盖混合防翼场吃耳
    head_core = bat_c & (np.abs(gxL - 767.0) < 115.0) & (gyL < 620.0)
    wH = np.clip(tf.ndi.gaussian_filter(head_core.astype(np.float32), 18.0), 0.0, 1.0)
    dyy = dyy * (1.0 - wH) + wH * ((1.0 / SCL - 1.0) * (gyL - CYB) - 19.0
                                   + (1.0 / 1.06 - 1.0) * (gyL - 620.0))
    dxx = dxx * (1.0 - wH) + wH * ((1.0 / SCL - 1.0) * (gxL - CXB)
                                   + (1.0 / 1.06 - 1.0) * (gxL - 767.0))

    # ④ 尾：拉伸 1.07（尾根 y880 固定）+ 侧摆 8px
    tail_core = bat_c & (np.abs(gxL - 770.0) < 30.0) & (gyL > 865.0)
    wT = np.clip(tf.ndi.gaussian_filter(tail_core.astype(np.float32), 6.0), 0.0, 1.0)
    tcur = np.clip((gyL - 880.0) / 110.0, 0.0, 1.0)          # 侧摆沿尾长渐增 → 平滑成弧
    tcur = tcur * tcur * (3.0 - 2.0 * tcur)
    dxx = dxx + wT * 13.0 * tcur
    dyy = dyy + wT * ((1.0 / 1.02 - 1.0) * (gyL - 880.0))

    wlay = smod.warp_layer(rgba_b, (dyy, dxx), order=1)
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
    wvis7 = np.clip(np.stack([wWL, bat_c * 0.3, wWR], -1) * 255, 0, 255).astype(np.uint8)
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
    print(f'[6978] lines={len(lines)} arc r={r:.0f}(orig) cap*1.02 shift+4deg spacing=2 bat=scale1.06+wings0.28up+contraction0.92+head-26+tailcurve')
    return out


# ---------------------------------------------------------------- pinterest6
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

    def _wregion(m, close_r=21, frac=0.12):
        m = tf.ndi.binary_dilation(m, structure=tf._disk(7))
        m = tf.ndi.binary_closing(m, structure=tf._disk(close_r))
        lab, n = tf.ndi.label(m, structure=np.ones((3, 3), bool))
        if n:
            sz = tf.ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sz >= sz.max() * frac          # union 大组件（碎片全收，防 v334 碎片化）
            m = keep[lab]
        m = tf.ndi.binary_fill_holes(m)
        w = tf.ndi.gaussian_filter(m.astype(np.float32), 36.0)
        return w / max(float(w.max()), 1e-6)          # 归一化峰值=1，部件内全场饱和

    wL = _wregion(brown6 & (xx6 < 1450) & (yy6 > 380) & (yy6 < 2380))
    wR = _wregion(brown6 & (xx6 > 2360) & (yy6 > 380) & (yy6 < 2380))
    wSK = _wregion(white6 & (((xx6 - 1850.0) / 620.0) ** 2
                             + ((yy6 - 3050.0) / 800.0) ** 2 <= 1.0), close_r=17, frac=0.20)

    def _rot6(dyy, dxx, w, px, py, theta):
        ang = -theta * w                              # 采样端 -θ ⇔ 显示端 +θ（θ>0 顺时针）
        ca, sa = np.cos(ang), np.sin(ang)
        dx0 = xx6 - px
        dy0 = yy6 - py
        return dyy + (dx0 * sa + dy0 * ca) - dy0, dxx + (dx0 * ca - dy0 * sa) - dx0

    dyy = np.zeros((H6, W6), np.float32)
    dxx = np.zeros((H6, W6), np.float32)
    dyy, dxx = _rot6(dyy, dxx, wL, 1450.0, 1700.0, 0.16)    # 左翼上扬 9.2°
    dyy, dxx = _rot6(dyy, dxx, wR, 2400.0, 1700.0, -0.13)   # 右翼上扬 7.4°（不对称）
    dyy, dxx = _rot6(dyy, dxx, wSK, 1850.0, 2950.0, -0.07)  # 骷髅左倾 4.0°
    # 下颌下沉 55px（张嘴）：y 3300..3750 带内、骷髅权重内，平滑过渡
    jw = smod.smoothstep(yy6, 3260.0, 3450.0) * (1.0 - smod.smoothstep(yy6, 3720.0, 3920.0))
    dyy = dyy - 55.0 * jw * wSK
    coords6 = [yy6 + dyy, xx6 + dxx]
    o6 = np.empty_like(a6)
    for c in range(3):
        o6[..., c] = tf.ndi.map_coordinates(a6[..., c], coords6, order=1,
                                            mode='nearest', prefilter=False)
    out = Image.fromarray(np.clip(o6, 0, 255).astype(np.uint8), 'RGB')
    wvis = np.clip(np.stack([wL, wSK, wR], -1) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(wvis, 'RGB').save(VIS / 'v335_p6_weights.jpg', quality=90)

    # ---- 字标贴回 ----
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
        sc = tgt / sh
        tw, th = int(round(sw * sc)), int(round(sh * sc))
        gly = Image.fromarray((sub * 255).astype(np.uint8), 'L').resize((tw, th), Image.LANCZOS)
        aa = np.asarray(gly, np.float32) / 255.0
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
    print(f'[pinterest6] erase_px={int(mask.sum())} title={tw}x{th}@{cy_place}')
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
