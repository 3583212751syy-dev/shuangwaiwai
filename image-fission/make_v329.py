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
    vis = np.asarray(det).copy(); vis[md] = [255, 0, 0]
    Image.fromarray(vis).save(VIS / 'v329_p3_mask.png')
    out = tf.erase(img, md, method='nn', nn_median=41)
    out = tf.match_fill_level(out, det, md, size=241)
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
    dots = np.zeros((H, W), bool)                # v335: 虚线点收集（轨迹级裂变用）
    for i in range(1, cn + 1):
        m_i = clab == i
        area = int(m_i.sum())
        if area < 2500:                          # v335: 点/噪声 → 收进 dots，轨迹级处理
            if area >= 60:
                dots |= m_i
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
            # v335（用户："这种有小元素的图片为什么不把小元素也裂变了"）：
            # 自转对近对称小蝶=自相似不可见（v333 ±0.40 用户无感）→ 改
            # **镜像翻转**（左右翼互换，剪影必然改变）+ 小角度旋转 + 缩放。
            # 顶蝶（comp1）：翻转+0.20rad+1.05x；底蝶（comp29）：-0.24rad+0.93x。
            rgba = np.ascontiguousarray(rgba[:, ::-1]) if n_bf % 2 == 0 else rgba
            th = 0.20 if n_bf % 2 == 0 else -0.24
            sc = 1.05 if n_bf % 2 == 0 else 0.93
            cxl, cyl = float(xs.mean()) - bx0, float(ys.mean()) - by0
            yyS, xxS = np.mgrid[0:hh, 0:ww].astype(np.float32)
            dx0 = xxS - cxl
            dy0 = yyS - cyl
            wgt = smod.smoothstep(np.hypot(dx0, dy0), 4.0, 0.6 * R)
            ang = -th * wgt
            ca_, sa2 = np.cos(ang), np.sin(ang)
            dyy = (dx0 * sa2 + dy0 * ca_) - dy0
            dxx = (dx0 * ca_ - dy0 * sa2) - dx0
            dyy = dyy + dy0 * (1.0 / sc - 1.0) * wgt         # 缩放（采样端 1/s）
            dxx = dxx + dx0 * (1.0 / sc - 1.0) * wgt
        wlay = smod.warp_layer(rgba, (dyy.astype(np.float32), dxx.astype(np.float32)), order=1)
        res = smod.paste_layer(res, wlay, mbox)
        n_bf += 1
    print(f'[pinterest3] butterflies morphed={n_bf}')

    # ---- v335: 虚线轨迹裂变（单点旋转不可见 → 膨胀聚类成轨迹，整条微旋+平移）----
    n_tr = 0
    if dots.any():
        dl = tf.ndi.binary_dilation(dots, structure=tf._disk(14))
        dlab, dn = tf.ndi.label(dl, structure=np.ones((3, 3), bool))
        for k_ in range(1, dn + 1):
            pts = dots & (dlab == k_)
            if int(pts.sum()) < 300:
                continue
            ys_, xs_ = np.where(pts)
            x0_, x1_ = int(xs_.min()), int(xs_.max()) + 1
            y0_, y1_ = int(ys_.min()), int(ys_.max()) + 1
            pad = 12
            bx0_, by0_ = max(0, x0_ - pad), max(0, y0_ - pad)
            bx1_, by1_ = min(W, x1_ + pad), min(H, y1_ + pad)
            sub_m = np.zeros((H, W), bool)
            sub_m[by0_:by1_, bx0_:bx1_] = pts[by0_:by1_, bx0_:bx1_]
            rgba_t, tbox = smod.make_layer(src_img, sub_m, feather=1.2,
                                           box=(bx0_, by0_, bx1_, by1_))
            hh_, ww_ = rgba_t.shape[:2]
            yyT, xxT = np.mgrid[0:hh_, 0:ww_].astype(np.float32)
            cxl, cyl = float(xs_.mean()) - bx0_, float(ys_.mean()) - by0_
            sgn = 1.0 if k_ % 2 else -1.0
            ang = -0.14 * sgn
            ca_, sa2 = np.cos(ang), np.sin(ang)
            dx0 = xxT - cxl
            dy0 = yyT - cyl
            dyy = (dx0 * sa2 + dy0 * ca_) - dy0 + 3.0 * sgn
            dxx = (dx0 * ca_ - dy0 * sa2) - dx0 + 7.0 * sgn
            wlay = smod.warp_layer(rgba_t, (dyy.astype(np.float32),
                                            dxx.astype(np.float32)), order=1)
            res = smod.paste_layer(res, wlay, tbox)
            n_tr += 1
    print(f'[pinterest3] trails morphed={n_tr}')

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

    # ---- 蝙蝠擦除掩膜（v330：深色主体 + 浅色描边，全 footprint 擦除后贴回形变层）----
    arr = np.asarray(img, np.float32)
    mx = arr.max(2)
    lum = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    ell = ((xx - 815) / 252.0) ** 2 + ((yy - 742) / 262.0) ** 2 <= 1.0
    core = (mx < 58) & ell
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

    # ---- v334 主体蝙蝠裂变：**部件级刚性旋转**（p6 同一教训：平滑位移场自相似不可见，
    #      三轮用户连续说"主体没裂变"）。左右翼绕翼根刚性旋转：
    #      左翼 +0.24rad（翼尖上扬 ~55px）、右翼 +0.10rad（翼尖下压 ~23px）
    #      → 姿态从"平展"变"左攻角"，剪影角度变化一眼可见，材质 100% 保留。
    #      权重=蝙蝠 mask(fill_holes+膨胀+高斯羽化)，碟面渐变在羽化带内随翼流动。
    #      位移铁律：θ>0=顺时针；左翼尖(-x)顺时针=上扬，右翼(+x)顺时针=下压。
    a7 = np.asarray(out, np.float32)
    yy7, xx7 = np.mgrid[0:H, 0:W].astype(np.float32)

    def _rot7(dyy, dxx, w, px, py, theta):
        ang = -theta * w
        ca, sa = np.cos(ang), np.sin(ang)
        dx0 = xx7 - px
        dy0 = yy7 - py
        return dyy + (dx0 * sa + dy0 * ca) - dy0, dxx + (dx0 * ca - dy0 * sa) - dx0

    bat_f = tf.ndi.binary_fill_holes(bat)
    # v335c：权重**避开躯干**（v335b 教训：左右半幅各自旋转时中轴线 x≈815 对拉剪切，
    # 把蝙蝠躯干撕开露出内里淡紫 = 淡紫横带）。翼场只罩翼膜（xx<760 / xx>870），
    # 枢轴移到翼根 → 躯干+肩部零位移完整保留，翼膜绕根旋转。
    wBL = tf.ndi.gaussian_filter(
        tf.ndi.binary_dilation(bat_f & (xx7 < 760), structure=tf._disk(6)).astype(np.float32), 7.0)
    wBR = tf.ndi.gaussian_filter(
        tf.ndi.binary_dilation(bat_f & (xx7 > 870), structure=tf._disk(6)).astype(np.float32), 7.0)
    dyy = np.zeros((H, W), np.float32)
    dxx = np.zeros((H, W), np.float32)
    # v335（用户："主体蝙蝠裂变能不能正常一些，改变头部四肢类的结构剪影"）：
    # 废 v334"左攻角"不对称姿态（用户觉得畸形）。改自然展翅 + **结构剪影**变化：
    #   · 双翼对称上扬 0.15rad（8.6°，翼尖各抬 ~52px，剪影一眼可见且自然）；
    #   · 头部（耳+头，中轴带 y<690）绕颈点 (815,690) 歪头 -0.10rad → 头/耳轮廓偏转；
    #   · 足/尾簇（中轴带 y>820）绕体底 (815,810) 摆动 +0.10rad → 四肢剪影变化。
    dyy, dxx = _rot7(dyy, dxx, wBL, 780.0, 650.0, 0.15)     # 左翼绕翼根上扬 8.6°
    dyy, dxx = _rot7(dyy, dxx, wBR, 850.0, 650.0, -0.15)    # 右翼绕翼根上扬 8.6°（对称）
    wHD = tf.ndi.gaussian_filter(
        (bat_f & (np.abs(xx7 - 815.0) < 95) & (yy7 < 700)).astype(np.float32), 5.0)
    dyy, dxx = _rot7(dyy, dxx, wHD, 815.0, 690.0, -0.09)    # 歪头（头部剪影变）
    # v335b 实测：足部旋转场会把碟面渐变拖出淡紫横带（收益小伪影大）→ 删除。
    # "四肢结构变化"由双翼上扬承担（翼=蝙蝠的前肢，肩/臂/翼尖全部位移）。
    coords7 = [yy7 + dyy, xx7 + dxx]
    o7 = np.empty_like(a7)
    for c in range(3):
        o7[..., c] = tf.ndi.map_coordinates(a7[..., c], coords7, order=1,
                                            mode='nearest', prefilter=False)
    out = Image.fromarray(np.clip(o7, 0, 255).astype(np.uint8), 'RGB')
    wvis7 = np.clip(np.stack([wBL, bat_f * 0.3, wBR], -1) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(wvis7, 'RGB').save(VIS / 'v334_6978_weights.jpg', quality=90)

    # ---- 重画：文字（原字为实心黑 Didone serif，材质=纯黑）----
    for (w, b, m) in lines:
        out = tf.draw_line(out, w, b, 'playfair_black', tf.text_color(img, m), fit='squeeze')
    # v334: 弧形字**排版**也裂变（用户："弧形的字体排版也还是没裂变啊"）——
    # v333 只换了词、排版原样 = "只裂变了字体"。本版：弧度压平（r*1.22）、
    # 字号加大（cap*1.12）、角跨各展 4° → 弧形/大小/字距全部可见变化，
    # 仍是同圆心顶部弧排、同字体语言、同材质（黑 Didone serif）。
    out = tf.draw_arc(out, "LA LUNA NELL'OMBRA", (cx, cy), r * 1.22, cap_h * 1.12,
                      'playfair_black', tf.text_color(img, m_arc),
                      start_deg=a0 - 4.0, end_deg=a1 + 4.0)
    out.save(OUT / '6978_variant.jpg', quality=93)
    print(f'[6978] lines={len(lines)} arc="LA LUNA NELL\'OMBRA" r*1.22 cap*1.12 span+8deg bat=rigid-wings')
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
    out = tf.erase(img, mask, method='lama', max_side=1100, margin=40)
    out.save(VIS / 'v332_p6_erased.jpg', quality=95)

    # ---- v335（用户："蓝色背景部分是原图没删干净吗"）：标题带内的**原图蓝烟**一并擦除。
    # v329 起刻意保留蓝烟是误判——用户要的是干净黑底。蓝烟判据 b>r+8 且 lum>40：
    # 白射线(低饱和)/棕羽(r>b)均不命中；y<1300（翼顶 ~1560 之上，安全）。
    blue = (a[..., 2] > a[..., 0] + 8.0) & (lum > 40)
    blue[int(1300 // DS):, :] = False
    blue = tf.ndi.binary_dilation(blue, structure=tf._disk(4))
    bmask = np.asarray(Image.fromarray((blue * 255).astype(np.uint8), 'L')
                       .resize((W, H), Image.NEAREST)) > 128
    out = tf.erase(out, bmask, method='nn', nn_median=41, margin=30)
    out.save(VIS / 'v335_p6_erased_smoke.jpg', quality=95)

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

    # ---- v335b 残余蓝烟二次压制：nn 填充取样自周边蓝烟会**回填蓝色**（实测顶部两角
    # 仍剩蓝云）。色域法软压制：蓝雾像素（b>r+6 渐进权重）去饱和+压暗融入黑底，
    # 烟雾纹理保留；lum>150 的白色字标/射线不受影响。
    o6s = np.asarray(out, np.float32)
    lum6s = o6s @ np.array([0.299, 0.587, 0.114], np.float32)
    blueness = np.clip((o6s[..., 2] - o6s[..., 0] - 6.0) / 24.0, 0.0, 1.0)
    band6 = 1.0 - smod.smoothstep(yy6, 1150.0, 1400.0)
    sw = blueness * band6 * (1.0 - smod.smoothstep(lum6s, 130.0, 170.0))
    gray6 = o6s.mean(2, keepdims=True)
    o6s = o6s * (1 - 0.88 * sw[..., None]) + gray6 * 0.35 * sw[..., None]
    out = Image.fromarray(np.clip(o6s, 0, 255).astype(np.uint8), 'RGB')

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
