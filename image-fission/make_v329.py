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
    res.save(OUT / 'pinterest3_variant.jpg', quality=93)
    print(f'[pinterest3] letters={len(letters)} word={word!r} -> {OUT}')
    return res


# ---------------------------------------------------------------- b78e60
def do_b78e60():
    """军标迷彩文字：原字本身就是**实心黑粗字**，材质=纯黑，故原位重画黑字即同材质。"""
    img0 = Image.open(BY['b78e60']['path'])
    excl = v328.region_mask(*img0.size,
                            pred=lambda xx, yy: (xx > 500) & (xx < 850) & (yy > 762) & (yy < 1120))
    out = v328.text_variant(
        'b78e60',
        [dict(box=(360, 430, 1200, 790),
              words=['WE DEFEND THE', 'STEEL', 'HAWKS'], min_area=120)],
        'blackopsone', erase_kw=dict(method='nn', nn_median=41), dilate=5, exclude=excl)
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
    # 碟面内允许取样区（蝙蝠以外的碟面像素）：蝙蝠填充只从**碟面自身**取色，
    # 才不会把碟外的粉色拖进来（实测碟面被打成一块糊紫）。
    yy0, xx0 = yy, xx
    disc_src = (((xx0 - 812) / 205.0) ** 2 + ((yy0 - 730) / 205.0) ** 2 <= 1.0) & (~bat)
    for tag, msk, kw in (('arc', m_arc, dict(method='nn', nn_median=31, src_allow=src_ribbon)),
                         # 品牌字：hybrid = nn 保边界色 + LaMa 补低频结构。
                         # ⚠️ 不用 patch 平移填充：平移 120px 会**复制到同一行文字本身** →
                         #    新字上叠出旧字双影（实测 NOCTAVEN 上透出 BACARDÍ）。
                         ('brand', brand, dict(method='hybrid', nn_median=31)),
                         # 蝙蝠：碟面内部有强径向渐变，nn 会留下"蝙蝠形状的浅色鬼影"。
                         # 改用调和平滑填充（解 Laplace，只用边界值延拓）→ 碟面自然过渡。
                         ('bat', bat, dict(method='diffuse', diffuse_down=1,
                                           src_allow=disc_src))):
        if not msk.any():
            continue
        mdd = tf.ndi.binary_dilation(msk, structure=tf._disk(9)) if tag != 'bat' else msk
        out = tf.erase(out, mdd, **kw)
    out.save(VIS / 'v329_6978_erased.jpg', quality=95)

    # ---- v330 重画：从原图蝙蝠做结构保持形变（subject_morph），保留原材质/描边/解剖结构 ----
    from styles import subject_morph as smod
    rgba, box = smod.make_layer(img, bat, feather=1.5, pad=14)
    # 形变参数：翼微展+翼尖上勾+尾部拉长 → 同构异姿，anatomy 100% 来自原图
    disp = smod.wing_pose(rgba.shape[:2][::-1], cx=815.0, cy=635.0,
                          theta=-0.12, pivot_dx=80.0, ramp=(28.0, 235.0),
                          k_up=10.0, span=1.04, tip_flick=8.0,
                          tip_ramp=(180.0, 250.0), tail_stretch=18.0, tail_y=780.0)
    warped = smod.warp_layer(rgba, disp, order=1)
    out = smod.paste_layer(out, warped, box)
    Image.fromarray(np.clip(np.asarray(warped[..., :3], np.float32), 0, 255).astype(np.uint8), 'RGB').save(VIS / 'v330_6978_morphed.jpg', quality=95)

    # ---- 重画：文字（原字为实心黑 Didone serif，材质=纯黑）----
    for (w, b, m) in lines:
        out = tf.draw_line(out, w, b, 'playfair_black', tf.text_color(img, m), fit='squeeze')
    out = tf.draw_arc(out, 'LA CASA DELLE OMBRE', (cx, cy), r, cap_h,
                      'playfair_black', tf.text_color(img, m_arc), start_deg=a0, end_deg=a1)
    out.save(OUT / '6978_variant.jpg', quality=93)
    print(f'[6978] lines={len(lines)} bat_px={int(bat.sum())}')
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

    # ---- 擦除掩膜 ----
    DS = 4
    sm = img.resize((W // DS, H // DS), Image.LANCZOS)
    a = np.asarray(sm, np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = a.max(2) - a.min(2)
    ink = (lum > 30) & (sat < 62)
    ink[int(1530 // DS):, :] = False
    cl = tf.ndi.binary_closing(ink, structure=tf._disk(12))
    lab, n = tf.ndi.label(cl)
    best, bA = np.zeros_like(cl), 0
    for i in range(1, n + 1):
        m = lab == i
        ys = np.where(m.any(1))[0]
        if len(ys) and ys.min() < int(300 // DS) and int(m.sum()) > bA:
            bA = int(m.sum()); best = m
    best = tf.ndi.binary_fill_holes(best)
    # v330 fix: diffuse（Laplace 平滑梯度）替代 nn（最近邻会把下方鹰/骷髅色拉进标题区）
    # 同时缩小膨胀半径（disk(6) 替代 disk(10)），减少 mask 吃进插图区的风险
    best = tf.ndi.binary_dilation(best, structure=tf._disk(6))
    mask = np.asarray(Image.fromarray((best * 255).astype(np.uint8), 'L')
                      .resize((W, H), Image.NEAREST)) > 128
    vis = np.asarray(img).copy(); vis[mask] = [255, 0, 0]
    Image.fromarray(vis).save(VIS / 'v329_p6_mask.png')
    # v330: diffuse 平滑填充（从四边界混合渐变），不再用 nn（会拉进下方插图色）或 const（黑色色块）
    out = tf.erase(img, mask, method='diffuse', diffuse_down=4)
    out.save(VIS / 'v329_p6_erased.jpg', quality=95)

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
        # 深蓝雾气（复刻原图标题背后的烟雾）
        glow = np.asarray(Image.fromarray((a * 255).astype(np.uint8), 'L')
                          .filter(ImageFilter.GaussianBlur(38)), np.float32) / 255.0
        glow = np.clip(glow * 1.5, 0, 1)
        o = np.asarray(out, np.float32)
        o = o * (1 - 0.62 * glow[..., None]) + \
            np.array([32, 52, 104], np.float32)[None, None, :] * (0.62 * glow[..., None])
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
