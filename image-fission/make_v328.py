"""make_v328.py — v328 五图裂变：以**原图为底**做原位裂变（保色 + 保相关性 + 禁遮挡）

针对用户 2026-09-14 反馈的路线修正：
  ❌ 旧路线：全图 SDXL 重生 -> 黑底变羊皮、迷彩糊成一团、旧字残影
  ✅ 新路线：原图不动，只做**局部原位裂变**
     - 文字：笔画级掩膜 -> LaMa 只擦笔画 -> 同字体/同字号/同位置重画新词（禁矩形掩膜=禁遮挡）
     - 迷彩(pinterest4)：色板量化后只形变**标签图** -> 色块区域/角度/大小变，颜色 100% 不变

用法：venv/Scripts/python.exe make_v328.py [iid ...]
"""
import json, sys, math
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from styles import textfix as tf

ROOT = Path('.')
OUT = ROOT / 'jobs' / 'router_out_v328'; OUT.mkdir(parents=True, exist_ok=True)
VIS = ROOT / 'jobs' / '_probe'; VIS.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / 'regression_set/set.json', encoding='utf-8'))
BY = {i['id']: i for i in cfg['images']}


def region_mask(W, H, pred):
    yy, xx = np.mgrid[0:H, 0:W]
    return pred(xx, yy)


def fit_circle(xs, ys):
    A = np.c_[2 * xs.astype(np.float64), 2 * ys.astype(np.float64), np.ones(len(xs))]
    b = (xs.astype(np.float64) ** 2 + ys.astype(np.float64) ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    return cx, cy, float(np.sqrt(sol[2] + cx * cx + cy * cy))


def arc_span(ang):
    """给一堆角度（度，可正可负），求**最大连续弧段**（找最大空档，取补集）。"""
    a = np.sort(np.mod(ang, 360.0))
    if len(a) < 10:
        return 0.0, 180.0
    d = np.diff(np.concatenate([a, a[:1] + 360.0]))
    i = int(np.argmax(d))
    return float(a[(i + 1) % len(a)]), float(a[i])


def camo_reshape(img, seed=7, colors=5, strength=0.06, freq=2, keep_thin=True):
    """迷彩「湖泊」色块重塑：量化成色板 -> 只形变**标签图**（最近邻 -> 硬边不糊）-> 回填原色板。

    颜色 100% 来自原图（不可能变色）；色块形状/角度/大小/选取范围改变（每 seed 不同）。
    keep_thin=True 时保护最暗标签（棕榈剪影）不被拉丝——物种完整保留，只随形变轻微移位/转向。
    """
    from scipy import ndimage as ndi
    pimg = img.convert("P", palette=Image.ADAPTIVE, colors=colors, dither=Image.NONE)
    raw = np.asarray(pimg)
    pal_raw = np.array(pimg.getpalette()[:256 * 3], np.int32).reshape(-1, 3)
    used = np.unique(raw)
    pal = pal_raw[used]
    remap = np.zeros(256, np.int32)
    for i, v in enumerate(used):
        remap[int(v)] = i
    idx = remap[np.asarray(pimg)].astype(np.float32)
    lum = pal.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    thin_label = int(np.argmin(lum))            # 最暗 = 棕榈剪影
    thin = (idx == thin_label)
    H, W = idx.shape
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, H, np.float32), np.linspace(0, 1, W, np.float32), indexing="ij")
    dx = np.zeros((H, W), np.float32); dy = np.zeros((H, W), np.float32)
    for k in range(1, freq + 1):
        amp = strength / k
        dx += amp * np.sin(2 * np.pi * k * xx + rng.uniform(0, 6.283)) * W
        dy += amp * np.cos(2 * np.pi * k * yy + rng.uniform(0, 6.283)) * H
    coords = np.stack([(yy * H + dy).ravel(), (xx * W + dx).ravel()])
    w = ndi.map_coordinates(idx, coords, order=0, mode="nearest").reshape(H, W)
    w = np.clip(np.rint(w), 0, len(pal) - 1).astype(np.int32)
    out = Image.fromarray(pal[w].astype(np.uint8), "RGB")
    if keep_thin:
        # 棕榈剪影保留原**像素**（含反锯齿边），用羽化掩膜贴回。
        # ⚠️ 不能只在标签图上把 thin 写回：原图棕榈边缘是中间调像素，不属于「最暗标签」，
        #     形变会把它们带走 → 每棵棕榈周围留下一圈浅色描边（肉眼可见的"贴纸边"）。
        from PIL import ImageFilter
        pm = np.zeros((H, W), np.uint8)
        pm[ndi.binary_dilation(thin, iterations=3)] = 255
        m = Image.fromarray(pm, "L").filter(ImageFilter.GaussianBlur(1.5))
        out = Image.composite(img, out, m)
    return out


# ---------------------------------------------------------------- 通用文字图
def text_variant(iid, specs, font, direction='dark', mode='lum', thr=None,
                 exclude=None, erase_kw=None, level_size=241, tag='', dilate=4):
    ic = BY[iid]
    img = Image.open(ic['path']).convert('RGB')
    W, H = img.size
    lines, mask = [], np.zeros((H, W), bool)
    for sp in specs:
        wl = list(sp['words'])
        ls = tf.detect_lines(img, sp['box'], direction=sp.get('direction', direction),
                             mode=mode, thr=sp.get('thr', thr), pad=sp.get('pad', 40),
                             exclude=exclude, min_area=sp.get('min_area', 300),
                             min_h=sp.get('min_h', 16))
        # 一个框里可能检出行带之外的小件（原字的下半截/细笔画）——一并擦掉，只取第一条画字
        for bb in sp.get('extra', []):
            mask |= extra_specks(img, bb, exclude=exclude)
        ls = ls[:len(wl)]
        for (b, m) in ls:
            lines.append((wl.pop(0), tuple(sp.get('draw_box') or b), m,
                          sp.get('cap_scale', 1.0), sp.get('color')))
            mask |= m
    if mask.any():
        mask = tf.ndi.binary_dilation(mask, structure=tf._disk(dilate))
    vis = np.asarray(img).copy(); vis[mask] = [255, 0, 0]
    Image.fromarray(vis).save(VIS / f'{iid}{tag}_m328.png')
    out = tf.erase(img, mask, **(erase_kw or {}))
    out.save(VIS / f'{iid}{tag}_erased.jpg', quality=95)
    if erase_kw is None or 'method' not in erase_kw or erase_kw.get('method') == 'lama':
        out = tf.match_fill_level(out, img, mask, size=level_size)
    for (w, b, m, cs, col) in lines:
        out = tf.draw_line(out, w, b, font,
                           tuple(col) if col else tf.text_color(img, m), cap_scale=cs)
    out.save(OUT / f'{iid}_variant.jpg', quality=93)
    print(f'[{iid}] lines={len(lines)} mask={int(mask.sum())}')
    for (w, b, m, cs, col) in lines:
        print(f'    {w!r:<20} box={b} h={b[3]-b[1]}')
    return out


def arc_letter_mask(img, box, cx, cy, thr=80, max_ang_span=26.0,
                    min_area=120, exclude=None, emblem=None, radial=None):
    """弧线上的**字母**掩膜（笔画级，绝不整带填充）。

    两条路线：
    - radial=None（第一遍，还没有 r/cap）：mode='lum' + fill=False，按连通域的**角向跨度**筛掉
      缎带外框那两条长弧线（跨度 150°+），只留字母（每个字母 ≤ 26°）。用于先量出 r/cap。
    - radial=(r, cap)（第二遍，已知几何）：直接用**径向带** |d-r| <= 0.63*cap 逐像素裁剪。
      ⚠️ 必须用径向带而不是角向跨度：弧线末端（近水平段）相邻字母会互相粘连成一个大连通域
      （实测跨度 177°），角向跨度判据会把它整块判为「缎带外框」丢掉 → 该处旧字残留，
      新字压上去就成了「双影」。径向带只按「离圆心多远」筛，粘连的字母也会被保住，
      而两条边界弧线（|d-r|≈45 ≈ 0.83cap）仍在带外被排除。
    返回 (mask, cx, cy, r, cap_h)。
    """
    raw = tf.stroke_mask(img, box, mode='lum', thr=thr, direction='dark', pad=0,
                         min_area=0, dilate=0, fill=False, exclude=emblem)
    m = raw
    if radial is not None:
        r0, cap0 = radial
        H, W = raw.shape
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.hypot(xx - cx, yy - cy)
        m = raw & (np.abs(d - r0) <= cap0 * 0.63)
    lab, n = tf.ndi.label(m, structure=np.ones((3, 3), bool))
    keep = np.zeros_like(m)
    for i, sl in enumerate(tf.ndi.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = np.where(lab[sl] == i)
        if len(ys) < min_area:
            continue
        if radial is None:
            yy, xx = ys + sl[0].start, xs + sl[1].start
            ang = np.sort(np.mod(np.degrees(np.arctan2(-(yy - cy), xx - cx)), 360.0))
            gaps = np.diff(np.concatenate([ang, ang[:1] + 360.0]))
            if 360.0 - float(gaps.max()) > max_ang_span:
                continue
        keep[sl] |= (lab[sl] == i)
    keep = tf.ndi.binary_fill_holes(keep)
    if exclude is not None:
        keep &= ~exclude
    return keep, cx, cy, 0.0, 0.0


def arc_metrics(mask, cx, cy, bin_deg=2.0):
    """从字母掩膜量出 (真实圆心, 半径, 角度范围, cap 高)。

    cap 高**不能**用全局分位数（会把不同位置的字母混在一起，实测低估 25%）。
    正确做法：按角度分箱，量每箱内字母的**径向跨度**，取中位数 = 单字高度。
    """
    ys, xs = np.where(mask)
    if len(xs) < 50:
        return cx, cy, 0.0, 0.0, 180.0, 0.0
    cx, cy, r = fit_circle(xs, ys)
    rad = np.hypot(xs - cx, ys - cy)
    ang = np.degrees(np.arctan2(-(ys - cy), xs - cx))
    a0, a1 = float(np.percentile(ang, 1)), float(np.percentile(ang, 99))
    spans, mids = [], []
    for lo in np.arange(a0, a1, bin_deg):
        sel = (ang >= lo) & (ang < lo + bin_deg)
        if sel.sum() < 12:
            continue
        rr = rad[sel]
        lo_r, hi_r = float(np.percentile(rr, 2)), float(np.percentile(rr, 98))
        if hi_r - lo_r < 4:
            continue
        spans.append(hi_r - lo_r)
        mids.append((lo_r + hi_r) / 2)
    if not spans:
        return cx, cy, float(np.percentile(rad, 50)), 0.0, a0, a1
    # cap 高取分箱跨度的 p88：中位数会被「字与字之间的窄箱」拉低（实测低估 ~12%）
    cap = float(np.percentile(spans, 88))
    # 半径：median(mids) 是**原字**的心线；arc_text 的逐字贴图会整体内偏 ~0.22cap，
    # 故 +0.26cap 让新字落在原字同一条心线上（实测对齐误差 <5px）
    return cx, cy, float(np.median(mids)) + cap * 0.26, cap, a0, a1


def extra_specks(img, box, thr=80, area_lo=40, area_hi=2600, exclude=None):
    """字行**外**的附属小符号（重音符/® 等）：面积落在 [area_lo, area_hi] 且整体在该 box 内。

    为什么需要：检测框是按「字母带」定的，原字带的撇/® 常落在带外，
    主检测漏掉 → 擦除后成品上会留下一个孤立的旧符号（实测 BACARDÍ 的 Í 重音、
    MCHEART 的 ® 都残留过）。所以对已知符号位置补小框单独收。
    """
    raw = tf.stroke_mask(img, box, mode='lum', thr=thr, direction='dark', pad=0,
                         min_area=0, dilate=0, fill=False, exclude=exclude)
    lab, n = tf.ndi.label(raw, structure=np.ones((3, 3), bool))
    keep = np.zeros_like(raw)
    for i, sl in enumerate(tf.ndi.find_objects(lab), 1):
        if sl is None:
            continue
        ar = int((lab[sl] == i).sum())
        if area_lo <= ar <= area_hi:
            keep[sl] |= (lab[sl] == i)
    return keep


def do_6978():
    """6978：弧字（拟合圆）+ Est/1862 + BACARDÍ/MCHEART 全部原位改写。

    擦除用 nn 填充（原图是平滑粉紫渐变+暗纹理）：LaMa 会把大字块抹成白雾/灰雾，
    nn 用边界色向内延伸 → 底色自然延续。其中**弧字只在缎带带内取样**，
    否则带外的粉底会被拖进带内，把缎带的白色染成粉色。
    """
    ic = BY['6978']
    img = Image.open(ic['path']).convert('RGB')
    W, H = img.size
    # 徽章保护圈（紫色圆盘+蝙蝠）：别把它当文字擦掉
    emblem = region_mask(W, H, lambda xx, yy: (xx - 813) ** 2 + (yy - 690) ** 2 <= 268 ** 2)
    # 1) 弧字：只留字母连通域 -> 分箱量半径/cap 高
    cx0, cy0 = 777.0, 728.0
    # 两遍：先靠角向跨度筛出孤立字母量出 r/cap，再用径向带 |d-r|<=0.63cap 重建完整字母掩膜
    # （弧线末端字母粘连，第一遍会整块丢掉 → 必须第二遍补回，否则末端旧字残留出双影）
    m_arc0, _, _, _, _ = arc_letter_mask(img, (235, 245, 1345, 825), cx0, cy0,
                                         thr=80, emblem=emblem)
    cx, cy, r, cap_h, a0, a1 = arc_metrics(m_arc0, cx0, cy0)
    m_arc, _, _, _, _ = arc_letter_mask(img, (235, 245, 1345, 825), cx0, cy0,
                                        thr=80, emblem=emblem, radial=(r, cap_h))
    print(f'[6978] arc c=({cx:.0f},{cy:.0f}) r={r:.0f} capH={cap_h:.0f} ang={a0:.1f}..{a1:.1f} px={int(m_arc.sum())}')
    # 缎带带（内外描边之间的白带）作为弧字擦除的**唯一取样源**
    yy, xx = np.mgrid[0:H, 0:W]
    band = np.hypot(xx - cx, yy - cy)
    ribbon = (band > r - cap_h * 0.95) & (band < r + cap_h * 0.95)
    # 缎带带内允许取样的区域（去掉笔画本身，避免取样源自我污染）
    src_ribbon = ribbon & (~tf.ndi.binary_dilation(m_arc, structure=tf._disk(3)))
    # 2) 品牌文字行（thr 80：只取近黑字，排除背景暗纹理）
    specs = [
        dict(box=(285, 800, 640, 935), words=['SET'], pad=10, min_area=120,
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
        print(f'   [{sp["words"][0]}] candidates={[ (b, b[3]-b[1]) for b, _ in ls ]}')
        if ls:
            b, m = ls[0]
            lines.append((sp['words'][0], tuple(sp.get('draw_box') or b), m)); brand |= m
    # 原字行的附属小符号（BACARDÍ 的 Í 重音、MCHEART 的 ®）——按字母带定的框收不到，补收
    for bb in [(1150, 930, 1300, 1010), (1120, 1170, 1200, 1235)]:
        brand |= extra_specks(img, bb, exclude=excl_brand)
    out = img
    for tag, msk, kw in (('arc', m_arc, dict(src_allow=src_ribbon)),
                         ('brand', brand, {})):
        if not msk.any():
            continue
        md = tf.ndi.binary_dilation(msk, structure=tf._disk(9))
        vis = np.asarray(img).copy(); vis[md] = [255, 0, 0]
        Image.fromarray(vis).save(VIS / f'6978_{tag}_m328.png')
        out = tf.erase(out, md, method='nn', nn_median=41, **kw)
    out.save(VIS / '6978_erased.jpg', quality=95)
    for (w, b, m) in lines:
        # fit='squeeze'：原字是**窄体重 Didone**（MCHEART 7 字占 686px / cap145），
        # Playfair Black 字面宽 → 用缩字号会在 cap 上差一倍（实测 82 vs 141）。
        # 改成保 cap 高、横向压缩墨迹，字号与原字一致。
        out = tf.draw_line(out, w, b, 'playfair_black', tf.text_color(img, m), fit='squeeze')
        print(f'    {w!r:<12} box={b} h={b[3]-b[1]}')
    out = tf.draw_arc(out, 'LA CASA DELLE OMBRE', (cx, cy), r, cap_h,
                      'playfair_black', tf.text_color(img, m_arc), start_deg=a0, end_deg=a1)
    out.save(OUT / '6978_variant.jpg', quality=93)
    return out


def do_pinterest4():
    img = Image.open(BY['pinterest4']['path']).convert('RGB')
    out = camo_reshape(img, seed=17, colors=5, strength=0.075, freq=2)
    out.save(OUT / 'pinterest4_variant.jpg', quality=93)
    print('[pinterest4] camo reshape done')
    return out


def do_pinterest6():
    """pinterest6：金属尖刺标题整体原位替换（黑底必须保持黑）。

    关键结论（实测）：
      1) 标题背后那层「深蓝烟雾」是标题自身的**光晕**，不是背景 → 必须一起擦掉，
         否则黑底上会留下一个**标题形状的蓝色鬼影轮廓**。
      2) 烟雾靠「蓝 > 红」判据区分（羽毛是棕=红>蓝），所以不会被误擦成羽毛。
      3) 擦除用**定值黑 + 羽化**（不是 LaMa/nn）：LaMa 会长蓝雾，nn 会灌亮点，
         定值黑+36px 羽化与周围黑底自然咬合，且绝不引入任何"猜测"色块。
      4) 字号/位置按原 logo **字母带**（y 480..1140）定 cap 高并居中（含尖刺的原框会偏高）。
    """
    ic = BY['pinterest6']
    img = Image.open(ic['path']).convert('RGB')
    W, H = img.size
    arr = np.asarray(img, np.float32)
    lum = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    ls = tf.detect_lines(img, (0, 150, W, 1140), mode='lum', thr=72,
                         direction='light', pad=30, min_area=600)
    logo = ls[0][1]
    # glow（标题背光/深蓝烟雾 + 上角深蓝三角）：lum 下限放到 8。
    # 原图黑底实测是 (22,22,22) 的近黑灰，烟雾外缘暗到 lum≈8~18，用 18 会漏掉一圈淡蓝残影。
    glow = (yy < 1135) & ((arr[..., 2] - arr[..., 0]) > 6) & (lum > 8) & (lum < 215)
    mask = tf.ndi.binary_dilation(logo | glow, structure=tf._disk(14))
    print(f'[pinterest6] logo={int(logo.sum())} glow={int(glow.sum())} mask={int(mask.sum())}')
    vis = np.asarray(img).copy(); vis[mask] = [255, 0, 0]
    Image.fromarray(vis).save(VIS / 'pinterest6_m328.png')
    # 填充色 = 原图实测底色（标题带上下的黑底中位亮度=0，纯黑）：
    # ⚠️ 别用近黑灰(22,22,22)——那比真实黑底亮，会在标题位留下一块肉眼可见的灰斑。
    out = tf.erase(img, mask, method='const', const_color=(2, 2, 6), const_feather=30)
    out.save(VIS / 'pinterest6_erased.jpg', quality=95)
    out = tf.draw_line(out, 'RAVEN', (12, 480, W - 6, 1140), 'metal', (222, 222, 228))
    out.save(OUT / 'pinterest6_variant.jpg', quality=93)
    return out


TASKS = {
    'b78e60': lambda: text_variant(
        'b78e60',
        [dict(box=(360, 430, 1200, 790),
              words=['WE DEFEND THE', 'STEEL', 'HAWKS'], min_area=120)],
        'blackopsone',
        erase_kw=dict(method='nn', nn_median=41), dilate=5,
        exclude=region_mask(*Image.open(BY['b78e60']['path']).size,
                            pred=lambda xx, yy: (xx > 500) & (xx < 850) & (yy > 762) & (yy < 1120))),
    'pinterest3': lambda: text_variant(
        'pinterest3',
        [dict(box=(0, 60, 736, 400), words=['DENIM'], pad=30, min_area=400,
              # 原 UPCY 的白色 Y 字只有下半截描边落在暗掩膜里，且自成一条"行"→ 主检测只取
              # 第一条行带，Y 的下摆会被漏 → 擦除后 DENIM 的 M 下方留一小段旧字。
              extra=[(430, 312, 660, 400)])],
        'denim', dilate=22),
    'pinterest6': do_pinterest6,
    '6978': do_6978,
    'pinterest4': do_pinterest4,
}

if __name__ == '__main__':
    ids = sys.argv[1:] or list(TASKS)
    for i in ids:
        print(f'===== {i} =====')
        TASKS[i]()
    print('[DONE] ->', OUT)
