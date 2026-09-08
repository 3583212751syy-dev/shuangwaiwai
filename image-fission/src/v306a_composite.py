# -*- coding: utf-8 -*-
"""v306A 混合管线 v2：
   1) plate 章底圆内重建（归一化卷积低频场 + 花瓣保护 + 黑尾条 inpaint）
   2) AI 生成的独立蝙蝠主体 -> 抠出 -> 贴回干净 plate
   3) final 的盘内区（圆 R306 ∪ 旧蝙蝠/残留差集）替换为 comp，环/大理石/文字零风险"""
import numpy as np, cv2, os, sys
from PIL import Image

JOBS = r"E:\Desktop\双接口\image-fission\jobs\v306"
SRC = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
CX, CY = 776, 734            # 盘/环圆心
MED_C = (772.8, 723.5)       # 章底圆心
MED_R = 224                  # 章底圆半径
MED_COL = np.array([94, 22, 122], np.float32)
SWAP_R = 306
PETAL_Y0 = 785               # 花瓣区上界（保护，不参与场填充）

TARGET_H = {"up": 452, "spread": 400, "fold": 452}
TARGET_CY = {"up": 750, "spread": 735, "fold": 758}


def rebuild_plate(plate):
    """章底圆内重建：干净紫像素低频场填充，花瓣与其上的黑尾条残留单独处理"""
    p = plate.astype(np.float32)
    H, W = p.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r_med = np.hypot(xx - MED_C[0], yy - MED_C[1])
    inside = r_med < (MED_R - 4)

    dist_med = np.linalg.norm(p - MED_COL[None, None], axis=2)

    # 归一化卷积低频场：仅用干净紫像素作源，先验拉向 MED_COL 防稀疏区漂移。
    # 章底圆内全量填充（旧内容=翼残留/黑尾条一并清零），花瓣稍后从源图贴回
    clean_src = (inside & (dist_med < 25)).astype(np.float32)
    src_rgb = p * clean_src[..., None]
    num = cv2.GaussianBlur(src_rgb, (61, 61), 0)
    den = cv2.GaussianBlur(clean_src, (61, 61), 0)
    prior = 0.12
    field = (num + MED_COL[None, None] * prior) / (den[..., None] + prior)

    fill = inside.astype(np.float32)
    fill = cv2.GaussianBlur(fill, (7, 7), 2.0)          # 边界羽化
    out = p * (1 - fill[..., None]) + field * fill[..., None]

    # 花瓣区从源图重建：章底色圆瓣（dist<32）+ 近贴浅粉描边（maxc>150 且不在蝙蝠暗体上）
    src = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)
    d_src = np.linalg.norm(src - MED_COL[None, None], axis=2)
    src_maxc = src.max(axis=2)
    src_dark = cv2.dilate(((src_maxc < 90).astype(np.uint8)) * 255, np.ones((9, 9), np.uint8)) > 0
    zone = (yy >= 690) & (r_med < 235)
    lobe = zone & (d_src < 32)
    rim = zone & (src_maxc > 150) & ~src_dark & (cv2.dilate(lobe.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0)
    layer = ((lobe | rim).astype(np.uint8) * 255)
    layer = cv2.morphologyEx(layer, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 只留大 CC（双瓣+圆底），去小碎屑
    ln, llab, lstats, _ = cv2.connectedComponentsWithStats(layer, 8)
    for i in range(1, ln):
        if lstats[i, 4] < 800:
            layer[llab == i] = 0
    la = cv2.GaussianBlur(layer, (5, 5), 1.5).astype(np.float32)[..., None] / 255.0
    out = out * (1 - la) + src * la
    return np.clip(out, 0, 255).astype(np.uint8)


def extract_bat(gen_path):
    """从生成图抠蝙蝠：色距 mask + 最大 CC + 封闭背景口袋剔除"""
    img = np.array(Image.open(gen_path).convert("RGB")).astype(np.float32)
    corner = np.concatenate([img[:60, :60].reshape(-1, 3), img[:60, -60:].reshape(-1, 3),
                             img[-120:-40, :60].reshape(-1, 3)])
    bg = np.median(corner, axis=0)
    dist_bg = np.linalg.norm(img - bg[None, None], axis=2)

    m = (dist_bg > 30).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    keep[lab == 1 + int(np.argmax(stats[1:, 4]))] = 255

    # 封闭背景口袋剔除（不触边、无版画暗纹理的平滑低色距区）
    dark = (img.max(axis=2) < 60).astype(np.uint8)
    cand = (dist_bg < 70).astype(np.uint8)
    cn, clab, cstats, _ = cv2.connectedComponentsWithStats(cand, 8)
    Hg, Wg = cand.shape
    for i in range(1, cn):
        x, y, w, h, area = cstats[i]
        if area < 1500 or x == 0 or y == 0 or x + w >= Wg or y + h >= Hg:
            continue
        cm = clab == i
        if dark[cm].mean() < 0.03:
            keep[cm] = 0

    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    soft = cv2.GaussianBlur(keep, (5, 5), 1.5).astype(np.float32) / 255.0
    img[dist_bg <= 30] = MED_COL           # 背景重着色，孔洞融合
    return img, soft, keep, bg


def composite(pose, gen_path, final_path, out_path, plate_clean, scale=1.0):
    final = np.array(Image.open(os.path.join(JOBS, final_path)).convert("RGB"))
    img, soft, keep, bg = extract_bat(gen_path)

    ys, xs = np.nonzero(keep)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, crop_soft = img[y0:y1, x0:x1], soft[y0:y1, x0:x1]
    th = TARGET_H[pose] * scale
    s = th / crop.shape[0]
    nw = max(1, int(round(crop.shape[1] * s)))
    crop = cv2.resize(crop, (nw, int(round(th))), interpolation=cv2.INTER_LANCZOS4)
    crop_soft = cv2.resize(crop_soft, (nw, crop.shape[0]), interpolation=cv2.INTER_LINEAR)

    ph, pw = crop.shape[:2]
    px, py = CX - pw // 2, int(TARGET_CY[pose] - ph * 0.52)
    comp = plate_clean.astype(np.float32).copy()
    region = comp[py:py + ph, px:px + pw]
    a = crop_soft[..., None]
    comp[py:py + ph, px:px + pw] = region * (1 - a) + crop * a

    # 盘内替换：圆 R340（底部 y<1005 限位保护大字；EST./1862 文字矩形保护）。
    # comp 与 final 在环/大理石处像素一致（plate 未重建那些区域），扩大圆安全
    H, W = final.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r_map = np.hypot(xx - CX, yy - CY)
    circ = (r_map < 340) & (yy < 1005)
    protect = ((xx > 415) & (xx < 550) & (yy > 720) & (yy < 805)) | \
              ((xx > 1010) & (xx < 1135) & (yy > 720) & (yy < 805))
    dm = (circ & ~protect).astype(np.float32)
    dm = np.clip(cv2.GaussianBlur(dm, (5, 5), 1.2), 0, 1)[..., None]
    out = final.astype(np.float32) * (1 - dm) + comp * dm

    # EST./1862 字形回贴：统一从 up 版 final 捐赠（三版文字层像素相同，
    # up 版该区无旧翼残块；只贴小 CC 字形，大 CC 拒贴）
    donor = np.array(Image.open(os.path.join(JOBS, "v306_up_final.png")).convert("RGB"))
    for (x0, x1, y0, y1) in [(415, 585, 660, 875), (1000, 1185, 660, 880)]:
        reg = donor[y0:y1, x0:x1]
        tmask = ((reg.max(axis=2) < 100).astype(np.uint8)) * 255
        tn, tl, ts, _ = cv2.connectedComponentsWithStats(tmask, 8)
        keep_t = np.zeros_like(tmask)
        for i in range(1, tn):
            if ts[i, 4] < 1500 and ts[i, 2] < 130 and ts[i, 3] < 130:
                keep_t[tl == i] = 255
        keep_t = cv2.dilate(keep_t, np.ones((3, 3), np.uint8))
        ta = cv2.GaussianBlur(keep_t, (3, 3), 0.8).astype(np.float32)[..., None] / 255.0
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - ta) + reg.astype(np.float32) * ta

    out = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(JOBS, out_path))
    print(f"  {pose}: gen bg={bg.astype(int)} bat {nw}x{crop.shape[0]} at ({px},{py}) -> {out_path}")


if __name__ == "__main__":
    plate = np.array(Image.open(os.path.join(JOBS, "_chk", "plate.png")).convert("RGB"))
    plate_clean = rebuild_plate(plate)
    Image.fromarray(plate_clean).save(os.path.join(JOBS, "_chk", "plateA.png"))
    print("plate rebuilt -> _chk/plateA.png")
    composite(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], plate_clean,
              scale=float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
