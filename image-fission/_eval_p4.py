# -*- coding: utf-8 -*-
"""p4 验收器：把"碎不碎"量化成可比数字。

用户第 19 轮否决原话：「碎成一块一块的效果图我不是严格明令禁止了吗」。
"一块块"有两个可测特征，必须同时盯：
  ① **悬浮孤块**：连通成分里"小而孤立"的块（面积 <400px 且离最近的大块 ≥25px）；
  ② **细缝被填死**：叶片之间的白色缝隙被 α 膝点膨胀糊住 → 原图的**孔洞**数量/面积塌陷。
本脚本同时输出这两项 + 基础形态（墨量/主块/笔宽），用 ORIG 当基准逐项对齐。
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
LUM = np.array([0.299, 0.587, 0.114], np.float32)
ORIG = "E:/Desktop/图裂变测试图/Pinterest (4).jpg"


def metrics(path, lum_thr=30.0, tag=None):
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    m = (a @ LUM) < lum_thr
    lab, n = ndi.label(m, np.ones((3, 3), bool))
    sz = np.bincount(lab.ravel()); sz[0] = 0
    main = sz >= 12                                  # 剔 JPEG 噪点（MEMORY：p4 判墨必须 lum<30）
    ncomp = int(main.sum())
    med = int(np.median(sz[main])) if ncomp else 0
    # 笔宽
    e = ndi.distance_transform_edt(m)
    sk = skeletonize(m)
    w = float(2.0 * e[sk].mean()) if sk.any() else 0.0
    # 孔洞（叶片间的白缝）
    holes = ndi.binary_fill_holes(m) & ~m
    hl, hn = ndi.label(holes, np.ones((3, 3), bool))
    hsz = np.bincount(hl.ravel()) if hn else np.array([0])
    hsz[0] = 0
    hk = hsz >= 8
    # 悬浮孤块：小块(<400px) 且 距最近大块(>=2000px) 的间距 >=25px
    big = np.zeros_like(m)
    idx_big = np.where(sz >= 2000)[0]
    if len(idx_big):
        big = np.isin(lab, idx_big)
    small = m & ~big & (sz[lab] < 400) & (lab > 0)
    if small.any() and big.any():
        d = ndi.distance_transform_edt(~big)
        orph = small & (d >= 25.0)
    else:
        orph = np.zeros_like(m)
    ol, on = ndi.label(orph, np.ones((3, 3), bool))
    osz = np.bincount(ol.ravel()) if on else np.array([0])
    osz[0] = 0
    res = dict(tag=tag or Path(path).name, ink=100 * m.mean(), ncomp=ncomp, med=med,
               w=w, holes=int(hk.sum()), hole_area=100 * holes.mean(),
               orph_n=int((osz >= 30).sum()), orph_area=100 * orph.mean())
    print(f"[{res['tag']:<22}] 墨={res['ink']:5.2f}%  主块={res['ncomp']:4d}  "
          f"中位={res['med']:5d}  笔宽={res['w']:4.2f}  "
          f"孔洞={res['holes']:5d}/面积{res['hole_area']:5.3f}%  "
          f"孤块={res['orph_n']:3d}/面积{res['orph_area']:4.3f}%")
    return res


if __name__ == "__main__":
    metrics(ORIG, tag="ORIG 基准")
    for p in sys.argv[1:]:
        metrics(p, tag=Path(p).stem)
