"""v347: p4 定稿路线 = **强幅度逐树姿态平滑场**（tree_wind 加强版）。

为什么放弃"逐树换位/逐树仿射"（v345f/v346 实测结论）：
  这张图的棕榈**在墨迹上互相连通**（r=2 膨胀就只剩 20 个大组、最大组占全墨 42%），
  所以：
    · 用树干 Voronoi 取"树" → 逐棵渲染发现大量"树干环纹条""树冠碎片"被当成独立的树
      （第 6 行几乎全是横纹条）→ 搬动后横纹条散落全图 = 悬空孤块；
    · 用连通域取"树" → 大域里常含 2-3 棵（相邻树冠相接）；
    · 用"bbox 内连通域" → 域总面积可达自身的 51 倍 → 把邻居整片卷进来 = 深色糊团。
  ⇒ **逐树切分在这张图上不可靠**，任何"切分+搬动"都会产生碎块/空洞/糊团。

定稿：不做切分，改**全局平滑形变场**（tree_wind 加强）——只做重采样：
  · 线宽/硬边/连通性**零损失**（不新建、不删除任何笔画）；
  · 天然不会出现碎块、空洞、糊团（整层一起变形）；
  · 四个逐树姿态维度（高度/倾斜/干弯/冠幅）由只依赖 (x,y) 的平滑场驱动
    （波长 ≫ 单棵树 → 树内一致、邻树不同，绝不撕裂），把幅度加大到肉眼一眼可辨。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp            # noqa: E402

OUT = ROOT / "jobs" / "v347_p4pose"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
INK = np.array([12.0, 10.0, 9.0], np.float32)

VARIANTS = [
    ("V1", dict(amp=0.085, h_var=0.25, lean_var=0.10, bow_var=0.035, w_var=0.0,
                uni_scale=True)),
    ("V2", dict(amp=0.100, h_var=0.32, lean_var=0.14, bow_var=0.050, w_var=0.0,
                uni_scale=True)),
]


def main():
    only = sys.argv[1:] or None
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    ink = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    wr = ROOT / "jobs" / "router_out_v329" / "_p4_warped.jpg"
    base = np.asarray(Image.open(wr).convert("RGB").resize((W, H), Image.LANCZOS),
                      np.float32)
    rows = []
    for name, p in VARIANTS:
        if only and name not in only:
            continue
        tw, alw = cpp.tree_wind(img, ink, dilate=1, seed=21,
                                lam=2.40, phase=0.55, power=1.35, **p)
        _twa = np.asarray(tw, np.float32)
        a3 = alw[..., None]
        out = base * (1.0 - a3) + _twa * a3
        im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
        im.save(str(OUT / f"{name}.jpg"), quality=95)
        mv = cpp.tree_ink_mask(im, lum_thr=55., sat_thr=14., min_px=25, thin_only=False)
        from skimage.morphology import skeletonize
        edt = ndi.distance_transform_edt(mv)
        wpx = 2 * edt[skeletonize(mv)]
        lab, k = ndi.label(mv, np.ones((3, 3), bool))
        sz = np.bincount(lab.ravel())[1:] if k else np.array([0])
        iou = (ink & mv).sum() / max(1, (ink | mv).sum())
        print(f"[{name}] 墨={100*mv.mean():.2f}% (原 {100*ink.mean():.2f}%) IoU={iou:.3f} "
              f"笔宽median={np.median(wpx):.1f}px 块={k}(>500 {int((sz>500).sum())}) "
              f"碎点<30px={int((sz<30).sum())}")
        rows.append((name, im))
    # 对照板：原图 + 各档（下半部树区放大）
    bo = (0, int(H * 0.42), W, H)
    w = 560
    panels = [img.crop(bo), *[r.crop(bo) for _, r in rows]]
    fit = [p.resize((w, int(p.size[1] * w / p.size[0])), Image.LANCZOS) for p in panels]
    hh = fit[0].size[1]
    sheet = Image.new("RGB", (w * len(fit) + 8 * (len(fit) + 1), hh + 8), (25, 25, 25))
    for i, f in enumerate(fit):
        sheet.paste(f, (8 + i * (w + 8), 4))
    sheet.save(str(OUT / "_sheet.jpg"), quality=93)
    print("[sheet]", OUT / "_sheet.jpg")


if __name__ == "__main__":
    main()
