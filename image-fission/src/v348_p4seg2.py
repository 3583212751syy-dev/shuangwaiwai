"""v348: p4 树分段 v2 —— 「大域=完整棕榈 + 其余小域按 bbox 归属」。

v345c/v346 的分段为什么错（本轮逐棵渲染才看清）：
  · `trunks()` 在 76 个竖结构里按高度筛出 34 个当"树"，但这 34 个里大量是
    **树干环纹条**或**树冠碎片**（把 34 棵单独渲染：第 6 行几乎全是横纹条）。
    搬动它们 → 横纹条散落全图（肉眼=悬空细碎块），这就是 v346 的丑根因。
  · 用"墨迹连通域"当树 → 这图的墨**互相连通**（膨胀 r=2 就只剩 20 个大组、
    最大组占全墨 42%），大域里常含 2-3 棵。
  · 用"bbox 内连通域"当树 → 域总面积可达自身的 51 倍 → 把邻居整片卷进来。

正解（本文件）：
  ① 连通域标记，取**大面积域（≥ AREA_MIN）当"树主域"**——逐域渲染确认这些就是
     完整棕榈（干+环+冠齐全）；
  ② 其余小域（环纹条/碎冠/草痕）按"质心是否落在某棵树主域 bbox 外扩 PAD 内"
     归并到那棵树 → 树形完整、不牵连邻居；
  ③ 归并不上的小域 = 就地装饰元素，**留在原位**（同位硬规则）。
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

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
AREA_MIN = 4800
PAD = 26


def segment(img):
    ink = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    lab, k = ndi.label(ink, structure=np.ones((3, 3), bool))
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    objs = ndi.find_objects(lab)
    main = [i for i in range(1, k + 1) if sz[i] >= AREA_MIN]
    main.sort(key=lambda i: -sz[i])
    # 棕榈特征筛选：真树 = **上宽下窄**（冠幅 > 干区宽度）。
    # 纯"树干环纹条"虽然面积够，但上下等宽 → 被判掉（v346 的碎块根因就是把这些
    # 环纹条当成独立的树搬走了）。判据用"冠区最大行宽 / 干区最大行宽"。
    def palm_like(i):
        sl = objs[i - 1]
        m = lab[sl] == i
        h = m.shape[0]
        if h < 60:
            return False
        wr = m.sum(1)                            # 每行墨宽
        top = int(wr[:int(h * 0.45)].max())      # 冠区（上部 45%）
        bot = int(max(1, wr[int(h * 0.55):].max()))   # 干区（下部 55%）
        return (top / bot) >= 1.35

    trees, weak = [], []
    for i in main:
        (trees if palm_like(i) else weak).append(i)
    main = trees
    trees = []
    for i in main:
        sl = objs[i - 1]
        m = lab[sl] == i
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        trees.append(dict(ids=[i], m=m, bbox=(x0, y0, x1, y1), off=(y0, x0),
                          area=int(sz[i])))
    H, W = ink.shape
    # 不合格的大域（环纹条/碎冠）+ 小域 一并按 bbox 归属并入最近的真树
    rest = [i for i in range(1, k + 1) if sz[i] < AREA_MIN] + weak
    placed, free = 0, []
    for i in rest:
        sl = objs[i - 1]
        cy = (sl[0].start + sl[0].stop) * 0.5
        cx = (sl[1].start + sl[1].stop) * 0.5
        best, bd = None, 1e18
        for t in trees:
            bx0, by0, bx1, by1 = t["bbox"]
            if bx0 - PAD <= cx <= bx1 + PAD and by0 - PAD <= cy <= by1 + PAD:
                d = (cx - (bx0 + bx1) / 2) ** 2 + (cy - (by0 + by1) / 2) ** 2
                if d < bd:
                    best, bd = t, d
        if best is None:
            free.append(i)
            continue
        best["ids"].append(i)
        placed += 1
    # 重算每棵树的完整 mask
    for t in trees:
        y0, x0 = t["off"]
        Hm, Wm = t["m"].shape
        acc = t["m"]
        for i in t["ids"][1:]:
            sl = objs[i - 1]
            m = np.zeros((H, W), bool)
            m[sl] = (lab[sl] == i)
            sy, sx = sl[0].start - y0, sl[1].start - x0
            need = (max(0, -sy), max(0, -sx),
                    max(0, sy + m.shape[0] - Hm), max(0, sx + m.shape[1] - Wm))
        # 简化：直接在全局拼装（下面用全局索引重算，避免越界补丁）
        g = np.zeros((H, W), bool)
        for i in t["ids"]:
            sl = objs[i - 1]
            g[sl] |= (lab[sl] == i)
        ys, xs = np.where(g)
        t["mask"] = g
        t["bbox"] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        t["area"] = int(g.sum())
    print(f"[seg] 连通域={k} 树主域≥{AREA_MIN}px={len(main)} 归并小域={placed} "
          f"就地装饰域={len(free)}")
    return ink, trees, free, lab, objs, sz


def main():
    img = Image.open(SRC).convert("RGB")
    ink, trees, free, lab, objs, sz = segment(img)
    H, W = ink.shape
    # 渲染每棵树（白底黑形）+ 就地装饰
    cols, cell = 8, 165
    n = len(trees)
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * cell, cols * cell), np.uint8) + 255
    for j, t in enumerate(trees):
        x0, y0, x1, y1 = t["bbox"]
        m = t["mask"][y0:y1 + 1, x0:x1 + 1]
        im = Image.fromarray((~m).astype(np.uint8) * 255)
        s = min((cell - 12) / max(1, im.size[0]), (cell - 12) / max(1, im.size[1]))
        im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))),
                       Image.NEAREST)
        r_, c_ = divmod(j, cols)
        oy = r_ * cell + 6 + (cell - 12 - im.size[1]) // 2
        ox = c_ * cell + 6 + (cell - 12 - im.size[0]) // 2
        sheet[oy:oy + im.size[1], ox:ox + im.size[0]] = np.asarray(im)
    out = ROOT / "jobs" / "v348_p4seg2"
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(str(out / "_trees.png"))
    # 就地装饰单独一张
    g = np.zeros((H, W), bool)
    for i in free:
        sl = objs[i - 1]
        g[sl] |= (lab[sl] == i)
    Image.fromarray((~g).astype(np.uint8) * 255).save(str(out / "_free.png"))
    print(f"[seg] 树={n} 树面积合计={sum(t['area'] for t in trees)} 就地装饰={int(g.sum())} "
          f"墨总={int(ink.sum())}")
    print("[seg] 面积排序:", [t["area"] for t in trees][:25])


if __name__ == "__main__":
    main()
