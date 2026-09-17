"""p4 树分割 v345c：用**树干**定位每棵树，再按"最近树干"把墨迹 Voronoi 划分。

为什么必须这样（第 14 轮真正根因）：
  · 旧掩膜 = 原墨迹 dilate1 → **新剪影只能是原剪影的子集** → 剪影数学上不可能变
    （这才是"裂变没看到明显变化"的根因，不是模型不行）。
  · 要让剪影真变，掩膜必须是**实心包络**（比原墨迹大一圈）。
  · 但简单 dilate20+closing30 的实心掩膜 = 87% 图幅 → 又退回"整图降采样 → 发虚"。
  ⇒ 唯一出路：**逐棵树**给实心包络，每棵树单独裁块、上采样到 1024 重生（= 蝙蝠的尺度）。

树干检测：调色板里树干有两种画法（连续竖线+横环纹 / 纯横纹梯子），所以
`closing(vline25)` 先把断纹桥接成连续竖线，再 `opening(vline60)` 取出长竖结构。
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

J = ROOT / "jobs" / "v345_p4tree"
J.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")


def _vline(n):
    return np.ones((n, 1), bool)


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def keep_big(m, min_px):
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return m, lab, n
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    k = np.zeros(n + 1, bool)
    k[1:] = sz >= min_px
    return k[lab], lab, n


def trunks(ink, H, W, close_n=25, open_n=60):
    c = ndi.binary_closing(ink, structure=_vline(close_n))
    o = ndi.binary_opening(c, structure=_vline(open_n))
    o = ndi.binary_opening(o, structure=np.ones((3, 3), bool))
    o, lab, n = keep_big(o, 40)
    segs = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        segs.append(dict(x=float(xs.mean()), y_top=float(ys.min()), y_bot=float(ys.max()),
                         h=float(ys.max() - ys.min() + 1), mask=(lab == i)))
    segs.sort(key=lambda s: -s["h"])
    return o, segs


def partition(ink, segs, H, W):
    """每个墨迹像素判给最近的树干（距离到树干的竖直段）。"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cost = np.full((len(segs), H, W), 1e18, np.float32)
    for i, s in enumerate(segs):
        dx = xx - s["x"]
        dy = np.clip(yy, s["y_top"], s["y_bot"]) - yy
        cost[i] = dx * dx + dy * dy
    win = cost.argmin(0)
    return win


def main():
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    a = np.asarray(img)
    ink = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    for cn, on in ((25, 60), (35, 90), (20, 45)):
        o, segs = trunks(ink, H, W, cn, on)
        hs = [s["h"] for s in segs]
        print(f"[vline close={cn} open={on}] 树干={len(segs)} 覆盖={100*o.mean():.2f}% "
              f"高度 min={min(hs) if hs else 0:.0f} max={max(hs) if hs else 0:.0f}")
    o, segs = trunks(ink, H, W, 25, 60)
    # 过滤：树干太长（>0.6H = 多棵树连在一起）或太短
    segs = [s for s in segs if 0.06 * H < s["h"] < 0.62 * H]
    print(f"[filter] 有效树干={len(segs)}")
    win = partition(ink, segs, H, W)
    # 每棵树：墨迹归属 + 包络
    envs, stats = [], []
    for i, s in enumerate(segs):
        own = (win == i) & ink
        if own.sum() < 400:
            continue
        ys, xs = np.where(own)
        e = ndi.binary_closing(ndi.binary_dilation(own, structure=_disk(9)),
                               structure=_disk(14))
        e = ndi.binary_fill_holes(e)
        e, _, _ = keep_big(e, 900)
        envs.append((s, own, e))
        stats.append((int(own.sum()), s["x"], s["y_top"], s["y_bot"],
                      xs.max() - xs.min(), ys.max() - ys.min()))
    print(f"[trees] 可用={len(envs)}")
    for t in sorted(stats, reverse=True)[:12]:
        print(f"   墨={t[0]:>6d} 干x={t[1]:>6.0f} y={t[2]:>5.0f}-{t[3]:<5.0f} own_bbox={t[4]}x{t[5]}")
    # 上色叠加
    rng = np.random.default_rng(3)
    ov = a.copy()
    for k, (s, own, e) in enumerate(envs):
        col = rng.integers(60, 255, 3).astype(np.float32)
        ov[e] = ov[e] * 0.45 + col * 0.55
        ov[own] = np.array([255, 255, 255], np.float32)
    Image.fromarray(ov.astype(np.uint8)).resize((W // 2, H // 2),
                                                Image.LANCZOS).save(
        str(J / "_seg_ov.jpg"), quality=90)
    print("saved", J / "_seg_ov.jpg")


if __name__ == "__main__":
    main()
