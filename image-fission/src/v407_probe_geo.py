# -*- coding: utf-8 -*-
"""v407_probe_geo.py —— 量测原图 p6 标题（MRCHOSR）的**排版位置 + 形状包络**

用户第26轮要求（原话）：
  「按照原图文本排版位置，大概形状构图去生成新的文本样子，
    不允许换位置生成，也不允许用色块背景遮盖原图内容」

=> 新文本必须落在**原位**，且**上下缘剖面（≈大致形状）**要跟原图一致。
所以先把原图标题的「每列上缘 y_top(x) / 下缘 y_bot(x)」量出来，
并给出低通平滑后的**包络**（去掉单字母尖刺、保留整体拱形/倾斜/收放）。

输出：
  jobs/_probe/v407_title_mask.png   标题墨迹掩膜可视化
  jobs/_probe/v407_envelope.png     剖面曲线 + 平滑包络可视化
  jobs/_probe/v407_envelope.npz     数值（ytop/ybot + 平滑版）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parent.parent
VIS = PROJ / "jobs" / "_probe"
VIS.mkdir(parents=True, exist_ok=True)
ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")

# 标题带：原图标题白墨 bbox y[88,1800]（见 make_v329 do_pinterest6 注释）
# 但标题真实下缘 ≈1490；鹰的白头从 y≳1470 起 → 用 1470 切，既保标题又躲开白头。
BAND_TOP = 0
BAND_BOT = 1900
TITLE_BOT = 1470


def title_mask(img: Image.Image) -> np.ndarray:
    """标题本体掩膜 = 白墨（亮 + 低饱和）∩ 标题带，**全部**面积≥40 的连通域。

    ⚠️ 别再要求「触顶」：擦字版要求触顶是因为它工作在 4x 下采样空间（阈值 90 ≈
    全分辨率 360）；全分辨率下标题只有 M 的尖刺到 y≈85，其余字母顶在 y≈130~300，
    触顶判据会把 RCHOSR 全丢掉（实测只剩 x[9,1435]）。
    这里改为：只取标题带（y < TITLE_BOT），保留所有像样的白簇。
    标题真实下缘 ≈1490（鹰白头 y≳1470 起 → 用 1470 切既保标题又躲白头）。
    """
    import scipy.ndimage as ndi
    a = np.asarray(img, np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = a.max(2) - a.min(2)
    ink = (lum > 170) & (sat < 50)
    ink[TITLE_BOT:, :] = False
    cl = ndi.binary_closing(ink, structure=_disk(3))
    lab, n = ndi.label(cl)
    keep = np.zeros_like(cl)
    for i in range(1, n + 1):
        m = lab == i
        if int(m.sum()) >= 40:
            keep |= m
    return keep & ink          # 收回到原始笔画（不保留闭运算的补洞）


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def profiles(mask: np.ndarray):
    """每列的上/下缘 y（无墨列返回 -1）。"""
    H, W = mask.shape
    xs = np.arange(W)
    ytop = np.full(W, -1, np.int32)
    ybot = np.full(W, -1, np.int32)
    any_col = mask.any(0)
    idx = np.where(any_col)[0]
    for x in idx:
        col = np.where(mask[:, x])[0]
        ytop[x] = col.min()
        ybot[x] = col.max()
    return ytop, ybot, idx


def smooth_envelope(ytop, ybot, idx, win=190):
    """低通平滑：去掉单字母尖刺，保留整体拱形/倾斜。

    对 ytop / ybot 在**有墨区间**上做滑动中位 + 均值，再向两端外推
    （题目只要求"大概形状"，不要把 M 的独苗尖刺搬过来）。
    """
    top = ytop[idx].astype(np.float64)
    bot = ybot[idx].astype(np.float64)
    k = max(3, int(win) | 1)

    def _ma(v, k):
        pad = k // 2
        vv = np.pad(v, pad, mode="edge")
        ker = np.ones(k) / k
        return np.convolve(vv, ker, mode="valid")

    st = _ma(_ma(top, k), k)      # 二次均值 ≈ 高斯低通
    sb = _ma(_ma(bot, k), k)
    return st, sb


def main() -> None:
    img = Image.open(ORIG).convert("RGB")
    W, H = img.size
    m = title_mask(img)
    ytop, ybot, idx = profiles(m)
    print(f"[orig] {W}x{H}  标题墨迹列数={len(idx)}  "
          f"x[{int(idx.min())},{int(idx.max())}]")
    ys, xs = np.where(m)
    print(f"[orig] 标题 bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")

    st, sb = smooth_envelope(ytop, ybot, idx, win=190)

    # 采样打印 12 个位置
    print("   x     ytop  ybot   | smooth_top smooth_bot  (thickness)")
    samp = np.linspace(0, len(idx) - 1, 12).astype(int)
    for s in samp:
        x = int(idx[s])
        print(f"  {x:5d}  {ytop[x]:5d} {ybot[x]:5d}   |   {st[s]:7.0f}   {sb[s]:7.0f}   "
              f"({sb[s]-st[s]:6.0f})")
    print(f"[env] smooth top 范围 [{st.min():.0f},{st.max():.0f}]  "
          f"bottom 范围 [{sb.min():.0f},{sb.max():.0f}]")
    # 倾斜：左端 vs 右端的上缘差
    print(f"[env] 上缘左端 {st[0]:.0f} → 右端 {st[-1]:.0f}  Δ={st[-1]-st[0]:+.0f}px "
          f"(>0 右低=右倾, <0 右高=右上扬)")

    # ---- 可视化：掩膜 ----
    vis = np.asarray(img).copy()
    vis[m] = [255, 40, 40]
    Image.fromarray(vis).save(VIS / "v407_title_mask.png")

    # ---- 可视化：剖面 ----
    cw, ch = 1200, 620
    pad_l, pad_t = 70, 40
    sx = (cw - pad_l - 30) / float(W)
    sy = (ch - pad_t - 40) / float(BAND_BOT)
    cv = Image.new("RGB", (cw, ch), (18, 18, 22))
    d = ImageDraw.Draw(cv)

    def P(x, y):
        return (pad_l + x * sx, pad_t + y * sy)

    d.line([P(0, 0), P(0, BAND_BOT)], fill=(70, 70, 80))
    d.line([P(0, 0), P(W, 0)], fill=(70, 70, 80))
    # 原图每列剖面（散点）
    for i, x in enumerate(idx):
        d.point(P(int(x), int(ytop[x])), fill=(90, 160, 255))
        d.point(P(int(x), int(ybot[x])), fill=(255, 160, 90))
    # 平滑包络（粗线）
    prev_t = prev_b = None
    for i, x in enumerate(idx):
        t = P(int(x), int(st[i]))
        b = P(int(x), int(sb[i]))
        if prev_t:
            d.line([prev_t, t], fill=(80, 255, 140), width=3)
            d.line([prev_b, b], fill=(255, 90, 200), width=3)
        prev_t, prev_b = t, b
    d.text((8, 8), "blue/orange = per-col orig  green/pink = smoothed envelope",
           fill=(230, 230, 230))
    cv.save(VIS / "v407_envelope.png")

    np.savez(VIS / "v407_envelope.npz",
             ytop=ytop, ybot=ybot, idx=idx, smooth_top=st, smooth_bot=sb,
             band_bot=BAND_BOT)
    print("[out] ->", VIS / "v407_title_mask.png", VIS / "v407_envelope.png",
          VIS / "v407_envelope.npz")


if __name__ == "__main__":
    main()
