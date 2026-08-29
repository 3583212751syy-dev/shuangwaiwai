"""v139 自检 v2：灰度皮尔逊相关(结构是否保留) + 红/白/黑/杂色占比(配色是否锁住)。
用法：python verify_v139.py <orig> <gen>"""
import sys
import numpy as np
from PIL import Image

SIZE = (512, 512)


def load_gray(p):
    return np.asarray(Image.open(p).convert("L").resize(SIZE), dtype=np.float64)


def pearson(a, b):
    a = a.flatten(); b = b.flatten()
    am = a - a.mean(); bm = b - b.mean()
    den = np.sqrt((am ** 2).sum() * (bm ** 2).sum())
    if den == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float((am * bm).sum() / den)


def color_profile(p, n=200):
    im = Image.open(p).convert("RGB").resize((n, n))
    arr = np.asarray(im, dtype=np.float64)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    flat = arr.reshape(-1, 3)
    R, G, B = flat[:, 0], flat[:, 1], flat[:, 2]
    total = len(R)
    red = np.sum((R > 120) & (R > 1.4 * G) & (R > 1.4 * B)) / total
    white = np.sum((R > 200) & (G > 200) & (B > 200)) / total
    black = np.sum((R < 30) & (G < 30) & (B < 30)) / total
    # 非红/白/黑 的“杂色”占比
    other = 1.0 - (red + white + black)
    return dict(red=red, white=white, black=black, other=other)


def main():
    if len(sys.argv) < 3:
        print("usage: verify_v139.py <orig> <gen>"); return
    orig, gen = sys.argv[1], sys.argv[2]
    ga = load_gray(orig); gb = load_gray(gen)
    corr = pearson(ga, gb)
    po = color_profile(orig); pg = color_profile(gen)
    print(f"灰度结构相关(corr)={corr:.3f}  (≈1=照抄, 0.5~0.85=同源裂变, <0.4=结构崩)")
    print(f"原图  红={po['red']*100:4.1f}% 白={po['white']*100:4.1f}% 黑={po['black']*100:4.1f}% 杂={po['other']*100:4.1f}%")
    print(f"裂变  红={pg['red']*100:4.1f}% 白={pg['white']*100:4.1f}% 黑={pg['black']*100:4.1f}% 杂={pg['other']*100:4.1f}%")
    print("判定: 红白黑占比应与原图接近(锁色), 杂色<15%; corr∈[0.5,0.85]为合格裂变")


if __name__ == "__main__":
    main()
