"""
v219 — bat_logo PIL 后处理（v215 底部装饰整块彻底去掉）

v218 软 mask 只覆盖水滴 + 菱形，没覆盖顶部细线 → 形状残留。
v219 一次性彻底去掉整个底部装饰：
  1. 检测 v215 底部 50% 区域所有 mask（黑色 + 高对比度边缘）
  2. 用连通域找最底部大型深色形状
  3. 大椭圆 mask 从 y_centre 向上扩展，覆盖底部所有装饰（菱形 + 水滴 + 顶部连接线）
  4. 边缘 10px Gaussian blur 软过渡
  5. 填 BG_PURPLE + 微噪声
"""
import numpy as np
import cv2
from PIL import Image, ImageFilter
from pathlib import Path

PROJECT = Path("E:/Desktop/双接口/image-fission")
JOB = PROJECT / "jobs" / "smoke_v219"
JOB.mkdir(parents=True, exist_ok=True)

REF = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
SRC = PROJECT / "jobs" / "smoke_v215" / "v215_bat_logo.jpg"


def sample_bg_purple(img_bgr):
    h, w = img_bgr.shape[:2]
    patches = [
        img_bgr[20:80, 20:80], img_bgr[20:80, w-80:w-20],
        img_bgr[h-80:h-20, 20:80], img_bgr[h-80:h-20, w-80:w-20],
    ]
    samples = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    bincount = {}
    for px in samples[::max(1, len(samples) // 200)]:
        key = tuple((px // 8) * 8)
        bincount[key] = bincount.get(key, 0) + 1
    return max(bincount, key=bincount.get)


def find_bottom_decoration(img_bgr):
    """找 v215 底部 0.50-0.96 区域所有 mask 元素，返回最大深色形状 bbox。"""
    h, w = img_bgr.shape[:2]
    y0 = int(h * 0.50)
    y1 = int(h * 0.96)
    band = img_bgr[y0:y1, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    # 黑色或接近黑色
    mask = (gray < 50).astype(np.uint8) * 255
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels < 2:
        return None
    # 找最大连通域（包括 v215 的菱形+水滴整体）
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    x, y, ww, hh, area = stats[largest_idx]
    if area < 100:
        return None
    return (x, y0 + y, ww, hh), area


def main():
    pil = Image.open(SRC).convert("RGB")
    img = np.array(pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    print(f"[v219] v215 size: {w}x{h}")

    bg = sample_bg_purple(img_bgr)
    print(f"[v219] sampled BG_PURPLE = {bg}")

    res = find_bottom_decoration(img_bgr)
    if res is None:
        print("[v219] no bottom decoration, skip")
        out = pil
    else:
        (rx, ry, rw, rh), area = res
        print(f"[v219] found bottom decoration bbox at ({rx},{ry}) {rw}x{rh}, area={area}")

        # 大椭圆 mask 覆盖整个底部装饰（菱形+水滴+顶部细线）
        cx = rx + rw // 2
        cy = ry + rh // 2
        # 椭圆半径：宽 * 0.9, 高 * 0.7（高略小因为是垂直形状）；额外再向上扩 rh * 0.3 cover 顶部细线
        rx_r = int(rw * 0.95)
        ry_r = int(rh * 0.70)

        # 软 mask
        yy, xx = np.mgrid[0:h, 0:w]
        norm = ((xx - cx) / max(1, rx_r)) ** 2 + ((yy - cy) / max(1, ry_r)) ** 2
        hard_mask = (norm < 1.0).astype(np.float32)
        soft_mask = cv2.GaussianBlur(hard_mask, (15, 15), 4)
        soft_mask_3 = soft_mask[..., None]

        bg_color = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
        new_img = img_bgr.astype(np.float32)
        new_img = new_img * (1 - soft_mask_3) + bg_color * soft_mask_3

        noise = np.random.normal(0, 2.5, img_bgr.shape).astype(np.float32)
        new_img = new_img + noise * soft_mask_3

        img_bgr = np.clip(new_img, 0, 255).astype(np.uint8)
        out = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    out = out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=2))

    out_final = JOB / "v219_bat_logo.jpg"
    out.save(str(out_final), quality=95)
    print(f"[v219] saved {out_final} ({out_final.stat().st_size//1024} KB)")

    ref = Image.open(REF).convert("RGB")
    if ref.size != out.size:
        out = out.resize(ref.size)
    cmp_path = JOB / "_compare_v219.jpg"
    cmp_img = Image.new("RGB", (ref.width * 2 + 30, ref.height), "white")
    cmp_img.paste(ref, (0, 0))
    cmp_img.paste(out, (ref.width + 30, 0))
    cmp_img.save(str(cmp_path), quality=95)
    print(f"[v219] saved compare {cmp_path}")


if __name__ == "__main__":
    main()
