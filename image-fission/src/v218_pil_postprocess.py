"""
v218 — bat_logo PIL 后处理（基于 v215，彻底消除底部吊坠）

v217 软 mask 没去干净——水滴形状还在。
v218 用硬覆盖：
  1. 找最大连通域（v215 底部最黑区域=水滴形状）
  2. 扩大 ellipse mask（覆盖水滴+挂绳+晕染，rx*=1.5, ry*=1.5）
  3. 硬覆盖 BG_PURPLE（不混 alpha，1.0 强度）
  4. 边缘 5 像素 Gaussian blur 软化（避免硬边）
  5. 噪声 + USM（统一纹理）
"""
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path

PROJECT = Path("E:/Desktop/双接口/image-fission")
JOB = PROJECT / "jobs" / "smoke_v218"
JOB.mkdir(parents=True, exist_ok=True)

REF = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
SRC = PROJECT / "jobs" / "smoke_v215" / "v215_bat_logo.jpg"


def sample_bg_purple(img_bgr, n_samples=200):
    h, w = img_bgr.shape[:2]
    patches = [
        img_bgr[20:80, 20:80],
        img_bgr[20:80, w-80:w-20],
        img_bgr[h-80:h-20, 20:80],
        img_bgr[h-80:h-20, w-80:w-20],
    ]
    samples = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    bincount = {}
    for px in samples[::max(1, len(samples) // n_samples)]:
        key = tuple((px // 8) * 8)
        bincount[key] = bincount.get(key, 0) + 1
    return max(bincount, key=bincount.get)


def find_bottom_dark_region(img_bgr, bottom_band=(0.72, 0.97)):
    h, w = img_bgr.shape[:2]
    y0 = int(h * bottom_band[0])
    y1 = int(h * bottom_band[1])
    band = img_bgr[y0:y1, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    mask = (gray < 35).astype(np.uint8) * 255
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    x, y, ww, hh, area = stats[largest_idx]
    if area < 50:
        return None
    return (x, y0 + y, ww, hh), area


def main():
    pil = Image.open(SRC).convert("RGB")
    img = np.array(pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    print(f"[v218] v215 size: {w}x{h}")

    bg = sample_bg_purple(img_bgr)
    print(f"[v218] sampled BG_PURPLE = {bg}")

    res = find_bottom_dark_region(img_bgr)
    if res is None:
        print("[v218] no bottom dark region, skip")
        out = pil
    else:
        (rx, ry, rw, rh), area = res
        print(f"[v218] found bottom dark region at ({rx},{ry}) {rw}x{rh}, area={area}")

        # 硬覆盖 ellipse mask（rx/ry 扩大 1.5× 包括晕染区）
        cx = rx + rw // 2
        cy = ry + rh // 2
        rx_r = int(rw * 0.85)
        ry_r = int(rh * 1.20)  # 高度扩大更多（包含上方连接的线）

        # mask: 椭圆内 = 1, 外 = 0
        yy, xx = np.mgrid[0:h, 0:w]
        norm = ((xx - cx) / max(1, rx_r)) ** 2 + ((yy - cy) / max(1, ry_r)) ** 2
        hard_mask = (norm < 1.0).astype(np.float32)
        # 边缘软化（5px Gaussian）
        soft_mask = cv2.GaussianBlur(hard_mask, (11, 11), 3)
        soft_mask_3 = soft_mask[..., None]

        # 硬覆盖 BG_PURPLE（mask 内基本是 BG）
        bg_color = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
        new_img = img_bgr.astype(np.float32)
        new_img = new_img * (1 - soft_mask_3) + bg_color * soft_mask_3

        # 加 noise 模拟原图纹理
        noise = np.random.normal(0, 2.5, img_bgr.shape).astype(np.float32)
        new_img = new_img + noise * soft_mask_3  # 只在改的区域内加噪声

        img_bgr = np.clip(new_img, 0, 255).astype(np.uint8)
        out = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    out = out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=2))

    out_final = JOB / "v218_bat_logo.jpg"
    out.save(str(out_final), quality=95)
    print(f"[v218] saved {out_final} ({out_final.stat().st_size//1024} KB)")

    # 对照拼图
    ref = Image.open(REF).convert("RGB")
    if ref.size != out.size:
        out = out.resize(ref.size)
    cmp_path = JOB / "_compare_v218.jpg"
    cmp_img = Image.new("RGB", (ref.width * 2 + 30, ref.height), "white")
    cmp_img.paste(ref, (0, 0))
    cmp_img.paste(out, (ref.width + 30, 0))
    cmp_img.save(str(cmp_path), quality=95)
    print(f"[v218] saved compare {cmp_path}")


if __name__ == "__main__":
    main()
