"""
v217 — bat_logo PIL 后处理（基于 v215，去掉底部水滴吊坠）

v215 QC FAIL 但视觉合规；v216 QC PASS 但出可读英文 BACLRET/SMIRAICE。
最佳折中：不重跑 ComfyUI，直接 PIL 后处理 v215 去掉底部黑色水滴吊坠。

后处理步骤：
  1. cv2.HoughCircles + connectedComponents 找底部最深色区域（水滴形状）
  2. 椭圆 mask 覆盖该区域 + 周围晕染区
  3. 填 BG_PURPLE（采样原图徽章外的紫底色）
  4. 加轻 Gaussian noise 模拟原图纹理
  5. 轻 USM 锐化主轮廓

输出：jobs/smoke_v217/v217_bat_logo.jpg + _compare_v217.jpg
"""
import numpy as np
import cv2
from PIL import Image, ImageFilter
from pathlib import Path

PROJECT = Path("E:/Desktop/双接口/image-fission")
JOB = PROJECT / "jobs" / "smoke_v217"
JOB.mkdir(parents=True, exist_ok=True)

REF = PROJECT / "ComfyUI" / "input" / "test_6978fabda2cc99629fa9e81f802762d3.jpg"
SRC = PROJECT / "jobs" / "smoke_v215" / "v215_bat_logo.jpg"


def sample_bg_purple(img_bgr, n_samples=200):
    """从图像四角采样 BG_PURPLE（最常见底色）。"""
    h, w = img_bgr.shape[:2]
    patches = [
        img_bgr[20:80, 20:80],
        img_bgr[20:80, w-80:w-20],
        img_bgr[h-80:h-20, 20:80],
        img_bgr[h-80:h-20, w-80:w-20],
    ]
    samples = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    # 取众数（最频繁的颜色）
    bincount = {}
    for px in samples[::max(1, len(samples) // n_samples)]:
        key = tuple((px // 8) * 8)  # 量化到 8 灰度级
        bincount[key] = bincount.get(key, 0) + 1
    return max(bincount, key=bincount.get)


def find_bottom_dark_region(img_bgr, bottom_band=(0.78, 0.96)):
    """在图像底部 band 找最深的黑色区域（水滴吊坠）。"""
    h, w = img_bgr.shape[:2]
    y0 = int(h * bottom_band[0])
    y1 = int(h * bottom_band[1])
    band = img_bgr[y0:y1, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    # 黑色区域（亮度 < 30）
    mask = (gray < 30).astype(np.uint8) * 255
    # 找最大连通域
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels < 2:
        return None
    # 排除背景（labels=0），找最大连通域
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    x, y, ww, hh, area = stats[largest_idx]
    if area < 50:
        return None
    # 转回原图坐标
    return (x, y0 + y, ww, hh), area


def main():
    pil = Image.open(SRC).convert("RGB")
    img = np.array(pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    print(f"[v217] v215 size: {w}x{h}")

    # 1) 采样 BG_PURPLE
    bg = sample_bg_purple(img_bgr)
    print(f"[v217] sampled BG_PURPLE = {bg}")

    # 2) 找底部水滴区域
    res = find_bottom_dark_region(img_bgr)
    if res is None:
        print("[v217] no bottom dark region found, skip processing")
        out = pil
    else:
        (rx, ry, rw, rh), area = res
        print(f"[v217] found bottom dark region at ({rx},{ry}) {rw}x{rh}, area={area}")

        # 3) 椭圆 mask 覆盖（水滴形状 → 椭圆）
        cx = rx + rw // 2
        cy = ry + rh // 2
        # 椭圆半径：宽 * 0.7, 高 * 0.9（包含晕染区）
        rx_r = int(rw * 0.7)
        ry_r = int(rh * 0.9)
        # 稍微外扩以包括晕染
        expansion = max(rx_r, ry_r) * 0.3

        # 软 mask：椭圆中心 = 1.0，边缘 fade 到 0.0
        yy, xx = np.mgrid[0:h, 0:w]
        # 椭圆距离（标准化到 [-1, 1]，<1 在椭圆内）
        norm = ((xx - cx) / max(1, rx_r + expansion)) ** 2 + ((yy - cy) / max(1, ry_r + expansion)) ** 2
        soft_mask = np.clip(1.0 - norm, 0.0, 1.0)  # 在椭圆内 = 1, 边缘 = 0
        # mask 加 soft 边界（向外扩散 30 像素）
        soft_mask = cv2.GaussianBlur((soft_mask * 255).astype(np.uint8), (0, 0), 12) / 255.0
        soft_mask_3 = soft_mask[..., None]

        # 4) 填充 BG_PURPLE（按 mask 权重混合）
        bg_color = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
        new_img = img_bgr.astype(np.float32)
        new_img = new_img * (1 - soft_mask_3 * 0.92) + bg_color * (soft_mask_3 * 0.92)

        # 5) 加轻 Gaussian noise 模拟原图纹理
        noise = np.random.normal(0, 2.5, img_bgr.shape).astype(np.float32)
        new_img = new_img + noise

        img_bgr = np.clip(new_img, 0, 255).astype(np.uint8)
        out = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # 6) 轻 USM 锐化（降 v215 边缘感知）
    out = out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=2))

    out_final = JOB / "v217_bat_logo.jpg"
    out.save(str(out_final), quality=95)
    print(f"[v217] saved {out_final} ({out_final.stat().st_size//1024} KB)")

    # 对照拼图
    ref = Image.open(REF).convert("RGB")
    if ref.size != out.size:
        out = out.resize(ref.size)
    cmp_path = JOB / "_compare_v217.jpg"
    cmp_img = Image.new("RGB", (ref.width * 2 + 30, ref.height), "white")
    cmp_img.paste(ref, (0, 0))
    cmp_img.paste(out, (ref.width + 30, 0))
    cmp_img.save(str(cmp_path), quality=95)
    print(f"[v217] saved compare {cmp_path}")


if __name__ == "__main__":
    main()
