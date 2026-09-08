#!/usr/bin/env python
# v312 text fission —— 方案 A（暴力矩形去字 + 相似词重写）
# 用于「无主体 / 统一背景」图（如 ARMED FORCES 迷彩）的文字裂变：
#   底图狗牌/链/迷彩花纹的变体由 mode3 双锁 AI 底图负责（v312b 已生成），
#   本脚本只负责把 AI 底图上糊掉的英文擦净、换上相似单词。
#
# 为什么暴力矩形：
#   AI 把字渲染成「中灰字 on 中灰迷彩」，局部对比度极低，PIL 邻域 inpaint
#   扩散不出干净底（四周也是同灰度迷彩）→ 永远擦不干净（v3/v4 实测）。
#   故放弃精细 ink mask：直接把字带整矩形填成「局部 camo 中值色」(羽化)，
#   字 100% 消失，再写新相似词。字带本就是文字区，平滑斑可接受。
#
# band 用相对比例（不写死分辨率），自动适配 768x1024 / 1556x2000。
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FEATHER = 12  # 矩形边缘羽化像素
CAMO_SRC_Y0 = 0.55  # 干净迷彩源区起点（字带之下）


def fill_band_texture(arr, band, feather=FEATHER):
    """纹理合成填字带：从下方干净迷彩采样等大矩形，替换字带。
    比 flat 色填充好 —— 保留迷彩纹理，矩形肉眼不可见。
    """
    y0, y1, x0, x1 = band
    bh, bw = y1 - y0, x1 - x0
    H, W = arr.shape[:2]
    # 干净迷彩源区（字带下方）
    src_y_min = int(CAMO_SRC_Y0 * H)
    src_y_max = H - bh - 5
    if src_y_max <= src_y_min:
        src_y = src_y_min
    else:
        # 随机偏移（避免三带采样同一段）
        np.random.seed(band[0] * 31 + band[2] * 7)
        src_y = src_y_min + int(np.random.randint(0, src_y_max - src_y_min))
    texture = arr[src_y:src_y + bh, x0:x1].copy()
    # 羽化 blend 边缘
    hh, ww = bh, bw
    ay = np.ones(hh); ax = np.ones(ww)
    for d in range(feather):
        a = (feather - d) / feather
        ay[d] = a; ay[hh - 1 - d] = a
        ax[d] = a; ax[ww - 1 - d] = a
    alpha = np.minimum.outer(ay, ax)[..., None]
    region = arr[y0:y1, x0:x1].astype(np.float32)
    blended = region * (1 - alpha) + texture.astype(np.float32) * alpha
    arr[y0:y1, x0:x1] = blended


def fill_band_bruteforce(arr, band, camo_ref, feather=FEATHER):
    """把字带整矩形填成 camo 参考色（羽化），字 100% 消失。
    camo_ref = 从字带外取的纯迷彩中值色（保证不是字色）。"""
    y0, y1, x0, x1 = band
    region = arr[y0:y1, x0:x1].astype(np.float32)
    fill = np.full_like(region, np.asarray(camo_ref, np.float32))
    hh, ww = y1 - y0, x1 - x0
    ay = np.ones(hh); ax = np.ones(ww)
    for d in range(feather):
        a = (feather - d) / feather
        ay[d] = a; ay[hh - 1 - d] = a
        ax[d] = a; ax[ww - 1 - d] = a
    alpha = np.minimum.outer(ay, ax)[..., None]
    blended = region * (1 - alpha) + fill * alpha
    arr[y0:y1, x0:x1] = blended


def measure(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def fit_font(draw, text, font_path, max_w, max_h, start=400, lo=18, step=3):
    size = start
    while size > lo:
        f = ImageFont.truetype(font_path, size)
        tw, th = measure(draw, text, f)
        if tw <= max_w and th <= max_h:
            return f
        size -= step
    return ImageFont.truetype(font_path, lo)


def draw_centered(draw, text, font, cx, cy, fill):
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = cx - tw // 2 - bb[0]
    y = cy - th // 2 - bb[1]
    draw.text((x, y), text, font=font, fill=fill)


def main():
    if len(sys.argv) < 3:
        print("usage: v312_text_fission.py <ai_base.jpg> <out.jpg> "
              "[--small TXT] [--big1 TXT] [--big2 TXT]")
        sys.exit(1)
    src_path, out_path = sys.argv[1], sys.argv[2]
    small_new = "WE HONOR OUR HEROES"
    big1_new = "BRAVE"
    big2_new = "LEGION"
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--small":
            small_new = args[i + 1]; i += 2
        elif args[i] == "--big1":
            big1_new = args[i + 1]; i += 2
        elif args[i] == "--big2":
            big2_new = args[i + 1]; i += 2
        else:
            i += 1

    src = Image.open(src_path).convert("RGB")
    W, H = src.size
    print(f"[src] {W}x{H}")
    arr = np.asarray(src, np.float32)

    # 字带按相对比例 —— AI mode3 渲染的字在 y 0.245~0.470（实测 AI 底图），
    # 不再按原图位置。原图小字+两行大字都在这个区间，分三段盖。
    small_band = (int(0.234 * H), int(0.286 * H), int(0.28 * W), int(0.72 * W))
    big1_band  = (int(0.286 * H), int(0.372 * H), int(0.24 * W), int(0.76 * W))
    big2_band  = (int(0.372 * H), int(0.466 * H), int(0.18 * W), int(0.82 * W))

    # 取 camo 参考色：字带外的纯迷彩区域 (y 0.08~0.18, x 0.10~0.90)
    # 排除暗像素（万一有暗斑）→ 拿真正的灰军迷彩中值
    ref_strip = arr[int(0.08 * H):int(0.18 * H), int(0.10 * W):int(0.90 * W)]
    flat = ref_strip.reshape(-1, 3)
    lum = flat.max(axis=1)
    order = np.argsort(lum)
    keep = order[int(len(order) * 0.30):]  # 丢最暗 30%
    camo_ref = np.median(flat[keep], axis=0)
    print(f"[camo_ref] = {camo_ref.astype(int)} (sampled from y 0.08-0.18 strip)")

    fill_band_bruteforce(arr, small_band, camo_ref)
    fill_band_bruteforce(arr, big1_band, camo_ref)
    fill_band_bruteforce(arr, big2_band, camo_ref)
    print(f"[fill] all 3 bands filled with camo_ref={camo_ref.astype(int)}, feathered={FEATHER}px")

    result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)

    # 新字颜色：深軍灰（camo 中值偏暗一档，保证对比）
    def ink_color(med):
        c = med.mean()
        return (int(max(c - 95, 25)),) * 3

    F_IMPACT = r"C:\Windows\Fonts\impact.ttf"
    F_ARIALBD = r"C:\Windows\Fonts\arialbd.ttf"

    def draw_in_band(band, text, font_path, name):
        y0, y1, x0, x1 = band
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        max_w = (x1 - x0) - 40
        max_h = (y1 - y0) - 14
        font = fit_font(draw, text, font_path, max_w, max_h)
        tw, th = measure(draw, text, font)
        print(f"[{name}] '{text}' size={font.size} -> {tw}x{th}px band {max_w}x{max_h}")
        draw_centered(draw, text, font, cx, cy, fill=ink_color(np.median(arr[y0:y1, x0:x1], axis=(0, 1))))

    draw_in_band(small_band, small_new, F_ARIALBD, "small")
    draw_in_band(big1_band, big1_new, F_IMPACT, "big1")
    draw_in_band(big2_band, big2_new, F_IMPACT, "big2")

    result.save(out_path, "JPEG", quality=92)
    print(f"[OK] saved: {out_path}  size={result.size}")


if __name__ == "__main__":
    main()
