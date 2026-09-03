"""v260_postproc — SDXL 输出 -> PIL 后期:
  1. resize 1024x1344 -> 1552x2000
  2. cv2.inpaint 擦掉残留 BACARDÍ/WHEART/LA CASA (真实坐标, v259 实测)
  3. 烧新字按原图字号/位置:
       顶弧 (776, 374) radius 371, 宽 ~344 -> "NOCTIS ALATA DOMVS"
       大主字 (776, 829) 宽 ~358        -> "NOCTWING"
       副字 (776, 1261) 宽 ~193          -> "MORS VINI"
  4. 严守:
       - 字必须按原图墨色 INK = (26, 10, 31)
       - 文字角度/弧度按原图实测
       - 配色锁死不引入新色
       - 保留主体鸟 (raven/owl/falcon) 不动
"""

import numpy as np
import cv2
import importlib.util
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, 'src')
spec = importlib.util.spec_from_file_location('v253', 'v253_bat_logo_inpaint.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

JOB = Path('../jobs/smoke_v260')
INK = m.INK                       # (26, 10, 31)
TARGET = (1552, 2000)             # 原图分辨率

# v259 实测文字坐标 (排除 bat 后通过逐行暗像素直方图得到)
TEXT_BANDS = [
    # (y0, y1, x0, x1) — 原图坐标 (1552x2000)
    # BACARDÍ 大主字 (徽章内下, 实测中心 776, 829)
    (700, 920, 600, 960),
    # Est./1862 数字 (徽章底内圈 y≈1020-1140, 也擦)
    (1010, 1150, 320, 1240),
    # WHEART 副字 (徽章外下, y≈1200-1320)
    (1200, 1330, 200, 1352),
]


def erase_text_residue(img):
    """用 cv2.inpaint 擦掉 SDXL 残留的黑色文字.
    仅在原图文字的实测坐标带内做, bat/owl/falcon 主体完全不受影响"""
    arr = np.array(img.convert('RGB'))[..., ::-1].copy()  # to BGR for cv2
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for (y0, y1, x0, x1) in TEXT_BANDS:
        if y1 > h or x1 > w:
            continue
        mask[y0:y1, x0:x1] = 255
    # 边界扩张
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), 2)
    erased = cv2.inpaint(arr, mask, 9, cv2.INPAINT_TELEA)
    img2 = Image.fromarray(erased[..., ::-1])
    return img2


def resize_to_target(img):
    if img.size != TARGET:
        img = img.resize(TARGET, Image.LANCZOS)
    return img


def burn_text_on(img, top_arc=True, main_word="NOCTWING", sub_word="MORS VINI",
                 top_text="NOCTIS ALATA DOMVS"):
    """按 v259 实测 v253 函数 (但传入 v260 的相对尺寸 1552x2000)"""
    h, w = img.height, img.width
    img = img.convert('RGB')
    # 顶弧字 (圆心 776,745 / radius 371), 字高 ~h*0.038
    if top_arc:
        img = m.burn_top_arc(img, top_text, int(h * 0.038))
    # 大主字 (原 BACARDÍ 位, 实测中心 776,829, 原字宽 ~358)
    fs_main = m.calibrate(main_word, 358)
    m.burn_centered(img, main_word, fs_main, 776, 829)
    # 副字 (原 WHEART 位, 实测中心 776,1261, 原字宽 ~193)
    fs_small = m.calibrate(sub_word, 193)
    m.burn_centered(img, sub_word, fs_small, 776, 1261)
    return img


def main():
    for tag, fname in [('raven', 'v260_test_raven_raw.png'),
                       ('owl', 'v260_owl_raw.png'),
                       ('falcon', 'v260_falcon_raw.png')]:
        src = JOB / fname
        if not src.exists():
            print(f'MISS {src}')
            continue
        img = Image.open(src).convert('RGB')
        print(f'[{tag}] input size={img.size}')
        img = resize_to_target(img)
        print(f'  resized to {img.size}')
        img = erase_text_residue(img)
        print(f'  text residue erased')
        img = burn_text_on(img)
        print(f'  text burned')
        out = JOB / f'v260_{tag}_final.jpg'
        if img.mode != 'RGB': img = img.convert('RGB')
        img.save(str(out), quality=95)
        print(f'  -> {out}')


if __name__ == '__main__':
    main()
