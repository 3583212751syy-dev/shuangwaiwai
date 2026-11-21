# -*- coding: utf-8 -*-
"""
v173 - 05 恶魔女郎专用修正（用户 2026-11-21 令：双色层 + 文字重排）
不动 v171/v172 主引擎，独立 monkeypatch 后调用 L.layout。

修正内容（全部基于实测，人物/车/盾像素零重绘）：
1. 双色层：底图 896x1152 等比缩到宽 741，居中放在 1024x1152 暗紫红(103,49,55)画布上，
   还原原图"外暗紫红大底 + 内层亮红矩形"两层结构。
2. 文字重排（896 基准坐标系，加边后 make_jobs 自动按 W/896 缩放）：
   - FREEWAY 弧排收窄，不压盾顶
   - ROUTE 下移进盾牌横条区（盾顶黑弧 y210-260，横条 y270-330）
   - GO WIDE OPEN 三行收拢到原图 NO SPEED LIMIT 的位置风格
   - TO HELL 上移+收窄，离开女郎腿部
   - 底部标语收窄
"""
import os, sys, shutil
import numpy as np
from PIL import Image

BASE = r'D:/Users/Administrator/Desktop/双接口'
SRC_CACHE = BASE + r'/outputs/_flux_v172_cache'
FIX_CACHE = BASE + r'/outputs/_fix05_v173_cache'
os.makedirs(FIX_CACHE, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fz_v171_layout as L

NAME = '05_route666'
SRC_NAME = 'v172_05_route666_s503.png'
BG_OUTER = (103, 49, 55)   # 原图外层暗紫红实测
CANVAS_W, CANVAS_H = 1024, 1152
INNER_W = 741              # 原图内层亮红矩形宽度实测 (x140-881)


def make_bicolor():
    """把底图等比缩到宽 INNER_W，居中贴到暗紫红画布。人物等比缩放，零变形。"""
    src = os.path.join(SRC_CACHE, SRC_NAME)
    im = Image.open(src).convert('RGB')
    w0, h0 = im.size
    scale = INNER_W / float(w0)
    nw, nh = INNER_W, int(round(h0 * scale))
    im2 = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), BG_OUTER)
    # 加轻微颗粒到暗紫红底，避免纯色块（原图外层有做旧纹理）
    arr = np.array(canvas).astype(np.int16)
    rng = np.random.default_rng(505)
    grain = rng.normal(0, 3.2, arr.shape)
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(arr)
    ox = (CANVAS_W - nw) // 2
    oy = (CANVAS_H - nh) // 2
    canvas.paste(im2, (ox, oy))
    dst = os.path.join(FIX_CACHE, 'v173_%s_s503.png' % NAME)
    canvas.save(dst)
    print('[bicolor] src %dx%d -> inner %dx%d at (%d,%d) on %dx%d %s' % (
        w0, h0, nw, nh, ox, oy, CANVAS_W, CANVAS_H, BG_OUTER))
    return dst, ox, oy, scale


def main():
    make_bicolor()
    # ---- 文字重排：896 基准坐标系（make_jobs 会按 W/896 自动缩放）----
    L.COORD[NAME] = [
        # FREEWAY 顶部：原图 HIGHWAY 是平直粗黑体微拱，不是大圆弧 → 改普通横排，横贯上方
        dict(box=[130, 78, 766, 178], text='FREEWAY', role='sans', font_name='ArchivoBlack',
             ink=(30, 17, 11), scale=0.87, tracking=0.01),
        # ROUTE 下移进盾牌横条（黑弧下方米黄区 y320-365）
        dict(box=[330, 318, 585, 372], text='ROUTE', role='slab', font_name='ArchivoBlack',
             ink=(30, 18, 10), scale=0.82),
        # 313 盾牌中央，缩小避免顶到 ROUTE（scale 0.9→0.68）
        dict(box=[345, 372, 590, 540], text='313', role='slab', font_name='Bevan',
             ink=(28, 16, 10), scale=0.68),
        # GO WIDE OPEN 收拢到右中，仿原图 NO SPEED LIMIT 三行紧凑
        dict(box=[648, 350, 748, 384], text='GO', role='slab', font_name='Bevan', ink=(28, 16, 10), scale=0.72),
        dict(box=[648, 388, 748, 422], text='WIDE', role='sans', font_name='Anton', ink=(28, 16, 10), scale=0.72),
        dict(box=[648, 426, 748, 460], text='OPEN', role='sans', font_name='Anton', ink=(28, 16, 10), scale=0.72),
        # TO HELL 上移+收窄，离开女郎腿部（原 y915-1015 压腿，改 y885-975 且左侧收进）
        dict(box=[210, 882, 690, 972], text='TO HELL', role='slab', font_name='Bevan',
             ink=(28, 16, 10), scale=0.88),
        # 底部标语收窄居中
        dict(box=[230, 985, 660, 1022], text='CHASE ALL YOUR DREAMS', role='sans', font_name='Anton',
             ink=(232, 206, 168), ink_mode='light', outline=(88, 40, 32), outline_w=2, tracking=0.02, scale=0.62),
    ]
    # 05 底图无伪字（wipe 已清），不需要擦除
    L.ERASE_DETECT[NAME] = []
    # 输出到独立目录，不覆盖 v172 成品
    L.OUT = BASE + r'/outputs/_fix05_out'
    os.makedirs(L.OUT, exist_ok=True)
    L.layout(NAME, cache=FIX_CACHE)
    op = os.path.join(L.OUT, 'A_%s' % NAME, 'v24_v1_fission.jpg')
    print('[DONE]', op)


if __name__ == '__main__':
    main()
