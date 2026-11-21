# -*- coding: utf-8 -*-
"""11 修正 v2：自定义竖排渲染，绕开引擎 fit_line 断字问题。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fz_v171_layout as L
from _zai_v173_verttext import render_vertical_word
from PIL import Image

BASE = r'D:/Users/Administrator/Desktop/双接口'
NAME = '11_st_tropez'
OUT = BASE + r'/outputs/_fix11_out'
os.makedirs(OUT, exist_ok=True)
CACHE = BASE + r'/outputs/_flux_v172_cache'

# 底图源文件
src = os.path.join(CACHE, 'v172_11_st_tropez_s501.png')


def build(font, word, outfile, vbox, tight=0.92):
    from PIL import Image
    im = Image.open(src).convert('RGB')
    # 1) 擦旧伪字（右侧竖排 + 左上 Voyage）
    # 右侧竖排旧伪字擦除框必须覆盖到底部(y1080)，否则残留散落孤字
    im = L.inpaint_by_level(im, [[520, 420, 720, 1120], [93, 164, 362, 268]], lo=60, hi=255, dilate=13)
    # 2) 左上 Voyage（横排花体，引擎原样绘制到独立层再合成）
    L.COORD[NAME] = [dict(box=[93, 172, 362, 250], text='Voyage', role='script',
                          font_name='MrDafoe', ink=(208,140,50), ink_mode='light',
                          outline=(16,12,8), outline_w=3, scale=0.9)]
    L.OUT = OUT
    # 用 draw_engraved_light 绘制 Voyage 到 im
    im2 = L.draw_engraved_light(im, L.COORD[NAME])
    # 3) 自定义竖排主花体
    im3 = render_vertical_word(im2, word, font, vbox, ink=(226,156,62),
                               outline=(16,12,8), outline_w=2, tight=tight)
    dst = os.path.join(OUT, outfile)
    im3.save(dst, quality=95, subsampling=0)
    print('[DONE]', dst)


if __name__ == '__main__':
    import sys as _s
    font = _s.argv[1] if len(_s.argv) > 1 else 'GreatVibes'
    build(font, 'Riviera', '11v2_%s.jpg' % font, [555, 360, 715, 1060], tight=1.0)
