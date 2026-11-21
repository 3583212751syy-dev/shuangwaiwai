# -*- coding: utf-8 -*-
"""
v173 - 11 黑金专用文字修正（竖排金色花体重建）
原图：右侧一个完整竖排金色花体词（连笔贯穿、居中偏右醒目）。
v172 问题：rotate=-80 整词旋转后超框被裁 → Riviera 断成散块。
修法：vertical=True（引擎横排渲染后 rotate90，expand=True 不裁切）+ 加宽框给足空间 + 多字体候选。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fz_v171_layout as L

BASE = r'D:/Users/Administrator/Desktop/双接口'
NAME = '11_st_tropez'
L.OUT = BASE + r'/outputs/_fix11_out'
os.makedirs(L.OUT, exist_ok=True)

# 候选花体字体（粗笔连笔优先，接近原图金色花体）
CANDIDATES = ['KaushanScript', 'MrDafoe', 'GreatVibes', 'Allura', 'MonsieurLaDoulaise']


def run(font, word, box, outfile):
    L.COORD[NAME] = [
        # 左上 Voyage：保留原位的 MrDafoe 横排花体
        dict(box=[93, 172, 362, 250], text='Voyage', role='script', font_name='MrDafoe',
             ink=(208, 140, 50), ink_mode='light', outline=(16, 12, 8), outline_w=3, scale=0.9),
        # 右侧竖排主花体：rotate=-90 整词旋转（字从上往下读），框给足避免裁切
        dict(box=box, text=word, role='script', font_name=font,
             ink=(226, 156, 62), ink_mode='light', outline=(16, 12, 8), outline_w=2,
             rotate=-90, tracking=-0.05, scale=0.68),
    ]
    L.ERASE_DETECT[NAME] = [
        ([538, 430, 698, 1020], 'level'),   # 擦掉底图旧的残缺竖排伪字
        ([93, 164, 362, 268], 'level'),     # 擦旧的 Voyage 伪花体
    ]
    L.layout(NAME, cache=BASE + r'/outputs/_flux_v172_cache')
    src = os.path.join(L.OUT, 'A_%s' % NAME, 'v24_v1_fission.jpg')
    dst = os.path.join(L.OUT, outfile)
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
        print('[OK]', dst)
    else:
        print('[FAIL]', font)


if __name__ == '__main__':
    # 候选：优雅连笔花体。fit_line 用 th 算字号、tw 算字距：
    # th 减小→字号缩小；tw 贴近词宽→字距紧凑不散。
    import sys as _s
    which = _s.argv[1] if len(_s.argv) > 1 else 'GreatVibes'
    # 竖排词长度靠 rotate 前的 tw 控制（约 480），th 控字号（约 300 → 字号适中）
    run(which, 'Riviera', [560, 560, 820, 880], '11_%s_riviera.jpg' % which)
