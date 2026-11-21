# -*- coding: utf-8 -*-
"""自定义竖排花体渲染：横排整词(紧凑) -> 旋转90 -> 贴图。绕开引擎 fit_line 的负字距断字问题。"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import _fz_v11_surgical as V11


def render_vertical_word(base_im, text, font_name, box, ink=(226,156,62),
                         outline=(16,12,8), outline_w=2, tight=1.0):
    """
    box = [x0,y0,x1,y1] 目标竖排区域（896基准系会自动缩放）。
    横排渲染 text 到透明图，按目标高度缩放，收紧到连笔紧凑，旋转90°（字从上往下读），贴入 box。
    """
    W, H = base_im.size
    sx, sy = W / 896.0, H / 1152.0
    x0, y0, x1, y1 = [int(round(v*k)) for v, k in zip(box, (sx, sy, sx, sy))]
    tw, th = x1-x0, y1-y0
    fp = V11.find_font(font_name)
    if not fp:
        print('NO FONT', font_name); return base_im

    # 1) 横排渲染到透明图，字号先取大值
    size = int(th * 0.95)
    f = ImageFont.truetype(fp, size)
    bb = f.getbbox(text)
    w0 = bb[2]-bb[0]; h0 = bb[3]-bb[1]
    pad = 20
    tile = Image.new('RGBA', (w0+pad*2, h0+pad*2), (0,0,0,0))
    ImageDraw.Draw(tile).text((pad-bb[0], pad-bb[1]), text, font=f, fill=(255,255,255,255))
    # 裁到实际字形
    b2 = tile.split()[-1].getbbox()
    if b2:
        tile = tile.crop(b2)
    # 2) 缩放到目标长度（竖排后高度=词宽方向），收紧字距靠 tight 控制缩放比
    target_len = int(th * tight)
    sc = target_len / tile.width
    nw, nh = max(1,int(tile.width*sc)), max(1,int(tile.height*sc))
    tile = tile.resize((nw, nh), Image.LANCZOS)
    # 3) 旋转 90°（PIL rotate(90)逆时针 -> 横排左端变底部；用270让左端变顶部=从上往下读）
    tile = tile.rotate(270, expand=True, resample=Image.BICUBIC)
    # 4) 上色 + 描边
    alpha = np.asarray(tile.split()[-1])
    k = int(outline_w * sx)
    ring = cv2.dilate(alpha, np.ones((k,k),np.uint8), iterations=1)
    ring = np.clip(ring.astype(np.int16)-alpha.astype(np.int16),0,255).astype(np.uint8)
    rgba = np.zeros((tile.height, tile.width, 4), np.uint8)
    rgba[...,:3] = np.array(ink, np.uint8)
    rgba[...,3] = alpha
    r2 = np.zeros_like(rgba)
    r2[...,:3] = np.array(outline, np.uint8)
    r2[...,3] = ring
    out = Image.alpha_composite(Image.fromarray(r2,'RGBA'), Image.fromarray(rgba,'RGBA'))
    # 5) 缩放适配框（如超框），居中贴入
    if out.width > tw or out.height > th:
        s2 = min(tw/out.width, th/out.height)
        out = out.resize((max(1,int(out.width*s2)), max(1,int(out.height*s2))), Image.LANCZOS)
    px = x0 + (tw-out.width)//2
    py = y0 + (th-out.height)//2
    base = base_im.convert('RGBA')
    base.alpha_composite(out, (px, py))
    print('[vert] %s size=%d final=%dx%d at (%d,%d)' % (text, size, out.width, out.height, px, py))
    return base.convert('RGB')
