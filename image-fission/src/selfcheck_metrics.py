"""selfcheck_metrics.py — 100% 预期量化自检。
原图 vs 裂变图 算三指标：
  1) color_intersect : 主导色 LAB 直方图交集 (配色保真, 目标>=0.85)
  2) ssim           : 结构相似度 (构图保真, 目标>=0.55)
  3) frag_ratio     : 局域颜色梯度比 (圆润块碎片度, 目标<=1.25 即不比原图碎)
用法: python selfcheck_metrics.py <orig> <fiss> [label]
"""
import sys
import numpy as np
from PIL import Image

def to_lab(arr):
    # arr: uint8 RGB (h,w,3) -> lab float, 简单近似 (非色彩精确但足够比差)
    r, g, b = arr[:,:,0].astype(float)/255, arr[:,:,1].astype(float)/255, arr[:,:,2].astype(float)/255
    # sRGB->linear
    def lin(c):
        return np.where(c<=0.04045, c/12.92, ((c+0.055)/1.055)**2.4)
    r,g,b = lin(r),lin(g),lin(b)
    x = r*0.4124+g*0.3576+b*0.1805
    y = r*0.2126+g*0.7152+b*0.0722
    z = r*0.0193+g*0.1192+b*0.9505
    # 白点 D65
    xn,yn,zn = 0.95047,1.0,1.08883
    def f(t):
        return np.where(t>0.008856, t**(1/3), 7.787*t+16/116)
    fx,fy,fz = f(x/xn),f(y/yn),f(z/zn)
    L = 116*fy-16
    A = 500*(fx-fy)
    B = 200*(fy-fz)
    return np.stack([L,A,B],axis=-1)

def color_intersect(orig, fiss):
    o = to_lab(np.array(Image.open(orig).convert('RGB').resize((128,128))))
    f = to_lab(np.array(Image.open(fiss).convert('RGB').resize((128,128))))
    # L:0..100 -> 5 bins ; a,b:-128..127 -> 9 bins
    def hist(im):
        L=im[:,:,0]; A=im[:,:,1]; B=im[:,:,2]
        Lb=np.clip((L/20).astype(int),0,4)
        Ab=np.clip(((A+128)/32).astype(int),0,7)
        Bb=np.clip(((B+128)/32).astype(int),0,7)
        h=np.zeros((5,8,8),dtype=float)
        for i in range(Lb.size):
            h[Lb.flat[i],Ab.flat[i],Bb.flat[i]]+=1
        return h.flatten()/h.sum()
    ho,hf=hist(o),hist(f)
    return float(np.minimum(ho,hf).sum())

def ssim(orig, fiss):
    o=np.array(Image.open(orig).convert('L').resize((256,256)),dtype=float)
    f=np.array(Image.open(fiss).convert('L').resize((256,256)),dtype=float)
    mu_o,mu_f=o.mean(),f.mean()
    sig_o,sig_f=o.var(),f.var()
    sig_of=((o-o.mean())*(f-f.mean())).mean()
    c1,c2=(0.01*255)**2,(0.03*255)**2
    num=(2*mu_o*mu_f+c1)*(2*sig_of+c2)
    den=(mu_o**2+mu_f**2+c1)*(sig_o+sig_f+c2)
    return float(num/den)

def frag_ratio(orig, fiss):
    o=np.array(Image.open(orig).convert('RGB').resize((256,256))).astype(float)
    f=np.array(Image.open(fiss).convert('RGB').resize((256,256))).astype(float)
    def grad(im):
        gx=np.abs(im[1:,:,:-1][...,:-1]-im[:-1,:,:-1][...,:-1]).mean(axis=2)
        return gx.mean()
    return grad(f)/max(grad(o),1e-6)

if __name__=='__main__':
    import os
    pairs = [
        ('camo_4','web_gallery/img/orig_camo_4.jpg','outputs/v175/v175_fix_camo_4.jpg'),
        ('denim_3','web_gallery/img/orig_denim_3.jpg','jobs/smoke_v164/v164_denim_3.jpg'),
        ('illust_1','web_gallery/img/orig_illust_1.jpg','jobs/smoke_v164/v164_illust_1.jpg'),
        ('skull_5','web_gallery/img/orig_skull_5.jpg','jobs/smoke_v164/v164_skull_5.jpg'),
        ('metal_6','web_gallery/img/orig_metal_6.jpg','jobs/smoke_v164/v164_metal_6.jpg'),
    ]
    print(f"{'img':10} {'color∩':>8} {'ssim':>7} {'fragR':>7}  verdict")
    for name,o,f in pairs:
        if not (os.path.exists(o) and os.path.exists(f)):
            print(f"{name:10} MISSING FILES"); continue
        ci=color_intersect(o,f); ss=ssim(o,f); fr=frag_ratio(o,f)
        # 验收: 配色>=0.85, ssim>=0.55, frag<=1.25
        ok = ci>=0.85 and ss>=0.55 and fr<=1.25
        v = 'PASS' if ok else 'FAIL'
        print(f"{name:10} {ci:8.3f} {ss:7.3f} {fr:7.3f}  {v}")
