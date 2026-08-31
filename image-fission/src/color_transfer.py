"""color_transfer.py — reinhard LAB 色彩迁移 (target=裂变图, ref=原图)。
不占 GPU，纯 numpy/PIL，确定性可复现。对应 ComfyUI 内置 ColorTransfer(reinhard_lab)。
strength: 0=不动, 1=完全对齐原图色域统计, 可>1 过冲。
用法: python color_transfer.py <orig> <base> <out_prefix> [--sweep]
"""
import sys, os
import numpy as np
from PIL import Image

def rgb_to_lab(img):
    a = np.array(img, dtype=float)/255.0
    r,g,b = a[:,:,0],a[:,:,1],a[:,:,2]
    def lin(c): return np.where(c<=0.04045, c/12.92, ((c+0.055)/1.055)**2.4)
    r,g,b = lin(r),lin(g),lin(b)
    x=r*0.4124+g*0.3576+b*0.1805; y=r*0.2126+g*0.7152+b*0.0722; z=r*0.0193+g*0.1192+b*0.9505
    xn,yn,zn=0.95047,1.0,1.08883
    def f(t): return np.where(t>0.008856, t**(1/3), 7.787*t+16/116)
    fx,fy,fz=f(x/xn),f(y/yn),f(z/zn)
    L=116*fy-16; A=500*(fx-fy); B=200*(fy-fz)
    return np.stack([L,A,B],-1)

def lab_to_rgb(lab):
    L,A,B=lab[:,:,0],lab[:,:,1],lab[:,:,2]
    fy=(L+16)/116; fx=fy+A/500; fz=fy-B/200
    def inv(t): return np.where(t>0.206897, t**3, (t-16/116)/7.787)
    x=inv(fx)*0.95047; y=inv(fy)*1.0; z=inv(fz)*1.08883
    r=x*3.2406 + y*-1.5372 + z*-0.4986
    g=x*-0.9689 + y*1.8758 + z*0.0415
    b=x*0.0557 + y*-0.2040 + z*1.0570
    def d(c): c=np.clip(c,0,1); return np.where(c>0.0031308, 1.055*c**(1/2.4)-0.055, 12.92*c)
    r,g,b=d(r),d(g),d(b)
    return np.stack([r,g,b],-1)

def reinhard(target_rgb, ref_rgb, strength=1.0):
    t=target_rgb.astype(float); r=ref_rgb.astype(float)
    tlab=rgb_to_lab(t); rlab=rgb_to_lab(r)
    out=np.empty_like(tlab)
    for c in range(3):
        mt,st=tlab[:,:,c].mean(),tlab[:,:,c].std()+1e-6
        mr,sr=rlab[:,:,c].mean(),rlab[:,:,c].std()+1e-6
        v=(tlab[:,:,c]-mt)/st*sr+mr
        out[:,:,c]=tlab[:,:,c]*(1-strength)+v*strength
    # L 截断到 [0,100], a/b 合理范围
    out[:,:,0]=np.clip(out[:,:,0],0,100)
    out[:,:,1:]=np.clip(out[:,:,1:],-128,127)
    return (np.clip(lab_to_rgb(out),0,1)*255).astype(np.uint8)

def color_intersect(o_path, f_path):
    o=rgb_to_lab(np.array(Image.open(o_path).convert('RGB').resize((128,128))))
    f=rgb_to_lab(np.array(Image.open(f_path).convert('RGB').resize((128,128))))
    def hist(im):
        L=im[:,:,0];A=im[:,:,1];B=im[:,:,2]
        Lb=np.clip((L/20).astype(int),0,4);Ab=np.clip(((A+128)/32).astype(int),0,7);Bb=np.clip(((B+128)/32).astype(int),0,7)
        h=np.zeros((5,8,8),float)
        for i in range(Lb.size): h[Lb.flat[i],Ab.flat[i],Bb.flat[i]]+=1
        return h.flatten()/h.sum()
    ho,hf=hist(o),hist(f); return float(np.minimum(ho,hf).sum())

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('orig'); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--sweep',action='store_true')
    a=ap.parse_args()
    orig=np.array(Image.open(a.orig).convert('RGB')); base=np.array(Image.open(a.base).convert('RGB'))
    if a.sweep:
        best=None
        for s in [0.5,0.7,0.85,1.0,1.15]:
            out=reinhard(base,orig,s); p=f"{a.out}_s{s:.2f}.png"
            Image.fromarray(out).save(p)
            ci=color_intersect(a.orig,p); print(f"s={s:.2f} color∩={ci:.3f} -> {p}")
            if best is None or ci>best[0]: best=(ci,s,p)
        print(f"BEST s={best[1]:.2f} color∩={best[0]:.3f}")
    else:
        out=reinhard(base,orig,1.0); Image.fromarray(out).save(a.out); print("saved",a.out,"color∩=",round(color_intersect(a.orig,a.out),3))
