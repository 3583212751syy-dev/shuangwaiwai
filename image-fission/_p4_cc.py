import sys, numpy as np
sys.path.insert(0,'.')
from PIL import Image
from scipy import ndimage as ndi
from styles import camo_palm_pattern as cpp
img=Image.open('E:/Desktop/图裂变测试图/Pinterest (4).jpg').convert('RGB')
W,H=img.size
ink=cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
lab,n=ndi.label(ink, np.ones((3,3),bool))
sz=np.bincount(lab.ravel()); sz[0]=0
order=np.argsort(sz)[::-1]
print('ink=%.2f%%  连通块=%d'%(100*ink.mean(),n))
print('最大 24 块 (面积 / bbox):')
for i in order[:24]:
    if sz[i]==0: continue
    ys,xs=np.where(lab==i)
    print('  #%d area=%d bbox x[%d-%d] y[%d-%d] (w%d h%d)'%(i,sz[i],xs.min(),xs.max(),ys.min(),ys.max(),xs.max()-xs.min()+1,ys.max()-ys.min()+1))
print('面积<50px: %d 个 / 合计 %d px'%(int((sz[1:]<50).sum()), int(sz[sz<50].sum())))
print('面积>=300px: %d 个 / 合计 %.2f%%'%(int((sz[1:]>=300).sum()), 100*sz[sz>=300].sum()/(W*H)))
segs=cpp.trunks(ink,H,W,25,60)
segs=[s for s in segs if 0.06*H<s['h']<0.62*H]
print('树干候选=%d'%len(segs))
