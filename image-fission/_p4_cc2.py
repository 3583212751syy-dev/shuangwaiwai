import sys, numpy as np
sys.path.insert(0,'.')
from PIL import Image
from scipy import ndimage as ndi
from styles import camo_palm_pattern as cpp
img=Image.open('E:/Desktop/图裂变测试图/Pinterest (4).jpg').convert('RGB')
W,H=img.size
ink=cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
def disk(r):
    y,x=np.ogrid[-r:r+1,-r:r+1]; return x*x+y*y<=r*r
for r in (1,2,3,4,5):
    er=ndi.binary_erosion(ink, structure=disk(r))
    back=ndi.binary_propagation(er, mask=ink) if False else None
    lab,n=ndi.label(er, np.ones((3,3),bool))
    sz=np.bincount(lab.ravel()); sz[0]=0
    big=int((sz[1:]>=300).sum())
    # 还原到原墨（每块膨胀回原 ink 的连通域）
    lab_full=ndi.grey_dilation(lab, footprint=disk(r), mode='constant')  # 近似
    med=int(np.median(sz[sz>=300])) if big else 0
    print('erode r=%d: 块=%d  >=300px块=%d  其中位面积=%d'%(r,n,big,med))
# 展示 r=3 时大块的 bbox（看是否变成单棵棕榈尺寸）
r=3
er=ndi.binary_erosion(ink, structure=disk(r))
lab,n=ndi.label(er, np.ones((3,3),bool))
sz=np.bincount(lab.ravel()); sz[0]=0
order=np.argsort(sz)[::-1]
print('--- r=3 最大 12 块 bbox（单棵棕榈≈300x400）---')
for i in order[:12]:
    if sz[i]<300: continue
    ys,xs=np.where(lab==i)
    print('  area=%d bbox %dx%d  x[%d-%d] y[%d-%d]'%(sz[i],xs.max()-xs.min()+1,ys.max()-ys.min()+1,xs.min(),xs.max(),ys.min(),ys.max()))
