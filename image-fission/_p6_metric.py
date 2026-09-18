import sys, glob
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
ROOT = "."
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src")
from v342_rebirth import mask_p6

src = Image.open('E:/Desktop/图裂变测试图/Pinterest (6).jpg').convert('RGB')
m = mask_p6(src)
so = np.asarray(src, np.float32)
lo = rgb2lab(so / 255.0)
sel = m & (so.max(2) > 2)

def rep(tag, p):
    im = Image.open(p).convert('RGB')
    if im.size != src.size:
        im = im.resize(src.size, Image.LANCZOS)
    a = np.asarray(im, np.float32)
    ln = rgb2lab(a / 255.0)
    d = np.sqrt(((ln - lo) ** 2).sum(2))
    ms = m & (a.max(2) > 25); mo = m & (so.max(2) > 25)
    iou = (ms & mo).sum() / max(1, (ms | mo).sum())
    mx = a.max(2); mn = a.min(2)
    yellow = (mx > 140) & ((mx - mn) > 70) & (a[..., 0] > a[..., 2] + 50)
    print('[%-16s] 主体ΔE=%5.1f  ΔE>15=%4.1f%%  IoU=%.3f  黄=%.3f%%' % (
        tag, d[sel].mean(), 100 * (d[sel] > 15).mean(), iou, 100 * (yellow & m).mean()))

rep('ORIG', 'E:/Desktop/图裂变测试图/Pinterest (6).jpg')
for p in sorted(glob.glob('jobs/v389_p6crisp/p6_*_snap.jpg')) + sorted(glob.glob('jobs/v390_p6final/p6_*_snap.jpg')):
    rep(p.split('/')[-1].replace('p6_', '').replace('_snap.jpg', ''), p)
