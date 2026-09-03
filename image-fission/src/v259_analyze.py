"""v259 前置分析: 量化原图环/文字的径向位置与颜色, 防止 inpaint 误伤外环。"""
import numpy as np, cv2
import importlib.util
spec = importlib.util.spec_from_file_location('v253', 'src/v253_bat_logo_inpaint.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

from PIL import Image
ref = m.COMFY_INPUT / m.REF_IMG
rgb_pil = np.array(Image.open(ref).convert('RGB'))
bgr = cv2.cvtColor(rgb_pil, cv2.COLOR_RGB2BGR)
rgb = rgb_pil
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
h, w = bgr.shape[:2]
print(f"size={w}x{h}")
print(f"RING cx={m.RING['cx']} cy={m.RING['cy']} outer_r={m.RING['outer_r']} inner_r={m.RING['inner_r']}")
print(f"ring top y = {m.RING['cy']-m.RING['outer_r']}  ring bottom y = {m.RING['cy']+m.RING['outer_r']}")

yy, xx = np.mgrid[:h, :w]
dist = np.sqrt((xx - m.RING['cx'])**2 + (yy - m.RING['cy'])**2)

# 各环带颜色
for name, (d0, d1) in [("外环带[407,421]",(m.RING['inner_r'],m.RING['outer_r'])),
                        ("内圈[0,407]",(0,m.RING['inner_r'])),
                        ("内圈靠边[380,407]",(380,m.RING['inner_r']))]:
    sel = (dist>=d0)&(dist<d1)
    print(f"{name}: 像素数={sel.sum()}  均值RGB={rgb[sel].reshape(-1,3).mean(0).round(1).tolist()}")

# 顶弧文字区 (LA CASA) 径向分布: 看 y=340-490 这条带里暗像素的 dist
band = (yy>=340)&(yy<490)&(xx>=400)&(xx<=1150)
dark = gray[band] < 110
yd = yy[band][dark]
xd = xx[band][dark]
dd = dist[band][dark]
if len(dd):
    print(f"顶弧暗像素: 数量={len(dd)} dist范围=[{dd.min():.0f},{dd.max():.0f}]  中位dist={np.median(dd):.0f}")
    # 这些暗像素里, 落在内圈(<inner_r) vs 环带的比例
    print(f"  在内圈(<407)={(dd<407).mean()*100:.0f}%  在环带[407,421]={( (dd>=407)&(dd<=421)).mean()*100:.0f}%  在外(>421)={(dd>421).mean()*100:.0f}%")

# BACARDÍ 大字区
band2 = (yy>=770)&(yy<1150)&(xx>=280)&(xx<=1260)
dark2 = gray[band2] < 110
print(f"BACARDÍ区暗像素: 数量={dark2.sum()}  dist中位={np.median(dist[band2][dark2]):.0f} (应在环外>421)")

# WHEART 区
band3 = (yy>=1180)&(yy<1330)&(xx>=450)&(xx<=1160)
dark3 = gray[band3] < 110
print(f"WHEART区暗像素: 数量={dark3.sum()}")

# bat 区
bx,by,bw,bh = m.BAT_BBOX
band4 = (yy>=by)&(yy<by+bh)&(xx>=bx)&(xx<bx+bw)
dark4 = gray[band4] < 50
print(f"bat区 gray<50 像素: 数量={dark4.sum()}  中位dist={np.median(dist[band4][dark4]):.0f}")
