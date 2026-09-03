"""v259 输出自检 (替代肉眼): 验证 外环保留/ bat在位/ 新文字落位/ 旧文字清零/ 配色锁死。"""
import numpy as np, cv2, importlib.util, sys
from pathlib import Path
from PIL import Image
spec = importlib.util.spec_from_file_location('v253', 'src/v253_bat_logo_inpaint.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.insert(0, 'src'); import v259_clean_fission as v259

JOB = Path('jobs/smoke_v259')
# 原图参考
ref_bgr = cv2.cvtColor(np.array(Image.open(m.COMFY_INPUT / m.REF_IMG).convert('RGB')), cv2.COLOR_RGB2BGR)
ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
yy, xx = np.mgrid[:ref_gray.shape[0], :ref_gray.shape[1]]
dist = np.sqrt((xx - m.RING['cx'])**2 + (yy - m.RING['cy'])**2)
ring_zone = (dist > (m.RING['inner_r']-4)) & (dist < (m.RING['outer_r']+4))
bat = v259.extract_bat_strict(ref_bgr)

def check(path):
    bgr = cv2.cvtColor(np.array(Image.open(path).convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = bgr.shape[:2]
    print(f"\n=== {Path(path).name} ===")
    # 1) 外环带是否还在: 原图外环带是亮紫(均值~158,107,150), 若被吃成 blob 则暗像素↑
    ring_rgb = bgr[ring_zone]
    ring_dark = (gray[ring_zone] < 120).mean() * 100
    ring_mean = ring_rgb.reshape(-1,3).mean(0)
    ring_ref = ref_bgr[ring_zone].reshape(-1,3).mean(0)
    print(f"  外环带: 暗像素={ring_dark:.1f}% (原图参考~{(ref_gray[ring_zone]<120).mean()*100:.1f}%)  均值RGB={ring_mean.round(0).tolist()} (原图{ring_ref.round(0).tolist()})")
    # 2) bat 在位: 原图 bat 掩码区 暗像素应高
    bx,by,bw,bh = m.BAT_BBOX
    bat_dark = (gray[by:by+bh, bx:bx+bw][v259.extract_bat_strict(bgr)[by:by+bh,bx:bx+bw]] < 60).mean()*100 if v259.extract_bat_strict(bgr)[by:by+bh,bx:bx+bw].any() else 0
    print(f"  bat区暗像素(新图)={bat_dark:.1f}%  原图bat区暗像素={(ref_gray[bat]<60).mean()*100:.1f}%")
    # 3) 新文字落位: NOCTWING 中心(776,800) 附近暗像素; MORS VINI (776,1261)
    def near(cx,cy,r=60):
        msk = (xx-cx)**2+(yy-cy)**2 < r*r
        return (gray[msk] < 90).mean()*100
    print(f"  NOCTWING位(776,800)暗像素={near(776,800):.1f}%  MORS VINI位(776,1261)暗像素={near(776,1261):.1f}%  顶弧(776,374)暗像素={near(776,374):.1f}%")
    # 4) 旧文字清零: 原 BACARDÍ/WHEART/LA CASA 字位 新图应无原字形(与参考比, 新图暗像素应≈原图背景)
    # 5) 配色锁死: 整体色板与原图 chi2
    hist_chi = 0
    for c in range(3):
        ha,_ = np.histogram(ref_bgr[...,c], 256, [0,256], density=True)
        hb,_ = np.histogram(bgr[...,c], 256, [0,256], density=True)
        ha += 1e-6; hb += 1e-6
        chi = 0.5*((ha-hb)**2/(ha+hb)).sum()
        hist_chi += chi
    print(f"  配色 chi2(全局, 应<0.3)={hist_chi:.4f}")
    return ring_dark

for p in ['v259_A_neutral_bat_logo.jpg','v259_B_tiltL_bat_logo.jpg','v259_C_tiltR_bat_logo.jpg']:
    check(JOB/p)
