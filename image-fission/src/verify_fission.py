import numpy as np
from pathlib import Path
from PIL import Image
SRC=Path(r"E:/Desktop/图裂变测试图")
OUT=Path(r"E:/Desktop/双接口/image-fission/jobs/text_fission_1787823676")
checks=[
 ("pinterest_denim_3.jpg","pinterest_denim_3__JEANS.png",(0,90,736,372)),
 ("pinterest_skull_5.jpg","pinterest_skull_5__REAPER.png",(170,95,560,275)),
 ("pinterest_eagle_2.jpg","pinterest_eagle_2__RAPTOR.png",(340,700,610,790)),
]
for src,out,box in checks:
    a=np.array(Image.open(SRC/src).convert("RGB"))
    b=np.array(Image.open(OUT/out).convert("RGB"))
    x1,y1,x2,y2=box
    diff=np.abs(a[y1:y2,x1:x2].astype(int)-b[y1:y2,x1:x2].astype(int)).mean()
    var=b[y1:y2,x1:x2].std()
    print(f"{out}: region_mean_abs_diff={diff:.1f}  out_region_std={var:.1f}")
