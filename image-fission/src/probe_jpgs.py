from PIL import Image
import os
d = r"E:\Desktop\双接口\image-fission\jobs\smoke_v126_1787808430"
for f in sorted(os.listdir(d)):
    if not f.endswith(".jpg"):
        continue
    p = os.path.join(d, f)
    try:
        im = Image.open(p)
        im.load()
        print(f"{f}: OK {im.size} mode={im.mode} fmt={im.format}")
    except Exception as e:
        print(f"{f}: FAIL {e}")
