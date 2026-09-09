from PIL import Image
import numpy as np
import os
import datetime

# Check file timestamps
for f in ['cleaned.png', 'fission.jpg', 'compare.jpg', 'debug_text_mask.png']:
    path = rf'E:\Desktop\双接口\image-fission\jobs\v319v_bat_style\{f}'
    st = os.path.getmtime(path)
    print(f'{f}: {datetime.datetime.fromtimestamp(st).strftime("%H:%M:%S")} ({os.path.getsize(path)} bytes)')

print()
# Check fission.jpg pixel at B position
img = np.array(Image.open(r'E:\Desktop\双接口\image-fission\jobs\v319v_bat_style\fission.jpg').convert('RGB'))
print('fission.jpg pixel at B (425,850):', img[850, 425])
print('fission.jpg pixel at B (430,860):', img[860, 430])
print('fission.jpg pixel at B (445,890):', img[890, 445])
print('fission.jpg pixel at MCKEART (600,1080):', img[1080, 600])
print('fission.jpg pixel at MCKEART (550,1060):', img[1060, 550])

# Also check cleaned
clean = np.array(Image.open(r'E:\Desktop\双接口\image-fission\jobs\v319v_bat_style\cleaned.png').convert('RGB'))
print()
print('cleaned pixel at B (425,850):', clean[850, 425])
print('cleaned pixel at B (430,860):', clean[860, 430])
print('cleaned pixel at MCKEART (600,1080):', clean[1080, 600])
