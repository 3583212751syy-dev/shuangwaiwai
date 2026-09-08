from PIL import Image, ImageDraw, ImageFont
import numpy as np

font = ImageFont.truetype(r'C:/Windows/Fonts/ariblk.ttf', 71)
text = 'WE HONOR'
bbox = font.getbbox(text)
print(f'font.getbbox: {bbox} w={bbox[2]-bbox[0]} h={bbox[3]-bbox[1]}')
print(f'  font metrics: ascent={font.getmetrics()[0]} descent={font.getmetrics()[1]}')

img = Image.new('RGBA', (600, 200), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
draw.text((50, 50), text, font=font, fill=(20, 20, 20, 255))
arr = np.array(img)
# R,G,B all < 80 (text is dark, transparent is 0)
mask = (arr[:,:,0] < 80) & (arr[:,:,3] > 200)
ys, xs = np.where(mask)
if len(ys) > 0:
    print(f'rendered text bbox: y={ys.min()}..{ys.max()} x={xs.min()}..{xs.max()}  height={ys.max()-ys.min()}')
img.save(r'E:\Desktop\双接口\image-fission\jobs\v319o\debug_text_layer.png')
