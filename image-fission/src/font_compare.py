from PIL import Image, ImageDraw, ImageFont
import os

# Candidate bold fonts
candidates = [
    (r'C:/Windows/Fonts/ariblk.ttf', 'Arial Black'),
    (r'C:/Windows/Fonts/seguibl.ttf', 'Segoe UI Black'),
    (r'C:/Windows/Fonts/georgiab.ttf', 'Georgia Bold'),
    (r'C:/Windows/Fonts/cambriab.ttf', 'Cambria Bold'),
    (r'C:/Windows/Fonts/timesbd.ttf', 'Times Bold'),
    (r'C:/Windows/Fonts/impact.ttf', 'Impact'),
    (r'C:/Windows/Fonts/STENCIL.TTF', 'STENCIL'),
    (r'E:/Desktop/双接口/image-fission/ComfyUI/models/fonts/Anton-Regular.ttf', 'Anton'),
    (r'E:/Desktop/双接口/image-fission/ComfyUI/models/fonts/AbrilFatface-Regular.ttf', 'AbrilFatface'),
]

target_h = 99
samples = ['ARMED', 'FORCES', 'BRAVE', 'LEGION']

# Find fsize for each font so 'H' cap height matches target
font_size_map = {}
for path, name in candidates:
    if not os.path.exists(path):
        continue
    for fsize in range(60, 250):
        font = ImageFont.truetype(path, fsize)
        hbbox = font.getbbox('H')
        hh = hbbox[3] - hbbox[1]
        if hh >= target_h:
            font_size_map[name] = (path, fsize)
            break

# Build comparison image
img = Image.new('RGB', (2100, 60 + 80*len(font_size_map)), (245, 245, 245))
draw = ImageDraw.Draw(img)
draw.text((10, 8), f'capH target={target_h} (ARMED/FORCES band height)', fill=(0, 100, 0),
          font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16))
y = 50
for name, (path, fsize) in font_size_map.items():
    font = ImageFont.truetype(path, fsize)
    line = '  |  '.join(samples)
    bbox = draw.textbbox((0, 0), line, font=font)
    text_h = bbox[3] - bbox[1]
    draw.text((10, y - bbox[1]), line, font=font, fill='black')
    # Measure width of 'ARMED'
    aw = font.getbbox('ARMED')[2] - font.getbbox('ARMED')[0]
    fw = font.getbbox('FORCES')[2] - font.getbbox('FORCES')[0]
    draw.text((1800, y + 10),
              f'{name} fsize={fsize}',
              fill='red', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16))
    draw.text((1800, y + 30),
              f'ARMED w={aw} FORCES w={fw}',
              fill='blue', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14))
    y += 80

out = r'E:/Desktop/双接口/image-fission/jobs/v319b/font_compare_all.png'
img.save(out)
print(f'Saved {out}')
print(f'{len(font_size_map)} fonts compared:')
for name, (path, fsize) in font_size_map.items():
    font = ImageFont.truetype(path, fsize)
    aw = font.getbbox('ARMED')[2] - font.getbbox('ARMED')[0]
    fw = font.getbbox('FORCES')[2] - font.getbbox('FORCES')[0]
    bw = font.getbbox('BRAVE')[2] - font.getbbox('BRAVE')[0]
    print(f'  {name:20s} fsize={fsize:3d} ARMED_w={aw:4d} FORCES_w={fw:4d} BRAVE_w={bw:4d}')