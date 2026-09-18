from PIL import Image, ImageDraw, ImageFilter, ImageFont
import math, os

OUT = r"e:\Desktop\双接口\codex\generated\tea_fission_result.png"
img = Image.new('RGBA', (1600, 1200), (243, 232, 214, 255))
d = ImageDraw.Draw(img)

# Warm wood background
for y in range(0, 1200, 24):
    for x in range(0, 1600, 24):
        base = 190 + ((x * 17 + y * 11) % 40)
        d.rectangle([x, y, x + 24, y + 24], fill=(base, 150 + (x % 25), 110 + (y % 20), 255))

# Glow overlays
for cx, cy, r, color in [
    (220, 120, 170, (255, 226, 170, 120)),
    (1320, 150, 200, (255, 209, 125, 100)),
    (1180, 980, 170, (235, 170, 105, 110)),
    (300, 980, 150, (255, 212, 168, 90)),
]:
    glow = Image.new('RGBA', (1600, 1200), (0,0,0,0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((cx-r, cy-r, cx+r, cy+r), fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(40))
    img = Image.alpha_composite(img, glow)

# panel dividers
pb = ImageDraw.Draw(img)
pb.rectangle([30, 30, 790, 580], outline=(255,255,255,180), width=4)
pb.rectangle([810, 30, 1570, 580], outline=(255,255,255,180), width=4)
pb.rectangle([30, 620, 790, 1170], outline=(255,255,255,180), width=4)
pb.rectangle([810, 620, 1570, 1170], outline=(255,255,255,180), width=4)

# Fonts
font_title = ImageFont.truetype(r"C:\\Windows\\Fonts\\Georgia.ttf", 32)
font_sub = ImageFont.truetype(r"C:\\Windows\\Fonts\\Georgia.ttf", 18)
font_small = ImageFont.truetype(r"C:\\Windows\\Fonts\\Georgia.ttf", 15)
font_med = ImageFont.truetype(r"C:\\Windows\\Fonts\\Georgia.ttf", 18)


def draw_pouch_panel(panel, x, y, angle, size=1.0, mood='hero'):
    bag = Image.new('RGBA', (420, 610), (0,0,0,0))
    bd = ImageDraw.Draw(bag)
    # bag body
    bd.rounded_rectangle([30, 10, 390, 590], radius=28, fill=(182, 128, 82, 255))
    # grain
    for i in range(0, 360, 10):
        bd.line((20 + i % 200, 30 + i//3 % 50, 30 + i % 200, 520), fill=(220, 190, 130, 18), width=1)
    # top zip strip
    bd.rounded_rectangle([80, 6, 340, 62], radius=15, fill=(120, 90, 58, 255))
    bd.ellipse((190, 10, 230, 52), fill=(220, 202, 166, 255))
    # label
    bd.rounded_rectangle([60, 120, 360, 420], radius=20, fill=(245, 237, 223, 255))
    # inner herb strip
    bd.rectangle([80, 140, 340, 155], fill=(160, 180, 120, 180))
    # badge
    bd.ellipse((140, 160, 280, 290), fill=(227, 205, 129, 200))
    # text
    bd.text((95, 95), 'TEVOYATEA', font=font_title, fill=(130, 92, 44, 255))
    bd.text((140, 135), 'Rooibos Tea', font=font_sub, fill=(130, 92, 44, 255))
    # strap text
    bd.multiline_text((105, 250), 'A blend of natural herbs\nto help relieve stress', font=font_small, fill=(68, 88, 65, 210), spacing=4, align='center')
    # cup
    cup = Image.new('RGBA', (145, 80), (0,0,0,0))
    cd = ImageDraw.Draw(cup)
    cd.rounded_rectangle([12, 12, 118, 60], radius=18, fill=(255, 255, 255, 255))
    cd.ellipse((30, 24, 98, 52), fill=(188, 93, 36, 200))
    bag.paste(cup, (130, 350), cup)
    # natural seal
    bd.rounded_rectangle([110, 450, 310, 485], radius=10, fill=(110, 160, 110, 220))
    bd.text((112, 452), '100%\nNATURAL', font=font_med, fill=(222, 186, 90, 255), spacing=2, align='left')
    bd.text((112, 515), 'NET WT: 100g (3.53oz)', font=font_small, fill=(118, 94, 62, 200))

    rotated = bag.rotate(angle, expand=True, fillcolor=(0,0,0,0))
    if mood == 'hero':
        panel.paste(rotated, (95, 70), rotated)
    elif mood == 'macro':
        panel.paste(rotated, (110, 55), rotated)
    elif mood == 'lifestyle':
        panel.paste(rotated, (90, 65), rotated)
    else:
        panel.paste(rotated, (110, 75), rotated)

    # add small props per panel
    if mood == 'hero':
        herb = Image.new('RGBA', (140, 140), (0,0,0,0))
        hd = ImageDraw.Draw(herb)
        hd.line((20, 110, 70, 20), fill=(125, 150, 90, 200), width=2)
        hd.line((70, 110, 90, 20), fill=(125, 150, 90, 200), width=2)
        hd.ellipse((38, 12, 62, 34), fill=(172, 182, 98, 200))
        panel.paste(herb, (15, 350), herb)
    elif mood == 'macro':
        # detail label closeup effect
        faded = Image.new('RGBA', (180, 150), (0,0,0,0))
        fd = ImageDraw.Draw(faded)
        fd.rounded_rectangle([10,10,170,140], radius=15, fill=(238, 228, 206, 200))
        fd.text((34, 38), 'TEVOYATEA', font=font_title, fill=(128, 90, 40, 220))
        panel.paste(faded, (400, 110), faded)
    elif mood == 'lifestyle':
        # tea cup and ingredients
        tea_cup = Image.new('RGBA', (180, 120), (0,0,0,0))
        td = ImageDraw.Draw(tea_cup)
        td.ellipse((12, 30, 150, 100), fill=(173, 92, 28, 230))
        tea_cup = tea_cup.rotate(8)
        panel.paste(tea_cup, (420, 290), tea_cup)
        pom = Image.new('RGBA', (120, 120), (0,0,0,0))
        pd = ImageDraw.Draw(pom)
        pd.ellipse((10,10,110,110), fill=(139, 32, 24, 220))
        for a in range(0, 360, 35):
            ax = 55 + int(math.cos(math.radians(a)) * 30)
            ay = 55 + int(math.sin(math.radians(a)) * 30)
            pd.line((55,55,ax,ay), fill=(193, 42, 48, 180), width=4)
        panel.paste(pom, (470, 330), pom)
    else:
        # editorial mood
        pom = Image.new('RGBA', (130, 130), (0,0,0,0))
        pd = ImageDraw.Draw(pom)
        pd.ellipse((12,12,118,118), fill=(130, 24, 22, 230))
        for a in range(0, 360, 28):
            ax = 60 + int(math.cos(math.radians(a)) * 30)
            ay = 60 + int(math.sin(math.radians(a)) * 30)
            pd.line((60,60,ax,ay), fill=(200, 42, 50, 200), width=4)
        panel.paste(pom, (470, 360), pom)

# Build final panels
panel_positions = [
    (40, 40, 'hero', -5),
    (820, 40, 'macro', 7),
    (40, 640, 'lifestyle', -7),
    (820, 640, 'editorial', 6)
]

for x, y, mood, angle in panel_positions:
    panel = Image.new('RGBA', (720, 520), (0,0,0,0))
    pbg = ImageDraw.Draw(panel)
    if mood == 'hero':
        pbg.rectangle([0, 0, 720, 520], fill=(245, 234, 210, 255))
    elif mood == 'macro':
        pbg.rectangle([0, 0, 720, 520], fill=(236, 223, 194, 255))
    elif mood == 'lifestyle':
        pbg.rectangle([0, 0, 720, 520], fill=(238, 240, 235, 255))
    else:
        pbg.rectangle([0, 0, 720, 520], fill=(226, 212, 198, 255))
    # soft blur effect
    blurred = Image.new('RGBA', (720, 520), (0,0,0,0))
    b = ImageDraw.Draw(blurred)
    b.ellipse((60, 30, 580, 420), fill=(245, 203, 120, 90))
    panel = Image.alpha_composite(panel, blurred)
    draw_pouch_panel(panel, 0, 0, angle, 1.0, mood)
    img.alpha_composite(panel, (x, y))

# header title at the top center
header = Image.new('RGBA', (360, 75), (0,0,0,0))
hd = ImageDraw.Draw(header)
hd.rounded_rectangle([0,0,360,75], radius=18, fill=(255,255,255,150))
hd.text((80, 16), 'Rooibos Tea', font=ImageFont.truetype(r"C:\\Windows\\Fonts\\Georgia.ttf", 28), fill=(118, 86, 48, 255))
img.alpha_composite(header, (620, 10))

img = img.convert('RGB')
img.save(OUT)
print(OUT)
