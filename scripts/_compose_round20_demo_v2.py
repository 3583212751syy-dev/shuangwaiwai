from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from pathlib import Path
from scipy import ndimage

out = Path(r"E:\Desktop\茶叶\成品_第二十轮_参考复刻")
bg = Image.open(out / "Create_a_premium_e_commerce_de_2026-07-28T03-35-55.png").convert("RGBA")
W,H = bg.size
# cover watermark area with matching warm panel
d = ImageDraw.Draw(bg)
d.rectangle((690, 942, 1024, 1024), fill=(255, 247, 202, 255))

# Use the original standing pouch silhouette, then replace the AI label with the real label artwork.
raw = Image.open(r"E:\Desktop\茶叶\成品_样稿_confirm\_jimeng_front.png").convert("RGBA")
a = np.array(raw)
# Fixed silhouette mask avoids importing the studio shadow or white background.
mask_img = Image.new("L", raw.size, 0)
md = ImageDraw.Draw(mask_img)
md.rounded_rectangle((260, 110, 764, 940), radius=28, fill=255)
rect = np.array(mask_img) > 0
rgb = a[:,:,:3]
mask = rect & (rgb[:,:,0] < 225) & (rgb[:,:,1] < 220) & (rgb[:,:,2] < 210)
mask = ndimage.binary_closing(mask, iterations=4)
mask = ndimage.binary_fill_holes(mask)
# discard the original cast shadow region below the bag base
mask[930:,:] = False
ys,xs = np.where(mask)
cut = raw.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
cm = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
ca = np.array(cut)
ca[:,:,3] = (cm*255).astype(np.uint8)
cut = Image.fromarray(ca, "RGBA")
# Real label, scaled to the label area of the pouch.
lbl = Image.open(r"E:\Desktop\茶叶\成品_样稿_confirm\_label.png").convert("RGBA")
la = np.array(lbl); ys2,xs2 = np.where(la[:,:,3]>10)
lbl = lbl.crop((xs2.min(),ys2.min(),xs2.max()+1,ys2.max()+1))
# source pouch label box in cropped coordinates (measured from _jimeng_front)
# crop starts near x=260,y=110, label approx x=355..660,y=355..825
cx0,cy0 = xs.min(), ys.min()
label_box = (355-cx0, 355-cy0, 660-cx0, 824-cy0)
lw = label_box[2]-label_box[0]; lh = label_box[3]-label_box[1]
ratio = lbl.width/lbl.height
if lw/lh > ratio: tw,th = int(lh*ratio), lh
else: tw,th = lw, int(lw/ratio)
lbl = lbl.resize((tw,th), Image.Resampling.LANCZOS)
cut.alpha_composite(lbl, (label_box[0]+(lw-tw)//2, label_box[1]+(lh-th)//2))
# resize hero pouch
max_h=585
sc=max_h/cut.height
cut=cut.resize((int(cut.width*sc),max_h), Image.Resampling.LANCZOS)
px,py=520,325
# clear the generated placeholder pouch behind the real product
bg.paste((255,247,202,255),(425,280,1024,930))
shadow=Image.new("RGBA",bg.size,(0,0,0,0)); sd=ImageDraw.Draw(shadow)
sd.ellipse((px-25,py+max_h-25,px+cut.width+30,py+max_h+35),fill=(90,55,30,80))
shadow=shadow.filter(ImageFilter.GaussianBlur(18))
bg=Image.alpha_composite(bg,shadow); bg.alpha_composite(cut,(px,py))

d=ImageDraw.Draw(bg); fonts=Path(r"C:\Windows\Fonts")
def F(name,size): return ImageFont.truetype(str(fonts/name),size)
red=(174,37,28,255); dark=(76,47,30,255); cream=(255,247,211,255)
for text,y in [("NATURAL ROOIBOS",52),("GOODNESS",118)]:
    f=F("arialbd.ttf",58); b=d.textbbox((0,0),text,font=f); d.text(((W-(b[2]-b[0]))/2,y),text,font=f,fill=cream)
f=F("georgia.ttf",27); sub="A smooth herbal tea for daily relaxation"; b=d.textbbox((0,0),sub,font=f); d.text(((W-(b[2]-b[0]))/2,188),sub,font=f,fill=dark)
for text,y in [("100% NATURAL",320),("CAFFEINE-FREE",405),("GENTLE HERBAL BLEND",490),("HOT OR ICED",575)]:
    f=F("arialbd.ttf",24); d.ellipse((62,y,94,y+32),fill=cream); d.text((108,y+3),text,font=f,fill=cream)
f3=F("arial.ttf",16); d.text((54,972),"BREW YOUR MOMENT",font=f3,fill=(117,73,39,255))
bg.convert("RGB").save(out / "01_reference_style_demo.png",quality=95)
print(out / "01_reference_style_demo.png")
