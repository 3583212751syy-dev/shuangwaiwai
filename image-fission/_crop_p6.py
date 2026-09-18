import numpy as np
from PIL import Image
from scipy import ndimage as ndi

def crop(p, box):
    im = Image.open(p).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = [int(round(v)) for v in (box[0]*W, box[1]*H, box[2]*W, box[3]*H)]
    return np.asarray(im.crop((x0, y0, x1, y1)), np.float32)

def sharp(a):
    g = a @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(ndi.laplace(g).var())

BOX = (0.24, 0.24, 0.76, 0.50)   # eagle region
o = crop("E:/Desktop/图裂变测试图/Pinterest (6).jpg", BOX)
m = crop("jobs/v386_p6mode1/01_custom_1.jpg", BOX)
print("ORIG eagle crop", o.shape, "Laplacian var %.1f  lum mean %.1f" % (sharp(o), (o@np.array([.299,.587,.114],np.float32)).mean()))
print("MODE1 eagle crop", m.shape, "Laplacian var %.1f  lum mean %.1f" % (sharp(m), (m@np.array([.299,.587,.114],np.float32)).mean()))
# high-freq energy = mean |grad|
def hf(a):
    g = a @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(np.abs(ndi.sobel(g, 0)).mean() + np.abs(ndi.sobel(g, 1)).mean())
print("ORIG HF %.2f | MODE1 HF %.2f" % (hf(o), hf(m)))

# side-by-side at 1:1 (native pixels, no resize)
h = min(o.shape[0], m.shape[0]); w = min(o.shape[1], m.shape[1])
o2 = o[:h, :w].astype(np.uint8); m2 = m[:h, :w].astype(np.uint8)
gap = np.full((h, 16, 3), 255, np.uint8)
out = np.concatenate([o2, gap, m2], 1)
Image.fromarray(out, "RGB").save("jobs/_cmpcrop/p6_eagle1to1.jpg", quality=95)
print("saved jobs/_cmpcrop/p6_eagle1to1.jpg", out.shape)
