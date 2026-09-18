from PIL import Image

base = r"E:/Desktop/双接口/image-fission/jobs/v381_p3final/p3_base.jpg"
orig = r"E:/Desktop/图裂变测试图/Pinterest (3).jpg"
OUT = r"E:/Desktop/双接口/image-fission/jobs/v381_p3final"

b = Image.open(base)
o = Image.open(orig)
print("base", b.size, "orig", o.size)
W, H = o.size


def crop(im, box, name, scale=3):
    c = im.crop(box)
    c = c.resize((c.width * scale, c.height * scale), Image.LANCZOS)
    c.save(f"{OUT}/{name}")
    print(name, c.size)


box_top = (int(W * 0.55), int(H * 0.24), W, int(H * 0.42))
crop(o, box_top, "Z_top_orig.jpg")
crop(b, box_top, "Z_top_new.jpg")

box_bot = (0, int(H * 0.76), int(W * 0.45), H)
crop(o, box_bot, "Z_bot_orig.jpg")
crop(b, box_bot, "Z_bot_new.jpg")

box_main = (0, int(H * 0.40), W, int(H * 0.80))
crop(o, box_main, "Z_main_orig.jpg", 2)
crop(b, box_main, "Z_main_new.jpg", 2)
