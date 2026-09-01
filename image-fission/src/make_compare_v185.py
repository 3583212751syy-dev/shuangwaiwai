"""
v185 裂变对照拼图生成器
------------------------------
读取 smoke_v185_batch_18.py 的 id->ref_img 映射，做：
  1. 全部 18 张：源图 | 裂变结果 双列网格 -> jobs/smoke_v185/_compare_all.jpg
  2. 3 张侵权图：源图 | 裂变(含擦除前) | 烧字后 三列 -> jobs/smoke_v185/_compare_burned.jpg
运行：python src/make_compare_v185.py
"""
import os, re
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "src", "smoke_v185_batch_18.py")
JOB = os.path.join(ROOT, "jobs", "smoke_v185")
INPUT = os.path.join(ROOT, "ComfyUI", "input")
FONT = os.path.join(ROOT, "fonts", "Rye-Regular.ttf")
BURNED = ["fireball_skull", "bat_logo", "camo_armed"]
COL_W = 760
PAD = 12
LABEL_H = 34


def get_pairs():
    txt = open(SCRIPT, encoding="utf-8").read()
    return re.findall(r'"id":\s*"([^"]+)",\s*"ref_img":\s*"([^"]+)"', txt)


def load_fit(path, w):
    im = Image.open(path).convert("RGB")
    r = w / im.width
    return im.resize((w, int(im.height * r)))


def labeled(im, text):
    canvas = Image.new("RGB", (im.width, im.height + LABEL_H), (20, 20, 20))
    canvas.paste(im, (0, LABEL_H))
    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, 22)
    d.text((10, 6), text, font=f, fill=(230, 230, 230))
    return canvas


def grid(rows, cols, title):
    # rows: list of list of images (same width COL_W)
    heights = [max(im.height for im in r) for r in rows]
    W = cols * COL_W + (cols + 1) * PAD
    H = sum(heights) + (len(rows) + 1) * PAD + 40
    canvas = Image.new("RGB", (W, H), (12, 12, 12))
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 8), title, font=ImageFont.truetype(FONT, 26), fill=(240, 240, 240))
    y = 40
    for r, rh in zip(rows, heights):
        x = PAD
        for im in r:
            canvas.paste(im, (x, y + (rh - im.height) // 2))
            x += COL_W + PAD
        y += rh + PAD
    return canvas


def main():
    pairs = get_pairs()
    print(f"解析到 {len(pairs)} 张映射")
    all_rows = []
    burned_rows = []
    for cid, ref in pairs:
        src = os.path.join(INPUT, ref)
        res = os.path.join(JOB, f"v185_{cid}.jpg")
        if not os.path.exists(src) or not os.path.exists(res):
            print(f"  [skip] 缺文件 {cid}")
            continue
        s = labeled(load_fit(src, COL_W), f"SRC {cid}")
        r = labeled(load_fit(res, COL_W), f"v185 {cid}")
        all_rows.append([s, r])
        if cid in BURNED:
            b = os.path.join(JOB, f"v185_{cid}_burned.jpg")
            if os.path.exists(b):
                bb = labeled(load_fit(b, COL_W), f"BURNED {cid}")
                burned_rows.append([s, r, bb])
    out = grid(all_rows, 2, "v185 图裂变对照 (源图 | 裂变结果)")
    p = os.path.join(JOB, "_compare_all.jpg")
    out.save(p, quality=92)
    print(f"  全量对照 -> {p}  ({out.width}x{out.height})")
    if burned_rows:
        out2 = grid(burned_rows, 3, "v185 侵权文字烧字对照 (源图 | 裂变 | 烧字后)")
        p2 = os.path.join(JOB, "_compare_burned.jpg")
        out2.save(p2, quality=92)
        print(f"  烧字对照 -> {p2}  ({out2.width}x{out2.height})")


if __name__ == "__main__":
    main()
