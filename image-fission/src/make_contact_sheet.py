"""bat_logo 裂变候选对比总表（2x2 grid）。

把 Original / v226 / v227 / v228 拼到一张图，便于用户一次目检对比。
用法: ./venv/Scripts/python.exe src/make_contact_sheet.py
"""
from PIL import Image, ImageDraw, ImageFont

PROJECT = "E:/Desktop/双接口/image-fission"
ORIG = f"{PROJECT}/ComfyUI/input/test_6978fabda2cc99629fa9e81f802762d3.jpg"

CANDS = [
    ("ORIGINAL (参考图)", ORIG),
    ("v226  (clean PASS, edge 1.12x, 1 紫误报)", f"{PROJECT}/jobs/smoke_v226/v226_bat_logo.jpg"),
    ("v227  (edge FAIL 1.39x, 动态姿态)", f"{PROJECT}/jobs/smoke_v227/v227_bat_logo.jpg"),
    ("v228  (zero-flag PASS, edge 1.23x, 推荐)", f"{PROJECT}/jobs/smoke_v228/v228_bat_logo.jpg"),
]

H = 1500          # 每格图高
GAP = 30          # 格间距
LABEL_H = 56      # 标签栏高


def try_font(size):
    for p in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageDraw.getfont()


def build_cell(name, path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = H / h
    im = im.resize((int(w * scale), H))
    canvas = Image.new("RGB", (im.width, H + LABEL_H), "white")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((12, H + 10), name, fill=(20, 20, 20), font=try_font(34))
    return canvas


def main():
    cells = [build_cell(n, p) for n, p in CANDS]
    cols, rows = 2, 2
    cell_w = max(c.width for c in cells)
    cell_h = max(c.height for c in cells)
    sheet = Image.new(
        "RGB",
        (cols * cell_w + (cols + 1) * GAP, rows * cell_h + (rows + 1) * GAP),
        "white",
    )
    for idx, c in enumerate(cells):
        r, col = divmod(idx, cols)
        x = GAP + col * (cell_w + GAP)
        y = GAP + r * (cell_h + GAP)
        sheet.paste(c, (x, y))
    out = f"{PROJECT}/jobs/contact_sheet_bat_logo.jpg"
    sheet.save(out, quality=95)
    print(f"saved {out} size={sheet.size}")


if __name__ == "__main__":
    main()
