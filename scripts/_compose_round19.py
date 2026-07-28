"""
Round 19 - 71JAfgkZB9L 复刻：上下分栏「热饮/冰饮」对比
- 1000×1250 竖版
- 上半 40% = 冰茶玻璃杯背景 + 左上红色圆形 icon + 黄色 pill
- 中间 12% = 黄色色带 + 红色衬线大字 + 黑色副标
- 下半 48% = 热茶玻璃杯背景 + 站立正面袋 hero（中央偏右）+ 右下红色圆形 icon + 黄色 pill
- 底部右下角 SC 认证小章
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, os, math

ROOT = "E:/Desktop/茶叶/成品_第十九轮_热冰对比C"
ICED_BG = ROOT + "/_bg/Top_down_45_degree_commercial__2026-07-24T05-01-51.png"  # 冰茶
HOT_BG = ROOT + "/_bg/Top_down_45_degree_commercial__2026-07-24T05-02-09.png"   # 热茶
BAG = ROOT + "/_assets/product_front_real.png"  # kraft 干净袋

W, H = 1000, 1250
TOP_H = 488      # 上半高度（取冰茶顶部 488）
MID_H = 150      # 中间色带高度
BOT_H = H - TOP_H - MID_H  # = 612

# ============== 颜色（按 design_brief §5.2 严格） ==============
AMBER   = (198, 123, 75)   # C67B4B 琥珀棕（标题）
SAGE    = (74, 124, 74)    # 4A7C4A 鼠尾草绿
GOLD    = (212, 168, 83)   # D4A853 暖金
CREAM   = (253, 249, 243)  # FDF9F3 暖白
IVORY   = (245, 240, 230)  # F5F0E6 米白
CHARCOAL= (44, 44, 44)     # 2C2C2C 炭灰
RED     = (200, 35, 45)    # pill/icon 红色（接近 #C8232D）
YELLOW_BAND = (252, 215, 130)  # 中间色带黄 #FCD782

# ============== 字体 ==============
def load_font(path, size):
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()

FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_SANS = "C:/Windows/Fonts/arial.ttf"
FONT_SERIF_BOLD = "C:/Windows/Fonts/georgiab.ttf"
FONT_SERIF = "C:/Windows/Fonts/georgia.ttf"

def text_wh(draw, txt, font):
    if not txt: return (0,0)
    try:
        l, t, r, b = draw.textbbox((0,0), txt, font=font)
        return (r-l, b-t)
    except:
        return draw.textsize(txt, font=font)

# ============== 裁掉水印 + 拼上半 + 下半 ==============
def crop_no_watermark(img, top_n=0, bottom_n=0):
    """top_n / bottom_n = 要裁掉的顶部/底部像素（去掉水印）"""
    w, h = img.size
    return img.crop((0, top_n, w, h - bottom_n))

iced = Image.open(ICED_BG).convert("RGB")
hot  = Image.open(HOT_BG).convert("RGB")

# 冰茶：取顶部 488（避开右下角水印）
iced_top = crop_no_watermark(iced, top_n=0, bottom_n=iced.height - TOP_H * (iced.height // iced.width))
# 实际上水印在右下角靠近底部，我们裁掉底部 110px 防万一
iced_use = iced.crop((0, 0, iced.width, iced.height - 110))  # 1024×914
hot_use  = hot.crop((0, 0, hot.width, hot.height - 110))     # 1024×914

# resize 到目标尺寸
iced_top_resized = iced_use.resize((W, TOP_H), Image.LANCZOS)
bot_resized = hot_use.resize((W, BOT_H), Image.LANCZOS)

# ============== 创建画布 ==============
canvas = Image.new("RGB", (W, H), CREAM)
canvas.paste(iced_top_resized, (0, 0))
canvas.paste(bot_resized, (0, TOP_H + MID_H))

# ============== 中间色带 ==============
band = Image.new("RGB", (W, MID_H), YELLOW_BAND)
draw_band = ImageDraw.Draw(band)

# 大字
title = "DELICIOUS HOT OR ICED"
sub = "Satisfying in Every Season"
fs_title = 56
fs_sub = 26
font_t = load_font(FONT_SERIF_BOLD, fs_title)
font_s = load_font(FONT_SANS, fs_title // 2)

tw, th = text_wh(draw_band, title, font_t)
draw_band.text(((W - tw)//2, 22), title, font=font_t, fill=AMBER)

sw, sh = text_wh(draw_band, sub, font_s)
draw_band.text(((W - sw)//2, 22 + th + 12), sub, font=font_s, fill=CHARCOAL)

# 画两条细装饰线
draw_band.line([(60, 18), (160, 18)], fill=AMBER, width=3)
draw_band.line([(W - 160, 18), (W - 60, 18)], fill=AMBER, width=3)
draw_band.line([(60, MID_H - 12), (160, MID_H - 12)], fill=AMBER, width=3)
draw_band.line([(W - 160, MID_H - 12), (W - 60, MID_H - 12)], fill=AMBER, width=3)

canvas.paste(band, (0, TOP_H))

# ============== 袋 hero 放在下半右侧 ==============
bag = Image.open(BAG).convert("RGBA")
bag_w, bag_h = bag.size
# 下半空位：右侧 ~W*0.55~W，留给袋
target_bag_h = 560  # 袋高接近下半高度的 90%
scale = target_bag_h / bag_h
target_bag_w = int(bag_w * scale)
bag_s = bag.resize((target_bag_w, target_bag_h), Image.LANCZOS)

# 放中央偏右（参考图是 hero 中央，我放到下半中央偏右一点）
bag_x = W - target_bag_w - 30
bag_y = TOP_H + MID_H + (BOT_H - target_bag_h) // 2 + 6  # 略偏下让出顶部 pill

# 接触阴影
shadow = Image.new("RGBA", (W, H), (0,0,0,0))
sd = ImageDraw.Draw(shadow)
sd.ellipse([bag_x + 30, bag_y + target_bag_h - 18,
            bag_x + target_bag_w - 30, bag_y + target_bag_h + 12],
           fill=(0,0,0,80))
shadow = shadow.filter(ImageFilter.GaussianBlur(8))
canvas = canvas.convert("RGBA")
canvas.alpha_composite(shadow)
canvas.alpha_composite(bag_s, (bag_x, bag_y))
canvas = canvas.convert("RGB")

# ============== 左上圆形 icon + 黄色 pill（冰） ==============
def draw_circle_icon(draw, cx, cy, r, letter):
    # 红色实心圆 + 金色细描边
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=RED, outline=AMBER, width=3)
    # 居中粗体字母（白色）
    font_letter = load_font(FONT_SERIF_BOLD, int(r * 1.1))
    lw, lh = text_wh(draw, letter, font_letter)
    draw.text((cx - lw//2, cy - lh//2 - 2), letter, font=font_letter, fill=(255,255,255))

# 上半的 pill（冰饮）
draw = ImageDraw.Draw(canvas)
fs_pill = 18
font_pill = load_font(FONT_SANS, fs_pill)

# 左上角
icon_cx, icon_cy = 110, 270
pill_x = icon_cx + 50
pill_y = icon_cy - 25
pill_text = "Chill with ice for a refreshing\npick-me-up."
pill_lines = pill_text.split("\n")
# 计算 pill 尺寸
line_w = max(text_wh(draw, ln, font_pill)[0] for ln in pill_lines)
line_h = sum(text_wh(draw, ln, font_pill)[1] for ln in pill_lines)
pw, ph = line_w + 36, line_h + 30
draw.rounded_rectangle([pill_x, pill_y, pill_x+pw, pill_y+ph],
                       radius=22, fill=YELLOW_BAND, outline=GOLD, width=2)
# 文字
ty = pill_y + 15
for ln in pill_lines:
    lw, lh = text_wh(draw, ln, font_pill)
    draw.text((pill_x + (pw - lw)//2, ty), ln, font=font_pill, fill=AMBER)
    ty += lh + 2

# icon 在 pill 左侧
draw_circle_icon(draw, icon_cx, icon_cy, 38, "C")

# ============== 右下圆形 icon + 黄色 pill（热饮） ==============
# 下半右侧（袋子在右下，所以 pill 放左下）
icon_cx2, icon_cy2 = 130, TOP_H + MID_H + 470
pill_x2 = icon_cx2 + 50
pill_y2 = icon_cy2 - 30
pill_text2 = "Steep with fresh boiling\nwater for 5 minutes"
pill_lines2 = pill_text2.split("\n")
line_w2 = max(text_wh(draw, ln, font_pill)[0] for ln in pill_lines2)
line_h2 = sum(text_wh(draw, ln, font_pill)[1] for ln in pill_lines2)
pw2, ph2 = line_w2 + 36, line_h2 + 30
draw.rounded_rectangle([pill_x2, pill_y2, pill_x2+pw2, pill_y2+ph2],
                       radius=22, fill=YELLOW_BAND, outline=GOLD, width=2)
ty = pill_y2 + 15
for ln in pill_lines2:
    lw, lh = text_wh(draw, ln, font_pill)
    draw.text((pill_x2 + (pw2 - lw)//2, ty), ln, font=font_pill, fill=AMBER)
    ty += lh + 2
draw_circle_icon(draw, icon_cx2, icon_cy2, 38, "H")

# ============== 底部 SC 认证小章（右下角） ==============
badge_x = W - 280
badge_y = H - 78
badge_w = 260
badge_h = 50
draw.rounded_rectangle([badge_x, badge_y, badge_x+badge_w, badge_y+badge_h],
                       radius=8, fill=CREAM, outline=GOLD, width=1)
fs_badge = 12
font_badge = load_font(FONT_SANS, fs_badge)
sc_text1 = "SC 1143552406532  ·  GH/T 1091"
sc_text2 = "Fujian Anxi  ·  24-Month Shelf Life"
tw1, th1 = text_wh(draw, sc_text1, font_badge)
tw2, th2 = text_wh(draw, sc_text2, font_badge)
draw.text((badge_x + (badge_w - tw1)//2, badge_y + 6), sc_text1, font=font_badge, fill=CHARCOAL)
draw.text((badge_x + (badge_w - tw2)//2, badge_y + 6 + th1 + 2), sc_text2, font=font_badge, fill=AMBER)

# ============== 保存 ==============
out_path = ROOT + "/01_hot_or_iced.png"
canvas.save(out_path, "PNG", optimize=True)
print(f"SAVED {out_path} {canvas.size}")