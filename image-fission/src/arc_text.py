"""
arc_text.py — 沿弧路径绘制文字（用于替换环形 logo 文字）

依赖：Pillow（已装），无需 numpy
公开函数：
  draw_arc_text(img_pil, text, font_path, font_size, color, center, radius,
                start_angle_deg, end_angle_deg, char_spacing_deg=0.05,
                flip_180=False) -> PIL Image
  fit_arc_text_width(text, font_path, font_size, radius) -> 文本占用的弧角度数（度）

坐标系：PIL 默认 +x=右 +y=下；角度 0°=3点钟方向（右侧），逆时针为正（数学惯例）
       旋转角 = 字符中心角 + 90°（让字符底部朝圆心，文字正立读）
       flip_180=True 时让字符底部朝外（文字头朝外侧）
"""

import math
from PIL import Image, ImageDraw, ImageFont


def _char_advance(font, ch):
    """单字符前移宽度 = getlength（更准确，包含 advance）"""
    adv = int(round(font.getlength(ch)))
    bbox = font.getbbox(ch)
    char_w = bbox[2] - bbox[0]
    return adv, char_w


def fit_arc_text_width(text, font_path, font_size, radius, char_spacing_px=4):
    """返回文字在指定半径上占用的总弧长（像素）"""
    font = ImageFont.truetype(font_path, font_size)
    total = 0
    for i, ch in enumerate(text):
        adv, _ = _char_advance(font, ch)
        total += adv
        if i < len(text) - 1:
            total += char_spacing_px
    return total


def draw_arc_text(img_pil, text, font_path, font_size, color,
                  center, radius, start_angle_deg, end_angle_deg,
                  char_spacing_px=4, flip_180=False):
    """
    沿弧路径绘制 text。
    img_pil: PIL Image (RGB 或 RGBA)
    返回：绘制后的 PIL Image（原图被修改）

    color: (r,g,b) 或 (r,g,b,a)
    """
    if isinstance(color, tuple) and len(color) == 3:
        color_rgba = color + (255,)
    else:
        color_rgba = color

    font = ImageFont.truetype(font_path, font_size)

    # 字符宽度表
    advances = []
    for ch in text:
        adv, _ = _char_advance(font, ch)
        advances.append(adv)

    # 总弧长 + 字符间距
    n_chars = len(text)
    total_arc_len = sum(advances) + char_spacing_px * max(0, n_chars - 1)

    # 弧的角度范围（输入 start → end，方向是 start → end）
    arc_deg = (end_angle_deg - start_angle_deg) % 360
    if arc_deg == 0 and start_angle_deg != end_angle_deg:
        arc_deg = 360

    # 计算每个字符中心角（沿弧均匀分布，居中在 [start, end] 区间内）
    char_angles = []
    cur_len = 0
    mid_angle_deg = (start_angle_deg + end_angle_deg) / 2.0
    deg_per_px_d = math.degrees(1.0 / radius)
    total_deg = total_arc_len * deg_per_px_d
    start_offset_deg = -total_deg / 2.0  # 居中：起点在 mid - total/2

    for i, adv in enumerate(advances):
        char_center_len = cur_len + adv / 2
        center_deg_offset = char_center_len * deg_per_px_d
        center_deg = mid_angle_deg + start_offset_deg + center_deg_offset
        char_angles.append(center_deg)
        cur_len += adv + (char_spacing_px if i < n_chars - 1 else 0)

    # 确保总弧长不超出给定范围（如果超出，文字会超出范围；这是 caller 的责任）
    # 这里不再自动缩放

    cx, cy = center
    img = img_pil.copy()
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    for i, ch in enumerate(text):
        adv = advances[i]
        deg = char_angles[i]
        rad = math.radians(deg)

        # 字符位置（弧上）
        x = cx + radius * math.cos(rad)
        y = cy + radius * math.sin(rad)

        # 字符旋转角（让字符底部朝向圆心 = 正立读）
        # PIL rotate 是逆时针为正，所以这里取 -deg - 90 (for flip_180=False)
        # 但 PIL 旋转也是 (x,y)→(x*cos - y*sin, x*sin + y*cos)
        # 字符切线方向：垂直于半径
        # 字符底部朝向圆心：rotate angle = -deg - 90 + 180 (180 让字符翻转到正立方向)
        # 简化：rotate = -deg - 90 让字符沿切线方向（底部朝圆心），
        # 但文字"上"是反的，所以再加 180 让字头朝外
        if flip_180:
            rotate_deg = -deg - 90 + 180
        else:
            rotate_deg = -deg - 90

        # 画单字符到透明图层
        ch_w = adv + 8
        ch_h = font_size * 2 + 8
        ch_img = Image.new("RGBA", (ch_w, ch_h), (0, 0, 0, 0))
        ch_draw = ImageDraw.Draw(ch_img)
        # text 位置：稍微居中
        ch_draw.text((4, ch_h // 4), ch, font=font, fill=color_rgba)
        # 旋转
        ch_img = ch_img.rotate(rotate_deg, resample=Image.BICUBIC, expand=True)
        # paste 到目标位置（中心对齐）
        px = int(round(x - ch_img.width / 2))
        py = int(round(y - ch_img.height / 2))
        img.paste(ch_img, (px, py), ch_img)

    return img


if __name__ == "__main__":
    # 测试：画一段弧字
    import os
    img = Image.new("RGB", (1552, 2000), (200, 180, 200))  # 粉紫底模拟
    out = draw_arc_text(
        img,
        text="NEW BAT CAVE",
        font_path="fonts/Lora-VF.ttf",
        font_size=72,
        color=(20, 10, 30),
        center=(776, 600),
        radius=450,
        start_angle_deg=210,  # 左下 7-8 点
        end_angle_deg=330,    # 右下 4-5 点
        char_spacing_px=8,
        flip_180=False,
    )
    out_rgb = out.convert("RGB")
    out_path = f"jobs/_test_arc_text_{os.getpid()}.jpg"
    out_rgb.save(out_path, quality=90)
    print(f"saved {out_path}")