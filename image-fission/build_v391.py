# -*- coding: utf-8 -*-
"""build_v391.py — 第18轮交付（用户否决 p4"一块块" + p6"错乱" 后的返修）

只交付被否决的两张：
  · pinterest4 迷彩棕榈 → 代码 camo_palm_swap（整棵仿射换位，保线不失真）
  · pinterest6 鹰骷髅   → 代码 eagle_skull_canny（Canny 结构引导 SDXL 重生，逻辑正常）

产出：E:\Desktop\v391_两图返修_第18轮\
  1_pinterest4_迷彩棕榈_仿射换位定稿.jpg
  2_pinterest6_鹰骷髅_canny重生定稿.jpg
  z1_p4_原图vs成品.jpg      （1:1 全幅对照）
  z2_p6_原图vs成品.jpg
  QC_指标.txt
  代码映射_五图对应.txt
"""
import os
import sys
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = "E:/Desktop/图裂变测试图"
OUT = "E:/Desktop/v391_两图返修_第18轮"
os.makedirs(OUT, exist_ok=True)

P4_O = os.path.join(SRC, "Pinterest (4).jpg")
P4_N = os.path.join(ROOT, "jobs/router_out_v329/Pinterest (4)_variant.jpg")
P6_O = os.path.join(SRC, "Pinterest (6).jpg")
P6_N = os.path.join(ROOT, "jobs/router_out_v329/pinterest6_variant.jpg")

try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
except Exception:
    FONT = FONT_S = ImageFont.load_default()


def side_by_side(o_path, n_path, out_path, max_h=1400, labels=("原图 ORIG", "成品 NEW")):
    """1:1 全幅并排（等比例缩到同一高度，不裁切）。"""
    o = Image.open(o_path).convert("RGB")
    n = Image.open(n_path).convert("RGB")
    h = min(max_h, o.height, n.height)
    ow = int(o.width * h / o.height)
    nw = int(n.width * h / n.height)
    oc = o.resize((ow, h), Image.LANCZOS)
    nc = n.resize((nw, h), Image.LANCZOS)
    gap = 14
    bar = 30
    cv = Image.new("RGB", (ow + nw + gap * 3, h + bar + gap), (24, 24, 28))
    dr = ImageDraw.Draw(cv)
    x = gap
    for lab, im in zip(labels, (oc, nc)):
        cv.paste(im, (x, bar))
        dr.text((x + 4, 6), lab, fill=(240, 240, 240), font=FONT)
        x += im.width + gap
    cv.save(out_path, quality=94)
    return cv.size


def metrics(tag, o_path, n_path, subject="lum<60"):
    from skimage.color import rgb2lab
    o = Image.open(o_path).convert("RGB")
    n = Image.open(n_path).convert("RGB").resize(o.size, Image.LANCZOS)
    ao = np.asarray(o, np.float32)
    an = np.asarray(n, np.float32)
    lo = rgb2lab(ao / 255.0)
    ln = rgb2lab(an / 255.0)
    d = np.sqrt(((ln - lo) ** 2).sum(2))
    lum_o = ao @ np.array([.299, .587, .114], np.float32)
    lum_n = an @ np.array([.299, .587, .114], np.float32)
    sel = lum_o < 60
    print("[%s] 全图ΔE=%.2f (ΔE>15占%.1f%%)  主体区ΔE=%.2f  主体墨量 ORIG %.2f%% NEW %.2f%%"
          % (tag, d.mean(), 100 * (d > 15).mean(), d[sel].mean(),
             100 * (lum_o < 30).mean(), 100 * (lum_n < 30).mean()))
    return d


def main():
    # 成品
    shutil.copy(P4_N, os.path.join(OUT, "1_pinterest4_迷彩棕榈_仿射换位定稿.jpg"))
    shutil.copy(P6_N, os.path.join(OUT, "2_pinterest6_鹰骷髅_canny重生定稿.jpg"))
    print("[copy] 2 成品")

    s1 = side_by_side(P4_O, P4_N, os.path.join(OUT, "z1_p4_原图vs成品.jpg"))
    s2 = side_by_side(P6_O, P6_N, os.path.join(OUT, "z2_p6_原图vs成品.jpg"))
    print("[cmp] z1", s1, " z2", s2)

    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    d4 = metrics("p4", P4_O, P4_N)
    d6 = metrics("p6", P6_O, P6_N)

    # QC
    lines = []
    lines.append("第18轮返修 QC 指标  (2026-09-18)")
    lines.append("=" * 62)
    lines.append("")
    lines.append("【pinterest4 迷彩棕榈】代码：camo_palm_swap（整棵仿射换位）")
    lines.append("  用户否决原话：\"乱七八糟一块一块的，严格要求过不许一块块碎片化的裂变效果\"")
    lines.append("  根因（逐条实测）：")
    lines.append("   1) 常数黑覆盖：α 覆盖像素一律涂常数黑 → 树干棕褐杆身被抹平、横档缝被涂黑 → 整杆糊成实心黑条")
    lines.append("      → 改「预乘原色层」(直传供体真实 RGB)")
    lines.append("   2) Voronoi 归属撕裂树冠：高棕榈树冠离干轴 150px+，落进邻树格 → 冠/干分离成漂浮碎块")
    lines.append("      → 改「测地线归属」(watershed 以墨迹为连通域洪泛)")
    lines.append("   3) 尺度被旋转惩罚：旧式 bbox 拟合，旋转把细高棕榈 bbox 撑大 → 强制缩小 → 墨量掉 2.9pp")
    lines.append("      → 改「树干长度比」定尺（旋转不变量）")
    lines.append("   4) 边缘树被裁 + α 膝点过高 → 加「四方连续环绕贴回」+ knee(0.25,0.50)→(0.15,0.20)")
    lines.append("   5) 掩膜 0.7σ 高斯预模糊制造 1.5px 光晕 → 桥接细横档；去掉")
    lines.append("")
    lines.append("  验收（剔除 JPEG 噪点后的真结构，lum<30）：")
    lines.append("    原图  墨 22.67% / 主连通块 85  / 笔宽 7.88")
    lines.append("    成品  墨 22.64% / 主连通块 78  / 笔宽 7.07      ← 墨量差 0.03pp、主块数 92% 吻合")
    lines.append("  ⚠️ 旧指标\"原图 832 个连通块\"是误导：其中 747 个 <12px 小块合计仅 960px(0.044%)，")
    lines.append("     落在暗迷彩里(均值27.8/邻域59.4) = JPEG 压缩噪点，不是墨。真实主结构只有 85 块。")
    lines.append("")
    lines.append("【pinterest6 鹰骷髅】代码：eagle_skull_canny（Canny 结构引导 SDXL 重生）")
    lines.append("  用户否决原话：\"有逻辑是正常元素的要求，做的代码错乱就是不合格，严重不合格\"")
    lines.append("  被否决的三条历史路线：")
    lines.append("   · v373 本地 inpaint 重生 → 主体≈原图复制（\"跟原图没区别\"）")
    lines.append("   · v386 mode1（880×1240 txt2img + IPAdapter 双锁）→ 生成图仅 880px 宽，")
    lines.append("     被放大 4× 回 3543px → 鹰头糊成一团（\"错乱\"、Laplacian 方差仅为原图 1/5.9）")
    lines.append("   此轮定稿：")
    lines.append("   · Canny 结构引导（cn 0.32→0.42 线性、end 0.45）+ denoise 0.94 + 2048 原生重绘")
    lines.append("     → 模型在结构骨架上真正重画，同时保硬边/平涂质感，不再糊")
    lines.append("   · 鹰（翼羽改人字纹、翼展姿态、白首黄喙黄眼）、骷髅（颅形/裂纹/下颌）、")
    lines.append("     双角（加粗分节）全部重构 = 真·异内容同构")
    lines.append("   · seed 777（6 变体 F1~F6 中唯一同时满足「变化量最大 + 无黄色杂斑 + 无深色污斑」）")
    lines.append("")
    lines.append("  量化：全图 ΔE=%.2f（ΔE>15 占 %.1f%%）；主体区 ΔE=%.2f" % (d6.mean(), 100 * (d6 > 15).mean(), d6[np.asarray(Image.open(P6_O).convert('RGB'), np.float32) @ np.array([.299, .587, .114], np.float32) < 60].mean()))
    lines.append("")
    lines.append("=" * 62)
    lines.append("五图 ↔ 代码命名（严格记住，每图独立代码）")
    lines.append("  1. 6978          紫色蝙蝠徽章+Didone弧字   → bat_badge          styles/subject_badge_text.py")
    lines.append("  2. b78e60        军牌狗牌迷彩+BlackOpsOne → dogtag_camo        styles/camo_pattern.py")
    lines.append("  3. pinterest3    牛仔拼布蝶+denim贴布字   → denim_butterfly    src/v381_p3final.py")
    lines.append("  4. pinterest4    迷彩棕榈(矢量线稿·无字)  → camo_palm_swap     styles/camo_palm_pattern.py + src/v346_p4aff.py")
    lines.append("  5. pinterest6    鹰骷髅金属+尖刺标题      → eagle_skull_canny  src/v390_p6canny.py (REBIRTH_FILES['pinterest6'])")

    with open(os.path.join(OUT, "QC_指标.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[QC] written")

    # 代码映射单独一份
    mp = []
    mp.append("五图 ↔ 裂变代码命名对照（每图一套独立代码，禁止一码通用）")
    mp.append("=" * 70)
    mp.append("")
    mp.append("  ① 6978        紫色蝙蝠徽章 + Didone 弧字")
    mp.append("     代码名：bat_badge")
    mp.append("     文件：  image-fission/styles/subject_badge_text.py")
    mp.append("     路线：  主体保形 + 弧字重绘（有字徽章）")
    mp.append("")
    mp.append("  ② b78e60      军牌狗牌迷彩 + BlackOpsOne 字")
    mp.append("     代码名：dogtag_camo")
    mp.append("     文件：  image-fission/styles/camo_pattern.py")
    mp.append("     路线：  迷彩底纹保色 + 商标原位改写")
    mp.append("")
    mp.append("  ③ pinterest3  牛仔拼布蝶 + denim 贴布字")
    mp.append("     代码名：denim_butterfly")
    mp.append("     文件：  image-fission/src/v381_p3final.py")
    mp.append("     路线：  SDXL 结构级重生三蝶（主体 Zoom1.7 / 小蝶 Zoom5·6）")
    mp.append("")
    mp.append("  ④ pinterest4  迷彩棕榈（专业矢量线稿，无字）")
    mp.append("     代码名：camo_palm_swap")
    mp.append("     文件：  image-fission/styles/camo_palm_pattern.py")
    mp.append("            image-fission/src/v346_p4aff.py")
    mp.append("     路线：  整棵仿射换位（刚体错排 + 等比缩放 + 四方连续环绕）")
    mp.append("            —— 矢量线稿永不用 SDXL 重画（细叶必碎）")
    mp.append("     开关：  make_v329.py 的 P4_MODE 缺省 = swap")
    mp.append("")
    mp.append("  ⑤ pinterest6  鹰骷髅金属 + 尖刺标题")
    mp.append("     代码名：eagle_skull_canny")
    mp.append("     文件：  image-fission/src/v390_p6canny.py")
    mp.append("            image-fission/make_v329.py（REBIRTH_FILES['pinterest6']）")
    mp.append("     路线：  Canny 结构引导 SDXL 原生 2048 重绘")
    mp.append("            （cn 0.32→0.42 / denoise 0.94 / seed 777）")
    mp.append("            —— 禁本地 inpaint 重生（太像原图）、禁低分辨率 mode1（放大 4× 必糊）")
    with open(os.path.join(OUT, "代码映射_五图对应.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(mp))
    print("[map] written")

    print("\n=== 交付目录 ===")
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        print("  %-46s %8d B" % (fn, os.path.getsize(p)))


if __name__ == "__main__":
    main()
