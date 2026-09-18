"""build_v386.py — 第17轮交付：只做被否决的两图（p4 + p6）修复 + 五图代码映射。

用户令：先只做这两个图裂变代码，优化升级完了再去改其他的。
产出目录：E:\\Desktop\\v386_两图返修_第17轮\\
"""
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DESK = Path("E:/Desktop")
OUT = DESK / "v386_两图返修_第17轮"
OUT.mkdir(parents=True, exist_ok=True)

P4_ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
P6_ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
P4_NEW = ROOT / "jobs" / "router_out_v329" / "Pinterest (4)_variant.jpg"
P6_NEW = ROOT / "jobs" / "router_out_v329" / "pinterest6_variant.jpg"


def load_font(sz):
    for f in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def side_by_side(orig_p, new_p, dst, tag, H=1100):
    o = Image.open(orig_p).convert("RGB")
    n = Image.open(new_p).convert("RGB")
    o = o.resize((int(o.width * H / o.height), H), Image.LANCZOS)
    n = n.resize((int(n.width * H / n.height), H), Image.LANCZOS)
    pad, top = 16, 74
    W = o.width + n.width + pad * 3
    canvas = Image.new("RGB", (W, H + top + pad), (245, 245, 247))
    canvas.paste(o, (pad, top))
    canvas.paste(n, (pad * 2 + o.width, top))
    d = ImageDraw.Draw(canvas)
    f = load_font(44)
    d.text((pad + 6, 14), f"{tag}  ORIG（原图）", fill=(20, 20, 20), font=f)
    d.text((pad * 2 + o.width + 6, 14), f"{tag}  NEW（裂变）", fill=(150, 20, 20), font=f)
    d.rectangle([pad - 1, top - 1, pad + o.width, top + H], outline=(180, 180, 185))
    d.rectangle([pad * 2 + o.width - 1, top - 1, pad * 2 + o.width + n.width, top + H],
                outline=(180, 180, 185))
    canvas.save(dst, quality=93)
    print("[build_v386]", dst.name, canvas.size)


# 1) 成品
shutil.copy2(P4_NEW, OUT / "1_pinterest4_迷彩棕榈_仿射换位定稿.jpg")
shutil.copy2(P6_NEW, OUT / "2_pinterest6_鹰骷髅_mode1真裂变.jpg")

# 2) 1:1 对照
side_by_side(P4_ORIG, P4_NEW, OUT / "z1_p4_原图vs成品.jpg", "pinterest4")
side_by_side(P6_ORIG, P6_NEW, OUT / "z2_p6_原图vs成品.jpg", "pinterest6")

# 3) 五图代码映射
mapping = """五图 ↔ 裂变代码 权威映射（2026-09-18 用户第17轮定，严格记住·勿混用）
================================================================
每张图的裂变方式都按**图片自身内容**选代码，不是一套通吃。

#  图ID          内容                          代码名              代码文件
1  6978          紫色蝙蝠徽章 + Didone 弧字      bat_badge          styles/subject_badge_text.py
2  b78e60        军牌狗牌迷彩 + BlackOpsOne      dogtag_camo        styles/camo_pattern.py
3  pinterest3    牛仔拼布蝶 + denim 贴布字       denim_butterfly    src/v381_p3final.py
4  pinterest4    迷彩棕榈（矢量线稿·无字）       camo_palm_swap     styles/camo_palm_pattern.py (+src/v346_p4aff.py)
5  pinterest6    鹰骷髅金属 + 尖刺标题           eagle_skull_mode1  styles/subject_badge_textless.py

各代码做什么
----------------------------------------------------------------
bat_badge          mode1 整幅 + Canny 锁轮廓 + Playfair 弧字；输出已冻结（ΔE≈4.5）
dogtag_camo        mode3 img2img + camo_blob_morph 保色湖泊形变 + 文字原位替换
denim_butterfly    SDXL 三蝶原生重生(zoom 1.7/5/6) + repair_dots + overlay_fringe + denim 字
camo_palm_swap     整棵仿射换位（刚体错排/各向异性缩放/切变/镜像）→ 保线零掉档
eagle_skull_mode1  mode1 整幅 txt2img + IPAdapter 双锁 + Canny 0.5 → 主体真换形

本轮两条铁律（都是被否决逼出来的）
----------------------------------------------------------------
* camo_palm_swap：矢量线稿**永不用 SDXL 重画**（细叶二值化必撕成短虚线 = "碎"）。
  默认 = 仿射换位；SDXL 重画仅 P4_MODE=hires/pre15 显式启用（调试）。
  实测：仿射换位 墨 21.2% ≈ 原 22.6%（线质零损失）；SDXL 重画碎块 126~177 vs 原图 91。
* eagle_skull_mode1：**禁用本地 inpaint 重生**（掩膜内小改 → 主体近乎原图复制）。
  正解 = src/fission.py --mode mode1（880x1240 txt2img + IPAdapter 双锁 + Canny 0.5）。
  注意 mode1 会把纯黑底生成成布纹灰底 → 交付前做「背景压黑」（低饱和中灰+与画框连通域置黑）。
"""
(OUT / "代码映射_五图对应.txt").write_text(mapping, encoding="utf-8")
print("[build_v386] done ->", OUT)
