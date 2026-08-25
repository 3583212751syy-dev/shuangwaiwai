"""
批量风格裂变 v7.5：在 v7 基础上修复三个迭代问题：
  1. 中央噪点/碎裂感 → 保留 4x USDU 真实超分（4096px 高清晰度，用户要的），
     但在超分前加「二阶细化 KSampler」（latent 低 denoise 重采）做二次清理：
     KSampler1(45步满采) → 二阶细化 KSampler(denoise=0.28, 30步) → VAE → 4x USDU → Save。
     潜空间先降噪再超分，避免把噪点放大成碎裂感（v7.5 初版误关 USDU 改 latent 1.55x
     导致分辨率从 4096 掉到 1536，已回退）。
  2. 糊字/乱码（denim_3 顶部 OLDE/UPLUI，eagle_2 骷髅盒 JACHE DJAONOES 等）
     - STYLE_PREFIX 加 6 个 anti-garbled-text 关键词；
     - negative_prompt 加强到 14 项 anti-text；
     - TSHIRT_LORA 强度从 0.45 降到 0.28（其训练集大量带 banner 字样样本）；
     - tasks 里 4 个 text-prone 类型 (eagle_2/denim_3/skull_5/metal_6) 关闭 tshirts LoRA，
       改用 VECTOR_LORA 治乱码；
  3. 像素清晰度 → steps 35→45（build.py 现读取 params["steps"]）/ cfg 6→7 /
     hires_steps 20→30（二阶细化步数）/ hires_denoise 0.35→0.28，
     IPA_NOISE 0.10→0.05（减参考噪）、IPA_END 0.65→0.78（参考用得更彻底）。

设计语言约束（继承 v7）：
  - 输出「可印在全幅服装上的平面印花图案」，非衣服效果图；
  - 每类裂变参考原图的构图与设计语言，禁止跨语言跳跃；
  - 排除侵权文字 / 品牌 / 真人。
"""
import os
import sys
import time
import shutil
import base64
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1
import config as _cfg

# 底模：v8 起改回强底模 Juggernaut Ragnarok v10 —— 原版 sd_xl_base_1.0 噪声大、易出乱码、
# 质量低（用户 8/25 反馈）。Juggernaut v10 质量更高、prompt 服从更强，仍配合
# IPAdapter + LoRA 主导风格。备选 CounterfeitXL（更平涂/动漫风）。
_cfg.SDXL_CHECKPOINT = "juggernautXL_ragnarokBy.safetensors"

COMFYUI_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
JOBS_BASE = r"E:\Desktop\双接口\image-fission\jobs"

# 全局印花/T恤 LoRA（chrisconyers LoRA 文件名带 tshirts，但实际是图案风格 LoRA，不生成衣服）
TEXTILE_LORA = "Canopus-Textile-Pattern-adp-LoRA.safetensors"
TEXTILE_LORA_STRENGTH = 0.50
TSHIRT_LORA = "chrisconyers-sdxl-tshirts-lora.safetensors"
# v7.5: 从 0.45 降到 0.28 —— 这套 LoRA 训练集含大量带 banner 文字的街头图案样本，
# 强度过高是糊字/乱码的主因之一。
TSHIRT_LORA_STRENGTH = 0.28

# 矢量线描 LoRA：用于 engraving/ornamental 类，增强 ornamental 图案感
VECTOR_LORA = "DD-vector-v2.safetensors"

# ControlNet Canny fp16：全局开启，给每一类提供像素级构图参考；
# 单类强度在 ORIGINALS_CONFIG 里分别控制。
CONTROLNET = "controlnet-canny-sdxl-1.0.fp16.safetensors"
USE_CONTROLNET = True

# v7.5: 保留 4x USDU 真实超分（1024→4096px，用户要的清晰度），但改在超分前加二阶细化
# KSampler（见 build.py USDU 路径）= 潜空间先低 denoise 重采清理噪点，再 4x 超分，
# 既治中央噪点又不掉分辨率。HIRES_DENOISE/HIRES_STEPS 即二阶细化 KSampler 的参数。
# v8: 超分模型 NMKD-Siax → 4x-UltraSharp —— 前者在矢量/插画平涂上易出颗粒噪点，
# UltraSharp 对插画边缘更干净（用户 8/25 反馈噪点多）。配合 USDU 前二阶细化 KSampler。
USDU_MODEL = "4x-UltraSharp.pth"
HIRES_SCALE = 1.5     # 仅在 USDU 关闭的 fallback 路径使用
HIRES_DENOISE = 0.28  # 二阶细化 KSampler 重采强度
HIRES_STEPS = 30      # 二阶细化 KSampler 步数

# 公共风格前缀：明确「全幅印花图案 / 纺织品印花艺术作品」，不是衣服效果图
# v7.5: 强化 anti-garbled-text（治糊字/乱码）—— 加 6 个关键词从源头引导模型远离 banner 文字。
STYLE_PREFIX = (
    "high quality all-over print graphic, textile print artwork, "
    "surface pattern design, 2d flat graphic illustration, "
    "bold clean outlines, flat color blocks, screen print aesthetic, "
    "scalable vector style, print ready artwork, no background scene, "
    "no photography, no 3d render, no realistic texture, no fabric folds, "
    "no wrinkles, no shadows, no depth of field, no product shot, no mockup, no garment, "
    "no embroidery, no stitched texture, no raised thread, no 3d fabric, no buttons, "
    "absolutely no readable text, no letters whatsoever, no alphabet, no words, "
    "no garbled text, no gibberish text, no pseudo-script, no banner writing, "
    "no ribbon writing, no sign writing, no carved text, no engraved text, no inscriptions, "
    "no character glyphs resembling text, no scribbles"
)

STYLE_SUFFIX = "sharp focus, crisp edges, professional textile print"


def make_config(
    type_label,
    style_words,
    subjects,
    extra_lora=None,
    extra_lora_strength=0.40,
    sim=0.70,
    comp=0.55,
    cn=0.40,
):
    """打包一类原图的生成配置。"""
    return {
        "type": type_label,
        "style_words": style_words,
        "subjects": subjects,
        "extra_lora": extra_lora,
        "extra_lora_strength": extra_lora_strength,
        "sim": sim,
        "comp": comp,
        "cn": cn,
    }


# 按原图类型配对生成策略
# sim  = IPAdapter 相似度（颜色/材质锁）
# comp = composition 构图锁（0=完全由 prompt 重新构图，1=紧贴原图布局）
# cn   = ControlNet Canny 强度（只在 USE_CONTROLNET=True 时生效）
ORIGINALS_CONFIG = {
    # illust_1：黑白高对比装饰花卉/孔雀卷草 → 保留「竖向不对称装饰卷草」构图
    "illust_1": make_config(
        type_label="bw_ornamental_scroll",
        style_words=(
            "black and white ornamental engraving, elegant botanical scrollwork, "
            "high contrast floral illustration, Art Nouveau inspired decorative pattern, "
            "peacock and flower silhouette, vertical asymmetrical composition"
        ),
        subjects=[
            ("peacock_floral",   "ornate peacock with flowing tail feathers intertwined with blooming flowers and curling vines, high contrast black and white decorative scroll"),
            ("hummingbird_vines", "hummingbird hovering among blossoming vines and swirling floral scrolls, monochrome ornamental illustration"),
            ("floral_cascade",    "cascading bouquet of flowers, leaves and curls forming an elegant vertical scroll, high contrast decorative botanical pattern"),
            ("butterfly_garden",  "ornate butterfly surrounded by daisies, leaves and curling tendrils, vertical asymmetrical botanical decorative motif"),
        ],
        extra_lora=VECTOR_LORA,
        extra_lora_strength=0.35,
        sim=0.78,   # 保留黑白装饰风格
        comp=0.72,  # 保留竖向卷草构图
        cn=0.55,
    ),

    # eagle_2：鹰+火焰+骷髅+锁链+横幅徽章 → 元素在「哥特机车徽章」语汇内裂变
    "eagle_2": make_config(
        type_label="gothic_biker_crest",
        style_words=(
            "gothic biker emblem print, black and orange flame graphic, "
            "eagle and skull crest, chain and banner details, dark streetwear badge"
        ),
        subjects=[
            # v8.1: "blank scroll / ribbon scroll" → "blank ornamental filigree cartouche"
            # —— scroll 是横幅的近义词，会被模型识别为放字的位置；filigree cartouche
            # 则被渲成装饰花纹/卷叶，不出字（v8 验证）。
            ("eagle_flame",   "spread-wing eagle clutching a flaming skull, surrounded by fire and chains, blank ornamental filigree cartouche with abstract pattern inside, symmetrical vertical emblem, no writing, no text, no letters, no characters, no glyphs, no script"),
            ("skull_wings",   "large skull with spread eagle wings, red flames and chain borders, blank ornamental filigree cartouche with abstract pattern inside, symmetrical crest, no writing, no text, no letters, no characters, no glyphs, no script"),
            ("raven_flame",   "black raven with outstretched wings, flaming skull below, chains and blank ornamental filigree cartouche with abstract pattern inside, dark gothic emblem, no writing, no text, no letters, no characters, no glyphs, no script"),
            ("winged_skull",  "winged skull with red flames, crossed chains and a blank ornamental filigree cartouche with abstract pattern inside, symmetrical biker crest, no writing, no text, no letters, no characters, no glyphs, no script"),
        ],
        sim=0.72,
        comp=0.58,  # 保留垂直徽章构图，但允许元素替换
        cn=0.45,
    ),

    # denim_3：牛仔贴布+蝴蝶+"UPCY"文字 → 元素在「牛仔再造贴布」语汇内裂变
    # 注意：原图是真实牛仔布贴布照片，裂变结果必须是「平面印花图案」而非再拍一张布贴布
    "denim_3": make_config(
        type_label="denim_patchwork",
        style_words=(
            "flat vector illustration, light blue and white color palette only, "
            "denim blue inspired graphic print, butterfly and abstract letterform motif, "
            "crisp clean shapes, solid flat color blocks, screen print style, "
            "no orange, no yellow, no gold, no warm tones, no brown, no beige, "
            "no fabric texture, no embroidery"
        ),
        subjects=[
            # v8.1: 把 "upper abstract geometric pattern band" / "across the top" 改成
            # "decorative filigree flourish / ornamental crest" —— 「band/top」 类词会
            # 触发模型往里填字母（v8 验证 denim_3 出 "CNST"）；ornament/filigree/crests
            # 则被渲成装饰花纹，不出字。
            ("butterfly_trail", "flat vector butterfly graphic, decorative filigree flourish above, central large butterfly, smaller butterflies trailing below along a dotted path, light blue and white solid colors, no text, no letters, no glyphs"),
            ("word_butterfly",  "flat vector ornamental filigree flourish across the top, large butterfly graphic in the center, small star and heart accents, light blue and white solid colors, no text, no letters, no glyphs"),
            ("shape_collage",   "flat vector collage of overlapping geometric shapes in denim blue, central butterfly graphic, star and heart accents, clean edges, no text, no letters, no glyphs"),
            ("floral_butterfly","flat vector butterfly surrounded by small flowers and dotted trail, decorative floral filigree above, light blue and white solid colors, no text, no letters, no glyphs"),
        ],
        sim=0.42,   # 极低材质锁，避免把真实牛仔布纹理带进来；构图交给 ControlNet
        comp=0.60,  # 保留上方装饰+中央主图+下方小元素 的构图
        cn=0.40,    # v8.1: 0.55→0.40 —— 原图带真字 "UPCY"，高 CN 让模型想复现文字
    ),

    # camo_4：迷彩+棕榈树全幅图案 → 保留「迷彩底+棕榈树剪影」构图
    "camo_4": make_config(
        type_label="tropical_camo",
        style_words=(
            "military camouflage all-over print, woodland and desert camo blocks, "
            "scattered palm tree silhouettes, brown green khaki tones, "
            "repeating tactical surface pattern"
        ),
        subjects=[
            ("palm_woodland", "woodland camouflage pattern with scattered black palm tree silhouettes, brown green khaki tones, repeating all-over print"),
            ("palm_desert",   "desert camouflage pattern with palm tree silhouettes, sand tan and olive tones, repeating surface pattern"),
            ("palm_jungle",   "jungle camouflage pattern with dense palm tree silhouettes, deep green and brown tones, repeating print"),
            ("palm_digital",  "digital pixelated camouflage with palm tree silhouettes, olive and grey tones, modern tactical repeating pattern"),
        ],
        sim=0.78,   # 保留迷彩颜色/风格
        comp=0.70,  # 保留迷彩全幅构图
        cn=0.55,
    ),

    # skull_5：骷髅+翅膀+蛇+玫瑰+血滴 → 元素在「哥特骷髅徽章」语汇内裂变
    "skull_5": make_config(
        type_label="gothic_skull_emblem",
        style_words=(
            "dark gothic skull emblem, red wings and roses, snake wrapped around skull, "
            "blood drip accents, symmetrical vertical badge, dark rock poster art"
        ),
        subjects=[
            ("skull_wing_snake", "skull with spread red wings, snake coiled around, red roses at sides, blood drips, symmetrical emblem, no text, no letters, no glyphs"),
            ("skull_bat_wings",  "skull with bat wings, snake and thorny roses, dark red accents, symmetrical gothic badge, no text, no letters, no glyphs"),
            ("skull_raven_wings","skull with raven black wings, snake and roses, blood drops, dark symmetrical crest, no text, no letters, no glyphs"),
            ("skull_roses",      "skull surrounded by red roses and thorns, wing-like floral frame, snake at base, symmetrical emblem, no text, no letters, no glyphs"),
        ],
        sim=0.72,
        comp=0.58,  # 保留中央骷髅+两侧翅膀/玫瑰+上下横幅 的构图
        cn=0.45,
    ),

    # metal_6：金属 logo+鹰+角骷髅+闪电 → 元素在「重金属乐队艺术」语汇内裂变
    "metal_6": make_config(
        type_label="heavy_metal_badge",
        style_words=(
            "heavy metal band art print, spiked ornamental abstract shape, no readable words, "
            "eagle and horned skull, radiating lightning bolts, "
            "dark underground metal emblem, black white and brown"
        ),
        subjects=[
            ("eagle_horned_skull",  "eagle with spread wings above a horned skull, radiating lightning bolts, spiked abstract ornamental shape banner at top, symmetrical emblem, no text, no letters, no glyphs"),
            ("skull_lightning",     "screaming skull with horns, lightning bolts radiating behind, spiked abstract ornamental shape banner above, death metal crest, no text, no letters, no glyphs"),
            ("raven_skull",         "raven with outstretched wings above horned skull, lightning and spikes, underground metal emblem, no text, no letters, no glyphs"),
            ("winged_horned_skull", "large horned skull with wings, lightning radiating, spiked abstract ornamental shape above, symmetrical metal badge, no text, no letters, no glyphs"),
        ],
        sim=0.72,
        comp=0.58,  # 保留顶部 logo+中央鹰/骷髅+放射闪电 的构图
        cn=0.45,
    ),
}

# 6 张原图
ORIGINALS = [
    ("pinterest_illust_1", "illust_1"),
    ("pinterest_eagle_2",  "eagle_2"),
    ("pinterest_denim_3_flat", "denim_3"),  # 原图是实物照片，用预处理后的平涂版作参考
    ("pinterest_camo_4",   "camo_4"),
    ("pinterest_skull_5",  "skull_5"),
    ("pinterest_metal_6",  "metal_6"),
]

IPA_WEIGHT_TYPE = "style transfer"
# v7.5: IPA_NOISE 0.10→0.05（参考图加更少噪，避免 base 1024 中央出现碎裂噪点）；
#        IPA_END 0.65→0.78（让 IPAdapter 的颜色/材质参考用到更靠后采样段，更彻底锁参考）。
IPA_NOISE = 0.05
IPA_END = 0.78
CN_LOW, CN_HIGH = 0.4, 0.8

_lock = threading.Lock()
_done = 0
_total = 0


def img_to_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_one(client, orig_label, sub_label, sub_prompt, seed_name, cfg, out_dir, idx):
    """生成单张；返回 (orig_label, sub_label, out_path) 或 None。"""
    global _done

    style_words = cfg.get("style_words", "")
    sim = cfg.get("sim", 0.70)
    comp = cfg.get("comp", 0.55)
    cn = cfg.get("cn", 0.40)
    extra_lora = cfg.get("extra_lora")
    extra_lora_strength = cfg.get("extra_lora_strength", 0.40)

    # 组合 prompt
    prompt = ", ".join(filter(None, [STYLE_PREFIX, style_words, sub_prompt, STYLE_SUFFIX]))

    # 独立随机 seed，确保同原图 4 张有差异
    seed = random.randint(1, 999999999)

    # v7.5: 治噪 + 治乱码 —— steps 35→45（更多采样降噪，build.py 现已读取 params["steps"]）/
    # cfg 6→7（更严守 prompt 治伪文字）。USDU_MODEL 恢复 4x 真实超分（4096px 清晰度），
    # 二阶细化 KSampler（hires_denoise/hires_steps）在超分前做潜空间二次清理治中央噪点。
    params = {
        "style_prompt": prompt,
        "similarity": sim,
        "ipadapter_weight_type": IPA_WEIGHT_TYPE,
        "ipadapter_noise": IPA_NOISE,
        "ipadapter_end": IPA_END,
        "composition_strength": comp,
        "controlnet_name": CONTROLNET if USE_CONTROLNET else "",
        "controlnet_strength": cn if USE_CONTROLNET else 0.0,
        "controlnet_low_threshold": CN_LOW,
        "controlnet_high_threshold": CN_HIGH,
        "usdu_model": USDU_MODEL,
        "hires_scale": HIRES_SCALE,
        "hires_denoise": HIRES_DENOISE,
        "hires_steps": HIRES_STEPS,
        # v7.5: 增强 anti-garbled-text —— 14 项覆盖字母/文字/品牌/伪脚本
        # v8: 增加 SDXL 原生负向 embedding —— NegativeXL（CounterfeitXL 作者出品，治乱码/
        # 画质伪影）+ unaestheticXL（ComfyUI 官方推荐通用质量负向）。EasyNegative 是 SD1.5
        # 的，对 SDXL 无效，勿用。embedding 放最前确保 CLIP 权重优先。
        "negative_prompt": (
            "embedding:NegativeXL, embedding:unaestheticXL, "
            "photography, product photo, 3d render, realistic texture, fabric folds, "
            "wrinkles, shadows, depth of field, blurry, deformed, low quality, "
            "readable text, real words, fake words, brand name, trademark, watermark, "
            "letters, lettering, alphabet, alphabet characters, words, font, typography, "
            "banner text, ribbon text, garbled text, gibberish text, pseudo-script, "
            "sign text, label text, inscription, motto, slogan, carved text, engraved text, "
            "scribbles resembling text, copyright logo, signature, cropped, out of frame, "
            "mockup, garment"
        ),
        "width": 1024, "height": 1024, "batch_per_run": 1,
        "steps": 45, "cfg": 7.0,
        "seed": seed,
        # 双 LoRA
        "lora_name": TEXTILE_LORA,
        "lora_strength": TEXTILE_LORA_STRENGTH,
        "lora_name_2": TSHIRT_LORA,
        "lora_strength_2": TSHIRT_LORA_STRENGTH,
    }

    # engraving 类用矢量线描 LoRA 替换第二个 LoRA，增强 ornamental 图案感
    if extra_lora:
        params["lora_name_2"] = extra_lora
        params["lora_strength_2"] = extra_lora_strength

    # v7.5: 4 个 text-prone 类型 (eagle_2 / denim_3 / skull_5 / metal_6) 关 tshirts LoRA，
    # 用 VECTOR_LORA 替代 —— chrisconyers-tshirts 训练集带大量 banner 文字样本，
    # 是「骷髅盒 JACHE DJAONOES」/「牛仔顶部 OLDE」这类乱码的直接诱因。
    if cfg["type"] in ("gothic_biker_crest", "denim_patchwork",
                       "gothic_skull_emblem", "heavy_metal_badge"):
        params["lora_name_2"] = VECTOR_LORA
        params["lora_strength_2"] = 0.15  # v8.1: 0.22→0.15 —— vector/engraving 风格本身诱发伪字母，再降

    g = build_mode1(seed_name, params, f"batch_{orig_label}_{sub_label}")
    t0 = time.time()
    try:
        res = client.run(g, timeout=600)
        data = next(iter(res.values()))[0]
        out_path = os.path.join(out_dir, f"{orig_label}_{sub_label}.jpg")
        with open(out_path, "wb") as f:
            f.write(data)
        dt = time.time() - t0
        with _lock:
            _done += 1
            print(f"[OK {_done}/{_total}] {orig_label} × {sub_label} ({cfg['type']}) "
                  f"seed={seed} SIM={sim:.2f} COMP={comp:.2f} CN={cn if USE_CONTROLNET else 0}  {dt:.0f}s")
        return (orig_label, sub_label, out_path)
    except Exception as e:
        with _lock:
            _done += 1
            print(f"[FAIL {_done}/{_total}] {orig_label} × {sub_label}: {repr(e)}")
        return None


def build_gallery(results, out_dir):
    """生成 base64 内联 gallery.html（兼容旧流程，但体积大）。"""
    rows_html = []
    for orig_seed, orig_label in ORIGINALS:
        orig_path = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(orig_path):
            continue
        seed_b64 = img_to_b64(orig_path)
        cfg = ORIGINALS_CONFIG.get(orig_label, ORIGINALS_CONFIG["illust_1"])
        cells = ""
        for sub_label, _ in cfg["subjects"]:
            img = next((p for o, s, p in results if o == orig_label and s == sub_label), None)
            if img:
                cells += f'<td><img src="data:image/jpeg;base64,{img_to_b64(img)}"><div class="cap">{sub_label}</div></td>'
            else:
                cells += f'<td class="empty">—</td>'
        rows_html.append(f"""
        <tr>
          <th>{orig_label}</th>
          <td class="seed"><img src="data:image/jpeg;base64,{seed_b64}"><div class="cap">原图</div></td>
          {cells}
        </tr>""")

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>6x4 批量风格裂变 v7.5</title>
<style>
body{{font-family:system-ui;background:#fafafa;padding:24px;margin:0;}}
h1{{margin:0 0 8px;font-size:22px;}}.sub{{color:#666;margin-bottom:24px;}}
.params{{background:#222;color:#fff;padding:10px 14px;border-radius:6px;display:inline-block;font-family:monospace;font-size:12px;margin-bottom:18px;}}
table{{border-collapse:collapse;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.1);width:100%;}}
th,td{{padding:6px;text-align:center;vertical-align:top;border-bottom:1px solid #eee;}}
th{{background:#f4f4f4;font-size:13px;text-align:left;padding-left:12px;}}
td img{{max-width:200px;display:block;margin:0 auto;}}
td.seed img{{max-width:120px;}}
.cap{{font-size:11px;color:#666;padding-top:4px;}}
.empty{{color:#aaa;}}
</style></head><body>
<h1>图裂变 · 6 原图 × 4 主题批量 v7.5（治噪 + 治乱码 + 像素清晰度提升）</h1>
<div class="sub">6 张设计稿 → 4 个同设计语言新主题 = 24 张可印全幅服装的平面图案</div>
<div class="params">hires fix (1.55x + refine 30 steps) | 45 steps / cfg 7.0 | text-prone 类型 VECTOR LORA 替代 TSHIRT | 强化 anti-garbled-text</div>
<table>
  <tr><th>原图</th><th>原图</th><th>主题1</th><th>主题2</th><th>主题3</th><th>主题4</th></tr>
  {''.join(rows_html)}
</table>
</body></html>"""
    html_path = os.path.join(out_dir, "gallery.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main():
    global _total
    out_dir = os.path.join(JOBS_BASE, f"batch_6x4_v75_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[cfg] checkpoint={_cfg.SDXL_CHECKPOINT} textile={TEXTILE_LORA}@{TEXTILE_LORA_STRENGTH} "
          f"tshirt={TSHIRT_LORA}@{TSHIRT_LORA_STRENGTH} steps=45 cfg=7.0 hires={HIRES_SCALE}x/{HIRES_STEPS}steps denoise={HIRES_DENOISE}")

    client = ComfyClient()

    # 准备所有任务（复制原图 + 构造参数）
    tasks = []
    for orig_seed, orig_label in ORIGINALS:
        seed_src = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(seed_src):
            print(f"[WARN] 跳过: {seed_src} 不存在")
            continue
        seed_name = f"batch_{orig_label}_{int(time.time()*1000)}.jpg"
        shutil.copy(seed_src, os.path.join(COMFYUI_INPUT, seed_name))
        cfg = ORIGINALS_CONFIG.get(orig_label, ORIGINALS_CONFIG["illust_1"])
        for sub_label, sub_prompt in cfg["subjects"]:
            tasks.append((orig_label, sub_label, sub_prompt, seed_name, cfg))

    _total = len(tasks)
    print(f"[queue] 提交 {_total} 张到 ComfyUI 队列（背靠背执行）...")

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {}
        for idx, (orig_label, sub_label, sub_prompt, seed_name, cfg) in enumerate(tasks):
            fut = ex.submit(run_one, client, orig_label, sub_label, sub_prompt,
                            seed_name, cfg, out_dir, idx)
            futures[fut] = (orig_label, sub_label)
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    total = time.time() - t_start
    print(f"\n[DONE] {len(results)}/{_total}  总耗时 {total:.0f}s")

    if not results:
        return

    # 同时生成 base64 版（旧）和轻量版（新）
    build_gallery(results, out_dir)
    from make_gallery import build as build_lean
    build_lean(out_dir, os.path.join(out_dir, "gallery_lean.html"))
    print(f"[gallery] {out_dir}/gallery.html  &  gallery_lean.html")
    return out_dir


if __name__ == "__main__":
    main()
