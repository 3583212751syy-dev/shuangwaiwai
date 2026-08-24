"""
批量风格裂变 v7：严格按原图构图 + 设计理念裂变。

核心修正（相对 v6.5）：
  1. 输出是「可印在全幅服装上的平面印花图案」，不是衣服效果图；
     prompt 里删除 "t-shirt design" 等可能触发 garment/mockup 的词，
     保留 "all-over print graphic, textile print artwork"。
  2. 每一类裂变都必须参考原图的「构图」和「设计理念」：
     - 阴阳/装饰图（illust_1）与迷彩图（camo_4）高构图锁，保留其构图风格；
     - 其余四类也提高构图锁到 0.50-0.58，让布局仍看得出原图影子。
  3. 「元素裂变」只允许在同类设计语言内部变化：
     - eagle_2 只能换鹰/乌鸦/翼、骷髅、火焰、锁链、横幅；
     - denim_3 只能换牛仔贴布、蝴蝶、缝线字母；
     - skull_5 只能换骷髅、翅膀、蛇、玫瑰、血滴；
     - metal_6 只能换金属 logo、鹰/乌鸦、角骷髅、闪电；
     禁止向原图没有的语言跳跃（ dragon / butterfly 乱入 / 热带植物乱入等）。
  4. 明确排除侵权内容：negative 用 "readable text, real words, brand name, trademark"
     代替简单 "text"，允许装饰性字形成分但不出现可读品牌名。
  5. 保留并发 + 实时进度 + 轻量画廊。
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

# 底模：中性 SDXL base，让 LoRA 与 IPAdapter 主导风格
_cfg.SDXL_CHECKPOINT = "sd_xl_base_1.0.safetensors"

COMFYUI_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
JOBS_BASE = r"E:\Desktop\双接口\image-fission\jobs"

# 全局印花/T恤 LoRA（chrisconyers LoRA 文件名带 tshirts，但实际是图案风格 LoRA，不生成衣服）
TEXTILE_LORA = "Canopus-Textile-Pattern-adp-LoRA.safetensors"
TEXTILE_LORA_STRENGTH = 0.55
TSHIRT_LORA = "chrisconyers-sdxl-tshirts-lora.safetensors"
TSHIRT_LORA_STRENGTH = 0.45

# 矢量线描 LoRA：用于 engraving/ornamental 类，增强 ornamental 图案感
VECTOR_LORA = "DD-vector-v2.safetensors"

# ControlNet Canny fp16：全局开启，给每一类提供像素级构图参考；
# 单类强度在 ORIGINALS_CONFIG 里分别控制。
CONTROLNET = "controlnet-canny-sdxl-1.0.fp16.safetensors"
USE_CONTROLNET = True

# 公共风格前缀：明确「全幅印花图案 / 纺织品印花艺术作品」，不是衣服效果图
STYLE_PREFIX = (
    "high quality all-over print graphic, textile print artwork, "
    "surface pattern design, 2d flat graphic illustration, "
    "bold clean outlines, flat color blocks, screen print aesthetic, "
    "scalable vector style, print ready artwork, no background scene, "
    "no photography, no 3d render, no realistic texture, no fabric folds, "
    "no wrinkles, no shadows, no depth of field, no product shot, no mockup, no garment, "
    "no embroidery, no stitched texture, no raised thread, no 3d fabric, no buttons"
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
            ("eagle_flame",   "spread-wing eagle clutching a flaming skull, surrounded by fire and chains, blank ornamental banner, symmetrical vertical emblem, no text"),
            ("skull_wings",   "large skull with spread eagle wings, red flames and chain borders, blank ribbon banner, symmetrical crest, no text"),
            ("raven_flame",   "black raven with outstretched wings, flaming skull below, chains and blank banner, dark gothic emblem, no text"),
            ("winged_skull",  "winged skull with red flames, crossed chains and a blank ribbon banner, symmetrical biker crest, no text"),
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
            "no brown, no beige, no fabric texture, no embroidery"
        ),
        subjects=[
            ("butterfly_trail", "flat vector butterfly graphic, upper abstract wordmark band, central large butterfly, smaller butterflies trailing below along a dotted path, light blue and white solid colors"),
            ("word_butterfly",  "flat vector abstract letters across the top, large butterfly graphic in the center, small star and heart accents, light blue and white solid colors"),
            ("shape_collage",   "flat vector collage of overlapping geometric shapes in denim blue, central butterfly graphic, star and heart accents, clean edges"),
            ("floral_butterfly","flat vector butterfly surrounded by small flowers and dotted trail, upper ornamental wordmark, light blue and white solid colors"),
        ],
        sim=0.42,   # 极低材质锁，避免把真实牛仔布纹理带进来；构图交给 ControlNet
        comp=0.60,  # 保留上方文字+中央主图+下方小元素 的构图
        cn=0.55,
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
            ("skull_wing_snake", "skull with spread red wings, snake coiled around, red roses at sides, blood drips, symmetrical emblem"),
            ("skull_bat_wings",  "skull with bat wings, snake and thorny roses, dark red accents, symmetrical gothic badge"),
            ("skull_raven_wings","skull with raven black wings, snake and roses, blood drops, dark symmetrical crest"),
            ("skull_roses",      "skull surrounded by red roses and thorns, wing-like floral frame, snake at base, symmetrical emblem"),
        ],
        sim=0.72,
        comp=0.58,  # 保留中央骷髅+两侧翅膀/玫瑰+上下横幅 的构图
        cn=0.45,
    ),

    # metal_6：金属 logo+鹰+角骷髅+闪电 → 元素在「重金属乐队艺术」语汇内裂变
    "metal_6": make_config(
        type_label="heavy_metal_badge",
        style_words=(
            "heavy metal band art print, spiked ornamental abstract lettering, no readable words, "
            "eagle and horned skull, radiating lightning bolts, "
            "dark underground metal emblem, black white and brown"
        ),
        subjects=[
            ("eagle_horned_skull",  "eagle with spread wings above a horned skull, radiating lightning bolts, spiked metal letterform banner at top, symmetrical emblem"),
            ("skull_lightning",     "screaming skull with horns, lightning bolts radiating behind, spiked ornamental letterform banner above, death metal crest"),
            ("raven_skull",         "raven with outstretched wings above horned skull, lightning and spikes, underground metal emblem"),
            ("winged_horned_skull", "large horned skull with wings, lightning radiating, spiked ornamental lettering above, symmetrical metal badge"),
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
IPA_NOISE = 0.10
IPA_END = 0.65
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
        "usdu_model": "4x_NMKD-Siax_200k.pth",
        "negative_prompt": (
            "photography, product photo, 3d render, realistic texture, fabric folds, "
            "wrinkles, shadows, depth of field, blurry, deformed, low quality, "
            "readable text, real words, fake words, banner text, ribbon text, pseudo-words, "
            "brand name, trademark, watermark, signature, brand logo, copyrighted character, "
            "cropped, out of frame, mockup, garment"
        ),
        "width": 1024, "height": 1024, "batch_per_run": 1,
        "steps": 35, "cfg": 6.0,
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

    g = build_mode1(seed_name, params, f"batch_{orig_label}_{sub_label}")
    t0 = time.time()
    try:
        res = client.run(g, timeout=360)
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

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>6x4 批量风格裂变 v7</title>
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
<h1>图裂变 · 6 原图 × 4 主题批量 v7（参考原图构图+设计理念裂变）</h1>
<div class="sub">6 张设计稿 → 4 个同设计语言新主题 = 24 张可印全幅服装的平面图案</div>
<div class="params">全类构图参考 | 同设计语言元素裂变 | 双 LoRA | 4x 真实超分 | 无侵权内容</div>
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
    out_dir = os.path.join(JOBS_BASE, f"batch_6x4_v7_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[cfg] checkpoint={_cfg.SDXL_CHECKPOINT} textile={TEXTILE_LORA} tshirt={TSHIRT_LORA}")

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
