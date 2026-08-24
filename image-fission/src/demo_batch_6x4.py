"""
批量风格裂变 v6.5：按原图类型差异化裂变 + 欧美全幅印花方向。
核心策略（相对 v6）：
  1. 阴阳/装饰图（illust_1）与迷彩图（camo_4）保留其「构图风格」：高 composition 锁 + 高相似度。
  2. 其余四类（eagle/denim/skull/metal）按「元素裂变」：低 composition 锁 + 中等相似度，
     由 prompt 内容主导，避免同组四张长得一样。
  3. 全局锁定「可印在全幅衣服上的图案」域：2D flat all-over print / t-shirt graphic，
     明确排除摄影/3D/实物褶皱/品牌商标/侵权内容。
  4. 双 LoRA：Canopus Textile Pattern（印花图案）+ chrisconyers T-shirts（T恤图案）。
  5. 并发 + 实时进度 + 轻量画廊。
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

# 全局印花/T恤 LoRA
TEXTILE_LORA = "Canopus-Textile-Pattern-adp-LoRA.safetensors"
TEXTILE_LORA_STRENGTH = 0.55
TSHIRT_LORA = "chrisconyers-sdxl-tshirts-lora.safetensors"
TSHIRT_LORA_STRENGTH = 0.45

# 矢量线描 LoRA：仅用于 engraving 类，增强 ornamental 图案感
VECTOR_LORA = "DD-vector-v2.safetensors"

# ControlNet Canny fp16（可选，默认关闭以保速度；需要强像素级构图锁时开启）
CONTROLNET = "controlnet-canny-sdxl-1.0.fp16.safetensors"
USE_CONTROLNET = False

# 公共风格前缀：明确「全幅衣服印花 / T恤图案」
STYLE_PREFIX = (
    "high quality all over print t-shirt design, surface pattern design, "
    "2d flat graphic illustration, bold clean outlines, flat color blocks, "
    "screen print aesthetic, centered composition, scalable vector style, "
    "print ready artwork, no background scene, "
    "no photography, no 3d render, no realistic texture, no fabric folds, "
    "no wrinkles, no shadows, no depth of field, no product shot, "
    "no text, no watermark, no signature, no brand logo, no trademark, no copyrighted character"
)

STYLE_SUFFIX = "sharp focus, crisp edges, professional apparel print"


def make_config(
    type_label,
    style_words,
    subjects,
    extra_lora=None,
    extra_lora_strength=0.45,
    sim=0.70,
    comp=0.35,
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
    # illust_1：黑白高对比装饰纹章 → 保留「阴阳/对称构图」风格，换不同文化母题
    "illust_1": make_config(
        type_label="yin_yang_ornamental",
        style_words=(
            "black and white ornamental engraving, art nouveau floral pattern, "
            "high contrast yin yang composition, elegant botanical silhouette, "
            "symmetrical decorative crest"
        ),
        subjects=[
            ("damask",     "intricate damask ornamental pattern with blooming botanical motifs, symmetrical scrollwork, monochrome decorative repeat"),
            ("creature",   "heraldic ornamental pattern with stylized beast and flowering vines, monochrome decorative symmetrical crest"),
            ("geometry",   "sacred geometry ornamental pattern with stars, circles and botanical frames, monochrome symmetrical mandala"),
            ("ornamental", "ornate ornamental pattern with corner flourishes, peonies and acanthus leaves, monochrome decorative symmetrical"),
        ],
        extra_lora=VECTOR_LORA,
        extra_lora_strength=0.40,
        sim=0.78,   # 保留原图风格
        comp=0.72,  # 保留对称构图
        cn=0.55,
    ),

    # eagle_2：黑鹰+红火焰 → 按元素裂变：机车徽章 / 火焰 / 龙 / 盾徽
    "eagle_2": make_config(
        type_label="gothic_biker",
        style_words=(
            "gothic biker emblem t-shirt print, red orange flame accents on black, "
            "bold graphic badge, tattoo flash style, streetwear crest"
        ),
        subjects=[
            ("eagle_alt",   "dark gothic eagle with spread black wings and roaring red orange flames, bold symmetrical shield emblem"),
            ("skull_flame", "gothic skull with red flame accents and barbed wire, symmetrical badge layout, dark background"),
            ("flame_wing",  "symmetrical shield with black wings, red flames and a lightning bolt, gothic graphic print"),
            ("dragon",      "dark gothic dragon coiled with red flame accents, symmetrical emblem, bold outline print"),
        ],
        sim=0.68,
        comp=0.28,
        cn=0.35,
    ),

    # denim_3：牛仔贴布+蝴蝶 → 按元素裂变：美式补丁 / 涂鸦字母 / 蝴蝶刺绣 / 工装标签
    # 注意：用户明确不要「实体牛仔布照片」，所以 prompt 里用 "denim style patch" 而非 "denim texture"
    "denim_3": make_config(
        type_label="vintage_patchwork",
        style_words=(
            "vintage americana patchwork t-shirt print, blue indigo tones, "
            "stitched patch aesthetic, distressed workwear graphic, "
            "flat graphic print, no real denim fabric texture"
        ),
        subjects=[
            ("denim_alt",      "blue denim style patch with stylized butterfly motif and stitched border, vintage workwear print"),
            ("patch_letters",  "blue patch with bold varsity letters and embroidered roses, stitched edges, vintage textile graphic"),
            ("denim_butterfly","blue denim style patch with butterflies, floral embroidery and stitch details, vintage americana print"),
            ("jeans_pattern",  "patchwork style pattern with various stitched patches, embroidered motifs and a name tag, workwear graphic"),
        ],
        sim=0.65,
        comp=0.25,
        cn=0.30,
    ),

    # camo_4：迷彩 → 保留「迷彩构图」风格，只换前景英雄元素
    "camo_4": make_config(
        type_label="military_camo",
        style_words=(
            "military camouflage all over print, woodland camo blocks, "
            "brown green khaki tones, streetwear graphic pattern, "
            "repeating camo background with central hero motif"
        ),
        subjects=[
            ("camo_alt",     "woodland camouflage pattern with a black palm tree silhouette as the hero motif, brown green tones, repeating fabric print"),
            ("camo_jungle",  "jungle camouflage pattern with tropical monstera leaves and a prowling panther silhouette, green brown tones"),
            ("camo_desert",  "desert camouflage pattern with a black scorpion silhouette and a compass rose, sand brown tones"),
            ("camo_digital", "digital military camouflage pattern with a black wolf head in pixel blocks, olive brown tones"),
        ],
        sim=0.78,   # 保留迷彩颜色/风格
        comp=0.70,  # 保留迷彩构图
        cn=0.55,
    ),

    # skull_5：骷髅头+红翅膀+蛇+玫瑰 → 按元素裂变，但保留哥特/亡灵节审美
    "skull_5": make_config(
        type_label="gothic_day_of_dead",
        style_words=(
            "gothic day-of-the-dead t-shirt print, red orange accents on dark ground, "
            "bold graphic skull art, tattoo inspired pattern, streetwear emblem"
        ),
        subjects=[
            ("skull_wing",  "detailed gothic skull with red wings and a rose in its teeth, symmetrical layout, dark background, bold outlines"),
            ("skull_snake", "detailed gothic skull entwined by a snake and a rose, symmetrical emblem, dark moody background"),
            ("skull_flame", "detailed gothic skull with red flame accents and a dagger, symmetrical emblem, dark background"),
            ("skull_cross", "detailed gothic skull with crossbones and a rose wreath, symmetrical emblem, dark background"),
        ],
        sim=0.68,
        comp=0.30,
        cn=0.35,
    ),

    # metal_6：金属骷髅+鹰 → 按元素裂变：重金属乐队艺术（避免具体乐队名/商标）
    "metal_6": make_config(
        type_label="heavy_metal",
        style_words=(
            "heavy metal band t-shirt art print, chrome silver and black, "
            "aggressive spiked emblem, underground metal poster style, "
            "symmetrical dark emblem"
        ),
        subjects=[
            ("metal_skull", "detailed metal skull with an eagle and industrial gears, spiked border, death metal emblem"),
            ("metal_eagle", "detailed metal eagle with a skull and spikes, death metal band logo style, chrome and black"),
            ("metal_cross", "detailed metal cross with a skull, eagle and chains, death metal emblem, symmetrical"),
            ("metal_band",  "death metal band logo with skull, eagle and gothic lettering, symmetrical emblem, chrome and black"),
        ],
        sim=0.68,
        comp=0.30,
        cn=0.35,
    ),
}

# 6 张原图
ORIGINALS = [
    ("pinterest_illust_1", "illust_1"),
    ("pinterest_eagle_2",  "eagle_2"),
    ("pinterest_denim_3",  "denim_3"),
    ("pinterest_camo_4",   "camo_4"),
    ("pinterest_skull_5",  "skull_5"),
    ("pinterest_metal_6",  "metal_6"),
]

IPA_WEIGHT_TYPE = "style transfer"
IPA_NOISE = 0.10
IPA_END = 0.65
CN_LOW, CN_HIGH = 100, 200

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
    comp = cfg.get("comp", 0.35)
    cn = cfg.get("cn", 0.40)
    extra_lora = cfg.get("extra_lora")
    extra_lora_strength = cfg.get("extra_lora_strength", 0.45)

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
            "text, watermark, signature, brand logo, trademark, copyrighted character, "
            "cropped, out of frame, mockup"
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

    # 可选第三类 LoRA（目前只有 engraving 类用矢量线描）
    if extra_lora:
        # build_mode1 目前只支持两个 LoRA；把 extra 当作第二个，tshirt 降下来或合并到 prompt
        # 策略：用 extra_lora 替换 tshirt LoRA（engraving 更需要矢量感）
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

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>6x4 批量风格裂变 v6.5</title>
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
<h1>图裂变 · 6 原图 × 4 主题批量 v6.5（全幅印花 / 差异化构图锁）</h1>
<div class="sub">6 张 Pinterest 设计稿 → 4 个新主题 = 24 张可印衣服的欧美风格图案</div>
<div class="params">阴阳/迷彩高构图锁 | 其余元素裂变 | 双 LoRA | 4x 真实超分 | 无侵权内容</div>
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
    out_dir = os.path.join(JOBS_BASE, f"batch_6x4_v65_{int(time.time())}")
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
