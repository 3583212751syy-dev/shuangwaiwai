"""
批量风格裂变：6 张 Pinterest 原图 × 4 主题 = 24 张，全部加 vector LoRA。
治"糊"用 Doctor Diffusion Vector Art XL LoRA（trigger word: vector + black line art）。
"""
import os
import sys
import time
import shutil
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1
import config as _cfg

_cfg.SDXL_CHECKPOINT = "juggernautXL_ragnarokBy.safetensors"

COMFYUI_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
JOBS_BASE = r"E:\Desktop\双接口\image-fission\jobs"

# 按原图类型配对不同的生成策略（写实 vs 纹章 vs 写实图案）
# 核心修复：之前用 Vector LoRA 0.85 强制矢量化所有原图，方向性错误 —
# 写实类原图（迷彩/牛仔/金属/骷髅）应该用写实 prompt，不能被压成"纹章"
ORIGINALS_CONFIG = {
    "illust_1": {  # 黑白花卉藤蔓纹章
        "type": "engraving",
        "lora": "DD-vector-v2.safetensors", "lora_strength": 0.85,
        "subjects": [
            ("damask",  "vector, black line art, keep the same color palette and composition as the original, an intricate damask ornamental pattern with different botanical motifs, monochrome, decorative symmetrical, no text"),
            ("creature", "vector, black line art, keep the same color palette and composition as the original, an intricate ornamental pattern with heraldic creature and detailed beast elements, monochrome, decorative symmetrical, no text"),
            ("geometry", "vector, black line art, keep the same color palette and composition as the original, an intricate sacred geometry pattern with stars and circles, monochrome, symmetrical mandala, no text"),
            ("ornamental", "vector, black line art, keep the same color palette and composition as the original, an ornate ornamental pattern with corner flourishes, monochrome, decorative symmetrical, no text"),
        ],
    },
    "eagle_2": {  # 黑鹰+红火焰（哥特/机车）
        "type": "gothic",
        "lora": None,  # 不用 LoRA，让 SDXL 写实出
        "subjects": [
            ("eagle_alt",  "keep the same color palette and composition as the original, a dark gothic eagle and flame emblem, detailed black wings spread, red orange flames, symmetrical shield layout, monochrome with red accent, commercial product photography, sharp focus, 8k"),
            ("skull_flame", "keep the same color palette and composition as the original, a gothic skull with red flame accents, symmetrical emblem layout, dark background, dramatic lighting, commercial product photography, sharp focus, 8k"),
            ("flame_wing",  "keep the same color palette and composition as the original, a symmetrical shield with black wings and red flames, gothic commercial design, dark background, sharp focus, 8k"),
            ("dragon",     "keep the same color palette and composition as the original, a dark gothic dragon with red flame accents, symmetrical emblem layout, dramatic lighting, sharp focus, 8k"),
        ],
    },
    "denim_3": {  # 牛仔贴布 UPGY+蝴蝶
        "type": "denim",
        "lora": None,
        "subjects": [
            ("denim_alt",  "keep the same color palette and composition as the original, a blue denim fabric patch with stylized letters and butterfly motifs, embroidered textile pattern, vintage worn texture, symmetrical layout, sharp focus, 8k photography"),
            ("patch_letters", "keep the same color palette and composition as the original, a blue denim patch with bold graffiti letters and embroidered motifs, vintage textile pattern, sharp focus, 8k"),
            ("denim_butterfly", "keep the same color palette and composition as the original, a blue denim patch with butterflies and floral embroidery, vintage textile pattern, sharp focus, 8k"),
            ("jeans_pattern", "keep the same color palette and composition as the original, a blue denim fabric pattern with various stitched patches and embroidered motifs, vintage worn texture, sharp focus, 8k"),
        ],
    },
    "camo_4": {  # 棕绿迷彩+黑色棕榈树
        "type": "camo",
        "lora": None,
        "subjects": [
            ("camo_alt",     "keep the same color palette and composition as the original, a military woodland camouflage pattern with black palm tree silhouettes, brown green tones, repeating pattern, fabric textile photography, sharp focus, 8k"),
            ("camo_jungle",  "keep the same color palette and composition as the original, a military jungle camouflage pattern with tropical leaves and trees, brown green tones, repeating pattern, fabric photography, sharp focus, 8k"),
            ("camo_desert",  "keep the same color palette and composition as the original, a military desert camouflage pattern with black palm tree silhouettes, brown sand tones, repeating pattern, fabric photography, sharp focus, 8k"),
            ("camo_digital", "keep the same color palette and composition as the original, a digital military camouflage pattern with black palm tree silhouettes, brown green tones, repeating pixel pattern, fabric photography, sharp focus, 8k"),
        ],
    },
    "skull_5": {  # 骷髅头+红翅膀+蛇+玫瑰
        "type": "gothic_skull",
        "lora": None,  # 写实骷髅不用 LoRA
        "subjects": [
            ("skull_wing",     "keep the same color palette and composition as the original, a detailed gothic skull with red wings and rose accents, symmetrical layout, dark background, dramatic lighting, sharp focus, 8k photography"),
            ("skull_snake",    "keep the same color palette and composition as the original, a detailed gothic skull with snake and rose, symmetrical emblem layout, dark moody background, sharp focus, 8k"),
            ("skull_flame",    "keep the same color palette and composition as the original, a detailed gothic skull with red flame accents, symmetrical emblem layout, dark background, sharp focus, 8k"),
            ("skull_cross",    "keep the same color palette and composition as the original, a detailed gothic skull with cross and rose, symmetrical emblem layout, dark background, sharp focus, 8k"),
        ],
    },
    "metal_6": {  # 金属骷髅+鹰+NEVERSEA 死亡金属
        "type": "death_metal",
        "lora": None,
        "subjects": [
            ("metal_skull",   "keep the same color palette and composition as the original, a detailed metal skull with eagle and industrial elements, death metal band logo style, sharp spikes, symmetrical emblem layout, sharp focus, 8k photography"),
            ("metal_eagle",   "keep the same color palette and composition as the original, a detailed metal eagle with skull and spikes, death metal band logo style, symmetrical emblem layout, sharp focus, 8k"),
            ("metal_cross",   "keep the same color palette and composition as the original, a detailed metal cross with skull and eagle, death metal band logo style, symmetrical emblem layout, sharp focus, 8k"),
            ("metal_band",    "keep the same color palette and composition as the original, a detailed metal death metal band logo with skull and eagle, symmetrical emblem layout, sharp focus, 8k"),
        ],
    },
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

LORA_DEFAULT = "DD-vector-v2.safetensors"
SIM = 0.80  # 提高到 0.80，强锁原图风格/色调/构图
IPA_WEIGHT_TYPE = "style transfer"   # SDXL 风格锁（IPAdapter Plus 1.5+ 推荐）
IPA_NOISE = 0.10                     # 加噪防止完全照搬
IPA_END = 0.80                       # 前 80% 强影响，后 20% 让模型自由发挥


def img_to_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    out_dir = os.path.join(JOBS_BASE, f"batch_6x4_vectorlora_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[cfg] LoRA={LORA}@{LORA_STRENGTH}  sim={SIM}")

    client = ComfyClient()
    base_seed = 8888
    results = []  # (orig_label, sub_label, path)
    t_start = time.time()

    for orig_seed, orig_label in ORIGINALS:
        seed_src = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(seed_src):
            print(f"[WARN] 跳过: {seed_src} 不存在")
            continue
        # 复制 seed 到 input（用带时间戳名字，避免冲突）
        seed_name = f"batch_{orig_label}_{int(time.time()*1000)}.jpg"
        shutil.copy(seed_src, os.path.join(COMFYUI_INPUT, seed_name))

        # 读取该原图的专属配置
        cfg = ORIGINALS_CONFIG.get(orig_label, ORIGINALS_CONFIG["illust_1"])
        cur_lora = cfg.get("lora")
        cur_lora_strength = cfg.get("lora_strength", 0.85)

        for sub_label, sub_prompt in cfg["subjects"]:
            params = {
                "similarity": SIM,
                "ipadapter_weight_type": IPA_WEIGHT_TYPE,
                "ipadapter_noise": IPA_NOISE,
                "ipadapter_end": IPA_END,
                "usdu_model": "4x_NMKD-Siax_200k.pth",  # 治糊：4x 真实超分
                "negative_prompt": ("blurry, deformed, low quality, "
                                     "text, watermark, signature, "
                                     "different style, different color palette, "
                                     "vector, line art, illustration, cartoon, "
                                     "3d, render, gradient, flat, posterized"),
                "width": 768, "height": 1344, "batch_per_run": 1,
                "steps": 30, "cfg": 5.0,
                "hires_scale": 1.5, "hires_denoise": 0.40, "hires_steps": 20,
                "seed": base_seed + hash((orig_label, sub_label)) % 9999,
                "style_prompt": sub_prompt,
            }
            if cur_lora:
                params["lora_name"] = cur_lora
                params["lora_strength"] = cur_lora_strength
            g = build_mode1(seed_name, params, f"batch_{orig_label}_{sub_label}")
            t0 = time.time()
            try:
                res = client.run(g, timeout=300)
                data = next(iter(res.values()))[0]
                out_path = os.path.join(out_dir, f"{orig_label}_{sub_label}.jpg")
                with open(out_path, "wb") as f:
                    f.write(data)
                dt = time.time() - t0
                lora_info = f" LoRA={cur_lora}@{cur_lora_strength}" if cur_lora else " no-LoRA"
                print(f"[OK ] {orig_label} × {sub_label} ({cfg['type']}){lora_info}  {dt:.0f}s  {len(data)} B")
                results.append((orig_label, sub_label, out_path))
            except Exception as e:
                print(f"[FAIL] {orig_label} × {sub_label}: {repr(e)}")

    total = time.time() - t_start
    print(f"\n[DONE] {len(results)}/24  总耗时 {total:.0f}s")

    if not results:
        return

    # 画廊：6 行（原图）× 4 列（主题）+ 顶部原图
    rows_html = []
    for orig_seed, orig_label in ORIGINALS:
        orig_path = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(orig_path):
            continue
        seed_b64 = img_to_b64(orig_path)
        cells = ""
        for sub_label, _ in SUBJECTS:
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

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>6x4 批量风格裂变</title>
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
<h1>图裂变 · 6 原图 × 4 主题批量（Vector LoRA 治糊）</h1>
<div class="sub">6 张 Pinterest 设计稿 → 4 个新主题 = 24 张同画风不同内容</div>
<div class="params">LoRA = {LORA} (strength {LORA_STRENGTH}) | similarity = {SIM} | prompt trigger: vector, black line art, white background</div>
<table>
  <tr><th>原图</th><th>原图</th><th>damask</th><th>creature</th><th>geometry</th><th>frame</th></tr>
  {''.join(rows_html)}
</table>
</body></html>"""
    html_path = os.path.join(out_dir, "gallery.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[gallery] {html_path}")


if __name__ == "__main__":
    main()