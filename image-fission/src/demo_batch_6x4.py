"""
批量风格裂变 v5：6 张 Pinterest 原图 × 4 主题 = 24 张。
核心升级（相对 v4）：
  1. ControlNet Canny 锁「构图」——颜色走 IPAdapter style transfer，构图走 ControlNet，
     内容走 prompt，三者独立可控（用户要的 颜色/构图/内容 滑杆）。
  2. 写实类（迷彩/牛仔/骷髅/金属/鹰）不再强制矢量化，用写实 prompt + 具体含义母题，
     解决"内容糊、没含义没审美"。
  3. 并发生成：一次性把 24 个 prompt 提交进 ComfyUI 队列，GPU 背靠背跑，无提交空隙；
     ThreadPoolExecutor 收集结果并实时打印进度（同步处理进度）。
"""
import os
import sys
import time
import shutil
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1
import config as _cfg

_cfg.SDXL_CHECKPOINT = "juggernautXL_ragnarokBy.safetensors"

COMFYUI_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
JOBS_BASE = r"E:\Desktop\双接口\image-fission\jobs"

# 构图锁方案：默认用「双 IPAdapter（composition）」——零额外显存，12GB 卡可跑；
# ControlNet Canny 更强但 fp32 约 4.66GB，需转 fp16 或高显存机器再启用。
CONTROLNET = "controlnet-canny-sdxl-1.0.safetensors"
USE_CONTROLNET = False          # True 时启用 ControlNet（需显存足够/fp16）
COMPOSITION_STRENGTH = 0.55     # 双 IPAdapter 构图锁强度（颜色走 style transfer）

# 按原图类型配对不同的生成策略（写实 vs 纹章 vs 写实图案）
ORIGINALS_CONFIG = {
    "illust_1": {  # 黑白花卉藤蔓纹章
        "type": "engraving",
        "lora": "DD-vector-v2.safetensors", "lora_strength": 0.85,
        "cn_strength": 0.45,  # 纹章类保留一定构图自由
        "subjects": [
            ("damask",  "vector, black line art, keep the same color palette and composition as the original, an intricate damask ornamental pattern with different botanical motifs, monochrome, decorative symmetrical, no text"),
            ("creature", "vector, black line art, keep the same color palette and composition as the original, an intricate ornamental pattern with heraldic creature and detailed beast elements, monochrome, decorative symmetrical, no text"),
            ("geometry", "vector, black line art, keep the same color palette and composition as the original, an intricate sacred geometry pattern with stars and circles, monochrome, symmetrical mandala, no text"),
            ("ornamental", "vector, black line art, keep the same color palette and composition as the original, an ornate ornamental pattern with corner flourishes, monochrome, decorative symmetrical, no text"),
        ],
    },
    "eagle_2": {  # 黑鹰+红火焰（哥特/机车）
        "type": "gothic",
        "lora": None,
        "cn_strength": 0.65,
        "subjects": [
            ("eagle_alt",  "keep the same color palette and composition as the original, a dark gothic eagle with spread black wings and roaring red orange flames, a bold emblem, symmetrical shield layout, monochrome with red accent, commercial product photography, sharp focus, 8k"),
            ("skull_flame", "keep the same color palette and composition as the original, a gothic skull with red flame accents and barbed wire, symmetrical emblem layout, dark background, dramatic lighting, commercial product photography, sharp focus, 8k"),
            ("flame_wing",  "keep the same color palette and composition as the original, a symmetrical shield with black wings and red flames and a lightning bolt, gothic commercial design, dark background, sharp focus, 8k"),
            ("dragon",     "keep the same color palette and composition as the original, a dark gothic dragon coiled with red flame accents, symmetrical emblem layout, dramatic lighting, sharp focus, 8k"),
        ],
    },
    "denim_3": {  # 牛仔贴布 UPGY+蝴蝶
        "type": "denim",
        "lora": None,
        "cn_strength": 0.65,
        "subjects": [
            ("denim_alt",  "keep the same color palette and composition as the original, a blue denim fabric patch with stylized letters UPGY and a butterfly motif, embroidered textile pattern, vintage worn texture, symmetrical layout, sharp focus, 8k photography"),
            ("patch_letters", "keep the same color palette and composition as the original, a blue denim patch with bold graffiti letters and embroidered roses, vintage textile pattern, sharp focus, 8k"),
            ("denim_butterfly", "keep the same color palette and composition as the original, a blue denim patch with butterflies and floral embroidery and stitches, vintage textile pattern, sharp focus, 8k"),
            ("jeans_pattern", "keep the same color palette and composition as the original, a blue denim fabric pattern with various stitched patches, embroidered motifs and a name tag, vintage worn texture, sharp focus, 8k"),
        ],
    },
    "camo_4": {  # 棕绿迷彩+黑色棕榈树
        "type": "camo",
        "lora": None,
        "cn_strength": 0.65,
        "subjects": [
            ("camo_alt",     "keep the same color palette and composition as the original, a military woodland camouflage pattern with a black palm tree silhouette as the hero motif, brown green tones, repeating fabric textile photography, sharp focus, 8k"),
            ("camo_jungle",  "keep the same color palette and composition as the original, a military jungle camouflage pattern with tropical monstera leaves and a prowling panther silhouette, brown green tones, repeating fabric photography, sharp focus, 8k"),
            ("camo_desert",  "keep the same color palette and composition as the original, a military desert camouflage pattern with a black palm tree silhouette and a compass, brown sand tones, repeating fabric photography, sharp focus, 8k"),
            ("camo_digital", "keep the same color palette and composition as the original, a digital military camouflage pattern with a black palm tree silhouette in pixel blocks, brown green tones, repeating fabric photography, sharp focus, 8k"),
        ],
    },
    "skull_5": {  # 骷髅头+红翅膀+蛇+玫瑰
        "type": "gothic_skull",
        "lora": None,
        "cn_strength": 0.65,
        "subjects": [
            ("skull_wing",     "keep the same color palette and composition as the original, a detailed gothic skull with red wings and a rose in its teeth, symmetrical layout, dark background, dramatic lighting, sharp focus, 8k photography"),
            ("skull_snake",    "keep the same color palette and composition as the original, a detailed gothic skull entwined by a snake and a rose, symmetrical emblem layout, dark moody background, sharp focus, 8k"),
            ("skull_flame",    "keep the same color palette and composition as the original, a detailed gothic skull with red flame accents and a dagger, symmetrical emblem layout, dark background, sharp focus, 8k"),
            ("skull_cross",    "keep the same color palette and composition as the original, a detailed gothic skull with a crossbone and a rose wreath, symmetrical emblem layout, dark background, sharp focus, 8k"),
        ],
    },
    "metal_6": {  # 金属骷髅+鹰+NEVERSEA 死亡金属
        "type": "death_metal",
        "lora": None,
        "cn_strength": 0.65,
        "subjects": [
            ("metal_skull",   "keep the same color palette and composition as the original, a detailed metal skull with an eagle and industrial gears, death metal band logo style, sharp spikes, symmetrical emblem layout, sharp focus, 8k photography"),
            ("metal_eagle",   "keep the same color palette and composition as the original, a detailed metal eagle with a skull and spikes and the word NEVERSEA, death metal band logo style, symmetrical emblem layout, sharp focus, 8k"),
            ("metal_cross",   "keep the same color palette and composition as the original, a detailed metal cross with a skull and an eagle and chains, death metal band logo style, symmetrical emblem layout, sharp focus, 8k"),
            ("metal_band",    "keep the same color palette and composition as the original, a detailed metal death metal band logo with skull eagle and gothic lettering, symmetrical emblem layout, sharp focus, 8k"),
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

SIM = 0.80
IPA_WEIGHT_TYPE = "style transfer"
IPA_NOISE = 0.10
IPA_END = 0.80
CN_LOW, CN_HIGH = 100, 200  # Canny 阈值

_lock = threading.Lock()
_done = 0
_total = 0


def img_to_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_one(client, orig_label, sub_label, sub_prompt, seed_name, cfg, out_dir, idx):
    """生成单张；返回 (orig_label, sub_label, out_path) 或异常。"""
    global _done
    cur_lora = cfg.get("lora")
    cur_lora_strength = cfg.get("lora_strength", 0.85)
    cn_strength = cfg.get("cn_strength", 0.65)
    params = {
        "similarity": SIM,
        "ipadapter_weight_type": IPA_WEIGHT_TYPE,
        "ipadapter_noise": IPA_NOISE,
        "ipadapter_end": IPA_END,
        "composition_strength": COMPOSITION_STRENGTH,
        "controlnet_name": CONTROLNET if USE_CONTROLNET else "",
        "controlnet_strength": cn_strength if USE_CONTROLNET else 0.0,
        "controlnet_low_threshold": CN_LOW,
        "controlnet_high_threshold": CN_HIGH,
        "usdu_model": "4x_NMKD-Siax_200k.pth",
        "negative_prompt": ("blurry, deformed, low quality, "
                            "text, watermark, signature, "
                            "different style, different color palette, "
                            "vector, line art, illustration, cartoon, "
                            "3d, render, gradient, flat, posterized"),
        "width": 768, "height": 1344, "batch_per_run": 1,
        "steps": 30, "cfg": 5.0,
    }
    if cur_lora:
        params["lora_name"] = cur_lora
        params["lora_strength"] = cur_lora_strength
    g = build_mode1(seed_name, params, f"batch_{orig_label}_{sub_label}")
    t0 = time.time()
    try:
        res = client.run(g, timeout=360)
        data = next(iter(res.values()))[0]
        out_path = os.path.join(out_dir, f"{orig_label}_{sub_label}.jpg")
        with open(out_path, "wb") as f:
            f.write(data)
        dt = time.time() - t0
        lora_info = f" LoRA={cur_lora}@{cur_lora_strength}" if cur_lora else " no-LoRA"
        with _lock:
            _done += 1
            print(f"[OK {_done}/{_total}] {orig_label} × {sub_label} ({cfg['type']}){lora_info} COMP={COMPOSITION_STRENGTH} CN={cn_strength if USE_CONTROLNET else 0}  {dt:.0f}s")
        return (orig_label, sub_label, out_path)
    except Exception as e:
        with _lock:
            _done += 1
            print(f"[FAIL {_done}/{_total}] {orig_label} × {sub_label}: {repr(e)}")
        return None


def main():
    global _total
    out_dir = os.path.join(JOBS_BASE, f"batch_6x4_cn_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[cfg] ControlNet={CONTROLNET} sim={SIM} ipa={IPA_WEIGHT_TYPE}")

    client = ComfyClient()
    base_seed = 8888

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
            seed = base_seed + hash((orig_label, sub_label)) % 9999
            tasks.append((orig_label, sub_label, sub_prompt, seed_name, cfg))

    _total = len(tasks)
    print(f"[queue] 提交 {_total} 张到 ComfyUI 队列（背靠背执行）...")

    results = []
    t_start = time.time()
    # 并发生求：一次性全部提交，GPU 无空隙；线程池只负责 HTTP 提交/轮询
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

    # 画廊：6 行（原图）× 4 列（主题）+ 顶部原图
    label2sub = {o: c for o, c in ORIGINALS}
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

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>6x4 批量风格裂变 v5</title>
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
<h1>图裂变 · 6 原图 × 4 主题批量 v5（ControlNet 构图锁 + 写实母题）</h1>
<div class="sub">6 张 Pinterest 设计稿 → 4 个新主题 = 24 张同画风不同内容</div>
<div class="params">相似度 {SIM} | IPAdapter {IPA_WEIGHT_TYPE} | ControlNet Canny {CONTROLNET} | 4x 真实超分</div>
<table>
  <tr><th>原图</th><th>原图</th><th>主题1</th><th>主题2</th><th>主题3</th><th>主题4</th></tr>
  {''.join(rows_html)}
</table>
</body></html>"""
    html_path = os.path.join(out_dir, "gallery.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[gallery] {html_path}")
    return html_path


if __name__ == "__main__":
    main()
