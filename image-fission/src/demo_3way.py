"""
3 模型对比 demo：同一 seed、同一 prompt、同一原图，三个 checkpoint 各出一张。
每个模型用各自的官方推荐参数（从 Civitai/HF 拉下来的）— 避免"用错的参数评对的模型"。

模型参数（已查证）：
  Juggernaut XL v9:  sampler=dpmpp_2m_karras, steps=30, cfg=5.0,  hires=1.5x/0.35/20
  Juggernaut XL v10: sampler=dpmpp_2m_sde,    steps=35, cfg=4.0,  hires=1.5x/0.30/15 (官方推荐)
  RealVisXL V5:     sampler=dpmpp_sde_karras, steps=35, cfg=2.0,  hires=1.5x/0.20/25 (官方推荐)

用法：python demo_3way.py [out_dir]
"""
import os
import sys
import time
import shutil
import base64
from io import BytesIO
from PIL import Image
from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1
import config as _cfg

JOBS_DIR = r"E:\Desktop\双接口\image-fission\jobs"
SEED_SRC = r"E:\Desktop\双接口\image-fission\jobs\test_mode1\test_mode1_v2_0_node10_0.jpg"
SEED_IN_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input\three_way_seed.jpg"
OUT_DIR = os.path.join(JOBS_DIR, sys.argv[1] if len(sys.argv) > 1 else "demo_3way")
SEED_NAME = "three_way_seed.jpg"

# 三个对比配置：每个模型用各自推荐参数
CONFIGS = [
    {
        "label": "JuggernautXL_v9 (现行)",
        "ckpt": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "params": {
            "sampler": "dpmpp_2m", "scheduler": "karras",
            "steps": 30, "cfg": 5.0,
            "similarity": 0.5,
            "negative_prompt": "low quality, blurry, deformed, watermark, text, extra fingers",
            "hires_scale": 1.5, "hires_denoise": 0.35, "hires_steps": 20,
        },
    },
    {
        "label": "JuggernautXL_Ragnarok (v10)",
        "ckpt": "juggernautXL_ragnarokBy.safetensors",
        "params": {
            "sampler": "dpmpp_2m_sde", "scheduler": "karras",
            "steps": 35, "cfg": 4.0,
            "similarity": 0.5,
            "negative_prompt": "",  # 官方：起步不加 negative
            "hires_scale": 1.5, "hires_denoise": 0.30, "hires_steps": 15,
        },
    },
    {
        "label": "RealVisXL_V5",
        "ckpt": "RealVisXL_V5.0_fp16.safetensors",
        "params": {
            "sampler": "dpmpp_sde", "scheduler": "karras",
            "steps": 35, "cfg": 2.0,
            "similarity": 0.5,
            "negative_prompt": "low quality, blurry, deformed, watermark, text, extra fingers",
            "hires_scale": 1.5, "hires_denoise": 0.20, "hires_steps": 25,
        },
    },
]

# 4 个场景：覆盖 商业白底/木质桌面/大理石/室外自然
SCENES = [
    ("白底商业", "a clean pure white studio background, seamless white, soft diffused lighting, premium product photography, Canon EOS R5, 85mm f/1.4, shallow depth of field, sharp focus, 8k, ultra detailed"),
    ("原木桌面", "a premium tea product photo, warm wooden table, soft natural light from window, cozy atmosphere, Canon EOS R5, 85mm f/1.4, shallow depth of field, sharp focus, 8k, ultra detailed"),
    ("大理石",  "a luxury tea product photo on a polished white marble surface, soft side lighting, editorial product photography, Canon EOS R5, 85mm f/1.4, shallow depth of field, sharp focus, 8k, ultra detailed"),
    ("山野自然", "a premium tea product photo in a lush mountain meadow, soft golden hour sunlight, fresh atmosphere, Canon EOS R5, 85mm f/1.4, shallow depth of field, sharp focus, 8k, ultra detailed"),
]


def setup():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SEED_IN_INPUT), exist_ok=True)
    if not os.path.exists(SEED_IN_INPUT):
        shutil.copy(SEED_SRC, SEED_IN_INPUT)
        print(f"[seed] copied -> {SEED_IN_INPUT}")


def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_one(client, cfg, scene_name, scene_prompt, seed):
    params = dict(cfg["params"])
    params["style_prompt"] = scene_prompt
    params["seed"] = seed
    params["width"] = 1024
    params["height"] = 1024
    params["batch_per_run"] = 1

    # 临时把 config.SDXL_CHECKPOINT 改成目标 ckpt — build.py 动态读这个常量
    _cfg.SDXL_CHECKPOINT = cfg["ckpt"]
    job_id = f"3way_{cfg['label'].split()[0]}_{scene_name}"
    g = build_mode1(SEED_NAME, params, job_id)
    t0 = time.time()
    out = None
    try:
        res = client.run(g, timeout=300)
        node_id, imgs = next(iter(res.items()))
        data = imgs[0]
        dst = os.path.join(OUT_DIR, f"{cfg['label'].split()[0]}_{scene_name}.jpg")
        with open(dst, "wb") as f:
            f.write(data)
        out = dst
    except Exception as e:
        print(f"[FAIL] {cfg['label']} × {scene_name}: {repr(e)}")
    dt = time.time() - t0
    if out:
        print(f"[OK ] {cfg['label']} × {scene_name}  {dt:.0f}s  -> {os.path.basename(out)} ({os.path.getsize(out)} bytes)")
    # 不还原 — 下个 config 会覆盖，结束后由 main 还原
    return out


def main():
    setup()
    print(f"[out] {OUT_DIR}")
    client = ComfyClient()
    base_seed = 7777
    grid = {}
    orig_ckpt = _cfg.SDXL_CHECKPOINT
    try:
        for cfg in CONFIGS:
            for scene_name, scene_prompt in SCENES:
                out = run_one(client, cfg, scene_name, scene_prompt, base_seed)
                grid[(cfg["label"], scene_name)] = out
    finally:
        _cfg.SDXL_CHECKPOINT = orig_ckpt  # 还原
    build_html(grid)
    print(f"\n[done] gallery: {OUT_DIR}/gallery.html")


def build_html(grid):
    rows = []
    header = "<th></th>" + "".join(f"<th>{c['label']}</th>" for c in CONFIGS)
    for scene_name, _ in SCENES:
        cells = "".join(
            f'<td>{"<img src=\"data:image/jpeg;base64," + img_to_b64(grid[(c["label"], scene_name)]) + "\">" if grid.get((c["label"], scene_name)) else "—"}</td>'
            for c in CONFIGS
        )
        rows.append(f"<tr><th>{scene_name}</th>{cells}</tr>")
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>3-way 写实模型对比</title>
<style>
body{{font-family:system-ui;background:#f5f5f5;padding:24px;margin:0;}}
h1{{font-size:20px;margin:0 0 16px;}}
table{{border-collapse:collapse;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.1);}}
th,td{{padding:8px;text-align:center;vertical-align:middle;}}
th{{background:#222;color:#fff;font-size:13px;}}
td img{{max-width:340px;display:block;}}
tr:nth-child(even) td{{background:#fafafa;}}
</style></head><body>
<h1>图裂变 · 写实 SDXL 模型 3 选对比（同一 seed / 同一 prompt）</h1>
<table><tr>{header}</tr>{''.join(rows)}</table>
</body></html>"""
    with open(os.path.join(OUT_DIR, "gallery.html"), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
