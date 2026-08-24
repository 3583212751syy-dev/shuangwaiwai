"""Smoke test v5：单张验证 写实(camo) + 矢量(floral) 双 IPAdapter 构图锁 + USDU。"""
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1
import config as _cfg
_cfg.SDXL_CHECKPOINT = "juggernautXL_ragnarokBy.safetensors"

COMFYUI_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
OUT = r"E:\Desktop\双接口\image-fission\jobs\smoke_v5"
os.makedirs(OUT, exist_ok=True)

CASES = [
    ("camo_alt", "pinterest_camo_4", None, 0.85, False,
     "keep the same color palette and composition as the original, a military woodland camouflage pattern with a black palm tree silhouette as the hero motif, brown green tones, repeating fabric textile photography, sharp focus, 8k"),
    ("damask", "pinterest_illust_1", "DD-vector-v2.safetensors", 0.85, True,
     "vector, black line art, keep the same color palette and composition as the original, an intricate damask ornamental pattern with different botanical motifs, monochrome, decorative symmetrical, no text"),
]

client = ComfyClient()
for label, seed, lora, lora_strength, is_vec, prompt in CASES:
    seed_src = os.path.join(COMFYUI_INPUT, f"{seed}.jpg")
    seed_name = f"smoke_{seed}_{int(time.time()*1000)}.jpg"
    shutil.copy(seed_src, os.path.join(COMFYUI_INPUT, seed_name))
    params = {
        "similarity": 0.80,
        "ipadapter_weight_type": "style transfer",
        "ipadapter_noise": 0.10,
        "ipadapter_end": 0.80,
        "composition_strength": 0.55,
        "controlnet_name": "", "controlnet_strength": 0.0,
        "usdu_model": "4x_NMKD-Siax_200k.pth",
        "negative_prompt": ("blurry, deformed, low quality, text, watermark, signature, "
                            "different style, different color palette, vector, line art, "
                            "illustration, cartoon, 3d, render, gradient, flat, posterized"),
        "width": 768, "height": 1344, "batch_per_run": 1,
        "steps": 30, "cfg": 5.0, "seed": 7777,
        "style_prompt": prompt,
    }
    if lora:
        params["lora_name"] = lora
        params["lora_strength"] = lora_strength
    g = build_mode1(seed_name, params, f"smoke_{label}")
    t0 = time.time()
    res = client.run(g, timeout=300)
    data = next(iter(res.values()))[0]
    path = os.path.join(OUT, f"{label}.jpg")
    with open(path, "wb") as f:
        f.write(data)
    print(f"[OK] {label}  {time.time()-t0:.0f}s  {len(data)//1024} KB -> {path}")
print("SMOKE DONE")
