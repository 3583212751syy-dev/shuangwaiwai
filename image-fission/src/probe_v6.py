"""
极简探测：sd_xl_base + Canopus LoRA，无 USDU，快速验证能否出图。
"""
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1

COMFYUI_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
JOBS_BASE = r"E:\Desktop\双接口\image-fission\jobs"


def main():
    orig_seed = "pinterest_denim_3"
    seed_src = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
    seed_name = f"probe_v6_denim_{int(time.time()*1000)}.jpg"
    shutil.copy(seed_src, os.path.join(COMFYUI_INPUT, seed_name))

    prompt = (
        "high quality surface pattern design, textile print, 2d flat illustration, "
        "bold clean outlines, flat color blocks, screen print aesthetic, "
        "vintage americana patchwork print, blue indigo tones, stitched patch aesthetic, "
        "blue denim style patch with stylized UPGY lettering and a butterfly motif, "
        "no photography, no 3d render, no realistic texture, no fabric folds"
    )
    params = {
        "style_prompt": prompt,
        "similarity": 0.70,
        "ipadapter_weight_type": "style transfer",
        "ipadapter_noise": 0.10,
        "ipadapter_end": 0.65,
        "composition_strength": 0.30,
        "controlnet_name": "",
        "controlnet_strength": 0.0,
        "usdu_model": "",           # 关闭 USDU，最快路径
        "negative_prompt": (
            "photography, product photo, 3d render, realistic texture, fabric folds, "
            "wrinkles, shadows, depth of field, blurry, deformed, low quality"
        ),
        "width": 1024, "height": 1024, "batch_per_run": 1,
        "steps": 20, "cfg": 5.0,
        "seed": 12345,
        "lora_name": "Canopus-Textile-Pattern-adp-LoRA.safetensors",
        "lora_strength": 0.65,
    }
    g = build_mode1(seed_name, params, "probe_v6_denim")

    out_dir = os.path.join(JOBS_BASE, f"probe_v6_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)

    client = ComfyClient()
    print("[run] start")
    t0 = time.time()
    try:
        res = client.run(g, timeout=600)
        data = next(iter(res.values()))[0]
        out_path = os.path.join(out_dir, "denim_probe.jpg")
        with open(out_path, "wb") as f:
            f.write(data)
        dt = time.time() - t0
        print(f"[OK] {dt:.0f}s -> {out_path}")
    except Exception as e:
        dt = time.time() - t0
        print(f"[FAIL] {dt:.0f}s: {repr(e)}")


if __name__ == "__main__":
    main()
