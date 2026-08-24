"""
单图验证 fp16 ControlNet 能否在 RTX 4070 Ti 12GB 上加载并跑通（不与其他图并发）。
目的：确认 build.py 的 ControlNet 路径在 12GB 卡上可用，作为双 IPAdapter 构图锁之外的「更强像素级构图锁」选项。

用法：python smoke_controlnet.py
"""
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1

COMFYUI_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
JOB_OUT = r"E:\Desktop\双接口\image-fission\jobs\smoke_controlnet"


def vram_mb(client):
    try:
        s = client.system_stats()
        dev = (s.get("devices") or [{}])[0]
        return dev.get("vram_total"), dev.get("vram_free")
    except Exception:
        return None, None


def main():
    os.makedirs(JOB_OUT, exist_ok=True)
    client = ComfyClient()

    seed = "pinterest_eagle_2.jpg"
    seed_name = f"batch_eagle_cn_{int(time.time()*1000)}.jpg"
    shutil.copy(os.path.join(COMFYUI_INPUT, seed),
                os.path.join(COMFYUI_INPUT, seed_name))

    params = {
        "similarity": 0.80,
        "ipadapter_weight_type": "style transfer",
        "ipadapter_noise": 0.10,
        "ipadapter_end": 0.80,
        "composition_strength": 0.0,   # 关掉双 IPAdapter 构图锁，单独测 ControlNet 显存
        "controlnet_name": "controlnet-canny-sdxl-1.0.fp16.safetensors",
        "controlnet_strength": 0.65,
        "controlnet_low_threshold": 100,
        "controlnet_high_threshold": 200,
        "usdu_model": "4x_NMKD-Siax_200k.pth",
        "negative_prompt": "blurry, deformed, low quality, text, watermark, signature, illustration, cartoon, 3d, render",
        "width": 768, "height": 1344, "batch_per_run": 1,
        "steps": 30, "cfg": 5.0,
    }
    prompt = ("keep the same color palette and composition as the original, "
              "a dark gothic eagle with spread black wings and roaring red orange flames, "
              "a bold emblem, symmetrical shield layout, monochrome with red accent, "
              "commercial product photography, sharp focus, 8k")

    vt, _ = vram_mb(client)
    print(f"[VRAM] total={vt/1024/1024/1024:.1f}GB" if vt else "[VRAM] unknown")

    g = build_mode1(seed_name, params, "cn_smoke")
    print(f"[graph] nodes={len(g)} (含 ControlNet 50/51/52)")

    t0 = time.time()
    try:
        res = client.run(g, timeout=600)
        out_path = os.path.join(JOB_OUT, "eagle_cn.jpg")
        data = next(iter(res.values()))[0]
        with open(out_path, "wb") as f:
            f.write(data)
        dt = time.time() - t0
        print(f"[OK] ControlNet 跑通！耗时 {dt:.0f}s，输出 {out_path} ({len(data)/1e6:.1f}MB)")
    except Exception as e:
        print(f"[FAIL] {repr(e)}")
        raise


if __name__ == "__main__":
    main()
