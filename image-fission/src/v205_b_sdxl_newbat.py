"""
v205_b — SDXL 独立生成新蝙蝠（最小 workflow，无 IPAdapter 无 ControlNet）

设计思路：
  - **完全不用 IPAdapter / ControlNet**（这两个会污染原图风格）
  - 仅用 LoRA DD-vector-v2 锁矢量插画风
  - prompt 主导蝙蝠姿态/角度/构图
  - 输出 1024x1024（白底方形，v205_c 切前景合成）
  - 模型基线用 Juggernaut XL v9 + LoRA DD-vector-v2 0.70
  - 背景：透明（v205_c 切前景时用 color threshold）

prompt 设计：
  - "new gothic bat with spread wings and downward dive pose"
  - "dark purple palette (#5C2A6E + #B77697)"
  - "vector illustration style, sharp angular wings, prominent claws"
  - 不用任何 "BACARDÍ / 文字 / logo / brand" 字样（避免 SDXL 学原图风格）

输出落地：
  - 异步跑 SDXL，保存到 ComfyUI/output/smoke_v205_b/mode1_00001_.png
  - 复制到 jobs/smoke_v205/v205_b_newbat.png（统一作业目录）
"""
import os
import sys
import time
import shutil
from pathlib import Path

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import COMFYUI_URL, COMFYUI_DIR, COMFYUI_WS
from engine.comfy_client import ComfyClient

PROJECT = Path("E:/Desktop/双接口/image-fission")
JOB_ID = "smoke_v205_b"
OUT_DIR = PROJECT / "ComfyUI" / "output" / JOB_ID
SAVE_PREFIX = f"{JOB_ID}/mode1"

# ---- prompt ----
POSITIVE = (
    "a single simple gothic heraldic crest centered in frame, "
    "dusk vintage distillation bottle label style, "
    "tightly cropped, NO outer frame, NO border, NO filigree, NO castle, "
    "ONLY the bat itself in center, "
    "gothic bat with sharp angular spread wings, "
    "downward swoop dive pose, prominent claws, "
    "two color monochrome purple duotone only, "
    "dark #2A0F40 base and #B77697 highlights, "
    "solid purple gradient background, "
    "no text, no watermark, no logo letters, no frame ornament"
)

NEGATIVE = (
    "low quality, blurry, deformed, watermark, text, letters, words, "
    "logo, brand, badge text, border frame, signature, "
    "extra fingers, mutated, jpeg artifacts, noise, "
    "cute, kawaii, cartoon, disney, chibi, anime, "
    "frame, border, filigree, scroll, castle, building, mountain, sky, moon, sun, "
    "ornament, decoration, ribbon, banner"
)

# ---- workflow（最小：ProteusV0.4 + Harrlogos_XL_v2 + prompt 主导）----
workflow = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}},
    "2": {"class_type": "LoraLoader",
          "inputs": {"model": ["1", 0],
                     "clip": ["1", 1],
                     "lora_name": "Harrlogos_XL_v2.safetensors",
                     "strength_model": 0.65,
                     "strength_clip": 0.65}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": POSITIVE, "clip": ["2", 1]}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"text": NEGATIVE, "clip": ["2", 1]}},
    "8": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "9": {"class_type": "KSampler",
          "inputs": {"seed": 1860210042,
                     "steps": 36,         # 多几步让细节更精致
                     "cfg": 8.5,          # 高 CFG 强 prompt 主导，避免徽章过装饰
                     "sampler_name": "dpmpp_2m",
                     "scheduler": "karras",
                     "denoise": 1.0,
                     "model": ["2", 0],
                     "positive": ["6", 0],
                     "negative": ["7", 0],
                     "latent_image": ["8", 0]}},
    "10": {"class_type": "VAEDecode",
           "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
    "11": {"class_type": "SaveImage",
           "inputs": {"images": ["10", 0], "filename_prefix": SAVE_PREFIX}},
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[job] {OUT_DIR}")

    client = ComfyClient()
    print(f"[submit] {len(workflow)} nodes  prompt='{POSITIVE[:60]}...'")
    prompt_id = client.queue_prompt(workflow)
    print(f"[queue] prompt_id={prompt_id}")

    # 监听产出
    outputs = ComfyClient.wait_for_images(
        COMFYUI_WS, client.client_id, prompt_id, timeout=600
    )
    print(f"[done] outputs={list(outputs.keys())}")

    # 收集 SaveImage 节点产出
    saved = []
    for nid, imgs in outputs.items():
        for i, info in enumerate(imgs):
            data = client.get_image_bytes(info["filename"], info["subfolder"], info["type"])
            out_path = OUT_DIR / f"{info['filename']}"
            out_path.write_bytes(data)
            saved.append(out_path)
            print(f"[save] {out_path} ({len(data)} bytes)")

    # 复制到 jobs/ 统一目录
    if saved:
        dst = PROJECT / "jobs" / "smoke_v205" / "v205_b_newbat.png"
        shutil.copy(saved[0], dst)
        print(f"[copy] → {dst}")
    else:
        print("[!!] 全部产出失败")


if __name__ == "__main__":
    main()
