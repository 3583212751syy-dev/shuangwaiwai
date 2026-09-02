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
    "a single gothic bat crest emblem centered in frame, "
    "wings fully spread wide in downward swoop, "
    "prominent claws extended forward, sharp angular wings with pointed tips, "
    "vector illustration style, professional badge logo, "
    "dark purple #5C2A6E base with medium purple #B77697 highlights and dark #2A0F40 shadows, "
    "isolated on a smooth purple gradient background, "
    "clean composition, sharp details, vector emblem, 8k quality, "
    "intricate line work, emblems, medallion, heraldic style, "
    "no text, no watermark, no logo, no letters, no words"
)
NEGATIVE = (
    "low quality, blurry, deformed, watermark, text, letters, words, "
    "logo, brand, badge, border, frame, signature, blurry wings, "
    "extra fingers, mutated, jpeg artifacts, noise"
)

# ---- workflow（精简版，无 IPAdapter）----
# 节点编号：
#   1 = CheckpointLoaderSimple
#   2 = LoraLoader (DD-vector-v2)
#   6 = CLIPTextEncode positive (从节点 2 模型走 CLIP)
#   7 = CLIPTextEncode negative
#   8 = EmptyLatentImage (1024x1024)
#   9 = KSampler
#   10 = VAEDecode (从节点 1 走 VAE，绕过 LoRA 干扰)
#   11 = SaveImage (前缀 smoke_v205_b/mode1)
workflow = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"}},
    "2": {"class_type": "LoraLoader",
          "inputs": {"model": ["1", 0],
                     "clip": ["1", 1],
                     "lora_name": "DD-vector-v2.safetensors",
                     "strength_model": 0.70,
                     "strength_clip": 0.70}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": POSITIVE, "clip": ["2", 1]}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"text": NEGATIVE, "clip": ["2", 1]}},
    "8": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "9": {"class_type": "KSampler",
          "inputs": {"seed": 1860210042,    # 固定种子便于复现
                     "steps": 32,
                     "cfg": 6.5,
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
