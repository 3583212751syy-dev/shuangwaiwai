"""
v206_b — SDXL 单只蝙蝠（无任何装饰，原图 bat bbox 位置）

设计原则：
  - 不加画框 / 盾形 / filigree / 装饰
  - 单只 gothic bat 居中
  - 紫底（与原图色完全一致）
  - 翼展凶狠、向下俯冲、dusk gothic vintage

输出：单只蝙蝠徽章，跟原图 bat bbox 中央位置一致
"""
import os
import sys
import shutil
from pathlib import Path

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import COMFYUI_WS
from engine.comfy_client import ComfyClient

PROJECT = Path("E:/Desktop/双接口/image-fission")
JOB_ID = "smoke_v206_b"
OUT_DIR = PROJECT / "ComfyUI" / "output" / JOB_ID
SAVE_PREFIX = f"{JOB_ID}/bat"

POSITIVE = (
    "a single gothic bat centered in frame, "
    "full spread sharp angular wings, downward swoop dive pose, "
    "prominent sharp claws extended forward, "
    "detailed feather texture on wings, "
    "vintage engraving style, dusk gothic, "
    "monochrome purple duotone only, "
    "dark #2A0F40 base and #B77697 highlights, "
    "smooth solid purple gradient background"
)

NEGATIVE = (
    "low quality, blurry, deformed, watermark, text, letters, words, "
    "cute, kawaii, cartoon, disney, chibi, anime, "
    "extra fingers, mutated, jpeg artifacts, noise, "
    "frame, border, filigree, scroll, shield, banner, ribbon, "
    "sky, cloud, mountain, building, castle, sun, moon, "
    "ornament, decoration, second bat, multiple bats"
)

# workflow：ProteusV0.4 + Harrlogos_XL_v2 0.55（轻）+ cfg 7.5 + steps 32
workflow = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "ProteusV0.4.safetensors"}},
    "2": {"class_type": "LoraLoader",
          "inputs": {"model": ["1", 0],
                     "clip": ["1", 1],
                     "lora_name": "Harrlogos_XL_v2.safetensors",
                     "strength_model": 0.55,
                     "strength_clip": 0.55}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": POSITIVE, "clip": ["2", 1]}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"text": NEGATIVE, "clip": ["2", 1]}},
    "8": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "9": {"class_type": "KSampler",
          "inputs": {"seed": 1860210077,
                     "steps": 32,
                     "cfg": 7.5,
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
    prompt_id = client.queue_prompt(workflow)
    print(f"[queue] {prompt_id}")
    outputs = ComfyClient.wait_for_images(COMFYUI_WS, client.client_id, prompt_id, timeout=300)
    print(f"[done] outputs={list(outputs.keys())}")
    for nid, imgs in outputs.items():
        for info in imgs:
            data = client.get_image_bytes(info["filename"], info["subfolder"], info["type"])
            out_path = OUT_DIR / info["filename"]
            out_path.write_bytes(data)
            print(f"[save] {out_path}")
    saved = list(OUT_DIR.glob("*.png"))
    if saved:
        dst = PROJECT / "jobs" / "smoke_v206" / "v206_b_newbat.png"
        shutil.copy(sorted(saved)[-1], dst)
        print(f"[copy] → {dst}")


if __name__ == "__main__":
    main()
