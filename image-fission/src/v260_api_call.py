"""v260 — 第一版真·SDXL 裂变（按用户给的标准 prompt 模板）

路线:
  ref image (ComfyUI input)
  -> Canny edge (ControlNet 锁构图)
  -> IPAdapter SDXL Plus (锁全图风格/颜色)
  -> VAEEncodeForInpaint (mask = bat 区 + 装饰, 让 SDXL 在此区重绘新主体)
  -> KSampler (denoise 让原图大部保留, prompt 引导主体变化)
  -> VAEDecode -> PNG

Prompt 完全采用你 (用户) 给的模板, 新主体替换为 raven/owl/falcon 三选一
合规避蝙蝠版权, 配色锁死, 反向 prompt 严守 10 条禁令
"""

import json
import time
import sys
import subprocess
from pathlib import Path
from urllib.parse import urljoin
import requests
from PIL import Image

API = "http://127.0.0.1:8188"
JOB = Path("jobs/smoke_v260")
JOB.mkdir(parents=True, exist_ok=True)
REF_NAME = "test_6978fabda2cc99629fa9e81f802762d3.jpg"   # 已存在 ComfyUI/input/
REF_PATH = Path("ComfyUI/input") / REF_NAME


SUBJECT_VARIANTS = [
    ("raven",   "raven with outspread wings, dark plumage"),
    ("owl",     "great horned owl with spread wings and stern gaze"),
    ("falcon",  "peregrine falcon gliding with wings extended"),
]


# 用户的统一 prompt 模板（你提供的原文, 严格 1:1 落地）
POSITIVE_PREFIX = (
    "referencing original composition, layout, subject position, color blocks — "
    "fully 1:1 preserve, never modify composition structure, "
    "never introduce new colors, reuse original color palette strictly. "
    "replace main subject with: "
)
POSITIVE_SUFFIX = (
    ", differentiated appearance to avoid copyright. "
    "erase all original text, in same screen-positions and matching font sizes "
    "render new english words adapted to the new subject. "
    "minor decorative elements may vary in quantity, size; "
    "background color blocks and lighting angle fully match original. "
    "vector flat 2d printed emblem, clean, symmetrical, no text residue, no gradient noise"
)

NEGATIVE = (
    "layout drift, color block change, new colors introduced, original text residue, "
    "text artifacts, font mismatch, subject position shift, image cropping, "
    "deformed subject, extra objects, color bleeding, jpeg artifacts, watermark, "
    "blurry, lowres, photorealistic, 3d render, gradient shading, "
    "modern minimalism, sans-serif typography, faded text, double exposure, "
    "out of frame, off-center composition"
)


def build_workflow(prompt_pos: str, seed: int) -> dict:
    """构造 ComfyUI workflow JSON (API 提交格式)
    节点: LoadImage -> VAEEncode (ref latent)
         ControlNetLoader -> CannyEdge -> ApplyAdvanced (构图锁)
         IPAdapterUnifiedLoader -> ApplyAdvanced (风格/色锁)
         CLIPTextEncode x2 (正/反)
         EmptyLatentImage 1024x1344 (SDXL 友好分辨率)
         KSampler (denoise=0.55, 原图退化 -> 重绘)
         VAEDecode -> SaveImage
    """
    return {
        # ---- 模型加载 ----
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "ProteusV0.4.safetensors"},
        },
        # ---- 参考图加载 ----
        "10": {
            "class_type": "LoadImage",
            "inputs": {"image": REF_NAME},
        },
        "11": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["10", 0], "vae": ["4", 2]},
        },
        # ---- Canny 边缘 (锁构图) ----
        "30": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "controlnet-canny-sdxl-1.0.safetensors"},
        },
        "31": {
            "class_type": "CannyEdgePreprocessor",
            "inputs": {
                "image": ["10", 0],
                "low_threshold": 0.08,
                "high_threshold": 0.20,
            },
        },
        "32": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "strength": 0.55,
                "start_percent": 0.0,
                "end_percent": 0.85,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "control_net": ["30", 0],
                "image": ["31", 0],
            },
        },
        # ---- IPAdapter SDXL Plus (锁全图色 + 风格) ----
        "20": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": "sdxl_models\\ip-adapter-plus_sdxl_vit-h.safetensors"},
        },
        "22": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": "_vit_h_complete\\models\\image_encoder\\model.safetensors"},
        },
        "21": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "weight": 0.65,
                "weight_type": "style transfer",
                "combine_embeds": "average",
                "start_at": 0.0,
                "end_at": 0.85,
                "embeds_scaling": "V only",
                "model": ["4", 0],
                "ipadapter": ["20", 0],
                "image": ["10", 0],
                "clip_vision": ["22", 0],
            },
        },
        # ---- 正反向 prompt ----
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt_pos, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE, "clip": ["4", 1]},
        },
        # ---- 采样 ----
        "12": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1344, "batch_size": 1},
        },
        "40": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 30,
                "cfg": 6.5,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 0.55,
                "model": ["21", 0],
                "positive": ["32", 0],
                "negative": ["32", 1],
                "latent_image": ["11", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["40", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "v260_<tag>", "images": ["8", 0]},
        },
    }


def post_workflow(prompt_id: str, workflow: dict) -> str:
    payload = {"prompt_id": prompt_id, "prompt": workflow}
    r = requests.post(urljoin(API, "/prompt"), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["prompt_id"]


def poll_done(prompt_id: str, timeout_s: int = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = requests.get(urljoin(API, "/history"), timeout=30)
        r.raise_for_status()
        h = r.json()
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("completed", False):
                return entry
            if status.get("error"):
                raise RuntimeError(f"workflow error: {status['error']}")
        time.sleep(3)
    raise TimeoutError(f"workflow {prompt_id} timeout after {timeout_s}s")


def collect_output(entry: dict) -> list:
    """从 entry.outputs 里提取 SaveImage 文件路径"""
    outs = []
    for nid, node in entry.get("outputs", {}).items():
        if "images" in node:
            for img in node["images"]:
                outs.append(Path("ComfyUI/output") / img["filename"])
    return outs


def main():
    assert REF_PATH.exists(), f"ref image not found: {REF_PATH}"
    seed_base = 42
    for i, (tag, desc) in enumerate(SUBJECT_VARIANTS):
        seed = seed_base + i * 10
        prompt_pos = POSITIVE_PREFIX + desc + POSITIVE_SUFFIX
        print(f"\n=== [{tag}] seed={seed} ===")
        print(f"prompt: {prompt_pos[:120]}...")
        wf = build_workflow(prompt_pos, seed)
        # 修正 SaveImage 的 prefix
        wf["9"]["inputs"]["filename_prefix"] = f"v260_{tag}"
        pid = post_workflow(tag + "_seed" + str(seed), wf)
        print(f"prompt_id={pid}, polling...")
        entry = poll_done(pid, timeout_s=900)
        files = collect_output(entry)
        print(f"got {len(files)} output(s)")
        for fp in files:
            print(f"  -> {fp}")
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
