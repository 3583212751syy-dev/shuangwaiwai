"""v8.5 illust_1 单图验证：D1 重做后看 sim 0.85 + 加细元素是否能复现原图主体。

只跑 illust_1 单图，6 主题全跑一遍对当下技术栈不划算。
"""
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_batch_6x4 import (
    COMFYUI_INPUT, JOBS_BASE, ORIGINALS, ORIGINALS_CONFIG, run_one
)
from engine.comfy_client import ComfyClient

# 只验证 illust_1 一张，看 v8.5 D1 重做后主体复现度
SUBJECT_PICK = {
    "illust_1": "peacock_floral",
}
ONLY_ORIG = "illust_1"


def main():
    out_dir = os.path.join(JOBS_BASE, f"smoke_v85_illust_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    client = ComfyClient()
    results = []

    for orig_seed, orig_label in ORIGINALS:
        if orig_label != ONLY_ORIG:
            continue
        cfg = ORIGINALS_CONFIG.get(orig_label)
        sub_label = SUBJECT_PICK[orig_label]
        sub_prompt = dict(cfg["subjects"])[sub_label]

        seed_src = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(seed_src):
            print(f"[WARN] skip {orig_seed}")
            continue
        seed_name = f"smoke_{orig_label}_{int(time.time()*1000)}.jpg"
        shutil.copy(seed_src, os.path.join(COMFYUI_INPUT, seed_name))

        r = run_one(client, orig_label, sub_label, sub_prompt, seed_name, cfg, out_dir, 0)
        if r:
            results.append(r)

    print(f"\n[DONE] {len(results)}/{len(SUBJECT_PICK)}")
    return out_dir


if __name__ == "__main__":
    main()
