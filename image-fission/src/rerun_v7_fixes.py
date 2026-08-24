"""v7 定向修复：重新生成 eagle_2 与 denim_3 两类，替换到已有 batch 目录并刷新画廊。"""
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_batch_6x4 import (
    COMFYUI_INPUT, ORIGINALS_CONFIG, run_one, ORIGINALS
)
from engine.comfy_client import ComfyClient

# 要修复的 batch 目录
BATCH_DIR = r"E:\Desktop\双接口\image-fission\jobs\batch_6x4_v7_1787558189"

# 需要重跑的两类
FIX_LABELS = ["eagle_2", "denim_3"]


def main():
    import demo_batch_6x4 as _m
    _m._total = 8  # 4 eagle + 4 denim

    client = ComfyClient()
    results = []

    # 构造原图文件名映射（与 demo_batch_6x4.py 一致）
    seed_map = {label: seed for seed, label in ORIGINALS}

    for orig_label in FIX_LABELS:
        cfg = ORIGINALS_CONFIG.get(orig_label)
        if not cfg:
            print(f"[WARN] 未知类别: {orig_label}")
            continue

        seed_src = os.path.join(COMFYUI_INPUT, f"{seed_map[orig_label]}.jpg")
        if not os.path.exists(seed_src):
            print(f"[WARN] 缺少原图: {seed_src}")
            continue

        for sub_label, sub_prompt in cfg["subjects"]:
            seed_name = f"fix_{orig_label}_{sub_label}_{int(time.time()*1000)}.jpg"
            shutil.copy(seed_src, os.path.join(COMFYUI_INPUT, seed_name))

            r = run_one(client, orig_label, sub_label, sub_prompt,
                        seed_name, cfg, BATCH_DIR, 0)
            if r:
                results.append(r)

    print(f"\n[DONE] 修复重跑 {len(results)}/8")
    if results:
        from make_gallery import build as build_lean
        build_lean(BATCH_DIR, os.path.join(BATCH_DIR, "gallery_lean.html"))
        print(f"[gallery] {BATCH_DIR}/gallery_lean.html 已刷新")


if __name__ == "__main__":
    main()
