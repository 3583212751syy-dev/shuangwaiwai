"""v7 快速方向验证：每类原图跑 1 个主题，共 6 张。"""
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_batch_6x4 import (
    COMFYUI_INPUT, JOBS_BASE, ORIGINALS, ORIGINALS_CONFIG, run_one
)
from engine.comfy_client import ComfyClient

# 每类取第一个主题做方向验证
SUBJECT_PICK = {
    "illust_1": "peacock_floral",
    "eagle_2":  "eagle_flame",
    "denim_3":  "butterfly_trail",
    "camo_4":   "palm_woodland",
    "skull_5":  "skull_wing_snake",
    "metal_6":  "eagle_horned_skull",
}


def main():
    out_dir = os.path.join(JOBS_BASE, f"smoke_v84_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    client = ComfyClient()
    results = []

    for orig_seed, orig_label in ORIGINALS:
        cfg = ORIGINALS_CONFIG.get(orig_label)
        sub_label = SUBJECT_PICK.get(orig_label)
        sub_prompt = dict(cfg["subjects"]).get(sub_label)

        seed_src = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(seed_src):
            print(f"[WARN] skip {orig_seed}")
            continue
        seed_name = f"smoke_{orig_label}_{int(time.time()*1000)}.jpg"
        shutil.copy(seed_src, os.path.join(COMFYUI_INPUT, seed_name))

        r = run_one(client, orig_label, sub_label, sub_prompt, seed_name, cfg, out_dir, 0)
        if r:
            results.append(r)

    print(f"\n[DONE] {len(results)}/6")
    if results:
        from make_gallery import build as build_lean
        build_lean(out_dir, os.path.join(out_dir, "gallery_lean.html"))
        print(f"[gallery] {out_dir}/gallery_lean.html")
    return out_dir


if __name__ == "__main__":
    main()
