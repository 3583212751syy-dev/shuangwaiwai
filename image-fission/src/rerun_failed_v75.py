"""精准重跑 v7.5 全量中超时失败的 8 张（illust_1×2, eagle_2×4, denim_3×2）。

直接复用 demo_batch_6x4 的 run_one / 配置 / ComfyClient，只跑失败的组合，
输出到既有成功批次目录 batch_6x4_v75_1787563855（与已成功的 16 张合并），
最后在同一个目录重建 gallery，凑齐 24 张。
"""
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import demo_batch_6x4 as B

OUT_DIR = r"E:\Desktop\双接口\image-fission\jobs\batch_6x4_v75_1787563855"
os.makedirs(OUT_DIR, exist_ok=True)

# (orig_label, sub_label) 失败清单 —— 来自 full_v75b.log 的 FAIL 条目
FAILED = [
    ("illust_1", "hummingbird_vines"),
    ("illust_1", "floral_cascade"),
    ("eagle_2",  "eagle_flame"),
    ("eagle_2",  "skull_wings"),
    ("eagle_2",  "raven_flame"),
    ("eagle_2",  "winged_skull"),
    ("denim_3",  "butterfly_trail"),
    ("denim_3",  "word_butterfly"),
]

# 每个原图只需复制一次参考图到 COMFYUI input
src_by_label = {}
for orig_seed, orig_label in B.ORIGINALS:
    src_by_label[orig_label] = orig_seed

client = B.ComfyClient()
B._done = 0
B._total = len(FAILED)
print(f"[out] {OUT_DIR}")

# 准备：每个原图复制一份参考图（带新时间戳名）
seed_names = {}
for orig_label, _ in FAILED:
    if orig_label in seed_names:
        continue
    seed_src = os.path.join(B.COMFYUI_INPUT, f"{src_by_label[orig_label]}.jpg")
    seed_name = f"rerun_{orig_label}_{int(time.time()*1000)}.jpg"
    shutil.copy(seed_src, os.path.join(B.COMFYUI_INPUT, seed_name))
    seed_names[orig_label] = seed_name
    print(f"[copy] {orig_label} -> {seed_name}")

tasks = []
for orig_label, sub_label in FAILED:
    cfg = B.ORIGINALS_CONFIG[orig_label]
    sub_prompt = next(s for s, p in cfg["subjects"] if s == sub_label)
    tasks.append((orig_label, sub_label, sub_prompt, seed_names[orig_label], cfg))

results = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(B.run_one, client, ol, sl, sp, sn, cfg, OUT_DIR, i): (ol, sl)
               for i, (ol, sl, sp, sn, cfg) in enumerate(tasks)}
    for fut in as_completed(futures):
        r = fut.result()
        if r:
            results.append(r)

print(f"\n[DONE-rerun] {len(results)}/{len(FAILED)}  耗时 {time.time()-t0:.0f}s")

if results:
    try:
        B.build_gallery(results, OUT_DIR)
    except Exception as e:
        print(f"[gallery-warn] {e}")
    try:
        from make_gallery import build as build_lean
        build_lean(OUT_DIR, os.path.join(OUT_DIR, "gallery_lean.html"))
    except Exception as e:
        print(f"[gallery-lean-warn] {e}")
    print(f"[gallery] {OUT_DIR}/gallery.html  &  gallery_lean.html")
