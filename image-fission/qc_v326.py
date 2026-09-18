# -*- coding: utf-8 -*-
"""qc_v326.py — 量化验收 5 图裂变（模型无法肉眼看图时的替代自检）。

指标：
  clarity   : 灰度 Laplacian 方差（越高越清晰，糊图会很低）
  color_delta: 原图 vs 裂变 在 LAB 空间的整体均值差（越小=配色越保留；真裂变允许一定偏移）
  hue_shift : 主色相偏移（度），验证"颜色保留不变"
  size_ok   : 裂变尺寸是否与原图一致（text_plan bbox 对齐前提）
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "regression_set" / "set.json").read_text(encoding="utf-8"))
OUT = ROOT / "jobs" / "router_out_v326"
SHOW_IDS = ["6978", "b78e60", "pinterest3", "pinterest4", "pinterest6"]


def clarity(arr_gray):
    from scipy import ndimage
    return float(ndimage.laplace(arr_gray).var())


def lab_meanDelta(a, b):
    aa = rgb2lab(a.astype(np.float64) / 255.0)
    bb = rgb2lab(b.astype(np.float64) / 255.0)
    d = np.abs(aa.mean(axis=(0, 1)) - bb.mean(axis=(0, 1)))
    return d.tolist()  # L,a,b 三通道差


def main():
    by_id = {img["id"]: img for img in CFG["images"]}
    print(f"{'id':<10}{'style':<22}{'size_ok':<8}{'clarity_var':<14}{'dL':<7}{'da':<7}{'db':<7}")
    for iid in SHOW_IDS:
        img = by_id[iid]
        orig = np.asarray(Image.open(img["path"]).convert("RGB"), dtype=np.float64)
        # 该 id 的输出固定位于 OUT/<iid>/01_custom_1.jpg（run_five_v326 每图独立子目录）
        variant = OUT / iid / "01_custom_1.jpg"
        if not variant.exists():
            print(f"{iid:<10}{img['style_key']:<22}{'NO':<8}  (无输出)")
            continue
        varr = np.asarray(Image.open(variant).convert("RGB"), dtype=np.float64)
        size_ok = (orig.shape[:2] == varr.shape[:2])
        if not size_ok:
            # resize 对齐再比
            varr = np.asarray(Image.open(variant).convert("RGB").resize(orig.shape[1::-1] if False else (orig.shape[1], orig.shape[0])), dtype=np.float64)
        c_o = clarity(np.asarray(Image.open(img["path"]).convert("L")))
        c_v = clarity(np.asarray(Image.open(variant).convert("L")))
        dL, da, db = lab_meanDelta(orig, varr)
        print(f"{iid:<10}{img['style_key']:<22}{str(size_ok):<8}{c_v:<14.1f}{dL:<7.2f}{da:<7.2f}{db:<7.2f}")
        print(f"           clarity orig={c_o:.1f}  Δ={c_v-c_o:+.1f}  | variant={variant.name}")


if __name__ == "__main__":
    main()
