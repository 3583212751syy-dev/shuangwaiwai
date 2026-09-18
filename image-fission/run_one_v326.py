# -*- coding: utf-8 -*-
"""run_one_v326.py — 单独重跑某张图，用于迭代调试。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fission_router import route_one

CFG_PATH = ROOT / "regression_set" / "set.json"
OUT_ROOT = ROOT / "jobs" / "router_out_v326"

def main(iid: str):
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    by_id = {img["id"]: img for img in cfg["images"]}
    img = by_id[iid]
    img_out = OUT_ROOT / iid
    img_out.mkdir(parents=True, exist_ok=True)
    print(f"==== {iid} ({img['style_key']}) ====")
    res = route_one(img["path"], img_out, seed=None)
    print(f"[DONE] {iid}: variants={len(res['variants'])}")


if __name__ == "__main__":
    iid = sys.argv[1] if len(sys.argv) > 1 else "b78e60"
    main(iid)
