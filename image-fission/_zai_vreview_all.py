# -*- coding: utf-8 -*-
"""批量：7 张成品全部过本地识图AI审图，结果写 outputs/_zai_vreview/ """
import json, os, sys, subprocess

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Users\Administrator\Desktop\双接口"
OUT = os.path.join(ROOT, "outputs", "_zai_vreview")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, HERE)
from _zai_vreview import REQS, review

PAIRS = [
    ("04", "A_04_route66_gold"),
    ("05", "A_05_route666"),
    ("06", "A_06_route66_rust"),
    ("07", "A_07_route66_purple"),
    ("08", "A_08_semi_truck"),
    ("10", "A_10_show_time"),
    ("11", "A_11_st_tropez"),
]

results = {}
for num, d in PAIRS:
    orig = os.path.join(ROOT, "outputs", "masks_1_9", f"_fz_in_A_{num:0>2}_{d.split('_',1)[1]}.jpg")
    if not os.path.exists(orig):
        # fallback: glob
        import glob
        cands = glob.glob(os.path.join(ROOT, "outputs", "masks_1_9", f"_fz_in_A_{num}_*.jpg"))
        orig = cands[0] if cands else None
    final = os.path.join(ROOT, "outputs", "原图裂变_FLUX_v9", d, "v24_v1_fission.jpg")
    print(f"=== {num} {d} orig={orig} final={final}", flush=True)
    if not (orig and os.path.exists(final)):
        results[num] = {"error": "missing file"}
        continue
    try:
        txt = review(orig, final, REQS.get(num, ""))
        results[num] = {"raw": txt}
        # try parse json
        try:
            j = txt[txt.index("{"): txt.rindex("}") + 1]
            results[num]["json"] = json.loads(j)
            results[num]["verdict"] = results[num]["json"].get("verdict", "?")
        except Exception as e:
            results[num]["verdict"] = "PARSE_ERR"
        with open(os.path.join(OUT, f"{num}_review.json"), "w", encoding="utf-8") as f:
            json.dump(results[num], f, ensure_ascii=False, indent=2)
    except Exception as e:
        results[num] = {"error": str(e)}
    print(json.dumps(results[num].get("verdict", "?")), flush=True)

with open(os.path.join(OUT, "all_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n==== SUMMARY ====")
for k, v in results.items():
    print(k, "->", v.get("verdict", v.get("error", "?")))
