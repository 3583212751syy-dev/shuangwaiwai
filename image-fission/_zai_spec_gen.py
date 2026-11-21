# -*- coding: utf-8 -*-
"""用拆解 spec 生成的 prompt 跑 FLUX，验证「拆解驱动生成」效果。
用法:
  python _zai_spec_gen.py --prompt outputs/_zai_spec/04/04_prompt.json --seed 601 --out <成品.jpg> [--name 04]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fz_v170_palette as V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=601)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--W", type=int, default=896)
    ap.add_argument("--H", type=int, default=1152)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    p = json.load(open(a.prompt, encoding="utf-8"))
    name = a.name or p.get("name", "spec")
    print(f"[gen] {name} seed={a.seed} pos={len(p['pos'])} neg={len(p['neg'])}", file=sys.stderr)
    im = V.generate(p["pos"], p["neg"], a.seed, a.steps, a.cfg, a.W, a.H)
    out = a.out or f"outputs/_zai_spec_gen/{name}_spec_s{a.seed}.jpg"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    im.save(out, quality=95)
    print(f"[saved] {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
