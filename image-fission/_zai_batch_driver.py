# -*- coding: utf-8 -*-
"""稳健分批驱动：逐图单独子进程调用 _zai_decompose_all.py --only <num>，跳过已完成，
单图超时则 kill 进程继续下一张，避免单点卡死拖垮整批。
用法: python _zai_batch_driver.py --models qwen2.5vl:7b,gemma3:12b --timeout 600
"""
import argparse, glob, os, subprocess, sys, time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "outputs", "_zai_spec")
IMGS = "D:/Users/Administrator/Desktop/双接口/outputs/masks_1_9"
NUMS = ["01","02","03","04","05","06","07","08","09","10","11"]

def done(n):
    d = os.path.join(ROOT, n)
    if not os.path.exists(os.path.join(d, "ensemble_spec.json")):
        return False
    specs = [s for s in glob.glob(os.path.join(d, "*_spec.json")) if "ensemble" not in s]
    return len(specs) >= 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5vl:7b,gemma3:12b")
    ap.add_argument("--timeout", type=int, default=600, help="单图超时秒")
    a = ap.parse_args()
    todo = [n for n in NUMS if not done(n)]
    print(f"[driver] 待处理 {todo}, 已完成 {[n for n in NUMS if done(n)]}", flush=True)
    for n in todo:
        t0 = time.time()
        cmd = [PY, os.path.join(HERE, "_zai_decompose_all.py"),
               "--imgs", IMGS, "--out-root", os.path.join(HERE, "outputs", "_zai_spec"),
               "--models", a.models, "--only", n]
        print(f"\n=== {n} START {time.strftime('%H:%M:%S')} ===", flush=True)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=a.timeout, cwd=HERE)
            tail = (r.stdout or "")[-300:] + (r.stderr or "")[-300:]
            print(f"{n} exit={r.returncode} {round(time.time()-t0)}s | {tail}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"{n} TIMEOUT {a.timeout}s → kill & continue", flush=True)
        ok = done(n)
        print(f"=== {n} {'DONE' if ok else 'INCOMPLETE'} ===", flush=True)
    print("[driver] ALL FINISHED", flush=True)

if __name__ == "__main__":
    main()
