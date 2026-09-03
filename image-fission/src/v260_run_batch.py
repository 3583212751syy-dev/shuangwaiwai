"""v260 — 跑 3 主体 (raven/owl/falcon) 完整批
已验证 raven workflow 通, 用同样 build_workflow + 不同 prompt
"""
import sys, time, requests, json
sys.path.insert(0, 'src')
from v260_api_call import build_workflow, post_workflow, collect_output, JOB
from urllib.parse import urljoin

# 三主体 prompt (合规避蝙蝠版权)
SUBJECTS = [
    ("raven",  "raven with outspread wings and angular wingspan, dark plumage"),  # seed 42 done
    ("owl",    "great horned owl with spread wings and stern gaze"),                # seed 52
    ("falcon", "peregrine falcon gliding with sharp wings extended, predatory stance"),  # seed 62
]

# 跑 raven (seed 42) 已被刚才的 test 占用, 这里只跑 owl, falcon
SEEDS = {tag: 42 + i*10 for i, (tag, _) in enumerate(SUBJECTS)}

# 直接 inline 跑 owl 和 falcon, raven 跳过 (已生成)
RUN_THESE = ["owl", "falcon"]


def run(tag, desc, seed):
    from v260_api_call import POSITIVE_PREFIX, POSITIVE_SUFFIX
    prompt_pos = POSITIVE_PREFIX + desc + POSITIVE_SUFFIX
    wf = build_workflow(prompt_pos, seed)
    wf["9"]["inputs"]["filename_prefix"] = f"v260_{tag}"
    pid = post_workflow(f"v260_{tag}_s{seed}", wf)
    print(f"[{tag}] pid={pid}, polling...")
    t0 = time.time()
    api = "http://127.0.0.1:8188"
    while time.time()-t0 < 600:
        r = requests.get(urljoin(api, f"/history/{pid}"), timeout=10).json()
        if pid in r:
            st = r[pid].get('status', {})
            if st.get('completed'):
                print(f"[{tag}] DONE in {int(time.time()-t0)}s")
                for img in [img for nid,n in r[pid].get('outputs',{}).items() if 'images' in n for img in n['images']]:
                    print(f"  -> {img['filename']}")
                    break
                return
            if st.get('status_str') == 'error':
                print(f"[{tag}] ERR: {json.dumps(st)[:300]}")
                return
        time.sleep(5)
        if int(time.time()-t0) % 60 < 5:
            print(f"[{tag}] t={int(time.time()-t0)}s")
    print(f"[{tag}] TIMEOUT")

if __name__ == "__main__":
    for tag, desc in SUBJECTS:
        if tag in RUN_THESE:
            run(tag, desc, SEEDS[tag])
            time.sleep(2)  # gap between prompts
    print("\n=== copy outputs to jobs/smoke_v260 ===")
    import shutil
    from pathlib import Path
    out_dir = Path("ComfyUI/output")
    job_dir = Path("jobs/smoke_v260")
    job_dir.mkdir(parents=True, exist_ok=True)
    for tag in ["v260_raven", "v260_owl", "v260_falcon"]:
        for fp in out_dir.glob(f"{tag}*.png"):
            dst = job_dir / fp.name
            if not dst.exists():
                shutil.copy(fp, dst)
                print(f"copied {fp.name}")
