"""
下载 Juggernaut XL Ragnarok (v10) + RealVisXL V5.0 + epiCRealism XL
对比测试三选一专用 — 走代理直连 huggingface.co
"""
import os
import sys
import time
from huggingface_hub import hf_hub_download

PROXY = "http://127.0.0.1:7897"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ[k] = PROXY

CKPT_DIR = r"E:\Desktop\双接口\image-fission\ComfyUI\models\checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# (repo_id, filename, label)
TARGETS = sys.argv[1:] if len(sys.argv) > 1 else None
DEFAULT_TARGETS = [
    # (repo, file, label) — 这三个就是要对比的候选
    ("KandooAI/Juggernaut-XL-Ragnarok", "JuggernautXL_Ragnarok_ByRunDiffusion.safetensors", "jugg_v10"),
    ("SG161222/RealVisXL_V5.0", "RealVisXL_V5.0_fp16.safetensors", "realvisxl_v5"),
]
if TARGETS is None:
    TARGETS = []
    for i, (repo, fn, label) in enumerate(DEFAULT_TARGETS):
        if i == 0:
            TARGETS.append(repo)
        elif i == 1:
            TARGETS.append(fn)
print(f"[TARGETS] {TARGETS}")

# 用三参数调用：repo, file, label
if len(TARGETS) == 3:
    repo, fn, label = TARGETS
else:
    print("Usage: python download_3way.py <repo> <filename> <label>")
    sys.exit(1)

print(f"[DL] {label}: {repo}/{fn}")
t0 = time.time()
path = hf_hub_download(
    repo_id=repo,
    filename=fn,
    local_dir=CKPT_DIR,
    local_dir_use_symlinks=False,
)
sz = os.path.getsize(path)
print(f"[DL] DONE {label}: {path} ({sz/1024/1024:.1f} MB) in {time.time()-t0:.0f}s")
