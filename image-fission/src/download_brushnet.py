#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后台下载 BrushNet SDXL 权重到 ComfyUI/models/inpaint/"""
import os, sys, time, requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEST = r"E:/Desktop/双接口/image-fission/ComfyUI/models/inpaint"
os.makedirs(DEST, exist_ok=True)

URLS = {
    "diffusion_pytorch_model.safetensors":  # 放 random_mask 的文件，BrushNet loader 会自动识别
    "https://huggingface.co/hngan/brushnet_random_mask_brushnet_ckpt_sdxl_v0/resolve/main/random_mask_brushnet_ckpt_sdxl_v0/diffusion_pytorch_model.safetensors",
}

# 期望 SHA256
EXPECTED_SHA = "d968334b1e1553bbc450dd1876840732ef8726593bde51f1f79a19dc82770a55"

def download(url, dst):
    s = requests.Session()
    s.verify = False
    # 先 HEAD 拿大小
    try:
        h = s.head(url, timeout=20, allow_redirects=True)
        total = int(h.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"[HEAD] {e}", flush=True); total = 0
    # 断点续传
    already = os.path.getsize(dst) if os.path.exists(dst) else 0
    print(f"[START] {os.path.basename(dst)} expect={total/1024/1024:.0f}MB have={already/1024/1024:.1f}MB", flush=True)
    headers = {"Range": f"bytes={already}-"} if already else {}
    s2 = requests.Session(); s2.verify = False
    r = s2.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)
    r.raise_for_status()
    mode = "ab" if already else "wb"
    chunk = 1024*1024
    start = time.time()
    written = already
    with open(dst, mode) as f:
        for buf in r.iter_content(chunk_size=chunk):
            if not buf: continue
            f.write(buf); written += len(buf)
            if total and (written - already) % (16*chunk) < chunk:
                pct = written/total*100
                speed = (written - already)/(time.time()-start)/1024/1024
                eta = (total - written)/max(speed*1024*1024, 1) if speed>0 else 0
                print(f"  [{os.path.basename(dst)}] {pct:.1f}% {written/1024/1024:.0f}/{total/1024/1024:.0f}MB {speed:.1f}MB/s eta={eta:.0f}s", flush=True)
    print(f"[DONE] {dst} {os.path.getsize(dst)/1024/1024:.1f}MB", flush=True)
    # SHA256
    import hashlib
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    got = h.hexdigest()
    print(f"  sha256={got}", flush=True)
    print(f"  expected={EXPECTED_SHA}", flush=True)
    print(f"  match={got == EXPECTED_SHA}", flush=True)
    return got == EXPECTED_SHA

if __name__ == "__main__":
    fname, url = list(URLS.items())[0]
    dst = os.path.join(DEST, fname)
    ok = download(url, dst)
    sys.exit(0 if ok else 2)
