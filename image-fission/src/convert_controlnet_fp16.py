"""
把 ControlNet Canny SDXL 的 fp32 safetensors 转成 fp16，体积减半(~2.3GB)，
使其能在 RTX 4070 Ti 12GB 上加载（原 fp32 4.66GB + Juggernaut 7GB fp16 易 OOM）。

用法：
    python convert_controlnet_fp16.py [源路径] [目标路径]
源默认：ComfyUI/models/controlnet/controlnet-canny-sdxl-1.0.safetensors
"""
import os
import sys
import time

from safetensors import safe_open
from safetensors.torch import load_file, save_file

DEFAULT_SRC = r"E:\Desktop\双接口\image-fission\ComfyUI\models\controlnet\controlnet-canny-sdxl-1.0.safetensors"


def convert(src, dst):
    t0 = time.time()
    print(f"[fp16] 读取元数据 {os.path.basename(src)} ...")
    with safe_open(src, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    print(f"[fp16] 加载权重（fp32→内存）...")
    tensors = load_file(src, device="cpu")
    n = len(tensors)
    print(f"[fp16] 共 {n} 个张量，逐块转 fp16 ...")
    out = {}
    for i, (k, v) in enumerate(tensors.items(), 1):
        out[k] = v.half()
        if i % 50 == 0 or i == n:
            print(f"  [{i}/{n}] {k}")
    print(f"[fp16] 写出 {os.path.basename(dst)} ...")
    save_file(out, dst, metadata=meta)
    dt = time.time() - t0
    src_mb = os.path.getsize(src) / 1e6
    dst_mb = os.path.getsize(dst) / 1e6
    print(f"[fp16] 完成：{src_mb:.0f}MB -> {dst_mb:.0f}MB，耗时 {dt:.0f}s")
    print(f"[fp16] 启用方法：demo_batch_6x4.py 里 USE_CONTROLNET=True 且 CONTROLNET 指向该 fp16 文件")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".safetensors", ".fp16.safetensors")
    convert(src, dst)
