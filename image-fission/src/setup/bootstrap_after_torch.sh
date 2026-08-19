#!/usr/bin/env bash
# torch(CUDA) 装完后执行：安装 ComfyUI 依赖(跳过 torch 三项以免覆盖 CUDA 版)、
# 应用/检测依赖、自定义节点。走代理。
set -e
export http_proxy=http://127.0.0.1:7897 https_proxy=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897
cd "$(dirname "$0")/../.."   # -> image-fission
source venv/Scripts/activate
echo "venv: $(which python)"

echo "=== 1) ComfyUI 依赖（剔除 torch/torchvision/torchaudio，保留 CUDA 版）==="
# 注意：必须用项目内相对路径，不能用 /tmp（Git Bash 的 /tmp 是 Cygwin 路径，
# 原生 Windows pip 解析不到，会报 No such file or directory）
grep -vE '^(torch|torchvision|torchaudio)$' ComfyUI/requirements.txt > _cu_req.txt
pip install -r _cu_req.txt 2>&1 | tail -15
rm -f _cu_req.txt

echo "=== 2) 应用层依赖 ==="
pip install -r src/requirements.txt 2>&1 | tail -10

echo "=== 3) 检测/质量额外依赖 ==="
pip install insightface onnxruntime-gpu ultralytics pyiqa huggingface_hub 2>&1 | tail -8

echo "=== 4) 自定义节点 ==="
bash src/setup/install_custom_nodes.sh 2>&1 | tail -20

echo "BOOTSTRAP_DONE"
