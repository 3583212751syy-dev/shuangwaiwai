#!/usr/bin/env bash
# 安装 ComfyUI 自定义节点（图片裂变 MVP 所需）
# 用法：在 image-fission/venv 激活后运行：bash src/setup/install_custom_nodes.sh
set -e
export http_proxy=http://127.0.0.1:7897 https_proxy=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897
cd "$(dirname "$0")/../.."   # -> image-fission
CUSTOM="ComfyUI/custom_nodes"
mkdir -p "$CUSTOM"
clone() {
  local repo="$1" name="$(basename "$1" .git)"
  if [ -d "$CUSTOM/$name" ]; then echo "[skip] $name 已存在"; else
    echo "==> clone $repo"; git clone --depth 1 "$repo" "$CUSTOM/$name"; fi
}
clone https://github.com/ltdrdata/ComfyUI-Manager.git
clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
clone https://github.com/1038lab/ComfyUI-RMBG.git
clone https://github.com/StevenGrove/ComfyUI-YOLOWorld.git
echo "=== 安装各节点 python 依赖 ==="
for d in ComfyUI_IPAdapter_plus ComfyUI-RMBG ComfyUI-YOLOWorld; do
  if [ -f "$CUSTOM/$d/requirements.txt" ]; then
    echo "--> $d"; pip install -r "$CUSTOM/$d/requirements.txt" || echo "[warn] $d 依赖安装有告警，见上"; fi
done
# InsightFace（侵权检测用，独立 Python 模块也需要）
pip install insightface onnxruntime-gpu ultralytics 2>&1 | tail -3 || true
echo "DONE: 自定义节点安装完成"
