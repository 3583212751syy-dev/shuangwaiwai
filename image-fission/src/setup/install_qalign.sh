#!/usr/bin/env bash
# 安装 Q-Align 质量/美学精评（独立 venv，隔离避免与主环境 transformers 冲突）。
# 用法：bash src/setup/install_qalign.sh
set -e
export http_proxy=http://127.0.0.1:7897 https_proxy=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897
cd "$(dirname "$0")/../.."   # -> image-fission
python -m venv qalign-venv
source qalign-venv/Scripts/activate
pip install --upgrade pip
# Q-Align 官方：克隆安装 + 依赖
pip install git+https://github.com/Q-Future/Q-Align.git
pip install transformers accelerate sentencepiece timm 2>&1 | tail -5
echo "QALIGN_VENV_DONE"
