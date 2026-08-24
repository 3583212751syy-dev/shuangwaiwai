#!/bin/bash
set -e
CV="/e/Desktop/双接口/image-fission/ComfyUI/models/clip_vision"
echo "[S1] 等待 curl 下载进程结束..."
for i in $(seq 1 180); do
  if ! tasklist 2>/dev/null | grep -q "curl.exe"; then echo "[S1] curl 已结束"; break; fi
  sleep 10
done
sleep 3
sz=$(stat -c '%s' "$CV/_vit_h_model.safetensors" 2>/dev/null || echo 0)
echo "[S1] 下载文件大小=$sz"
if [ "$sz" -lt 1000000000 ]; then echo "[S1] ERROR: 文件过小(<1GB)，下载可能失败"; exit 1; fi
# 复制为 cubiq 约定的 ViT-H 编码器文件名
cp -f "$CV/_vit_h_model.safetensors" "$CV/sd1.5model.safetensors"
cp -f "$CV/_vit_h_config.json" "$CV/sd1.5model.safetensors.config.json"
echo "[S1] 已替换 sd1.5model.safetensors 为 ViT-H(1280)"
# 验证 config
python -c "import json;d=json.load(open('$CV/sd1.5model.safetensors.config.json'));print('[S1] hidden_size=',d.get('hidden_size'))"
# 杀旧 ComfyUI (当前 PID 40644)
if tasklist 2>/dev/null | grep -q " 40644 "; then
  taskkill /PID 40644 /F 2>/dev/null && echo "[S1] 已杀旧 ComfyUI(40644)" || echo "[S1] 杀旧 ComfyUI 失败"
else
  echo "[S1] 旧 ComfyUI(40644) 未运行"
fi
echo "STEP1_DONE"
