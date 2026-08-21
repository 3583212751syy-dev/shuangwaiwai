#!/bin/bash
# 启动 ComfyUI（适配 commit cc0fc21：动态显存，无 --medvram/--max-queue-size）
COMFY="/e/Desktop/双接口/image-fission/ComfyUI"
VENV="/e/Desktop/双接口/image-fission/venv/Scripts/python.exe"
LOG="/e/Desktop/双接口/image-fission/comfyui.log"
cd "$COMFY"
echo "[LAUNCH] 启动 ComfyUI (加载 ViT-H 编码器, 动态显存)..."
nohup "$VENV" main.py --cuda-device 0 --listen 127.0.0.1 --port 8188 > "$LOG" 2>&1 &
PID=$!
echo "[LAUNCH] ComfyUI pid=$PID"
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8188/system_stats 2>/dev/null)
  if [ "$code" = "200" ]; then echo "COMFY_UP pid=$PID"; break; fi
  sleep 5
done
echo "[LAUNCH] 节点校验:"
for n in IPAdapterUnifiedLoader IPAdapterAdvanced CheckpointLoaderSimple KSampler VAEDecode SaveImage EmptyLatentImage VAEEncode LoadImage CLIPTextEncode; do
  c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "http://127.0.0.1:8188/object_info/$n")
  echo "  $n -> $c"
done
echo "[LAUNCH] 保持运行 (tail 日志)..."
tail -f "$LOG"
