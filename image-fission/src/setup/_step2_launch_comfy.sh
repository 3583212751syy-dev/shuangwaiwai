#!/bin/bash
# 步骤2：启动 ComfyUI（持久后台任务承载），等待 /system_stats 200 后保持运行。
COMFY="/e/Desktop/双接口/image-fission/ComfyUI"
VENV="/e/Desktop/双接口/image-fission/venv/Scripts/python.exe"
LOG="/e/Desktop/双接口/image-fission/comfyui.log"

cd "$COMFY"
echo "[S2] 启动 ComfyUI ..."
nohup "$VENV" main.py --cuda-device 0 --listen 127.0.0.1 --port 8188 --max-queue-size 20 --medvram > "$LOG" 2>&1 &
echo "[S2] ComfyUI 已拉起, pid $!"

# 轮询直到可响应
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8188/system_stats 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "COMFY_UP"
    break
  fi
  sleep 5
done

# 校验关键节点已注册
echo "[S2] 校验节点注册..."
for n in IPAdapterUnifiedLoader IPAdapterAdvanced CheckpointLoaderSimple KSampler; do
  c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "http://127.0.0.1:8188/object_info/$n")
  echo "  $n -> $c"
done

# 保持任务存活，承载 ComfyUI 进程（tail 日志）
echo "[S2] ComfyUI 运行中，tail 日志保持任务..."
tail -f "$LOG"
