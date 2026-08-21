#!/bin/bash
# 等待 ComfyUI 上线后运行 mode1 端到端验证
echo "[VERIFY] 等待 ComfyUI 上线 (/system_stats)..."
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8188/system_stats 2>/dev/null)
  if [ "$code" = "200" ]; then echo "[VERIFY] COMFY_READY"; break; fi
  sleep 5
done

cd /e/Desktop/双接口/image-fission/src
source /e/Desktop/双接口/image-fission/venv/Scripts/activate
echo "[VERIFY] 运行 test_mode1.py ..."
python test_mode1.py 2>&1
echo "VERIFY_DONE exit=$?"
echo "=== 产出文件 ==="
ls -la /e/Desktop/双接口/image-fission/jobs/test_mode1/ 2>/dev/null
