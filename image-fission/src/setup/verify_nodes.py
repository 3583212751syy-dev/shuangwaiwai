"""
校验 image-fission 工作流用到的 ComfyUI 节点真实规范。
连本地 ComfyUI(/object_info)，打印我们 build.py 使用的节点 input 规范，
便于发现节点名/参数不一致，避免首次提交就 node_errors。
"""
import os
import sys
import json

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

# 依赖检查
missing = []
for m in ("requests", "websocket", "websocket_client"):
    try:
        __import__(m)
    except ImportError:
        missing.append(m)
print("依赖检查 -> requests/websocket 是否可用:", "OK" if not missing else f"缺失: {missing}")

import requests

URL = "http://127.0.0.1:8188/object_info"

# 我们 build.py 用到的节点
WANT = [
    "CheckpointLoaderSimple", "IPAdapterModelLoader", "IPAdapter",
    "LoadImage", "SaveImage", "KSampler", "VAEEncode",
    "EmptyLatentImage", "CLIPTextEncode", "VAEDecode",
]

try:
    r = requests.post(URL, json={"nodes": WANT}, timeout=30)
    info = r.json()
except Exception as e:
    print("查询失败:", e)
    sys.exit(1)

for name in WANT:
    if name not in info:
        print(f"\n[缺失] {name} —— ComfyUI 未注册此节点（可能节点未装/未加载）")
        continue
    inputs = info[name].get("input", {}).get("required", {})
    opt = info[name].get("input", {}).get("optional", {})
    print(f"\n=== {name} ===")
    for k, v in inputs.items():
        print(f"  required.{k}: {v[0] if isinstance(v, list) and v else v}")
    for k, v in opt.items():
        print(f"  optional.{k}: {v[0] if isinstance(v, list) and v else v}")

# 列出所有含 IPAdapter 的节点名，确认命名
all_nodes = requests.post(URL, json={}, timeout=30).json()
ipa = [n for n in all_nodes if "IPAdapter" in n or "ipadapter" in n.lower()]
print("\n--- 当前已注册的 IP-Adapter 相关节点 ---")
for n in ipa:
    print("  ", n)
