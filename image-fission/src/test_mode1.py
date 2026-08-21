"""
P1 验证：mode1（换景换风格）端到端跑通。
构造工作流 -> 提交 ComfyUI -> WS 等待 -> 下载 4 张结果 -> 存到 jobs/test_mode1/
仅用于本地验证，不进生产。
"""
import os
import sys

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import JOBS_DIR, DEFAULTS
from pipelines.build import build_mode1
from engine.comfy_client import ComfyClient


def main():
    job_id = "test_mode1"
    out_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    params = {
        "similarity": 0.6,
        "style_prompt": "a premium product photo, clean studio background, soft lighting, high detail, commercial photography",
        "negative_prompt": "low quality, blurry, deformed, watermark, text, extra fingers",
        "width": 1024,
        "height": 1024,
        "batch_per_run": 4,
        "steps": 25,
        "cfg": 6.0,
        "seed": 12345,
    }

    prompt = build_mode1("test_product.png", params, job_id)
    print(f"[1] 工作流已构造，节点数={len(prompt)}")

    client = ComfyClient()
    print("[2] 提交到 ComfyUI ...")
    try:
        result = client.run(prompt, timeout=600)
    except Exception as e:
        print("[FAIL] 提交/执行失败:", repr(e))
        return

    print(f"[3] 收到产出节点: {list(result.keys())}")
    saved = 0
    for node_id, imgs in result.items():
        # imgs 是该节点产出的多张 bytes 列表（batch）
        for i, data in enumerate(imgs):
            p = os.path.join(out_dir, f"{job_id}_node{node_id}_{i}.jpg")
            with open(p, "wb") as f:
                f.write(data)
            saved += 1
            print(f"    保存 {p} ({len(data)} bytes)")
    print(f"[DONE] mode1 验证完成，共保存 {saved} 张到 {out_dir}")


if __name__ == "__main__":
    main()
