"""
P1 验证：mode1（换景换风格）端到端跑通（v2：质量改进）。
构造工作流 -> 提交 ComfyUI -> 轮询 history -> 下载结果 -> 存到 jobs/test_mode1/
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

    base_params = {
        "similarity": 0.5,        # IP-Adapter 权重，降低过度还原原图扁平感
        "style_prompt": (
            "a premium product photo, clean studio background, soft lighting, "
            "high detail, commercial photography, sharp focus, 8k, ultra detailed"
        ),
        "negative_prompt": (
            "low quality, blurry, deformed, watermark, text, extra fingers, "
            "flat, illustration, vector, lowres, cartoon, posterized"
        ),
        "width": 1024,
        "height": 1024,
        "batch_per_run": 1,       # 每张单独 seed，保证 4 张差异化
        "steps": 30,              # 第一轮步数（steps_base）
        "cfg": 7.0,
        "hires_scale": 1.5,       # 潜空间放大倍数
        "hires_denoise": 0.35,    # 第二轮细化强度
        "hires_steps": 20,        # 第二轮步数
    }

    client = ComfyClient()
    saved = 0
    for i in range(4):
        params = dict(base_params)
        params["seed"] = 12345 + i * 777
        params["steps_base"] = base_params["steps"]
        prompt = build_mode1("test_product.png", params, f"{job_id}_v2_{i}")
        print(f"[1] 工作流 {i} 已构造，节点数={len(prompt)} (seed={params['seed']})")
        print("[2] 提交到 ComfyUI ...")
        try:
            result = client.run(prompt, timeout=600)
        except Exception as e:
            print(f"[FAIL] 提交/执行失败: {repr(e)}")
            continue
        print(f"[3] 收到产出节点: {list(result.keys())}")
        for node_id, imgs in result.items():
            for j, data in enumerate(imgs):
                p = os.path.join(out_dir, f"{job_id}_v2_{i}_node{node_id}_{j}.jpg")
                with open(p, "wb") as f:
                    f.write(data)
                saved += 1
                print(f"    保存 {p} ({len(data)} bytes)")
    print(f"[DONE] mode1 v2 完成，共保存 {saved} 张到 {out_dir}")


if __name__ == "__main__":
    main()
