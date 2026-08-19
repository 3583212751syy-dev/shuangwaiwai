"""
作业编排核心：对一张原图跑完整个裂变作业。
流程：侵权检测(P2钩子) -> 100批×4张生成 -> 逐张质量评分 -> 不达标自动重绘 ->
      JPG 落盘 + meta.json。
MVP 用进程内串行/受限并行（单卡 12G），不引入 Redis/Celery 以降低复杂度；
后续多机可平滑替换为 Celery worker。
"""
import os
import json
import time
from config import JOBS_DIR, DEFAULTS
from engine.comfy_client import ComfyClient
from pipelines.build import build
from quality.score import score_image
from orchestrator.io_utils import stage_input, save_result
from detection.infringement import detect as detect_infringement  # P2 启用


class JobRunner:
    def __init__(self, comfy: ComfyClient = None):
        self.comfy = comfy or ComfyClient()

    def run_job(self, original_path: str, mode: str, params: dict, job_id: str) -> dict:
        params = {**DEFAULTS, **params}
        batch_size = params["batch_per_run"]
        total = params["total_target"]
        max_retry = params["max_retry"]
        batches = max(1, total // batch_size)

        fname = stage_input(original_path, job_id)

        # —— P2 侵权检测（先做，蒙版供后续生成屏蔽；当前 build 未消费，下一步接入）——
        mask_dir = os.path.join(JOBS_DIR, job_id, "masks")
        infr = {"has_infringement": False, "regions": [], "mask": None}
        try:
            os.makedirs(mask_dir, exist_ok=True)
            infr = detect_infringement(original_path,
                                       os.path.join(mask_dir, "infringement_mask.png"),
                                       dilation_frac=params.get("mask_dilation", 0.05))
        except Exception as e:
            print(f"[warn] 侵权检测失败（不阻塞生成）: {e}")

        results = []
        passed = 0
        t0 = time.time()
        for b in range(batches):
            seed = (params.get("seed", 0) or 0) + b * 100003
            wf = build(mode, fname, {**params, "seed": seed}, job_id)
            ok_this_batch = []
            for attempt in range(max_retry + 1):
                try:
                    imgs = self.comfy.run(wf, timeout=1200)
                except Exception as e:
                    print(f"[err] batch {b} 生成异常: {e}")
                    break
                raw = imgs.get("10")  # SaveImage 节点
                if not raw:
                    print(f"[warn] batch {b} 未取到产出")
                    break
                path = save_result(raw, job_id, b, 0)
                sc = score_image(path, allow_text=params.get("allow_text_in_image", False))
                if sc["passed"]:
                    ok_this_batch.append({"path": path, "score": sc})
                    break
                else:
                    print(f"[retry] batch {b} 质量不达标 {sc['issues']}，重绘(seed+1)")
                    seed += 1
                    wf = build(mode, fname, {**params, "seed": seed}, job_id)
            if ok_this_batch:
                passed += 1
                results.append(ok_this_batch[0])
            else:
                results.append({"path": path if 'path' in dir() else None,
                                "score": sc if 'sc' in dir() else None, "passed": False})
            print(f"  批次 {b+1}/{batches} 完成，累计通过 {passed}")

        meta = {
            "job_id": job_id,
            "mode": mode,
            "original": original_path,
            "params": params,
            "infringement": infr,
            "batches": batches,
            "passed_batches": passed,
            "elapsed_sec": round(time.time() - t0, 1),
            "outputs": results,
        }
        meta_path = os.path.join(JOBS_DIR, job_id, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"作业完成：{passed}/{batches} 批通过，meta -> {meta_path}")
        return meta
