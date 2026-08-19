"""
图裂变 Web 服务（MVP）：上传原图 + 4 控件滑杆 + 作业监控 + 画廊(原图vs产物整体对照)。
启动：venv 内  uvicorn orchestrator.server:app --port 8000
静态资源：src/web/gallery.html
"""
import os
import uuid
import threading
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import JOBS_DIR, DEFAULTS, BASE
from orchestrator.runner import JobRunner
from engine.comfy_client import ComfyClient

app = FastAPI(title="图裂变 image-fission")
WEB = os.path.join(BASE, "src", "web")
runner = JobRunner(ComfyClient())
_jobs = {}  # job_id -> meta（运行中存进度）


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(WEB, "gallery.html"))


@app.post("/job/start")
def start_job(original: UploadFile = File(...),
              mode: str = Form("mode1"),
              similarity: float = Form(DEFAULTS["similarity"]),
              style_strength: float = Form(DEFAULTS["style_strength"]),
              redraw_amount: float = Form(DEFAULTS["redraw_amount"]),
              style_prompt: str = Form("studio lighting, high quality product photo")):
    job_id = uuid.uuid4().hex[:12]
    src = os.path.join(JOBS_DIR, job_id, "_upload" + os.path.splitext(original.filename)[1])
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src, "wb") as f:
        shutil.copyfileobj(original.file, f)
    params = {
        "similarity": similarity, "style_strength": style_strength,
        "redraw_amount": redraw_amount, "style_prompt": style_prompt,
        "total_target": DEFAULTS["total_target"],
        "batch_per_run": DEFAULTS["batch_per_run"],
        "max_retry": DEFAULTS["max_retry"],
    }
    _jobs[job_id] = {"status": "running", "mode": mode, "params": params, "original": src}

    def _run():
        try:
            meta = runner.run_job(src, mode, params, job_id)
            _jobs[job_id] = {**_jobs[job_id], "status": "done", "meta": meta}
        except Exception as e:
            _jobs[job_id] = {**_jobs[job_id], "status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"job_id": job_id, "status": "running"})


@app.get("/job/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    return JSONResponse(_jobs[job_id])


@app.get("/gallery/{job_id}")
def gallery(job_id: str):
    d = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(d):
        raise HTTPException(404, "job not found")
    imgs = sorted([f for f in os.listdir(d) if f.endswith((".jpg", ".png"))])
    items = [f"/file/{job_id}/{f}" for f in imgs if not f.startswith("_")]
    meta_path = os.path.join(d, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        import json
        meta = json.load(open(meta_path, encoding="utf-8"))
    return JSONResponse({"job_id": job_id, "images": items, "meta": meta})


@app.get("/file/{job_id}/{name}")
def file(job_id: str, name: str):
    p = os.path.join(JOBS_DIR, job_id, name)
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


# 挂载静态目录
if os.path.isdir(WEB):
    app.mount("/static", StaticFiles(directory=WEB), name="static")
