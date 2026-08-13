Model Worker (Enhanced)

- Endpoints:
  - `POST /infer_enhanced` - form: `image` file, `variants` int. Returns JSON with `outputs` list of `{filename, b64}`.
  - `POST /infer_batch_enhanced` - multipart form with multiple `image` files.

- Env vars:
  - `USE_REAL_MODEL` (true/false) - if true will attempt to load SDXL/ControlNet pipelines.
  - `MODEL_ID` - Hugging Face model id for SDXL.
  - `CONTROLNET_ID` - optional ControlNet model id.
  - `DEVICE` - `cuda` or `cpu`.

- To run (dev/simulated):
  - `uvicorn worker_infer_enhanced:app --host 0.0.0.0 --port 7860`

- To run in Docker (CPU):
  - `docker build -t model_worker:latest model_worker`
  - `docker run -p 7860:7860 model_worker:latest`

- Notes:
  - Real model support requires GPU, large models, and credentials for HF if private.
