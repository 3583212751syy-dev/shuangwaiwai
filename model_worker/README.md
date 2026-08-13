Model Worker (Enhanced)

- Endpoints:
  - `POST /infer_enhanced` - form: `image` file, optional `variants`, `prompt`, `negative_prompt`, `control_image`. Returns JSON with `outputs` list of `{filename, b64}`.
  - `POST /infer_batch_enhanced` - multipart form with multiple `image` files plus optional `prompt` and `negative_prompt`.

- Env vars:
  - `USE_REAL_MODEL` (true/false) - if true attempts to load SDXL/ControlNet pipelines.
  - `MODEL_ID` - Hugging Face model id for SDXL.
  - `CONTROLNET_ID` - optional ControlNet model id.
  - `DEVICE` - `cuda` or `cpu`.
  - `DEFAULT_PROMPT` - default generation prompt.
  - `NEGATIVE_PROMPT` - default negative prompt.
  - `CONTROL_IMAGE_PATH` - optional local file path to a conditioning image for real inference.

- To run (dev/simulated):
  - `uvicorn worker_infer_enhanced:app --host 0.0.0.0 --port 7860`

- To run in Docker (CPU):
  - `docker build -t model_worker:latest model_worker`
  - `docker run -p 7860:7860 model_worker:latest`

- To call the worker with prompt/control image:
  - `curl -F "image=@sample.png" -F "prompt=studio portrait" -F "negative_prompt=blurry" -F "control_image=@layout.png" http://localhost:7860/infer_enhanced -v`

- Notes:
  - Real model support requires GPU, large models, and credentials for Hugging Face if the model is gated.
