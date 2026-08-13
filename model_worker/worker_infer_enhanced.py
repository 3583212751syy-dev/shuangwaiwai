"""
Simple model worker for image-fission (simulated).
Supports environment flag USE_REAL_MODEL to switch to real inference (not implemented here).
Provides HTTP endpoints for `/infer_enhanced` and `/infer_batch_enhanced`.
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import uuid
from PIL import Image, ImageFilter
import io
import asyncio
import logging
import base64
from typing import Optional
import concurrent.futures

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model_worker')

USE_REAL_MODEL = os.environ.get('USE_REAL_MODEL', 'false').lower() == 'true'

WORK_DIR = os.environ.get('WORK_DIR', '/tmp/model_worker')
os.makedirs(WORK_DIR, exist_ok=True)

async def simulate_fission(image_bytes: bytes, variants: int = 4):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
    results = []
    for i in range(variants):
        # simple aesthetic transforms for prototype
        out = img.copy()
        out = out.filter(ImageFilter.GaussianBlur(radius=1 + i))
        w, h = out.size
        out = out.crop((0, 0, int(w*0.9), int(h*0.9))).resize((w, h))
        buf = io.BytesIO()
        out.save(buf, format='PNG')
        results.append(buf.getvalue())
    await asyncio.sleep(0.1)
    return results


# --- Real model lazy loader (skeleton) ---
_real_pipeline = None
_real_lock = asyncio.Lock()

async def load_real_pipeline():
    """Lazy-load a real SDXL/ControlNet pipeline if requested. This is a skeleton
    — heavy dependencies are imported inside this function. Returns pipeline or None."""
    global _real_pipeline
    if _real_pipeline is not None:
        return _real_pipeline
    async with _real_lock:
        if _real_pipeline is not None:
            return _real_pipeline
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline, UniPCMultistepScheduler
            from diffusers import ControlNetModel

            model_id = os.environ.get('MODEL_ID') or 'stabilityai/stable-diffusion-xl-base-1.0'
            controlnet_id = os.environ.get('CONTROLNET_ID')
            device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

            # load base pipeline
            pipe_kwargs = {}
            dtype = torch.float16 if device.startswith('cuda') else torch.float32
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, **pipe_kwargs)

            # optional ControlNet
            if controlnet_id:
                try:
                    cn = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
                    # If library supports adding controlnet, user should adapt
                    pipe.controlnet = cn
                except Exception as e:
                    logger.warning(f'ControlNet 加载失败: {e}')

            # scheduler selection
            try:
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            except Exception:
                pass

            pipe = pipe.to(device)
            _real_pipeline = pipe
            logger.info('Real pipeline loaded (skeleton)')
            return _real_pipeline
        except Exception as e:
            logger.warning(f'加载真实模型失败: {e}')
            _real_pipeline = None
            return None

async def infer_real(image_bytes: bytes, variants: int = 4, prompt: Optional[str] = None, negative_prompt: Optional[str] = None, control_bytes: Optional[bytes] = None):
    pipe = await load_real_pipeline()
    if pipe is None:
        return None

    # Note: this is a minimal implementation. Real-world usage needs prompt engineering,
    # safety checking, batch handling, and device/precision tuning.
    try:
        import torch
        from PIL import Image
        from io import BytesIO

        prompt = prompt or os.environ.get('DEFAULT_PROMPT', 'best quality portrait')
        negative_prompt = negative_prompt or os.environ.get('NEGATIVE_PROMPT', '')

        # blocking sync call wrapper for the (potentially) heavy pipeline
        def _sync_infer(pipeline, prompt, negative_prompt, steps, control_pil=None):
            try:
                if control_pil is not None and hasattr(pipeline, 'controlnet'):
                    try:
                        out = pipeline(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, image=control_pil)
                    except TypeError:
                        out = pipeline(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, control_image=control_pil)
                else:
                    out = pipeline(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps)
                return out.images
            except Exception as e:
                logger.warning(f"_sync_infer pipeline call failed: {e}")
                return None

        # prepare optional control image from direct bytes > env path > nothing
        control_pil = None
        if control_bytes:
            control_pil = Image.open(io.BytesIO(control_bytes)).convert('RGB')
        else:
            control_path = os.environ.get('CONTROL_IMAGE_PATH')
            if control_path and os.path.exists(control_path):
                control_pil = Image.open(control_path).convert('RGB')

        loop = asyncio.get_running_loop()
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for i in range(int(variants)):
                imgs = await loop.run_in_executor(pool, _sync_infer, pipe, prompt, negative_prompt, 20, control_pil)
                if imgs is None:
                    continue
                for img in imgs:
                    buf = BytesIO()
                    img.save(buf, format='PNG')
                    results.append(buf.getvalue())
        return results
    except Exception as e:
        logger.warning(f'infer_real failed: {e}')
        return None


# compatibility helper for non-cuda contexts
class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

@app.post('/infer_enhanced')
async def infer_enhanced(
    image: UploadFile = File(...),
    variants: int = Form(4),
    prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    control_image: UploadFile | None = File(default=None),
):
    """Return generated variant images (base64 or saved paths)."""
    content = await image.read()
    control_bytes = None
    if control_image is not None and getattr(control_image, 'filename', None):
        control_bytes = await control_image.read()

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"infer_enhanced request {request_id} variants={variants} use_real={USE_REAL_MODEL}")

    if USE_REAL_MODEL:
        real = await infer_real(content, variants=int(variants), prompt=prompt, negative_prompt=negative_prompt, control_bytes=control_bytes)
        if real is None:
            return JSONResponse({'success': False, 'error': 'Real model not available or inference failed'})
        saved = []
        for idx, b in enumerate(real):
            fname = f"{request_id}_{idx}.png"
            path = os.path.join(WORK_DIR, fname)
            with open(path, 'wb') as f:
                f.write(b)
            b64 = base64.b64encode(b).decode('utf-8')
            saved.append({'path': path, 'filename': fname, 'b64': b64})
        return {'success': True, 'request_id': request_id, 'outputs': saved}

    imgs = await simulate_fission(content, variants=int(variants))
    saved = []
    for idx, b in enumerate(imgs):
        fname = f"{request_id}_{idx}.png"
        path = os.path.join(WORK_DIR, fname)
        with open(path, 'wb') as f:
            f.write(b)
        b64 = base64.b64encode(b).decode('utf-8')
        saved.append({'path': path, 'filename': fname, 'b64': b64})

    return {'success': True, 'request_id': request_id, 'outputs': saved}

@app.post('/infer_batch_enhanced')
async def infer_batch_enhanced(images: list[UploadFile] | None = None, variants: int = Form(4), prompt: Optional[str] = Form(None), negative_prompt: Optional[str] = Form(None)):
    # simple batch loop
    results = []
    if images is None:
        return {'success': False, 'error': 'no images provided'}
    for im in images:
        content = await im.read()
        if USE_REAL_MODEL:
            res = await infer_real(content, variants=int(variants))
        else:
            res = await simulate_fission(content, variants=int(variants))
        results.append({'filename': im.filename, 'variants': len(res) if res else 0})
    return {'success': True, 'count': len(results), 'results': results}
