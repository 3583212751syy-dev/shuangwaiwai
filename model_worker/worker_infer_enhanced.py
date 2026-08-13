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

async def infer_real(image_bytes: bytes, variants: int = 4):
    pipe = await load_real_pipeline()
    if pipe is None:
        return None

    # Note: this is a minimal implementation. Real-world usage needs prompt engineering,
    # safety checking, batch handling, and device/precision tuning.
    try:
        import torch
        from PIL import Image
        from io import BytesIO

        prompt = os.environ.get('DEFAULT_PROMPT', 'best quality portrait')
        negative_prompt = os.environ.get('NEGATIVE_PROMPT', '')
        results = []
        for i in range(int(variants)):
            with torch.autocast("cuda") if torch.cuda.is_available() else nullcontext():
                out = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=20)
            img = out.images[0]
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
async def infer_enhanced(image: UploadFile = File(...), variants: int = Form(4)):
    """Return generated variant images (base64 or saved paths)."""
    content = await image.read()
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"infer_enhanced request {request_id} variants={variants} use_real={USE_REAL_MODEL}")

    if USE_REAL_MODEL:
        real = await infer_real(content, variants=int(variants))
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
async def infer_batch_enhanced(images: list[UploadFile] | None = None, variants: int = Form(4)):
    # simple batch loop
    results = []
    if images is None:
        return {'success': False, 'error': 'no images provided'}
    for im in images:
        content = await im.read()
        res = await simulate_fission(content, variants=int(variants))
        results.append({'filename': im.filename, 'variants': len(res)})
    return {'success': True, 'count': len(results), 'results': results}
