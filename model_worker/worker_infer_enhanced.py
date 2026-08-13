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

@app.post('/infer_enhanced')
async def infer_enhanced(image: UploadFile = File(...), variants: int = Form(4)):
    """Return generated variant images (base64 or saved paths)."""
    content = await image.read()
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"infer_enhanced request {request_id} variants={variants} use_real={USE_REAL_MODEL}")

    if USE_REAL_MODEL:
        # Placeholder: call real SDXL+ControlNet pipeline here
        return JSONResponse({'success': False, 'error': 'Real model not configured in this worker image'})

    imgs = await simulate_fission(content, variants=int(variants))
    saved = []
    for idx, b in enumerate(imgs):
        fname = f"{request_id}_{idx}.png"
        path = os.path.join(WORK_DIR, fname)
        with open(path, 'wb') as f:
            f.write(b)
        saved.append({'path': path, 'filename': fname})

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
