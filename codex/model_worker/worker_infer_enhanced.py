"""
Simple model worker for image-fission (simulated).
Supports environment flag USE_REAL_MODEL to switch to real inference (not implemented here).
Provides HTTP endpoints for `/infer_enhanced` and `/infer_batch_enhanced`.
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import uuid
from PIL import Image, ImageFilter, ImageDraw
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

# Use local directory for Windows compatibility
WORK_DIR = os.environ.get('WORK_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'generated', 'outputs'))
os.makedirs(WORK_DIR, exist_ok=True)

def _to_float(value, default=0.6):
    if value is None:
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _theme_colors(theme: Optional[str]):
    theme = (theme or 'warm herbal').lower()
    palette = {
        'warm herbal': ((235, 214, 178), (122, 162, 110), (185, 123, 72)),
        'forest calm': ((211, 227, 217), (70, 122, 87), (169, 146, 106)),
        'luxury gold': ((241, 228, 197), (120, 101, 64), (184, 138, 66)),
        'ocean fresh': ((204, 228, 236), (62, 118, 149), (92, 102, 126)),
        'dark premium': ((228, 214, 200), (57, 65, 74), (150, 112, 71)),
        'sunset glow': ((248, 220, 180), (164, 91, 66), (211, 152, 72)),
    }
    for key, value in palette.items():
        if key in theme:
            return value
    return palette['warm herbal']


def _blend_bg_and_subject(img: Image.Image, bg_top, bg_bottom, accent, strength: float):
    w, h = img.size
    base = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(bg_top[0] * (1 - t) + bg_bottom[0] * t)
        g = int(bg_top[1] * (1 - t) + bg_bottom[1] * t)
        b = int(bg_top[2] * (1 - t) + bg_bottom[2] * t)
        for x in range(w):
            base.putpixel((x, y), (r, g, b, 255))
    # Soft vignette / theme arcs to make it feel intentionally designed
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ov = ImageDraw.Draw(overlay)
    ov.ellipse((-w * 0.1, -h * 0.15, w * 1.15, h * 1.15), fill=(*accent, int(70 * strength)))
    base = Image.alpha_composite(base, overlay)
    base = Image.alpha_composite(base, img)
    return base


def _full_redraw_like_reference(img: Image.Image, theme: Optional[str], reference_range: float):
    # "AI redraw" simulation: same object remains identifiable, but layout/background are deliberately changed.
    w, h = img.size
    bg_top, bg_bottom, accent = _theme_colors(theme)
    # apply a strong theme gradient and then compose the subject into a new composition.
    new_bg = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(bg_top[0] * (1 - t) + bg_bottom[0] * t)
        g = int(bg_top[1] * (1 - t) + bg_bottom[1] * t)
        b = int(bg_top[2] * (1 - t) + bg_bottom[2] * t)
        for x in range(w):
            new_bg.putpixel((x, y), (r, g, b, 255))

    blurred = img.resize((int(w * (0.82 + reference_range * 0.18)), int(h * (0.82 + reference_range * 0.18))))
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=1 + int((1 - reference_range) * 4)))
    # set new subject pose and position to remove original composition identity
    subject = img.copy().resize((int(w * (0.54 + (1 - reference_range) * 0.2)), int(h * (0.7 + reference_range * 0.18))))
    x = int(w * (0.35 + (1 - reference_range) * 0.15))
    y = int(h * (0.2 + reference_range * 0.1))
    new_bg.paste(subject, (x, y), subject)

    # extra mood shapes to make it feel like a different picture
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ov = ImageDraw.Draw(overlay)
    ov.ellipse((int(w * 0.64), int(h * 0.12), int(w * 0.91), int(h * 0.42)), fill=(*accent, 130))
    ov.rounded_rectangle((int(w * 0.08), int(h * 0.72), int(w * 0.92), int(h * 0.92)), radius=30, fill=(255, 255, 255, 30))
    new_bg = Image.alpha_composite(new_bg, overlay)
    return new_bg


async def simulate_fission(image_bytes: bytes, variants: int = 4, mode: str = 'theme_background', theme: Optional[str] = None, theme_strength: float = 0.7, reference_range: float = 0.6):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
    theme_strength = max(0.0, min(1.0, float(theme_strength)))
    reference_range = _to_float(reference_range, default=0.6)
    results = []

    for i in range(variants):
        out = img.copy()
        if mode == 'full_redraw':
            out = _full_redraw_like_reference(out, theme, reference_range)
        elif mode == 'theme_background':
            bg_top, bg_bottom, accent = _theme_colors(theme)
            out = _blend_bg_and_subject(out, bg_top, bg_bottom, accent, theme_strength)
        else:
            # default: preserve original composition but loosely stylize the background
            bg_top, bg_bottom, accent = _theme_colors(theme)
            out = _blend_bg_and_subject(out, bg_top, bg_bottom, accent, max(0.2, theme_strength * 0.7))
            out = out.filter(ImageFilter.GaussianBlur(radius=1 + i))
            w, h = out.size
            crop_scale = 0.92 - (reference_range * 0.1)
            out = out.crop((0, 0, int(w * crop_scale), int(h * crop_scale))).resize((w, h))

        # keep output stable for simple screenshot/demo use
        if mode == 'full_redraw':
            out = out.resize((max(1, img.width), max(1, img.height)))
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
    mode: str = Form('theme_background'),
    theme: Optional[str] = Form(None),
    theme_strength: float = Form(0.7),
    reference_range: float = Form(0.6),
    background_change: bool = Form(True),
):
    """Return generated variant images (base64 or saved paths).

    Supported modes:
      - reference_lock: keep original composition, only mild stylistic shifts
      - theme_background: strong theme-driven background change, subject stays recognizable
      - full_redraw: redraw into a completely different composition while respecting a reference range

    reference_range: 0.0 ~ 1.0, controls how far the result is allowed to drift from the source image.
    theme_strength: 0.0 ~ 1.0, controls how strongly the background is linked to the theme.
    """
    content = await image.read()
    control_bytes = None
    if control_image is not None and getattr(control_image, 'filename', None):
        control_bytes = await control_image.read()

    request_id = uuid.uuid4().hex[:8]
    mode = (mode or 'theme_background').lower()
    mode = mode if mode in {'reference_lock', 'theme_background', 'full_redraw'} else 'theme_background'
    logger.info(
        f"infer_enhanced request {request_id} variants={variants} mode={mode} "
        f"theme={theme} theme_strength={theme_strength} reference_range={reference_range} "
        f"background_change={background_change} use_real={USE_REAL_MODEL}"
    )

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
        return {'success': True, 'request_id': request_id, 'mode': mode, 'outputs': saved}

    imgs = await simulate_fission(
        content,
        variants=int(variants),
        mode=mode,
        theme=theme,
        theme_strength=float(theme_strength),
        reference_range=float(reference_range),
    )
    saved = []
    for idx, b in enumerate(imgs):
        fname = f"{request_id}_{idx}.png"
        path = os.path.join(WORK_DIR, fname)
        with open(path, 'wb') as f:
            f.write(b)
        b64 = base64.b64encode(b).decode('utf-8')
        saved.append({'path': path, 'filename': fname, 'b64': b64})

    return {'success': True, 'request_id': request_id, 'mode': mode, 'outputs': saved}

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
