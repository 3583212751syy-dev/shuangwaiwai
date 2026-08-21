"""
图裂变 image-fission —— 全局配置
所有路径均落在 E 盘（C 盘已满，仅剩 ~6.6G）。
"""
import os

# ---------- 根目录 ----------
BASE = r"E:\Desktop\双接口\image-fission"
COMFYUI_DIR = os.path.join(BASE, "ComfyUI")
VENV = os.path.join(BASE, "venv")
JOBS_DIR = os.path.join(BASE, "jobs")          # 作业输出：JPG + meta.json
MODELS_EXTRA = os.path.join(BASE, "models_extra")
SRC = os.path.join(BASE, "src")

# ---------- ComfyUI 运行 ----------
COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_API_PROMPT = COMFYUI_URL + "/prompt"
COMFYUI_API_HISTORY = COMFYUI_URL + "/history"
COMFYUI_API_VIEW = COMFYUI_URL + "/view"
COMFYUI_WS = "ws://127.0.0.1:8188/ws"
# 启动命令（在 venv 内）：python ComfyUI/main.py --cuda-device 0 --listen 127.0.0.1
COMFYUI_LAUNCH = [
    os.path.join(COMFYUI_DIR, "main.py"),
    "--cuda-device", "0",
    "--listen", "127.0.0.1",
    "--port", "8188",
    # 12G 显存保护：限队列 + 中低显存模式
    "--max-queue-size", "20",
    "--medvram",
]

# ---------- 模型路径（ComfyUI/models 下）----------
M = os.path.join(COMFYUI_DIR, "models")
CHECKPOINTS_DIR = os.path.join(M, "checkpoints")
VAE_DIR = os.path.join(M, "vae")
IPADAPTER_DIR = os.path.join(M, "ipadapter")
INSIGHTFACE_DIR = os.path.join(M, "insightface", "models")
BIREFNET_DIR = os.path.join(M, "BiRefNet")
YOLOWORLD_DIR = os.path.join(M, "ultralytics")

# 关键模型文件名（下载后请核对）
SDXL_CHECKPOINT = "sd_xl_base_1.0.safetensors"          # CreativeML OpenRAIL-M，允许商用
SDXL_REFINER = "sd_xl_refiner_1.0.safetensors"          # 可选
SDXL_VAE = "sdxl_vae.safetensors"
IPADAPTER_SDXL = "sdxl_models\\ip-adapter_sdxl_vit-h.safetensors"            # 基础版
IPADAPTER_PLUS_SDXL = "sdxl_models\\ip-adapter-plus_sdxl_vit-h.safetensors"  # plus 版(更强参考保持)
BIREFNET_MATTE = "BiRefNet-matting.safetensors"

# ---------- 默认裂变参数（MVP 控件初值）----------
DEFAULTS = {
    # 模式1：换景换风格
    "similarity": 0.6,        # IP-Adapter weight（与原图相似度滑杆 0~1）
    "style_strength": 0.7,    # 风格 LoRA / prompt 强度
    # 模式2：内容重绘
    "redraw_amount": 0.55,    # img2img denoise（重绘幅度 0~1）
    "keep_subject": True,     # 自动锁定主体（默认开）
    # 通用
    "steps": 30,
    "cfg": 7.0,
    "width": 1024,
    "height": 1024,
    "batch_per_run": 4,       # 每批 4 张
    "total_target": 400,      # 一次作业总量
    "max_retry": 3,           # 质量不达标时最多重绘次数
}

# ---------- 质量闭环阈值 ----------
QUALITY = {
    "clip_iqa_min": 0.55,     # CLIP-IQA 无参考清晰度下限
    "qalign_quality_min": 3.0,  # Q-Align 质量分(1~5)下限
    "qalign_aesthetic_min": 3.0,  # Q-Align 美学分下限
    "laplacian_blur_max": 100.0,  # Laplacian 方差上限（越大越糊）
    "allow_text_in_image": False,  # 是否允许图内文字（False=出现乱码文字即判不合格）
    "allow_extra_fingers": False,  # 是否允许手部畸变（False=检测多指即判不合格）
}

# ---------- 侵权检测 ----------
INFRINGEMENT = {
    "detect_logo": True,      # YOLO-World 检测 logo/brand/trademark
    "detect_face": True,      # InsightFace 检测真人肖像
    "detect_watermark": False,# 水印检测后置
    "logo_classes": ["logo", "brand", "trademark", "watermark"],
    "mask_dilation": 0.05,    # 侵权区域蒙版外扩比例，确保生成时不重现
}

# ---------- 输出 ----------
OUTPUT_FORMAT = "jpg"
OUTPUT_QUALITY = 92

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(MODELS_EXTRA, exist_ok=True)
