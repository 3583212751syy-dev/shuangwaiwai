import os
import gc
import signal
import sys
import time
import traceback

os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/huggingface_cache/hub'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache/transformers'
os.environ['U2NET_HOME'] = 'D:/u2net_models'

import cv2
import numpy as np
import logging
import threading
import uuid
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime

# Optional persistence imports (import failures are tolerated at runtime)
try:
    from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
except Exception:
    create_engine = None

try:
    import boto3
    from botocore.client import Config
except Exception:
    boto3 = None

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 1200

app = Flask(__name__)
CORS(app)

DOUBLE_IFACE = r'D:\Users\Administrator\Desktop\双接口'
os.makedirs(DOUBLE_IFACE, exist_ok=True)

UPLOAD_FOLDER = os.path.join(DOUBLE_IFACE, 'uploads')
OUTPUT_FOLDER = os.path.join(DOUBLE_IFACE, 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_REQUESTS = 1
current_requests = 0
request_lock = threading.Lock()
model_load_lock = threading.Lock()
server_running = True

torch = None
transformers = None
rembg = None

# Segformer B2 Clothes 标签定义 (基于LIP数据集)
# 0: Background
# 1: Hat, 2: Hair, 3: Glove, 4: Sunglasses
# 5: Upper-clothes(上装), 6: Dress(连衣裙), 7: Coat(外套)
# 8: Socks(袜子), 9: Pants(裤子), 10: Jumpsuits(连体裤)
# 11: Scarf(围巾), 12: Skirt(半身裙)
# 13: Face, 14: Left-arm, 15: Right-arm
# 16: Left-leg, 17: Right-leg, 18: Left-shoe, 19: Right-shoe

CLOTHING_CLASSES = [5, 6, 7, 9, 10, 12]  # 所有服装类
BODY_CLASSES = [2, 13, 14, 15, 16, 17]  # 身体部位
ACCESSORY_CLASSES = [1, 3, 4, 8, 11, 18, 19]  # 配件
TOP_CLASSES = [5, 7]  # 上装: Upper-clothes, Coat
BOTTOM_CLASSES = [9, 12]  # 下装: Pants, Skirt
FULL_BODY_CLASSES = [6, 10]  # 全身装: Dress, Jumpsuits

class UltraFashionSegmenter:
    def __init__(self):
        self.models_loaded = False
        self.u2net_session = None
        self.u2net_cloth_session = None
        self.u2net_human_session = None
        self.isnet_session = None
        self.segformer_processor = None
        self.segformer_model = None
        self.device = None

    def _load_dependencies(self):
        global torch, transformers, rembg
        if torch is None:
            logger.info("加载PyTorch...")
            import torch as torch_module
            torch = torch_module
            logger.info(f"PyTorch版本: {torch.__version__}")
        
        if transformers is None:
            logger.info("加载Transformers...")
            import transformers as transformers_module
            transformers = transformers_module
        
        if rembg is None:
            try:
                logger.info("加载rembg库...")
                import rembg as rembg_module
                rembg = rembg_module
                logger.info("rembg加载成功")
                try:
                    logger.info("创建ISNet session...")
                    self.isnet_session = rembg.new_session("isnet-general-use")
                    logger.info("ISNet session创建成功")
                except Exception as e:
                    logger.warning(f"ISNet加载失败: {e}")
                    self.isnet_session = None
            except Exception as e:
                logger.warning(f"rembg加载失败: {e}")
                rembg = None
                self.isnet_session = None
            
            self.u2net_session = None
            self.u2net_cloth_session = None
            self.u2net_human_session = None
            if self.isnet_session is None and rembg is not None:
                try:
                    self.isnet_session = rembg.new_session("isnet-general-use")
                    logger.info("ISNet session重新创建成功")
                except Exception as e:
                    logger.warning(f"ISNet重新创建失败: {e}")

    def _load_segformer(self):
        if self.segformer_model is None:
            logger.info("加载Segformer B2服装分割模型...")
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            try:
                self.segformer_processor = AutoImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes', local_files_only=True)
            except Exception as e:
                logger.warning(f"Image processor加载失败，使用默认配置: {e}")
                self.segformer_processor = None
            self.segformer_model = AutoModelForSemanticSegmentation.from_pretrained(
                'mattmdjaga/segformer_b2_clothes',
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.segformer_model = self.segformer_model.to(self.device)
            logger.info(f"Segformer B2加载成功，使用设备: {self.device}")

    def load_models(self):
        with model_load_lock:
            if self.models_loaded:
                return
            try:
                self._load_dependencies()
                self._load_segformer()
                self.models_loaded = True
                logger.info("=== 所有模型加载完成 ===")
            except Exception as e:
                logger.error(f"模型加载失败: {e}", exc_info=True)
                self.models_loaded = False
                raise

    def segment_with_segformer(self, image_np):
        if self.segformer_model is None:
            return None, None, None, None, None, None

        try:
            image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            if self.segformer_processor is not None:
                inputs = self.segformer_processor(images=image_pil, return_tensors="pt")
            else:
                inputs = {"pixel_values": torch.randn(1, 3, 512, 512)}
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.segformer_model(**inputs)
            
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=image_pil.size[::-1],
                mode="bilinear",
                align_corners=False
            )
            
            predicted_mask = logits.argmax(dim=1).cpu().numpy()[0]
            
            clothing_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
            body_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
            accessory_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
            top_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
            bottom_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
            full_body_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
            
            for cls in CLOTHING_CLASSES:
                clothing_mask[predicted_mask == cls] = 255
            
            for cls in BODY_CLASSES:
                body_mask[predicted_mask == cls] = 255
            
            for cls in ACCESSORY_CLASSES:
                accessory_mask[predicted_mask == cls] = 255
            
            for cls in TOP_CLASSES:
                top_mask[predicted_mask == cls] = 255
            
            for cls in BOTTOM_CLASSES:
                bottom_mask[predicted_mask == cls] = 255
            
            for cls in FULL_BODY_CLASSES:
                full_body_mask[predicted_mask == cls] = 255
            
            return clothing_mask, body_mask, accessory_mask, top_mask, bottom_mask, full_body_mask
        except Exception as e:
            logger.error(f"Segformer分割失败: {e}", exc_info=True)
            return None, None, None, None, None, None

    def _process_mask_result_fixed(self, result, h, w):
        try:
            if result is None:
                return np.zeros((h, w), dtype=np.uint8)
            
            if isinstance(result, Image.Image):
                result = np.array(result)
            
            mask = np.array(result)
            
            if mask.size == 0:
                return np.zeros((h, w), dtype=np.uint8)
            
            if len(mask.shape) == 2:
                pass
            elif len(mask.shape) == 3:
                if mask.shape[0] == 3 and mask.shape[1] == h and mask.shape[2] == w:
                    mask = np.mean(mask, axis=0).astype(np.uint8)
                elif mask.shape[0] == 3 and mask.shape[1] == w and mask.shape[2] == h:
                    mask = np.mean(mask, axis=0).astype(np.uint8)
                    mask = mask.T
                elif mask.shape[-1] == 4:
                    mask = mask[:, :, 3]
                elif mask.shape[-1] == 3:
                    mask = np.max(mask, axis=2)
                elif mask.shape[0] == 1:
                    mask = mask[0]
            
            if len(mask.shape) == 1:
                mask = np.expand_dims(mask, axis=0)
            
            mask = mask.astype(np.float32)
            if mask.max() > 0:
                mask = (mask / mask.max()) * 255
            mask = mask.astype(np.uint8)
            
            if mask.ndim == 3:
                if mask.shape[0] == 3:
                    mask = np.mean(mask, axis=0).astype(np.uint8)
            
            if mask.shape[:2] != (h, w):
                try:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                except Exception as e:
                    logger.warning(f"resize失败: {e}, 尝试备用方案")
                    temp = np.zeros((h, w), dtype=np.uint8)
                    h_min = min(mask.shape[0], h)
                    w_min = min(mask.shape[1], w)
                    if mask.ndim == 2:
                        temp[:h_min, :w_min] = mask[:h_min, :w_min]
                    mask = temp
            
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            return mask
        except Exception as e:
            logger.warning(f"处理蒙版失败: {e}", exc_info=True)
            return np.zeros((h, w), dtype=np.uint8)

    def _mask_ratio(self, mask):
        if mask is None or mask.size == 0:
            return 0.0
        return (np.sum(mask > 0) / mask.size) * 100

    def remove_small_regions(self, mask, min_area=300):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(result, [contour], 0, 255, -1)
        return result

    def get_largest_connected_component(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        
        result = np.zeros_like(mask)
        if max_contour is not None:
            cv2.drawContours(result, [max_contour], 0, 255, -1)
        return result

    def fill_holes_color_aware(self, mask, image_np, max_hole_area=50000):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                x, y, w, h = cv2.boundingRect(contour)
                hole_area = cv2.contourArea(contour)
                
                if hole_area < max_hole_area:
                    mask_copy = mask.copy()
                    cv2.drawContours(mask_copy, [contour], 0, 0, -1)
                    
                    surrounding_pixels = []
                    search_radius = 15
                    cx, cy = x + w // 2, y + h // 2
                    for dx in range(-search_radius, search_radius + 1):
                        for dy in range(-search_radius, search_radius + 1):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0]:
                                if mask_copy[ny, nx] > 0:
                                    surrounding_pixels.append(image_np[ny, nx])
                    
                    if surrounding_pixels:
                        avg_color = np.mean(surrounding_pixels, axis=0)
                        hole_mask = np.zeros_like(mask)
                        cv2.drawContours(hole_mask, [contour], 0, 255, -1)
                        
                        hole_pixels = image_np[hole_mask > 0]
                        if len(hole_pixels) > 0:
                            hole_avg = np.mean(hole_pixels, axis=0)
                            color_diff = np.linalg.norm(avg_color - hole_avg)
                            
                            if color_diff < 180:
                                cv2.drawContours(mask, [contour], 0, 255, -1)

        return mask

    def detect_accessories_strict(self, image_np, mask):
        h, w = mask.shape[:2]
        accessory_mask = np.zeros_like(mask)
        
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        
        gold_lower = np.array([15, 150, 150])
        gold_upper = np.array([25, 255, 255])
        gold_mask = cv2.inRange(hsv, gold_lower, gold_upper)
        
        silver_lower = np.array([0, 0, 200])
        silver_upper = np.array([180, 20, 255])
        silver_mask = cv2.inRange(hsv, silver_lower, silver_upper)
        
        metal_mask = cv2.bitwise_or(gold_mask, silver_mask)
        
        combined_accessory = cv2.bitwise_and(metal_mask, mask)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined_accessory = cv2.morphologyEx(combined_accessory, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        combined_accessory = cv2.morphologyEx(combined_accessory, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        contours, _ = cv2.findContours(combined_accessory, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        neck_region = np.zeros_like(mask)
        neck_region[0:int(h * 0.25), :] = 255
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w_cnt, h_cnt = cv2.boundingRect(contour)
            aspect_ratio = max(w_cnt, h_cnt) / min(w_cnt, h_cnt) if min(w_cnt, h_cnt) > 0 else 1
            
            if area < 100 or area > 800:
                continue
            
            center_x = x + w_cnt // 2
            center_y = y + h_cnt // 2
            
            is_accessory = False
            
            if y < h * 0.25:
                if (aspect_ratio > 8 and w_cnt > w * 0.4) or (aspect_ratio > 3 and area < 200):
                    if neck_region[center_y, center_x] > 0:
                        is_accessory = True
            elif aspect_ratio > 20:
                if area < 300:
                    is_accessory = True
            
            if is_accessory:
                cv2.drawContours(accessory_mask, [contour], 0, 255, -1)
        
        return accessory_mask

    def detect_backpack_straps(self, image_np, mask):
        h, w = mask.shape[:2]
        strap_mask = np.zeros_like(mask)
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        
        line_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)
        line_mask = cv2.bitwise_and(line_mask, mask)
        
        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        side_margin = w // 8
        leg_start = int(h * 0.45)
        leg_end = int(h * 0.9)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w_cnt, h_cnt = cv2.boundingRect(contour)
            aspect_ratio = max(w_cnt, h_cnt) / min(w_cnt, h_cnt) if min(w_cnt, h_cnt) > 0 else 1
            
            if area < 50:
                continue
            
            is_strap = False
            
            if x < side_margin or x + w_cnt > w - side_margin:
                if h_cnt > w * 0.2 and w_cnt < w * 0.03:
                    is_strap = True
            elif y >= leg_start and y <= leg_end:
                if (x < w * 0.15 or x > w * 0.85):
                    if aspect_ratio > 10 and w_cnt < w * 0.04:
                        is_strap = True
            
            if is_strap:
                cv2.drawContours(strap_mask, [contour], 0, 255, -1)
                padding = 3
                cv2.rectangle(strap_mask, 
                            (max(0, x-padding), max(0, y-padding)), 
                            (min(w-1, x+w_cnt+padding), min(h-1, y+h_cnt+padding)), 
                            255, -1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        strap_mask = cv2.morphologyEx(strap_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return strap_mask

    def detect_skin_enhanced(self, image_np):
        try:
            ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
            rgb = image_np
            
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            
            white_cloth_mask = cv2.inRange(gray, 201, 264)
            kernel_w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            white_cloth_mask = cv2.morphologyEx(white_cloth_mask, cv2.MORPH_CLOSE, kernel_w, iterations=3)
            
            lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
            upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
            
            lower_hsv = np.array([0, 15, 30], dtype=np.uint8)
            upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
            
            lower_hsv2 = np.array([170, 15, 30], dtype=np.uint8)
            upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
            
            skin_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            skin_hsv1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
            skin_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
            skin_hsv = cv2.bitwise_or(skin_hsv1, skin_hsv2)
            
            r, g, b = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
            skin_rgb = ((r > 95) & (g > 40) & (b > 20) & 
                        (r > g) & (r > b) & 
                        (abs(r - g) > 15) & 
                        (r > 100) & (g > 80) & (b > 60)).astype(np.uint8) * 255
            
            skin_mask = cv2.bitwise_or(skin_ycrcb, skin_hsv)
            skin_mask = cv2.bitwise_or(skin_mask, skin_rgb)
            
            high_light = gray > 220
            skin_mask[high_light] = 0
            
            low_light = gray < 40
            skin_mask[low_light] = 0
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:
                    cv2.drawContours(skin_mask, [contour], 0, 0, -1)
            
            skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(white_cloth_mask))
            
            return skin_mask
        except Exception as e:
            logger.warning(f"皮肤检测失败: {e}")
            try:
                return np.zeros(image_np.shape[:2], dtype=np.uint8)
            except:
                return None

    def detect_skin_saturated(self, image_np):
        """S型高精度皮肤检测：使用更窄的HSV范围，精准检测皮肤而避免误判白色"""
        try:
            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
            
            # 窄范围皮肤检测：只检测明显的皮肤颜色
            lower1 = np.array([0, 20, 40], dtype=np.uint8)
            upper1 = np.array([25, 180, 230], dtype=np.uint8)
            lower2 = np.array([160, 20, 40], dtype=np.uint8)
            upper2 = np.array([180, 180, 230], dtype=np.uint8)
            
            skin1 = cv2.inRange(hsv, lower1, upper1)
            skin2 = cv2.inRange(hsv, lower2, upper2)
            skin_mask = cv2.bitwise_or(skin1, skin2)
            
            # 去除高亮区域（白色/浅色衣服）
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            high_brightness = gray > 210
            skin_mask[high_brightness] = 0
            
            # 去除低饱和度区域
            h, s, v = cv2.split(hsv)
            low_saturation = s < 30
            skin_mask[low_saturation] = 0
            
            # 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 去除小噪点
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:
                    cv2.drawContours(skin_mask, [contour], 0, 0, -1)
            
            return skin_mask
        except Exception as e:
            logger.warning(f"S型皮肤检测失败: {e}")
            return None

    def find_waist_by_contour_narrowing(self, mask):
        """
        通过分析服装轮廓宽度变化找到腰部位置
        腰部通常是服装宽度最窄的区域
        """
        h, w = mask.shape[:2]
        
        # 计算每行的有效像素数（宽度）
        row_counts = np.zeros(h, dtype=np.int32)
        for y in range(h):
            row_counts[y] = np.sum(mask[y, :] > 0)
        
        # 找到服装的垂直范围
        non_zero_rows = np.where(row_counts > 0)[0]
        if len(non_zero_rows) < 50:
            return None
        
        top_y = non_zero_rows[0]
        bottom_y = non_zero_rows[-1]
        clothing_height = bottom_y - top_y
        
        # 只考虑中间40%-80%区域（腰部通常在此范围）
        search_start = int(top_y + clothing_height * 0.35)
        search_end = int(top_y + clothing_height * 0.75)
        search_start = max(search_start, top_y + 20)
        search_end = min(search_end, bottom_y - 20)
        
        if search_end <= search_start:
            return None
        
        # 平滑处理
        smoothed = cv2.GaussianBlur(row_counts.astype(np.float32).reshape(-1, 1), (15, 1), 3).flatten()
        
        # 在搜索区域内找到宽度最小值（腰部）
        search_region = smoothed[search_start:search_end]
        if len(search_region) == 0:
            return None
        
        min_idx = np.argmin(search_region)
        waist_y = search_start + min_idx
        
        # 确保腰部位置合理
        waist_ratio = (waist_y - top_y) / clothing_height if clothing_height > 0 else 0.5
        if waist_ratio < 0.3 or waist_ratio > 0.8:
            return None
        
        return waist_y

    def recover_clothing_after_skin_removal(self, original_mask, after_skin_mask, image_np, seg_clothing, mask_cloth):
        """
        皮肤狂删后恢复被误删的服装区域
        
        策略：
        1. 找出被删除的区域 = original - after_skin
        2. 对每个被删除的连通组件，检查是否为服装
        3. 如果是服装（与已知服装蒙版重叠大或颜色匹配），则恢复
        """
        h, w = original_mask.shape[:2]
        result = after_skin_mask.copy()
        
        # 找出被删除的区域
        removed = cv2.bitwise_and(original_mask, cv2.bitwise_not(after_skin_mask))
        
        if np.sum(removed) == 0:
            return result
        
        # 膨胀已知服装蒙版作为参考
        clothing_ref = np.zeros((h, w), dtype=np.uint8)
        if seg_clothing is not None:
            clothing_ref = cv2.bitwise_or(clothing_ref, seg_clothing)
        if mask_cloth is not None:
            clothing_ref = cv2.bitwise_or(clothing_ref, mask_cloth)
        clothing_ref = cv2.dilate(clothing_ref, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
        
        # 连通组件分析被删除的区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(removed, 8)
        
        for i in range(1, num_labels):
            comp_mask = (labels == i).astype(np.uint8) * 255
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < 100:
                continue
            
            # 检查该组件是否与已知服装蒙版重叠
            overlap = cv2.bitwise_and(comp_mask, clothing_ref)
            overlap_ratio = np.sum(overlap) / (255 * max(area, 1))
            
            # 如果与服装蒙版重叠超过20%，很可能是被误删的衣服
            if overlap_ratio > 0.2:
                result = cv2.bitwise_or(result, comp_mask)
                continue
            
            # 检查颜色：如果是高饱和度（皮肤）就不恢复
            comp_indices = np.where(comp_mask > 0)
            comp_colors = image_np[comp_indices]
            if len(comp_colors) > 0:
                comp_hsv = cv2.cvtColor(comp_colors.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
                mean_s = np.mean(comp_hsv[:, :, 1])
                # 低饱和度 = 可能是白色/黑色衣服，恢复
                if mean_s < 40:
                    result = cv2.bitwise_or(result, comp_mask)
        
        return result

    def _recover_components(self, removed_mask, clothing_ref, image_np, min_overlap=0.15):
        """恢复被误删的服装组件（保守）"""
        result = np.zeros_like(removed_mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(removed_mask, 8)
        for i in range(1, num_labels):
            comp = (labels == i).astype(np.uint8) * 255
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                continue
            overlap = cv2.bitwise_and(comp, clothing_ref)
            overlap_ratio = np.sum(overlap) / (255 * max(area, 1))
            if overlap_ratio > min_overlap:
                result = cv2.bitwise_or(result, comp)
        return result

    def _recover_below_waist_components(self, removed_mask, image_np, skin_mask, waist_y):
        """恢复腰部以下被误删的组件（非常保守，几乎不恢复）"""
        result = np.zeros_like(removed_mask)
        h, w = removed_mask.shape[:2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(removed_mask, 8)
        for i in range(1, num_labels):
            comp = (labels == i).astype(np.uint8) * 255
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                continue
            
            top_of_comp = stats[i, cv2.CC_STAT_TOP]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            
            # 条件1：组件顶部必须在腰线50像素以内（连接裤子）
            if top_of_comp > waist_y + 50:
                continue
            
            # 条件2：不能是长条形（腿的特征）
            aspect = comp_h / max(comp_w, 1)
            if aspect > 3.0:
                continue
            
            # 条件3：颜色检查 - 高饱和度=皮肤，低饱和度=衣服
            comp_indices = np.where(comp > 0)
            global_y = comp_indices[0]
            global_x = comp_indices[1]
            comp_colors = image_np[global_y, global_x]
            
            if len(comp_colors) > 0:
                comp_hsv = cv2.cvtColor(comp_colors.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
                mean_s = np.mean(comp_hsv[:, :, 1])
                # 高饱和度=皮肤色，不恢复
                if mean_s > 50:
                    continue
            
            result = cv2.bitwise_or(result, comp)
        return result

    def _find_waist_position(self, mask):
        """快速找到服装的腰部分割位置"""
        h, w = mask.shape[:2]
        # 计算每行的服装像素数
        row_counts = np.zeros(h, dtype=np.int32)
        for y in range(h):
            row_counts[y] = np.sum(mask[y, :] > 0)
        
        non_zero = np.where(row_counts > 0)[0]
        if len(non_zero) < 50:
            return None
        
        top_y, bottom_y = non_zero[0], non_zero[-1]
        clothing_height = bottom_y - top_y
        # 腰部在服装高度的55%-70%处
        return int(top_y + clothing_height * 0.63)

    def force_clean_below_waist(self, mask, waist_y, image_np, skin_mask=None):
        """
        强力清除腰部以下区域中的大腿/腿部残留
        策略：
        1. 对腰部以下区域做连通组件分析
        2. 移除皮肤色的组件（使用皮肤蒙版验证）
        3. 移除长条形组件（like legs）
        4. 只保留紧凑、非皮肤色的组件
        """
        h, w = mask.shape[:2]
        result = mask.copy()
        
        # 只处理腰部以下的区域
        above = mask[:waist_y, :].copy()
        below = mask[waist_y:, :].copy()
        
        if np.sum(below) == 0:
            return result
        
        # 膨胀皮肤蒙版（如果在腰部以下）
        below_skin = np.zeros_like(below)
        if skin_mask is not None:
            below_skin_raw = skin_mask[waist_y:, :]
            below_skin = cv2.dilate(below_skin_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
        
        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(below, 8)
        
        cleaned_below = np.zeros_like(below)
        
        for i in range(1, num_labels):
            comp_mask = (labels == i).astype(np.uint8) * 255
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < 200:
                continue
            
            # 检查该组件是否主要是皮肤色
            skin_overlap = np.sum(cv2.bitwise_and(comp_mask, below_skin)) / 255
            skin_ratio = skin_overlap / max(area, 1)
            
            # 如果该组件超过50%是皮肤色，移除它
            if skin_ratio > 0.5:
                continue
            
            # 检查长宽比（长条形 = 可能是腿）
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            aspect_ratio = comp_h / max(comp_w, 1)
            
            # 长条形且在图像下半部分 = 很可能是腿
            if aspect_ratio > 4.0:
                continue
            
            # 获取该组件的颜色
            comp_indices = np.where(comp_mask > 0)
            if len(comp_indices[0]) > 0:
                # 转换为全局坐标
                global_y = comp_indices[0] + waist_y
                global_x = comp_indices[1]
                comp_colors = image_np[global_y, global_x]
                comp_hsv = cv2.cvtColor(comp_colors.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
                mean_s = np.mean(comp_hsv[:, :, 1])
                mean_v = np.mean(comp_hsv[:, :, 2])
                
                # 皮肤色判断：中等饱和度 + 中高亮度
                if 20 < mean_s < 100 and mean_v > 80:
                    # 可能是皮肤，进一步检查
                    skin_pixels = comp_colors[skin_mask[global_y, global_x] > 0] if skin_mask is not None else np.array([])
                    if len(skin_pixels) > len(comp_colors) * 0.3:
                        continue
            
            cleaned_below[comp_mask > 0] = 255
        
        # 合并上下部分
        result[waist_y:, :] = cleaned_below
        return result

    def separate_top_bottom_optimized(self, mask, top_mask=None, bottom_mask=None, full_body_mask=None):
        h, w = mask.shape[:2]
        
        # 检查是否穿着全身服装（连衣裙/连体裤）
        # 关键判断：真正的连衣裙不会同时有单独的裤子/裙子标签
        use_full_body_strategy = False
        if full_body_mask is not None:
            full_body_ratio = self._mask_ratio(full_body_mask)
            
            # 检查是否同时检测到下装（裤子/裙子）
            has_bottom = bottom_mask is not None and self._mask_ratio(bottom_mask) > 0.5
            
            # 如果同时有全身装和下装，说明不是真正的连衣裙，而是上衣+裤子的组合
            if full_body_ratio > 3.0 and not has_bottom:
                logger.info(f"检测到全身服装占比{full_body_ratio:.1f}%（无独立下装），使用特殊分割策略")
                use_full_body_strategy = True
            elif full_body_ratio > 3.0 and has_bottom:
                logger.info(f"检测到全身服装占比{full_body_ratio:.1f}%，但同时检测到下装，视为上衣+裤子组合")
        
        if use_full_body_strategy:
                # 对于全身服装，分割线设在腰部（约60%处）
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    all_points = []
                    for contour in contours:
                        for point in contour:
                            all_points.append((point[0][0], point[0][1]))
                    if all_points:
                        ys = [y for _, y in all_points]
                        min_y, max_y = min(ys), max(ys)
                        mid_y = int(min_y + (max_y - min_y) * 0.60)
                        
                        top_result = mask.copy()
                        top_result[mid_y:, :] = 0
                        bottom_result = mask.copy()
                        bottom_result[:mid_y, :] = 0
                        
                        top_result = self.get_largest_connected_component(top_result)
                        bottom_result = self.get_largest_connected_component(bottom_result)
                        return top_result, bottom_result
        
        # 首先尝试使用Segformer的精确分割结果
        if top_mask is not None and bottom_mask is not None:
            top_ratio = self._mask_ratio(top_mask)
            bottom_ratio = self._mask_ratio(bottom_mask)
            
            # 只有当Segformer同时检测到上衣和下装时才使用它的结果
            # 如果只检测到其中一个，说明Segformer不可靠，应该回退到智能分割
            if top_ratio > 2.0 and bottom_ratio > 2.0:
                top_result = cv2.bitwise_and(mask, top_mask)
                bottom_result = cv2.bitwise_and(mask, bottom_mask)
                
                top_result = self.get_largest_connected_component(top_result)
                bottom_result = self.get_largest_connected_component(bottom_result)
                
                top_ratio_final = self._mask_ratio(top_result)
                bottom_ratio_final = self._mask_ratio(bottom_result)
                
                if top_ratio_final > 1.0 and bottom_ratio_final > 1.0:
                    logger.info(f"使用Segformer分割: 上衣{top_ratio_final:.1f}% 下装{bottom_ratio_final:.1f}%")
                    return top_result, bottom_result
            elif top_ratio > 0.5 or bottom_ratio > 0.5:
                # Segformer只检测到了一部分，不可靠，跳过使用
                logger.info(f"Segformer部分检测失败（上衣{top_ratio:.1f}% 下装{bottom_ratio:.1f}%），回退到智能分割")
        
        # 使用基于轮廓缩窄的智能分割
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return np.zeros_like(mask), np.zeros_like(mask)
        
        # 获取所有轮廓点
        all_points = []
        for contour in contours:
            for point in contour:
                all_points.append((point[0][0], point[0][1]))
        
        if not all_points:
            return np.zeros_like(mask), np.zeros_like(mask)
        
        ys = [y for _, y in all_points]
        xs = [x for x, _ in all_points]
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)
        body_height = max_y - min_y
        body_width = max_x - min_x
        
        # 策略1: 尝试通过轮廓缩窄找到腰部
        waist_y = self.find_waist_by_contour_narrowing(mask)
        
        if waist_y is not None:
            logger.info(f"通过轮廓缩窄找到腰部位置: y={waist_y}")
        else:
            # 策略2: 使用基于人体比例的固定分割
            # 正常人体：头部10-15%，躯干35-40%，腿部45-50%
            # 从服装顶部算起：上衣约占60-65%
            
            # 计算垂直方向的像素分布，找到服装主体区域
            y_hist = np.zeros(h, dtype=np.int32)
            for y in range(h):
                y_hist[y] = np.sum(mask[y, :] > 0)
            
            non_zero_y = np.where(y_hist > 0)[0]
            if len(non_zero_y) > 0:
                start_y, end_y = non_zero_y[0], non_zero_y[-1]
                clothing_height = end_y - start_y
                
                # 对于上衣+裤子的组合，分割线约在服装高度的62-65%处
                # 这样上衣包含躯干部分，下装从腰部开始
                waist_y = int(start_y + clothing_height * 0.63)
                
                # 确保分割线在合理范围内
                waist_y = max(start_y + int(clothing_height * 0.45), 
                             min(waist_y, end_y - int(clothing_height * 0.25)))
            else:
                waist_y = int(min_y + body_height * 0.60)
        
        logger.info(f"智能分割: 腰部y={waist_y}, 图像高度={h}, 服装范围={min_y}-{max_y}")
        
        # 创建分割线（稍微平滑过渡）
        # 在分割线附近创建一个过渡区域，避免硬切割
        transition = 5  # 过渡像素数
        
        top_result = np.zeros_like(mask)
        bottom_result = np.zeros_like(mask)
        
        # 上衣：分割线以上 + 过渡区域的一半
        top_end = min(waist_y + transition, h)
        top_result[:top_end, :] = mask[:top_end, :]
        
        # 下装：分割线以下 + 过渡区域的一半  
        bottom_start = max(waist_y - transition, 0)
        bottom_result[bottom_start:, :] = mask[bottom_start:, :]
        
        # 后处理：确保上衣和下装都有合理的内容
        top_result = self.get_largest_connected_component(top_result)
        bottom_result = self.get_largest_connected_component(bottom_result)
        
        top_ratio = self._mask_ratio(top_result)
        bottom_ratio = self._mask_ratio(bottom_result)
        
        # 如果分割结果严重不平衡，尝试自适应调整
        if top_ratio < 3.0 and bottom_ratio > 8.0:
            # 上衣太少，分割线太靠上，向下调整
            new_waist_y = int(waist_y + body_height * 0.15)
            new_waist_y = min(new_waist_y, max_y - 30)
            top_result = mask.copy()
            top_result[new_waist_y:, :] = 0
            top_result = self.get_largest_connected_component(top_result)
            bottom_result = mask.copy()
            bottom_result[:new_waist_y, :] = 0
            bottom_result = self.get_largest_connected_component(bottom_result)
            logger.info(f"自适应调整分割线到 y={new_waist_y}")
        elif bottom_ratio < 3.0 and top_ratio > 8.0:
            # 下装太少，分割线太靠下，向上调整
            new_waist_y = int(waist_y - body_height * 0.15)
            new_waist_y = max(new_waist_y, min_y + 30)
            top_result = mask.copy()
            top_result[new_waist_y:, :] = 0
            top_result = self.get_largest_connected_component(top_result)
            bottom_result = mask.copy()
            bottom_result[:new_waist_y, :] = 0
            bottom_result = self.get_largest_connected_component(bottom_result)
            logger.info(f"自适应调整分割线到 y={new_waist_y}")
        
        return top_result, bottom_result

    def refine_contour_with_gradient(self, mask, image_np):
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = cv2.magnitude(gradient_x, gradient_y)
            gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            result = np.zeros_like(mask)
            
            for contour in contours:
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                refined_points = []
                for i in range(len(approx)):
                    x, y = approx[i][0]
                    
                    if 0 <= y < gradient_mag.shape[0] and 0 <= x < gradient_mag.shape[1]:
                        grad_val = gradient_mag[y, x]
                        
                        if i > 0 and i < len(approx) - 1:
                            prev_x, prev_y = approx[i-1][0]
                            next_x, next_y = approx[i+1][0]
                            
                            dx = next_x - prev_x
                            dy = next_y - prev_y
                            length = max(np.sqrt(dx*dx + dy*dy), 1)
                            nx = dx / length
                            ny = dy / length
                            
                            if grad_val > 200:
                                offset = -1
                            elif grad_val > 120:
                                offset = 0
                            else:
                                offset = 1
                            
                            new_x = x + int(nx * offset)
                            new_y = y + int(ny * offset)
                            
                            new_x = max(0, min(new_x, mask.shape[1]-1))
                            new_y = max(0, min(new_y, mask.shape[0]-1))
                            
                            refined_points.append([[new_x, new_y]])
                        else:
                            refined_points.append([[x, y]])
                    else:
                        refined_points.append([[x, y]])
                
                if refined_points:
                    refined_contour = np.array(refined_points)
                    cv2.drawContours(result, [refined_contour], 0, 255, -1)
            
            return result
        except Exception as e:
            logger.warning(f"轮廓细化失败，使用原始蒙版: {e}")
            return mask

    def optimize_edge_quality(self, image_np, mask):
        try:
            # 使用高斯模糊创建平滑的alpha过渡
            alpha = mask.copy().astype(np.float32) / 255.0
            
            # 多尺度边缘平滑
            # 小半径用于精细边缘，大半径用于平滑区域
            alpha_smooth_small = cv2.GaussianBlur(alpha, (3, 3), 0.8)
            alpha_smooth_medium = cv2.GaussianBlur(alpha, (5, 5), 1.5)
            alpha_smooth_large = cv2.GaussianBlur(alpha, (7, 7), 2.5)
            
            # 检测边缘区域
            edge = cv2.Canny(mask, 50, 150)
            edge_dilated = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            
            # 根据距离边缘的远近使用不同的平滑程度
            distance = cv2.distanceTransform(cv2.bitwise_not(edge_dilated), cv2.DIST_L2, 5)
            distance = distance / (distance.max() + 1e-5)
            
            # 混合三种平滑程度：边缘附近用小半径，远离边缘用大半径
            alpha_smooth = (
                alpha_smooth_small * (1 - distance) * 0.6 +
                alpha_smooth_medium * 0.3 +
                alpha_smooth_large * distance * 0.1
            )
            
            # 确保主要区域保持不透明
            core_region = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
            alpha_smooth = np.maximum(alpha_smooth, core_region.astype(np.float32) / 255.0 * 0.95)
            
            # 转回0-255范围
            result = (alpha_smooth * 255).clip(0, 255).astype(np.uint8)
            
            return result
        except Exception as e:
            logger.warning(f"边缘优化失败，使用原始蒙版: {e}")
            return mask

    def smooth_contour_precision(self, mask, image_np):
        """高精度轮廓平滑 - 使用图像引导的轮廓优化"""
        try:
            h, w = mask.shape[:2]
            
            # 找到主要轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return mask
            
            # 获取最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 计算图像梯度
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gradient = cv2.magnitude(grad_x, grad_y)
            
            # 平滑轮廓点
            epsilon = 0.0005 * cv2.arcLength(max_contour, True)  # 更精细的近似
            smooth_contour = cv2.approxPolyDP(max_contour, epsilon, True)
            
            # 创建平滑后的掩码
            result = np.zeros_like(mask)
            cv2.drawContours(result, [smooth_contour], 0, 255, -1)
            
            # 使用高斯模糊进行边缘抗锯齿
            result_float = result.astype(np.float32) / 255.0
            result_blurred = cv2.GaussianBlur(result_float, (5, 5), 1.0)
            
            # 使用图像梯度作为权重，在强边缘处保持清晰
            gradient_norm = gradient / (gradient.max() + 1e-5)
            edge_weight = np.clip(gradient_norm * 2, 0, 1)  # 强边缘权重高
            
            # 混合：强边缘处使用清晰轮廓，弱边缘处使用平滑轮廓
            result_final = result_float * edge_weight + result_blurred * (1 - edge_weight)
            result_final = (result_final * 255).clip(0, 255).astype(np.uint8)
            
            # 确保主体区域不透明
            result_final = cv2.max(result_final, cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1))
            
            return result_final
        except Exception as e:
            logger.warning(f"轮廓平滑失败: {e}")
            return mask

    def skin_detection_combined(self, img_bgr):
        """HSV+YCbCr联合皮肤检测 - 更准确的肤色识别"""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 10, 50], dtype=np.uint8)
        upper1 = np.array([30, 220, 255], dtype=np.uint8)
        mask_hsv1 = cv2.inRange(hsv, lower1, upper1)
        lower2 = np.array([155, 10, 50], dtype=np.uint8)
        upper2 = np.array([180, 220, 255], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv, lower2, upper2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0, 125, 70], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        return skin_mask

    def fill_holes(self, mask):
        """填充蒙版中的孔洞"""
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        mask_u = (mask > 128).astype(np.uint8)
        H, W = mask_u.shape
        mask_border = np.zeros((H+2, W+2), np.uint8)
        mask_border[1:H+1, 1:W+1] = mask_u
        for x in range(W):
            if mask_border[1, x+1] == 0:
                cv2.floodFill(mask_border, None, (x, 0), 255)
            if mask_border[H, x+1] == 0:
                cv2.floodFill(mask_border, None, (x, H-1), 255)
        for y in range(H):
            if mask_border[y+1, 1] == 0:
                cv2.floodFill(mask_border, None, (0, y), 255)
            if mask_border[y+1, W] == 0:
                cv2.floodFill(mask_border, None, (W-1, y), 255)
        holes = 255 - mask_border[1:H+1, 1:W+1]
        result = cv2.bitwise_or(mask_u * 255, holes)
        return result

    def clean_mask(self, mask, min_area_ratio=0.005):
        """清理蒙版，去除小区域"""
        contours, _ = cv2.findContours((mask>128).astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas)
        H, W = mask.shape
        clean = np.zeros((H, W), dtype=np.uint8)
        for c, a in zip(contours, areas):
            if a >= max(W*H*min_area_ratio, 100) and a >= max_area * 0.05:
                cv2.drawContours(clean, [c], -1, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        return clean

    def get_largest_cc(self, mask, min_area=500):
        """获取最大连通组件"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 128).astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return mask
        max_area, best_label = 0, 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area and area >= min_area:
                max_area = area
                best_label = i
        return (labels == best_label).astype(np.uint8) * 255 if best_label > 0 else mask

    def smooth_contour(self, mask, smoothness=5):
        """平滑轮廓"""
        contours, _ = cv2.findContours((mask>128).astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        largest = max(contours, key=cv2.contourArea)
        epsilon = smoothness * cv2.arcLength(largest, True) / 1000
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) < 10:
            approx = largest
        hull = cv2.convexHull(approx)
        H, W = mask.shape
        result = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(result, [hull], -1, 255, -1)
        return result

    def repair_top_mask_smart(self, top_mask):
        """智能修复上衣蒙版"""
        H, W = top_mask.shape
        rows = np.any(top_mask > 128, axis=1)
        if not np.any(rows):
            return top_mask
        y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        h = y2 - y1
        filled = self.fill_holes(top_mask)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
        smoothed = self.get_largest_cc(smoothed, min_area=2000)
        smoothed = self.clean_mask(smoothed, min_area_ratio=0.008)
        return smoothed

    def find_split_y_precise(self, clothing_mask, skin_mask):
        """精确的腰部定位算法"""
        H, W = clothing_mask.shape
        rows = np.any(clothing_mask > 128, axis=1)
        if not np.any(rows):
            return H // 2
        y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        height = y2 - y1
        widths = []
        y_pos = []
        for y in range(y1, y2):
            cols = np.where(clothing_mask[y, :] > 128)[0]
            if len(cols) > 0:
                widths.append(cols[-1] - cols[0])
                y_pos.append(y)
        if len(widths) < 20:
            return y1 + height // 2
        widths_smooth = np.convolve(widths, np.ones(21)/21, mode='same')
        n = len(widths_smooth)
        search_start = int(n * 0.40)
        search_end = int(n * 0.65)
        min_w = float('inf')
        min_idx = search_start
        for i in range(search_start, search_end):
            if widths_smooth[i] < min_w and widths_smooth[i] > 50:
                min_w = widths_smooth[i]
                min_idx = i
        hip_idx = min_idx
        hip_w = min_w
        for i in range(min_idx + 5, min(min_idx + 200, n)):
            if widths_smooth[i] > hip_w:
                hip_w = widths_smooth[i]
                hip_idx = i
            if widths_smooth[i] < hip_w * 0.75 and i > min_idx + 80:
                break
        hip_y = y_pos[hip_idx]
        pants_top_y = y_pos[min_idx] + max(15, int((hip_y - y_pos[min_idx]) * 0.80))
        if skin_mask is not None:
            skin_rows = np.any(skin_mask > 128, axis=1)
            if np.any(skin_rows):
                sy_top = int(np.where(skin_rows)[0][0])
                sy_bottom = int(np.where(skin_rows)[0][-1])
                skin_h = sy_bottom - sy_top
                if skin_h > 50 and sy_top > y1 + int(height * 0.35):
                    if pants_top_y > sy_top + 10:
                        pants_top_y = sy_top + 3
        if pants_top_y < y1 + int(height * 0.55):
            pants_top_y = y1 + int(height * 0.62)
        if pants_top_y > y1 + int(height * 0.80):
            pants_top_y = y1 + int(height * 0.68)
        return pants_top_y

    def extract_pants_refined(self, bottom_mask, skin_mask, split_y):
        """精细的裤子提取算法"""
        H, W = bottom_mask.shape
        rows = np.any(bottom_mask > 128, axis=1)
        if not np.any(rows):
            return bottom_mask
        y1 = int(np.where(rows)[0][0])
        pure_clothing = cv2.bitwise_and(bottom_mask, 255 - skin_mask) if skin_mask is not None else bottom_mask.copy()
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        pure_clothing = cv2.morphologyEx(pure_clothing, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (pure_clothing > 128).astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                (bottom_mask > 128).astype(np.uint8), connectivity=8)
            if num_labels <= 1:
                return bottom_mask
        valid = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            top_y = stats[i, cv2.CC_STAT_TOP]
            if area > 800 and top_y < y1 + 100:
                valid.append(i)
        if not valid:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 500:
                    valid.append(i)
        max_area, best_label = 0, 0
        for i in valid:
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                best_label = i
        if best_label == 0:
            return bottom_mask
        pants_cc = (labels == best_label).astype(np.uint8) * 255
        pants_rows = np.any(pants_cc > 128, axis=1)
        if np.any(pants_rows):
            py1, py2 = int(np.where(pants_rows)[0][0]), int(np.where(pants_rows)[0][-1])
            ph = py2 - py1
            band_h = max(int(ph * 0.12), 30)
            if py2 - band_h > py1:
                bottom_band = pants_cc[py2 - band_h:py2, :]
                skin_band = skin_mask[py2 - band_h:py2, :] if skin_mask is not None else np.zeros_like(bottom_band)
                overlap = cv2.bitwise_and(bottom_band, skin_band)
                if np.sum(overlap > 128) > 80:
                    cut = py2 - int(ph * 0.08)
                    if cut > py1:
                        pants_cc[cut:, :] = 0
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.bitwise_and(bottom_mask, cv2.dilate(pants_cc, kernel_dilate))
        result = self.clean_mask(combined, min_area_ratio=0.008)
        result = self.fill_holes(result)
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)
        result = self.clean_mask(result, min_area_ratio=0.008)
        return result

    def remove_attachments(self, mask, image_np):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
        
        result = mask.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max_area * 0.005:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                
                if aspect_ratio > 12 or area < 100:
                    cv2.drawContours(result, [contour], 0, 0, -1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return result

    def refine_mask_with_edges_conservative(self, mask, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 120, 200)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        expanded = cv2.dilate(mask, kernel, iterations=1)
        
        edge_mask = cv2.bitwise_and(edges_dilated, expanded)
        
        result = cv2.bitwise_or(mask, edge_mask)
        
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_area = max(cv2.contourArea(c) for c in contours)
            filtered = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) >= max_area * 0.1:
                    cv2.drawContours(filtered, [contour], 0, 255, -1)
            result = filtered
        
        return result

    def ensure_clothing_completeness(self, mask, image_np):
        h, w = mask.shape[:2]
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        result = mask.copy()
        
        for contour in contours:
            x, y, w_cnt, h_cnt = cv2.boundingRect(contour)
            
            padding = 3
            search_x1 = max(0, x - padding)
            search_x2 = min(w, x + w_cnt + padding)
            search_y1 = max(0, y - padding)
            search_y2 = min(h, y + h_cnt + padding)
            
            search_region = image_np[search_y1:search_y2, search_x1:search_x2]
            search_mask = mask[search_y1:search_y2, search_x1:search_x2]
            
            if search_region.size == 0:
                continue
            
            edge = cv2.Canny(cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY), 50, 150)
            edge = cv2.bitwise_and(edge, cv2.bitwise_not(search_mask))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            edge_dilated = cv2.dilate(edge, kernel, iterations=1)
            
            edge_contours, _ = cv2.findContours(edge_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for edge_contour in edge_contours:
                area = cv2.contourArea(edge_contour)
                if area < 50 and area > 5:
                    ecx, ecy, ecw, ech = cv2.boundingRect(edge_contour)
                    if ecw < 10 and ech < 10:
                        result[search_y1 + ecy:search_y1 + ecy + ech, 
                               search_x1 + ecx:search_x1 + ecx + ecw] = 255
        
        return result

    def process_image(self, input_path, output_path, model_type="Best", return_layers=False, clothing_type="all"):
        try:
            logger.info(f"=== 开始终极服装精准抠图 === (类型: {clothing_type})")
            
            logger.info(f"  - 尝试读取图片: {input_path}")
            logger.info(f"  - 文件是否存在: {os.path.exists(input_path)}")
            if os.path.exists(input_path):
                logger.info(f"  - 文件大小: {os.path.getsize(input_path)} bytes")
            
            try:
                image = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                logger.info(f"  - cv2.imdecode结果: {image is not None}")
            except Exception as e:
                image = None
                logger.error(f"  - cv2.imdecode失败: {e}")
            
            if image is None:
                try:
                    image = cv2.imread(input_path)
                    logger.info(f"  - cv2.imread结果: {image is not None}")
                except Exception as e:
                    image = None
                    logger.error(f"  - cv2.imread失败: {e}")
            
            if image is None:
                logger.error(f"无法读取图片: {input_path}")
                return (False, "无法读取图片", None, None) if return_layers else (False, "无法读取图片")
            
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            image_np = image.copy()
            h, w = image_np.shape[:2]
            
            # 如果图像过大，缩小处理以节省内存
            MAX_DIM = 700
            scale = 1.0
            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = image_np.shape[:2]
                logger.info(f"  - 图像已缩放至: {w}x{h} (scale={scale:.2f})")
            
            logger.info(f"  - 图像尺寸: {w}x{h}")
            logger.info(f"  - 服装类型: {clothing_type}")
            
            if not self.models_loaded:
                self.load_models()
            
            all_masks = []
            seg_clothing = None
            seg_body = None
            seg_accessory = None
            seg_top = None
            seg_bottom = None
            seg_full_body_mask = None
            fg_alpha = None
            
            logger.info("【1】ISNet + Alpha Matting精细抠图...")
            if self.isnet_session and rembg is not None:
                try:
                    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                    image_small = image_pil.resize((min(700, w), min(700, h)), Image.LANCZOS)
                    try:
                        result = rembg.remove(image_small, session=self.isnet_session,
                                               alpha_matting=True,
                                               alpha_matting_foreground_threshold=240,
                                               alpha_matting_background_threshold=10,
                                               alpha_matting_erode_size=10)
                    except Exception:
                        result = rembg.remove(image_small, session=self.isnet_session, alpha_matting=False)
                    
                    fg_alpha = cv2.resize(np.array(result.getchannel('A')), (w, h), interpolation=cv2.INTER_CUBIC)
                    fg_mask = self.clean_mask((fg_alpha > 128).astype(np.uint8) * 255)
                    fg_mask = self.fill_holes(fg_mask)
                    all_masks.append(fg_mask)
                    logger.info(f"  - ISNet前景占比: {self._mask_ratio(fg_mask):.1f}%")
                except Exception as e:
                    logger.warning(f"ISNet失败: {e}")
            
            logger.info("【2】Segformer B2精细服装分割...")
            seg_clothing, seg_body, seg_accessory, seg_top, seg_bottom, seg_full_body = self.segment_with_segformer(image_np)
            seg_full_body_mask = seg_full_body
            if seg_clothing is not None:
                logger.info(f"  - Segformer服装占比: {self._mask_ratio(seg_clothing):.1f}%")
                logger.info(f"  - Segformer身体占比: {self._mask_ratio(seg_body):.1f}%")
                logger.info(f"  - Segformer配饰占比: {self._mask_ratio(seg_accessory):.1f}%")
                logger.info(f"  - Segformer上衣占比: {self._mask_ratio(seg_top):.1f}%")
                logger.info(f"  - Segformer下装占比: {self._mask_ratio(seg_bottom):.1f}%")
                if seg_full_body is not None:
                    logger.info(f"  - Segformer全身装占比: {self._mask_ratio(seg_full_body):.1f}%")
                all_masks.append(seg_clothing)
            
            logger.info("【3】多模型蒙版融合...")
            if len(all_masks) == 0:
                return (False, "所有模型都失败了", None, None) if return_layers else (False, "所有模型都失败了")
            
            combined_mask = all_masks[0].copy()
            for i, mask in enumerate(all_masks[1:], 1):
                combined_mask = cv2.bitwise_or(combined_mask, mask)
                logger.info(f"  - 融合第{i+1}个模型后占比: {self._mask_ratio(combined_mask):.1f}%")
            
            clothing_only = combined_mask.copy()
            
            logger.info("【4】HSV+YCbCr联合皮肤检测...")
            skin_mask = self.skin_detection_combined(image_np)
            if skin_mask is not None:
                skin_mask = cv2.bitwise_and(skin_mask, combined_mask)
                logger.info(f"  - 皮肤占比: {self._mask_ratio(skin_mask):.1f}%")
            
            logger.info("【5】去除身体区域...")
            clothing_only = self.clean_mask(cv2.bitwise_and(clothing_only, 255 - skin_mask))
            clothing_only = self.fill_holes(clothing_only)
            clothing_only = self.get_largest_cc(clothing_only, min_area=5000)
            clothing_only = self.clean_mask(clothing_only)
            clothing_only = self.fill_holes(clothing_only)
            logger.info(f"  - 去除身体后占比: {self._mask_ratio(clothing_only):.1f}%")
            
            logger.info("【6】去除Segformer识别的配饰区域...")
            if seg_accessory is not None:
                accessory_dilated = cv2.dilate(seg_accessory, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                clothing_protect = np.zeros((h, w), dtype=np.uint8)
                if seg_clothing is not None:
                    clothing_protect = cv2.bitwise_or(clothing_protect, seg_clothing)
                if clothing_protect.any():
                    accessory_dilated = cv2.bitwise_and(accessory_dilated, cv2.bitwise_not(clothing_protect))
                clothing_only = cv2.bitwise_and(clothing_only, cv2.bitwise_not(accessory_dilated))
            logger.info(f"  - 去除配饰后占比: {self._mask_ratio(clothing_only):.1f}%")
            
            logger.info("【7】精确腰部定位与上下装分离...")
            # 保存分割结果用于分图层输出，但不替换主蒙版
            top_mask_for_layers, bottom_mask_for_layers = self.separate_top_bottom_optimized(clothing_only, seg_top, seg_bottom, seg_full_body_mask)
            logger.info(f"  - 上衣占比: {self._mask_ratio(top_mask_for_layers):.1f}%")
            logger.info(f"  - 下装占比: {self._mask_ratio(bottom_mask_for_layers):.1f}%")
            
            # 使用精确腰部定位算法
            split_y = self.find_split_y_precise(clothing_only, skin_mask)
            logger.info(f"  - 精确分割位置: y={split_y}")
            
            # 保存分割结果用于分图层输出
            saved_top_mask = top_mask_for_layers if self._mask_ratio(top_mask_for_layers) > 1.0 else None
            saved_bottom_mask = bottom_mask_for_layers if self._mask_ratio(bottom_mask_for_layers) > 1.0 else None
            
            # 【8】精细裤子提取（去除腿部残留）
            bottom_mask_refined = self.extract_pants_refined(bottom_mask_for_layers, skin_mask, split_y)
            if self._mask_ratio(bottom_mask_refined) > 0.5:
                saved_bottom_mask = bottom_mask_refined
            logger.info(f"  - 精细裤子提取后占比: {self._mask_ratio(saved_bottom_mask):.1f}%" if saved_bottom_mask is not None else "  - 未检测到裤子")
            
            # 【9】智能上衣修复
            if saved_top_mask is not None:
                saved_top_mask = self.repair_top_mask_smart(saved_top_mask)
            logger.info(f"  - 上衣修复后占比: {self._mask_ratio(saved_top_mask):.1f}%" if saved_top_mask is not None else "  - 未检测到上衣")
            
            logger.info("【10】形态学修复...")
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clothing_only = cv2.morphologyEx(clothing_only, cv2.MORPH_CLOSE, kernel_small, iterations=5)
            clothing_only = cv2.morphologyEx(clothing_only, cv2.MORPH_OPEN, kernel_small, iterations=2)
            logger.info(f"  - 修复后占比: {self._mask_ratio(clothing_only):.1f}%")
            
            logger.info("【11】孔洞修复...")
            clothing_only = self.fill_holes(clothing_only)
            logger.info(f"  - 孔洞修复后占比: {self._mask_ratio(clothing_only):.1f}%")
            
            logger.info("【12】去除小区域...")
            clothing_only = self.clean_mask(clothing_only, min_area_ratio=0.008)
            logger.info(f"  - 去小区域后占比: {self._mask_ratio(clothing_only):.1f}%")
            
            logger.info("【13】边缘质量优化...")
            if fg_alpha is not None:
                alpha = cv2.GaussianBlur(cv2.bitwise_and(fg_alpha, clothing_only).astype(np.float32), (3,3), 0.7)
                alpha = np.clip(alpha, 0, 255).astype(np.uint8)
            else:
                alpha = self.optimize_edge_quality(image_np, clothing_only)
            
            logger.info("【14】生成透明PNG...")
            result = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
            result[:, :, 3] = alpha
            
            cv2.imwrite(output_path, result)
            
            logger.info(f"=== 最终蒙版占比: {self._mask_ratio(alpha):.1f}% ===")
            logger.info(f"=== 处理完成: {output_path} ===")
            
            # 如果请求分图层，则处理上衣和下装
            top_output_path = None
            bottom_output_path = None
            
            if return_layers:
                logger.info(f"【15】生成分图层结果... (类型: {clothing_type})")
                
                if saved_top_mask is not None and saved_bottom_mask is not None:
                    top_mask_final = saved_top_mask
                    bottom_mask_final = saved_bottom_mask
                    logger.info("  - 使用优化后的分割结果")
                else:
                    top_mask_final, bottom_mask_final = self.separate_top_bottom_optimized(clothing_only, seg_top, seg_bottom, seg_full_body_mask)
                    logger.info("  - 重新分割")
                
                # 根据服装类型生成不同的输出
                if clothing_type == "top":
                    if self._mask_ratio(top_mask_final) > 0.5:
                        if fg_alpha is not None:
                            top_alpha = cv2.GaussianBlur(cv2.bitwise_and(fg_alpha, top_mask_final).astype(np.float32), (3,3), 0.7)
                            top_alpha = np.clip(top_alpha, 0, 255).astype(np.uint8)
                        else:
                            top_alpha = self.optimize_edge_quality(image_np, top_mask_final)
                        top_result = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
                        top_result[:, :, 3] = top_alpha
                        cv2.imwrite(output_path, top_result)
                        logger.info(f"  - 仅上衣已保存为主输出")
                        top_output_path = output_path
                    else:
                        logger.warning("  - 未检测到上衣，使用完整蒙版")
                        result = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
                        result[:, :, 3] = alpha
                        cv2.imwrite(output_path, result)
                        
                elif clothing_type == "bottom":
                    if self._mask_ratio(bottom_mask_final) > 0.5:
                        if fg_alpha is not None:
                            bottom_alpha = cv2.GaussianBlur(cv2.bitwise_and(fg_alpha, bottom_mask_final).astype(np.float32), (3,3), 0.7)
                            bottom_alpha = np.clip(bottom_alpha, 0, 255).astype(np.uint8)
                        else:
                            bottom_alpha = self.optimize_edge_quality(image_np, bottom_mask_final)
                        bottom_result = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
                        bottom_result[:, :, 3] = bottom_alpha
                        cv2.imwrite(output_path, bottom_result)
                        logger.info(f"  - 仅下装已保存为主输出")
                        bottom_output_path = output_path
                    else:
                        logger.warning("  - 未检测到下装，使用完整蒙版")
                        result = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
                        result[:, :, 3] = alpha
                        cv2.imwrite(output_path, result)
                        
                else:  # "all"
                    if self._mask_ratio(top_mask_final) > 0.5:
                        if fg_alpha is not None:
                            top_alpha = cv2.GaussianBlur(cv2.bitwise_and(fg_alpha, top_mask_final).astype(np.float32), (3,3), 0.7)
                            top_alpha = np.clip(top_alpha, 0, 255).astype(np.uint8)
                        else:
                            top_alpha = self.optimize_edge_quality(image_np, top_mask_final)
                        top_result = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
                        top_result[:, :, 3] = top_alpha
                        top_output_path = output_path.replace('.png', '_top.png')
                        cv2.imwrite(top_output_path, top_result)
                        logger.info(f"  - 上衣已保存: {top_output_path}")
                    
                    if self._mask_ratio(bottom_mask_final) > 0.5:
                        if fg_alpha is not None:
                            bottom_alpha = cv2.GaussianBlur(cv2.bitwise_and(fg_alpha, bottom_mask_final).astype(np.float32), (3,3), 0.7)
                            bottom_alpha = np.clip(bottom_alpha, 0, 255).astype(np.uint8)
                        else:
                            bottom_alpha = self.optimize_edge_quality(image_np, bottom_mask_final)
                        bottom_result = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
                        bottom_result[:, :, 3] = bottom_alpha
                        bottom_output_path = output_path.replace('.png', '_bottom.png')
                        cv2.imwrite(bottom_output_path, bottom_result)
                        logger.info(f"  - 下装已保存: {bottom_output_path}")
            
            if return_layers:
                gc.collect()
                return True, None, top_output_path, bottom_output_path
            gc.collect()
            return True, None
            
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            gc.collect()
            if return_layers:
                return False, str(e), None, None
            return False, str(e)

segmenter = UltraFashionSegmenter()

# --- Persistence setup (Postgres / MinIO) ---
POSTGRES_URL = os.environ.get('POSTGRES_URL')
S3_ENDPOINT = os.environ.get('MINIO_ENDPOINT') or os.environ.get('S3_ENDPOINT')
S3_BUCKET = os.environ.get('S3_BUCKET')
S3_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY') or os.environ.get('S3_ACCESS_KEY')
S3_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY') or os.environ.get('S3_SECRET_KEY')

DB_ENGINE = None
DB_Session = None
Base = None

if create_engine is not None:
    try:
        if POSTGRES_URL:
            DB_ENGINE = create_engine(POSTGRES_URL, echo=False, future=True)
        else:
            # fallback to sqlite file in repo
            sqlite_path = os.path.join(os.path.dirname(__file__), 'data.db')
            DB_ENGINE = create_engine(f'sqlite:///{sqlite_path}', echo=False, connect_args={"check_same_thread": False})

        from sqlalchemy.orm import declarative_base
        Base = declarative_base()

        class Task(Base):
            __tablename__ = 'tasks'
            id = Column(Integer, primary_key=True)
            request_id = Column(String(64), index=True)
            input_path = Column(Text)
            output_path = Column(Text)
            s3_input_url = Column(Text)
            s3_output_url = Column(Text)
            qa_result = Column(Text)
            created_at = Column(DateTime, default=datetime.utcnow)
            completed_at = Column(DateTime, nullable=True)

        Base.metadata.create_all(DB_ENGINE)
        DB_Session = sessionmaker(bind=DB_ENGINE)
        logger.info("数据库（Task）初始化完成")
    except Exception as e:
        logger.warning(f"数据库初始化失败: {e}")
        DB_ENGINE = None

# MinIO/S3 client
S3_CLIENT = None
if boto3 is not None and S3_ENDPOINT and S3_BUCKET and S3_ACCESS_KEY and S3_SECRET_KEY:
    try:
        s3_config = Config(signature_version='s3v4')
        S3_CLIENT = boto3.client('s3', endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY, config=s3_config)
        logger.info("S3/MinIO 客户端初始化完成")
    except Exception as e:
        logger.warning(f"S3 客户端初始化失败: {e}")

def upload_file_to_s3(local_path, key):
    """Upload a local file to configured S3/MinIO bucket and return an accessible URL."""
    if S3_CLIENT is None:
        return None
    try:
        S3_CLIENT.upload_file(local_path, S3_BUCKET, key)
        # Construct URL (path-style). Users may need to adapt to their MinIO setup.
        url = f"{S3_ENDPOINT.rstrip('/')}/{S3_BUCKET}/{key}"
        return url
    except Exception as e:
        logger.warning(f"上传到 S3 失败: {e}")
        return None

def create_task_record(request_id, input_path):
    if DB_Session is None:
        return None
    try:
        session = DB_Session()
        t = Task(request_id=request_id, input_path=input_path, created_at=datetime.utcnow())
        session.add(t)
        session.commit()
        session.refresh(t)
        session.close()
        return t.id
    except Exception as e:
        logger.warning(f"创建任务记录失败: {e}")
        return None

def update_task_record(request_id, **kwargs):
    if DB_Session is None:
        return
    try:
        session = DB_Session()
        t = session.query(Task).filter(Task.request_id == request_id).first()
        if not t:
            session.close()
            return
        for k, v in kwargs.items():
            if hasattr(t, k):
                setattr(t, k, v)
        if 'completed_at' in kwargs and kwargs['completed_at'] is None:
            t.completed_at = None
        session.commit()
        session.close()
    except Exception as e:
        logger.warning(f"更新任务记录失败: {e}")

@app.route('/')
def index():
    return jsonify({'status': 'ok', 'message': '服装抠图API服务', 'endpoints': ['POST /api/segment', 'GET /api/health', 'GET /api/model-status']})

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': segmenter.models_loaded,
        'current_requests': current_requests,
        'server_running': server_running
    })

@app.route('/api/model-status')
def model_status():
    logger.info(f"isnet_session type: {type(segmenter.isnet_session)}, value: {segmenter.isnet_session}")
    return jsonify({
        'loaded': segmenter.models_loaded,
        'models': {
            'u2net': segmenter.u2net_session is not None,
            'u2net_cloth_seg': segmenter.u2net_cloth_session is not None,
            'u2net_human_seg': segmenter.u2net_human_session is not None,
            'isnet-general-use': segmenter.isnet_session is not None,
            'segformer': segmenter.segformer_model is not None
        }
    })

@app.route('/api/load-model', methods=['POST'])
def load_model():
    try:
        segmenter.load_models()
        return jsonify({'success': True, 'message': '模型加载成功'})
    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/segment', methods=['POST'])
def segment():
    global current_requests
    
    with request_lock:
        if current_requests >= MAX_CONCURRENT_REQUESTS:
            return jsonify({'success': False, 'error': '系统繁忙，请稍后重试'}), 429
        current_requests += 1
    
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"=== 请求 {request_id} 开始 ===")
    
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '未上传图片'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名不能为空'})
        
        filename = f"{uuid.uuid4().hex}.png"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        logger.info(f"  - 保存图片到: {input_path}")
        try:
            with open(input_path, 'wb') as f:
                f.write(file.read())
            if os.path.exists(input_path):
                file_size = os.path.getsize(input_path)
                logger.info(f"  - 文件保存成功，大小: {file_size} bytes")
            else:
                logger.error(f"  - 文件保存失败")
        except Exception as e:
            logger.error(f"  - 文件保存异常: {e}")
            return jsonify({'success': False, 'error': '文件保存失败'})
        # Create DB task record (if enabled) and optionally upload input to S3/MinIO
        try:
            create_task_record(request_id, input_path)
            if S3_CLIENT is not None:
                s3_key_in = f"inputs/{filename}"
                s3_url_in = upload_file_to_s3(input_path, s3_key_in)
                if s3_url_in:
                    update_task_record(request_id, s3_input_url=s3_url_in)
        except Exception as e:
            logger.warning(f"任务记录/上传输入失败: {e}")
        
        model_type = request.form.get('model_type', 'Best')
        return_layers = request.form.get('return_layers', 'false').lower() == 'true'
        clothing_type = request.form.get('clothing_type', 'all')  # 服装类型：all, top, bottom
        
        logger.info(f"  - 请求参数: model={model_type}, layers={return_layers}, type={clothing_type}")
        
        if return_layers:
            success, error, top_path, bottom_path = segmenter.process_image(
                input_path, output_path, model_type, 
                return_layers=True, 
                clothing_type=clothing_type
            )
            
            if success:
                logger.info(f"=== 请求 {request_id} 成功（分图层） ===")
                result = {
                    'success': True,
                    'output_url': f'/outputs/{filename}'
                }
                if top_path:
                    top_filename = os.path.basename(top_path)
                    result['top_url'] = f'/outputs/{top_filename}'
                if bottom_path:
                    bottom_filename = os.path.basename(bottom_path)
                    result['bottom_url'] = f'/outputs/{bottom_filename}'
                # Upload outputs to S3 and update DB record if configured
                try:
                    if S3_CLIENT is not None:
                        s3_key_out = f"outputs/{filename}"
                        s3_url_out = upload_file_to_s3(output_path, s3_key_out)
                        if s3_url_out:
                            update_task_record(request_id, s3_output_url=s3_url_out)
                        # also upload layer files if present
                        if top_path and os.path.exists(top_path):
                            upload_file_to_s3(top_path, f"outputs/{os.path.basename(top_path)}")
                        if bottom_path and os.path.exists(bottom_path):
                            upload_file_to_s3(bottom_path, f"outputs/{os.path.basename(bottom_path)}")
                    update_task_record(request_id, output_path=output_path, completed_at=datetime.utcnow())
                except Exception as e:
                    logger.warning(f"上传输出或更新任务记录失败: {e}")
                return jsonify(result)
        else:
            success, error = segmenter.process_image(input_path, output_path, model_type)
            
            if success:
                logger.info(f"=== 请求 {request_id} 成功 ===")
                # Upload output to S3 and update DB record if configured
                try:
                    if S3_CLIENT is not None:
                        s3_key_out = f"outputs/{filename}"
                        s3_url_out = upload_file_to_s3(output_path, s3_key_out)
                        if s3_url_out:
                            update_task_record(request_id, s3_output_url=s3_url_out)
                    update_task_record(request_id, output_path=output_path, completed_at=datetime.utcnow())
                except Exception as e:
                    logger.warning(f"上传输出或更新任务记录失败: {e}")
                return jsonify({
                    'success': True,
                    'output_url': f'/outputs/{filename}'
                })
        
        logger.error(f"=== 请求 {request_id} 失败: {error} ===")
        return jsonify({'success': False, 'error': error})
            
    except Exception as e:
        logger.error(f"=== 请求 {request_id} 异常: {e} ===", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})
    finally:
        with request_lock:
            current_requests -= 1
        logger.info(f"=== 请求 {request_id} 结束 ===")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success': False, 'error': '文件过大，最大支持50MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"内部错误: {error}", exc_info=True)
    return jsonify({'success': False, 'error': '服务器内部错误，请稍后重试'}), 500

def signal_handler(sig, frame):
    global server_running
    logger.info("收到停止信号，正在关闭服务...")
    server_running = False
    time.sleep(2)
    logger.info("服务已关闭")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("=== 启动终极版服装精准抠图服务 ===")
        logger.info("服务将在 http://127.0.0.1:5088 运行")
        logger.info("正在预加载所有模型...")
        logger.info(f"最大并发请求数: {MAX_CONCURRENT_REQUESTS}")
        
        segmenter.load_models()
        
        logger.info("所有模型预加载完成，服务开始运行...")
        
        from waitress import serve
        serve(app, host='0.0.0.0', port=5088, threads=4, channel_timeout=300)
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)