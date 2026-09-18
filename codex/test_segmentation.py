import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import FashionSegmenter

def test_segmentation():
    segmenter = FashionSegmenter()
    segmenter._init_models()
    
    upload_folder = 'uploads'
    test_output_folder = 'test_outputs'
    os.makedirs(test_output_folder, exist_ok=True)
    
    for filename in os.listdir(upload_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(upload_folder, filename)
            output_path = os.path.join(test_output_folder, f'test_{filename}')
            
            print(f'\n=== 测试图片: {filename} ===')
            try:
                result = segmenter.process_image(input_path, output_path)
                print(f'  ✓ 处理成功')
                
                result_np = np.array(result)
                alpha = result_np[:, :, 3]
                clothing_area = np.sum(alpha > 0)
                total_area = alpha.shape[0] * alpha.shape[1]
                print(f'  服装区域占比: {clothing_area/total_area*100:.1f}%')
                
                holes = cv2.bitwise_not(alpha)
                contours, _ = cv2.findContours(holes, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                hole_count = sum(1 for i in range(len(contours)) if cv2.contourArea(contours[i]) > 100)
                print(f'  孔洞数量(面积>100): {hole_count}')
                
            except Exception as e:
                print(f'  ✗ 处理失败: {e}')

if __name__ == '__main__':
    test_segmentation()