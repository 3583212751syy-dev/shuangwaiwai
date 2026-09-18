import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from rembg import remove

class MultiModelFashionSegmenter:
    """
    多层模型服装分割器 - 使用多种模型进行层层筛选处理
    优先保证质量，不追求速度
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 初始化所有模型
        self._init_models()
    
    def _init_models(self):
        """初始化所有分割模型"""
        print("正在加载Segformer服装分割模型...")
        self.segformer_processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.segformer_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(self.device)
        
        print("所有模型加载完成")
    
    def segment_with_segformer(self, image):
        """使用Segformer进行服装分割"""
        original_size = image.size  # (width, height)
        inputs = self.segformer_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.segformer_model(**inputs)
        
        logits = outputs.logits.cpu()
        mask = torch.argmax(logits, dim=1).squeeze().numpy()
        
        # 服装类别: 1=上衣, 2=下装, 3=全身装等
        # 创建服装蒙版（合并多个服装类别）
        cloth_mask = np.zeros_like(mask)
        for cloth_class in [1, 2, 3, 4, 5, 6, 7, 8]:  # 所有服装类别
            cloth_mask[mask == cloth_class] = 255
        
        # 将蒙版调整回原始图像尺寸
        cloth_mask = cv2.resize(cloth_mask.astype(np.uint8), (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)
        
        return cloth_mask
    
    def segment_with_rembg(self, image):
        """使用RMBG进行背景去除"""
        result = remove(image)
        mask = np.array(result)[:, :, 3]  # 获取alpha通道作为蒙版
        return mask
    
    def detect_skin(self, image):
        """检测皮肤区域"""
        image_np = np.array(image)
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # HSV皮肤范围
        lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
        upper_skin1 = np.array([30, 200, 255], dtype=np.uint8)
        lower_skin2 = np.array([150, 20, 50], dtype=np.uint8)
        upper_skin2 = np.array([180, 200, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # RGB皮肤检测补充
        r, g, b = cv2.split(image_np)
        rgb_skin = ((abs(r - g) < 20) & (r > b) & (g > b) & (r > 90) & (g > 80) & (b > 60))
        rgb_skin = rgb_skin.astype(np.uint8) * 255
        
        combined = cv2.bitwise_or(skin_mask, rgb_skin)
        
        # 形态学操作去除噪点
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return combined
    
    def refine_mask(self, mask, image):
        """优化蒙版边缘"""
        # 形态学闭运算填充空洞
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 形态学开运算去除噪点
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 连通区域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 保留最大的几个连通区域
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_indices = np.argsort(areas)[::-1]
            
            # 保留前2-3个最大的区域（通常是上衣+下装）
            max_components = min(3, len(sorted_indices))
            new_mask = np.zeros_like(mask)
            
            for i in range(max_components):
                component_label = sorted_indices[i] + 1
                new_mask[labels == component_label] = 255
            
            mask = new_mask
        
        # 边缘平滑
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask[mask > 127] = 255
        mask[mask <= 127] = 0
        
        return mask
    
    def composite_masks(self, masks):
        """多层蒙版融合"""
        if not masks:
            return None
        
        # 初始蒙版
        final_mask = masks[0].copy()
        
        for mask in masks[1:]:
            if mask is None:
                continue
            
            # 取并集（保留任何模型检测到的服装区域）
            final_mask = cv2.bitwise_or(final_mask, mask)
        
        return final_mask
    
    def remove_body_parts(self, mask, skin_mask):
        """从服装蒙版中去除身体部分"""
        # 创建身体区域的反掩码
        body_inv = cv2.bitwise_not(skin_mask)
        
        # 从服装蒙版中减去身体区域
        result = cv2.bitwise_and(mask, body_inv)
        
        # 形态学操作修复边缘
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return result
    
    def process_image(self, input_path, output_path):
        """处理图片 - 多层模型筛选"""
        print(f"\n正在处理: {input_path}")
        image = Image.open(input_path).convert("RGB")
        image_np = np.array(image)
        
        # 第一步：使用Segformer进行服装分割
        print("步骤1: Segformer服装分割...")
        segformer_mask = self.segment_with_segformer(image)
        
        # 第二步：使用RMBG进行背景去除
        print("步骤2: RMBG背景去除...")
        rembg_mask = self.segment_with_rembg(image)
        
        # 第三步：检测皮肤区域
        print("步骤3: 皮肤检测...")
        skin_mask = self.detect_skin(image)
        
        # 第四步：融合多层蒙版
        print("步骤4: 蒙版融合...")
        masks = [segformer_mask, rembg_mask]
        combined_mask = self.composite_masks(masks)
        
        # 第五步：去除身体部分
        print("步骤5: 去除身体部分...")
        final_mask = self.remove_body_parts(combined_mask, skin_mask)
        
        # 第六步：优化蒙版边缘
        print("步骤6: 蒙版优化...")
        final_mask = self.refine_mask(final_mask, image)
        
        # 第七步：应用蒙版生成透明背景图片
        print("步骤7: 生成透明背景图片...")
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        result.paste(image, mask=Image.fromarray(final_mask))
        
        # 保存结果
        result.save(output_path)
        print(f"处理完成，结果保存到: {output_path}")
        
        return result

if __name__ == "__main__":
    import os
    
    # 输入输出路径 - 模特合集文件夹
    input_image_path = r"D:\Users\Administrator\Desktop\模特合集\T恤详情页\女T\2.jpg"
    output_image_path = r"D:\Users\Administrator\Desktop\模特合集\T恤详情页\女T\output_clothes.png"
    
    if not os.path.exists(input_image_path):
        print(f"错误：输入文件不存在: {input_image_path}")
        exit(1)
    
    # 创建分割器（加载所有模型）
    segmenter = MultiModelFashionSegmenter()
    
    # 处理图片
    segmenter.process_image(input_image_path, output_image_path)
    
    print("\n✅ 服装分割处理完成！")
