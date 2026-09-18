#!/usr/bin/env python3
"""
Test the image fission API with the uploaded TEVOYATEA image.
Save results to generated/outputs/
"""
import sys
from pathlib import Path

# Add the attachment image path - this will be saved in a temp location
# For now, create a synthetic test image that represents the tea pouch

from PIL import Image, ImageDraw, ImageFont
import requests
import json
import base64

def create_test_image():
    """Create a simple test image to represent the tea pouch."""
    # Create a kraft paper-like background with a label
    img = Image.new('RGB', (400, 500), color=(210, 180, 150))  # kraft paper color
    draw = ImageDraw.Draw(img)
    
    # Draw a label area (white rectangle)
    label_x1, label_y1 = 50, 80
    label_x2, label_y2 = 350, 400
    draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=(245, 240, 230))
    
    # Draw border
    draw.rectangle([label_x1, label_y1, label_x2, label_y2], outline=(180, 140, 80), width=3)
    
    # Add some text indication
    try:
        draw.text((100, 150), "TEVOYATEA", fill=(139, 69, 19))
        draw.text((100, 200), "Rooibos Tea", fill=(180, 100, 50))
        draw.text((100, 250), "Natural Herbs", fill=(100, 150, 80))
    except:
        pass
    
    return img

def test_modes():
    """Test all three fission modes."""
    
    print("=" * 70)
    print("图裂变 API 测试")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = Path("generated/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create or load test image
    test_img = create_test_image()
    test_img_path = output_dir / "test_input.png"
    test_img.save(test_img_path)
    print(f"✓ 测试图片已保存: {test_img_path}")
    print()
    
    # API endpoint
    api_url = "http://localhost:7860/infer_enhanced"
    
    # Test configurations
    tests = [
        {
            "name": "模式 1: reference_lock（保留原图构图）",
            "params": {
                "mode": "reference_lock",
                "theme": "warm herbal",
                "theme_strength": "0.3",
                "reference_range": "0.2",
                "variants": "1"
            }
        },
        {
            "name": "模式 2: theme_background（强主题背景）",
            "params": {
                "mode": "theme_background",
                "theme": "luxury gold",
                "theme_strength": "0.9",
                "reference_range": "0.4",
                "variants": "1"
            }
        },
        {
            "name": "模式 3: full_redraw（完全重绘）",
            "params": {
                "mode": "full_redraw",
                "theme": "sunset glow",
                "theme_strength": "0.8",
                "reference_range": "0.7",
                "variants": "1"
            }
        }
    ]
    
    # Run tests
    for i, test_config in enumerate(tests, 1):
        print(f"[测试 {i}/3] {test_config['name']}")
        print(f"  参数: {json.dumps(test_config['params'], ensure_ascii=False, indent=2)}")
        
        try:
            with open(test_img_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(api_url, files=files, data=test_config['params'], timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ 成功！请求ID: {result.get('request_id')}")
                
                # Save output images
                for j, output in enumerate(result.get('outputs', [])):
                    output_path = output_dir / f"result_{i}_{j}.png"
                    
                    # Decode base64 image
                    img_data = base64.b64decode(output['b64'])
                    with open(output_path, 'wb') as f:
                        f.write(img_data)
                    
                    file_size = len(img_data) / 1024  # KB
                    print(f"    └─ 已保存: {output_path} ({file_size:.1f} KB)")
                
                print()
            else:
                print(f"  ✗ 失败: HTTP {response.status_code}")
                print(f"    {response.text}")
                print()
        
        except requests.exceptions.ConnectionError:
            print(f"  ✗ 连接失败 - API 服务器未运行 (localhost:7860)")
            print(f"     请确保在另一个终端中运行了: python -m uvicorn worker_infer_enhanced:app --host 0.0.0.0 --port 7860")
            print()
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            print()
    
    print("=" * 70)
    print("✓ 测试完成！")
    print(f"✓ 结果已保存到: {output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    test_modes()
