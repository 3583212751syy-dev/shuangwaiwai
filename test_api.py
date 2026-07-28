import requests

url = 'http://127.0.0.1:5088/api/segment'
image_path = 'test_image.png'

with open(image_path, 'rb') as f:
    files = {'image': ('test_image.png', f, 'image/png')}
    data = {'model_type': 'Best'}
    
    print(f"正在发送请求...")
    response = requests.post(url, files=files, data=data, timeout=300)
    
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print(f"输出URL: {result.get('output_url')}")
        else:
            print(f"错误: {result.get('error')}")