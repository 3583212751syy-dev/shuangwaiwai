# 图裂变 API 接口文档

## 已实现功能

✓ **可选模式**（三种工作流）
✓ **强主题关联背景**（内置 6 种主题）  
✓ **参考范围滑块**（控制生成自由度）

---

## 端点：POST /infer_enhanced

### 基础参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| image | file | ✓ | — | 输入图片（PNG/JPG） |
| variants | int | — | 4 | 生成变体数量 |
| prompt | string | — | — | AI 补充提示词 |
| negative_prompt | string | — | — | 反向提示词 |
| control_image | file | — | — | 控制图（可选） |

### 新增裂变参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| mode | string | theme_background | reference_lock / theme_background / full_redraw | 工作模式 |
| theme | string | warm herbal | 见主题列表 | 背景主题 |
| theme_strength | float | 0.7 | 0.0-1.0 | 主题强度（0 = 弱，1 = 强） |
| reference_range | float | 0.6 | 0.0-1.0 | 参考范围（0 = 紧贴原图，1 = 完全自由） |
| background_change | bool | true | — | 是否允许背景变化 |

---

## 三种模式详解

### 模式 1：reference_lock（保留原始构图）

**场景**：需要保留原图的整体布局和视角，只做轻微风格改变

**参数示例**：
```json
{
  "mode": "reference_lock",
  "theme": "forest calm",
  "theme_strength": 0.5,
  "reference_range": 0.2
}
```

**结果特点**：
- 主体位置不变
- 背景略有风格化
- 最贴近原图

---

### 模式 2：theme_background（强主题背景）

**场景**：换主题/换背景，但主体产品保持可识别性

**参数示例**：
```json
{
  "mode": "theme_background",
  "theme": "luxury gold",
  "theme_strength": 0.9,
  "reference_range": 0.4
}
```

**结果特点**：
- 背景强烈变化
- 产品仍清晰可见
- 适合电商多色系展示

---

### 模式 3：full_redraw（完全重绘）

**场景**：基于原图做"全新创意"，生成视觉上完全不同的图片

**参数示例**：
```json
{
  "mode": "full_redraw",
  "theme": "sunset glow",
  "theme_strength": 0.8,
  "reference_range": 0.8
}
```

**结果特点**：
- 主体位置可改变
- 背景、光影都变
- 最自由的生成

---

## 内置主题

| 主题名 | 特点 | 用途 |
|--------|------|------|
| warm herbal | 暖棕色+草本绿 | 茶叶、保健品 |
| forest calm | 浅绿+森林深绿 | 自然、有机产品 |
| luxury gold | 奶油色+金色 | 高端、奢华品 |
| ocean fresh | 浅蓝+海蓝 | 清爽、饮品 |
| dark premium | 深灰+棕色 | 高级感、专业 |
| sunset glow | 橙色+红色 | 温暖、活力 |

---

## 调用示例

### 使用 curl

**场景：TEVOYATEA 茶叶，换成 luxury gold 主题**

```bash
curl -X POST "http://localhost:7860/infer_enhanced" \
  -F "image=@tea_pouch.png" \
  -F "variants=4" \
  -F "mode=theme_background" \
  -F "theme=luxury gold" \
  -F "theme_strength=0.85" \
  -F "reference_range=0.3" \
  -F "background_change=true"
```

### 使用 Python requests

```python
import requests

files = {'image': open('tea_pouch.png', 'rb')}
data = {
    'variants': 4,
    'mode': 'theme_background',
    'theme': 'warm herbal',
    'theme_strength': 0.8,
    'reference_range': 0.4,
    'background_change': True
}

response = requests.post(
    'http://localhost:7860/infer_enhanced',
    files=files,
    data=data
)

result = response.json()
print(f"请求 ID: {result['request_id']}")
print(f"模式: {result['mode']}")
print(f"生成了 {len(result['outputs'])} 张图片")

for out in result['outputs']:
    print(f"  - {out['filename']}: {out['path']}")
```

### 使用 JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('image', fs.createReadStream('tea_pouch.png'));
form.append('variants', '4');
form.append('mode', 'full_redraw');
form.append('theme', 'sunset glow');
form.append('theme_strength', '0.8');
form.append('reference_range', '0.7');

axios.post('http://localhost:7860/infer_enhanced', form, {
  headers: form.getHeaders()
}).then(res => {
  console.log(`生成了 ${res.data.outputs.length} 张图片`);
  res.data.outputs.forEach((out, idx) => {
    console.log(`  变体 ${idx + 1}: ${out.filename}`);
  });
}).catch(err => console.error(err));
```

---

## 返回值

```json
{
  "success": true,
  "request_id": "abc12345",
  "mode": "theme_background",
  "outputs": [
    {
      "path": "/tmp/model_worker/abc12345_0.png",
      "filename": "abc12345_0.png",
      "b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    },
    {
      "path": "/tmp/model_worker/abc12345_1.png",
      "filename": "abc12345_1.png",
      "b64": "..."
    }
  ]
}
```

---

## 参数调优指南

### 快速试验模板

| 场景 | mode | theme | theme_strength | reference_range |
|------|------|-------|-----------------|-----------------|
| 保留原图，换背景色 | reference_lock | * | 0.3-0.5 | 0.1-0.3 |
| 中等改变，可识别 | theme_background | * | 0.7-0.9 | 0.3-0.5 |
| 大胆创意，完全不同 | full_redraw | * | 0.8-1.0 | 0.7-0.9 |

### 调参建议

- **theme_strength 太低** → 背景看不出主题感，显得平淡
- **theme_strength 太高** → 主体被背景色压住，可能看不清
- **reference_range 太低** → 生成图和原图太像，没新意
- **reference_range 太高** → 可能偏离原图认知，风险大

**推荐起点**：theme_strength=0.7 + reference_range=0.5

---

## 部署说明

### 启动 worker

```bash
cd e:\Desktop\双接口\codex\model_worker
uvicorn worker_infer_enhanced:app --host 0.0.0.0 --port 7860 --reload
```

### Docker 运行（可选）

```bash
docker build -t image-fission:latest model_worker/
docker run -p 7860:7860 image-fission:latest
```

---

## 性能与限制

| 指标 | 值 |
|------|-----|
| 单个请求最大文件 | 50 MB |
| 最大变体数 | 10 |
| 超时时间 | 300 秒 |
| 输出格式 | PNG（无损，可透明） |

---

## 错误处理

### 400 Bad Request

```json
{
  "success": false,
  "error": "mode must be one of: reference_lock, theme_background, full_redraw"
}
```

### 422 Unprocessable Entity

缺少必需参数或类型错误。

### 500 Internal Server Error

模型加载失败或推理出错。

---

## 下一步

1. ✓ 启动 `/infer_enhanced` 服务
2. ✓ 用上述示例测试三种模式
3. ✓ 调参找到最佳组合
4. ✓ 集成到前端或批处理流程
5. ✓（可选）接入真实 AI 模型（SDXL/ControlNet）

---

## 常见问题

**Q: 如果不提供 theme，会怎样？**  
A: 默认使用 `warm herbal`（适合食品、茶叶）

**Q: reference_range 的数学含义是什么？**  
A: 0 = 最严格保留原图 → 1 = 完全自由生成。内部控制产品大小、位置、模糊度

**Q: 可以同时指定 prompt 和 mode 吗？**  
A: 可以。prompt 会传给真实模型；mode/theme 参数对模拟模式和真实模式都有效

**Q: 生成速度多快？**  
A: 模拟模式 <0.2 秒/图；真实 SDXL <10 秒/图（GPU）

---

## 技术细节（代码路径）

- API 端点实现：[model_worker/worker_infer_enhanced.py](../model_worker/worker_infer_enhanced.py#L157)
- 裂变函数库：[model_worker/worker_infer_enhanced.py](../model_worker/worker_infer_enhanced.py#L27-L95)
- 主题配色表：[model_worker/worker_infer_enhanced.py](../model_worker/worker_infer_enhanced.py#L37-L48)

---

**接口版本**：1.0  
**最后更新**：2026-08-18  
**维护者**：Codex AI  
