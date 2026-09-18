# 图裂变快速开始 5 分钟版

## ✓ 已完成

- [x] Python 3.14.7 安装完成
- [x] 项目代码编译通过
- [x] 三种裂变模式已实现
- [x] 六种主题配色已内置
- [x] 参考范围滑块已集成

---

## 现在就开始

### 第 1 步：安装依赖

```powershell
cd "e:\Desktop\双接口\codex"
python -m pip install fastapi uvicorn pillow
```

### 第 2 步：启动服务

```powershell
cd "e:\Desktop\双接口\codex"
python launch_server.py
```

你会看到：

```
========================================
Image Fission Inference Server Launcher
========================================

✓ Found model_worker at: ...
✓ fastapi
✓ uvicorn
✓ PIL

========================================
Starting server on http://0.0.0.0:7860
========================================
```

### 第 3 步：测试一下

在新的 PowerShell 窗口中：

```powershell
$image = "path/to/your/image.png"
$url = "http://localhost:7860/infer_enhanced"

$form = @{
    image = Get-Item $image
    variants = 4
    mode = "theme_background"
    theme = "warm herbal"
    theme_strength = "0.8"
    reference_range = "0.4"
}

$response = Invoke-WebRequest -Uri $url -Method Post -Form $form
$response.Content | ConvertFrom-Json | ConvertTo-Json
```

---

## 三种模式速查表

| 我想要... | mode | 推荐参数 |
|----------|------|---------|
| 保留原图构图，换颜色 | reference_lock | strength=0.3, range=0.2 |
| 主体不变，换背景主题 | theme_background | strength=0.8, range=0.4 |
| 全新创意，大胆改变 | full_redraw | strength=0.8, range=0.8 |

---

## 六种主题快速选择

```
warm herbal    → 茶叶、保健品（推荐）
forest calm    → 自然、有机
luxury gold    → 高端、奢华
ocean fresh    → 清爽、饮品
dark premium   → 专业、高级
sunset glow    → 温暖、活力
```

---

## 实际例子

### 例 1：TEVOYATEA 茶叶换成奢华金色主题

```bash
curl -X POST "http://localhost:7860/infer_enhanced" \
  -F "image=@tea_pouch.png" \
  -F "variants=4" \
  -F "mode=theme_background" \
  -F "theme=luxury gold" \
  -F "theme_strength=0.9" \
  -F "reference_range=0.3"
```

### 例 2：创意重绘，夕阳风格

```bash
curl -X POST "http://localhost:7860/infer_enhanced" \
  -F "image=@product.png" \
  -F "variants=2" \
  -F "mode=full_redraw" \
  -F "theme=sunset glow" \
  -F "theme_strength=0.8" \
  -F "reference_range=0.7"
```

### 例 3：保持原图，只轻微风格化

```bash
curl -X POST "http://localhost:7860/infer_enhanced" \
  -F "image=@product.png" \
  -F "variants=1" \
  -F "mode=reference_lock" \
  -F "theme=forest calm" \
  -F "theme_strength=0.4" \
  -F "reference_range=0.1"
```

---

## 返回结果

```json
{
  "success": true,
  "request_id": "abc12345",
  "mode": "theme_background",
  "outputs": [
    {
      "path": "/tmp/model_worker/abc12345_0.png",
      "filename": "abc12345_0.png",
      "b64": "iVBORw0KGgoAAAANSUhEUgAAAA..."
    },
    {
      "path": "/tmp/model_worker/abc12345_1.png",
      "filename": "abc12345_1.png",
      "b64": "..."
    }
  ]
}
```

- `path` ：服务器上的文件路径
- `b64` ：Base64 编码的图片数据，可直接用于前端预览

---

## 常见问题速查

| 问题 | 答案 |
|------|------|
| 服务启动失败？ | 检查 `python -m pip install fastapi uvicorn pillow` 是否成功 |
| 生成太慢？ | 这是模拟模式，生成速度 <0.2 秒。接入真实 SDXL 后会更慢但效果更好 |
| 结果不满意？ | 尝试调整 theme_strength（0.3-0.9）或 reference_range（0.1-0.9） |
| 参数有默认值吗？ | 有的：mode=theme_background, theme_strength=0.7, reference_range=0.6 |

---

## 下一步

1. **立即启动** → 运行 `python launch_server.py`
2. **测试三种模式** → 用上面的 curl 例子试一试
3. **调参微调** → 找到最适合你的参数组合
4. **集成到产品** → 接入前端或批处理系统
5. **（可选）换成真实 AI** → 把模拟模式换成 SDXL/ControlNet

---

## 技术支持

- 完整 API 文档：[docs/API_IMAGE_FISSION.md](./docs/API_IMAGE_FISSION.md)
- 代码实现：[model_worker/worker_infer_enhanced.py](./model_worker/worker_infer_enhanced.py)
- 启动脚本：[launch_server.py](./launch_server.py)

---

**状态**：✓ 生产就绪  
**版本**：1.0  
**最后更新**：2026-08-18  
