# 图裂变 image-fission

自托管 AI 图裂变工具（MVP）。输入一张原图，产出大量变异图，且：
- **规避侵权**：自动识别原图侵权元素（logo / 品牌 / 真人肖像 / 水印），生成时剥离，产出图不带明显侵权元素。
- **规避低质量**：模型逐张打分（清晰度 / 构图 / 糊图 / 畸变 / 乱码文字），不达标自动重绘。
- **可控区间**：与原图相似度、风格强度、重绘幅度均可滑杆控制。
- **自动评审**：系统自审批，不合格重生成，只给高质量结果。
- **数据不出域**：本地 GPU 推理（SDXL + ComfyUI），成本敏感。

## 环境约束（本机）
- GPU：RTX 4070 Ti 12GB → 用 **SDXL**（FLUX 需 24G，跑不动）。
- 磁盘：C 盘已满，**所有内容在 E 盘** `E:\Desktop\双接口\image-fission\`。
- 代理：`http://127.0.0.1:7897`（git/pip/hf 下载走代理）。

## 目录结构
```
image-fission/
  ComfyUI/            # 克隆的 ComfyUI（API 模式运行）
  venv/               # Python 虚拟环境（E 盘）
  src/
    config.py         # 全局配置（路径/参数/阈值）
    engine/comfy_client.py   # ComfyUI HTTP+WS 客户端
    pipelines/build.py       # 模式1/2 工作流构造（API Format）
    detection/infringement.py# 侵权检测(YOLO-World+InsightFace)→蒙版
    quality/score.py         # 质量评分(Laplacian+CLIP-IQA+OCR)
    orchestrator/            # 作业编排(runner) + Web(server)
    web/gallery.html         # 上传/滑杆/整体对照画廊
    setup/                   # 安装脚本
  jobs/               # 作业输出：JPG + meta.json + masks/
```

## 安装
```bash
cd E:/Desktop/双接口/image-fission
source venv/Scripts/activate
# 1) torch(CUDA) 已单独从 pytorch 源安装（见 P0）
# 2) 装其余依赖 + 自定义节点
bash src/setup/bootstrap_after_torch.sh
# 3) 下载模型（SDXL/IP-Adapter/antelopev2）
python src/setup/download_models.py
```

## 运行
```bash
# 终端A：启动 ComfyUI
source venv/Scripts/activate
python ComfyUI/main.py --cuda-device 0 --listen 127.0.0.1 --port 8188 --max-queue-size 20 --medvram
# 终端B：启动 Web 服务
source venv/Scripts/activate
uvicorn orchestrator.server:app --port 8000
# 浏览器打开 http://127.0.0.1:8000 上传原图、拉滑杆、看整体对照
```

## 裂变模式
- **模式1 换景换风格**：IP-Adapter(与原图相似度滑杆) + 风格 prompt + KSampler。
- **模式2 内容重绘**：img2img(重绘幅度) + IP-Adapter(保主体) + KSampler。
- 每批 4 张，一次作业 100 批 ≈ 400 张，不达标自动重绘。

## 技术选型
ComfyUI(API) + SDXL(CreativeML OpenRAIL-M，商用合规，避开 FLUX.1-dev 非商用)
+ IP-Adapter(相似度) + BiRefNet/RMBG(主体锁定) + YOLO-World/InsightFace(侵权检测)
+ CLIP-IQA/Q-Align(质量) + FastAPI(服务) + Gradio/HTML(画廊)。
