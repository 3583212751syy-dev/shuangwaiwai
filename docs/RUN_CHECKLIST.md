运行与基准检查清单

目的：快速在目标机器上启动堆栈、执行集成测试与吞吐基准，并收集可供调优的数据。

前提
- Docker 已安装：`docker --version`
- （GPU）NVIDIA 驱动与 Container Toolkit：`nvidia-smi`
- （可选）设置 Hugging Face Token：`export HF_TOKEN=<token>`（Windows PowerShell 使用 `setx HF_TOKEN <token>` 或手动在运行时设置）

准备模型（可选，真实推理）
```bash
python model_worker/download_models.py stabilityai/stable-diffusion-xl-base-1.0 --dest ./models/sdxl
python model_worker/download_models.py lllyasviel/control_v11p_sd15_tile --dest ./models/controlnet
```

生成示例图片
```bash
python3 tests/generate_sample.py
# Windows
python tests/generate_sample.py
```

本地开发堆栈（模拟，CPU）
```bash
docker-compose up --build
# 或一键脚本
./run.sh
# PowerShell
./run.ps1
```

GPU（真实推理）
```bash
# 设置 HF_TOKEN（如需要）
export HF_TOKEN=<your_token>
# 构建并运行 GPU 容器
docker compose -f docker-compose.real.yml up --build
```

集成测试（上传→轮询→下载）
```bash
# 一键运行会执行集成测试
./run.sh
# 或手动运行
python tests/integration_test.py
```

吞吐基准（示例）
```bash
# 单张延迟测试
python bench/throughput_benchmark.py --url http://localhost:8502/infer_enhanced --image tests/sample.jpg --mode single --iters 20

# 批量吞吐测试
python bench/throughput_benchmark.py --url http://localhost:8502/infer_batch_enhanced --images tests/ --mode batch --batch-size 4 --iters 10 --micro-batch 1
```

日志与故障排查
- 查看容器日志：`docker-compose logs -f server qa model_worker`
- GPU 容器日志：`docker compose -f docker-compose.real.yml logs -f model_worker_real`
- 查看运行容器：`docker ps`

停止堆栈
```bash
docker-compose down
# GPU 堆栈
docker compose -f docker-compose.real.yml down
```

收集数据并反馈
- 运行基准后，粘贴：`images/sec`、median latency、迭代次数、GPU 型号与显存（`nvidia-smi` 输出）。
- 我会基于该数据给出 `micro_batch` / `CONCURRENCY` / `fp16` 的具体默认值并把最终 `docker-compose.override.yml` 提交到仓库。
