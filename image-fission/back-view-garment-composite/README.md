# 背面图合成（back-view-garment-composite）

> 本目录是 `image-fission` 主工程（图裂变 MVP）下的一个**子项目**：把正面模特的衣服印花转身成纯背面图。
> 与 `src/` 里的裂变管线相互独立，仅共享"本地 SDXL + ComfyUI"环境。

把**正面模特图**的衣服（LIMITLESS 泼墨字、印花、抽绳）**100% 像素级保留**，只把模特转身成纯背面（背对镜头、后脑勺、双手插兜），输出 1350×1800 竖版、暖米色背景，与正面配对成电商产品图。

## 最终交付

- `results/back_v34_pose_back_view_v34_s_00002_.png` —— **主交付**：纯背、双手插兜（与正面 pose 一致）、暖米色背景，正面印花 100% 复用（非镜像）。衬衫墨水占比 27.0% vs 正面 27.5%（仅差 0.5%）。
- `results/2panel_front_v34pose.png` —— 正/背并排对比。
- `results/3panel_old_vs_new.png` —— 旧版（双臂下垂、灰底）vs 新版（双手插兜、米底）对比。
- `results/3panel_front_cv2_catvton.png` —— 左:正面 / 中:cv2 交付 / 右:CatVTON git 模型（失败对照，生成 flesh blob 无印花）。
- `results/back_v33_v23s2024_beige.png` —— 上一版交付（双臂下垂底图，灰底后处理换米色）。

## 方法（为什么是这样）

**生成式换装模型做不到"印花 100% 一模一样"**，已逐一验证：

| 模型 | 结论 |
|---|---|
| CatVTON (zhengchong/CatVTON) | 本机已完整跑通（attn+base+vae+SCHP 就位，diffusers pin 0.32.2），但输出是 flesh blob，印花/文字完全消失 |
| OOTDiffusion / IDM-VTON / fal-tryon | 都是"重新画"衣服，需付费 API 或 diffusers+大显存，8GB 不现实 |
| IP-Adapter / ControlNet 直接生成背面 | 只能风格迁移，文字会被画歪、镜像或重绘模糊 |

**唯一满足"印花像素级一致"的路线 = cv2 透视合成**（`composite_v34_pose.py`）：

1. 用 ComfyUI+SDXL 生成**纯背 AI 底图** `back_view_v34_s_00002_.png`（IP-Adapter 借姿态、ControlNet 控形、warm beige 背景、双手插兜、白T白短裤）。
2. 预处理：在衣服 y 范围内对 `g<60` 像素 `cv2.inpaint` —— 清除底图自带的 IP-Adapter 泼墨 + 黑色腰头伪影，得到干净白T白短裤画布。
3. 衣服检测改用 **GrabCut figure**（底图衣服都是纯白，灰度范围检测会失效）。
4. `cv2.getPerspectiveTransform` 把正面印花 warp 到背面衣服区域（**绝不做水平镜像**，否则 LIMITLESS → SSELITMIL）。
5. 衬衫整张贴、裤子只贴墨水（lum<130，抽绳自然消失），Gaussian 羽化融合。

## 复现

环境：主 `image-fission/venv`（torch 2.11+cu128 / diffusers 0.32.2 / opencv / PIL / numpy）。

```bash
# 最终一步法（pose 匹配版）
python composite_v34_pose.py <back_base.png>
# 例：python composite_v34_pose.py inputs/back_view_v34_s_00002_.png
```

> 脚本内路径为原工作区绝对路径（`D:\.workbuddy\2026-08-16-00-13-40\image-fission\...`），
> 在本仓库仅作归档；要重跑请把 `inputs/` 与对应底图放好并调整脚本顶部常量。

## 文件清单

```
image-fission/
├── README.md
├── .gitignore
├── scripts/
│   ├── composite_v34_pose.py        # 最终：pose 匹配 + 米底，单步 13s
│   ├── composite_v33.py             # 上一版：灰度范围检测衣服，双臂下垂底图
│   ├── beige_bg_v33.py              # v33 米色背景后处理（GrabCut 换底）
│   ├── run_catvton.py               # CatVTON git 模型推理（对照，已确认失败）
│   ├── make_3panel.py               # 正面/cv2/CatVTON 三连对比
│   ├── make_v34_panels.py           # 新版 vs 旧版 对比图
│   ├── download_base_modelscope.py  # 基础模型(SD1.5-inpainting) ModelScope 镜像下载
│   ├── download_catvton_weights.py  # CatVTON 权重 HF+Xet 下载（备份）
│   └── make_skeleton_v23b.py        # mediapipe 骨架翻转生成同 pose 背面（需降 protobuf）
├── inputs/
│   ├── front_model.jpg              # 正面英雄图（源印花）
│   ├── back_reference_real.png      # 真实背面照（参考）
│   └── back_view_v34_s_00002_.png   # AI 纯背底图（最终合成底）
└── results/
    ├── back_v34_pose_back_view_v34_s_00002_.png  # ★ 主交付
    ├── 2panel_front_v34pose.png
    ├── 3panel_old_vs_new.png
    ├── 3panel_front_cv2_catvton.png
    └── back_v33_v23s2024_beige.png
```

## 关键经验

- **Xet CDN**：huggingface.co 大文件走 Xet 数据面，本机需 `HF_HUB_ENABLE_HF_XET=1` + 代理；小文件快、大文件很慢但可达（base 3.44GB 跑过 2h23m 完整拿到）。
- **diffusers 必须 pin 0.32.2**：CatVTON `pipeline.py` 顶层 import `StableDiffusionSafetyChecker`，diffusers git main 已删该类。
- **复用主 venv**：别新建 venv（继承系统 python 无 torch / 装 torch 会与 ComfyUI 冲突）；直接在主 venv 加 diffusers+accelerate。
- **Git-Bash `/d/...` 路径坑**：Windows Python 会把 `/d/foo` 解析成 `D:\d\foo`，必须写 `D:\foo` 原生 Windows 路径。
- **绝不做水平镜像**：正面印花 warp 到背面时镜像会让文字反向。
