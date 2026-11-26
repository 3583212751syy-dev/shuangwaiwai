# 图片 AI：产品图提取与对比

统一入口：本目录。整理日期：2026-11-23 15:03（北京时间）。

## 本次已经完成

从保存后的“包装-产品图.xlsx”提取所有嵌入原图，保留原始字节。文件大小 411,092,924 字节；校验 SHA-256：`8b5a5c32299bd1350279cf485a64149b996480b334f0b1d5e628a43924d412af`。

| 核对项目 | 结果 |
|---|---:|
| 实际工作表 | Sheet1 |
| 图片文件 | 2442 |
| 不同图片 SHA-256 | 2442 |
| 图片所在位置 | 2648 |
| 缺失图片位置 | 0 |
| 未引用的媒体文件 | 0 |
| 图片解码、ZIP CRC、逐文件字节校验 | 全部通过 |

2648 是图片引用位置，2442 是不同原图数量。重复引用保存在 occurrences 清单里，不会丢失来源。该文件只有 Sheet1，不能据此证明在线原表“莆田、深圳”两张工作表都已完整复制。

## 按用途找文件

| 路径 | 内容 |
|---|---|
| [docs/method.md](docs/method.md) | 处理思路、失败原因、筛选规则和后续工作 |
| [extraction/extract_workbook.py](extraction/extract_workbook.py) | 可重复执行的提取与校验程序 |
| [reports/summary.json](reports/summary.json) | 本次提取统计与范围 |
| [reports/inventory.csv](reports/inventory.csv) | 每个原图的哈希、大小、尺寸与引用次数 |
| [reports/occurrences.csv](reports/occurrences.csv) | 每个图片位置对应的工作表、单元格和原图 |
| [site-source/](site-source/) | 图片对比网站当前源码快照，包含已有后续修改 |
| [model-tools/](model-tools/) | 本地模型分析、特征写回程序与依赖锁定 |
| [local-runner/](local-runner/) | 本地 HTTP 服务封装快照 |
| [history/](history/) | 前几轮已使用的旧处理脚本，仅供追溯 |
| [code-inventory.json](code-inventory.json) | 本目录代码文件哈希清单 |

## 重新提取

Python 3.11+，先安装 Pillow，然后运行：

```powershell
python -m pip install Pillow
python extraction/extract_workbook.py 'C:\Users\Administrator\Desktop\包装-产品图.xlsx' .\data\full-extraction
```

输入必须是已保存的 xlsx 文件。文件被 WPS 占用时，可先以共享读取方式复制到工作目录，不能直接修改其 ZIP 内容。输出 `originals/` 原图和各份清单；同名输出内容不一致时程序会停止。

## 原图与大文件位置

本次原图：`D:\GPT\2026-11-21\new-chat-2\work\full-extraction\originals`。

原始工作簿：`C:\Users\Administrator\Desktop\包装-产品图.xlsx`。

Git 保存代码、方法与全部校验清单。411 MB 工作簿、约 410 MB 原图、模型权重和特征缓存没有放入普通 Git；可以按清单校验并从原工作簿重建。没有上传登录凭据或私有访问令牌。

## 筛选与入库进展（2026-11-23 续做）

| 项目 | 结果 |
|---|---:|
| 联系表目检 | 39 页，全部 2442 张逐页看过 |
| 保守剔除的纯包装照 | 12 张（见 reports/selection-decisions.json，逐张带理由） |
| 保留入库原图 | 2430 张 |
| 来源行同款候选组 | 761 组 |
| 已生成 CLIP 512 维特征 | 2430 / 2430，失败 0 |
| 数量守恒 | 12 + 2430 + 0 = 2442，通过（reports/reconcile.json） |

新产物：

| 路径 | 内容 |
|---|---|
| [reports/selection.csv](reports/selection.csv) | 每张图的 keep/exclude 决策、来源行列、SHA-256 |
| [reports/groups.csv](reports/groups.csv) | 761 个同款候选组及成员 |
| [tools/make-contact-sheet.py](tools/make-contact-sheet.py) | 联系表生成（原图只读，不重编码） |
| [tools/build-selection.mjs](tools/build-selection.mjs) | 筛选定稿、分组、数量守恒 |
| [tools/analyze-all.mjs](tools/analyze-all.mjs) | 本机 CLIP 批量特征（断点续跑） |
| [tools/pack-site.mjs](tools/pack-site.mjs) | 组装成可运行的自包含站点（含双模型向量/本地模型/人工定名回写） |
| [tools/analyze-dinov2.mjs](tools/analyze-dinov2.mjs) | DINOv2-small 全量 384 维对照特征 |
| [tools/eval-model-recall.mjs](tools/eval-model-recall.mjs) | 1486 次留一法单模型召回对比 |
| [tools/eval-ensemble.mjs](tools/eval-ensemble.mjs) | CLIP / DINOv2 / 融合三方案召回对比 |
| [tools/verify-fusion.mjs](tools/verify-fusion.mjs) | 本地端到端验收双模型融合与首位同款 |

## 尚未完成 / 边界

- 只确认了**已保存工作簿 Sheet1** 的完整提取；无法证明覆盖在线原表的其它工作表（副本被禁止，不能自行确认）。
- 类目和品名是 CLIP 初筛建议（2430 张里绝大多数为「待确认」），不是人工定论；同款分组按表格来源行归为**候选**（同行平均相似度 0.826 高于随机 0.700，但只有约 11.6% 超过 0.9），都需在网页人工核对。相似度只是视觉参考，不代表型号或尺寸相同。
- 我没有把机器猜测当成人工筛选结果；`unreviewed`/`待确认` 是真实状态。
- site-source/.openai/hosting.json 保留原站点身份，仅供维护原站点；不要用于创建另一站点。
- 本地可运行目录仍是 `D:/GPT/2026-11-21/new-chat-2/work/deploy-product-library`（2430 张 + 149MB 本机模型）。
- **线上已发布（2026-11-23 17:20，17:37 修复入口）**：`https://product-image-match.app.workbuddy.host/?sharecode=MTCP4KDYCY`，复用同一应用槽位覆盖更新，线上运行的是本批 2430 张新库。上线自验收：首页 200、功能标记齐全、`/api/images` 返回 2430 张且全部带向量 / 761 组 / 12 个排除项零泄漏、图片端点与模型文件（vision onnx 89MB）均 200、线上向量以图搜自身/同组首位命中 5/5。所有 API 需带 `sharecode` 参数。
- **入口修复**：从 WorkBuddy 应用面板以非根路径打开曾显示 "Not found"（worker 对未匹配路径一律返回 404）。已在 `site-source/worker/index.js` 把兜底改为「GET 且非 `/api/` 的未匹配路径回首页」，API 404 保留；重建重发布后，CDP 真机从 `/preview` 打开 25 个请求全成功、0 个 4xx。
- **真机功能测试**：用桌面 `1.webp` 在真实 Chrome 走「上传图片找同款」全流程，识别为「金刚石打磨盘/五金工具（待确认，置信度 0.385）」，自动扩大到全库 2430 张，按同款分组返回 8 组候选（切割片套装 0.703 等），无异常、截图正常。

## 检索判据升级与模型对照（2026-11-24）

- **问题定位**：用户指出「表格一行=同一产品不同角度；底图是实拍、对比图是展示图，视觉相似度天然对不上；应产品相似度优先、视觉为辅」。实测确认：CLIP 把 17 种不同产品误标成「切割片套装」（高压软管/砂纸/离心风机/波纹管等），且旧算法按「绝对角度数」加分，导致 6 张一组的软管压过 3 张一组的真切割片。
- **验证基准**（reports/verification-baseline.json）：桌面 `1.webp`（4 片彩色花纹切割片展示图）的真实同款 = 第473行 image1540/1541（原图目检花纹、片数一致）。
- **算法改造**（site-source/search.mjs）：产品相似度 = 最高单张×0.6 + 强命中(≥0.65)比例×0.25 + 组内平均×0.15；不再用 CLIP 文本标签加权（错误率高）；不再奖励绝对角度数。真机 CDP 验证：修复前首位是软管组，修复后首位正确为第473行彩色切割片（3 角度，产品匹配度 0.604，领先第二名 0.574）。
- **模型对照**：从 hf-mirror 下载 `onnx-community/dinov2-small`（量化 24MB，tools/analyze-dinov2.mjs 全量特征）。快速对照中 DINOv2 对同款/软管区分度 0.50 vs 0.38，好于单张 CLIP；但同为纯视觉特征，无法单独解决实拍vs展示图域差异。结论：算法改造是根本，模型升级是辅助。
- **逐组定名**（reports/product-labels-manual.json / work/labels-manual.json）：761 组全部逐格目检定名并回写站点元数据，其中 105 组保守标注「待复核」，不编造品牌型号；17 个误标「切割片套装」组已逐张原图纠正真实品名。同时修正了早期 5 条按「来源行号」误当组索引写入的标签（如 1.webp 真同款实际是 groups.json 索引 471 / 来源行 473，检索按组名不受影响）。外部佐证（reports/external-sources.json）：义乌购「4.5寸115烧结金刚石锯片…切割片」、Alibaba「4.5 inch diamond cutting and grinding discs」、1688 百科「岩板切割片/磨片/倒角」共同支持「4.5寸彩色花纹切割片/打磨片（4片装）」，类目五金工具>磨具磨料。不写图片无法证明的品牌和型号。
- **全量模型对照**（reports/model-comparison.json）：不只测 `1.webp`，用 1486 次留一查询全量对比。CLIP 召回@1 41.59% / @5 63.32%；DINOv2-small 44.15% / 64.00%；z-score 等权融合 50.07% / 69.92%。因此站点接入双模型融合（`ai.mjs` 的 `createDinoEmbedder`、`search.mjs` 的融合评分），DINOv2-small q8 模型随站点本地发布，加载失败自动退回单模型。`tools/verify-fusion.mjs` 本地验收：2430 张 CLIP 向量 + 2430 张 DINO 向量全部就绪，模型文件 200，`1.webp` 首位仍是第473行彩色切割片。

## Git 仓库说明（2026-11-26）

本仓库包含「对比图片找产品」工具的全部代码、逻辑与小型数据；可直接克隆后本地运行。

### 随仓库发布的内容

| 内容 | 说明 |
|---|---|
| site-source/ | 网站源码：上传图片 → CLIP+DINOv2 融合检索底图库 → PP-OCRv6 读图规格 → 规格冲突对比（如 C16 ≠ C20 直接否决） |
| site-source/models/ocr/ | PP-OCRv6 tiny 检测/识别模型 + onnxruntime-web wasm（含 jsep/jspi 变体，缺 jsep.wasm 会崩）随仓库发布（约 66MB） |
| site-source/vendor/ | tesseract.js 运行时与依赖平铺包（local-runner 离线 OCR 用，约 61MB） |
| site-source/specs.json / overrides.json / references.json | 245 张底图 OCR 规格、人工定名回写、查证锚点库（气弹簧/防臭地漏/断路器/防火毯等 4 条，余弦≥0.82 采信） |
| work/*.json | 底图库分组标签、人工定名、筛选决策（embeddings*.json 为 2430 张特征缓存） |
| tools/、local-runner/、model-tools/、reports/、docs/、extraction/、history/ | 处理脚本、本地服务、模型分析工具、全部校验清单 |

### 未随仓库发布（可重建）

- 411MB 原始工作簿与约 410MB 原图（`data/`、`originals/`），按 `extraction/extract_workbook.py` 从原工作簿重建。
- 视觉模型不进仓库，首次运行由 @huggingface/transformers 自动下载：`Xenova/clip-vit-base-patch32`（CLIP）与 `onnx-community/dinov2-small`（DINOv2 q8）；国内可用 `HF_ENDPOINT=https://hf-mirror.com`。

### 本地运行

```bash
# 底图库版站点（含完整检索）
node site-source/build.mjs          # 组装自包含站点
node local-runner/server.mjs        # 本地服务（默认 4322）
# 发布固化
python tools/prepare-publish.py
```

线上地址：`https://product-image-match.app.workbuddy.host/?sharecode=MTCP4KDYCY`
