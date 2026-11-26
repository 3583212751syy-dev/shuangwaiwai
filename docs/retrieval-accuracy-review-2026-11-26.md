# 同款产品识别准确率评审

日期：2026-11-26。范围：product-image-match 分支的 README、site-source/ai.mjs、search.mjs、ai-client.mjs、specs.mjs、worker/index.js、reports/model-comparison.json。本次为静态代码评审与上游模型调研；未修改生产逻辑、未重新训练、未证明准确率提升。

## 结论

当前不应只替换模型。先修复双模型持久化和降级逻辑、同款判定与OCR证据，再用外部展示图对实拍底库建立真实验证集。品名识别、同款召回、SKU规格判定是三个任务，需要分别评估。

仓库报告的1486次底图库留一评估：CLIP Recall@1=41.59%，DINOv2-small=44.15%，融合=50.07%；融合Recall@5=69.92%。这些是已有报告数据，未在本次复测，不代表真实用户上传图的准确率。按来源行作为真值可能既有错分组，也存在跨组同款，需人工标定。

## 代码发现与优先级

### P0：DINO向量未持久化，且单条缺失关闭全库融合

worker/index.js 的 PATCH ai 分支只保存CLIP字段，没有保存输入的 dinoEmbedding。ai-client.mjs prepareIndex 调用 rawAnalyze 得到DINO后提交，但服务端返回记录会覆盖本地记录；旧图库DINO由overrides按filename补回，新上传文件未必覆盖。

search.mjs useDino 要求所有组每张图片都具备DINO，否则全库退回CLIP。一条新记录缺DINO即可触发。应测试：上传新底图、完成索引、刷新页面、重启服务后DINO是否保留；混合缺失记录是否仍保留双模型候选召回。

修复建议：服务端独立校验并保存DINO向量与model/revision/pooling/preprocess/dim；元数据编辑不得抹掉已有向量。两路独立召回后采用候选并集和校准排序，缺失模型只影响对应证据；不要直接把不同模型的原始余弦分混比。

### P0：融合z分数的比值不能当同款概率

ai-client.mjs verdict 对productScore用3.0/4.5和1.35倍阈值；search.mjs融合分数为当次图库组分数z-score均值。z-score可能为负且随候选分布改变，不是概率；next<=0直接令ratio=Infinity，不能说明真实同款。单模型余弦量纲又不同，统一阈值会造成另一种误判。界面对每个候选与下一候选调用verdict，还可能把后排孤立候选标为疑似同款。

修复建议：召回排序与同款判定分开；校准原始绝对相似度、差值、两路一致性、局部证据及规格证据。无负样本校准前仅输出候选，不直接断言“库里没有该产品同款”。

### P0：OCR次数不等于商品件数

specs.mjs compareSpecs 把C16等token出现次数视作商品件数并硬否决。详情图同一产品可能有正反面、多处说明或重复文本；OCR漏读也常见。两次C16不能证明2件装。单个C16/C20误识别同样会误杀真同款。

修复建议：OCR返回文本框、置信度、来源图及字段上下文；数量仅接受明确包装数量字段或可靠实例计数；同产品不同套装分开给出“主体同款/规格差异”，不要混为外观不同。高可信且字段语义相同的规格冲突才可强排除，其余标待确认。

### P1：规格只验证首位代表图

ai-client.mjs execute流程只取topId代表图进行规格比较，其余角度、其他候选未核验；首位冲突后没有看到对剩余候选重新排序的步骤。CLIP选出的代表图也未必是DINO最匹配视图。

修复建议：先召回Top30产品组，对Top10–20组的相关视图查规格、做精排，再输出最终候选；未知规格不能当作规格一致。

### P1：全图全局特征受背景与拍摄域影响

ai.mjs DINO将last_hidden_state所有token直接平均，包含CLS；这是一种实现选择，不是可直接认定的错误，但应对照官方CLS、patch-only均值和局部patch匹配。所有池化变更必须重建底库特征并带版本，不能混用旧向量。

建议并行保留原图与商品主体裁剪，多视图检索。裁剪可能漏配件，保留原图回退；不要只删背景后丢失套装结构。当前CLIP/DINO使用同一个strong>=0.65阈值，需分别校准，不能假设同一余弦阈值代表同样可靠性。

### P1：品名识别仍是有限词典与近邻推断

ai.mjs createRecognizer使用CATALOG英文短语与CLIP相似度；它不是开放式中文产品理解。identify的referenceName来自近邻，即便未确认，也会在verdict的nameStrong路径帮助降低判定门槛，存在同一视觉证据重复加信心。ai-client的4个参考锚点无法代表完整761组。

建议结构化产品属性：中文名、用途、主体形状、连接头、孔位、接口、颜色、明确尺寸/型号、套装关系、未知字段。用可信人工标签与图像证据交叉核验，未知不补写。名称与匹配身份分别显示。

### P1：人工反馈只记录，未参与检索

前端确认/否定保存的是decision、query文件名和时间；检索代码未读取confirmations。文件名不是稳定图像身份，不能支持可靠学习。

建议记录查询图SHA-256、候选productId/groupId、确认者、版本、same/different/uncertain及明确变体关系。形成可导出样本、困难负样本和固定测试集，再用于精排校准或微调。不能让同一确认对泄漏到独立测试集。

## 优先对照的模型与项目

| 候选 | 用途 | 接入判断 |
|---|---|---|
| QwenLM/Qwen3-VL-Embedding | 官方Embedding-2B/8B与Reranker-2B/8B，图文混合召回及候选精排 | 优先测试Reranker-2B对Top候选重排；相关性分数不等于SKU相同，仍需规格核验。现有浏览器WASM不能直接当替换文件，建议后端Python推理服务，测内存/显存、延迟 |
| google-research/big_vision SigLIP2 | 多语言图文语义特征、零样本分类；有保留长宽比的NaFlex变体 | 对照当前CLIP，重点测商品命名与跨域召回；官方通用基准不保证本图库同款更准 |
| facebookresearch/dinov3 | 视觉特征和局部密集特征候选 | 对照DINOv2，关注小配件结构和主体区域；需核对自定义模型许可和权重取得条件，不能直接称Apache开源替代 |
| cvg/LightGlue | 图像对局部特征匹配 | 适合作为纹理/图案/孔位等可见对应证据；大视角、无纹理和反光件可能失败，失败不等于非同款 |
| polysius/visual-search | 社区产品检索架构：裁剪、CLIP/DINO、LoRA、FAISS、ONNX、FastAPI | 参考流程而非直接替换成品；需检查许可证、训练样本和维护情况，不能采信介绍作为效果保证 |

来源：
- https://github.com/QwenLM/Qwen3-VL-Embedding
- https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md
- https://github.com/facebookresearch/dinov3
- https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- https://github.com/cvg/LightGlue
- https://github.com/polysius/visual-search
- https://huggingface.co/docs/transformers/model_doc/dinov2

Skill只能组织调用和业务规则，不会自行让模型权重获得更高识别能力。优先选择上述可测模型、可复现管线和真实标注集。

## 建议实施顺序与验收

1. 修复P0逻辑并补针对性回归：新图刷新后双模型仍存在；一条缺DINO不关闭全库；z分数正负边界不误判；重复OCR不推断数量；首位规格不符后可考虑后续候选。
2. 收集至少100–200张独立查询图：展示图对实拍、不同角度、包装、背景、多产品、形近不同规格、库外商品。按产品ID标定同主体和同SKU两种答案；保留验证集和锁定测试集。
3. 先测原流程与修复版，再逐项对照主体裁剪、池化、SigLIP2、DINOv3、Qwen精排；一次改一项以归因。
4. 指标：Recall@1/5/20、自动同款判定precision/误报率、库外拒识、不同规格误判率、人工待确认率、耗时和内存。不用“搜自身首位5/5”证明真实准确。
5. 推荐流程：原图+主体裁剪 → 双路Top30召回 → Top10–20组精排 → 高可信规格核验 → 同SKU/同主体但规格不同/形近候选/无法判断。
6. 2430图规模可先精确向量搜索；引入FAISS只会改善索引效率，本身不会修复语义错配。不要为了框架复杂度跳过模型和标注验证。
7. 保留现有线上版本；先用独立测试分支、本地服务和固定样本验收，再决定上线，不宣称尚未测出的提升。

## 访问与测试限制

已从GitHub连接读取本分支源码。README中的workbuddy.host地址经网页读取失败，本机HTTP读取遇到TLS错误；本次没有完成在线交互上传复现，因此不能证明线上当前运行的脚本与本分支完全一致。需以实际部署资源版本及真实失败样本再核对。
