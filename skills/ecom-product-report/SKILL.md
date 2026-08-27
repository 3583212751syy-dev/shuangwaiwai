---
name: ecom-product-report
description: 电商产品图片分析报表生成器 v2.0（工具化）。接收产品图片路径 + 结构化参数，自动核查平台资质、记忆库去重、采集竞品数据（AI搜索或内置库）、计算利润矩阵，生成 5-sheet 专业选品分析报表（产品与竞品图/利润明细/竞品全景/决策结论/供应链物流）。核心约束：负利润 / 需资质 / 重复使用 任一命中即停止出表并输出原因报告。触发词：选品报表、产品分析报表、利润分析、专业版选品分析、电商报表、传图出报表。
agent_created: true
version: 2.0.0
---

# 电商产品图片分析报表生成器（ecom-product-report）v2.0

## 定位
根据用户上传的产品图片（产品图 / 1688采购图 / 平台售卖图），生成专业选品数据报表（.xlsx，5-sheet），支持多图同页对照、物流三选、平台多选、记忆库去重与三大约束自动停止。当前会话模型若无法读图，则接受文字结构化参数作为输入，并在报表中标注"数据来源：用户描述 + 行业估算"。

## 核心工具
`scripts/ecom_report_tool.py`（openpyxl + sqlite3，零第三方依赖除 openpyxl）

```bash
# 基本用法：产品图+采购图+售卖图 嵌入同一 sheet
python ecom_report_tool.py \
  --product "产品图.jpg" --purchase-image "1688采购图.png" --platform-image "平台售卖图.png" \
  --product-name "浴室壁挂式脏衣篮HSB808" \
  --platform shopee-id --logistics mixed --purchase 4.00 --benchmark 109000 --auto-search

# 图片目录自动匹配（文件名含 product/产品 等）
python ecom_report_tool.py --image-dir "D:\产品图" --platform lazada-id --logistics sea

# 注入 AI 搜索结果（自动搜索）
python ecom_report_tool.py --product-name "XXX" --platform shopee-id --search-json "竞品数据.json"

# 配置文件模式
python ecom_report_tool.py --config config.json
```

## 参数速查（CLI）
| 参数 | 说明 | 默认 |
|---|---|---|
| --product / --purchase-image / --platform-image | 三图路径（嵌入同 sheet 对应列） | 无 |
| --image-dir | 图片目录，按文件名自动匹配（product/产品、1688/采购、shopee/售卖） | 无 |
| --platform | shopee-id(本土) / shopee-id-cb(跨境) / lazada-id / tiktok-id / amazon-us | shopee-id |
| --logistics | sea 海运 / air 空运 / mixed 海运+空运结合 | sea |
| --purchase | 采购价 CNY | 4.00 |
| --benchmark | 基准售价（当地货币） | 109000 |
| --ad | 广告率%，逗号分隔 | 0,5,10,15 |
| --discount | 降价阶梯%，逗号分隔 | 5,10,15,20,25 |
| --auto-search | 内置平台价格库自动匹配竞品 | 关 |
| --search-json | AI 搜索结果 JSON 注入（competitors_1688 / competitors_platform） | 无 |
| --force | 跳过记忆库去重 | 关 |
| --output | 输出 xlsx 路径 | 桌面/产品名_平台_选品分析报表.xlsx |

## 平台费率（内置）
| 平台 | 佣金 | 交易费 | 支付 | 货币/汇率 |
|---|---|---|---|---|
| shopee-id 本土 | 5% | 2% | 1% | IDR 0.0003782 |
| shopee-id 跨境 | 8% | 6% | 1% | IDR 0.0003782 |
| lazada-id | 6% | 2% | 1% | IDR 0.0003782 |
| tiktok-id | 2% | 2% | 1% | IDR 0.0003782 |
| amazon-us | 15% | 1% | 1% | USD 7.20 |

## 三大约束（任一命中 → 停止出表，输出"产品名_停止原因报告.txt"）
1. **负利润**：主力场景（半价档+常规广告+所选物流）单件净利 ≤ 0 → 停止
2. **需资质**：产品名命中强制资质规则（SNI 食品接触塑料 7323:2008 / BPOM 化妆品药品 / SNI 电子产品 / SNI 儿童用品 / Halal 食品 / 平台禁售词）→ 停止
3. **重复使用**：记忆库（ecom_report_memory.db，按产品名 hash + 平台）已存在 → 停止（--force 可覆盖）

## 记忆库
- 位置：`scripts/ecom_report_memory.db`（SQLite 自动创建）
- 表 reports：product_hash, product_name, platform, logistics, purchase, benchmark, result, note, file_path, created_at
- 作用：去重判断 + 历史追踪（list_reports 可查最近10条）

## 工作流
1. **Phase 0 前置筛查**：资质核查 → 记忆库去重 → 任一命中立即停止（不生成报表）
2. **Phase 1 参数澄清**：AskUserQuestion 一次一问（平台→采购价→店铺→物流→阶梯→广告→基准售价→输出），复用 ecom-product-report v1 的 13 问清单
3. **Phase 2 数据采集**：AI 用 WebSearch 搜平台竞品 → 组装 competitors_1688 / competitors_platform 写入 JSON → --search-json 注入；或 --auto-search 用内置价格库
4. **Phase 3 计算**：单件净利 = 售价×汇率×(1-佣金-交易-支付-广告) - 采购 - 国际运费 - 本地配送 - 包装；物流 mixed 时海运基准 + 空运补货参考
5. **Phase 4 生成**：5-sheet（产品与竞品图[3图对照] / 利润明细[降价×广告矩阵] / 竞品全景 / 决策结论[三重检查+定价] / 供应链物流[海运空运对比]）
6. **Phase 5 交付**：result.json + 明确请用户验收；同步到 shuangwaiwai git

## 注意
- 印尼强制资质仅限：食品接触塑料(SNI 7323:2008)、食品/化妆品(BPOM)、电子产品/儿童用品(SNI)、药品；衣物/家居收纳类无强制门槛，软合规必做（印尼语标签、PT PMA、原产国标注）
- 图片无法读取时：接受文字结构化参数，报表标注"数据来源：用户描述+行业估算"，提示切多模态后补图
- 嵌入图片 openpyxl 以像素设置 width/height，实际保存为 EMU（110px=1,047,750 EMU），Excel 显示正确
