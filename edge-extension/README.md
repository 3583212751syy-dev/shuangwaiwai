# 选品排重助手 · Edge 插件

网页一键扫描：读取当前商品页的**主图、名称、规格描述、SKU**，用自有 SKU 集合判定是否**产品重复**。

## 安装（Edge）

1. 打开 `edge://extensions/`，开启「开发人员模式」。
2. 点「加载已解压的扩展程序」，选择本目录 `edge-extension`。
3. 工具栏固定插件，打开任意商品详情页（例：美客多墨西哥商品页），点插件图标 → 「一键扫描当前网页」。

## 判定规则

- 自有 SKU 集合内置 `sku-set.json`，共 1832 条：
  - 「产品排重」A 列「同行SKU」：1728 条（如 `MLM4405140516`、`MLM2011620355`）。
  - 「通用品」G 列「同行SKU」：合并去重（如 `MLM1409046617`、`MLM2226332683`）。
  - 两张表「主管审核备注」列补充，以及少量 Amazon ASIN。
- 扫描页面时从 JSON-LD、meta 标签、URL、页面文本提取 `MLM\d+` 与内部 SKU。
- 任一 SKU 命中集合 → 显示「产品重复（SKU 命中自有集合）」，并列出命中编号与来源行。
- 未命中 → 显示「SKU 未命中，不判为重复」；扫描不到 SKU → 「未识别到 SKU，无法判定」。
- SKU 归一化：`MLM-2850659521` 与 `MLM2850659521` 视为同一个；大小写不敏感。
- 额外 SKU 可在插件「SKU 集合设置」里手动追加（每行一个），不用改代码。

## 字段提取

- **名称**：JSON-LD `name` → `og:title` → `h1` → `document.title`。
- **图片**：只取商品图库容器与左侧竖排缩略图；跳过视频格、广告横幅、店铺 logo、相关推荐；按图片核心 ID 去重，避免同一张图的 srcset 多尺寸重复。DOM 抓不到时才回退 JSON-LD / `og:image`。
- **规格**：规格区/商品信息的 `li` 与 `tr`（覆盖 Mercado Libre、Shopee、Amazon、1688 常见结构），最多 40 条。
- **SKU**：JSON-LD `sku`/`mpn`/`productID`、meta `product:retailer_item_id`、URL、页面文本。
- **价格**：JSON-LD `offers.price` → `og:price:amount`。

## 与图片对比工具联动

「一键扫描当前网页」完成后会自动在后台打开图库工具（不抢前台焦点），把扫描出的商品图带过去检索，并把「网页商品图 / 图库匹配图」并列回填到插件面板；也可点「图片对比工具」重试。默认地址 `https://product-image-match.app.workbuddy.host/?sharecode=MTCP4KDYCY`（可在设置里改）。扫描结果仍可一键复制。

## 重新生成 SKU 集合

```bash
node tools-build-sku.mjs <产品排重A列.json> <通用品G列.json> <MLM_ID.json> sku-set.json
node -e "const fs=require('fs'),vm=require('vm');const c={module:{exports:{}},self:{}};vm.runInNewContext(fs.readFileSync('lib/sku.js','utf8'),c);global.SKUMatch=c.module.exports||c.self.SKUMatch;require('./test-extension.mjs')"
```

## 测试

8 项测试覆盖 SKU 归一化、文本提取、命中/未命中判定、模拟商品页 DOM 扫描，全部通过。

## 数据来源

金山文档「选品定价表.xlsx」：https://www.kdocs.cn/l/ch0h8m4dZ6ho
