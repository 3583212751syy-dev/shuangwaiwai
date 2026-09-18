# 茶叶电商图生成 · 模型技能手册（TEVOYATEA Rooibos Tea 100g）

> 沉淀自多轮迭代的经验。核心目标：用 AI 出图能力 + 程序化合成，稳定产出亚马逊风格、全英文、包装标签 100% 保真的电商主图/详情图。

## 1. 硬约束（用户反复强调，违反即重做）
- 包装袋形态（挂孔、拉链封口、底部弧度、倾斜角度、贴纸位置）**不许改变**。
- 标签内容按 `标签.png` **一字不差、一图不改**（含 100% NATURAL 徽章、NET WT. 100g、金色水滴 logo、花草边框）。
- **全英文、无中文**。
- 产品必须**站立正面（standing upright）**，用**真实袋子素材**，禁止 AI 直接重绘包装（会糊标签、变形）。
- 构图要像商业广告：产品为 hero、浅景深虚化背景、三分法、充足留白、光影聚焦。

## 2. 核心工作流（混合工艺，非纯 AI 出图）
1. **AI 出"无产品"底图**：ImageGen 图生图（hunyuan-image-v3.0-art），`input_fidelity=high`，prompt 必须强调 `absolutely no pouch, no packaging, no product, no labels, no readable text`，只生成版式/场景/背景。
2. **PIL 合成真实袋**：用**固定 rounded_rectangle 区域 + 阈值掩码**抠 `_jimeng_front.png`（去白底/阴影）；scipy 二值化精确找袋 bbox；袋内部强制填深 kraft `(155,105,65)` 让袋子从背景跳出。
3. **贴真标签**：`alpha_composite` 把 `_label.png` 覆盖 AI 标签占位区，`cover-mode≈1.05x` 做装裱边。
4. **PIL 叠加 crisp 英文排版**：`ImageDraw` 绘制标题/卖点胶囊/副标/SC 角章。**零 AI 生图文字**（AI 文字必乱码/变形）。圆形 icon 用红圆+白色粗体字母（如 C=冷/H=热），**不要用 PIL 线稿**（小尺寸不可读）。

## 3. 配色规范（design_brief.md §5.2）
| 用途 | 色值 |
|---|---|
| 标题棕 | `#C67B4B` |
| 绿（草本/有机） | `#4A7C4A` |
| 金（点缀） | `#D4A853` |
| 暖白（背景） | `#FDF9F3` |
| 色带黄 | `#FCD782` |

认证信息：产地写 **福建安溪**；SC `1143552406532`；标准 `GH/T 1091`。

## 4. 模型现状与选型
- **混元 ImageGen（默认）**：做"无产品底图"够用；做"产品摆拍"易糊包装 → 必须走上面混合工艺。
- **Gemini 3 Pro Image（nano-banana-pro）**：视觉营销质量最强，但免费档配额 `limit:0`，需绑卡付费才能用。
- **即梦**：参考图质量标杆，但本环境未接调用通道。

## 5. 常见坑（踩过的）
- **误判袋为 kraft 棕色**：实际 `_jimeng_front.png` 是**浅米色立袋**（背景白色不透明），之前多轮"白底板"瑕疵根因。修正：固定区域抠图 + 强制深 kraft 填内部。
- **奶油色大矩形溢出**：AI 标签占位比真标签大 → 用奶油色填充溢出。修正：仅比真标签大 10px 装裱边。
- **near_white 抠图把原袋踢掉** → 改用固定 rounded_rectangle 区域掩码。
- **占位袋背后留 AI 假阴影** → 同色面板 `paste((255,247,202,255), box)` 覆盖再放真袋。
- **掩码边界漏色** → 合成前 `expand 掩码 8px` 或二次覆盖底色。
- **Windows CRLF**：git 提示 `LF will be replaced by CRLF`，无害。

## 6. 文件索引
- `scripts/`：全部合成与抠图工具脚本（`_compose_*` 合成、`_extract_*`/`_remove_bg_*`/`_crop_*`/`_mask_*` 抠图裁剪、`_rebuild_*` 重建袋、`extract_xls.py` 抽表、`_round17_prompts.txt` 出图 prompt 模板）。
- `docs/design_brief.md`：完整配色与认证规范。
- `docs/codex_skills.md`：Codex CLI 模型/工具技能清单。
- `setup/setup_env.ps1`：Codex/OpenAI 环境配置脚本。
