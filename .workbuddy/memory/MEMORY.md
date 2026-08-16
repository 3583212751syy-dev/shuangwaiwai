# Project Memory

## ColorForge（图片改色工具）
- 主文件：`colorforge-pro.html`（v9.4，单文件，双模式）
- 全图模式：目标色直接映射（结果图主色 = 所选目标色），仅极低饱和/极暗/极亮像素被保护，非中性像素严格着色为目标色；预设主题 30 个
- 单色模式：点图取源色 → 选目标色，容差+羽化，保留纹理
- 预览：主画布 + 全部预览网格（全局 30 预设 / 单色 30 常用色），主/缩略图均使用 `image-rendering: high-quality` 与 DPR 感知的缩略图采样，避免拉伸模糊
- 本地依赖：`colorforge-libs/vendor/{jszip,chroma,color-thief}.min.js`
- 同步路径：`\Os-20250608fqtb\双接口\workbuddy\`
- 可复用技能：`image-color-tool`（用户级）
