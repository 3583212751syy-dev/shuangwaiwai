# Project Memory

## ColorForge（图片改色工具）
- 主文件：`colorforge-pro.html`（v9.6，单文件，双模式）
- 全图模式：目标色直接映射（结果图主色 = 所选目标色），中性保护（白/黑/灰不染色）；支持「自然」与「电影」两种风格
  - 电影风格（cinematic）：暗部/亮部双端点渐变映射 + 保留原图亮度纹理 + 鲜艳度滑块，接近 PS 调色/3D LUT 效果
  - 清晰度（clarity，v9.6）：亮度通道 unsharp 局部对比增强（boxBlurLum+detail 缓存），只动亮度不碰色相，改色同时保留/提升清晰度
  - 关键算法：hueLerp 色相最短路径、低饱和像素直接采用映射色相、softLight 混合辅助、clarity detail 叠加
- 预设主题 30 个（27 个彩色主题全部 cinematic + vibrance 参数）
- 单色模式：点图取源色 → 选目标色，容差+羽化，保留纹理
- 预览：主画布 + 全部预览网格（全局 30 预设 / 单色 30 常用色），`image-rendering: high-quality` + DPR 感知缩略图
- 参考库：`colorforge-libs/{grader-npm,AlloyImage,glfx.js}/`（3D LUT / PS 混合模式学习）
- 本地依赖：`colorforge-libs/vendor/{jszip,chroma,color-thief}.min.js`
- 同步路径：`\Os-20250608fqtb\双接口\workbuddy\`
- 可复用技能：`image-color-tool`（用户级）
