---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 075210f9385b33e628b87fbddda4ab41_1a186b068a5811f1be80525400f8a581
    ReservedCode1: +YYqdDd9m9q+BG7OExDsEMU3XcHdDT1MIFXCzsTTHA4HPm3U4yZ87RxfiZH3+HXF5BLdvs6tWwhdYvgSMCxVkFq2/43CDD1Lny+ZUV4qBEZlia5uh+2OzEGGd7EuoVPd6xmlUQG+ltwh7obsw9lPCN33xvfPktwOoWGP1l0YwaJ3kxoiC6kDzg2loeQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 075210f9385b33e628b87fbddda4ab41_1a186b068a5811f1be80525400f8a581
    ReservedCode2: +YYqdDd9m9q+BG7OExDsEMU3XcHdDT1MIFXCzsTTHA4HPm3U4yZ87RxfiZH3+HXF5BLdvs6tWwhdYvgSMCxVkFq2/43CDD1Lny+ZUV4qBEZlia5uh+2OzEGGd7EuoVPd6xmlUQG+ltwh7obsw9lPCN33xvfPktwOoWGP1l0YwaJ3kxoiC6kDzg2loeQ=
---

# Codex CLI 技能清单

## 安装与配置

Codex CLI 通过 npm 全局安装到 E:\npm-global\，使用代理访问 OpenAI API。

```bash
npm install -g @openai/codex
```

## 环境变量

需要配置以下环境变量才能联网:
```
HTTP_PROXY=http://127.0.0.1:7897
HTTPS_PROXY=http://127.0.0.1:7897
OPENAI_API_KEY=<your-key>
```

## 核心能力

### 代码生成与编辑
- 自然语言描述生成完整代码文件
- 对已有代码进行修改和重构
- 多文件项目脚手架搭建

### 代码审查
- 逐行分析代码逻辑
- 指出潜在 bug 和性能问题
- 提供优化建议

### 调试支持
- 错误信息分析
- 堆栈追踪解读
- 修复方案建议

### 项目分析
- 理解代码库结构
- 生成项目文档
- 依赖关系梳理

## VS Code 集成

安装 Codex 扩展 (v26.721.41059) 后:
- `Ctrl+Shift+I` 打开 Codex 面板
- 选中代码后可直接提问
- 支持内联补全和对话式编码

## 使用技巧

1. **明确上下文**: 提供文件路径和相关代码片段
2. **分步请求**: 复杂任务拆成多个小步骤
3. **指定输出格式**: 明确期望的代码风格和规范
4. **利用 Git 历史**: 让 Codex 参考最近的 commit 理解项目演进
*（内容由AI生成，仅供参考）*
