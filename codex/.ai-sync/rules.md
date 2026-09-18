---
# AI 协同规则

## 工作流（每次对话开始前必须执行）

1. **同步状态**：读取 .ai-sync/board.json，了解当前任务分布和进度
2. **申领导向**：如果要操作的文件已被其他 AI 标记为工作中，先检查对方是否已释放
3. **更新进度**：完成任务或阶段后，更新 oard.json 中自己的条目
4. **变更通报**：Push 后 CI 自动生成变更摘要并更新 board.json

## 核心文件

| 文件 | 用途 |
|---|---|
| .ai-sync/board.json | 实时任务看板，所有 AI 共享 |
| .ai-sync/rules.md | 本规则文件 |
| .github/workflows/ai-sync.yml | 自动化中枢，push时触发 |

## board.json 结构

- ctive_tasks: 各 AI 当前正在做的任务 {ai_name, task, files_locked, started_at}
- completed_tasks: 已完成任务 {ai_name, task, files_changed, completed_at}
- last_sync: 上次同步时间
- warnings: 冲突预警

## 各 AI 身份标识

| AI 名称 | board.json 中的 ai_name |
|---|---|
| Claude Code | claude-code |
| Wokebuddy | wokebuddy |
| Trae Code | trae-code |
| 扣子 | kouzi |
| Marvis | marvis |
| Codex | codex |

## 冲突处理

- 如果 board.json 中某文件已被锁定（files_locked），其他 AI 必须等待或与对方协商
- 同一文件同时只能被一个 AI 锁定
- 任务完成后立即释放锁定

---
