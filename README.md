# shuangwaiwai - 双接口项目

## 环境配置

### 开发工具链 (E 盘)
| 工具 | 路径 | 版本 |
|------|------|------|
| Node.js | E:\node.exe | v24.18.0 |
| Git | E:\Git\ | 2.47.1 |
| Codex CLI | E:\npm-global\ | 0.145.0 |
| VS Code | E:\Microsoft VS Code\ | 最新 |
| Codex 扩展 | VS Code 内 | v26.721.41059 |

### 代理配置
- 代理工具: GW (D:\eagle\gw\GW.exe)
- 代理端口: http://127.0.0.1:7897
- 当前节点: 香港 (Vmess, 24ms)

### Git 远程仓库
- 平台: GitHub
- 账号: 3583212751syy-dev
- 仓库: shuangwaiwai (Private)
- SSH: git@github.com:3583212751syy-dev/shuangwaiwai.git

## 快速开始

```powershell
# 设置代理
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"

# 克隆仓库
git clone git@github.com:3583212751syy-dev/shuangwaiwai.git

# 使用 Codex
codex
```
