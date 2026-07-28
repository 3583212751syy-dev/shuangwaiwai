---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 075210f9385b33e628b87fbddda4ab41_1971eba18a5811f1be80525400f8a581
    ReservedCode1: BUcGLbZDEC9oXx99OJ+kdWn5gFhYGpQfwVhL9e28WnLRLqDzU749AS5G0fyrfNyqWJLGlSl73jLztQYxXoAFnSeWYP+RXckTdliQgI3ZjYBN/6E1xkOVLIOYJ79Z1Tg25WPUQpd1QYDFvaV5jbnApuuP4/iFx6SgfxezKH8Zh2uxU4NbzEi4Fmi1tBM=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 075210f9385b33e628b87fbddda4ab41_1971eba18a5811f1be80525400f8a581
    ReservedCode2: BUcGLbZDEC9oXx99OJ+kdWn5gFhYGpQfwVhL9e28WnLRLqDzU749AS5G0fyrfNyqWJLGlSl73jLztQYxXoAFnSeWYP+RXckTdliQgI3ZjYBN/6E1xkOVLIOYJ79Z1Tg25WPUQpd1QYDFvaV5jbnApuuP4/iFx6SgfxezKH8Zh2uxU4NbzEi4Fmi1tBM=
---

# Bash on Windows - 完整配置记录

## 安装步骤

### 1. 安装 Git for Windows (2.47.1)
- 下载链接: https://git-scm.com/downloads/win
- 安装路径: E:\Git\
- 选择组件: Git Bash Here, Git GUI Here, Git LFS, Associate .sh

### 2. 配置 Git 用户
```bash
git config --global user.name "3583212751syy-dev"
git config --global user.email "3583212751syy-dev@users.noreply.github.com"
```

### 3. 生成 SSH 密钥
```bash
ssh-keygen -t ed25519 -C "3583212751syy-dev@users.noreply.github.com" -f ~/.ssh/id_ed25519
```

### 4. 配置 Git SSH
```bash
git config --global core.sshCommand "ssh -o StrictHostKeyChecking=accept-new"
```

### 5. 关联远程仓库
```bash
git remote add origin git@github.com:3583212751syy-dev/shuangwaiwai.git
```

## 常用 Git 命令

```bash
# 查看状态
git status

# 添加并提交
git add .
git commit -m "描述"

# 推送到远程
git push origin master

# 拉取更新
git pull origin master

# 查看远程配置
git remote -v

# 查看提交历史
git log --oneline
```

## 代理设置

当 Git 直连被墙时:
```bash
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

取消代理:
```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```
*（内容由AI生成，仅供参考）*
