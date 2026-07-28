# 环境配置脚本
# 一键设置代理和 Git 配置

Write-Host "=== 设置系统代理 ==="
[System.Environment]::SetEnvironmentVariable("HTTP_PROXY", "http://127.0.0.1:7897", "User")
[System.Environment]::SetEnvironmentVariable("HTTPS_PROXY", "http://127.0.0.1:7897", "User")

Write-Host "=== 配置 Git 代理 ==="
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

Write-Host "=== 配置 npm 代理 ==="
npm config set proxy http://127.0.0.1:7897
npm config set https-proxy http://127.0.0.1:7897

Write-Host "=== 配置 Git SSH ==="
git config --global core.sshCommand "ssh -o StrictHostKeyChecking=accept-new"

Write-Host "`n环境配置完成!"
Write-Host "代理: http://127.0.0.1:7897"
Write-Host "Git 用户: 3583212751syy-dev"
