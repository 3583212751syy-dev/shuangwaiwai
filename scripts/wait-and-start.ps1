param([int]$DownloaderPid=0,[string]$Root='D:\图片AI',[int]$TimeoutMinutes=180)
$ErrorActionPreference='Continue'
$log=Join-Path $Root 'logs\wait-and-start.log'
function W($m){ Add-Content -LiteralPath $log -Value ('['+(Get-Date).ToString('HH:mm:ss')+'] '+$m) -Encoding UTF8 }
W ('守护启动，等待下载进程 PID='+$DownloaderPid)
$deadline=(Get-Date).AddMinutes($TimeoutMinutes)
while((Get-Date) -lt $deadline){
  if($DownloaderPid -gt 0){
    $p=Get-Process -Id $DownloaderPid -ErrorAction SilentlyContinue
    if(-not $p){ W '下载进程已退出'; break }
  }
  Start-Sleep -Seconds 20
}
Start-Sleep -Seconds 5
# 校验主模型完整性
$main=Join-Path $Root 'models\diffusion_models\qwen-image-edit-2511-Q4_K_M.gguf'
if(Test-Path -LiteralPath $main){
  $gb=[math]::Round((Get-Item -LiteralPath $main).Length/1GB,2)
  W ('主模型大小 '+$gb+' GB')
} else { W '主模型文件不存在' }
try{
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'scripts\stop.ps1') -Root $Root 2>&1 | Out-String | ForEach-Object { W ('stop: '+$_.Trim()) }
}catch{ W ('stop 异常: '+$_.Exception.Message) }
Start-Sleep -Seconds 3
try{
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'scripts\start.ps1') -Root $Root -NoBrowser 2>&1 | Out-String | ForEach-Object { W ('start: '+$_.Trim()) }
}catch{ W ('start 异常: '+$_.Exception.Message) }
W '守护结束'
