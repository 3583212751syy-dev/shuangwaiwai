param([string]$Root='D:\图片AI')
. (Join-Path $PSScriptRoot 'env.ps1') -Root $Root
$processFile=Join-Path $Root 'data\processes.json'
if(-not(Test-Path -LiteralPath $processFile)){Write-Host '没有由本工具启动的服务记录。';return}
$managed=@(Get-Content -LiteralPath $processFile -Raw -Encoding UTF8 | ConvertFrom-Json)
foreach($record in ($managed | Sort-Object @{Expression={if($_.name -eq 'tunnel'){0}elseif($_.name -eq 'web'){1}else{2}}})){
 $p=Get-Process -Id $record.pid -ErrorAction SilentlyContinue
 if($p -and $p.StartTime.ToUniversalTime().ToString('o') -eq $record.started -and $p.Path -eq $record.exe){Stop-Process -Id $p.Id;Write-Host ('已停止：'+$record.name)}
}
Start-Sleep -Milliseconds 500
Remove-Item -LiteralPath $processFile -ErrorAction SilentlyContinue
$lock=Join-Path $Root 'data\server.lock'
if(Test-Path -LiteralPath $lock){$old=[int](Get-Content -LiteralPath $lock -Raw);if(-not(Get-Process -Id $old -ErrorAction SilentlyContinue)){Remove-Item -LiteralPath $lock}}
$remoteFile=Join-Path $Root 'data\remote.json';if(Test-Path -LiteralPath $remoteFile){Remove-Item -LiteralPath $remoteFile}
Write-Host '停止程序只处理本工具启动且身份匹配的进程，不会关闭其他 Python 或 Node 程序。'
