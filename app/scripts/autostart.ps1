param([string]$Root='D:\图片AI',[ValidateSet('enable','disable','status')][string]$Action='status')
. (Join-Path $PSScriptRoot 'env.ps1') -Root $Root
$startup=[Environment]::GetFolderPath('Startup')
$linkPath=Join-Path $startup 'MXProductStudio.lnk'
if($Action -eq 'enable'){
 $shell=New-Object -ComObject WScript.Shell;$link=$shell.CreateShortcut($linkPath)
 $link.TargetPath=Join-Path $PSHOME 'powershell.exe';$link.Arguments='-NoProfile -ExecutionPolicy Bypass -File "'+(Join-Path $AppRoot 'scripts\remote.ps1')+'" -Root "'+$Root+'" -NoBrowser';$link.WorkingDirectory=$AppRoot;$link.WindowStyle=7;$link.Save()
}
if($Action -eq 'disable' -and (Test-Path -LiteralPath $linkPath)){Remove-Item -LiteralPath $linkPath}
@{enabled=(Test-Path -LiteralPath $linkPath);startsAfterWindowsLogin=$true} | ConvertTo-Json -Compress
