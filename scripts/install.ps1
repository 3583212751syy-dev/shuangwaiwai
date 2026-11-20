param([string]$Root = 'D:\图片AI',[switch]$SkipModels)
$ErrorActionPreference='Stop'
$SourceApp = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'env.ps1') -Root $Root
Write-Host ('安装目录：' + $Root) -ForegroundColor Green
$free=(Get-PSDrive D).Free
if ($free -lt 60GB) { throw 'D 盘可用空间不足 60GB，请先清理或调整下载计划。' }
try { $ram=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; Write-Host ('系统内存：{0:N1} GB' -f ($ram/1GB)); if ($ram -lt 30GB) { Write-Warning '当前量化工作流需要大量内存卸载；建议 32GB 以上系统内存。低于此容量可能非常慢或无法出图。' } } catch { Write-Warning '无法读取系统内存，安装后需做实际出图检查。' }
$smi=Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $smi) { $candidate='C:\Windows\System32\nvidia-smi.exe'; if (Test-Path -LiteralPath $candidate) { $smi=Get-Item -LiteralPath $candidate } }
if ($smi) { $smiPath=if($smi.Source){$smi.Source}else{$smi.FullName}; & $smiPath --query-gpu=name,memory.total,driver_version --format=csv,noheader } else { Write-Warning '未找到 nvidia-smi。请先安装 NVIDIA 官方驱动；本程序不会替你修改显卡驱动。' }
$manifest=Get-Content -LiteralPath (Join-Path $SourceApp 'downloads.json') -Raw -Encoding UTF8 | ConvertFrom-Json
function Get-VerifiedFile($item,[string]$destination) {
 New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
 if (Test-Path -LiteralPath $destination) { if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $item.sha256) { Write-Host ('已校验，跳过：'+$item.file); return } else { throw ('现有文件校验失败，已保留原文件。请手动检查后移走再重新安装：'+$destination) } }
 $part=$destination+'.part'
 Write-Host ('下载：'+$item.file) -ForegroundColor Cyan
 $curlArgs=@('--fail','--location','--retry','3','--connect-timeout','30','--output',$part)
 if (Test-Path -LiteralPath $part) { $curlArgs+=@('--continue-at','-') }
 & curl.exe @curlArgs $item.url
 if ($LASTEXITCODE -ne 0) { throw ('下载失败，可重跑继续下载：'+$item.file) }
 $hash=(Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash.ToLowerInvariant()
 if ($hash -ne $item.sha256) { throw ('文件校验不匹配，停止安装；未执行该文件：'+$item.file) }
 Move-Item -LiteralPath $part -Destination $destination
}
foreach($kind in @('node','seven','comfy','ollama','cloudflared')) {
 $item=$manifest.assets | Where-Object kind -eq $kind
 $dest=Join-Path $Root ('downloads\'+$item.file)
 Get-VerifiedFile $item $dest
 if ($kind -eq 'node' -and -not (Test-Path -LiteralPath $NodeExe)) { Expand-Archive -LiteralPath $dest -DestinationPath (Join-Path $Root 'tools') -Force }
 if ($kind -eq 'seven') { Copy-Item -LiteralPath $dest -Destination (Join-Path $Root 'tools\7zr.exe') -Force }
 if ($kind -eq 'comfy' -and -not (Test-Path -LiteralPath $PythonExe)) { & (Join-Path $Root 'tools\7zr.exe') x $dest ('-o'+(Join-Path $Root 'tools')) '-y'; if ($LASTEXITCODE -ne 0) { throw 'ComfyUI 解压失败' } }
 if ($kind -eq 'ollama' -and -not (Test-Path -LiteralPath (Join-Path $Root 'tools\ollama\ollama.exe'))) { Expand-Archive -LiteralPath $dest -DestinationPath (Join-Path $Root 'tools\ollama') -Force }
 if ($kind -eq 'cloudflared') { Copy-Item -LiteralPath $dest -Destination (Join-Path $Root 'tools\cloudflared.exe') -Force }
}
if (-not (Test-Path -LiteralPath $NodeExe)) { throw 'Node.js 安装不完整' }
New-Item -ItemType Directory -Path $AppRoot -Force | Out-Null
if ([IO.Path]::GetFullPath($SourceApp) -ne [IO.Path]::GetFullPath($AppRoot)) {
 foreach($name in @('lib','web','scripts','test','docs','workflows','vendor','package.json','package-lock.json','server.mjs','downloads.json','README.md','USER_REQUIREMENTS.md','NOTICE.md')) { $src=Join-Path $SourceApp $name; if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination $AppRoot -Recurse -Force } }
}
$npm=Join-Path $NodeDirectory 'node_modules\npm\bin\npm-cli.js'
Push-Location $AppRoot
try { & $NodeExe $npm install --omit=dev --no-audit --no-fund; if ($LASTEXITCODE -ne 0) { throw '网站依赖安装失败' } } finally { Pop-Location }
$customNodes=Join-Path $ComfyRoot 'ComfyUI\custom_nodes'
$pluginTarget=Join-Path $customNodes 'ComfyUI-GGUF'
if (-not (Test-Path -LiteralPath (Join-Path $pluginTarget 'nodes.py'))) {
 $vendor=Join-Path $AppRoot 'vendor\ComfyUI-GGUF.zip'
 if (-not (Test-Path -LiteralPath $vendor)) { throw '安装包缺少经过校验的 ComfyUI-GGUF 源码归档' }
 $sourceHash=(Get-FileHash -LiteralPath $vendor -Algorithm SHA256).Hash.ToLowerInvariant()
 if ($sourceHash -ne $manifest.ggufArchiveSHA256) { throw 'GGUF 插件归档校验失败' }
 $extract=Join-Path $Root ('temp\gguf-'+[Guid]::NewGuid().ToString('N'))
 Expand-Archive -LiteralPath $vendor -DestinationPath $extract
 $inner=Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
 Copy-Item -LiteralPath $inner.FullName -Destination $pluginTarget -Recurse
}
& $PythonExe -s -m pip install -r (Join-Path $pluginTarget 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'GGUF 依赖安装失败' }
$modelRoot=Join-Path $ComfyRoot 'ComfyUI\models'
foreach($kind in @('diffusion_models','text_encoders','vae','loras')) {
 $destDir=Join-Path $Root ('models\'+$kind)
 New-Item -ItemType Directory -Path $destDir -Force | Out-Null
 if (-not $SkipModels) { $item=$manifest.assets | Where-Object kind -eq $kind; Get-VerifiedFile $item (Join-Path $destDir $item.file) }
}
$yaml="mx_studio:`n  base_path: $($Root.Replace('\','/'))/models`n  diffusion_models: diffusion_models`n  unet: diffusion_models`n  text_encoders: text_encoders`n  clip: text_encoders`n  vae: vae`n  loras: loras`n"
[IO.File]::WriteAllText((Join-Path $ComfyRoot 'ComfyUI\extra_model_paths.yaml'),$yaml,(New-Object Text.UTF8Encoding($false)))
Copy-Item -Path (Join-Path $AppRoot 'workflows\*') -Destination (Join-Path $Root 'workflows') -Recurse -Force
Copy-Item -Path (Join-Path $AppRoot 'docs\*') -Destination (Join-Path $Root 'docs') -Recurse -Force
Copy-Item -LiteralPath (Join-Path $AppRoot 'USER_REQUIREMENTS.md') -Destination (Join-Path $Root 'docs\用户偏好与需求.md') -Force
if (-not $SkipModels) {
 $ollama=Join-Path $Root 'tools\ollama\ollama.exe'
 $ready=$false;try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null;$ready=$true } catch {}
 if (-not $ready) { Start-Process -FilePath $ollama -ArgumentList 'serve' -WorkingDirectory (Split-Path $ollama -Parent) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Root 'logs\ollama-install.out.log') -RedirectStandardError (Join-Path $Root 'logs\ollama-install.err.log') | Out-Null; Start-Sleep -Seconds 3 }
 & $ollama pull qwen2.5vl:3b
 if ($LASTEXITCODE -ne 0) { throw '识别模型下载失败；其他已下载文件保留，可重跑安装' }
}
if (-not (Test-Path -LiteralPath (Join-Path $Root 'data\owner.json'))) { & (Join-Path $AppRoot 'scripts\set-account.ps1') -Root $Root }
# Only the current Windows user, Administrators and SYSTEM can read secrets and projects.
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
foreach($private in @('data','projects')) { & icacls.exe (Join-Path $Root $private) '/inheritance:r' '/grant:r' ('*'+$sid+':(OI)(CI)F') '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' '/T' '/Q' | Out-Null; if($LASTEXITCODE -ne 0){throw '本地数据权限设置失败，请勿启动远程访问'} }
$shell=New-Object -ComObject WScript.Shell
$desktop=[Environment]::GetFolderPath('Desktop')
foreach($item in @(@('图片AI工作室','start.ps1'),@('图片AI远程访问','remote.ps1'),@('停止图片AI服务','stop.ps1'))) {
 $link=$shell.CreateShortcut((Join-Path $desktop ($item[0]+'.lnk')))
 $link.TargetPath=Join-Path $PSHOME 'powershell.exe'
 $link.Arguments='-NoProfile -ExecutionPolicy Bypass -File "'+(Join-Path $AppRoot ('scripts\'+$item[1]))+'" -Root "'+$Root+'"'
 $link.WorkingDirectory=$AppRoot;$link.WindowStyle=7;$link.Save()
}
Write-Host '安装完成。双击桌面“图片AI工作室”启动；远程地址请运行“图片AI远程访问”。' -ForegroundColor Green
Write-Host '注意：尚需在本机验证模型是否能正常生成；8GB 显存不代表量化模型全部装入显存。'
