param([string]$Root='D:\图片AI',[switch]$NoBrowser)
. (Join-Path $PSScriptRoot 'env.ps1') -Root $Root
$processFile=Join-Path $Root 'data\processes.json'
$managed=@();if(Test-Path -LiteralPath $processFile){$managed=@(Get-Content -LiteralPath $processFile -Raw -Encoding UTF8 | ConvertFrom-Json)}
function Available([string]$url){try{Invoke-RestMethod $url -TimeoutSec 2 | Out-Null;return $true}catch{return $false}}
function Start-Managed([string]$name,[string]$exe,[string]$arguments,[string]$work){
 if(-not(Test-Path -LiteralPath $exe)){throw ('组件未安装：'+$exe)}
 $p=Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $work -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Root ('logs\'+$name+'.out.log')) -RedirectStandardError (Join-Path $Root ('logs\'+$name+'.err.log'))
 $script:managed=@($script:managed | Where-Object name -ne $name)+@(@{name=$name;pid=$p.Id;started=$p.StartTime.ToUniversalTime().ToString('o');exe=$exe})
 $script:managed | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $processFile -Encoding UTF8
}
if(-not(Test-Path -LiteralPath (Join-Path $Root 'data\owner.json'))){throw '请先运行安装程序设置账号，暂不启动远程服务。'}
if(-not(Available 'http://127.0.0.1:11434/api/tags')){Start-Managed 'ollama' (Join-Path $Root 'tools\ollama\ollama.exe') 'serve' (Join-Path $Root 'tools\ollama')}
if(-not(Available 'http://127.0.0.1:8188/system_stats')){
 $gpuArgs=' --lowvram --cpu-vae --reserve-vram 1'
 $smi=Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
 if(-not $smi){ if(Test-Path -LiteralPath 'C:\Windows\System32\nvidia-smi.exe'){ $smi=$true } }
 if(-not $smi){ $gpuArgs=' --cpu --use-split-cross-attention --disable-smart-memory' }
 $args='-s "'+(Join-Path $ComfyRoot 'ComfyUI\main.py')+'" --windows-standalone-build --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-assets'+$gpuArgs+' --input-directory "'+(Join-Path $Root 'projects\comfy-input')+'" --output-directory "'+(Join-Path $Root 'projects\comfy-output')+'" --temp-directory "'+(Join-Path $Root 'temp\comfy')+'"'
 Start-Managed 'comfy' $PythonExe $args $ComfyRoot
}
if(-not(Available 'http://127.0.0.1:3188/api/session')){
 $lockPath=Join-Path $Root 'data\server.lock'
 if(Test-Path -LiteralPath $lockPath){$old=Get-Content -LiteralPath $lockPath -Raw;$running=Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue;if($running){throw '发现正在运行的服务锁，请先使用停止程序，不启动重复实例。'};Remove-Item -LiteralPath $lockPath}
 Start-Managed 'web' $NodeExe ('"'+(Join-Path $AppRoot 'server.mjs')+'"') $AppRoot
}
if(-not $NoBrowser){Start-Sleep -Seconds 2;Start-Process 'http://127.0.0.1:3188'}
Write-Host '后台服务已启动。网址：http://127.0.0.1:3188；首次模型加载可能需要较长时间。'
