param([string]$Root = 'D:\图片AI')
$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root).TrimEnd('\')
if (-not $Root.StartsWith('D:\',[StringComparison]::OrdinalIgnoreCase) -or $Root.Length -lt 6) { throw '安装目录必须是 D 盘下的专用子目录。' }
$env:MX_ROOT = $Root
$env:OLLAMA_MODELS = Join-Path $Root 'models\ollama'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:HF_HOME = Join-Path $Root 'cache\huggingface'
$env:TORCH_HOME = Join-Path $Root 'cache\torch'
$env:PIP_CACHE_DIR = Join-Path $Root 'cache\pip'
$env:npm_config_cache = Join-Path $Root 'cache\npm'
$env:TEMP = Join-Path $Root 'temp'
$env:TMP = $env:TEMP
foreach ($dir in @('data','projects','models','models\ollama','downloads','tools','logs','cache','temp','workflows','docs')) { New-Item -ItemType Directory -Path (Join-Path $Root $dir) -Force | Out-Null }
$NodeDirectory = Join-Path $Root 'tools\node-v24.22.3-win-x64'
$NodeExe = Join-Path $NodeDirectory 'node.exe'
$env:PATH = $NodeDirectory + ';' + $env:PATH
$ComfyRoot = Join-Path $Root 'tools\ComfyUI_windows_portable'
$PythonExe = Join-Path $ComfyRoot 'python_embeded\python.exe'
$AppRoot = Join-Path $Root 'app'
