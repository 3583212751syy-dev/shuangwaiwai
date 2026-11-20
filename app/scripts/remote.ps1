param([string]$Root='D:\图片AI',[switch]$NoBrowser)
. (Join-Path $PSScriptRoot 'env.ps1') -Root $Root
& (Join-Path $PSScriptRoot 'start.ps1') -Root $Root -NoBrowser
$ready=$false
for($i=0;$i -lt 20;$i++){try{Invoke-RestMethod 'http://127.0.0.1:3188/api/session' -TimeoutSec 2 | Out-Null;$ready=$true;break}catch{Start-Sleep -Seconds 1}}
if(-not $ready){throw '网站未启动，远程连接没有开放。请查看 logs/web.err.log。'}
$processFile=Join-Path $Root 'data\processes.json'
$managed=@();if(Test-Path -LiteralPath $processFile){$managed=@(Get-Content -LiteralPath $processFile -Raw -Encoding UTF8 | ConvertFrom-Json)}
$previous=$managed | Where-Object name -eq 'tunnel' | Select-Object -First 1
if($previous){$p=Get-Process -Id $previous.pid -ErrorAction SilentlyContinue;if($p -and $p.StartTime.ToUniversalTime().ToString('o') -eq $previous.started){$r=Get-Content -LiteralPath (Join-Path $Root 'data\remote.json') -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host $r.url;if(-not $NoBrowser){Start-Process $r.url};return}}
$exe=Join-Path $Root 'tools\cloudflared.exe';if(-not(Test-Path -LiteralPath $exe)){throw 'cloudflared 未安装，请重跑安装程序'}
$out=Join-Path $Root 'logs\tunnel.out.log';$err=Join-Path $Root 'logs\tunnel.err.log'
$process=Start-Process -FilePath $exe -ArgumentList 'tunnel --url http://127.0.0.1:3188 --no-autoupdate' -WorkingDirectory (Join-Path $Root 'tools') -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
$managed=@($managed | Where-Object name -ne 'tunnel')+@(@{name='tunnel';pid=$process.Id;started=$process.StartTime.ToUniversalTime().ToString('o');exe=$exe})
$managed | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $processFile -Encoding UTF8
for($i=0;$i -lt 45;$i++){
 $text='';foreach($log in @($out,$err)){if(Test-Path -LiteralPath $log){$text+=Get-Content -LiteralPath $log -Raw -ErrorAction SilentlyContinue}}
 $match=[regex]::Match($text,'https://[a-z0-9-]+\.trycloudflare\.com')
 if($match.Success){$remote=@{url=$match.Value;updatedAt=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds();temporary=$true};$json=$remote | ConvertTo-Json;[IO.File]::WriteAllText((Join-Path $Root 'data\remote.json'),$json,(New-Object Text.UTF8Encoding($false)));Write-Host ('当次远程地址：'+$match.Value) -ForegroundColor Green;Write-Host '地址重启后可能变化。需要账号密码登录，电脑需保持运行。';if(-not $NoBrowser){Start-Process $match.Value};return}
 if($process.HasExited){throw '远程连接启动失败，请查看 logs/tunnel.err.log'}
 Start-Sleep -Seconds 1
}
throw '暂未取得远程地址，请查看 logs/tunnel.err.log；不要反复启动。'
