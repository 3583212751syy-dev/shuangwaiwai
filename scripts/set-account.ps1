param([string]$Root='D:\图片AI')
. (Join-Path $PSScriptRoot 'env.ps1') -Root $Root
$username=Read-Host '设置登录账号（3–50 位英文字母、数字、下划线）'
$secure=Read-Host '设置密码（至少 12 个字符）' -AsSecureString
$again=Read-Host '再次输入密码' -AsSecureString
$ptr=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$ptrAgain=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($again)
try {
 $password=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
 $passwordAgain=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptrAgain)
 if($password -ne $passwordAgain){throw '两次输入的密码不一致'}
 $OutputEncoding=New-Object Text.UTF8Encoding($false)
 @{username=$username;password=$password} | ConvertTo-Json -Compress | & $NodeExe (Join-Path $AppRoot 'scripts\account.mjs')
 if($LASTEXITCODE -ne 0){throw '账号设置失败'}
} finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr);[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptrAgain);$password=$null;$passwordAgain=$null }
