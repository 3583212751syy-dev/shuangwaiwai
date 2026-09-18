@echo off
chcp 65001 >nul
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo 请右键此文件，选择"以管理员身份运行"。
  pause
  exit /b 1
)
echo [1/3] 清理 NVIDIA 驱动下载缓存...
if exist "C:\ProgramData\NVIDIA Corporation\Downloader" (
  del /f /s /q "C:\ProgramData\NVIDIA Corporation\Downloader\*" >nul 2>&1
  for /d %%d in ("C:\ProgramData\NVIDIA Corporation\Downloader\*") do rd /s /q "%%d" 2>nul
)
echo [2/3] 清理 C:\Windows\Temp ...
if exist "C:\Windows\Temp" (
  del /f /s /q "C:\Windows\Temp\*" >nul 2>&1
  for /d %%d in ("C:\Windows\Temp\*") do rd /s /q "%%d" 2>nul
)
echo [3/3] 清理 Windows 更新缓存...
net stop wuauserv /y >nul 2>&1
net stop bits /y >nul 2>&1
if exist "C:\Windows\SoftwareDistribution\Download" (
  del /f /s /q "C:\Windows\SoftwareDistribution\Download\*" >nul 2>&1
  for /d %%d in ("C:\Windows\SoftwareDistribution\Download\*") do rd /s /q "%%d" 2>nul
)
net start bits >nul 2>&1
net start wuauserv >nul 2>&1
echo.
echo 完成！可关闭本窗口。
pause >nul
