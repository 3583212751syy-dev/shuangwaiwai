---
name: c-drive-cleanup
description: Windows C盘空间清理和优化工具。安全清理临时文件、更新缓存、回收站，并指导用户文件夹迁移到非系统盘。
---

# C盘空间清理技能

## 使用场景
- C盘空间不足（红色警告）
- 系统运行缓慢
- 需要释放磁盘空间

## 安全清理步骤

### 1. 清理临时文件（安全）
```powershell
# 清理用户临时文件
Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue

# 清理系统临时文件
Remove-Item -Path "C:\Windows\Temp\*" -Recurse -Force -ErrorAction SilentlyContinue
```

### 2. 清理Windows更新缓存（安全）
```powershell
Stop-Service -Name wuauserv -Force -ErrorAction SilentlyContinue
Remove-Item -Path "C:\Windows\SoftwareDistribution\Download\*" -Recurse -Force -ErrorAction SilentlyContinue
Start-Service -Name wuauserv -ErrorAction SilentlyContinue
```

### 3. 清空回收站（安全）
```powershell
$shell = New-Object -ComObject Shell.Application
$recycleBin = $shell.Namespace(0xA)
$recycleBin.Self.InvokeVerb("empty")
```

### 4. 分析用户文件夹大小
```powershell
$folders = @(
    "$env:USERPROFILE\Downloads",
    "$env:USERPROFILE\Videos", 
    "$env:USERPROFILE\Pictures",
    "$env:USERPROFILE\Desktop",
    "$env:USERPROFILE\Documents"
)
foreach ($f in $folders) {
    if (Test-Path $f) {
        $size = (Get-ChildItem $f -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $gb = [math]::Round($size/1GB, 2)
        Write-Host "$f : ${gb}GB"
    }
}
```

### 5. 迁移用户文件夹到D盘（最有效）
**手动方法（推荐，最安全）**:
1. 打开"此电脑"
2. 右键点击"下载" → 属性 → 位置
3. 点击"移动"，选择 `D:\Downloads`
4. 点击"应用"，Windows自动迁移
5. 同样处理：视频、图片、桌面、文档

**预期效果**: 可释放10-50GB空间

### 6. 禁用休眠文件（释放内存等量空间）
```powershell
# 禁用（管理员权限）
powercfg -h off

# 重新启用
powercfg -h on
```
**注意**: 禁用后无法使用休眠功能，但睡眠模式正常

## 危险操作（不建议）
- ❌ 手动删除 `C:\Windows\` 下的文件
- ❌ 移动 `C:\Program Files\` 文件夹
- ❌ 删除注册表

## 检查磁盘空间
```powershell
$disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
Write-Host "总容量: $($disk.Size/1GB) GB"
Write-Host "已使用: $($disk.Size/1GB - $disk.FreeSpace/1GB) GB"
Write-Host "剩余: $($disk.FreeSpace/1GB) GB"
```

## Python自动化脚本
可以使用Python创建自动化清理脚本：

```python
import os
import shutil
import tempfile

def clean_temp_files():
    """清理临时文件"""
    temp_dirs = [
        tempfile.gettempdir(),
        r"C:\Windows\Temp"
    ]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            os.makedirs(temp_dir, exist_ok=True)
    print("临时文件已清理")

if __name__ == "__main__":
    clean_temp_files()
```

## 技能要点
- ✅ 总是先检查磁盘空间再清理
- ✅ 优先使用Windows内置功能（文件夹位置移动到非系统盘）
- ✅ 清理前告知用户将释放多少空间
- ⚠️ 禁用休眠文件需确认用户不需要休眠功能
- ❌ 不要删除系统文件或注册表
