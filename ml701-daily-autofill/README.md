# 美客多 701 店铺 · 每日退货码 / 广告费自动登记

紫鸟浏览器（美客多墨西哥 G-701 店铺）采集 → 钉钉表格「退货登记码」「广告费消耗」写入的 Windows 桌面自动化脚本。
一条链路串行走完：当日退货授权码写入「退货登记码」，昨日营业额/广告费/点击写入「广告费消耗」。

## 文件

| 文件 | 职责 |
|---|---|
| `daily_701_oneflow.py` | **唯一入口**。调度全流程：紫鸟取码 → 采集昨日数据 → 同一钉钉窗口内写两张表 |
| `daily_ads_cost.py` | 紫鸟侧采集业务指标/广告费（单日、两轮一致、0-800 校验） |
| `write_return_code.py` | 「退货登记码」表：打开、701 锚点对版、单格写入、逐格真值验证、凭证 |
| `write_ads_cost.py` | 「广告费消耗」表：打开、名称框/公式栏真值读写 |
| `finish_ads_cost.py` | 广告费 B/C/D 逐格补写与回读验证 |

## 运行环境

- Windows，受管 Python：`C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe`
- 依赖：标准库 + `comtypes`（Windows UI Automation COM 接口）
- 前置：紫鸟浏览器已登录该店铺、钉钉桌面客户端已登录且进程在运行

## 用法

```bash
# 全链路：采今日退货码 + 昨日数据并写入两张表
python daily_701_oneflow.py

# 只读体检（不写、不切窗口）
python daily_701_oneflow.py --probe

# 当日码已知：零采集，直接写入
python daily_701_oneflow.py --code XXXXXXXX

# 数据已知：营业额,广告费,点击 零采集写入
python daily_701_oneflow.py --data 1642,269,184

# 指定广告费目标日 / 无障碍树软修复
python daily_701_oneflow.py --date 2026-11-19 --fix-a11y
```

后台无控制台执行（整轮单进程、不置顶、不锁桌面）：

```python
subprocess.Popen([py, "daily_701_oneflow.py"], cwd=BASE,
                 stdin=subprocess.DEVNULL, creationflags=0x08000000)  # CREATE_NO_WINDOW
```

## 结果文件（运行目录 `.workbuddy/tmp/`）

- `oneflow_last.json` —— 权威结果，`overall_ok` 为真才算成功
- `oneflow.log` —— 运行日志
- `oneflow_pending.json` —— 已采到但未写成的数据，恢复后零采集补写
- `return_write_last.json` / `daily_ads_cost_last.json` —— 两张表各自的写入凭证

## 硬约束（勿改）

- **不 taskkill / 不重启 / 不自启动钉钉进程**；未运行则 ABORT，不假报成功。
- 定位一律走 UIA 名称/控件类型/Pattern，**禁止用坐标或截图确认数据**。
- 写入前按真实文档标题（Pane 名称 `XXX · 钉钉文档`）对版，标题 + 名称框都对版才放行。
- 写入只在 701 锚点右邻**单格**替换，写后逐格读公式栏真值验证。
- 采集不达标即停，不填异常值；失败如实报告，最多重试 1 次。
- 整轮最多「紫鸟一次 + 钉钉一次」必要激活，结束恢复用户原前台窗口。
- 退货码取自卖家后台 `Código de autorización para hoy:` 模块，8 位十六进制；与上次采信码相同视为页面未刷新，不采信。

## 安全

本目录**不含任何账号密码、API key 或真实授权码**。真实凭证仅存在运行目录的 `.workbuddy/tmp/` 本地文件中，未纳入版本控制。脚本内出现的 8 位字符串（`220DF71D`、`BE1B261A`、`C0FFEE01` 等）均为用法示例或单元测试用的假值。
