# LSTM 模型训练调度配置指南

## 概述

本项目支持灵活配置 LSTM 模型训练的调度时间，支持每日、每周、每月以及自定义调度模式。

## 快速配置

### 方法一：直接修改配置文件

编辑 `config/config.yaml` 中的 `trainLstm` 部分：

```yaml
cron:
  trainLstm:
    mode: "weekly"    # 调度模式: daily, weekly, monthly, custom
    day: 0           # 具体含义取决于模式
    hour: 1          # 小时 (0-23)
    minute: 30       # 分钟 (0-59)
    # custom_cron: "0 30 1 * * 1"  # 自定义模式时使用
```

### 方法二：使用命令行工具

```bash
# 查看当前配置
python util/schedule_manager.py show

# 设置每周执行 (每周一 01:30)
python util/schedule_manager.py weekly 0 1 30

# 设置每月执行 (每月1号 01:30)
python util/schedule_manager.py monthly 1 1 30

# 设置每日执行 (每天 02:00)
python util/schedule_manager.py daily 2 0

# 设置自定义调度 (每周三和周六 03:15)
python util/schedule_manager.py custom "15 3 * * 3,6"
```

## 调度模式详解

### 1. 每日执行 (daily)
```yaml
trainLstm:
  mode: "daily"
  hour: 2
  minute: 0
```
- 每天指定时间执行
- `day` 参数被忽略

### 2. 每周执行 (weekly)
```yaml
trainLstm:
  mode: "weekly"
  day: 0      # 0=周一, 1=周二, ..., 6=周日
  hour: 1
  minute: 30
```
- 每周指定星期的指定时间执行
- `day`: 0=周一, 1=周二, 2=周三, 3=周四, 4=周五, 5=周六, 6=周日

### 3. 每月执行 (monthly)
```yaml
trainLstm:
  mode: "monthly"
  day: 1      # 每月第1天
  hour: 1
  minute: 30
```
- 每月指定日期的指定时间执行
- `day`: 1-31 (每月第几天)

### 4. 自定义调度 (custom)
```yaml
trainLstm:
  mode: "custom"
  hour: 1     # 这些值在自定义模式下被忽略
  minute: 30
  custom_cron: "30 1 1 * *"  # 每月1号 01:30
```
- 使用标准 cron 表达式
- 格式: `"分 时 日 月 周"`

## 常用调度示例

### 每周执行
```bash
# 每周一凌晨 1:30
python util/schedule_manager.py weekly 0 1 30

# 每周五晚上 23:00
python util/schedule_manager.py weekly 4 23 0
```

### 每月执行
```bash
# 每月1号凌晨 2:00
python util/schedule_manager.py monthly 1 2 0

# 每月15号中午 12:00
python util/schedule_manager.py monthly 15 12 0
```

### 自定义调度
```bash
# 每两周的周日凌晨 1:30
python util/schedule_manager.py custom "30 1 * * 0/2"

# 每季度第一天凌晨 2:00 (1月、4月、7月、10月的1号)
python util/schedule_manager.py custom "0 2 1 1,4,7,10 *"

# 工作日每天凌晨 3:00
python util/schedule_manager.py custom "0 3 * * 1-5"
```

## Cron 表达式格式

```
分 时 日 月 周
│ │ │ │ │
│ │ │ │ └─── 周几 (0-7, 0和7都表示周日)
│ │ │ └───── 月份 (1-12)
│ │ └─────── 日期 (1-31)
│ └───────── 小时 (0-23)
└─────────── 分钟 (0-59)
```

### 特殊字符
- `*`: 匹配任意值
- `,`: 分隔多个值 (如: `1,3,5`)
- `-`: 表示范围 (如: `1-5`)
- `/`: 表示间隔 (如: `*/2` 表示每2个单位)

### 常用表达式
- `0 2 * * *`: 每天凌晨2点
- `30 1 * * 1`: 每周一凌晨1:30
- `0 0 1 * *`: 每月1号午夜
- `0 9 * * 1-5`: 工作日上午9点
- `0 */6 * * *`: 每6小时执行一次

## 重启应用

修改配置后需要重启应用才能生效：

```bash
# 停止当前应用 (Ctrl+C)
# 然后重新启动
python main.py
```

## 验证配置

启动应用后，查看日志输出，会显示当前的调度配置：

```
INFO - LSTM 模型训练调度: 每周一 01:30 执行
INFO - Application started
```

## 故障排除

1. **配置不生效**: 确保重启了应用
2. **时间格式错误**: 检查 hour (0-23) 和 minute (0-59) 的范围
3. **自定义 cron 错误**: 验证 cron 表达式格式是否正确
4. **权限问题**: 确保有写入配置文件的权限

## 监控调度任务

可以通过日志文件监控调度任务的执行情况：

```bash
# 查看最新日志
tail -f logs/training.log

# 搜索调度相关日志
grep "train by lstm" logs/training.log