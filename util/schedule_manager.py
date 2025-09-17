#!/usr/bin/env python3
"""
调度配置管理工具
用于快速切换 LSTM 模型训练的调度模式
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ScheduleManager:
    """调度配置管理器"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        self.config_path = Path(config_path)
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def set_weekly_schedule(self, day_of_week: int = 0, hour: int = 1, minute: int = 30):
        """
        设置每周执行
        
        Args:
            day_of_week: 周几执行 (0=周一, 6=周日)
            hour: 小时 (0-23)
            minute: 分钟 (0-59)
        """
        config = self.load_config()
        config['cron']['trainLstm'] = {
            'mode': 'weekly',
            'day': day_of_week,
            'hour': hour,
            'minute': minute
        }
        self.save_config(config)
        
        day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        print(f"✅ 已设置为每周执行: 每{day_names[day_of_week]} {hour:02d}:{minute:02d}")
    
    def set_monthly_schedule(self, day_of_month: int = 1, hour: int = 1, minute: int = 30):
        """
        设置每月执行
        
        Args:
            day_of_month: 每月第几天 (1-31)
            hour: 小时 (0-23)
            minute: 分钟 (0-59)
        """
        config = self.load_config()
        config['cron']['trainLstm'] = {
            'mode': 'monthly',
            'day': day_of_month,
            'hour': hour,
            'minute': minute
        }
        self.save_config(config)
        print(f"✅ 已设置为每月执行: 每月{day_of_month}号 {hour:02d}:{minute:02d}")
    
    def set_daily_schedule(self, hour: int = 1, minute: int = 30):
        """
        设置每日执行
        
        Args:
            hour: 小时 (0-23)
            minute: 分钟 (0-59)
        """
        config = self.load_config()
        config['cron']['trainLstm'] = {
            'mode': 'daily',
            'hour': hour,
            'minute': minute
        }
        self.save_config(config)
        print(f"✅ 已设置为每日执行: 每天 {hour:02d}:{minute:02d}")
    
    def set_custom_schedule(self, cron_expression: str):
        """
        设置自定义调度
        
        Args:
            cron_expression: cron 表达式 (格式: "分 时 日 月 周")
        """
        config = self.load_config()
        config['cron']['trainLstm'] = {
            'mode': 'custom',
            'hour': 1,  # 默认值，在自定义模式下会被忽略
            'minute': 30,
            'custom_cron': cron_expression
        }
        self.save_config(config)
        print(f"✅ 已设置为自定义调度: {cron_expression}")
    
    def get_current_schedule(self) -> Dict[str, Any]:
        """获取当前调度配置"""
        config = self.load_config()
        return config['cron']['trainLstm']
    
    def show_current_schedule(self):
        """显示当前调度配置"""
        schedule = self.get_current_schedule()
        mode = schedule.get('mode', 'monthly')
        hour = schedule.get('hour', 1)
        minute = schedule.get('minute', 30)
        day = schedule.get('day')
        
        print(f"📅 当前 LSTM 训练调度配置:")
        print(f"   模式: {mode}")
        print(f"   时间: {hour:02d}:{minute:02d}")
        
        if mode == 'weekly':
            day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            day_name = day_names[day] if day is not None and 0 <= day <= 6 else "未知"
            print(f"   执行: 每{day_name}")
        elif mode == 'monthly':
            print(f"   执行: 每月{day}号")
        elif mode == 'daily':
            print(f"   执行: 每天")
        elif mode == 'custom':
            custom_cron = schedule.get('custom_cron', '未设置')
            print(f"   自定义: {custom_cron}")

def main():
    """命令行工具主函数"""
    import sys
    
    manager = ScheduleManager()
    
    if len(sys.argv) < 2:
        print("🔧 调度配置管理工具")
        print("\n用法:")
        print("  python schedule_manager.py show                    # 显示当前配置")
        print("  python schedule_manager.py weekly [day] [hour] [minute]   # 设置每周执行")
        print("  python schedule_manager.py monthly [day] [hour] [minute]  # 设置每月执行")
        print("  python schedule_manager.py daily [hour] [minute]          # 设置每日执行")
        print("  python schedule_manager.py custom 'cron_expression'       # 设置自定义调度")
        print("\n示例:")
        print("  python schedule_manager.py weekly 0 1 30        # 每周一 01:30")
        print("  python schedule_manager.py monthly 1 2 0        # 每月1号 02:00")
        print("  python schedule_manager.py daily 3 15           # 每天 03:15")
        print("  python schedule_manager.py custom '0 2 * * 1,3,5'  # 每周一三五 02:00")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'show':
        manager.show_current_schedule()
    
    elif command == 'weekly':
        day = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        hour = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        minute = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        manager.set_weekly_schedule(day, hour, minute)
    
    elif command == 'monthly':
        day = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        hour = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        minute = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        manager.set_monthly_schedule(day, hour, minute)
    
    elif command == 'daily':
        hour = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        minute = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        manager.set_daily_schedule(hour, minute)
    
    elif command == 'custom':
        if len(sys.argv) < 3:
            print("❌ 错误: 请提供 cron 表达式")
            return
        cron_expr = sys.argv[2]
        manager.set_custom_schedule(cron_expr)
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == '__main__':
    main()