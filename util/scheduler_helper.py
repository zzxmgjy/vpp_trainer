from apscheduler.triggers.cron import CronTrigger
from config.app_config import CronConfig
from util.logger import logger

class SchedulerHelper:
    """调度器辅助类，用于创建灵活的调度触发器"""
    
    @staticmethod
    def create_trigger(cron_config: CronConfig) -> CronTrigger:
        """
        根据配置创建相应的调度触发器
        
        Args:
            cron_config: 调度配置对象
            
        Returns:
            CronTrigger: 调度触发器
        """
        mode = cron_config.mode.lower()
        hour = cron_config.hour
        minute = cron_config.minute
        day = cron_config.day
        
        logger.info(f"创建调度触发器 - 模式: {mode}, 时间: {hour:02d}:{minute:02d}, 日期: {day}")
        
        if mode == "daily":
            # 每日执行
            return CronTrigger(hour=hour, minute=minute)
            
        elif mode == "weekly":
            # 每周执行，day 表示周几 (0=周一, 6=周日)
            if day is None:
                day = 0  # 默认周一
            return CronTrigger(day_of_week=day, hour=hour, minute=minute)
            
        elif mode == "monthly":
            # 每月执行，day 表示每月第几天
            if day is None:
                day = 1  # 默认每月1号
            return CronTrigger(day=day, hour=hour, minute=minute)
            
        elif mode == "custom":
            # 自定义 cron 表达式
            if cron_config.custom_cron:
                # 解析自定义 cron 表达式 (格式: "minute hour day month day_of_week")
                parts = cron_config.custom_cron.split()
                if len(parts) == 5:
                    return CronTrigger(
                        minute=parts[0],
                        hour=parts[1], 
                        day=parts[2],
                        month=parts[3],
                        day_of_week=parts[4]
                    )
                else:
                    logger.warning(f"自定义 cron 表达式格式错误: {cron_config.custom_cron}")
                    return CronTrigger(hour=hour, minute=minute)
            else:
                logger.warning("自定义模式但未提供 custom_cron，使用默认每日执行")
                return CronTrigger(hour=hour, minute=minute)
                
        else:
            logger.warning(f"未知的调度模式: {mode}，使用默认每日执行")
            return CronTrigger(hour=hour, minute=minute)
    
    @staticmethod
    def get_schedule_description(cron_config: CronConfig) -> str:
        """
        获取调度配置的描述信息
        
        Args:
            cron_config: 调度配置对象
            
        Returns:
            str: 调度描述
        """
        mode = cron_config.mode.lower()
        hour = cron_config.hour
        minute = cron_config.minute
        day = cron_config.day
        
        if mode == "daily":
            return f"每日 {hour:02d}:{minute:02d} 执行"
            
        elif mode == "weekly":
            day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            day_name = day_names[day] if day is not None and 0 <= day <= 6 else "周一"
            return f"每周{day_name} {hour:02d}:{minute:02d} 执行"
            
        elif mode == "monthly":
            day_str = f"{day}号" if day is not None else "1号"
            return f"每月{day_str} {hour:02d}:{minute:02d} 执行"
            
        elif mode == "custom":
            if cron_config.custom_cron:
                return f"自定义调度: {cron_config.custom_cron}"
            else:
                return f"自定义调度 (默认每日 {hour:02d}:{minute:02d})"
                
        else:
            return f"未知模式 (默认每日 {hour:02d}:{minute:02d})"