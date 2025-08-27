import datetime
from typing import Iterator


def iter_months(start_date: datetime.date, end_date: datetime.date) -> Iterator[datetime.date]:
    """
    遍历两个日期之间的所有月份

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Yields:
        每个月的第一天日期对象
    """
    current = start_date.replace(day=1)
    end_date = end_date.replace(day=1)

    while current <= end_date:
        yield current
        # 移动到下一个月
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def iter_months_with_range(start_date: datetime.date, end_date: datetime.date) -> Iterator[
    tuple[datetime.date, datetime.date]]:
    """
    遍历两个日期之间的所有月份，并返回每月的开始和结束日期

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Yields:
        元组(每月开始日期, 每月结束日期)
    """
    for month_start in iter_months(start_date, end_date):
        # 计算该月的结束日期
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1, day=1) - datetime.timedelta(days=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1, day=1) - datetime.timedelta(days=1)

        # 如果是最后一个月，结束日期不应超过实际的结束日期
        actual_end = min(month_end, end_date)
        yield (month_start, actual_end)