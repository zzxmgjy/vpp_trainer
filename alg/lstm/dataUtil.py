from util.logger import logger
from pathlib import Path
import pandas as pd

# 合并数据文件
def merge_station_data(station_id, output_path):
    """
    合并场站数据文件

    1. 如果data-{station_id}-all.csv不存在，所有数据合并进去
    2. 如果data-{station_id}-all.csv存在，读取数据中最后一条，获取time字段，
       该时间之后的数据合并到data-{station_id}-all.csv中
    """
    logger.info(f"开始合并场站 {station_id} 的数据文件")

    # 构建数据目录路径
    data_dir = Path(output_path) / station_id / "data"
    if not data_dir.exists():
        logger.warning(f"场站 {station_id} 的数据目录不存在: {data_dir}")
        return False

    # 合并后的文件路径
    all_data_file = data_dir / f"data-{station_id}-all.csv"

    # 查找所有月度数据文件
    pattern = f"data-{station_id}-????-??.csv"
    monthly_files = list(data_dir.glob(pattern))

    if not monthly_files:
        logger.warning(f"场站 {station_id} 没有找到月度数据文件")
        return False

    # 按文件名排序（确保按时间顺序处理）
    monthly_files.sort()

    # 检查合并文件是否存在
    if all_data_file.exists():
        # 读取现有文件的最后一条记录
        try:
            existing_data = pd.read_csv(all_data_file)
            if existing_data.empty:
                last_time = None
                logger.warning(f"合并文件 {all_data_file} 存在但为空")
            else:
                last_time = existing_data['time'].iloc[-1]
                logger.info(f"合并文件最后记录时间: {last_time}")
        except Exception as e:
            logger.error(f"读取合并文件失败: {e}")
            last_time = None
    else:
        last_time = None
        logger.info(f"合并文件不存在，将创建新文件")

    # 处理所有月度文件
    dfs_to_concat = []
    if last_time is None:
        # 如果合并文件不存在或为空，读取所有月度文件
        for file in monthly_files:
            try:
                df = pd.read_csv(file)
                dfs_to_concat.append(df)
                logger.info(f"读取文件: {file}, 记录数: {len(df)}")
            except Exception as e:
                logger.error(f"读取文件 {file} 失败: {e}")
    else:
        # 如果合并文件存在且有数据，只读取最后时间之后的数据
        dfs_to_concat.append(existing_data)
        for file in monthly_files:
            try:
                # 提取文件名中的年月，例如 "2023-08"
                file_year_month = '-'.join(file.stem.split('-')[-2:])

                # 提取最后记录时间中的年月，例如 "2023-08"
                last_time_year_month = '-'.join(last_time.split('-')[:2])

                # 如果文件的年月大于或等于最后记录的年月，才读取
                if file_year_month >= last_time_year_month:
                    df = pd.read_csv(file)
                    # 筛选最后时间之后的数据
                    new_data = df[pd.to_datetime(df['time']) > pd.to_datetime(last_time)]
                    if not new_data.empty:
                        dfs_to_concat.append(new_data)
                        logger.info(f"读取文件: {file}, 新增记录数: {len(new_data)}")
            except Exception as e:
                logger.error(f"读取文件 {file} 失败: {e}")

    # 合并数据并保存
    if dfs_to_concat:
        try:
            merged_df = pd.concat(dfs_to_concat, ignore_index=True)
            # 按时间排序并去重
            merged_df['time'] = pd.to_datetime(merged_df['time'])
            merged_df = merged_df.sort_values('time').drop_duplicates(subset=['time'])
            merged_df.to_csv(all_data_file, index=False)
            logger.info(f"合并完成，保存到 {all_data_file}，总记录数: {len(merged_df)}")
            return True
        except Exception as e:
            logger.error(f"合并数据失败: {e}")
            return False
    else:
        logger.warning(f"没有找到需要合并的数据")
        return False