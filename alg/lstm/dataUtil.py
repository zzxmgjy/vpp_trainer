from util.logger import logger
from pathlib import Path
import pandas as pd

# 合并数据文件
def merge_station_data(station_id, output_path):
    """
    合并场站数据文件（优化版）

    核心逻辑：
    1. 高效读取 `all.csv` 的最后一条记录，获取其时间戳 `last_time`。
    2. 遍历月度数据文件，只读取并筛选出时间戳严格大于 `last_time` 的新数据。
    3. 如果找到新数据，将它们合并后，以追加（append）模式写入 `all.csv` 的末尾。
    4. 整个过程避免了将庞大的 `all.csv` 文件完整读入内存再重写的低效操作。
    """
    logger.info(f"开始合并场站 {station_id} 的数据文件")

    # 构建数据目录和文件路径
    data_dir = Path(output_path) / station_id / "data"
    if not data_dir.exists():
        logger.warning(f"场站 {station_id} 的数据目录不存在: {data_dir}")
        return False

    all_dir = data_dir / "all"
    all_dir.mkdir(exist_ok=True)
    all_data_file = all_dir / f"data-{station_id}-all.csv"

    # 查找并排序所有月度数据文件
    monthly_files = sorted(list(data_dir.glob(f"data-{station_id}-????-??.csv")))
    if not monthly_files:
        logger.warning(f"场站 {station_id} 没有找到月度数据文件")
        return False

    # --- 高效获取最后的时间戳 ---
    last_time = None
    if all_data_file.exists() and all_data_file.stat().st_size > 0:
        try:
            # 为了效率，只读取最后一行的 'time' 列，避免加载整个文件
            # 对于中小型CSV，直接读取最后一列也可以
            df_tail = pd.read_csv(all_data_file, usecols=['time']).tail(1)
            if not df_tail.empty:
                last_time = pd.to_datetime(df_tail['time'].iloc[0])
                logger.info(f"合并文件 {all_data_file.name} 最后记录时间: {last_time}")
        except Exception as e:
            logger.error(f"读取 {all_data_file.name} 的最后时间失败: {e}。将进行全量合并。")
            last_time = None # 如果读取失败，则退回至全量合并模式
    else:
        logger.info(f"合并文件 {all_data_file.name} 不存在或为空，将创建新文件。")

    # --- 遍历月度文件，只寻找新数据 ---
    new_data_dfs = []

    # 确定从哪个文件开始检查
    start_checking = False if last_time else True

    for file in monthly_files:
        if not start_checking:
            # 提取文件名中的年月，例如 "2023-08"
            file_year_month = '-'.join(file.stem.split('-')[-2:])
            # 如果文件的年月大于或等于最后记录的年月，就需要开始检查了
            if file_year_month >= last_time.strftime('%Y-%m'):
                start_checking = True

        if start_checking:
            try:
                df = pd.read_csv(file)
                if df.empty:
                    continue

                # 确保 time 列是 datetime 类型
                df['time'] = pd.to_datetime(df['time'])

                if last_time:
                    # 核心逻辑：只筛选出时间戳严格大于 last_time 的数据
                    new_rows = df[df['time'] > last_time]
                else:
                    # 如果是首次合并，所有数据都是新数据
                    new_rows = df

                if not new_rows.empty:
                    new_data_dfs.append(new_rows)
                    logger.info(f"在文件 {file.name} 中发现 {len(new_rows)} 条新记录")

            except Exception as e:
                logger.error(f"处理文件 {file.name} 失败: {e}")

    # --- 合并新数据并以追加模式保存 ---
    if not new_data_dfs:
        logger.info(f"场站 {station_id} 没有发现需要合并的新数据。")
        return True

    try:
        # 将所有找到的新数据行合并成一个 DataFrame
        all_new_df = pd.concat(new_data_dfs, ignore_index=True)

        # 在追加前，对新数据自身进行排序和去重，确保新数据内部的唯一性和顺序
        all_new_df = all_new_df.sort_values('time').drop_duplicates(subset=['time'])

        # 使用追加模式(mode='a')写入文件，如果文件不存在则会创建
        # header=not all_data_file.exists() 确保只有在文件首次创建时才写入表头
        all_new_df.to_csv(
            all_data_file,
            mode='a',
            header=not all_data_file.exists() or all_data_file.stat().st_size == 0,
            index=False,
            lineterminator='\n'
        )

        logger.info(f"成功向 {all_data_file.name} 追加 {len(all_new_df)} 条新记录。")
        return True
    except Exception as e:
        logger.error(f"追加新数据到 {all_data_file.name} 失败: {e}")
        return False