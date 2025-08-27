#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================
#  预测结果对比分析工具
#  对比 forcast.py 输出的JSON文件中2025-08-11预测值和data-08-11.csv中的实际数据
# =========================================================

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def calculate_mape(actual, predicted):
    """
    计算MAPE (Mean Absolute Percentage Error)
    使用改进的方法处理接近零值的情况
    
    参数:
    actual: 实际值数组
    predicted: 预测值数组
    
    返回:
    MAPE值 (百分比)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 使用对称MAPE (sMAPE) 的思想，避免除零问题
    # sMAPE = 100 * |actual - predicted| / ((|actual| + |predicted|) / 2)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    
    # 避免分母为0的情况
    denominator = np.where(denominator < 1e-6, 1e-6, denominator)
    
    mape = np.mean(np.abs(actual - predicted) / denominator) * 100
    return mape

def calculate_mae(actual, predicted):
    """
    计算MAE (Mean Absolute Error)
    
    参数:
    actual: 实际值数组
    predicted: 预测值数组
    
    返回:
    MAE值
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs(actual - predicted))

def calculate_rmse(actual, predicted):
    """
    计算RMSE (Root Mean Square Error)
    
    参数:
    actual: 实际值数组
    predicted: 预测值数组
    
    返回:
    RMSE值
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))

def calculate_mape_traditional(actual, predicted, threshold=0.1):
    """
    计算传统MAPE，但对小值进行处理
    
    参数:
    actual: 实际值数组
    predicted: 预测值数组
    threshold: 阈值，低于此值的实际值将被排除或特殊处理
    
    返回:
    MAPE值 (百分比)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 找出实际值大于阈值的索引
    valid_indices = np.abs(actual) > threshold
    
    if np.sum(valid_indices) == 0:
        return np.inf  # 如果没有有效值，返回无穷大
    
    actual_valid = actual[valid_indices]
    predicted_valid = predicted[valid_indices]
    
    mape = np.mean(np.abs((actual_valid - predicted_valid) / actual_valid)) * 100
    return mape

def load_prediction_json(json_file_path):
    """
    加载预测结果JSON文件
    
    参数:
    json_file_path: JSON文件路径
    
    返回:
    预测数据DataFrame
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取预测数据
        predictions = data.get('predictions', [])
        if not predictions:
            logger.error("JSON文件中没有找到预测数据")
            return None
        
        # 转换为DataFrame
        df = pd.DataFrame(predictions)
        df['time'] = pd.to_datetime(df['time'])
        
        logger.info(f"成功加载预测数据，共{len(df)}条记录")
        logger.info(f"时间范围: {df['time'].min()} 到 {df['time'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"加载预测JSON文件失败: {e}")
        return None

def load_actual_csv(csv_file_path):
    """
    加载实际数据CSV文件
    
    参数:
    csv_file_path: CSV文件路径
    
    返回:
    实际数据DataFrame
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        # 检查必要的列是否存在
        required_cols = ['time', 'load', 'meter']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"CSV文件缺少必要的列: {missing_cols}")
            logger.info(f"CSV文件现有列: {df.columns.tolist()}")
            return None
        
        # 转换时间列
        df['time'] = pd.to_datetime(df['time'])
        
        # 确保数值列为数值类型
        df['load'] = pd.to_numeric(df['load'], errors='coerce')
        df['meter'] = pd.to_numeric(df['meter'], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna(subset=['load', 'meter'])
        
        logger.info(f"成功加载实际数据，共{len(df)}条记录")
        logger.info(f"时间范围: {df['time'].min()} 到 {df['time'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"加载实际CSV文件失败: {e}")
        return None

def merge_data(pred_df, actual_df, target_date='2025-08-11'):
    """
    合并预测数据和实际数据，筛选指定日期
    
    参数:
    pred_df: 预测数据DataFrame
    actual_df: 实际数据DataFrame
    target_date: 目标日期，格式为'YYYY-MM-DD'
    
    返回:
    合并后的DataFrame
    """
    try:
        # 筛选目标日期的数据
        target_date = pd.to_datetime(target_date).date()
        
        pred_filtered = pred_df[pred_df['time'].dt.date == target_date].copy()
        actual_filtered = actual_df[actual_df['time'].dt.date == target_date].copy()
        
        if len(pred_filtered) == 0:
            logger.error(f"预测数据中没有找到{target_date}的数据")
            return None
            
        if len(actual_filtered) == 0:
            logger.error(f"实际数据中没有找到{target_date}的数据")
            return None
        
        # 按时间合并数据
        merged_df = pd.merge(pred_filtered, actual_filtered, on='time', suffixes=('_pred', '_actual'))
        
        if len(merged_df) == 0:
            logger.error("合并后没有匹配的数据")
            return None
        
        logger.info(f"成功合并数据，共{len(merged_df)}条记录")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"合并数据失败: {e}")
        return None

def calculate_15min_metrics(merged_df):
    """
    计算每15分钟的多种误差指标
    
    参数:
    merged_df: 合并后的数据DataFrame
    
    返回:
    包含每15分钟多种误差指标的DataFrame
    """
    try:
        results = []
        
        for _, row in merged_df.iterrows():
            time_str = row['time'].strftime('%H:%M')
            
            # 计算load的各种误差指标
            load_mape = calculate_mape([row['load_actual']], [row['load_pred']])
            load_mape_traditional = calculate_mape_traditional([row['load_actual']], [row['load_pred']])
            load_mae = calculate_mae([row['load_actual']], [row['load_pred']])
            load_rmse = calculate_rmse([row['load_actual']], [row['load_pred']])
            
            # 计算meter的各种误差指标
            meter_mape = calculate_mape([row['meter_actual']], [row['meter_pred']])
            meter_mape_traditional = calculate_mape_traditional([row['meter_actual']], [row['meter_pred']])
            meter_mae = calculate_mae([row['meter_actual']], [row['meter_pred']])
            meter_rmse = calculate_rmse([row['meter_actual']], [row['meter_pred']])
            
            # 计算相对误差百分比（避免MAPE的问题）
            load_relative_error = abs(row['load_actual'] - row['load_pred']) / max(abs(row['load_actual']), abs(row['load_pred']), 1e-6) * 100
            meter_relative_error = abs(row['meter_actual'] - row['meter_pred']) / max(abs(row['meter_actual']), abs(row['meter_pred']), 1e-6) * 100
            
            results.append({
                'time': time_str,
                'datetime': row['time'],
                'load_actual': row['load_actual'],
                'load_pred': row['load_pred'],
                'load_mape': load_mape,
                'load_mape_traditional': load_mape_traditional,
                'load_mae': load_mae,
                'load_rmse': load_rmse,
                'load_relative_error': load_relative_error,
                'meter_actual': row['meter_actual'],
                'meter_pred': row['meter_pred'],
                'meter_mape': meter_mape,
                'meter_mape_traditional': meter_mape_traditional,
                'meter_mae': meter_mae,
                'meter_rmse': meter_rmse,
                'meter_relative_error': meter_relative_error
            })
        
        result_df = pd.DataFrame(results)
        
        logger.info("成功计算每15分钟的多种误差指标")
        
        return result_df
        
    except Exception as e:
        logger.error(f"计算每15分钟误差指标失败: {e}")
        return None

def calculate_daily_metrics(merged_df):
    """
    计算整天的多种误差指标
    
    参数:
    merged_df: 合并后的数据DataFrame
    
    返回:
    包含整天多种误差指标的字典
    """
    try:
        # 计算load的各种整天误差指标
        load_daily_mape = calculate_mape(merged_df['load_actual'].values, merged_df['load_pred'].values)
        load_daily_mape_traditional = calculate_mape_traditional(merged_df['load_actual'].values, merged_df['load_pred'].values)
        load_daily_mae = calculate_mae(merged_df['load_actual'].values, merged_df['load_pred'].values)
        load_daily_rmse = calculate_rmse(merged_df['load_actual'].values, merged_df['load_pred'].values)
        
        # 计算meter的各种整天误差指标
        meter_daily_mape = calculate_mape(merged_df['meter_actual'].values, merged_df['meter_pred'].values)
        meter_daily_mape_traditional = calculate_mape_traditional(merged_df['meter_actual'].values, merged_df['meter_pred'].values)
        meter_daily_mae = calculate_mae(merged_df['meter_actual'].values, merged_df['meter_pred'].values)
        meter_daily_rmse = calculate_rmse(merged_df['meter_actual'].values, merged_df['meter_pred'].values)
        
        # 计算平均相对误差
        load_avg_relative_error = np.mean([
            abs(actual - pred) / max(abs(actual), abs(pred), 1e-6) * 100
            for actual, pred in zip(merged_df['load_actual'], merged_df['load_pred'])
        ])
        
        meter_avg_relative_error = np.mean([
            abs(actual - pred) / max(abs(actual), abs(pred), 1e-6) * 100
            for actual, pred in zip(merged_df['meter_actual'], merged_df['meter_pred'])
        ])
        
        daily_results = {
            'load_daily_mape': load_daily_mape,
            'load_daily_mape_traditional': load_daily_mape_traditional,
            'load_daily_mae': load_daily_mae,
            'load_daily_rmse': load_daily_rmse,
            'load_avg_relative_error': load_avg_relative_error,
            'meter_daily_mape': meter_daily_mape,
            'meter_daily_mape_traditional': meter_daily_mape_traditional,
            'meter_daily_mae': meter_daily_mae,
            'meter_daily_rmse': meter_daily_rmse,
            'meter_avg_relative_error': meter_avg_relative_error,
            'total_records': len(merged_df),
            'date': merged_df['time'].dt.date.iloc[0]
        }
        
        logger.info("成功计算整天的多种误差指标")
        
        return daily_results
        
    except Exception as e:
        logger.error(f"计算整天误差指标失败: {e}")
        return None

def create_comparison_plots(minute_results, daily_results, output_dir='comparison_results'):
    """
    创建对比图表
    
    参数:
    minute_results: 每15分钟结果DataFrame
    daily_results: 整天结果字典
    output_dir: 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('default')
        
        # 1. 创建预测值vs实际值对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'预测结果对比分析 - {daily_results["date"]}', fontsize=16, fontweight='bold')
        
        # Load预测vs实际值时间序列
        axes[0, 0].plot(minute_results['datetime'], minute_results['load_actual'], 
                       label='实际值', linewidth=2, color='blue', alpha=0.8)
        axes[0, 0].plot(minute_results['datetime'], minute_results['load_pred'], 
                       label='预测值', linewidth=2, color='red', alpha=0.8)
        axes[0, 0].set_title('Load - 预测值vs实际值')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('Load值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Meter预测vs实际值时间序列
        axes[0, 1].plot(minute_results['datetime'], minute_results['meter_actual'], 
                       label='实际值', linewidth=2, color='blue', alpha=0.8)
        axes[0, 1].plot(minute_results['datetime'], minute_results['meter_pred'], 
                       label='预测值', linewidth=2, color='red', alpha=0.8)
        axes[0, 1].set_title('Meter - 预测值vs实际值')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('Meter值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Load MAPE时间序列
        axes[1, 0].plot(minute_results['datetime'], minute_results['load_mape'], 
                       linewidth=2, color='green', marker='o', markersize=3)
        axes[1, 0].axhline(y=daily_results['load_daily_mape'], color='red', 
                          linestyle='--', label=f'整天平均MAPE: {daily_results["load_daily_mape"]:.2f}%')
        axes[1, 0].set_title('Load - 每15分钟MAPE')
        axes[1, 0].set_xlabel('时间')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Meter MAPE时间序列
        axes[1, 1].plot(minute_results['datetime'], minute_results['meter_mape'], 
                       linewidth=2, color='green', marker='o', markersize=3)
        axes[1, 1].axhline(y=daily_results['meter_daily_mape'], color='red', 
                          linestyle='--', label=f'整天平均MAPE: {daily_results["meter_daily_mape"]:.2f}%')
        axes[1, 1].set_title('Meter - 每15分钟MAPE')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(output_dir, f'comparison_plot_{daily_results["date"]}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"对比图表已保存到: {plot_path}")
        
        # 2. 创建散点图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'预测准确性散点图 - {daily_results["date"]}', fontsize=14, fontweight='bold')
        
        # Load散点图
        axes[0].scatter(minute_results['load_actual'], minute_results['load_pred'], 
                       alpha=0.6, color='blue')
        min_val = min(minute_results['load_actual'].min(), minute_results['load_pred'].min())
        max_val = max(minute_results['load_actual'].max(), minute_results['load_pred'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='完美预测线')
        axes[0].set_xlabel('实际Load值')
        axes[0].set_ylabel('预测Load值')
        axes[0].set_title(f'Load预测准确性 (MAPE: {daily_results["load_daily_mape"]:.2f}%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Meter散点图
        axes[1].scatter(minute_results['meter_actual'], minute_results['meter_pred'], 
                       alpha=0.6, color='green')
        min_val = min(minute_results['meter_actual'].min(), minute_results['meter_pred'].min())
        max_val = max(minute_results['meter_actual'].max(), minute_results['meter_pred'].max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='完美预测线')
        axes[1].set_xlabel('实际Meter值')
        axes[1].set_ylabel('预测Meter值')
        axes[1].set_title(f'Meter预测准确性 (MAPE: {daily_results["meter_daily_mape"]:.2f}%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存散点图
        scatter_path = os.path.join(output_dir, f'scatter_plot_{daily_results["date"]}.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"散点图已保存到: {scatter_path}")
        
    except Exception as e:
        logger.error(f"创建对比图表失败: {e}")

def save_results_to_csv(minute_results, daily_results, output_dir='comparison_results'):
    """
    保存结果到CSV文件
    
    参数:
    minute_results: 每15分钟结果DataFrame
    daily_results: 整天结果字典
    output_dir: 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存每15分钟结果
        minute_csv_path = os.path.join(output_dir, f'minute_comparison_{daily_results["date"]}.csv')
        minute_results.to_csv(minute_csv_path, index=False, encoding='utf-8-sig')
        
        # 保存整天结果
        daily_csv_path = os.path.join(output_dir, f'daily_comparison_{daily_results["date"]}.csv')
        daily_df = pd.DataFrame([daily_results])
        daily_df.to_csv(daily_csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"每15分钟结果已保存到: {minute_csv_path}")
        logger.info(f"整天结果已保存到: {daily_csv_path}")
        
    except Exception as e:
        logger.error(f"保存结果到CSV失败: {e}")

def print_summary(minute_results, daily_results):
    """
    打印结果摘要
    
    参数:
    minute_results: 每15分钟结果DataFrame
    daily_results: 整天结果字典
    """
    print("\n" + "="*80)
    print(f"预测结果对比分析摘要 - {daily_results['date']}")
    print("="*80)
    
    print(f"\n数据概况:")
    print(f"  总记录数: {daily_results['total_records']} 条")
    print(f"  时间范围: {minute_results['time'].iloc[0]} - {minute_results['time'].iloc[-1]}")
    
    print(f"\n整天误差指标结果:")
    print(f"  Load指标:")
    print(f"    改进MAPE (sMAPE): {daily_results['load_daily_mape']:.2f}%")
    print(f"    传统MAPE: {daily_results['load_daily_mape_traditional']:.2f}%")
    print(f"    MAE: {daily_results['load_daily_mae']:.4f}")
    print(f"    RMSE: {daily_results['load_daily_rmse']:.4f}")
    print(f"    平均相对误差: {daily_results['load_avg_relative_error']:.2f}%")
    
    print(f"  Meter指标:")
    print(f"    改进MAPE (sMAPE): {daily_results['meter_daily_mape']:.2f}%")
    print(f"    传统MAPE: {daily_results['meter_daily_mape_traditional']:.2f}%")
    print(f"    MAE: {daily_results['meter_daily_mae']:.4f}")
    print(f"    RMSE: {daily_results['meter_daily_rmse']:.4f}")
    print(f"    平均相对误差: {daily_results['meter_avg_relative_error']:.2f}%")
    
    print(f"\n每15分钟误差统计 (改进MAPE):")
    print(f"  Load MAPE - 最小值: {minute_results['load_mape'].min():.2f}%")
    print(f"  Load MAPE - 最大值: {minute_results['load_mape'].max():.2f}%")
    print(f"  Load MAPE - 平均值: {minute_results['load_mape'].mean():.2f}%")
    print(f"  Load MAPE - 标准差: {minute_results['load_mape'].std():.2f}%")
    
    print(f"  Meter MAPE - 最小值: {minute_results['meter_mape'].min():.2f}%")
    print(f"  Meter MAPE - 最大值: {minute_results['meter_mape'].max():.2f}%")
    print(f"  Meter MAPE - 平均值: {minute_results['meter_mape'].mean():.2f}%")
    print(f"  Meter MAPE - 标准差: {minute_results['meter_mape'].std():.2f}%")
    
    print(f"\n每15分钟相对误差统计:")
    print(f"  Load相对误差 - 最小值: {minute_results['load_relative_error'].min():.2f}%")
    print(f"  Load相对误差 - 最大值: {minute_results['load_relative_error'].max():.2f}%")
    print(f"  Load相对误差 - 平均值: {minute_results['load_relative_error'].mean():.2f}%")
    
    print(f"  Meter相对误差 - 最小值: {minute_results['meter_relative_error'].min():.2f}%")
    print(f"  Meter相对误差 - 最大值: {minute_results['meter_relative_error'].max():.2f}%")
    print(f"  Meter相对误差 - 平均值: {minute_results['meter_relative_error'].mean():.2f}%")
    
    # 找出误差最高的时间点
    max_load_mape_idx = minute_results['load_mape'].idxmax()
    max_meter_mape_idx = minute_results['meter_mape'].idxmax()
    max_load_rel_idx = minute_results['load_relative_error'].idxmax()
    max_meter_rel_idx = minute_results['meter_relative_error'].idxmax()
    
    print(f"\n预测误差最大的时间点:")
    print(f"  Load最大MAPE时间: {minute_results.loc[max_load_mape_idx, 'time']} "
          f"(改进MAPE: {minute_results.loc[max_load_mape_idx, 'load_mape']:.2f}%)")
    print(f"  Meter最大MAPE时间: {minute_results.loc[max_meter_mape_idx, 'time']} "
          f"(改进MAPE: {minute_results.loc[max_meter_mape_idx, 'meter_mape']:.2f}%)")
    print(f"  Load最大相对误差时间: {minute_results.loc[max_load_rel_idx, 'time']} "
          f"(相对误差: {minute_results.loc[max_load_rel_idx, 'load_relative_error']:.2f}%)")
    print(f"  Meter最大相对误差时间: {minute_results.loc[max_meter_rel_idx, 'time']} "
          f"(相对误差: {minute_results.loc[max_meter_rel_idx, 'meter_relative_error']:.2f}%)")
    
    print(f"\n说明:")
    print(f"  - 改进MAPE使用对称MAPE方法，避免了传统MAPE在实际值接近0时的数值问题")
    print(f"  - 相对误差使用max(|actual|, |predicted|)作为分母，更加稳健")
    print(f"  - MAE和RMSE提供了绝对误差的度量")
    
    print("\n" + "="*80)

def main(prediction_json_path, actual_csv_path, target_date='2025-08-11', output_dir='comparison_results'):
    """
    主函数：执行完整的对比分析流程
    
    参数:
    prediction_json_path: 预测结果JSON文件路径
    actual_csv_path: 实际数据CSV文件路径
    target_date: 目标日期，格式为'YYYY-MM-DD'
    output_dir: 输出目录
    """
    logger.info("开始预测结果对比分析...")
    
    # 1. 加载数据
    logger.info("正在加载预测数据...")
    pred_df = load_prediction_json(prediction_json_path)
    if pred_df is None:
        return
    
    logger.info("正在加载实际数据...")
    actual_df = load_actual_csv(actual_csv_path)
    if actual_df is None:
        return
    
    # 2. 合并数据
    logger.info(f"正在合并{target_date}的数据...")
    merged_df = merge_data(pred_df, actual_df, target_date)
    if merged_df is None:
        return
    
    # 3. 计算多种误差指标
    logger.info("正在计算每15分钟的误差指标...")
    minute_results = calculate_15min_metrics(merged_df)
    if minute_results is None:
        return
    
    logger.info("正在计算整天的误差指标...")
    daily_results = calculate_daily_metrics(merged_df)
    if daily_results is None:
        return
    
    # 4. 生成图表
    logger.info("正在生成对比图表...")
    create_comparison_plots(minute_results, daily_results, output_dir)
    
    # 5. 保存结果
    logger.info("正在保存结果到CSV文件...")
    save_results_to_csv(minute_results, daily_results, output_dir)
    
    # 6. 打印摘要
    print_summary(minute_results, daily_results)
    
    logger.info("预测结果对比分析完成！")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python compare_prediction.py <预测JSON文件路径> <实际CSV文件路径> [目标日期] [输出目录]")
        print("示例: python compare_prediction.py prediction_12345_20250811_000000.json data-08-11.csv 2025-08-11 comparison_results")
        sys.exit(1)
    
    prediction_json_path = sys.argv[1]
    actual_csv_path = sys.argv[2]
    target_date = sys.argv[3] if len(sys.argv) > 3 else '2025-08-11'
    output_dir = sys.argv[4] if len(sys.argv) > 4 else 'comparison_results'
    
    # 检查文件是否存在
    if not os.path.exists(prediction_json_path):
        print(f"错误: 预测JSON文件不存在: {prediction_json_path}")
        sys.exit(1)
    
    if not os.path.exists(actual_csv_path):
        print(f"错误: 实际CSV文件不存在: {actual_csv_path}")
        sys.exit(1)
    
    # 执行对比分析
    main(prediction_json_path, actual_csv_path, target_date, output_dir)