#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================
#  场站功率预测接口
# =========================================================

import os
import numpy as np
import pandas as pd
import torch
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
import joblib
import yaml
import datetime
from util.logger import logger
import warnings

warnings.filterwarnings("ignore")


# 加载配置文件
def load_config(config_path="config.yml"):
    """加载配置文件"""
    logger.info(f"正在加载配置文件 {config_path}...")

    if not os.path.exists(config_path):
        logger.warning(f"配置文件 {config_path} 不存在，将使用默认配置")
        return {
            "output": {
                "path": "output",           # 默认输出文件夹目录
                "model_path": "model"       # 默认模型输出目录
            }
        }

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("配置文件加载成功")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {
            "output": {
                "path": "output",
                "model_path": "model"
            }
        }

# 检查Mamba模型是否可用以及CUDA是否可用
MAMBA_AVAILABLE = False
USE_FALLBACK_MAMBA = False

try:
    from mamba_ssm import Mamba as MambaSSM
    if torch.cuda.is_available():
        MAMBA_AVAILABLE = True
        logger.info("mamba_ssm 可用且CUDA可用，使用原生Mamba")
    else:
        USE_FALLBACK_MAMBA = True
        logger.warning("CUDA不可用，将使用fallback Mamba实现")
except Exception:
    USE_FALLBACK_MAMBA = True
    logger.warning("mamba_ssm 未找到，使用脚本内 fallback Mamba")

# 定义fallback Mamba实现（向量化近似，避免逐步循环）
class FallbackMamba(torch.nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_fast_path=True):
        super().__init__()
        self.d_model = d_model
        self.expand_dim = int(expand * d_model)
        self.in_proj = torch.nn.Linear(d_model, self.expand_dim * 2)
        pad = max(0, d_conv // 2)
        self.dw_conv = torch.nn.Conv1d(self.expand_dim, self.expand_dim, d_conv, padding=pad, groups=self.expand_dim)
        self.pw = torch.nn.Linear(self.expand_dim, self.expand_dim)
        self.out_proj = torch.nn.Linear(self.expand_dim, d_model)
        self.act = torch.nn.SiLU()
        self.gate = torch.nn.Sigmoid()

    def forward(self, x):  # x: [B,L,D]
        B, L, _ = x.shape
        x_proj, z = self.in_proj(x).chunk(2, dim=-1)
        y = self.dw_conv(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)
        y = self.act(self.pw(y))
        y = y * self.gate(z)
        return self.out_proj(y)

# 根据环境选择Mamba实现
if USE_FALLBACK_MAMBA:
    Mamba = FallbackMamba
else:
    Mamba = MambaSSM

# 简单的进程内缓存（按站点与设备类型区分）
_MODEL_CACHE = {}

# 定义模型架构
class Patching(torch.nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
    def forward(self, x):                                   # x:[B,C,L]
        n_patches = (max(x.size(2), self.patch_len)-self.patch_len)//self.stride+1
        tgt_len   = self.patch_len + self.stride*(n_patches-1)
        x = torch.nn.functional.pad(x, (0, tgt_len - x.size(2)))
        x = x.unfold(-1, self.patch_len, self.stride)       # [B,C,N,P]
        return x.transpose(1,2), n_patches                  # [B,N,C,P]

class LearnablePositionalEmbedding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = torch.nn.Parameter(torch.zeros(1, max_len, d_model))
        torch.nn.init.trunc_normal_(self.pe, std=0.02)
    def forward(self, x):  # x: [B, N, D]
        return self.pe[:, :x.size(1), :]

class AddNorm(torch.nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm    = torch.nn.LayerNorm(d_model)
    def forward(self, new, old): return self.norm(old + self.dropout(new))

class BiMambaLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout, d_state, d_conv, expand, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        # 根据CUDA可用性选择Mamba实现
        if USE_FALLBACK_MAMBA:
            self.mamba_f = FallbackMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True)
            if bidirectional:
                self.mamba_b = FallbackMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True)
        else:
            self.mamba_f = MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True)
            if bidirectional:
                self.mamba_b = MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True)

        self.add_f = AddNorm(d_model, dropout)
        self.add_b = AddNorm(d_model, dropout) if bidirectional else None
        self.ffn   = torch.nn.Sequential(torch.nn.Linear(d_model, d_ff), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(d_ff, d_model))
        self.add_ff = AddNorm(d_model, dropout)

    def forward(self, x):                                     # x:[B,N,D]
        out_f = self.add_f(self.mamba_f(x), x)
        if self.bidirectional:
            out_b = self.mamba_b(x.flip(1)).flip(1)
            x = .5*(out_f + self.add_b(out_b, x))
        else:
            x = out_f
        return self.add_ff(self.ffn(x), x)

class BiMambaBackbone(torch.nn.Module):
    def __init__(self, c_in, seq_len, patch_len, stride, d_model, n_layers, d_ff, dropout, d_state, d_conv, expand, bidirectional):
        super().__init__()
        self.patching = Patching(patch_len, stride)
        max_patches = (max(seq_len, patch_len)-patch_len)//stride+1
        self.pos   = LearnablePositionalEmbedding(max_patches, d_model)
        self.proj  = torch.nn.Linear(c_in*patch_len, d_model)
        self.layers= torch.nn.ModuleList([BiMambaLayer(d_model, d_ff, dropout, d_state, d_conv, expand, bidirectional) for _ in range(n_layers)])
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x):                              # x:[B,C,L]
        patches,n = self.patching(x)
        h = self.proj(patches.flatten(2)) + self.pos(patches)
        h = self.dropout(h)
        for layer in self.layers: h = layer(h)
        return h                                       # [B,N,D]

class BiMambaPowerModel(torch.nn.Module):
    def __init__(self, c_in, seq_len, pred_len, patch_len, stride, d_model, n_layers, d_ff, dropout, d_state, d_conv, expand, bidirectional=True):
        super().__init__()
        self.total_input_len = seq_len + pred_len
        self.backbone = BiMambaBackbone(c_in, self.total_input_len, patch_len, stride, d_model, n_layers, d_ff, dropout, d_state, d_conv, expand, bidirectional)
        n_patches = (max(self.total_input_len, patch_len)-patch_len)//stride+1
        flat = n_patches*d_model
        self.head_p = torch.nn.Linear(flat, pred_len)
        self.head_n = torch.nn.Linear(flat, pred_len)
    def forward(self, x):                      # x:[B, total_input_len, C]
        h = self.backbone(x.permute(0,2,1)).flatten(1)
        return self.head_p(h), self.head_n(h)

def prepare_simple_features(df, is_past_data=False):
    """简化的特征准备函数，只处理原始字段（参照train_a.py的实现）"""
    df = df.copy()

    # 确保时间列存在并转换为datetime
    if 'time' not in df.columns and 'energy_date' in df.columns:
        df = df.rename(columns={'energy_date': 'time'})
    elif 'energy_date' not in df.columns and 'time' in df.columns:
        df = df.rename(columns={'time': 'energy_date'})

    # 确保有时间列用于特征提取，并且保证 'time' 列始终存在
    time_col = 'energy_date' if 'energy_date' in df.columns else 'time'
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        # 确保 'time' 列存在
        if 'time' not in df.columns:
            df['time'] = df[time_col]

        # 添加时间特征分解（与train_a.py保持一致）
        df['day_of_week'] = df[time_col].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df[time_col].dt.day
        df['day_of_year'] = df[time_col].dt.dayofyear
        df['week_of_year'] = df[time_col].dt.isocalendar().week
        df['hour'] = df[time_col].dt.hour
        df['minute'] = df[time_col].dt.minute
    else:
        # 如果没有时间列，设置默认值
        df['day_of_week'] = 0
        df['day_of_month'] = 0
        df['day_of_year'] = 0
        df['week_of_year'] = 0
        df['hour'] = 0
        df['minute'] = 0
        # 确保 'time' 列存在
        if 'time' not in df.columns:
            df['time'] = pd.Timestamp.now()

    # 处理缺失值和数据类型（与train_a.py保持一致）
    if 'quarter' in df.columns:
        df['quarter'] = df['quarter'].fillna(0).astype(int)
    else:
        df['quarter'] = 0

    if 'holiday' in df.columns:
        df['holiday'] = df['holiday'].fillna(0).astype(int)
    else:
        df['holiday'] = 0

    if 'is_peak' in df.columns:
        df['is_peak'] = df['is_peak'].fillna(0).astype(int)
    else:
        df['is_peak'] = 0

    if 'text' in df.columns:
        df['text'] = df['text'].fillna('unknown')
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['text'] = le.fit_transform(df['text'].astype(str))
    else:
        df['text'] = 0

    if 'code' in df.columns:
        df['code'] = df['code'].fillna(0).astype(int)
    else:
        df['code'] = 0

    # 数值型字段填充缺失值
    numeric_cols = ['temperature', 'humidity', 'windSpeed', 'cloud']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    # --- MODIFIED: 根据是否为过去数据决定如何处理目标变量 ---
    if is_past_data:
        # 对于过去数据，保留 NaN 值（与 train_a.py 保持一致）
        if 'load' not in df.columns:
            df['load'] = np.nan
        if 'meter' not in df.columns:
            df['meter'] = np.nan
        # 不再使用 .fillna(0.0)，保留 NaN
    else:
        # 对于未来数据，仍然填充为 0
        if 'load' not in df.columns:
            df['load'] = 0.0
        if 'meter' not in df.columns:
            df['meter'] = 0.0
        df['load'] = df['load'].fillna(0.0)
        df['meter'] = df['meter'].fillna(0.0)

    return df

def prepare_data_for_prediction(past_data, future_data, feature_cols, knowable_future_features, unknowable_future_features):
    """准备预测数据（参照train_a.py的简化特征工程）"""
    # 合并过去和未来数据
    past_data = past_data.copy()
    future_data = future_data.copy()

    # 确保列名一致
    past_data.columns = [col.lower() for col in past_data.columns]
    future_data.columns = [col.lower() for col in future_data.columns]

    # 确保time列存在
    if 'time' not in past_data.columns and 'energy_date' in past_data.columns:
        past_data = past_data.rename(columns={'energy_date': 'time'})
    if 'time' not in future_data.columns and 'energy_date' in future_data.columns:
        future_data = future_data.rename(columns={'energy_date': 'time'})

    # 确保时间列是datetime类型
    past_data['time'] = pd.to_datetime(past_data['time'])
    future_data['time'] = pd.to_datetime(future_data['time'])

    # 排序数据
    past_data = past_data.sort_values('time').reset_index(drop=True)
    future_data = future_data.sort_values('time').reset_index(drop=True)

    if 'load' in future_data.columns:
        future_data['load'] = 0.0
    if 'meter' in future_data.columns:
        future_data['meter'] = 0.0

    # --- MODIFIED: 分别处理过去数据和未来数据 ---
    # 处理过去数据（保留 NaN）
    past_data = prepare_simple_features(past_data, is_past_data=True)

    # 处理未来数据（填充为 0）
    future_data = prepare_simple_features(future_data, is_past_data=False)

    # 合并数据
    combined_data = pd.concat([past_data, future_data], ignore_index=True)

    # 确保时间列存在后再排序
    if 'time' not in combined_data.columns and 'energy_date' in combined_data.columns:
        combined_data['time'] = combined_data['energy_date']
    elif 'energy_date' not in combined_data.columns and 'time' in combined_data.columns:
        combined_data['energy_date'] = combined_data['time']

    # 确保 'time' 列存在后再进行排序
    if 'time' in combined_data.columns:
        combined_data = combined_data.sort_values('time').reset_index(drop=True)
    else:
        logger.warning("合并后的数据中没有 'time' 列，跳过排序")

    # --- MODIFIED: 只对非目标变量的特征列进行填充 ---
    # 对于特征列（非 load/meter），填充缺失值
    feature_only_cols = [col for col in feature_cols if col not in ['load', 'meter']]
    for col in feature_only_cols:
        if col in combined_data.columns:
            combined_data[col] = combined_data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # 确保所有特征都存在
    for feature in feature_cols:
        if feature not in combined_data.columns:
            combined_data[feature] = 0.0

    return combined_data

def predict_power(station_id, start_datetime, model_base_path=None, data_path=None, config_path="config.yml"):
    """
    预测场站功率

    参数:
    station_id: 场站ID
    start_datetime: 预测开始日期时间，格式为"2025-01-01 00:00:00"
    model_base_path: 模型文件基础路径，如果为None则从配置文件读取
    data_path: 数据文件基础路径，如果为None则从配置文件读取

    返回:
    预测结果数组，包含time, load, meter字段
    """
    try:
        # 如果没有传入路径参数，则从配置文件加载
        if model_base_path is None or data_path is None:
            config = load_config(config_path)
            if model_base_path is None:
                model_base_path = config['output'].get('model_path', 'model')
            if data_path is None:
                data_path = config['output'].get('path', 'output')

        output_path = data_path

        # 转换开始时间为datetime对象
        start_dt = pd.to_datetime(start_datetime)

        # 计算过去7天的开始时间
        past_start_dt = start_dt - pd.Timedelta(days=7)

        # 计算未来7天的结束时间
        future_end_dt = start_dt + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)

        # 检查场站ID目录是否存在
        station_dir = os.path.join(output_path, str(station_id))
        if not os.path.exists(station_dir):
            logger.error(f"场站ID {station_id} 目录不存在: {station_dir}")
            return []

        # 检查模型目录是否存在
        model_dir = os.path.join(model_base_path, str(station_id), "lstm")
        if not os.path.exists(model_dir):
            logger.error(f"模型目录不存在: {model_dir}")
            return []

        # 加载模型和相关文件（带简单缓存）
        try:
            # 直接加载 AutoTS 模型与配置
            cache_key = str(station_id)
            meter_model_path = os.path.join(model_dir, 'autots_meter.pkl')
            load_model_path = os.path.join(model_dir, 'autots_load.pkl')
            config_path_file = os.path.join(model_dir, 'config.pkl')

            if not (os.path.exists(meter_model_path) and os.path.exists(load_model_path) and os.path.exists(config_path_file)):
                logger.error("缺少必要的 AutoTS 模型或配置文件")
                return []

            model_mtime = max(os.path.getmtime(p) for p in [meter_model_path, load_model_path, config_path_file])
            bundle = _MODEL_CACHE.get(cache_key)

            if bundle and bundle.get('mtime') == model_mtime:
                feature_cols = bundle['feature_cols']
                expected_features = bundle['expected_features']
                knowable_future_features = bundle['knowable_future_features']
                unknowable_future_features = bundle['unknowable_future_features']
                model_config = bundle['model_config']
                model_meter = bundle['model_meter']
                model_load = bundle['model_load']
                logger.info("复用缓存的 AutoTS 模型与资源")
            else:
                # 加载特征列表
                feature_cols = joblib.load(os.path.join(model_dir, 'enc_cols.pkl'))
                expected_features = ['quarter','holiday','is_peak','text','code','temperature','humidity','windSpeed','cloud',
                                     'day_of_week','day_of_month','day_of_year','week_of_year','hour','minute']
                if not all(f in feature_cols for f in expected_features):
                    logger.warning(f"加载的特征列表不完整，使用预期的特征列表: {expected_features}")
                    feature_cols = expected_features
                feature_classification = joblib.load(os.path.join(model_dir, 'feature_classification.pkl'))
                knowable_future_features = feature_classification.get('knowable_future_features', expected_features)
                unknowable_future_features = feature_classification.get('unknowable_future_features', [])
                model_config = joblib.load(config_path_file)

                # 加载 AutoTS 模型
                model_meter = joblib.load(meter_model_path)
                model_load = joblib.load(load_model_path)

                _MODEL_CACHE[cache_key] = {
                    'mtime': model_mtime,
                    'feature_cols': feature_cols,
                    'expected_features': expected_features,
                    'knowable_future_features': knowable_future_features,
                    'unknowable_future_features': unknowable_future_features,
                    'model_config': model_config,
                    'model_meter': model_meter,
                    'model_load': model_load,
                }

        except Exception as e:
            logger.error(f"加载模型文件失败: {e}")
            return []

        # 1. 读取过去7天的历史数据
        past_data = None
        past_month = past_start_dt.strftime('%Y-%m')
        past_data_path = os.path.join(station_dir, 'data', f'data-{station_id}-{past_month}.csv')

        # 如果跨月，可能需要读取两个月的数据
        if past_start_dt.month != start_dt.month:
            current_month = start_dt.strftime('%Y-%m')
            current_data_path = os.path.join(station_dir, 'data', f'data-{station_id}-{current_month}.csv')

            try:
                past_month_data = pd.read_csv(past_data_path) if os.path.exists(past_data_path) else pd.DataFrame()
                current_month_data = pd.read_csv(current_data_path) if os.path.exists(current_data_path) else pd.DataFrame()
                past_data = pd.concat([past_month_data, current_month_data], ignore_index=True)
            except Exception as e:
                logger.error(f"读取历史数据失败: {e}")
                # 尝试读取全部数据
                all_data_path = os.path.join(station_dir, 'data', 'all', f'data-{station_id}-all.csv')
                if os.path.exists(all_data_path):
                    past_data = pd.read_csv(all_data_path)
                else:
                    logger.error("无法找到历史数据文件")
                    return []
        else:
            # 尝试读取单月数据
            if os.path.exists(past_data_path):
                past_data = pd.read_csv(past_data_path)
            else:
                # 尝试读取全部数据
                all_data_path = os.path.join(station_dir, 'data', 'all', f'data-{station_id}-all.csv')
                if os.path.exists(all_data_path):
                    past_data = pd.read_csv(all_data_path)
                else:
                    logger.error("无法找到历史数据文件")
                    return []

        # 确保列名一致性
        if 'time' not in past_data.columns and 'energy_date' in past_data.columns:
            past_data = past_data.rename(columns={'energy_date': 'time'})

        # 转换时间列为datetime
        past_data['time'] = pd.to_datetime(past_data['time'])

        # 过滤出过去7天的数据
        past_data = past_data[(past_data['time'] >= past_start_dt) & (past_data['time'] < start_dt)]

        # 检查是否有足够的历史数据
        expected_records = 96 * 7  # 每15分钟一条记录，7天
        if len(past_data) < expected_records:
            logger.warning(f"历史数据不足，预期{expected_records}条记录，实际{len(past_data)}条")

            # 创建完整的时间索引
            full_time_index = pd.date_range(start=past_start_dt, end=start_dt - pd.Timedelta(minutes=15), freq='15min')

            # 创建完整的DataFrame
            full_past_data = pd.DataFrame({'time': full_time_index})

            # 合并实际数据
            full_past_data = pd.merge(full_past_data, past_data, on='time', how='left')

            # --- MODIFIED: 对于缺失的历史数据，除了时间字段和预测字段外，其他字段都填充为0 ---
            # 确保目标变量存在但保持 NaN（与 train_a.py 保持一致）
            if 'meter' not in full_past_data.columns:
                full_past_data['meter'] = np.nan
            if 'load' not in full_past_data.columns:
                full_past_data['load'] = np.nan

            # 对于其他非时间字段，填充为0
            for col in past_data.columns:
                if col != 'time' and col not in ['meter', 'load']:
                    full_past_data[col] = full_past_data[col].fillna(0)

            past_data = full_past_data

        # 2. 读取未来数据（只读取预测开始日期的文件，该文件已包含未来7天的数据）
        year_month = start_dt.strftime('%Y-%m')
        day = start_dt.strftime('%d')

        forecaster_path = os.path.join(station_dir, 'forcaster', f'forcaster-{station_id}-{year_month}-{day}.csv')

        if os.path.exists(forecaster_path):
            try:
                future_data = pd.read_csv(forecaster_path)
                logger.info(f"成功读取未来数据文件: {forecaster_path}")
            except Exception as e:
                logger.error(f"读取未来数据失败 {forecaster_path}: {e}")
                return []
        else:
            logger.error(f"未来数据文件不存在: {forecaster_path}")
            return []

        # 确保列名一致性
        if 'time' not in future_data.columns and 'energy_date' in future_data.columns:
            future_data = future_data.rename(columns={'energy_date': 'time'})

        # 转换时间列为datetime
        future_data['time'] = pd.to_datetime(future_data['time'])

        # 检查未来数据的字段
        logger.info(f"未来数据字段: {future_data.columns.tolist()}")

        # 过滤出未来7天的数据
        future_data = future_data[(future_data['time'] >= start_dt) & (future_data['time'] <= future_end_dt)]

        # 检查是否有足够的未来数据
        if len(future_data) < 96 * 7:
            logger.warning(f"未来数据不足，预期{96 * 7}条记录，实际{len(future_data)}条")

            # 创建完整的时间索引
            full_time_index = pd.date_range(start=start_dt, end=future_end_dt, freq='15min')

            # 创建完整的DataFrame
            full_future_data = pd.DataFrame({'time': full_time_index})

            # 合并实际数据
            full_future_data = pd.merge(full_future_data, future_data, on='time', how='left')

            # 填充缺失值
            for col in future_data.columns:
                if col not in full_future_data.columns:
                    full_future_data[col] = 0
                else:
                    full_future_data[col] = full_future_data[col].fillna(0)

            future_data = full_future_data

        # 3. 准备预测数据
        combined_data = prepare_data_for_prediction(past_data, future_data, feature_cols, knowable_future_features, unknowable_future_features)

        # 4. 构建预测窗口
        past_steps = model_config['past_steps']
        future_steps = model_config['future_steps']
        total_len = past_steps + future_steps

        # 确保数据长度足够
        if len(combined_data) < total_len:
            logger.error(f"合并后的数据长度不足，需要{total_len}条记录，实际{len(combined_data)}条")
            return []

        # 确保所有特征列都存在（简化版本，只检查基本特征）
        missing_features = [f for f in feature_cols if f not in combined_data.columns]
        if missing_features:
            logger.warning(f"缺少以下特征列: {missing_features}")
            # 为缺失的特征列添加默认值
            for feature in missing_features:
                combined_data[feature] = 0.0

        # 提取特征（AutoTS无需缩放，这里仅确保列存在）
        feature_data = combined_data[feature_cols].copy()

        # 5. 进行预测（AutoTS）
        try:
            # 仅使用未来可知特征作为回归量
            regressor_cols = [c for c in feature_cols if c in knowable_future_features]
            future_regressor = combined_data[regressor_cols].iloc[past_steps:past_steps+future_steps].reset_index(drop=True)

            pred_power_df = model_meter.predict(future_regressor=future_regressor)
            pred_load_df = model_load.predict(future_regressor=future_regressor)
            pred_power = pred_power_df.forecast.iloc[:, 0].values.astype(float)
            pred_load = pred_load_df.forecast.iloc[:, 0].values.astype(float)
        except Exception as e:
            logger.error(f"AutoTS 预测失败: {e}")
            return []

        # 7. 构建结果数据
        # 确保时间列存在
        time_col = None
        for col in ['time', 'energy_date']:
            if col in combined_data.columns:
                time_col = col
                break

        if time_col is None:
            # 如果没有时间列，根据预测开始时间生成时间序列
            result_times = pd.date_range(start=start_dt, periods=future_steps, freq='15min')
        else:
            result_times = combined_data[time_col].iloc[past_steps:past_steps+future_steps]

        result = []

        for i in range(len(result_times)):
            # 转换numpy.datetime64为pandas.Timestamp，然后格式化
            time_str = pd.to_datetime(result_times.iloc[i] if hasattr(result_times, 'iloc') else result_times[i]).strftime('%Y-%m-%d %H:%M:%S')
            result.append({
                'time': time_str,
                'load': float(pred_load[i]),
                'meter': float(pred_power[i])
            })

        return result

    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def predict_all_stations(start_datetime, config_path="config.yml"):
    """
    预测所有场站的功率
    
    参数:
    start_datetime: 预测开始日期时间，格式为"2025-01-01 00:00:00"
    
    返回:
    预测结果字典，键为场站ID，值为预测结果数组
    """
    try:
        # 加载配置
        config = load_config(config_path)
        output_path = config['output'].get('path', 'output')

        # 获取所有场站ID
        station_ids = []
        for item in os.listdir(output_path):
            if os.path.isdir(os.path.join(output_path, item)) and item.isdigit():
                station_ids.append(item)

        if not station_ids:
            logger.warning("未找到任何场站ID")
            return {}

        # 预测每个场站
        results = {}
        for station_id in station_ids:
            logger.info(f"正在预测场站 {station_id} 的功率...")
            station_result = predict_power(station_id, start_datetime, config_path)
            results[station_id] = station_result

        return results

    except Exception as e:
        logger.error(f"预测所有场站时发生错误: {e}")
        return {}

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("用法: python forcast.py <场站ID> <预测开始日期时间>")
        print("示例: python forcast.py 12345 '2025-01-01 00:00:00'")
        sys.exit(1)

    station_id = sys.argv[1]
    start_datetime = sys.argv[2]

    result = predict_power(station_id, start_datetime)

    if result:
        print(f"成功预测场站 {station_id} 的功率，共 {len(result)} 条记录")
        for i, item in enumerate(result[:5]):
            print(f"样例 {i+1}: {item}")

        # 保存结果到JSON文件
        try:
            # 创建输出目录
            output_dir = "prediction_results"
            os.makedirs(output_dir, exist_ok=True)

            # 生成文件名
            start_dt = pd.to_datetime(start_datetime)
            filename = f"prediction_{station_id}_{start_dt.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(output_dir, filename)

            # 构建保存的数据结构
            save_data = {
                "station_id": station_id,
                "start_datetime": start_datetime,
                "prediction_count": len(result),
                "generated_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "predictions": result
            }

            # 保存到JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            print(f"预测结果已保存到: {filepath}")

        except Exception as e:
            logger.error(f"保存预测结果到JSON文件失败: {e}")
            print(f"保存预测结果失败: {e}")
    else:
        print(f"预测场站 {station_id} 的功率失败")
