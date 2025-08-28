#!/usr/bin/env python3
# =========================================================
#  单站点功率预测 —— Bi-Mamba4TS 简化版（使用过去7天+未来7天外部特征）
#  - 训练策略：使用过去7天历史数据 + 未来7天已知外部特征（天气预报、节假日）
#  - 过去7天：包含所有特征（负荷、天气、节假日等）
#  - 未来7天：仅使用可预知的外部特征（天气预报、节假日、时间特征等）
#  - 原始字段：time,quarter,holiday,is_peak,text,code,temperature,humidity,windSpeed,cloud,load,meter
# =========================================================

import os, warnings, time, argparse, gc
import numpy  as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics      import mean_absolute_percentage_error, mean_squared_error
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data      import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from util.logger import logger
import joblib
import yaml
import logging
import datetime
import shutil

warnings.filterwarnings("ignore")

# 加载配置文件
def load_config(config_path):
    """加载配置文件"""
    logger = logging.getLogger(__name__)
    logger.info(f"正在加载配置文件 {config_path}...")

    if not os.path.exists(config_path):
        logger.warning(f"配置文件 {config_path} 不存在，将使用默认配置")
        return {
            "task": {"cron": "0 0 1 * *"},
            "output": {
                "path": "output",
                "model_path": "model"
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
            "task": {"cron": "0 0 1 * *"},
            "output": {
                "path": "output",
                "model_path": "model"
            }
        }
# --- optional deps ---
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# Bi-Mamba (优先官方 mamba_ssm)
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except Exception:
    MAMBA_AVAILABLE = False
    logger.info("mamba_ssm 未找到，使用脚本内 fallback Mamba（仅用于快速测试，建议在生产安装官方包）。")
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_fast_path=True):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.expand_dim = int(expand * d_model)
            self.in_proj = nn.Linear(d_model, self.expand_dim * 2)
            self.conv = nn.Conv1d(self.expand_dim, self.expand_dim, d_conv, padding=d_conv-1, groups=self.expand_dim)
            self.A = nn.Parameter(torch.randn(self.expand_dim, d_state) * 0.01)
            self.C = nn.Parameter(torch.randn(self.expand_dim, d_state) * 0.01)
            self.D = nn.Parameter(torch.ones(self.expand_dim) * 0.01)
            self.out_proj = nn.Linear(self.expand_dim, d_model)
        def forward(self, x):
            B, L, _ = x.shape
            x_and_z = self.in_proj(x)
            x_proj, z = x_and_z.chunk(2, dim=-1)
            x_conv = self.conv(x_proj.transpose(1,2))[:, :, :L].transpose(1,2)
            x_s4 = torch.zeros_like(x_conv)
            h = torch.zeros(B, self.A.shape[1], device=x.device)
            for i in range(L):
                u = torch.einsum('be,ed->bd', x_conv[:, i], self.A)
                h = torch.tanh(u + h)
                s4_out = torch.einsum('bd,ed->be', h, self.C) + self.D * x_conv[:, i]
                x_s4[:, i] = s4_out
            y = torch.sigmoid(z) * x_s4
            return self.out_proj(y)

# ---------- 配置 ----------
CFG = dict(
    # === 时间序列窗口配置 ===
    past_steps   = 96*7,        # 历史回看步长 (7天，每天96个15分钟点)
    future_steps = 96*7,        # 未来预测步长 (7天，每天96个15分钟点)

    # === Patch化配置 ===
    patch_len    = 16,          # 每个patch的长度 (将时间序列分割成小块)
    stride       = 8,           # patch之间的步长 (控制patch重叠程度)

    # === Mamba模型架构配置 ===
    d_model      = 128,         # 模型隐藏维度 (特征嵌入维度)
    d_ff         = 256,         # 前馈网络隐藏层维度 (通常是d_model的2倍)
    n_layers     = 5,           # Bi-Mamba层数 (模型深度)
    drop_rate    = 0.5,         # Dropout比率 (防止过拟合)

    # === Mamba状态空间模型配置 ===
    d_state      = 64,          # 状态空间维度 (控制模型记忆容量)
    d_conv       = 4,           # 卷积核大小 (局部特征提取)
    expand       = 2,           # 扩展因子 (内部维度扩展倍数)
    bidirectional= True,        # 是否使用双向Mamba (前向+后向)

    # === 训练配置 ===
    batch_size   = 128,         # 批次大小 (每次训练的样本数)
    epochs       = 75,         # 最大训练轮数
    patience     = 15,          # 早停耐心值 (验证损失不改善的轮数)
    lr           = 1e-4,        # 学习率 (Adam优化器)

    # === 损失函数权重配置 ===
    power_weight = 1.2,         # 总功率预测损失权重
    not_use_power_weight = 0.7,  # 不可用功率预测损失权重 (相对较小)
)
# 超参数搜索空间定义
HYPEROPT_SPACE = {
    'd_model': [64, 128, 256, 512],
    'd_ff_ratio': [1.5, 2.0, 3.0, 4.0],  # d_ff = d_model * d_ff_ratio
    'n_layers': [2, 3, 4, 5, 6],
    'drop_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'd_state': [16, 32, 64, 128],
    'd_conv': [2, 4],
    'expand': [1.5, 2.0, 2.5, 3.0],
    'patch_len': [8, 12, 16, 20, 24],
    'stride_ratio': [0.25, 0.5, 0.75],  # stride = patch_len * stride_ratio
    'batch_size': [128],
    'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    'power_weight': [0.8, 1.0, 1.2],
    'not_use_power_weight': [0.5, 0.7, 0.9, 1.0],
}
# =========================================================
# 网络模块（patching + learnable pos emb + BiMamba）
# =========================================================
class Patching(nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
    def forward(self, x):                                   # x:[B,C,L]
        n_patches = (max(x.size(2), self.patch_len)-self.patch_len)//self.stride+1
        tgt_len   = self.patch_len + self.stride*(n_patches-1)
        x = F.pad(x, (0, tgt_len - x.size(2)))
        x = x.unfold(-1, self.patch_len, self.stride)       # [B,C,N,P]
        return x.transpose(1,2), n_patches                  # [B,N,C,P]

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)
    def forward(self, x):  # x: [B, N, D]
        return self.pe[:, :x.size(1), :]

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)
    def forward(self, new, old): return self.norm(old + self.dropout(new))

class BiMambaLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, d_state, d_conv, expand, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True)
        if bidirectional:
            self.mamba_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True)
        self.add_f = AddNorm(d_model, dropout)
        self.add_b = AddNorm(d_model, dropout) if bidirectional else None
        self.ffn   = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.add_ff = AddNorm(d_model, dropout)
    def forward(self, x):                                     # x:[B,N,D]
        out_f = self.add_f(self.mamba_f(x), x)
        if self.bidirectional:
            out_b = self.mamba_b(x.flip(1)).flip(1)
            x = .5*(out_f + self.add_b(out_b, x))
        else:
            x = out_f
        return self.add_ff(self.ffn(x), x)

class BiMambaBackbone(nn.Module):
    def __init__(self, c_in, seq_len, patch_len, stride, d_model, n_layers, d_ff, dropout, d_state, d_conv, expand, bidirectional):
        super().__init__()
        self.patching = Patching(patch_len, stride)
        max_patches = (max(seq_len, patch_len)-patch_len)//stride+1
        self.pos   = LearnablePositionalEmbedding(max_patches, d_model)
        self.proj  = nn.Linear(c_in*patch_len, d_model)
        self.layers= nn.ModuleList([BiMambaLayer(d_model, d_ff, dropout, d_state, d_conv, expand, bidirectional) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):                              # x:[B,C,L]
        patches,n = self.patching(x)
        h = self.proj(patches.flatten(2)) + self.pos(patches)
        h = self.dropout(h)
        for layer in self.layers: h = layer(h)
        return h                                       # [B,N,D]

class BiMambaPowerModel(nn.Module):
    def __init__(self, c_in, seq_len, pred_len, patch_len, stride, d_model, n_layers, d_ff, dropout, d_state, d_conv, expand, bidirectional=True):
        super().__init__()
        self.total_input_len = seq_len + pred_len
        self.backbone = BiMambaBackbone(c_in, self.total_input_len, patch_len, stride, d_model, n_layers, d_ff, dropout, d_state, d_conv, expand, bidirectional)
        n_patches = (max(self.total_input_len, patch_len)-patch_len)//stride+1
        flat = n_patches*d_model
        self.head_p = nn.Linear(flat, pred_len)
        self.head_n = nn.Linear(flat, pred_len)
    def forward(self, x):                      # x:[B, total_input_len, C]
        h = self.backbone(x.permute(0,2,1)).flatten(1)
        return self.head_p(h), self.head_n(h)

# =========================================================
#  辅助类和函数
# =========================================================
class WeightedSmoothL1(nn.Module):
    def __init__(self, fut, w):
        super().__init__()
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32)[:fut])
        self.crit = nn.SmoothL1Loss(reduction='none')
    def forward(self, pred, target):
        loss = self.crit(pred, target)
        weighted_loss = self.w * loss
        return weighted_loss  # 返回标量值

# =========================================================
# 全局占位
# =========================================================
feature_cols = None

# =========================================================
# 主流程
# =========================================================
def main(station_id=None, data_file='merged_station_test.csv', enable_hyperopt=False, enable_incremental=False, args_cli=None, output_path=None, model_base_path=None):
    global feature_cols
    tic=time.time()
    #赋值超参数调优
    if enable_hyperopt:
        CFG['enable_hyperopt'] = True
        if CFG.get('n_trials') is None:
            CFG['n_trials'] = 10
        CFG['optuna_timeout'] = 3600
    # CLI 参数
    sched_type = getattr(args_cli, 'sched_type', 'cosine')
    max_lr = getattr(args_cli, 'max_lr', CFG['lr'])
    weight_decay = getattr(args_cli, 'weight_decay', 1e-5)
    horizon_weights_csv = getattr(args_cli, 'horizon_weights', None)

    # 如果没有传递路径参数，则从配置文件加载
    if output_path is None or model_base_path is None:
        root_config = load_config("config.yml")
        if output_path is None:
            output_path = root_config['output'].get('path', 'output')
        if model_base_path is None:
            model_base_path = root_config['output'].get('model_path', 'model')

    if station_id:
        # 创建模型保存路径
        model_save_dir = f"{model_base_path}/{station_id}/lstm"
        os.makedirs(model_save_dir, exist_ok=True)

        # 检查目录中是否已有文件，如果有则移动到历史模型目录
        history_dir = f"{model_base_path}/{station_id}/history/lstm"
        os.makedirs(history_dir, exist_ok=True)

        if os.path.exists(model_save_dir) and os.listdir(model_save_dir):
            # 创建带时间戳的历史目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            history_dir_with_timestamp = f"{history_dir}/{timestamp}"
            os.makedirs(history_dir_with_timestamp, exist_ok=True)

            # 移动现有文件到历史目录
            for file_name in os.listdir(model_save_dir):
                src_file = os.path.join(model_save_dir, file_name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, os.path.join(history_dir_with_timestamp, file_name))
                    logger.info(f"已将文件 {file_name} 备份到历史目录 {history_dir_with_timestamp}")

        data_file = f"{output_path}/{station_id}/data/all/data-{station_id}-all.csv"

    logger.info(f"加载 {data_file} ...")
    df = pd.read_csv(data_file, low_memory=False)

    # 统一时间列名
    if 'time' in df.columns and 'energy_date' not in df.columns:
        df = df.rename(columns={'time':'energy_date'})
    if 'energy_date' not in df.columns:
        for c in df.columns:
            if 'date' in c.lower() or 'time' in c.lower():
                df = df.rename(columns={c:'energy_date'}); break

    df['energy_date'] = pd.to_datetime(df['energy_date'])
    df = df.sort_values('energy_date').reset_index(drop=True)

    if not station_id:
        path_parts = data_file.split('/')
        for i, part in enumerate(path_parts):
            if part == 'output' and i + 1 < len(path_parts):
                station_id = path_parts[i + 1]; break
        else:
            station_id = "default_station"

    out_dir=f"{model_base_path}/{station_id}/lstm"; os.makedirs(out_dir,exist_ok=True)

    # 添加时间特征分解
    df['day_of_week'] = df['energy_date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['energy_date'].dt.day
    df['day_of_year'] = df['energy_date'].dt.dayofyear
    df['week_of_year'] = df['energy_date'].dt.isocalendar().week
    df['hour'] = df['energy_date'].dt.hour
    df['minute'] = df['energy_date'].dt.minute


    # 处理缺失值和数据类型
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

    # --- MODIFIED: 使用 np.nan 初始化并保留 NaN 以便创建掩码 ---
    if 'load' not in df.columns:
        df['load'] = np.nan
    if 'meter' not in df.columns:
        df['meter'] = np.nan
    # --- 不再使用 .fillna(0.0) ---
    # df['load'] = df['load'].fillna(0.0)
    # df['meter'] = df['meter'].fillna(0.0)

    feature_cols = ['quarter', 'holiday', 'is_peak', 'text', 'code', 'temperature', 'humidity', 'windSpeed', 'cloud',
                    'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year', 'hour', 'minute']

    knowable_future_features = feature_cols.copy()
    unknowable_future_features = []

    logger.info(f"使用原始特征: {feature_cols}")
    logger.info(f"特征总数: {len(feature_cols)}")

    # 划分窗口与 scaler（fit 只在 train_df 上）
    hold_out=df.iloc[-CFG['future_steps']:]
    train_df=df.iloc[:-CFG['future_steps']]
    if len(train_df) < CFG['past_steps'] + CFG['future_steps']:
        logger.info(f" 错误: 训练数据太少，无法创建至少一个完整的输入窗口。")
        return

    # Scaler 拟合时会忽略NaN，这是正确的行为
    sc_e=RobustScaler().fit(train_df[feature_cols])
    sc_y_p=RobustScaler().fit(train_df[['meter']])
    sc_y_n=RobustScaler().fit(train_df[['load']])

    # <--- MODIFIED: 函数现在返回 X, Yp, Yn, Mask_p, Mask_n ---
    def build_windows_local(data, past, fut):
        """
        构建训练窗口：
        - 过去7天的历史数据（包括负荷数据）
        - 未来7天的已知外部特征（天气预报、节假日等，不包括负荷数据）
        """
        total_len = past + fut

        # 分离历史特征和未来外部特征
        # 历史特征：包含所有特征（包括负荷相关）
        historical_features = feature_cols.copy()
        future_external_features = knowable_future_features

        X, Yp, Yn, Mask_p, Mask_n = [], [], [], [], []
        for i in range(len(data) - total_len + 1):
            past_data = data.iloc[i : i + past]
            past_features = sc_e.transform(past_data[historical_features])
            future_data = data.iloc[i + past : i + total_len]
            future_all_features = sc_e.transform(future_data[feature_cols])
            future_features = future_all_features.copy()
            for j, col in enumerate(feature_cols):
                if col not in future_external_features:
                    future_features[:, j] = 0.0
            combined_features = np.vstack([past_features, future_features])
            X.append(combined_features)

            # --- ADDED: 创建目标值和掩码 ---
            yp_slice = future_data[['meter']].values
            yn_slice = future_data[['load']].values

            # 掩码: 1.0 代表有效值, 0.0 代表 NaN
            mask_p = 1.0 - pd.isna(yp_slice).astype(float)
            mask_n = 1.0 - pd.isna(yn_slice).astype(float)

            # 用0替换NaN，因为它们不会参与损失计算
            Yp.append(np.nan_to_num(sc_y_p.transform(yp_slice)).flatten())
            Yn.append(np.nan_to_num(sc_y_n.transform(yn_slice)).flatten())
            Mask_p.append(mask_p.flatten())
            Mask_n.append(mask_n.flatten())

        X = np.array(X, dtype=np.float32)
        Yp = np.array(Yp, dtype=np.float32)
        Yn = np.array(Yn, dtype=np.float32)
        Mask_p = np.array(Mask_p, dtype=np.float32) # <--- ADDED
        Mask_n = np.array(Mask_n, dtype=np.float32) # <--- ADDED

        if X.shape[0] > 0:
            assert X.shape[1] == total_len, f"窗口长度错误: got {X.shape[1]}, expected {total_len}"
        return X, Yp, Yn, Mask_p, Mask_n # <--- MODIFIED

    # <--- MODIFIED: 解包出掩码 ---
    X_tr, Yp_tr, Yn_tr, Mask_p_tr, Mask_n_tr = build_windows_local(train_df, CFG['past_steps'], CFG['future_steps'])
    val_raw_len = CFG['past_steps'] + CFG['future_steps']; val_raw = train_df.iloc[-val_raw_len:]
    X_va, Yp_va, Yn_va, Mask_p_va, Mask_n_va = build_windows_local(val_raw, CFG['past_steps'], CFG['future_steps'])

    assert X_tr.shape[1] == CFG['past_steps'] + CFG['future_steps'], "训练窗口长度不匹配！"
    assert X_va.shape[1] == CFG['past_steps'] + CFG['future_steps'], "验证窗口长度不匹配！"

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device:", device)

    # =========================================================
    # 6) 超参数优化 (可选)
    # =========================================================

    # 如果启用超参数优化，先进行搜索
    if CFG.get('enable_hyperopt', False):
        from alg.lstm.incremental_training_new import apply_hyperparameter_optimization

        train_data = (X_tr, Yp_tr, Yn_tr, Mask_p_tr, Mask_n_tr)
        val_data = (X_va, Yp_va, Yn_va, Mask_p_va, Mask_n_va)
        scalers = {'enc': sc_e, 'y_power': sc_y_p, 'y_not_use': sc_y_n}

        best_params, best_score = apply_hyperparameter_optimization(
            train_data, val_data, feature_cols, scalers, device, out_dir, CFG
        )

        if best_params:
            logger.info(f"\n 使用优化后的超参数进行训练")
            # 更新配置
            for key, value in best_params.items():
                if key in ['d_model', 'n_layers', 'drop_rate', 'd_state', 'd_conv', 'expand', 'patch_len', 'stride', 'batch_size', 'lr', 'power_weight', 'not_use_power_weight']:
                    CFG[key] = value
                elif key == 'd_ff':
                    CFG['d_ff'] = value

        # 重新创建数据加载器（如果batch_size改变）
        tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Yp_tr), torch.from_numpy(Yn_tr), torch.from_numpy(Mask_p_tr), torch.from_numpy(Mask_n_tr))
        tr_loader = DataLoader(tr_ds, batch_size=CFG['batch_size'], shuffle=True, pin_memory=True)

        va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Yp_va), torch.from_numpy(Yn_va), torch.from_numpy(Mask_p_va), torch.from_numpy(Mask_n_va))
        va_loader = DataLoader(va_ds, batch_size=CFG['batch_size'], pin_memory=True)
    else:
        tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Yp_tr), torch.from_numpy(Yn_tr), torch.from_numpy(Mask_p_tr), torch.from_numpy(Mask_n_tr))
        tr_loader = DataLoader(tr_ds, batch_size=CFG['batch_size'], shuffle=True, pin_memory=True)

        va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Yp_va), torch.from_numpy(Yn_va), torch.from_numpy(Mask_p_va), torch.from_numpy(Mask_n_va))
        va_loader = DataLoader(va_ds, batch_size=CFG['batch_size'], pin_memory=True)

    # =========================================================
    # 模型创建 & 训练
    # =========================================================
    model=BiMambaPowerModel(c_in=len(feature_cols), seq_len=CFG['past_steps'], pred_len=CFG['future_steps'], patch_len=CFG['patch_len'], stride=CFG['stride'],
                            d_model=CFG['d_model'], n_layers=CFG['n_layers'], d_ff=CFG['d_ff'], dropout=CFG['drop_rate'], d_state=CFG['d_state'],
                            d_conv=CFG['d_conv'], expand=CFG['expand'], bidirectional=CFG['bidirectional']).to(device)

    def default_weight_vector(fut):
        w = np.concatenate([np.ones(96*2)*2.0, np.ones(96)*1.3, np.ones(96)*1.5, np.ones(96*3)*1.2])
        return w[:fut]

    if horizon_weights_csv:
        try:
            wvec = np.array([float(x) for x in horizon_weights_csv.split(',')], dtype=float)[:CFG['future_steps']]
        except Exception as e:
            wvec = default_weight_vector(CFG['future_steps'])
    else:
        wvec = default_weight_vector(CFG['future_steps'])

    crit = WeightedSmoothL1(CFG['future_steps'], wvec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = None
    if sched_type == 'onecycle':
        steps_per_epoch = max(1, len(tr_loader))
        scheduler = OneCycleLR(opt, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=CFG['epochs'], pct_start=0.1, anneal_strategy='cos')
    else:
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=max_lr*1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    patience = CFG['patience']
    wait = 0
    best_val = float('inf')
    logger.info("开始训练 ...")
    epsilon = 1e-9 # 防止除以零
    for ep in range(1, CFG['epochs']+1):
        model.train()
        tl = 0.0
        n_batches = 0
        # <--- MODIFIED: 从 loader 中解包出掩码 ---
        for xe, yp, yn, mask_p, mask_n in tr_loader:
            xe, yp, yn = xe.to(device), yp.to(device), yn.to(device)
            mask_p, mask_n = mask_p.to(device), mask_n.to(device) # <--- ADDED: 移动掩码到设备

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                pp, pn = model(xe)
                # <--- MODIFIED: 使用掩码计算损失 ---
                loss_p_raw = crit(pp, yp)
                loss_n_raw = crit(pn, yn)
                loss_p_masked = loss_p_raw * mask_p
                loss_n_masked = loss_n_raw * mask_n
                loss_p = loss_p_masked.sum() / (mask_p.sum() + epsilon)
                loss_n = loss_n_masked.sum() / (mask_n.sum() + epsilon)
                loss = CFG['power_weight'] * loss_p + CFG['not_use_power_weight'] * loss_n

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched_type == 'onecycle' and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            tl += loss.item(); n_batches += 1
        tl = tl / max(1, n_batches)

        model.eval()
        vl = 0.0; v_batches = 0
        with torch.no_grad():
            # <--- MODIFIED: 从 loader 中解包出掩码 ---
            for xe, yp, yn, mask_p, mask_n in va_loader:
                xe, yp, yn = xe.to(device), yp.to(device), yn.to(device)
                mask_p, mask_n = mask_p.to(device), mask_n.to(device) # <--- ADDED
                with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                    pp, pn = model(xe)
                    # <--- MODIFIED: 使用掩码计算损失 ---
                    loss_p_raw = crit(pp, yp)
                    loss_n_raw = crit(pn, yn)
                    loss_p_masked = loss_p_raw * mask_p
                    loss_n_masked = loss_n_raw * mask_n
                    loss_p = loss_p_masked.sum() / (mask_p.sum() + epsilon)
                    loss_n = loss_n_masked.sum() / (mask_n.sum() + epsilon)
                    vloss = CFG['power_weight'] * loss_p + CFG['not_use_power_weight'] * loss_n
                vl += vloss.item(); v_batches += 1
        vl = vl / max(1, v_batches)

        if sched_type != 'onecycle' and isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(ep)

        if ep % 5 == 0 or ep == 1:
            logger.info(f"Epoch {ep:03d}/{CFG['epochs']} | train_loss={tl:.6f} | val_loss={vl:.6f} | lr={opt.param_groups[0]['lr']:.2e}")

        if vl < best_val:
            best_val = vl; wait = 0
            torch.save(model.state_dict(), f"{out_dir}/bi_mamba.pth")
        else:
            wait += 1
            if wait >= patience:
                logger.info("触发早停 (patience reached).")
                break

    logger.info("\n--- 保留集评估 ---")
    if os.path.exists(os.path.join(out_dir, 'bi_mamba.pth')):
        model.load_state_dict(torch.load(os.path.join(out_dir, 'bi_mamba.pth'), map_location=device))
    model.eval()

    total_input_len = CFG['past_steps'] + CFG['future_steps']
    # <--- MODIFIED: 解包时用 _ 忽略不需要的掩码和目标值 ---
    X_test, _, _, _, _ = build_windows_local(df.iloc[-total_input_len:], CFG['past_steps'], CFG['future_steps'])
    assert X_test.shape[1] == total_input_len, "测试窗口长度不匹配！"
    xe_realistic = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            pp_realistic, pn_realistic = model(xe_realistic)

    pred_p = sc_y_p.inverse_transform(pp_realistic.cpu().numpy()).flatten()
    pred_n = sc_y_n.inverse_transform(pn_realistic.cpu().numpy()).flatten()
    tgt_p = hold_out.meter.values
    tgt_n = hold_out.load.values

    def mape(y, yhat):
        # 筛选出真实值大于1且非NaN的数据点进行评估
        valid_mask = (~np.isnan(y)) & (y > 1.) # <--- MODIFIED
        if np.any(valid_mask):
            return mean_absolute_percentage_error(y[valid_mask], yhat[valid_mask]) * 100
        return np.nan

    p_mape = mape(tgt_p, pred_p)
    p_rmse = np.sqrt(mean_squared_error(tgt_p[~np.isnan(tgt_p)], pred_p[~np.isnan(tgt_p)]))
    n_mape = mape(tgt_n, pred_n)
    n_rmse = np.sqrt(mean_squared_error(tgt_n[~np.isnan(tgt_n)], pred_n[~np.isnan(tgt_n)]))

    logger.info("\n--- 评估结果 ---")
    logger.info(f"{station_id} 总功率 MAPE={p_mape:.2f}% RMSE={p_rmse:.2f}")
    logger.info(f"{station_id} 负荷 MAPE={n_mape:.2f}% RMSE={n_rmse:.2f}")

    logger.info("\n--- 每日 MAPE ---")
    daily = {}
    for d in range(7):
        s, e = d*96, (d+1)*96
        dp, pp = tgt_p[s:e], pred_p[s:e]
        dn, pn = tgt_n[s:e], pred_n[s:e]
        date = hold_out.iloc[s].energy_date.strftime('%Y-%m-%d')
        m1, m2 = mape(dp, pp), mape(dn, pn)
        daily[f"day_{d+1}"] = {'date': date, 'power_mape': m1, 'load_mape': m2}
        logger.info(f"{date}: meter {m1:.2f}%  load {m2:.2f}%")

    joblib.dump({'overall': {'power_mape': p_mape, 'load_mape': n_mape}, 'daily': daily}, os.path.join(out_dir, 'mape.pkl'))

    # 保存模型到指定路径
    if station_id:
        # 保存模型和相关文件到指定路径
        model_save_dir = f"{model_base_path}/{station_id}/lstm"
        torch.save(model.state_dict(), f"{model_save_dir}/bi_mamba.pth")
        joblib.dump(sc_e, os.path.join(model_save_dir, 'scaler_enc.pkl'))
        joblib.dump(sc_y_p, os.path.join(model_save_dir, 'scaler_y_power.pkl'))
        joblib.dump(sc_y_n, os.path.join(model_save_dir, 'scaler_y_not_use.pkl'))
        joblib.dump(feature_cols, os.path.join(model_save_dir, 'enc_cols.pkl'))
        joblib.dump({'knowable_future_features': knowable_future_features, 'unknowable_future_features': unknowable_future_features}, os.path.join(model_save_dir, 'feature_classification.pkl'))
        joblib.dump(CFG, os.path.join(model_save_dir, 'config.pkl'))
        joblib.dump({'overall': {'power_mape': p_mape, 'load_mape': n_mape}, 'daily': daily}, os.path.join(model_save_dir, 'mape.pkl'))

        logger.info(f"\n 模型已保存至: {model_save_dir}")
        logger.info(f" 历史模型目录: {model_base_path}/{station_id}/history/lstm")

    logger.info(f"\n 完成！总耗时 {time.time()-tic:.1f}s ; 结果已保存至 {out_dir}")

# =========================================================
# CLI
# =========================================================
if __name__=="__main__":
    ap=argparse.ArgumentParser(description="单站点 Bi-Mamba4TS 功率预测 (简化版：仅使用原始字段)")
    ap.add_argument('--station_id', type=str, help='站点ID')
    ap.add_argument('--data_file', type=str, default='merged_station_test.csv', help='数据文件路径')
    ap.add_argument('--enable_hyperopt', action='store_true', help='启用自动超参数调优')
    ap.add_argument('--enable_incremental', action='store_true', help='启用增量微调')
    ap.add_argument('--n_trials', type=int, default=10, help='超参数搜索试验次数')
    ap.add_argument('--sched_type', choices=['cosine','onecycle'], default='cosine', help='学习率调度策略（默认为 cosine）')
    ap.add_argument('--max_lr', type=float, default=1e-4, help='调度器最大学习率（用于 OneCycle/ Cosine 初始 lr）')
    ap.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW 权重衰减')
    ap.add_argument('--horizon_weights', type=str, default=None, help='预测 horizon 权重，逗号分隔（可选）')
    args = ap.parse_args()

    if args.enable_hyperopt:
        CFG['enable_hyperopt'] = True
    CFG['n_trials'] = args.n_trials

    try:
        main(args.station_id, args.data_file, args.enable_hyperopt, args.enable_incremental, args_cli=args)
    except Exception as e:
        import traceback, sys
        logger.info(f"\n 运行错误: {e}")
        traceback.print_exc(file=sys.stdout)