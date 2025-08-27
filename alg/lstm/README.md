# Bi-Mamba4TS 增量微调和自动超参数调优功能

本文档介绍了为 Bi-Mamba4TS 功率预测模型新增的增量微调和自动超参数调优功能。

## 📦 依赖安装

### 1. 基础依赖安装
首先安装所有基础依赖,要求py 3.12版本：
```bash
# 安装所有基础依赖
# pytorch指定版本
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
# 自动安装对应版本
pip install -r requirements.txt

### 2. Mamba-SSM 安装（重要）
Mamba-SSM 需要特殊安装方式，请根据您的环境选择以下方法：

#### 方法A：在线安装（推荐）
# 使用清华源安装（需要能连接GitHub）
pip install mamba-ssm causal-conv1d 
```

#### 方法B：离线安装（适用于网络受限环境）
1. 首先确定您的CUDA版本：
```bash
nvidia-smi
```
2. 根据CUDA版本下载对应的whl文件：
```bash
# 示例：CUDA 12.x + Python 3.12 + PyTorch 2.7
wget https://gh.llkk.cc/https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

wget https://gh.llkk.cc/https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

```

3. 安装下载的whl文件：
```bash
pip install *.whl
```

### 3. PyTorch 安装确认
确保安装正确版本的PyTorch：
```bash
# 查看PyTorch版本和CUDA支持
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 6. 故障排除
- **CUDA版本不匹配**：访问 https://pytorch.org/get-started/previous-versions/ 查找对应版本
- **mamba-ssm安装失败**：尝试使用conda安装或检查CUDA兼容性

## 🆕 功能

### 1. 增量微调 (Incremental Fine-tuning)
- **功能描述**: 在已有预训练模型基础上进行微调，适用于模型更新和适应新数据
- **主要特性**:
  - 支持冻结部分层进行微调
  - 使用较小的学习率进行精细调整
  - 自动保存微调历史和最佳模型
  - 支持早停机制

### 2. 自动超参数调优 (Hyperparameter Optimization)
- **功能描述**: 使用 Optuna 框架自动搜索最优超参数组合
- **主要特性**:
  - 支持多种超参数的自动搜索
  - 使用 TPE 采样器和中位数剪枝器
  - 自动保存优化历史和最佳参数
  - 支持超时和试验次数限制


## 🚀 使用方法

### 基础训练 (原有功能)
```bash
python train.py --station_id YOUR_STATION_ID 
```

### 启用自动超参数调优和天数比重
```bash
W=$(python - <<'PY'
import numpy as np
pred_len=96*7
w=np.ones(pred_len); per_day=96
w[0:2*per_day]=1.5; w[2*per_day:4*per_day]=1.2; w[4*per_day:5*per_day]=1.0; w[5*per_day:7*per_day]=0.9
w=w/np.mean(w)
print(','.join(map(str,w.tolist())))
PY
)
python train.py --station_id 1716387625733984256  --horizon_weights "$W" --n_trials 5  --enable_hyperopt
```

### 启用增量微调
```bash
python train_mbm.py --station_id YOUR_STATION_ID --data_file merged_station_test.csv  --enable_incremental
```


## ⚙️ 配置参数

### 增量微调配置
```python
CFG = {
    'incremental_training': False,    # 是否启用增量微调
    'finetune_lr_ratio': 0.1,        # 微调学习率相对于原始学习率的比例
    'finetune_epochs': 50,           # 微调最大轮数
    'finetune_patience': 15,         # 微调早停耐心值
    'freeze_backbone_layers': 2,     # 冻结前N层backbone (0表示不冻结)
}
```

### 超参数调优配置
```python
CFG = {
    'enable_hyperopt': False,        # 是否启用自动超参数调优
    'n_trials': 100,                # 超参数搜索试验次数
    'optuna_timeout': 3600,         # 超参数搜索超时时间(秒)
    'optuna_pruning': True,         # 是否启用早期剪枝
}
```

### 超参数搜索空间
```python
HYPEROPT_SPACE = {
    'd_model': [64, 128, 256, 512],                    # 模型隐藏维度
    'd_ff_ratio': [1.5, 2.0, 3.0, 4.0],              # 前馈网络维度比例
    'n_layers': [2, 3, 4, 5, 6],                     # 模型层数
    'drop_rate': [0.1, 0.2, 0.3, 0.4, 0.5],         # Dropout比率
    'd_state': [16, 32, 64, 128],                     # 状态空间维度
    'd_conv': [3, 4, 5, 6],                          # 卷积核大小
    'expand': [1.5, 2.0, 2.5, 3.0],                 # 扩展因子
    'patch_len': [8, 12, 16, 20, 24],               # Patch长度
    'stride_ratio': [0.25, 0.5, 0.75],              # Stride比例
    'batch_size': [128, 256, 512],                   # 批次大小
    'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],          # 学习率
    'power_weight': [0.8, 1.0, 1.2],                # 总功率损失权重
    'not_use_power_weight': [0.5, 0.7, 0.9, 1.0],  # 不可用功率损失权重
}
```

## 📁 输出文件

### 增量微调输出
- `bi_mamba_finetuned.pth`: 微调后的最佳模型权重
- `finetune_history.json`: 微调训练历史记录

### 超参数调优输出
- `hyperopt_results.json`: 超参数优化结果和历史
- 包含最佳参数、最佳分数和所有试验记录

### 特征分类输出
- `feature_classification.pkl`: 特征分类信息
  - `knowable_future_features`: 可预知的未来特征
  - `unknowable_future_features`: 不可预知的未来特征

## 🔄 工作流程

### 1. 超参数调优流程
```
数据加载 → 特征工程 → 超参数搜索 → 使用最佳参数训练 → 增量微调(可选) → 评估
```

### 2. 增量微调流程
```
加载预训练模型 → 冻结指定层 → 小学习率微调 → 保存最佳模型 → 解冻所有层
```

## 📊 性能监控

### 超参数调优监控
- 实时显示搜索进度
- 自动剪枝低性能试验
- 保存完整的优化历史

### 增量微调监控
- 显示微调进度和损失变化
- 早停机制防止过拟合
- 保存训练历史供分析

## 🎯 最佳实践

### 1. 超参数调优建议
- 首次使用建议设置较多试验次数 (100-200)
- 根据计算资源调整超时时间
- 启用剪枝可以加速搜索过程

### 2. 增量微调建议
- 确保有预训练模型存在
- 冻结层数根据数据变化程度调整
- 微调学习率通常设为原学习率的 0.1-0.3 倍

### 3. 组合使用建议
- 先进行超参数调优找到最佳配置
- 再使用增量微调进行精细调整
- 定期重新进行超参数搜索

## 🐛 故障排除

### 常见问题
1. **Optuna 未安装**: 安装 `pip install optuna`
2. **VMD 未安装**: 安装 `pip install vmdpy`
3. **预训练模型不存在**: 先运行基础训练生成模型
4. **内存不足**: 减少 batch_size 或 n_trials

### 日志信息
- `⚠️` 警告信息: 缺少依赖或配置问题
- `🔍` 超参数搜索进度
- `🔄` 增量微调进度
- `✅` 成功完成信息

## 📈 示例结果

### 超参数调优结果示例
```json
{
  "best_params": {
    "d_model": 256,
    "n_layers": 4,
    "lr": 0.0001,
    "batch_size": 256
  },
  "best_score": 0.0234,
  "n_trials": 100
}
```

### 增量微调历史示例
```json
[
  {
    "epoch": 1,
    "train_loss": 0.0456,
    "val_loss": 0.0398,
    "lr": 0.00001
  }
]
```

通过这些新功能，您可以更好地优化模型性能并适应不断变化的数据分布。