#!/usr/bin/env python3
# =========================================================
#   增量微调和自动超参数调优模块
#  支持在已有模型基础上进行增量训练和自动超参数搜索
# =========================================================

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy
import joblib

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class IncrementalTrainer:
    """增量训练器 - 支持在预训练模型基础上进行微调"""
    
    def __init__(self, model, device, out_dir: str):
        self.model = model
        self.device = device
        self.out_dir = out_dir
        self.training_history = []
        
    def freeze_layers(self, n_layers: int):
        """冻结模型的前N层"""
        if n_layers <= 0:
            return
            
        print(f" 冻结前 {n_layers} 层 backbone 参数")
        frozen_params = 0
        total_params = 0
        
        # 冻结backbone的前n_layers层
        for i, layer in enumerate(self.model.backbone.layers):
            total_params += sum(p.numel() for p in layer.parameters())
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                    
        print(f"已冻结 {frozen_params:,} / {total_params:,} 个backbone参数")
        
    def unfreeze_all_layers(self):
        """解冻所有层"""
        print(" 解冻所有模型参数")
        for param in self.model.parameters():
            param.requires_grad = True
            
    def load_pretrained_weights(self, model_path: str):
        """加载预训练权重"""
        if os.path.exists(model_path):
            print(f" 加载预训练模型: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            return True
        else:
            print(f"  预训练模型不存在: {model_path}")
            return False
            
    def incremental_train(self, train_loader, val_loader, criterion, 
                         finetune_lr: float, finetune_epochs: int, 
                         finetune_patience: int, freeze_layers: int = 0):
        """执行增量训练"""
        print("\n 开始增量微调...")
        
        # 冻结指定层数
        if freeze_layers > 0:
            self.freeze_layers(freeze_layers)
            
        # 设置微调优化器（使用较小的学习率）
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=finetune_lr, 
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=finetune_patience//3, factor=0.5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        epsilon = 1e-9  # 防止除以零
        
        for epoch in range(1, finetune_epochs + 1):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (xe, yp, yn, mask_p, mask_n) in enumerate(train_loader):
                xe, yp, yn = xe.to(self.device), yp.to(self.device), yn.to(self.device)
                mask_p, mask_n = mask_p.to(self.device), mask_n.to(self.device)
                
                optimizer.zero_grad()
                pp, pn = self.model(xe)
                
                # 使用与原训练相同的损失函数和掩码
                from train import CFG
                loss_p_raw = criterion(pp, yp)
                loss_n_raw = criterion(pn, yn)
                loss_p_masked = loss_p_raw * mask_p
                loss_n_masked = loss_n_raw * mask_n
                loss_p = loss_p_masked.sum() / (mask_p.sum() + epsilon)
                loss_n = loss_n_masked.sum() / (mask_n.sum() + epsilon)
                loss = CFG['power_weight'] * loss_p + CFG['not_use_power_weight'] * loss_n
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for xe, yp, yn, mask_p, mask_n in val_loader:
                    xe, yp, yn = xe.to(self.device), yp.to(self.device), yn.to(self.device)
                    mask_p, mask_n = mask_p.to(self.device), mask_n.to(self.device)
                    
                    pp, pn = self.model(xe)
                    
                    # 使用掩码计算损失
                    loss_p_raw = criterion(pp, yp)
                    loss_n_raw = criterion(pn, yn)
                    loss_p_masked = loss_p_raw * mask_p
                    loss_n_masked = loss_n_raw * mask_n
                    loss_p = loss_p_masked.sum() / (mask_p.sum() + epsilon)
                    loss_n = loss_n_masked.sum() / (mask_n.sum() + epsilon)
                    vloss = CFG['power_weight'] * loss_p + CFG['not_use_power_weight'] * loss_n
                    
                    val_loss += vloss.item()
                    
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # 记录训练历史
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            if epoch % 10 == 0:
                print(f"微调 Epoch {epoch:03d} | train {train_loss:.5f} | valid {val_loss:.5f}")
                
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), f"{self.out_dir}/bi_mamba_finetuned.pth")
            else:
                patience_counter += 1
                if patience_counter >= finetune_patience:
                    print(f" 微调早停! (epoch {epoch})")
                    break
                    
        # 解冻所有层
        self.unfreeze_all_layers()
        
        # 保存训练历史
        with open(f"{self.out_dir}/finetune_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
            
        print(f" 增量微调完成! 最佳验证损失: {best_val_loss:.5f}")
        return best_val_loss


class HyperparameterOptimizer:
    """超参数优化器 - 使用Optuna进行自动超参数搜索"""
    
    def __init__(self, train_data, val_data, feature_cols, scalers, device, out_dir):
        self.train_data = train_data
        self.val_data = val_data
        self.feature_cols = feature_cols
        self.scalers = scalers
        self.device = device
        self.out_dir = out_dir
        self.best_params = None
        self.best_score = float('inf')
        
    def objective(self, trial):
        """Optuna目标函数"""
        try:
            # 从搜索空间中采样超参数
            from train import HYPEROPT_SPACE
            
            d_model = trial.suggest_categorical('d_model', HYPEROPT_SPACE['d_model'])
            d_ff_ratio = trial.suggest_categorical('d_ff_ratio', HYPEROPT_SPACE['d_ff_ratio'])
            d_ff = int(d_model * d_ff_ratio)
            n_layers = trial.suggest_categorical('n_layers', HYPEROPT_SPACE['n_layers'])
            drop_rate = trial.suggest_categorical('drop_rate', HYPEROPT_SPACE['drop_rate'])
            d_state = trial.suggest_categorical('d_state', HYPEROPT_SPACE['d_state'])
            d_conv = trial.suggest_categorical('d_conv', HYPEROPT_SPACE['d_conv'])
            expand = trial.suggest_categorical('expand', HYPEROPT_SPACE['expand'])
            patch_len = trial.suggest_categorical('patch_len', HYPEROPT_SPACE['patch_len'])
            stride_ratio = trial.suggest_categorical('stride_ratio', HYPEROPT_SPACE['stride_ratio'])
            stride = max(1, int(patch_len * stride_ratio))
            batch_size = trial.suggest_categorical('batch_size', HYPEROPT_SPACE['batch_size'])
            lr = trial.suggest_categorical('lr', HYPEROPT_SPACE['lr'])
            power_weight = trial.suggest_categorical('power_weight', HYPEROPT_SPACE['power_weight'])
            not_use_power_weight = trial.suggest_categorical('not_use_power_weight', HYPEROPT_SPACE['not_use_power_weight'])
            
            # 构建模型配置
            config = {
                'd_model': d_model,
                'd_ff': d_ff,
                'n_layers': n_layers,
                'drop_rate': drop_rate,
                'd_state': d_state,
                'd_conv': d_conv,
                'expand': expand,
                'patch_len': patch_len,
                'stride': stride,
                'batch_size': batch_size,
                'lr': lr,
                'power_weight': power_weight,
                'not_use_power_weight': not_use_power_weight,
            }
            
            # 训练模型并返回验证损失
            val_loss = self._train_with_config(config, trial)
            
            return val_loss
            
        except Exception as e:
            print(f" Trial {trial.number} 失败: {e}")
            return float('inf')
            
    def _train_with_config(self, config, trial):
        """使用给定配置训练模型"""
        from train import BiMambaPowerModel, WeightedSmoothL1, CFG
        from torch.utils.data import DataLoader, TensorDataset
        
        # 创建数据加载器 - 包含掩码数据
        X_tr, Yp_tr, Yn_tr, Mask_p_tr, Mask_n_tr = self.train_data
        X_va, Yp_va, Yn_va, Mask_p_va, Mask_n_va = self.val_data
        
        tr_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Yp_tr), torch.from_numpy(Yn_tr), 
                         torch.from_numpy(Mask_p_tr), torch.from_numpy(Mask_n_tr)),
            batch_size=config['batch_size'], shuffle=True, pin_memory=True
        )
        va_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Yp_va), torch.from_numpy(Yn_va),
                         torch.from_numpy(Mask_p_va), torch.from_numpy(Mask_n_va)),
            batch_size=config['batch_size'], pin_memory=True
        )
        
        # 创建模型
        model = BiMambaPowerModel(
            c_in=len(self.feature_cols),
            seq_len=CFG['past_steps'],
            pred_len=CFG['future_steps'],
            patch_len=config['patch_len'],
            stride=config['stride'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            dropout=config['drop_rate'],
            d_state=config['d_state'],
            d_conv=config['d_conv'],
            expand=config['expand'],
            bidirectional=CFG['bidirectional']
        ).to(self.device)
        
        # 损失函数和优化器
        wvec_simple = np.ones(CFG['future_steps']) # 或使用 train.py 中的 default_weight_vector
        criterion = WeightedSmoothL1(CFG['future_steps'], wvec_simple).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # 训练循环（缩短版本用于超参数搜索）
        max_epochs = 50  # 减少epoch数以加速搜索
        patience = 15
        best_val_loss = float('inf')
        patience_counter = 0
        epsilon = 1e-9  # 防止除以零
        
        for epoch in range(1, max_epochs + 1):
            # 训练
            model.train()
            train_loss = 0.0
            
            for xe, yp, yn, mask_p, mask_n in tr_loader:
                xe, yp, yn = xe.to(self.device), yp.to(self.device), yn.to(self.device)
                mask_p, mask_n = mask_p.to(self.device), mask_n.to(self.device)
                
                optimizer.zero_grad()
                pp, pn = model(xe)
                
                # 使用掩码计算损失
                loss_p_raw = criterion(pp, yp)
                loss_n_raw = criterion(pn, yn)
                loss_p_masked = loss_p_raw * mask_p
                loss_n_masked = loss_n_raw * mask_n
                loss_p = loss_p_masked.sum() / (mask_p.sum() + epsilon)
                loss_n = loss_n_masked.sum() / (mask_n.sum() + epsilon)
                loss = config['power_weight'] * loss_p + config['not_use_power_weight'] * loss_n
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(tr_loader)
            
            # 验证
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for xe, yp, yn, mask_p, mask_n in va_loader:
                    xe, yp, yn = xe.to(self.device), yp.to(self.device), yn.to(self.device)
                    mask_p, mask_n = mask_p.to(self.device), mask_n.to(self.device)
                    
                    pp, pn = model(xe)
                    
                    # 使用掩码计算损失
                    loss_p_raw = criterion(pp, yp)
                    loss_n_raw = criterion(pn, yn)
                    loss_p_masked = loss_p_raw * mask_p
                    loss_n_masked = loss_n_raw * mask_n
                    loss_p = loss_p_masked.sum() / (mask_p.sum() + epsilon)
                    loss_n = loss_n_masked.sum() / (mask_n.sum() + epsilon)
                    vloss = config['power_weight'] * loss_p + config['not_use_power_weight'] * loss_n
                    
                    val_loss += vloss.item()
                    
            val_loss /= len(va_loader)
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
            # Optuna剪枝
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return best_val_loss
        
    def optimize(self, n_trials: int = 100, timeout: int = 3600, enable_pruning: bool = True):
        """执行超参数优化"""
        if not OPTUNA_AVAILABLE:
            print(" Optuna未安装，无法进行超参数优化")
            return None
            
        print(f" 开始超参数搜索 (试验次数: {n_trials}, 超时: {timeout}s)")
        
        # 创建study
        pruner = MedianPruner() if enable_pruning else None
        sampler = TPESampler(seed=42)
        
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            sampler=sampler
        )
        
        # 执行优化
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # 保存结果
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # 保存优化结果
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'optimization_history': [
                {'trial': i, 'value': trial.value, 'params': trial.params}
                for i, trial in enumerate(study.trials)
                if trial.value is not None
            ]
        }
        
        with open(f"{self.out_dir}/hyperopt_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f" 超参数优化完成!")
        print(f"最佳验证损失: {self.best_score:.5f}")
        print(f"最佳参数: {self.best_params}")
        
        return self.best_params, self.best_score


def apply_incremental_training(model, train_loader, val_loader, criterion, device, out_dir, cfg):
    """应用增量微调"""
    if not cfg.get('incremental_training', False):
        return model
        
    print("\n" + "="*50)
    print(" 启动增量微调模式")
    print("="*50)
    
    trainer = IncrementalTrainer(model, device, out_dir)
    
    # 尝试加载预训练模型
    pretrained_path = f"{out_dir}/bi_mamba.pth"
    if trainer.load_pretrained_weights(pretrained_path):
        # 执行增量训练
        finetune_lr = cfg['lr'] * cfg.get('finetune_lr_ratio', 0.1)
        
        trainer.incremental_train(
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            finetune_lr=finetune_lr,
            finetune_epochs=cfg.get('finetune_epochs', 50),
            finetune_patience=cfg.get('finetune_patience', 15),
            freeze_layers=cfg.get('freeze_backbone_layers', 2)
        )
        
        # 加载微调后的最佳模型
        finetuned_path = f"{out_dir}/bi_mamba.pth"
        if os.path.exists(finetuned_path):
            model.load_state_dict(torch.load(finetuned_path, map_location=device))
            print(" 已加载微调后的最佳模型")
    else:
        print("  未找到预训练模型，跳过增量微调")
        
    return model


def apply_hyperparameter_optimization(train_data, val_data, feature_cols, scalers, device, out_dir, cfg):
    """应用超参数优化"""
    if not cfg.get('enable_hyperopt', False):
        return None
        
    print("\n" + "="*50)
    print(" 启动自动超参数调优")
    print("="*50)
    
    optimizer = HyperparameterOptimizer(
        train_data=train_data,
        val_data=val_data,
        feature_cols=feature_cols,
        scalers=scalers,
        device=device,
        out_dir=out_dir
    )
    
    best_params, best_score = optimizer.optimize(
        n_trials=cfg.get('n_trials', 10),
        timeout=cfg.get('optuna_timeout', 3600),
        enable_pruning=cfg.get('optuna_pruning', True)
    )
    
    return best_params, best_score