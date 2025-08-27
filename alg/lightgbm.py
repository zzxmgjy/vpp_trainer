import pickle
import pandas as pd
import lightgbm as lgb
import os

def train_load(train_data):
    feature_cols = ['quarter', 'holiday', 'is_peak', 'code', 'temperature']
    target_col = ['load']
    x = train_data[feature_cols]
    y = train_data[target_col]
    train_x, train_y = x.values, y.values
    return train(train_x,train_y)

def train_meter(train_data):
    feature_cols = ['quarter', 'holiday', 'is_peak', 'code', 'temperature']
    target_col = ['meter']
    x = train_data[feature_cols]
    y = train_data[target_col]
    train_x, train_y = x.values, y.values
    return train(train_x,train_y)

def train(train_x, train_y):
    # 参数
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': 'rmse',  # 评估函数
        'num_leaves': 31,  # 叶子节点数
        'max_depth': 6,
        'learning_rate': 0.07,  # 学习速率
        'feature_fraction': 0.8,  # 建树的特征选择比例
        'bagging_fraction': 0.9,  # 建树的样本采样比例
        'bagging_freq': 10,  # k 意味着每 k 次迭代执行bagging
        'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(train_x, train_y)
    return model

def save_model(model_path, model):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model, open(model_path, 'wb'))

def predict_load(model_path, data_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    feature_cols = ['quarter', 'holiday', 'is_peak', 'code', 'temperature']
    data = pd.read_csv(data_path)
    numerical_cols = data.columns
    feature_cols = [col for col in numerical_cols if col in feature_cols]
    x_data = data[feature_cols]
    predictions = model.predict(x_data)
    return data['time'],predictions

def predict_meter(model_path, data_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    feature_cols = ['quarter', 'holiday', 'is_peak', 'code', 'temperature']
    data = pd.read_csv(data_path)
    numerical_cols = data.columns
    feature_cols = [col for col in numerical_cols if col in feature_cols]
    x_data = data[feature_cols]
    predictions = model.predict(x_data)
    return data['time'],predictions