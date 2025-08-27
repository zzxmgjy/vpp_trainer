import pickle
import pandas as pd
import lightgbm as lgb
import os
from dateutil import parser
from datetime import timedelta

def train_load(train_data):
    train_data,eval_data = train_data_transfer(train_data)
    feature_cols = ['quarter', 'holiday', 'is_peak', 'code', 'temperature']
    categorical_cols = ['quarter', 'holiday', 'is_peak', 'code']
    target_col = ['load']
    x = train_data[feature_cols]
    y = train_data[target_col]
    eval_x = eval_data[feature_cols]
    eval_y = eval_data[target_col]
    for col in categorical_cols:
        if col in x.columns:
            x[col] = x[col].astype('category')
    for col in categorical_cols:
        if col in eval_x.columns:
            eval_x[col] = eval_x[col].astype('category')
    return train(x,y,eval_x,eval_y,categorical_cols)

def train_meter(train_data):
    train_data,eval_data = train_data_transfer(train_data)
    feature_cols = ['quarter', 'holiday', 'is_peak', 'code', 'temperature']
    categorical_cols = ['quarter', 'holiday', 'is_peak', 'code']
    target_col = ['meter']
    x = train_data[feature_cols]
    y = train_data[target_col]
    eval_x = eval_data[feature_cols]
    eval_y = eval_data[target_col]
    for col in categorical_cols:
        if col in x.columns:
            x[col] = x[col].astype('category')
    for col in categorical_cols:
        if col in eval_x.columns:
            eval_x[col] = eval_x[col].astype('category')
    return train(x,y,eval_x,eval_y,categorical_cols)

def train_data_transfer(train_data):
    train_data['date'] = pd.to_datetime(train_data['time'])
    max_date = train_data['date'].max().strftime('%Y-%m-%d')
    last_date_time = parser.parse(max_date)
    train_data_last30 = train_data[train_data['date'] > last_date_time - timedelta(days=29)]
    train_data_last14 = train_data[train_data['date'] > last_date_time - timedelta(days=13)]
    train_data_last7 = train_data[train_data['date'] > last_date_time - timedelta(days=6)]
    train_data_last3 = train_data[train_data['date'] > last_date_time - timedelta(days=2)]
    for i in range(1, 5):
        train_data = pd.concat([
            train_data,
            train_data_last30,
            train_data_last14
        ])
    for i in range(1, 10):
        train_data = pd.concat([
            train_data,
            train_data_last7
        ])
    for i in range(1, 20):
        train_data = pd.concat([
            train_data,
            train_data_last3
        ])
    return train_data,train_data_last14

def train(train_x, train_y, eval_x, eval_y, categorical_cols):
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
    model = lgb.LGBMRegressor(**params, n_estimators=500, n_jobs=-1)
    model.fit(train_x, train_y,
              categorical_feature=categorical_cols,
              eval_set=[(eval_x, eval_y)],
              callbacks=[lgb.early_stopping(stopping_rounds=10)])
    return model

def save_model(model_path, model):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model, open(model_path, 'wb'))

def predict_load(model_path, data_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    feature_cols = ['quarter', 'holiday', 'is_peak', 'code', 'temperature']
    categorical_cols = ['quarter', 'holiday', 'is_peak', 'code']
    data = pd.read_csv(data_path)
    numerical_cols = data.columns
    feature_cols = [col for col in numerical_cols if col in feature_cols]
    x_data = data[feature_cols]
    for col in categorical_cols:
        if col in x_data.columns:
            x_data[col] = x_data[col].astype('category')
    predictions = model.predict(x_data)
    return data['time'],predictions

def predict_meter(model_path, data_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    feature_cols = ['quarter', 'holiday', 'is_peak', 'code', 'temperature']
    categorical_cols = ['quarter', 'holiday', 'is_peak', 'code']
    data = pd.read_csv(data_path)
    numerical_cols = data.columns
    feature_cols = [col for col in numerical_cols if col in feature_cols]
    x_data = data[feature_cols]
    for col in categorical_cols:
        if col in x_data.columns:
            x_data[col] = x_data[col].astype('category')
    predictions = model.predict(x_data)
    return data['time'],predictions