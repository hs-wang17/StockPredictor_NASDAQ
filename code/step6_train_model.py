import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import gc
import joblib
import concurrent.futures

# 配置路径
TRADE_DATE_DIR = '/root/autodl-tmp/.autodl/project/data/trade_date_data'
FACTOR_DATA_DIR = '/root/autodl-tmp/.autodl/project/data/daily_factor_data'
LABEL_DATA_DIR = '/root/autodl-tmp/.autodl/project/data/daily_label_data'
MODEL_DIR = '/root/autodl-tmp/.autodl/project/model'

# 训练参数
N_FOLDS = 4  # 4折交叉验证
LGB_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,  # 稍微降低，提高稳定性
    'max_depth': 5,  # 降低深度，防止过拟合
    'num_leaves': 31,  # 2^5-1
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,  # 提高，防止过拟合噪声
    'min_split_gain': 0.01,  # 新增，最小分裂增益
    'reg_alpha': 0.1,  # L1正则化
    'reg_lambda': 0.1,  # L2正则化
    'n_estimators': 100,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': 100  # 每100轮输出一次
}


def get_period_files():
    """获取所有周期文件"""
    period_files = []
    for filename in os.listdir(TRADE_DATE_DIR):
        if filename.startswith('period_') and filename.endswith('.csv'):
            period_files.append(filename)
    # 按周期编号排序
    period_files.sort()
    return period_files


def get_train_dates(period_file):
    """从周期文件中获取训练日期"""
    file_path = os.path.join(TRADE_DATE_DIR, period_file)
    df = pd.read_csv(file_path)
    train_dates = df[df['type'] == 'train']['date'].tolist()
    return train_dates


def load_and_merge_single_day(date):
    """加载指定日期的数据"""
    factor_path = os.path.join(FACTOR_DATA_DIR, f'{date}.csv')
    label_path = os.path.join(LABEL_DATA_DIR, f'{date}.csv')
    
    df_f = pd.read_csv(factor_path)
    df_l = pd.read_csv(label_path)
    
    # 处理列名重复和数据类型
    df_f.drop(columns=['stock.1'], inplace=True)
    df_f['stock'] = df_f['stock'].astype(str)
    df_l['stock'] = df_l['stock'].astype(str)
    
    # 合并因子和标签
    merged = pd.merge(df_f, df_l, on=['stock'], how='inner')
    merged['date'] = date
    
    # 类型转换
    fcols = merged.select_dtypes('float64').columns
    merged[fcols] = merged[fcols].astype('float32')
    
    return merged


def fast_load_period_data(dates):
    """多线程并行加载"""
    all_data = []
    # 使用 ThreadPoolExecutor 并行读取
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_and_merge_single_day, dates))
    
    # 过滤掉 None 并合并
    all_data = [r for r in results if r is not None]
    
    return pd.concat(all_data, ignore_index=True)


def train_lgbm_with_kfold(X, y, period_name, n_folds=4):
    """使用K折交叉验证训练LGBM模型（多输出）"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    models = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  训练第 {fold_idx + 1} 折...")
        
        # 划分训练和验证数据
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        
        # 训练模型
        start_time = time.time()
        
        # 创建LGBM回归器
        lgb_regressor = lgb.LGBMRegressor(**LGB_PARAMS)
        
        # 使用MultiOutputRegressor包装
        model = MultiOutputRegressor(lgb_regressor, n_jobs=-1)
        model.fit(X_train, y_train_scaled)
        
        train_time = time.time() - start_time
        
        # 验证模型
        y_pred = model.predict(X_val)
        
        # 计算每个标签的指标
        rmse_per_label = []
        r2_per_label = []
        for i, label in enumerate(y.columns):
            rmse = np.sqrt(mean_squared_error(y_val_scaled[:, i], y_pred[:, i]))
            r2 = r2_score(y_val_scaled[:, i], y_pred[:, i])
            rmse_per_label.append(rmse)
            r2_per_label.append(r2)
        
        # 计算平均指标
        avg_rmse = np.mean(rmse_per_label)
        avg_r2 = np.mean(r2_per_label)
        
        fold_metrics.append({
            'fold': fold_idx + 1,
            'avg_rmse': avg_rmse,
            'avg_r2': avg_r2,
            'rmse_per_label': rmse_per_label,
            'r2_per_label': r2_per_label,
            'train_time': train_time
        })
        print(f"  第 {fold_idx + 1} 折完成: 平均RMSE = {avg_rmse:.4f}, 平均R2 = {avg_r2:.4f}, 训练时间 = {train_time:.2f}秒")
        
        # 保存模型
        model_filename = f"{period_name}_fold{fold_idx + 1}.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"  第 {fold_idx + 1} 折模型保存到: {model_path}")
        
        models.append(model)
        
        # 清理内存
        del X_train, X_val, y_train_scaled, y_val_scaled, model, y_pred
        gc.collect()
    
    return models, fold_metrics


def process_period(period_file):
    """处理单个周期"""
    print(f"\n处理周期文件: {period_file}")
    
    # 提取周期名称（不含.csv后缀）
    period_name = period_file.replace('.csv', '')
    
    # 获取训练日期
    train_dates = get_train_dates(period_file)
    print(f"  训练日期数量: {len(train_dates)}")
    print(f"  日期范围: {train_dates[0]} ~ {train_dates[-1]}")
    
    # 并行加载因子和标签数据
    print("  并行加载因子和标签数据...")
    merged_data = fast_load_period_data(train_dates)
    # 填充缺失值为0
    merged_data.fillna(0.0, inplace=True)
    print(f"  合并后数据形状: {merged_data.shape}")
    
    # 选择特征列（排除非特征列）
    # 定义所有标签列
    label_cols = ['return_5d', 'return_10d', 'return_20d', 'return_60d', 'return_120d', 'return_252d',
                 'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d', 'volatility_120d', 'volatility_252d',
                 'max_drawdown_5d', 'max_drawdown_10d', 'max_drawdown_20d', 'max_drawdown_60d', 'max_drawdown_120d', 'max_drawdown_252d']
    
    # 排除所有与stock相关的列和日期列
    exclude_cols = ['date', 'stock'] + label_cols
    feature_cols = [col for col in merged_data.columns if col not in exclude_cols]
    X = merged_data[feature_cols]
    y = merged_data[label_cols]
    
    print(f"  特征数量: {X.shape[1]}")
    print(f"  样本数量: {X.shape[0]}")
    
    # 训练模型
    print("  开始训练模型...")
    models, fold_metrics = train_lgbm_with_kfold(X, y, period_name, N_FOLDS)
    
    # 计算平均指标
    avg_rmse = np.mean([m['avg_rmse'] for m in fold_metrics])
    avg_r2 = np.mean([m['avg_r2'] for m in fold_metrics])
    total_time = np.sum([m['train_time'] for m in fold_metrics])
    
    print(f"\n  周期 {period_name} 训练完成:")
    print(f"  平均RMSE: {avg_rmse:.4f}")
    print(f"  平均R2: {avg_r2:.4f}")
    print(f"  总训练时间: {total_time:.2f}秒")
    
    # 清理内存
    del merged_data, X, y
    gc.collect()
    
    return {
        'period': period_name,
        'metrics': fold_metrics,
        'avg_rmse': avg_rmse,
        'avg_r2': avg_r2,
        'total_time': total_time
    }


def main():
    print("=" * 80)
    print("开始执行模型训练任务")
    print("=" * 80)
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 获取所有周期文件
    period_files = get_period_files()
    print(f"找到 {len(period_files)} 个周期文件")
    
    # 处理每个周期
    results = []
    for period_file in period_files:
        result = process_period(period_file)
        results.append(result)
    
    # 生成训练报告
    if results:
        print("\n" + "=" * 80)
        print("训练完成报告")
        print("=" * 80)
        
        for result in results:
            print(f"\n周期: {result['period']}")
            print(f"平均RMSE: {result['avg_rmse']:.4f}")
            print(f"平均R2: {result['avg_r2']:.4f}")
            print(f"总训练时间: {result['total_time']:.2f}秒")
        
        # 保存训练结果
        results_df = pd.DataFrame(results)
        report_path = os.path.join(MODEL_DIR, 'training_report.csv')
        results_df.to_csv(report_path, index=False)
        print(f"\n训练报告已保存到: {report_path}")
    
    print("\n" + "=" * 80)
    print("模型训练任务完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
