import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import time
import gc
import joblib
import concurrent.futures

# 配置路径
TRADE_DATE_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/trade_date_data'
FACTOR_DATA_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/daily_factor_data'
LABEL_DATA_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/daily_label_data'
MODEL_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/model/mlp_L1loss'

# 训练参数
N_FOLDS = 4  # 4折交叉验证
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.00001  # 降低学习率以避免梯度爆炸
GRADIENT_CLIP = 1.0  # 梯度裁剪

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        # 激活函数
        self.relu = nn.ReLU()
        # 批归一化
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        # dropout
        self.dropout = nn.Dropout(0.2)
        
        # 初始化参数
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming Normal初始化权重
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # 偏置初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


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


def train_mlp_with_kfold(X, y, period_name, n_folds=4):
    """使用K折交叉验证训练MLP模型（多输出）"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    models = []
    fold_metrics = []
    
    # 定义所有标签列
    label_cols = ['return_5d', 'return_10d', 'return_20d', 'return_60d', 'return_120d', 'return_252d',
                 'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d', 'volatility_120d', 'volatility_252d',
                 'max_drawdown_5d', 'max_drawdown_10d', 'max_drawdown_20d', 'max_drawdown_60d', 'max_drawdown_120d', 'max_drawdown_252d']
    
    # 数据标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_np = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    y_np = np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    input_dim = X_np.shape[1]
    output_dim = y_np.shape[1]

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        print(f"  训练第 {fold_idx + 1} 折...")
        
        # 划分训练和验证数据
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]
        
        # 转换为PyTorch张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 初始化模型
        model = MLP(input_dim, output_dim)
        
        # 移动到GPU（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 定义损失函数和优化器
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 训练模型
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # 前向传播
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_X.size(0)
            
            # 计算平均损失
            epoch_loss = running_loss / len(train_loader.dataset)
            
            if (epoch + 1) % 10 == 0:
                print(f"    第 {epoch + 1} 轮, 损失: {epoch_loss:.4f}")
        
        train_time = time.time() - start_time
        
        # 验证模型
        model.eval()
        with torch.no_grad():
            X_val_tensor = X_val_tensor.to(device)
            y_val_tensor = y_val_tensor.to(device)
            y_pred = model(X_val_tensor).cpu().numpy()
        
        # 计算每个标签的指标
        rmse_per_label = []
        r2_per_label = []
        for i, label in enumerate(label_cols):
            rmse = np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i]))
            r2 = r2_score(y_val[:, i], y_pred[:, i])
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
        model_filename = f"{period_name}_mlp_fold{fold_idx + 1}.pt"
        model_path = os.path.join(MODEL_DIR, model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"  第 {fold_idx + 1} 折模型保存到: {model_path}")
        
        models.append(model)
        
        # 清理内存
        del X_train, X_val, y_train, y_val, model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
        gc.collect()
    
    # 保存标准化器
    scaler_filename = f"{period_name}_mlp_scaler.joblib"
    scaler_path = os.path.join(MODEL_DIR, scaler_filename)
    joblib.dump(scaler_y, scaler_path)
    print(f"  标签标准化器保存到: {scaler_path}")
    
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
    merged_data = merged_data.replace([np.inf, -np.inf], 0.0)
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
    print("  开始训练MLP模型...")
    models, fold_metrics = train_mlp_with_kfold(X, y, period_name, N_FOLDS)
    
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
    print("开始执行MLP模型训练任务")
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
    print("MLP模型训练任务完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
