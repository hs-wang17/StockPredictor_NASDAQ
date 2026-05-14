import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import concurrent.futures
import time
import gc
import logging

# 配置路径
TRADE_DATE_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/trade_date_data'
FACTOR_DATA_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/daily_factor_data'
MODEL_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/model/mlp_L1loss'
PREDICTION_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/outputs/prediction_mlp_L1loss'

# 定义所有标签列（与训练脚本保持一致）
label_cols = ['return_5d', 'return_10d', 'return_20d', 'return_60d', 'return_120d', 'return_252d',
             'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d', 'volatility_120d', 'volatility_252d',
             'max_drawdown_5d', 'max_drawdown_10d', 'max_drawdown_20d', 'max_drawdown_60d', 'max_drawdown_120d', 'max_drawdown_252d']

# 训练参数（与训练脚本保持一致）
N_FOLDS = 4


class MLP(nn.Module):
    """MLP模型类（与训练脚本完全一致）"""
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_period_files():
    """获取所有周期文件"""
    period_files = []
    for filename in os.listdir(TRADE_DATE_DIR):
        if filename.startswith('period_') and filename.endswith('.csv') and 'summary' not in filename:
            period_files.append(filename)
    # 按周期编号排序
    period_files.sort()
    return period_files


def get_test_dates(period_file):
    """从周期文件中获取测试日期"""
    file_path = os.path.join(TRADE_DATE_DIR, period_file)
    try:
        df = pd.read_csv(file_path)
        test_dates = df[df['type'] == 'test']['date'].tolist()
        return test_dates
    except Exception as e:
        logger.error(f"读取周期文件 {period_file} 失败: {e}")
        return []


def load_factor_data(date):
    """加载指定日期的因子数据（与训练脚本保持一致的数据处理）"""
    factor_path = os.path.join(FACTOR_DATA_DIR, f'{date}.csv')
    
    if not os.path.exists(factor_path):
        logger.warning(f"因子数据文件不存在: {factor_path}")
        return None
    
    try:
        df_f = pd.read_csv(factor_path)
        
        # 处理列名重复和数据类型（与训练脚本保持一致）
        if 'stock.1' in df_f.columns:
            df_f.drop(columns=['stock.1'], inplace=True)
        df_f['stock'] = df_f['stock'].astype(str)
        
        # 类型转换
        fcols = df_f.select_dtypes('float64').columns
        df_f[fcols] = df_f[fcols].astype('float32')
        
        return df_f
    except Exception as e:
        logger.error(f"加载日期 {date} 的因子数据时出错: {e}")
        return None


def fast_load_test_data(dates):
    """多线程并行加载测试数据，返回日期到数据的映射字典"""
    data_dict = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_date = {executor.submit(load_factor_data, date): date for date in dates}
        
        for future in concurrent.futures.as_completed(future_to_date):
            date = future_to_date[future]
            try:
                data = future.result()
                if data is not None:
                    data_dict[date] = data
            except Exception as e:
                logger.error(f"处理日期 {date} 的数据时出错: {e}")
    
    return data_dict


def load_models(period_name):
    """加载指定周期的所有MLP模型"""
    models = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold in range(1, N_FOLDS + 1):
        model_filename = f"{period_name}_mlp_fold{fold}.pt"
        model_path = os.path.join(MODEL_DIR, model_filename)
        
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            continue
        
        try:
            # 先加载一个模型获取输入输出维度
            if len(models) == 0:
                # 需要先获取输入维度，这里通过dummy方式处理
                # 实际上在预测时会根据数据动态设置
                pass
            
            model = None
            models.append((model, model_path))
        except Exception as e:
            logger.error(f"加载模型 {model_path} 失败: {e}")
    
    return models


def predict_single_date(date, factor_data, period_name, device):
    """对单个日期进行预测"""
    if factor_data is None:
        logger.warning(f"日期 {date} 的因子数据为空，跳过")
        return None
    
    try:
        # 选择特征列（排除非特征列，与训练脚本保持一致）
        exclude_cols = ['date', 'stock'] + label_cols
        feature_cols = [col for col in factor_data.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            logger.error(f"日期 {date} 没有可用的特征列")
            return None
        
        X = factor_data[feature_cols]
        
        # 填充缺失值为0（与训练脚本保持一致）
        X = X.replace([np.inf, -np.inf], 0.0)
        X = X.fillna(0.0)
        
        # 加载模型和scaler
        scaler_path = os.path.join(MODEL_DIR, f"{period_name}_mlp_scaler.joblib")
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler文件不存在: {scaler_path}")
            return None
        
        scaler_y = joblib.load(scaler_path)
        
        # 加载所有模型
        predictions = []
        for fold in range(1, N_FOLDS + 1):
            model_filename = f"{period_name}_mlp_fold{fold}.pt"
            model_path = os.path.join(MODEL_DIR, model_filename)
            
            if not os.path.exists(model_path):
                logger.warning(f"跳过缺失的模型: {model_path}")
                continue
            
            # 创建模型实例
            model = MLP(input_dim=X.shape[1], output_dim=len(label_cols))
            
            # 加载模型权重
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            
            # 转换为PyTorch张量
            X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
            
            # 预测
            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().numpy()
            
            predictions.append(pred_scaled)
            
            # 清理模型
            del model
            gc.collect()
        
        if len(predictions) == 0:
            logger.error(f"没有可用的模型进行预测")
            return None
        
        # 计算平均预测值（在标准化空间）
        avg_pred_scaled = np.mean(predictions, axis=0)
        
        # 反标准化到原始空间（使用训练时的y_scaler）
        avg_pred = scaler_y.inverse_transform(avg_pred_scaled)
        
        # 构建结果DataFrame
        result = pd.DataFrame(avg_pred, columns=label_cols)
        result['stock'] = factor_data['stock'].values
        
        # 调整列顺序，将stock列放在第一位
        cols = ['stock'] + label_cols
        result = result[cols]
        
        return result
    
    except Exception as e:
        logger.error(f"对日期 {date} 进行预测时出错: {e}")
        return None


def process_period(period_file):
    """处理单个周期"""
    logger.info(f"\n处理周期文件: {period_file}")
    
    # 提取周期名称（不含.csv后缀）
    period_name = period_file.replace('.csv', '')
    
    # 获取测试日期
    test_dates = get_test_dates(period_file)
    logger.info(f"  测试日期数量: {len(test_dates)}")
    if test_dates:
        logger.info(f"  日期范围: {test_dates[0]} ~ {test_dates[-1]}")
    
    # 加载因子数据
    logger.info("  并行加载因子数据...")
    factor_data_dict = fast_load_test_data(test_dates)
    logger.info(f"  成功加载 {len(factor_data_dict)} 天的因子数据")
    
    # 检查是否有可用模型
    model_exists = False
    for fold in range(1, N_FOLDS + 1):
        model_path = os.path.join(MODEL_DIR, f"{period_name}_mlp_fold{fold}.pt")
        if os.path.exists(model_path):
            model_exists = True
            break
    
    if not model_exists:
        logger.warning(f"  未找到周期 {period_name} 的任何模型，跳过此周期")
        return
    
    # 检查scaler是否存在
    scaler_path = os.path.join(MODEL_DIR, f"{period_name}_mlp_scaler.joblib")
    if not os.path.exists(scaler_path):
        logger.warning(f"  未找到周期 {period_name} 的scaler文件，跳过此周期")
        return
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  使用设备: {device}")
    
    # 对每个测试日期进行预测
    logger.info("  开始预测...")
    start_time = time.time()
    
    for date in test_dates:
        # 从字典中获取对应日期的因子数据
        if date not in factor_data_dict:
            logger.warning(f"  未找到 {date} 的因子数据，跳过")
            continue
        
        factor_data = factor_data_dict[date]
        
        # 预测
        prediction = predict_single_date(date, factor_data, period_name, device)
        
        if prediction is not None:
            # 保存预测结果
            output_file = os.path.join(PREDICTION_DIR, f"{date}.csv")
            prediction.to_csv(output_file, index=False)
            logger.info(f"  保存预测结果到: {output_file}")
    
    total_time = time.time() - start_time
    logger.info(f"  周期 {period_name} 预测完成，耗时: {total_time:.2f}秒")
    
    # 清理内存
    del factor_data_dict
    gc.collect()


def main():
    global logger
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("开始执行MLP-L1loss模型预测任务")
    logger.info("=" * 80)
    
    # 确保预测目录存在
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    # 获取所有周期文件
    period_files = get_period_files()
    logger.info(f"找到 {len(period_files)} 个周期文件")
    
    if len(period_files) == 0:
        logger.warning("未找到任何周期文件，任务结束")
        return
    
    # 处理每个周期
    for period_file in period_files:
        process_period(period_file)
    
    logger.info("\n" + "=" * 80)
    logger.info("MLP-L1loss模型预测任务完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
