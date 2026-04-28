import os
import pandas as pd
import numpy as np
import joblib
import concurrent.futures
import time
import gc
import logging
from typing import Dict, List, Optional, Any

# 配置路径
TRADE_DATE_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/trade_date_data'
FACTOR_DATA_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/daily_factor_data'
MODEL_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/model/lgbm'
PREDICTION_DIR = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/outputs/prediction_lgbm'

# 定义所有标签列（与训练脚本保持完全一致）
label_cols = [
    'return_5d', 'return_10d', 'return_20d', 'return_60d', 'return_120d', 'return_252d',
    'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d', 'volatility_120d', 'volatility_252d',
    'max_drawdown_5d', 'max_drawdown_10d', 'max_drawdown_20d', 'max_drawdown_60d', 'max_drawdown_120d', 'max_drawdown_252d'
]

# 训练参数（与训练脚本保持一致）
N_FOLDS = 4


def setup_logging() -> logging.Logger:
    """配置日志记录系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_period_files() -> List[str]:
    """获取所有周期文件，按周期编号排序"""
    period_files = []
    try:
        for filename in os.listdir(TRADE_DATE_DIR):
            if filename.startswith('period_') and filename.endswith('.csv') and 'summary' not in filename:
                period_files.append(filename)
        # 按周期编号排序
        period_files.sort()
        logger.info(f"发现 {len(period_files)} 个周期文件")
    except Exception as e:
        logger.error(f"获取周期文件列表失败: {e}")
    
    return period_files


def get_test_dates(period_file: str) -> List[str]:
    """从周期文件中获取测试日期"""
    file_path = os.path.join(TRADE_DATE_DIR, period_file)
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"周期文件不存在: {file_path}")
            return []
        
        df = pd.read_csv(file_path)
        
        if 'type' not in df.columns or 'date' not in df.columns:
            logger.error(f"周期文件格式不正确，缺少必要列: {file_path}")
            return []
        
        test_dates = df[df['type'] == 'test']['date'].tolist()
        logger.debug(f"从周期文件 {period_file} 中提取 {len(test_dates)} 个测试日期")
        
        return test_dates
    
    except Exception as e:
        logger.error(f"读取周期文件 {period_file} 失败: {e}")
        return []


def load_factor_data(date: str) -> Optional[pd.DataFrame]:
    """
    加载指定日期的因子数据
    
    Args:
        date: 日期字符串
        
    Returns:
        因子数据DataFrame，如果加载失败返回None
    """
    factor_path = os.path.join(FACTOR_DATA_DIR, f'{date}.csv')
    
    if not os.path.exists(factor_path):
        logger.warning(f"因子数据文件不存在: {factor_path}")
        return None
    
    try:
        df_f = pd.read_csv(factor_path)
        
        # 处理列名重复（与训练脚本保持一致）
        if 'stock.1' in df_f.columns:
            df_f.drop(columns=['stock.1'], inplace=True)
        
        # 确保stock列为字符串类型
        df_f['stock'] = df_f['stock'].astype(str)
        
        # 类型转换（float64 -> float32，节省内存）
        fcols = df_f.select_dtypes('float64').columns
        df_f[fcols] = df_f[fcols].astype('float32')
        
        logger.debug(f"成功加载日期 {date} 的因子数据，形状: {df_f.shape}")
        return df_f
    
    except Exception as e:
        logger.error(f"加载日期 {date} 的因子数据时出错: {e}")
        return None


def fast_load_test_data(dates: List[str]) -> Dict[str, pd.DataFrame]:
    """
    多线程并行加载测试数据
    
    Args:
        dates: 日期列表
        
    Returns:
        日期到数据的映射字典
    """
    data_dict: Dict[str, pd.DataFrame] = {}
    
    if not dates:
        logger.warning("没有需要加载的日期")
        return data_dict
    
    logger.info(f"开始并行加载 {len(dates)} 天的因子数据...")
    
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
    
    logger.info(f"成功加载 {len(data_dict)} / {len(dates)} 天的因子数据")
    return data_dict


def load_models(period_name: str) -> List[Any]:
    """
    加载指定周期的所有LGBM模型
    
    Args:
        period_name: 周期名称
        
    Returns:
        模型列表
    """
    models = []
    
    for fold in range(1, N_FOLDS + 1):
        model_filename = f"{period_name}_fold{fold}.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)
        
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            continue
        
        try:
            model = joblib.load(model_path)
            models.append(model)
            logger.debug(f"成功加载模型: {model_filename}")
        except Exception as e:
            logger.error(f"加载模型 {model_path} 失败: {e}")
    
    return models


def predict_single_date(
    date: str,
    factor_data: pd.DataFrame,
    models: List[Any]
) -> Optional[pd.DataFrame]:
    """
    对单个日期进行预测
    
    Args:
        date: 日期字符串
        factor_data: 因子数据DataFrame
        models: 模型列表
        
    Returns:
        预测结果DataFrame，如果预测失败返回None
    """
    if factor_data is None or len(models) == 0:
        if factor_data is None:
            logger.warning(f"日期 {date} 的因子数据为空")
        if len(models) == 0:
            logger.warning(f"日期 {date} 没有可用模型")
        return None
    
    try:
        # 选择特征列（与训练脚本保持完全一致）
        exclude_cols = ['date', 'stock'] + label_cols
        feature_cols = [col for col in factor_data.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            logger.error(f"日期 {date} 没有可用的特征列")
            return None
        
        X = factor_data[feature_cols]
        
        # 数据预处理（与训练脚本保持完全一致）
        X = X.replace([np.inf, -np.inf], 0.0)
        X.fillna(0.0, inplace=True)
        
        # 多模型预测并平均
        predictions = []
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # 计算平均预测值
        avg_pred = np.mean(predictions, axis=0)
        
        # 构建结果DataFrame
        result = pd.DataFrame(avg_pred, columns=label_cols)
        result['stock'] = factor_data['stock'].values
        
        # 调整列顺序，将stock列放在第一位
        cols = ['stock'] + label_cols
        result = result[cols]
        
        logger.debug(f"日期 {date} 预测完成，样本数: {len(result)}")
        return result
    
    except Exception as e:
        logger.error(f"对日期 {date} 进行预测时出错: {e}")
        return None


def process_period(period_file: str) -> None:
    """
    处理单个周期的预测
    
    Args:
        period_file: 周期文件名
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"处理周期文件: {period_file}")
    
    # 提取周期名称（不含.csv后缀）
    period_name = period_file.replace('.csv', '')
    
    # 获取测试日期
    test_dates = get_test_dates(period_file)
    logger.info(f"测试日期数量: {len(test_dates)}")
    if test_dates:
        logger.info(f"日期范围: {test_dates[0]} ~ {test_dates[-1]}")
    
    if not test_dates:
        logger.warning("没有测试日期，跳过此周期")
        return
    
    # 加载因子数据
    factor_data_dict = fast_load_test_data(test_dates)
    
    if not factor_data_dict:
        logger.warning("没有成功加载任何因子数据，跳过此周期")
        return
    
    # 加载模型
    logger.info("加载模型...")
    models = load_models(period_name)
    logger.info(f"成功加载 {len(models)} 个模型")
    
    if len(models) == 0:
        logger.warning("未找到任何模型，跳过此周期")
        return
    
    # 对每个测试日期进行预测
    logger.info("开始预测...")
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for date in test_dates:
        # 从字典中获取对应日期的因子数据
        if date not in factor_data_dict:
            logger.warning(f"未找到 {date} 的因子数据，跳过")
            fail_count += 1
            continue
        
        factor_data = factor_data_dict[date]
        
        # 预测
        prediction = predict_single_date(date, factor_data, models)
        
        if prediction is not None:
            # 确保输出目录存在
            os.makedirs(PREDICTION_DIR, exist_ok=True)
            
            # 保存预测结果
            output_file = os.path.join(PREDICTION_DIR, f"{date}.csv")
            prediction.to_csv(output_file, index=False)
            success_count += 1
            logger.debug(f"保存预测结果到: {output_file}")
        else:
            fail_count += 1
    
    total_time = time.time() - start_time
    logger.info(f"周期 {period_name} 预测完成")
    logger.info(f"成功: {success_count}, 失败: {fail_count}")
    logger.info(f"耗时: {total_time:.2f}秒")
    
    # 清理内存
    del models, factor_data_dict
    gc.collect()


def main() -> None:
    """主函数"""
    global logger
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("开始执行LGBM模型预测任务")
    logger.info("=" * 80)
    
    # 确保预测目录存在
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    # 获取所有周期文件
    period_files = get_period_files()
    
    if not period_files:
        logger.warning("未找到任何周期文件，任务结束")
        return
    
    # 处理每个周期
    start_total_time = time.time()
    for period_file in period_files:
        process_period(period_file)
    
    total_time = time.time() - start_total_time
    logger.info("\n" + "=" * 80)
    logger.info("LGBM模型预测任务完成!")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
