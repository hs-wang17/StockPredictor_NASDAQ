import pandas_datareader.data as web
import pandas as pd
import os
import time

# 确保数据目录存在
data_dir = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data'
os.makedirs(data_dir, exist_ok=True)

# 函数：获取数据，带重试机制
def get_data_with_retry(symbol, start_date, max_retries=3):
    for i in range(max_retries):
        try:
            data = web.DataReader(symbol, 'famafrench', start=start_date)[0]
            return data
        except Exception as e:
            print(f"获取 {symbol} 时出错: {e}")
            if i < max_retries - 1:
                print(f"第 {i+1} 次重试...")
                time.sleep(2)
            else:
                print(f"获取 {symbol} 失败，将跳过该因子")
                return None

# 开始日期
start_date = '2011-01-01'

# 获取FF5因子
ff5 = get_data_with_retry('F-F_Research_Data_5_Factors_2x3_daily', start_date)
if ff5 is None:
    print("无法获取FF5因子，程序退出")
    exit(1)

# 合并所有因子
daily_factors = ff5.copy()

# 获取动量因子
mom = get_data_with_retry('F-F_Momentum_Factor_daily', start_date)
if mom is not None:
    daily_factors['MOM'] = mom['Mom']

# 获取短期反转因子
st_rev = get_data_with_retry('F-F_ST_Reversal_Factor_daily', start_date)
if st_rev is not None:
    daily_factors['ST_REV'] = st_rev['ST_Rev']
    
# 获取长期反转因子
lt_rev = get_data_with_retry('F-F_LT_Reversal_Factor_daily', start_date)
if lt_rev is not None:
    daily_factors['LT_REV'] = lt_rev['LT_Rev']

# 重命名列以提高可读性
daily_factors.rename(columns={
    'Mkt-RF': 'Market',
    'SMB': 'Size',
    'HML': 'Value',
    'RMW': 'Profitability',
    'CMA': 'Investment',
    'RF': 'Risk-Free',
    'MOM': 'Momentum',
    'ST_REV': 'STReversal',
    'LT_REV': 'LTReversal',
}, inplace=True)

# 保存到CSV文件
output_path = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/daily_market_factor_data/daily_market_factor_data.csv'
daily_factors.to_csv(output_path)

print(f"市场因子数据已保存到: {output_path}")
print(f"数据形状: {daily_factors.shape}")
print(f"因子列表: {list(daily_factors.columns)}")
