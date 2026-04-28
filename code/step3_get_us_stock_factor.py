import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing

data_dir = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/daily_stock_data'
output_dir = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/daily_factor_data'
os.makedirs(output_dir, exist_ok=True)

csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
print(f"Found {len(csv_files)} daily stock files")
print(f"Date range: {csv_files[0]} to {csv_files[-1]}")


def calculate_alpha360_factors(close, open_p, high, low, volume, adj_close):
    """
    QLib Alpha360 系列因子 - 360个因子的精简版本
    来源: QLib Alpha360 | 主要包含量价关系的各种时间窗口组合
    适用场景: 日频交易，捕捉不同时间尺度的量价关系
    """
    factors = {}
    
    close_aligned = adj_close.reindex(close.index).fillna(adj_close)
    volume_aligned = volume.reindex(close.index).fillna(0)
    windows = [5, 10, 20, 30, 60]
    
    for w in windows:
        factors[f'rank_return_{w}d'] = close_aligned.pct_change(w)
        factors[f'rank_volatility_{w}d'] = close_aligned.pct_change().rolling(w).std()
        factors[f'rank_volume_ratio_{w}d'] = volume_aligned / volume_aligned.rolling(w).mean() - 1
        factors[f'rank_price_volume_corr_{w}d'] = close_aligned.rolling(w).corr(volume_aligned)
        factors[f'rank_high_low_range_{w}d'] = (high - low).rolling(w).mean() / close_aligned.replace(0, np.nan)
    
    return pd.DataFrame(factors)


def calculate_biquant_factors(close, open_p, high, low, volume, adj_close):
    """
    BigQuant 因子 - 常见技术分析因子
    来源: BigQuant 平台
    适用场景: 趋势跟踪、均值回归
    """
    factors = {}
    
    close_aligned = adj_close.reindex(close.index).fillna(adj_close)
    volume_aligned = volume.reindex(close.index).fillna(0)
    
    # Moving averages
    sma5 = close_aligned.rolling(5).mean()
    sma10 = close_aligned.rolling(10).mean()
    sma20 = close_aligned.rolling(20).mean()
    sma60 = close_aligned.rolling(60).mean()
    
    factors['bi_quant_ma_golden_5_20'] = (sma5 - sma20) / sma20.replace(0, np.nan)
    factors['bi_quant_ma_golden_10_60'] = (sma10 - sma60) / sma60.replace(0, np.nan)
    
    # Momentum
    factors['bi_quant_momentum_5'] = close_aligned / close_aligned.shift(5) - 1
    factors['bi_quant_momentum_10'] = close_aligned / close_aligned.shift(10) - 1
    factors['bi_quant_momentum_20'] = close_aligned / close_aligned.shift(20) - 1
    
    # VWAP
    factors['bi_quant_vwap_5'] = (close_aligned * volume_aligned).rolling(5).sum() / volume_aligned.rolling(5).sum() / close_aligned.replace(0, np.nan) - 1
    factors['bi_quant_vwap_20'] = (close_aligned * volume_aligned).rolling(20).sum() / volume_aligned.rolling(20).sum() / close_aligned.replace(0, np.nan) - 1
    
    # Price oscillator
    factors['bi_quant_price_oscillator_5_20'] = (sma5 - sma20) / sma20.replace(0, np.nan)
    factors['bi_quant_price_oscillator_10_60'] = (sma10 - sma60) / sma60.replace(0, np.nan)
    
    return pd.DataFrame(factors)


def calculate_technical_indicators(close, open_p, high, low, volume, adj_close):
    """
    传统技术指标因子 - 来自经典技术分析理论
    来源: Welles Wilder (RSI, ATR), John Bollinger (Bollinger Bands), Gerald Appel (MACD)
    适用场景: 趋势确认、超买超卖判断、波动率交易
    """
    factors = {}
    
    close_aligned = adj_close.reindex(close.index).fillna(adj_close)
    
    # RSI
    delta = close_aligned.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    factors['tech_RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close_aligned.ewm(span=12, adjust=False).mean()
    ema26 = close_aligned.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    factors['tech_MACD'] = macd
    factors['tech_MACD_signal'] = signal
    factors['tech_MACD_hist'] = macd - signal
    
    # Bollinger Bands
    sma20 = close_aligned.rolling(20).mean()
    std20 = close_aligned.rolling(20).std()
    factors['tech_BB_upper'] = sma20 + 2 * std20
    factors['tech_BB_lower'] = sma20 - 2 * std20
    factors['tech_BB_width'] = (factors['tech_BB_upper'] - factors['tech_BB_lower']) / sma20.replace(0, np.nan)
    factors['tech_BB_position'] = (close_aligned - factors['tech_BB_lower']) / (factors['tech_BB_upper'] - factors['tech_BB_lower']).replace(0, np.nan)
    
    # ATR
    tr1 = high - low
    tr2 = (high - close_aligned.shift(1)).abs()
    tr3 = (low - close_aligned.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    factors['tech_ATR_14'] = tr.rolling(14).mean()
    factors['tech_ATR_20'] = tr.rolling(20).mean()
    
    # Simple moving averages
    for w in [5, 10, 20, 60]:
        factors[f'tech_SMA_{w}'] = close_aligned / close_aligned.rolling(w).mean() - 1
    
    # Price ratios
    factors['tech_high_low_ratio'] = (high - low) / low.replace(0, np.nan)
    factors['tech_close_open_ratio'] = (close_aligned - open_p) / open_p.replace(0, np.nan)
    factors['tech_hl_position'] = (close_aligned - low) / (high - low).replace(0, np.nan)
    
    # Momentum
    factors['tech_momentum_5'] = close_aligned / close_aligned.shift(5) - 1
    factors['tech_momentum_10'] = close_aligned / close_aligned.shift(10) - 1
    factors['tech_momentum_20'] = close_aligned / close_aligned.shift(20) - 1
    factors['tech_momentum_60'] = close_aligned / close_aligned.shift(60) - 1
    
    # Stochastic
    factors['tech_stoch_K'] = 100 * (close_aligned - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min()).replace(0, np.nan)
    factors['tech_stoch_D'] = factors['tech_stoch_K'].rolling(3).mean()
    
    # Williams %R
    williams_high = high.rolling(14).max()
    williams_low = low.rolling(14).min()
    factors['tech_williams_R'] = -100 * (williams_high - close_aligned) / (williams_high - williams_low).replace(0, np.nan)
    
    # CCI
    cci = (close_aligned - close_aligned.rolling(20).mean()) / (0.015 * close_aligned.rolling(20).std())
    factors['tech_CCI_20'] = cci
    
    return pd.DataFrame(factors)


def process_single_stock(stock_data_tuple):
    """
    处理单个股票的因子计算
    
    参数:
    - stock_data_tuple: (stock, stock_data) 元组，包含股票代码和对应的股票数据
    """
    stock, stock_data = stock_data_tuple
    adj_close = stock_data['Adj Close'].dropna()
    close = stock_data['Close'].dropna().reindex(adj_close.index)
    open_p = stock_data['Open'].dropna().reindex(adj_close.index)
    high = stock_data['High'].dropna().reindex(adj_close.index)
    low = stock_data['Low'].dropna().reindex(adj_close.index)
    volume = stock_data['Volume'].dropna().reindex(adj_close.index)
    
    factor_row = {'stock': stock}
    
    alpha360 = calculate_alpha360_factors(close, open_p, high, low, volume, adj_close)
    biquant = calculate_biquant_factors(close, open_p, high, low, volume, adj_close)
    technical = calculate_technical_indicators(close, open_p, high, low, volume, adj_close)
    
    for df in [alpha360, biquant, technical]:
        for col in df.columns:
            factor_row[col] = df[col].iloc[-1] if len(df) > 0 else np.nan
    
    return factor_row


def calculate_factors_for_stocks(price_history, current_date, max_workers=None):
    """
    整合所有因子计算，为每只股票计算所有因子值
    
    因子分类:
    1. Alpha360扩展 - 多种时间窗口的量价关系
    2. BigQuant因子 - 常见技术分析因子
    3. 技术指标 - RSI/MACD/布林带等经典指标
    
    输出: DataFrame，行是股票代码，列是因子名称
    
    参数:
    - price_history: 价格历史数据
    - current_date: 当前日期
    - max_workers: 最大工作进程数，默认使用所有CPU核心
    """
    start_time = time.time()
    factors_dict = {}
    
    stocks = price_history.index.get_level_values('stock').unique()
    
    # 准备股票数据元组
    stock_data_tuples = []
    for stock in stocks:
        stock_data = price_history.xs(stock, level='stock')
        stock_data_tuples.append((stock, stock_data))
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {executor.submit(process_single_stock, stock_data_tuple): stock_data_tuple[0] for stock_data_tuple in stock_data_tuples}
        
        for future in tqdm(as_completed(future_to_stock), total=len(stocks), desc=f"Processing stocks for {current_date}"):
            stock = future_to_stock[future]
            try:
                factor_row = future.result()
                factors_dict[stock] = factor_row
            except Exception as e:
                print(f"Error processing stock {stock}: {e}")
    
    end_time = time.time()
    print(f"Parallel processing time for {len(stocks)} stocks: {end_time - start_time:.2f} seconds")
    
    return pd.DataFrame.from_dict(factors_dict, orient='index')


print("Factor calculation functions defined successfully!")
print("Factors include:")
print("  - Alpha360 extensions: multiple time windows")
print("  - BigQuant factors: momentum, reversal, MA crossover")
print("  - Technical indicators: RSI, MACD, Bollinger Bands, ATR, CCI, Williams %R, Stochastic")

all_dates = sorted([f.replace('.csv', '') for f in csv_files])
print(f"Processing {len(all_dates)} trading days...")

price_history = pd.DataFrame()
lookback_days = 252
max_workers = multiprocessing.cpu_count()
print(f"Using {max_workers} CPU cores for parallel processing")

# 主处理循环
for i, date_str in enumerate(tqdm(all_dates)):
    df_daily = pd.read_csv(os.path.join(data_dir, f'{date_str}.csv'))
    df_daily['Date'] = pd.to_datetime(date_str, format='%Y%m%d')
    df_daily = df_daily.set_index(['stock', 'Date'])
    
    price_history = pd.concat([price_history, df_daily])
    
    if len(price_history.index.get_level_values('Date').unique()) > lookback_days:
        oldest_date = sorted(price_history.index.get_level_values('Date').unique())[0]
        price_history = price_history.drop(oldest_date, level='Date')
    
    factors_df = calculate_factors_for_stocks(price_history, date_str, max_workers)
    factors_df = factors_df.reset_index().rename(columns={'index': 'stock'})
    
    output_path = os.path.join(output_dir, f'{date_str}.csv')
    factors_df.to_csv(output_path, index=False)

print(f"Factor calculation completed. Output saved to {output_dir}")
