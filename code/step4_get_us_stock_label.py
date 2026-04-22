import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict

DATA_DIR = "/root/autodl-tmp/.autodl/project/data/daily_stock_data"
OUTPUT_DIR = "/root/autodl-tmp/.autodl/project/data/daily_label_data"

RETURN_PERIODS = [5, 10, 20, 60, 120, 252]
VOLATILITY_PERIODS = [5, 10, 20, 60, 120, 252]
MAX_DRAWDOWN_PERIODS = [5, 10, 20, 60, 120, 252]


def get_date_files(data_dir):
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(glob.glob(pattern))


def calculate_labels_vectorized(prices, current_idx):
    record = {}

    current_price = prices[current_idx]

    # ========= Future Returns =========
    for days in RETURN_PERIODS:
        col_name = f'return_{days}d'
        future_idx = current_idx + days

        if future_idx < len(prices) and current_price != 0:
            future_price = prices[future_idx]
            record[col_name] = (future_price / current_price - 1) * 100
        else:
            record[col_name] = np.nan

    # ========= Future Volatility =========
    for period in VOLATILITY_PERIODS:
        col_name = f'volatility_{period}d'

        future_window = prices[current_idx: current_idx + period + 1]

        if len(future_window) == period + 1:
            future_returns = np.diff(future_window) / future_window[:-1] * 100
            record[col_name] = np.std(future_returns) * np.sqrt(252)
        else:
            record[col_name] = np.nan

    # ========= Future Max Drawdown =========
    for period in MAX_DRAWDOWN_PERIODS:
        col_name = f'max_drawdown_{period}d'

        future_window = prices[current_idx: current_idx + period + 1]

        if len(future_window) >= 2:
            rolling_max = np.maximum.accumulate(future_window)
            drawdowns = (future_window - rolling_max) / rolling_max * 100
            record[col_name] = np.min(drawdowns)
        else:
            record[col_name] = np.nan

    return record


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    date_files = get_date_files(DATA_DIR)
    date_strs = [os.path.basename(f).replace('.csv', '') for f in date_files]

    print(f"Found {len(date_files)} daily data files")

    # ========= Collect stock prices =========
    stock_data = defaultdict(dict)

    for file_idx, file_path in enumerate(date_files):
        date_str = date_strs[file_idx]
        df = pd.read_csv(file_path)
        stocks = df['stock'].values
        adj_closes = df['Adj Close'].values
        for stock, price in zip(stocks, adj_closes):
            stock_data[stock][date_str] = price

    print(f"Collected data for {len(stock_data)} unique stocks")

    # ========= Preprocess stock price series =========
    stock_processed = {}
    for stock, price_dict in stock_data.items():
        sorted_dates = sorted(price_dict.keys())
        filtered_dates = []
        prices_list = []
        for d in sorted_dates:
            price = price_dict[d]
            if not pd.isna(price):
                filtered_dates.append(d)
                prices_list.append(price)

        prices_array = np.array(prices_list, dtype=np.float64)
        date_to_idx = {d: idx for idx, d in enumerate(filtered_dates)}
        stock_processed[stock] = {
            'prices': prices_array,
            'date_to_idx': date_to_idx
        }

    print(f"Prepared {len(stock_processed)} valid stocks")

    # ========= Generate labels =========
    processed_count = 0

    for file_idx, file_path in enumerate(date_files):
        date_str = date_strs[file_idx]
        output_file = os.path.join(OUTPUT_DIR, f"{date_str}.csv")
        df = pd.read_csv(file_path)
        stocks = df['stock'].values
        label_records = []

        for stock in stocks:
            if stock not in stock_processed:
                continue

            stock_info = stock_processed[stock]
            prices_array = stock_info['prices']
            date_to_idx = stock_info['date_to_idx']

            if date_str not in date_to_idx:
                continue

            current_idx = date_to_idx[date_str]

            labels = calculate_labels_vectorized(prices_array, current_idx)
            record = {'stock': stock}
            record.update(labels)
            label_records.append(record)

        if label_records:
            label_df = pd.DataFrame(label_records)
            label_df = label_df.sort_values('stock').reset_index(drop=True)

            column_order = (
                ['stock'] +
                [f'return_{d}d' for d in RETURN_PERIODS] +
                [f'volatility_{p}d' for p in VOLATILITY_PERIODS] +
                [f'max_drawdown_{p}d' for p in MAX_DRAWDOWN_PERIODS]
            )

            label_df = label_df[column_order]
            label_df.to_csv(output_file, index=False)

            processed_count += 1

            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")

    print("\nProcessing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()