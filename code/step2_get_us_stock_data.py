import pandas as pd
import os
import glob
from tqdm.notebook import tqdm

stocks_dir = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/stock_market_data/stocks/'

csv_files = glob.glob(os.path.join(stocks_dir, '*.csv'))
print(f"Found {len(csv_files)} stock files")

all_data = {}
for file in tqdm(csv_files):
    symbol = os.path.basename(file).replace('.csv', '')
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df.columns = [f'{symbol}_{col}' for col in df.columns]
    all_data[symbol] = df

merged_df = pd.concat(all_data.values(), axis=1)
merged_df = merged_df.sort_index()
print(f"Merged data shape: {merged_df.shape}")
print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
merged_df.head()

output_dir = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/daily_stock_data'
os.makedirs(output_dir, exist_ok=True)

for date in tqdm([date for date in merged_df.index.unique() if date.year >= 2011]):
    date_str = date.strftime('%Y%m%d')
    daily_row = merged_df.loc[date]
    
    records = []
    for col in merged_df.columns:
        symbol = col.split('_')[0]
        field = col[len(symbol)+1:]
        records.append({'stock': symbol, 'field': field, 'value': daily_row[col]})
    
    daily_df = pd.DataFrame(records)
    daily_df = daily_df.pivot(index='stock', columns='field', values='value')
    
    output_path = os.path.join(output_dir, f'{date_str}.csv')
    daily_df.to_csv(output_path, index=True)

print(f"Saved {len(merged_df.index.unique())} csv files to {output_dir}")
