import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats.mstats import winsorize

class DataLoader:
    def __init__(self, prediction_dir, label_dir):
        self.prediction_dir = prediction_dir
        self.label_dir = label_dir
        self.prediction_data = {}
        self.label_data = {}
        self.dates = []
    
    def load_data(self):
        pred_files = sorted(glob.glob(os.path.join(self.prediction_dir, '*.csv')))
        label_files = sorted(glob.glob(os.path.join(self.label_dir, '*.csv')))
        
        pred_dates = [os.path.basename(f).replace('.csv', '') for f in pred_files]
        label_dates = [os.path.basename(f).replace('.csv', '') for f in label_files]
        
        common_dates = sorted(set(pred_dates) & set(label_dates))
        
        for date in common_dates:
            pred_path = os.path.join(self.prediction_dir, f'{date}.csv')
            label_path = os.path.join(self.label_dir, f'{date}.csv')
            
            try:
                pred_df = pd.read_csv(pred_path)
                label_df = pd.read_csv(label_path)
                
                pred_df['date'] = date
                label_df['date'] = date
                
                self.prediction_data[date] = pred_df
                self.label_data[date] = label_df
            except Exception as e:
                print(f"Error loading data for {date}: {e}")
        
        self.dates = sorted(common_dates)
        print(f"Loaded {len(self.dates)} common dates from {len(pred_files)} prediction files and {len(label_files)} label files")
    
    def get_data_by_date(self, date):
        return self.prediction_data.get(date), self.label_data.get(date)
    
    def get_all_dates(self):
        return self.dates

class OutlierHandler:
    def __init__(self, method='winsorize', limits=(0.01, 0.01)):
        self.method = method
        self.limits = limits
    
    def handle_outliers(self, data, columns):
        if self.method == 'winsorize':
            for col in columns:
                if col in data.columns:
                    data[col] = winsorize(data[col], limits=self.limits)
        elif self.method == 'iqr':
            for col in columns:
                if col in data.columns:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    data[col] = np.clip(data[col], lower_bound, upper_bound)
        return data

class PositionManager:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.portfolio_history = []
        self.transaction_history = []
    
    def get_positions(self):
        return self.positions.copy()
    
    def get_portfolio_value(self, prices):
        position_value = sum(
            pos['quantity'] * prices.get(stock, pos['avg_cost'])
            for stock, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def buy(self, stock, price, quantity):
        cost = price * quantity
        if cost <= self.cash and price > 0 and quantity > 0:
            self.cash -= cost
            if stock in self.positions:
                total_qty = self.positions[stock]['quantity'] + quantity
                total_cost = self.positions[stock]['avg_cost'] * self.positions[stock]['quantity'] + cost
                self.positions[stock] = {
                    'quantity': total_qty,
                    'avg_cost': total_cost / total_qty,
                    'current_price': price
                }
            else:
                self.positions[stock] = {
                    'quantity': quantity,
                    'avg_cost': price,
                    'current_price': price
                }
            self.transaction_history.append({
                'type': 'BUY',
                'stock': stock,
                'price': price,
                'quantity': quantity,
                'cost': cost,
                'date': datetime.now().strftime('%Y-%m-%d')
            })
            return True
        return False
    
    def sell(self, stock, price, quantity=None):
        if stock in self.positions:
            if quantity is None:
                quantity = self.positions[stock]['quantity']
            quantity = min(quantity, self.positions[stock]['quantity'])
            
            if quantity > 0 and price > 0:
                proceeds = price * quantity
                self.cash += proceeds
                
                self.positions[stock]['quantity'] -= quantity
                self.positions[stock]['current_price'] = price
                
                if self.positions[stock]['quantity'] <= 0:
                    del self.positions[stock]
                
                self.transaction_history.append({
                    'type': 'SELL',
                    'stock': stock,
                    'price': price,
                    'quantity': quantity,
                    'proceeds': proceeds,
                    'date': datetime.now().strftime('%Y-%m-%d')
                })
                return True
        return False
    
    def update_prices(self, prices):
        for stock in self.positions:
            if stock in prices:
                self.positions[stock]['current_price'] = prices[stock]
    
    def record_history(self, date, portfolio_value):
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy()
        })

class RiskController:
    def __init__(self, max_drawdown=0.2, max_position_size=0.1, stop_loss=0.05):
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.peak_value = 0
    
    def check_drawdown(self, current_value, initial_value):
        self.peak_value = max(self.peak_value, current_value)
        drawdown = (self.peak_value - current_value) / self.peak_value
        return drawdown > self.max_drawdown, drawdown
    
    def check_position_size(self, position_value, portfolio_value):
        return (position_value / portfolio_value) > self.max_position_size
    
    def check_stop_loss(self, current_price, avg_cost):
        if avg_cost <= 0:
            return False
        loss = (avg_cost - current_price) / avg_cost
        return loss > self.stop_loss

class StrategyEngine:
    def __init__(self, data_loader, position_manager, risk_controller):
        self.data_loader = data_loader
        self.position_manager = position_manager
        self.risk_controller = risk_controller
        self.signal_column = 'return_5d_pred'
        self.holding_period = 5
        self.rebalance_interval = 5
        self.top_n = 10
        self.bottom_n = 10
    
    def run_long_strategy(self, signals, prices):
        current_positions = self.position_manager.get_positions()
        
        signals = signals.sort_values(self.signal_column, ascending=False)
        long_stocks = set(signals.head(self.top_n)['stock'])
        
        for stock in list(current_positions.keys()):
            pos = current_positions[stock]
            if pos['quantity'] > 0 and stock not in long_stocks:
                self.position_manager.sell(stock, prices.get(stock, pos['avg_cost']))
        
        available_cash = self.position_manager.cash
        if len(long_stocks) > 0:
            capital_per_stock = available_cash / len(long_stocks)
            
            for stock in long_stocks:
                if stock not in current_positions or current_positions.get(stock, {}).get('quantity', 0) <= 0:
                    price = prices.get(stock, 1)
                    if price > 0 and capital_per_stock > 0:
                        quantity = int(capital_per_stock / price)
                        if quantity > 0:
                            self.position_manager.buy(stock, price, quantity)
    
    def run_short_strategy(self, signals, prices):
        current_positions = self.position_manager.get_positions()
        
        signals = signals.sort_values(self.signal_column, ascending=True)
        short_stocks = set(signals.head(self.bottom_n)['stock'])
        
        for stock in list(current_positions.keys()):
            pos = current_positions[stock]
            if pos['quantity'] < 0 and stock not in short_stocks:
                self.position_manager.sell(stock, prices.get(stock, abs(pos['avg_cost'])))
        
        available_cash = self.position_manager.cash
        if len(short_stocks) > 0:
            capital_per_stock = available_cash / len(short_stocks)
            
            for stock in short_stocks:
                if stock not in current_positions or current_positions.get(stock, {}).get('quantity', 0) >= 0:
                    price = prices.get(stock, 1)
                    if price > 0 and capital_per_stock > 0:
                        quantity = -int(capital_per_stock / price)
                        if quantity < 0:
                            self.position_manager.buy(stock, price, quantity)
    
    def run_long_short_strategy(self, signals, prices):
        current_positions = self.position_manager.get_positions()
        
        signals = signals.sort_values(self.signal_column, ascending=False)
        long_stocks = set(signals.head(self.top_n)['stock'])
        
        signals_sorted = signals.sort_values(self.signal_column, ascending=True)
        short_stocks = set(signals_sorted.head(self.bottom_n)['stock'])
        
        long_stocks = long_stocks - short_stocks
        short_stocks = short_stocks - long_stocks
        
        for stock in list(current_positions.keys()):
            pos = current_positions[stock]
            if pos['quantity'] > 0 and stock not in long_stocks:
                self.position_manager.sell(stock, prices.get(stock, pos['avg_cost']))
            elif pos['quantity'] < 0 and stock not in short_stocks:
                self.position_manager.sell(stock, prices.get(stock, abs(pos['avg_cost'])))
        
        available_cash = self.position_manager.cash
        total_stocks = len(long_stocks) + len(short_stocks)
        
        if total_stocks > 0:
            capital_per_stock = available_cash / total_stocks
            
            for stock in long_stocks:
                if stock not in current_positions or current_positions.get(stock, {}).get('quantity', 0) <= 0:
                    price = prices.get(stock, 1)
                    if price > 0 and capital_per_stock > 0:
                        quantity = int(capital_per_stock / price)
                        if quantity > 0:
                            self.position_manager.buy(stock, price, quantity)
            
            for stock in short_stocks:
                if stock not in current_positions or current_positions.get(stock, {}).get('quantity', 0) >= 0:
                    price = prices.get(stock, 1)
                    if price > 0 and capital_per_stock > 0:
                        quantity = -int(capital_per_stock / price)
                        if quantity < 0:
                            self.position_manager.buy(stock, price, quantity)

class PerformanceAnalyzer:
    def __init__(self, position_manager, risk_free_rate=0.02):
        self.position_manager = position_manager
        self.risk_free_rate = risk_free_rate
        self.portfolio_history = position_manager.portfolio_history
        self.transaction_history = position_manager.transaction_history
    
    def calculate_metrics(self):
        if not self.portfolio_history:
            return {}
        
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        df['daily_return'] = df['portfolio_value'].pct_change().fillna(0)
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()
        
        total_return = df['cumulative_return'].iloc[-1] - 1
        
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        daily_std = df['daily_return'].std()
        annualized_volatility = daily_std * np.sqrt(252)
        
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
        
        df['peak'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min()
        
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None
        
        for idx, row in df.iterrows():
            if row['drawdown'] < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = row['date']
            elif row['drawdown'] == 0 and in_drawdown:
                drawdown_periods.append((drawdown_start, row['date']))
                in_drawdown = False
        
        max_drawdown_duration = 0
        if drawdown_periods:
            max_drawdown_duration = max((end - start).days for start, end in drawdown_periods)
        elif in_drawdown and drawdown_start:
            max_drawdown_duration = (df['date'].iloc[-1] - drawdown_start).days
        
        transactions = pd.DataFrame(self.transaction_history)
        num_trades = len(transactions)
        
        profitable_trades = 0
        total_profit = 0
        total_loss = 0
        max_profit = 0
        max_loss = 0
        
        for _, trade in transactions.iterrows():
            if trade['type'] == 'BUY':
                continue
            
            pos = next((p for p in self.position_manager.portfolio_history 
                      if trade['stock'] in p['positions']), None)
            if pos:
                avg_cost = pos['positions'][trade['stock']]['avg_cost']
                profit = (trade['price'] - avg_cost) * trade['quantity']
                if profit > 0:
                    profitable_trades += 1
                    total_profit += profit
                    max_profit = max(max_profit, profit)
                else:
                    total_loss += abs(profit)
                    max_loss = min(max_loss, profit)
        
        win_rate = profitable_trades / num_trades if num_trades > 0 else 0
        profit_loss_ratio = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_profit = total_profit / profitable_trades if profitable_trades > 0 else 0
        avg_loss = total_loss / (num_trades - profitable_trades) if (num_trades - profitable_trades) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'num_trades': num_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': num_trades - profitable_trades,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_single_profit': max_profit,
            'max_single_loss': max_loss,
            'risk_free_rate': self.risk_free_rate,
            'start_date': df['date'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'duration_days': days
        }
    
    def plot_equity_curve(self, save_path=None):
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['portfolio_value'], label='Portfolio Value', color='blue')
        plt.plot(df['date'], df['cash'], label='Cash', color='green', alpha=0.5)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_drawdown_curve(self, save_path=None):
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        df['peak'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['peak']) / df['peak']
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df['date'], df['drawdown'], 0, where=df['drawdown'] < 0, 
                        color='red', alpha=0.3)
        plt.plot(df['date'], df['drawdown'], label='Drawdown', color='red')
        plt.title('Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_monthly_returns(self, save_path=None):
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['daily_return'] = df['portfolio_value'].pct_change().fillna(0)
        
        df['month'] = df['date'].dt.to_period('M')
        monthly_returns = df.groupby('month')['daily_return'].sum()
        
        plt.figure(figsize=(12, 6))
        monthly_returns.plot(kind='bar', color='skyblue')
        plt.title('Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Return')
        plt.grid(True)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_performance_heatmap(self, save_path=None):
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['daily_return'] = df['portfolio_value'].pct_change().fillna(0)
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        pivot = df.pivot_table(values='daily_return', index='year', columns='month', aggfunc='sum')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def print_metrics(self, metrics):
        print("=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Backtest Period: {metrics['start_date']} to {metrics['end_date']} ({metrics['duration_days']} days)")
        print(f"Risk-Free Rate: {metrics['risk_free_rate']:.2%}")
        print("-" * 60)
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print("-" * 60)
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Max Drawdown Duration: {metrics['max_drawdown_duration']} days")
        print("-" * 60)
        print(f"Number of Trades: {metrics['num_trades']}")
        print(f"Profitable Trades: {metrics['profitable_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit/Loss Ratio: {metrics['profit_loss_ratio']:.2f}")
        print("-" * 60)
        print(f"Average Profit per Winning Trade: ${metrics['avg_profit']:,.2f}")
        print(f"Average Loss per Losing Trade: ${metrics['avg_loss']:,.2f}")
        print(f"Max Single Profit: ${metrics['max_single_profit']:,.2f}")
        print(f"Max Single Loss: ${metrics['max_single_loss']:,.2f}")
        print("=" * 60)

class Backtester:
    def __init__(self, prediction_dir, label_dir, output_dir='backtest_results'):
        self.data_loader = DataLoader(prediction_dir, label_dir)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_backtest(self, strategy_type='long', initial_capital=1000000, 
                     max_drawdown=0.2, max_position_size=0.1, stop_loss=0.05,
                     top_n=10, bottom_n=10, rebalance_interval=5):
        
        self.data_loader.load_data()
        dates = self.data_loader.get_all_dates()
        
        if not dates:
            print("No data available for backtesting")
            return None
        
        position_manager = PositionManager(initial_capital)
        risk_controller = RiskController(max_drawdown, max_position_size, stop_loss)
        strategy_engine = StrategyEngine(self.data_loader, position_manager, risk_controller)
        
        strategy_engine.top_n = top_n
        strategy_engine.bottom_n = bottom_n
        strategy_engine.rebalance_interval = rebalance_interval
        
        outlier_handler = OutlierHandler(method='winsorize', limits=(0.01, 0.01))
        
        for i, date in enumerate(dates):
            pred_df, label_df = self.data_loader.get_data_by_date(date)
            
            if pred_df is None or label_df is None:
                continue
            
            merged_df = pd.merge(pred_df, label_df, on='stock', suffixes=('_pred', '_actual'))
            
            merged_df = outlier_handler.handle_outliers(merged_df, ['return_5d_pred', 'return_5d_actual'])
            
            prices = dict(zip(merged_df['stock'], merged_df.get('return_5d_actual', merged_df['return_5d_pred']) + 100))
            
            position_manager.update_prices(prices)
            
            if i % rebalance_interval == 0:
                if strategy_type == 'long':
                    strategy_engine.run_long_strategy(merged_df, prices)
                elif strategy_type == 'short':
                    strategy_engine.run_short_strategy(merged_df, prices)
                elif strategy_type == 'long_short':
                    strategy_engine.run_long_short_strategy(merged_df, prices)
            
            portfolio_value = position_manager.get_portfolio_value(prices)
            position_manager.record_history(date, portfolio_value)
            
            _, drawdown = risk_controller.check_drawdown(portfolio_value, initial_capital)
            if drawdown > max_drawdown:
                print(f"Drawdown limit reached at {date}: {drawdown:.2%}")
        
        analyzer = PerformanceAnalyzer(position_manager)
        metrics = analyzer.calculate_metrics()
        
        analyzer.print_metrics(metrics)
        
        analyzer.plot_equity_curve(os.path.join(self.output_dir, f'{strategy_type}_equity_curve.png'))
        analyzer.plot_drawdown_curve(os.path.join(self.output_dir, f'{strategy_type}_drawdown_curve.png'))
        analyzer.plot_monthly_returns(os.path.join(self.output_dir, f'{strategy_type}_monthly_returns.png'))
        analyzer.plot_performance_heatmap(os.path.join(self.output_dir, f'{strategy_type}_performance_heatmap.png'))
        
        pd.DataFrame(position_manager.portfolio_history).to_csv(
            os.path.join(self.output_dir, f'{strategy_type}_portfolio_history.csv'), index=False)
        pd.DataFrame(position_manager.transaction_history).to_csv(
            os.path.join(self.output_dir, f'{strategy_type}_transaction_history.csv'), index=False)
        
        with open(os.path.join(self.output_dir, f'{strategy_type}_metrics.txt'), 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        return metrics

if __name__ == '__main__':
    prediction_dir = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/outputs/prediction_lgbm'
    label_dir = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/daily_label_data'
    output_dir = '/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/outputs/backtest_lgbm'
    
    backtester = Backtester(prediction_dir, label_dir, output_dir)
    
    print("Running Long Strategy...")
    long_metrics = backtester.run_backtest(strategy_type='long')
    
    print("\nRunning Short Strategy...")
    short_metrics = backtester.run_backtest(strategy_type='short')
    
    print("\nRunning Long-Short Strategy...")
    long_short_metrics = backtester.run_backtest(strategy_type='long_short')
    
    all_metrics = pd.DataFrame({
        'Long': pd.Series(long_metrics),
        'Short': pd.Series(short_metrics),
        'Long-Short': pd.Series(long_short_metrics)
    })
    
    all_metrics.to_csv(os.path.join(output_dir, 'strategy_comparison.csv'))
    
    print("\nStrategy Comparison:")
    print(all_metrics)