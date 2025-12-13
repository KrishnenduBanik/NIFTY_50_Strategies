import pandas as pd
import numpy as np
import yfinance as yf

# Fetch stock data
def get_data(ticker='GRASIM.NS', start='2021-01-01'):
    """Fetch stock data using yfinance"""
    data = yf.download(ticker, start=start, progress=False)
    return data

# Strategy 1: Moving Average Crossover
def momentum_strategy(data, short=20, long=50):
    """Moving Average Crossover Strategy"""
    df = data.copy()
    df['Short_MA'] = df['Close'].rolling(short).mean()
    df['Long_MA'] = df['Close'].rolling(long).mean()
    df['Signal'] = np.where(df['Short_MA'] > df['Long_MA'], 1, -1)
    df['Position'] = df['Signal'].shift(1)
    return df

# Strategy 2: Mean Reversion (Bollinger Bands)
def mean_reversion_strategy(data, window=20, num_std=2):
    """Bollinger Bands Mean Reversion Strategy"""
    df = data.copy()
    df['SMA'] = df['Close'].rolling(window).mean()
    df['Std'] = df['Close'].rolling(window).std()
    df['Upper_Band'] = df['SMA'] + (df['Std'] * num_std)
    df['Lower_Band'] = df['SMA'] - (df['Std'] * num_std)
    
    df['Signal'] = 0
    df.loc[df['Close'] <= df['Lower_Band'], 'Signal'] = 1
    df.loc[df['Close'] >= df['Upper_Band'], 'Signal'] = -1
    df['Position'] = df['Signal'].shift(1)
    return df

# Strategy 3: RSI
def rsi_strategy(data, period=14, oversold=30, overbought=70):
    """RSI Based Strategy"""
    df = data.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Signal'] = 0
    df.loc[df['RSI'] < oversold, 'Signal'] = 1
    df.loc[df['RSI'] > overbought, 'Signal'] = -1
    df['Position'] = df['Signal'].shift(1)
    return df

# Strategy 4: Buy and Hold
def buy_hold_strategy(data):
    """Buy and Hold Benchmark"""
    df = data.copy()
    df['Signal'] = 1
    df['Position'] = 1
    return df

# Backtest
def backtest(data, initial_capital=100000, transaction_cost=0.0025):
    """Calculate returns and portfolio value"""
    df = data.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    
    # Transaction costs
    df['Trade'] = df['Position'].diff().fillna(0)
    df['Transaction_Cost'] = abs(df['Trade']) * transaction_cost
    df['Strategy_Returns_Net'] = df['Strategy_Returns'] - df['Transaction_Cost']
    
    # Cumulative returns
    df['Cumulative_Market'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns_Net']).cumprod()
    df['Portfolio_Value'] = initial_capital * df['Cumulative_Strategy']
    df['Market_Value'] = initial_capital * df['Cumulative_Market']
    
    # Calculate metrics
    total_return = df['Cumulative_Strategy'].iloc[-1] - 1
    market_return = df['Cumulative_Market'].iloc[-1] - 1
    excess_return = total_return - market_return
    volatility = df['Strategy_Returns_Net'].std() * np.sqrt(252)
    sharpe = (df['Strategy_Returns_Net'].mean() * 252) / volatility if volatility != 0 else 0
    
    # Max drawdown
    cumulative = df['Cumulative_Strategy']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate and trades
    win_rate = (df['Strategy_Returns_Net'] > 0).mean()
    num_trades = (df['Trade'] != 0).sum()
    
    metrics = {
        'Total Return': total_return,
        'Market Return': market_return,
        'Excess Return': excess_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Trades': num_trades,
        'Final Value': df['Portfolio_Value'].iloc[-1]
    }
    
    return df, metrics

# Main execution
def main():
    print("📊 Stock Trading Strategy Analysis")
    print("=" * 70)
    
    # Get data
    stock_data = get_data('GRASIM.NS', '2021-01-01')
    print(f"✅ Data fetched: {len(stock_data)} rows from {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
    
    # Define all strategies
    strategies = {
        'Momentum (MA 20/50)': lambda d: momentum_strategy(d, 20, 50),
        'Mean Reversion (BB)': lambda d: mean_reversion_strategy(d, 20, 2),
        'RSI Strategy': lambda d: rsi_strategy(d, 14, 30, 70),
        'Buy & Hold': buy_hold_strategy
    }
    
    # Test all strategies
    results = []
    
    for strategy_name, strategy_func in strategies.items():
        print(f"\n{'='*70}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*70}")
        
        # Apply strategy
        strategy_data = strategy_func(stock_data.copy())
        
        # Backtest
        backtest_data, metrics = backtest(strategy_data)
        
        # Print metrics
        print(f"Total Return:    {metrics['Total Return']:.2%}")
        print(f"Market Return:   {metrics['Market Return']:.2%}")
        print(f"Excess Return:   {metrics['Excess Return']:.2%}")
        print(f"Sharpe Ratio:    {metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown:    {metrics['Max Drawdown']:.2%}")
        print(f"Win Rate:        {metrics['Win Rate']:.2%}")
        print(f"Total Trades:    {metrics['Trades']}")
        print(f"Final Value:     ₹{metrics['Final Value']:,.2f}")
        
        results.append({**{'Strategy': strategy_name}, **metrics})
    
    # Compare all strategies
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(results).sort_values('Total Return', ascending=False)
    
    print("\n" + results_df.to_string(index=False, 
          columns=['Strategy', 'Total Return', 'Excess Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Trades'],
          formatters={
              'Total Return': '{:.2%}'.format,
              'Excess Return': '{:.2%}'.format,
              'Sharpe Ratio': '{:.2f}'.format,
              'Max Drawdown': '{:.2%}'.format,
              'Win Rate': '{:.2%}'.format
          }))
    
    # Best strategy
    best = results_df.iloc[0]
    print(f"\n🏆 BEST STRATEGY: {best['Strategy']}")
    print(f"   Return: {best['Total Return']:.2%} | Sharpe: {best['Sharpe Ratio']:.2f} | Drawdown: {best['Max Drawdown']:.2%}")
    
    # Return results for further analysis if needed
    return results_df

if __name__ == "__main__":
    results = main()
