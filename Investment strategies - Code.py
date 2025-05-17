import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from datetime import datetime
import plotly.graph_objects as go
import yfinance as yf
import warnings

# Set page config
st.set_page_config(page_title="Investment Strategy Analysis", layout="wide")

# Initialize session state variables if they don't exist
if 'z_threshold' not in st.session_state:
    st.session_state['z_threshold'] = 1.5
if 'high_vol_threshold' not in st.session_state:
    st.session_state['high_vol_threshold'] = 0.32
if 'low_vol_threshold' not in st.session_state:
    st.session_state['low_vol_threshold'] = 0.30

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set seaborn theme
sns.set_theme()

# Add caching for price data retrieval
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def retrieve_prices_yf(tickers: list, start_date: str, end_date: str, frequency: str = 'D',
                    common_inception: bool = False) -> pd.DataFrame:
    """
    Retrieve adjusted close prices for given investment tickers within a specified date range.
    """
    if not isinstance(tickers, list) or not all(isinstance(ticker, str) for ticker in tickers):
        raise ValueError('Please supply tickers in a list of strings')

    df_prices = yf.download(tickers, start_date, end_date)['Close']

    if frequency == 'W':
        df_prices = df_prices.resample('W-FRI').last()
    elif frequency == 'M':
        df_prices = df_prices.resample('ME').last()

    if common_inception:
        df_prices.dropna(inplace=True)

    df_prices.index = df_prices.index.tz_localize(None)
    return df_prices

def cumulative_return_graph_plotly(returns, start_date=None, end_date=None):
    """
    Generates a line graph showing the cumulative return of investments over time.
    """
    if start_date is None:
        start_date = returns.index[0]
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    if end_date is None:
        end_date = returns.index[-1]
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    returns = returns.loc[start_date:end_date]
    df_cumulative = ((1 + returns).cumprod() - 1).mul(100)
    x_values = df_cumulative.index.to_numpy()

    fig = go.Figure()
    for col in df_cumulative.columns:
        fig.add_trace(go.Scatter(x=x_values, y=df_cumulative[col], mode='lines', name=col))

    fig.update_layout(
        title='Cumulative Percent Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        yaxis_tickformat=',',
    )
    return fig

def rolling_returns(prices, period, frequency='D', annualize=True):
    """
    Calculate rolling returns for a given DataFrame of prices over a specified period.
    """
    df_rolling = prices.pct_change(period).dropna()

    if annualize:
        match frequency:
            case 'D':
                periods_per_year = 252
            case 'M':
                periods_per_year = 12
        df_rolling = (1 + df_rolling) ** (periods_per_year / period) - 1

    return df_rolling

def plot_rolling_returns(returns, highlight_col=None, alpha=1.0, title=None):
    """
    Plot rolling returns from a DataFrame.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, col in enumerate(returns.columns):
        line_alpha = alpha if col != highlight_col else 1.0
        sns.lineplot(data=returns[col], alpha=line_alpha, ax=ax, label=col)

    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Returns')
    ax.set_title(title, loc='center')
    ax.legend(frameon=False, loc='best')
    ax.yaxis.tick_left()
    return fig

def rolling_return_stats(rolling_returns, benchmark):
    """
    Calculate statistics for rolling returns compared to a benchmark.
    """
    df_stats = pd.DataFrame()
    for col in rolling_returns.columns:
        percent_greater = (rolling_returns[col] > rolling_returns[benchmark]).sum() / rolling_returns.shape[0]
        df_stats.loc[col, '% Periods Greater Than Benchmark'] = np.round(percent_greater * 100, 2)

    df_stats['Maximum Return (%)'] = rolling_returns.max().round(2)
    df_stats['Minimum Return (%)'] = rolling_returns.min().round(2)
    return df_stats

def expanding_zscore(series):
    """
    Calculate expanding z-scores up to each point.
    """
    expanding_mean = series.expanding().mean()
    expanding_std = series.expanding().std()
    return (series - expanding_mean) / expanding_std

def backtest_zscore_strategy_with_volatility(df: pd.DataFrame, z_col: str, threshold: float,
                                             spy_return_col: str, rsp_return_col: str,
                                             volatility_col: str,
                                             high_vol_threshold: float = 0.26, low_vol_threshold: float = 0.24,
                                             benchmark_return_col: str = None, initial_pos: str = None,
                                             plot=False) -> pd.DataFrame:
    """
    Backtests a strategy based on rolling z-score differentials between SPY and RSP, incorporates a volatility filter
    with tolerance bands, and compares it with a benchmark.
    """
    if benchmark_return_col is None:
        benchmark_return_col = spy_return_col

    df = df.dropna(subset=[z_col]).copy()
    df['Position'] = np.nan

    df.loc[df[z_col] > threshold, 'Position'] = 'RSP'
    df.loc[df[z_col] < -threshold, 'Position'] = 'SPY'

    if pd.isna(df['Position'].iloc[0]):
        if initial_pos is None:
            raise ValueError("Please provide an initial position (SPY or RSP) as the z-score does not exceed thresholds at the beginning.")
        df.loc[df.index[0], 'Position'] = initial_pos

    df['Position'] = df['Position'].fillna(method='ffill')

    in_cash = df.iloc[0][volatility_col] > high_vol_threshold

    for i in range(1, len(df)):
        vol = df.iloc[i][volatility_col]
        if in_cash:
            if vol < low_vol_threshold:
                in_cash = False
        else:
            if vol > high_vol_threshold:
                in_cash = True
        if in_cash:
            df.loc[df.index[i], 'Position'] = 'Cash'

    df['Position'] = df['Position'].shift(1)
    df = df.dropna(subset=['Position'])

    df['Strategy_Return'] = np.where(df['Position'] == 'RSP', df[rsp_return_col],
                                     np.where(df['Position'] == 'SPY', df[spy_return_col], 0))
    df['Strategy_Return'] = np.where(df['Position'] == 'Cash', 0, df['Strategy_Return'])
    df['Strategy_Return'] = df['Strategy_Return'].fillna(0)

    # Properly initialize the cumulative returns
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1
    
    # Make sure benchmark column exists and is properly initialized
    if benchmark_return_col in df.columns:
        # Initialize the first row to 0 to start benchmark at same point as strategy
        df.loc[df.index[0], benchmark_return_col] = 0
        df['Cumulative_Benchmark_Return'] = (1 + df[benchmark_return_col]).cumprod() - 1
    else:
        st.warning(f"Benchmark column {benchmark_return_col} not found in data. Using SPY returns as fallback.")
        benchmark_return_col = spy_return_col
        df.loc[df.index[0], benchmark_return_col] = 0
        df['Cumulative_Benchmark_Return'] = (1 + df[benchmark_return_col]).cumprod() - 1

    if plot:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[volatility_col], label='SPY Rolling Volatility', color='blue')
        plt.axhline(y=high_vol_threshold, color='red', linestyle='--', label=f'High Vol Threshold ({high_vol_threshold})')
        plt.axhline(y=low_vol_threshold, color='green', linestyle='--', label=f'Low Vol Threshold ({low_vol_threshold})')

        cash_days = df[df['Position'] == 'Cash'].index
        for cash_day in cash_days:
            plt.axvline(x=cash_day, color='orange', linestyle=':', label='In Cash' if cash_day == cash_days[0] else "")

        plt.title('Volatility and Strategy Positioning')
        plt.xlabel('Date')
        plt.ylabel('Volatility (Annualized)')
        plt.legend(frameon=False)
        st.pyplot(fig)

    return df[['Position', 'Strategy_Return', 'Cumulative_Strategy_Return',
               benchmark_return_col, 'Cumulative_Benchmark_Return', volatility_col]]

# Add caching to performance metric calculations
@st.cache_data
def calculate_rolling_sharpe(returns: pd.Series, window: int = 36, annualization_factor: int = 12) -> pd.Series:
    """
    Calculate the rolling Sharpe ratio over a specified window.
    """
    # Compute the rolling mean of the returns over the specified window
    rolling_mean = returns.rolling(window=window).mean()

    # Compute the rolling standard deviation of the returns over the specified window
    rolling_std = returns.rolling(window=window).std()

    # Calculate the rolling Sharpe ratio: mean divided by standard deviation, annualized
    rolling_sharpe = (rolling_mean / rolling_std) * (annualization_factor ** 0.5)

    return rolling_sharpe

@st.cache_data
def calculate_rolling_drawdown(returns: pd.Series, window: int = 36) -> pd.Series:
    """
    Calculate the rolling maximum drawdown over a specified window based on returns.
    """
    # Step 1: Convert returns to cumulative returns
    cumulative_returns = (1 + returns).cumprod()

    # Step 2: Calculate the rolling maximum of cumulative returns within the window
    rolling_max = cumulative_returns.rolling(window=window, min_periods=1).max()

    # Step 3: Calculate drawdown from the rolling max
    drawdown = (cumulative_returns / rolling_max) - 1

    # Step 4: Calculate the maximum drawdown within each window
    rolling_drawdown = drawdown.rolling(window=window, min_periods=1).min()

    return rolling_drawdown

@st.cache_data
def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate the percentage of positive return periods.
    """
    positive_periods = (returns > 0).sum()
    total_periods = len(returns.dropna())
    
    if total_periods > 0:
        return (positive_periods / total_periods) * 100
    else:
        return 0

@st.cache_data
def calculate_calmar_ratio(returns: pd.Series, window: int = 252) -> float:
    """
    Calculate the Calmar ratio (annualized return divided by maximum drawdown).
    """
    # Calculate annualized return
    annualized_return = ((1 + returns).prod()) ** (252 / len(returns)) - 1
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    
    # Calculate Calmar ratio
    if max_drawdown != 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = float('inf')  # Avoid division by zero
    
    return calmar_ratio

# Add the optimization function with caching
@st.cache_data
def optimize_strategy_parameters(df_is, first_ticker, second_ticker, volatility_source):
    """
    Find optimal strategy parameters using grid search on in-sample data.
    
    Parameters:
    -----------
    df_is : DataFrame
        In-sample price data
    first_ticker, second_ticker : str
        Ticker symbols
    volatility_source : str
        Which ticker to use for volatility calculation
    
    Returns:
    --------
    DataFrame
        Results sorted by performance metric
    """
    # Parameter ranges to test
    z_thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    high_vol_thresholds = [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]
    low_vol_thresholds = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
    
    # Create a copy of the in-sample data
    df = df_is.copy()
    
    # Store results
    results = []
    
    # Progress bar
    progress_bar = st.progress(0)
    total_iterations = len(z_thresholds) * len(high_vol_thresholds) * len(low_vol_thresholds)
    current_iteration = 0
    
    # Create volatility column
    volatility_col = f'{volatility_source}_Rolling_Volatility'
    if volatility_col not in df.columns:
        df[volatility_col] = df[f'{volatility_source}_Return'].rolling(window=20).std().mul(np.sqrt(252))
    
    # Create relative z-score
    rel_return_col = f'Relative_{first_ticker}_{second_ticker}_Return'
    if rel_return_col not in df.columns:
        df[rel_return_col] = rolling_returns(df[first_ticker], 21, annualize=False) - rolling_returns(df[second_ticker], 21, annualize=False)
        df['Relative_Z_Score'] = expanding_zscore(df[rel_return_col])
    
    for z in z_thresholds:
        for high_vol in high_vol_thresholds:
            for low_vol in low_vol_thresholds:
                # Skip invalid combinations where low_vol >= high_vol
                if low_vol >= high_vol:
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_iterations)
                    continue
                
                # Run backtest
                try:
                    result = backtest_zscore_strategy_with_volatility(
                        df.copy(),
                        'Relative_Z_Score',
                        threshold=z,
                        spy_return_col=f'{first_ticker}_Return',
                        rsp_return_col=f'{second_ticker}_Return',
                        volatility_col=volatility_col,
                        high_vol_threshold=high_vol,
                        low_vol_threshold=low_vol,
                        plot=False,
                        initial_pos=first_ticker
                    )
                    
                    # Calculate performance metrics
                    strategy_returns = result['Strategy_Return']
                    cumulative_return = result['Cumulative_Strategy_Return'].iloc[-1]
                    
                    # Calculate Sharpe ratio (annualized)
                    sharpe = calculate_rolling_sharpe(
                        strategy_returns, 
                        window=min(252, len(strategy_returns)), 
                        annualization_factor=252
                    ).iloc[-1]
                    
                    # Calculate maximum drawdown
                    max_drawdown = calculate_rolling_drawdown(
                        strategy_returns, 
                        window=len(strategy_returns)
                    ).iloc[-1]
                    
                    # Calculate win rate
                    win_rate = calculate_win_rate(strategy_returns)
                    
                    # Calculate Calmar ratio
                    calmar = calculate_calmar_ratio(strategy_returns)
                    
                    # Store results
                    results.append({
                        'Z_Threshold': z,
                        'High_Vol_Threshold': high_vol,
                        'Low_Vol_Threshold': low_vol,
                        'Total_Return': cumulative_return * 100,  # as percentage
                        'Sharpe_Ratio': sharpe,
                        'Max_Drawdown': max_drawdown * 100,  # as percentage
                        'Win_Rate': win_rate,
                        'Calmar_Ratio': calmar
                    })
                    
                except Exception as e:
                    st.error(f"Error with parameters Z:{z}, High:{high_vol}, Low:{low_vol}: {str(e)}")
                
                # Update progress
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio by default
    if not results_df.empty:
        results_df = results_df.sort_values('Sharpe_Ratio', ascending=False)
    
    return results_df

# Function to run separate in-sample/out-of-sample backtests
def run_separate_backtest(df_prices, oos_date, z_threshold, high_vol_threshold, 
                     low_vol_threshold, initial_position, volatility_source):
    """
    Run backtest with separate in-sample and out-of-sample periods.
    """
    # Define benchmark return column for performance metrics at the beginning
    benchmark_return_col = 'SPY_Return'  # Default benchmark
    
    # Separate in-sample and out-of-sample data
    df_is = df_prices.loc[:oos_date].copy()
    df_oos = df_prices.loc[oos_date:].copy()
    
    # Define first and second ticker (assuming SPY and RSP)
    first_ticker = 'SPY'
    second_ticker = 'RSP'
    
    # Use the selected volatility source (default to first ticker if not selected)
    volatility_source = volatility_source if volatility_source else first_ticker
    volatility_col = f'{volatility_source}_Rolling_Volatility'
    
    # Calculate metrics for in-sample period
    df_is[volatility_col] = df_is[f'{volatility_source}_Return'].rolling(window=20).std().mul(np.sqrt(252))
    period_days = 21  # 1 Month
    rel_return_col = f'Relative_{first_ticker}_{second_ticker}_Return'
    df_is[rel_return_col] = rolling_returns(df_is[first_ticker], period_days, annualize=False) - rolling_returns(df_is[second_ticker], period_days, annualize=False)
    df_is['Relative_Z_Score'] = expanding_zscore(df_is[rel_return_col])
    
    # Calculate metrics for out-of-sample period
    df_oos[volatility_col] = df_oos[f'{volatility_source}_Return'].rolling(window=20).std().mul(np.sqrt(252))
    df_oos[rel_return_col] = rolling_returns(df_oos[first_ticker], period_days, annualize=False) - rolling_returns(df_oos[second_ticker], period_days, annualize=False)
    # Use expanding z-score based only on out-of-sample data
    df_oos['Relative_Z_Score'] = expanding_zscore(df_oos[rel_return_col])
    
    # Run in-sample backtest
    result_is = backtest_zscore_strategy_with_volatility(
        df_is,
        'Relative_Z_Score',
        threshold=z_threshold,
        spy_return_col=f'{first_ticker}_Return',
        rsp_return_col=f'{second_ticker}_Return',
        volatility_col=volatility_col,
        high_vol_threshold=high_vol_threshold,
        low_vol_threshold=low_vol_threshold,
        plot=False,
        initial_pos=initial_position,
        benchmark_return_col=benchmark_return_col
    )
    
    # Run out-of-sample backtest
    result_oos = backtest_zscore_strategy_with_volatility(
        df_oos,
        'Relative_Z_Score',
        threshold=z_threshold,
        spy_return_col=f'{first_ticker}_Return',
        rsp_return_col=f'{second_ticker}_Return',
        volatility_col=volatility_col,
        high_vol_threshold=high_vol_threshold,
        low_vol_threshold=low_vol_threshold,
        plot=False,
        initial_pos=initial_position,
        benchmark_return_col=benchmark_return_col
    )
    
    # Display in-sample results
    st.subheader("In-Sample Results (Training Period)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Strategy Return", f"{result_is['Cumulative_Strategy_Return'].iloc[-1]*100:.2f}%")
    with col2:
        st.metric("Benchmark Return", f"{result_is['Cumulative_Benchmark_Return'].iloc[-1]*100:.2f}%")
    with col3:
        outperformance = (result_is['Cumulative_Strategy_Return'].iloc[-1] - result_is['Cumulative_Benchmark_Return'].iloc[-1])*100
        st.metric("Outperformance", f"{outperformance:.2f}%")

    # Add performance metrics for in-sample period
    st.subheader("In-Sample Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    # Calculate metrics for in-sample period
    strategy_returns_is = result_is['Strategy_Return']
    benchmark_returns_is = result_is[benchmark_return_col]

    # Sharpe Ratio for in-sample
    sharpe_is = calculate_rolling_sharpe(strategy_returns_is, window=min(252, len(strategy_returns_is)), annualization_factor=252).iloc[-1]
    benchmark_sharpe_is = calculate_rolling_sharpe(benchmark_returns_is, window=min(252, len(benchmark_returns_is)), annualization_factor=252).iloc[-1]

    # Maximum Drawdown for in-sample
    max_drawdown_is = calculate_rolling_drawdown(strategy_returns_is, window=len(strategy_returns_is)).iloc[-1] * 100
    benchmark_drawdown_is = calculate_rolling_drawdown(benchmark_returns_is, window=len(benchmark_returns_is)).iloc[-1] * 100

    # Win Rate for in-sample
    win_rate_is = calculate_win_rate(strategy_returns_is)
    benchmark_win_rate_is = calculate_win_rate(benchmark_returns_is)

    # Calmar Ratio for in-sample
    calmar_is = calculate_calmar_ratio(strategy_returns_is)
    benchmark_calmar_is = calculate_calmar_ratio(benchmark_returns_is)

    # Display metrics with comparison to benchmark for in-sample
    with col1:
        st.metric("Sharpe Ratio", f"{sharpe_is:.2f}", f"{sharpe_is - benchmark_sharpe_is:.2f} vs Benchmark")
    with col2:
        # For drawdown, lower is better, so we invert the delta
        st.metric("Maximum Drawdown", f"{max_drawdown_is:.2f}%", f"{benchmark_drawdown_is - max_drawdown_is:.2f}% vs Benchmark")
    with col3:
        st.metric("Win Rate", f"{win_rate_is:.2f}%", f"{win_rate_is - benchmark_win_rate_is:.2f}% vs Benchmark")
    with col4:
        st.metric("Calmar Ratio", f"{calmar_is:.2f}", f"{calmar_is - benchmark_calmar_is:.2f} vs Benchmark")

    # Plot in-sample volatility
    fig_vol_is = plt.figure(figsize=(10, 6))
    plt.plot(df_is.index, df_is[volatility_col], label=f'{volatility_source} Rolling Volatility', color='blue')
    plt.axhline(y=high_vol_threshold, color='red', linestyle='--', label=f'High Vol Threshold ({high_vol_threshold})')
    plt.axhline(y=low_vol_threshold, color='green', linestyle='--', label=f'Low Vol Threshold ({low_vol_threshold})')
    
    # Add cash position markers
    cash_days_is = result_is[result_is['Position'] == 'Cash'].index
    if len(cash_days_is) > 0:
        for cash_day in cash_days_is:
            plt.axvline(x=pd.Timestamp(cash_day), color='orange', linestyle=':', label='In Cash' if cash_day == cash_days_is[0] else "")
    
    plt.title('In-Sample Volatility and Strategy Positioning')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Annualized)')
    plt.legend(frameon=False)
    st.pyplot(fig_vol_is)
    
    # Plot in-sample cumulative returns
    fig_is = go.Figure()
    fig_is.add_trace(go.Scatter(x=result_is.index, y=result_is['Cumulative_Strategy_Return']*100, 
                            name='Strategy', mode='lines'))
    fig_is.add_trace(go.Scatter(x=result_is.index, y=result_is['Cumulative_Benchmark_Return']*100, 
                            name='Benchmark', mode='lines'))
    fig_is.update_layout(title='In-Sample Cumulative Returns Comparison',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return (%)',
                    yaxis_tickformat=',')
    st.plotly_chart(fig_is, use_container_width=True)
    
    # Display out-of-sample results
    st.subheader("Out-of-Sample Results (Testing Period)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Strategy Return", f"{result_oos['Cumulative_Strategy_Return'].iloc[-1]*100:.2f}%")
    with col2:
        st.metric("Benchmark Return", f"{result_oos['Cumulative_Benchmark_Return'].iloc[-1]*100:.2f}%")
    with col3:
        outperformance = (result_oos['Cumulative_Strategy_Return'].iloc[-1] - result_oos['Cumulative_Benchmark_Return'].iloc[-1])*100
        st.metric("Outperformance", f"{outperformance:.2f}%")
    
    # Plot out-of-sample volatility
    fig_vol_oos = plt.figure(figsize=(10, 6))
    plt.plot(df_oos.index, df_oos[volatility_col], label=f'{volatility_source} Rolling Volatility', color='blue')
    plt.axhline(y=high_vol_threshold, color='red', linestyle='--', label=f'High Vol Threshold ({high_vol_threshold})')
    plt.axhline(y=low_vol_threshold, color='green', linestyle='--', label=f'Low Vol Threshold ({low_vol_threshold})')
    
    # Add cash position markers
    cash_days_oos = result_oos[result_oos['Position'] == 'Cash'].index
    if len(cash_days_oos) > 0:
        for cash_day in cash_days_oos:
            plt.axvline(x=pd.Timestamp(cash_day), color='orange', linestyle=':', label='In Cash' if cash_day == cash_days_oos[0] else "")
    
    plt.title('Out-of-Sample Volatility and Strategy Positioning')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Annualized)')
    plt.legend(frameon=False)
    st.pyplot(fig_vol_oos)
    
    # Plot out-of-sample cumulative returns
    fig_oos = go.Figure()
    fig_oos.add_trace(go.Scatter(x=result_oos.index, y=result_oos['Cumulative_Strategy_Return']*100, 
                            name='Strategy', mode='lines'))
    fig_oos.add_trace(go.Scatter(x=result_oos.index, y=result_oos['Cumulative_Benchmark_Return']*100, 
                            name='Benchmark', mode='lines'))
    fig_oos.update_layout(title='Out-of-Sample Cumulative Returns Comparison',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return (%)',
                    yaxis_tickformat=',')
    st.plotly_chart(fig_oos, use_container_width=True)
    
    # Plot combined results for comparison
    fig_combined = go.Figure()
    # In-sample
    fig_combined.add_trace(go.Scatter(x=result_is.index, y=result_is['Cumulative_Strategy_Return']*100, 
                                name='In-Sample Strategy', mode='lines', line=dict(color='blue')))
    fig_combined.add_trace(go.Scatter(x=result_is.index, y=result_is['Cumulative_Benchmark_Return']*100, 
                                name='In-Sample Benchmark', mode='lines', line=dict(color='red')))
    # Out-of-sample
    fig_combined.add_trace(go.Scatter(x=result_oos.index, y=result_oos['Cumulative_Strategy_Return']*100, 
                                name='Out-of-Sample Strategy', mode='lines', line=dict(color='blue', dash='dash')))
    fig_combined.add_trace(go.Scatter(x=result_oos.index, y=result_oos['Cumulative_Benchmark_Return']*100, 
                                name='Out-of-Sample Benchmark', mode='lines', line=dict(color='red', dash='dash')))
    # Add a vertical line at the out-of-sample date
    oos_date_str = pd.Timestamp(oos_date).strftime('%Y-%m-%d')
    fig_combined.add_shape(
        type="line",
        x0=oos_date_str,
        y0=0,
        x1=oos_date_str,
        y1=1,
        line=dict(color="black", width=1, dash="dash"),
        xref="x",
        yref="paper"
    )
    fig_combined.add_annotation(
        x=oos_date_str,
        y=1,
        text="Out-of-Sample Start",
        showarrow=False,
        yref="paper"
    )
    fig_combined.update_layout(title='Combined Cumulative Returns Comparison',
                        xaxis_title='Date',
                        yaxis_title='Cumulative Return (%)',
                        yaxis_tickformat=',')
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Strategy Return", f"{result_oos['Cumulative_Strategy_Return'].iloc[-1]*100:.2f}%")
    with col2:
        st.metric("Benchmark Return", f"{result_oos['Cumulative_Benchmark_Return'].iloc[-1]*100:.2f}%")
    with col3:
        outperformance = (result_oos['Cumulative_Strategy_Return'].iloc[-1] - result_oos['Cumulative_Benchmark_Return'].iloc[-1])*100
        st.metric("Outperformance", f"{outperformance:.2f}%")

    # Add performance metrics for out-of-sample period
    st.subheader("Out-of-Sample Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    # Calculate metrics for out-of-sample period
    strategy_returns = result_oos['Strategy_Return']
    benchmark_returns = result_oos[benchmark_return_col]

    # Sharpe Ratio
    sharpe = calculate_rolling_sharpe(strategy_returns, window=min(252, len(strategy_returns)), annualization_factor=252).iloc[-1]
    benchmark_sharpe = calculate_rolling_sharpe(benchmark_returns, window=min(252, len(benchmark_returns)), annualization_factor=252).iloc[-1]

    # Maximum Drawdown
    max_drawdown = calculate_rolling_drawdown(strategy_returns, window=len(strategy_returns)).iloc[-1] * 100
    benchmark_drawdown = calculate_rolling_drawdown(benchmark_returns, window=len(benchmark_returns)).iloc[-1] * 100

    # Win Rate
    win_rate = calculate_win_rate(strategy_returns)
    benchmark_win_rate = calculate_win_rate(benchmark_returns)

    # Calmar Ratio
    calmar = calculate_calmar_ratio(strategy_returns)
    benchmark_calmar = calculate_calmar_ratio(benchmark_returns)

    # Display metrics with comparison to benchmark
    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", f"{sharpe - benchmark_sharpe:.2f} vs Benchmark")
    with col2:
        # For drawdown, lower is better, so we invert the delta
        st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%", f"{benchmark_drawdown - max_drawdown:.2f}% vs Benchmark")
    with col3:
        st.metric("Win Rate", f"{win_rate:.2f}%", f"{win_rate - benchmark_win_rate:.2f}% vs Benchmark")
    with col4:
        st.metric("Calmar Ratio", f"{calmar:.2f}", f"{calmar - benchmark_calmar:.2f} vs Benchmark")

    return result_is, result_oos

# Streamlit UI
st.title("Investment Strategy Analysis")

# Add Strategy Overview and Parameters Guide
with st.expander("Strategy Overview and Parameters Guide", expanded=False):
    st.write("""
    ### Strategy Overview
    This application implements a strategy for switching between ETF pairs based on relative z-scores and volatility thresholds. The strategy consists of three key elements:
    
    1. **Z-Score Switching**: The strategy switches between two ETFs (e.g., SPY and RSP) based on their relative performance z-scores. 
       - When the z-score exceeds the threshold in one direction, it allocates to one ETF.
       - When it exceeds the threshold in the opposite direction, it allocates to the other ETF.
       
    2. **Volatility Filter**: A volatility-based filter is used to move to cash during high volatility periods.
       - When volatility exceeds the high threshold, the strategy moves to cash.
       - It stays in cash until volatility drops below the low threshold.
       
    3. **Parameter Optimization**: The application can find optimal z-score and volatility threshold parameters based on in-sample data.
    
    ### Key Parameters
    
    - **Z-Score Threshold**: Controls how extreme the relative performance must be to trigger a switch between ETFs. Higher values mean less frequent switches.
    
    - **High Volatility Threshold**: When volatility exceeds this level, the strategy moves to cash. Higher values mean staying invested during more volatile periods.
    
    - **Low Volatility Threshold**: Once in cash, the strategy will remain there until volatility drops below this level. This creates a "buffer zone" to prevent frequent entries and exits.
    
    - **Initial Position**: The starting position (SPY or RSP) when the z-score doesn't exceed thresholds initially.
    
    - **Volatility Source**: Which ETF's volatility is used for cash timing decisions.
    
    ### Analysis Process
    
    1. **Data Loading**: Load historical price data for the selected ETFs.
    
    2. **Exploratory Analysis**: View cumulative returns, rolling returns, correlations, and z-scores.
    
    3. **Backtesting**: Run the strategy with selected parameters and view performance metrics including returns, drawdowns, Sharpe ratio, and win rate.
    
    4. **Parameter Optimization**: Find optimal parameter combinations based on Sharpe ratio maximization on in-sample data.
    
    5. **In-Sample vs. Out-of-Sample**: Validate the strategy by comparing in-sample and out-of-sample performance.
    """)

# Sidebar controls
st.sidebar.header("Parameters")
start_date = st.sidebar.date_input("Start Date", datetime(1997, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 11, 30))
oos_date = st.sidebar.date_input("Out-of-Sample Date", datetime(2020, 1, 1))
tickers = st.sidebar.multiselect("Select Tickers", 
    ["SPY", "RSP", "IWDA.AS", "EEM", "AGG", "GLD", "QQQ", "IWM", "EFA", "TLT"], 
    default=["SPY", "RSP"]
)

# Add dropdown for volatility source selection
volatility_source = None
if len(tickers) >= 2:
    volatility_source = st.sidebar.selectbox(
        "Volatility Source for Cash Decisions", 
        tickers[:2],  # Only show the first two tickers
        index=0,      # Default to first ticker
        help="Select which ETF's volatility will determine when to move to cash"
    )

# Add checkbox for separate in-sample/out-of-sample backtesting
separate_oos_backtest = st.sidebar.checkbox("Separate In-Sample/Out-of-Sample Backtesting", value=False)

# Convert dates to string format
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
oos_date_str = oos_date.strftime('%Y-%m-%d')

# Retrieve and process data
if tickers:
    df_prices = retrieve_prices_yf(tickers, start_date_str, end_date_str).sort_index().dropna()
    df_prices.index = pd.to_datetime(df_prices.index)

    # Display cumulative returns
    st.subheader("Cumulative Returns")
    fig_cumulative = cumulative_return_graph_plotly(df_prices.pct_change().dropna())
    st.plotly_chart(fig_cumulative, use_container_width=True)

    # Calculate and display rolling returns
    st.subheader("Rolling Returns Analysis")
    rolling_period = st.sidebar.slider("Rolling Period (days)", 21, 756, 252)
    df_rolling_returns = rolling_returns(df_prices, rolling_period, 'D', True)
    
    fig_rolling = plot_rolling_returns(df_rolling_returns, title=f'{rolling_period}-Day Rolling Returns')
    st.pyplot(fig_rolling)

    # Display rolling return statistics
    if len(tickers) > 1:
        st.subheader("Rolling Return Statistics")
        benchmark = st.sidebar.selectbox("Select Benchmark", tickers)
        df_rr_stats = rolling_return_stats(df_rolling_returns.mul(100).round(2), benchmark)
        st.dataframe(df_rr_stats)

    # Calculate and display rolling correlations
    if len(tickers) > 1:
        st.subheader("Rolling Correlations")
        df_monthly = df_prices.resample('M').last()
        df_monthly_returns = df_monthly.pct_change().dropna()
        
        correlation_period = st.sidebar.slider("Correlation Period (months)", 12, 36, 12)
        df_monthly_returns[f'{correlation_period}M_Rolling_Correlation'] = df_monthly_returns[tickers[0]].rolling(correlation_period).corr(df_monthly_returns[tickers[1]])
        
        fig_corr = plt.figure(figsize=(12, 6))
        df_monthly_returns[f'{correlation_period}M_Rolling_Correlation'].plot(title=f'{correlation_period}-Month Rolling Correlation')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.legend(frameon=False)
        st.pyplot(fig_corr)

    # Calculate and display z-scores
    st.subheader("Rolling Return Z-Scores")
    periods = {
        "1 Month": 21,
        "6 Months": 126,
        "1 Year": 252
    }
    
    selected_period = st.sidebar.selectbox("Select Period for Z-Scores", list(periods.keys()))
    period_days = periods[selected_period]
    
    # Calculate rolling returns for selected period for each ticker
    rolling_returns_df = pd.DataFrame()
    for ticker in tickers:
        rolling_returns_df[f'{ticker}_Rolling_{selected_period}' ] = rolling_returns(df_prices[ticker], period_days, annualize=False)
    
    # Calculate z-scores for each ticker
    z_scores_df = pd.DataFrame()
    for ticker in tickers:
        col_name = f'{ticker}_Rolling_{selected_period}'
        z_scores_df[f'{ticker}_Z'] = expanding_zscore(rolling_returns_df[col_name])
        z_scores_df[f'{ticker}_Z'] = z_scores_df[f'{ticker}_Z'][rolling_returns_df[col_name].notna()]
    
    # Plot z-scores
    fig_zscore = plt.figure(figsize=(12, 6))
    for ticker in tickers:
        sns.lineplot(data=z_scores_df[f'{ticker}_Z'], label=f'{ticker} {selected_period} Z-Score')
    plt.title(f'{selected_period} Rolling Return Z-Scores')
    plt.xlabel('Date')
    plt.ylabel('Z-Score')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    plt.legend(frameon=False)
    st.pyplot(fig_zscore)

    # Calculate Z-score differences between ETF pairs
    if len(tickers) >= 2:
        st.subheader("ETF Pair Z-Score Differences")
        
        # Map the selected period to the corresponding period mapping key
        period_mapping = {
            "1 Month": "1M",
            "6 Months": "6M", 
            "1 Year": "1Y"
        }
        selected_period_key = period_mapping[selected_period]
        period_days = periods[selected_period]
        
        # Calculate rolling return differences and Z-scores for the selected period
        # Calculate rolling returns for each ticker
        for ticker in tickers[:2]:  # Using only the first two tickers
            df_prices[f'{ticker}_Rolling_{selected_period_key}'] = rolling_returns(df_prices[ticker], period_days, annualize=False)
        
        # Calculate the difference between the rolling returns
        diff_col = f'Rolling_{selected_period_key}_Diff'
        df_prices[diff_col] = df_prices[f'{tickers[0]}_Rolling_{selected_period_key}'] - df_prices[f'{tickers[1]}_Rolling_{selected_period_key}']
        
        # Calculate expanding Z-score of the difference
        df_prices[f'{diff_col}_Z'] = expanding_zscore(df_prices[diff_col])
        
        # Set up the figure and axes
        fig = plt.figure(figsize=(10, 6))
        
        # Plot the selected period's Rolling Return Z-Scores
        sns.lineplot(data=df_prices, x=df_prices.index, y=f'{diff_col}_Z',
                    label=f'{selected_period} Rolling Return Difference Z-Score')
        
        # Add current Z-score threshold lines
        current_z_threshold = st.session_state.get('z_threshold', 1.5)
        plt.axhline(y=current_z_threshold, color='green', linestyle='--', 
                   label=f'Current Threshold: +{current_z_threshold}')
        plt.axhline(y=-current_z_threshold, color='green', linestyle='--',
                   label=f'Current Threshold: -{current_z_threshold}')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        
        # Add vertical line for out-of-sample split if separate backtesting is enabled
        if separate_oos_backtest:
            plt.axvline(x=oos_date, color='black', linestyle='--', 
                      label='Out-of-Sample Start', alpha=0.7)
        
        # Set labels
        plt.title(f'{selected_period} Rolling Return Differences (Z-Scores): {tickers[0]} vs {tickers[1]}')
        plt.xlabel('Date')
        plt.ylabel('Z-Score')
        plt.legend(frameon=False)
        
        # Show the plot
        st.pyplot(fig)

    # Calculate the daily returns
    df_prices['SPY_Return'] = df_prices['SPY'].pct_change().fillna(0)
    df_prices['RSP_Return'] = df_prices['RSP'].pct_change().fillna(0)
    
    # Calculate rolling returns for different periods
    df_prices['SPY_Rolling_1M'] = rolling_returns(df_prices['SPY'], period=21, annualize=False)
    df_prices['RSP_Rolling_1M'] = rolling_returns(df_prices['RSP'], period=21, annualize=False)
    df_prices['SPY_Rolling_6M'] = rolling_returns(df_prices['SPY'], period=126, annualize=False)
    df_prices['RSP_Rolling_6M'] = rolling_returns(df_prices['RSP'], period=126, annualize=False)
    df_prices['SPY_Rolling_1Y'] = rolling_returns(df_prices['SPY'], period=252, annualize=False)
    df_prices['RSP_Rolling_1Y'] = rolling_returns(df_prices['RSP'], period=252, annualize=False)

    # Calculate rolling return differences
    df_prices['Rolling_1M_Diff'] = df_prices['SPY_Rolling_1M'] - df_prices['RSP_Rolling_1M']
    df_prices['Rolling_6M_Diff'] = df_prices['SPY_Rolling_6M'] - df_prices['RSP_Rolling_6M']
    df_prices['Rolling_1Y_Diff'] = df_prices['SPY_Rolling_1Y'] - df_prices['RSP_Rolling_1Y']

    # Calculate z-scores of the differences
    df_prices['Rolling_1M_Diff_Z'] = expanding_zscore(df_prices['Rolling_1M_Diff'])
    df_prices['Rolling_6M_Diff_Z'] = expanding_zscore(df_prices['Rolling_6M_Diff'])
    df_prices['Rolling_1Y_Diff_Z'] = expanding_zscore(df_prices['Rolling_1Y_Diff'])

    # Ensure z-scores only start from the first valid window of data
    df_prices['Rolling_1M_Diff_Z'] = df_prices['Rolling_1M_Diff_Z'][df_prices['Rolling_1M_Diff'].notna()]
    df_prices['Rolling_6M_Diff_Z'] = df_prices['Rolling_6M_Diff_Z'][df_prices['Rolling_6M_Diff'].notna()]
    df_prices['Rolling_1Y_Diff_Z'] = df_prices['Rolling_1Y_Diff_Z'][df_prices['Rolling_1Y_Diff'].notna()]
    
    # Recreate the training and test set data
    df_is_prices = df_prices.loc[:oos_date]
    df_is_prices.tail()

    # Add backtesting section to the UI
    if len(tickers) == 2 and 'SPY' in tickers and 'RSP' in tickers:
        st.subheader("Strategy Backtesting")
        
        # Add controls for backtesting parameters with session state
        col1, col2 = st.columns(2)
        with col1:
            z_threshold = st.number_input("Z-Score Threshold", value=st.session_state.get('z_threshold', 1.5), step=0.1, key="z_threshold_input")
            high_vol_threshold = st.number_input("High Volatility Threshold", value=st.session_state.get('high_vol_threshold', 0.32), step=0.01, key="high_vol_threshold_input")
        with col2:
            initial_position = st.selectbox("Initial Position", ["SPY", "RSP"])
            low_vol_threshold = st.number_input("Low Volatility Threshold", value=st.session_state.get('low_vol_threshold', 0.30), step=0.01, key="low_vol_threshold_input")
        
        # Update session state from UI
        st.session_state['z_threshold'] = z_threshold
        st.session_state['high_vol_threshold'] = high_vol_threshold
        st.session_state['low_vol_threshold'] = low_vol_threshold
        
        # Add rolling volatility calculation
        df_prices['SPY_Rolling_Volatility'] = df_prices['SPY_Return'].rolling(window=20).std().mul(np.sqrt(252))
        
        # Run backtest
        if st.button("Run Backtest"):
            if separate_oos_backtest:
                # Use separate in-sample/out-of-sample backtesting
                result_is, result_oos = run_separate_backtest(
                    df_prices, 
                    oos_date, 
                    z_threshold, 
                    high_vol_threshold, 
                    low_vol_threshold, 
                    initial_position, 
                    volatility_source if volatility_source else 'SPY'
                )
            else:
                # Use standard backtesting on entire data
                result = backtest_zscore_strategy_with_volatility(
                    df_prices,
                    f'Rolling_1M_Diff_Z',
                    threshold=z_threshold,
                    spy_return_col='SPY_Return',
                    rsp_return_col='RSP_Return',
                    volatility_col='SPY_Rolling_Volatility',
                    high_vol_threshold=high_vol_threshold,
                    low_vol_threshold=low_vol_threshold,
                    plot=True,
                    initial_pos=initial_position
                )
                
                # Define benchmark return column for performance metrics
                benchmark_return_col = 'SPY_Return'  # Default benchmark

                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Strategy Return", f"{result['Cumulative_Strategy_Return'].iloc[-1]*100:.2f}%")
                with col2:
                    st.metric("Benchmark Return", f"{result['Cumulative_Benchmark_Return'].iloc[-1]*100:.2f}%")
                with col3:
                    outperformance = (result['Cumulative_Strategy_Return'].iloc[-1] - result['Cumulative_Benchmark_Return'].iloc[-1])*100
                    st.metric("Outperformance", f"{outperformance:.2f}%")
                
                # Add performance metrics
                st.subheader("Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)

                # Calculate metrics
                strategy_returns = result['Strategy_Return']
                benchmark_returns = result[benchmark_return_col]

                # Sharpe Ratio
                sharpe = calculate_rolling_sharpe(strategy_returns, window=min(252, len(strategy_returns)), annualization_factor=252).iloc[-1]
                benchmark_sharpe = calculate_rolling_sharpe(benchmark_returns, window=min(252, len(benchmark_returns)), annualization_factor=252).iloc[-1]

                # Maximum Drawdown
                max_drawdown = calculate_rolling_drawdown(strategy_returns, window=len(strategy_returns)).iloc[-1] * 100
                benchmark_drawdown = calculate_rolling_drawdown(benchmark_returns, window=len(benchmark_returns)).iloc[-1] * 100

                # Win Rate
                win_rate = calculate_win_rate(strategy_returns)
                benchmark_win_rate = calculate_win_rate(benchmark_returns)

                # Calmar Ratio
                calmar = calculate_calmar_ratio(strategy_returns)
                benchmark_calmar = calculate_calmar_ratio(benchmark_returns)

                # Display metrics with comparison to benchmark
                with col1:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}", f"{sharpe - benchmark_sharpe:.2f} vs Benchmark")
                with col2:
                    # For drawdown, lower is better, so we invert the delta
                    st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%", f"{benchmark_drawdown - max_drawdown:.2f}% vs Benchmark")
                with col3:
                    st.metric("Win Rate", f"{win_rate:.2f}%", f"{win_rate - benchmark_win_rate:.2f}% vs Benchmark")
                with col4:
                    st.metric("Calmar Ratio", f"{calmar:.2f}", f"{calmar - benchmark_calmar:.2f} vs Benchmark")
                
                # Plot cumulative returns
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result.index, y=result['Cumulative_Strategy_Return']*100, 
                                       name='Strategy', mode='lines'))
                fig.add_trace(go.Scatter(x=result.index, y=result['Cumulative_Benchmark_Return']*100, 
                                       name='Benchmark', mode='lines'))
                
                # Add a vertical line at the out-of-sample date to show the split
                oos_date_str = pd.Timestamp(oos_date).strftime('%Y-%m-%d')
                fig.add_shape(
                    type="line",
                    x0=oos_date_str,
                    y0=0,
                    x1=oos_date_str,
                    y1=1,
                    line=dict(color="black", width=1, dash="dash"),
                    xref="x",
                    yref="paper"
                )
                fig.add_annotation(
                    x=oos_date_str,
                    y=1,
                    text="Out-of-Sample Start",
                    showarrow=False,
                    yref="paper"
                )
                
                fig.update_layout(title='Cumulative Returns Comparison',
                                xaxis_title='Date',
                                yaxis_title='Cumulative Return (%)',
                                yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot position changes
                fig_pos = plt.figure(figsize=(12, 6))
                result_reset = result.reset_index()
                result_reset = result_reset.rename(columns={'index': 'Date'})
                sns.scatterplot(data=result_reset, x='Date', y='Cumulative_Strategy_Return', 
                              hue='Position', palette='Set1', s=50)
                plt.title('Strategy Positions Over Time')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Strategy Return')
                plt.axvline(x=oos_date, color='black', linestyle='--', alpha=0.7, label='Out-of-Sample Start')
                plt.legend(frameon=False)
                plt.xticks(rotation=45)
                st.pyplot(fig_pos)

    # Add Parameters Optimizer section
    st.subheader("Parameters Optimizer")
    st.write("Find optimal Z-Score, High Volatility, and Low Volatility threshold parameters based on in-sample data.")
    
    # Create containers to store results
    optimization_container = st.container()

    # Add a button to start optimization
    if st.button("Run Parameter Optimization"):
        with st.spinner("Optimizing strategy parameters... This may take several minutes."):
            # Ensure we have the volatility source
            volatility_source = 'SPY'  # Default to first ticker
            
            # Run optimization on in-sample data only
            df_is = df_prices.loc[:oos_date].copy()
            optimization_results = optimize_strategy_parameters(
                df_is, 
                first_ticker='SPY',
                second_ticker='RSP',
                volatility_source=volatility_source
            )
            
            # Display optimization results
            with optimization_container:
                if not optimization_results.empty:
                    st.subheader("Top 10 Parameter Combinations")
                    st.dataframe(optimization_results.head(10))
                    
                    # Store best parameters for easy access
                    best_params = optimization_results.iloc[0]
                    
                    # Parameter sensitivity analysis
                    st.subheader("Parameter Sensitivity Analysis")
                    
                    # Create a top 30 subset for analysis
                    top_30 = optimization_results.head(30)
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3, tab4 = st.tabs(["Threshold Levels", "Sharpe Ratio", "Maximum Drawdown", "Calmar Ratio"])
                    
                    with tab1:
                        # Create a figure with multiple subplots for threshold levels
                        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                        
                        # Plot distribution of Z-Score thresholds
                        sns.boxplot(y='Z_Threshold', data=top_30, ax=axs[0])
                        axs[0].set_title('Z-Score Threshold Distribution')
                        axs[0].axhline(y=best_params['Z_Threshold'], color='red', linestyle='--', 
                                     label=f'Best: {best_params["Z_Threshold"]:.2f}')
                        axs[0].legend()
                        
                        # Plot distribution of High Volatility thresholds
                        sns.boxplot(y='High_Vol_Threshold', data=top_30, ax=axs[1])
                        axs[1].set_title('High Volatility Threshold Distribution')
                        axs[1].axhline(y=best_params['High_Vol_Threshold'], color='red', linestyle='--',
                                     label=f'Best: {best_params["High_Vol_Threshold"]:.2f}')
                        axs[1].legend()
                        
                        # Plot distribution of Low Volatility thresholds
                        sns.boxplot(y='Low_Vol_Threshold', data=top_30, ax=axs[2])
                        axs[2].set_title('Low Volatility Threshold Distribution')
                        axs[2].axhline(y=best_params['Low_Vol_Threshold'], color='red', linestyle='--',
                                     label=f'Best: {best_params["Low_Vol_Threshold"]:.2f}')
                        axs[2].legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with tab2:
                        # Sharpe ratio analysis
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
                        
                        # Scatterplot of Z-Score vs Sharpe Ratio
                        sns.scatterplot(x='Z_Threshold', y='Sharpe_Ratio', 
                                      data=top_30, 
                                      size='Total_Return', 
                                      hue='High_Vol_Threshold',
                                      palette='viridis',
                                      ax=ax1)
                        ax1.set_title('Z-Score Threshold vs Sharpe Ratio')
                        ax1.grid(True)
                        
                        # Heatmap of High Vol vs Low Vol Threshold (color = Sharpe)
                        pivot_sharpe = pd.pivot_table(
                            top_30, 
                            values='Sharpe_Ratio', 
                            index=pd.cut(top_30['High_Vol_Threshold'], 5), 
                            columns=pd.cut(top_30['Low_Vol_Threshold'], 5),
                            aggfunc='mean'
                        )
                        sns.heatmap(pivot_sharpe, annot=True, cmap='viridis', fmt='.2f', ax=ax2)
                        ax2.set_title('Volatility Thresholds vs Sharpe Ratio')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with tab3:
                        # Maximum Drawdown analysis
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
                        
                        # Scatterplot of Z-Score vs Max Drawdown
                        sns.scatterplot(x='Z_Threshold', y='Max_Drawdown', 
                                      data=top_30, 
                                      size='Total_Return', 
                                      hue='High_Vol_Threshold',
                                      palette='coolwarm',
                                      ax=ax1)
                        ax1.set_title('Z-Score Threshold vs Maximum Drawdown')
                        ax1.grid(True)
                        
                        # Relationship between volatility thresholds and drawdown
                        sns.scatterplot(x='High_Vol_Threshold', y='Low_Vol_Threshold', 
                                      size='Max_Drawdown', 
                                      hue='Max_Drawdown',
                                      palette='coolwarm_r',  # Reversed to make lower drawdowns better
                                      data=top_30,
                                      ax=ax2)
                        ax2.set_title('Volatility Thresholds vs Maximum Drawdown')
                        ax2.grid(True)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with tab4:
                        # Calmar Ratio analysis
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
                        
                        # Parallel coordinates plot for key metrics
                        pd.plotting.parallel_coordinates(
                            top_30, 'Z_Threshold', 
                            cols=['Calmar_Ratio', 'Total_Return', 'Max_Drawdown'],
                            colormap=plt.cm.viridis,
                            ax=ax1
                        )
                        ax1.set_title('Parameter Impact on Key Metrics')
                        ax1.grid(True)
                        
                        # 2D scatter plot with Calmar ratio as color
                        scatter = ax2.scatter(
                            x=top_30['Z_Threshold'],
                            y=top_30['High_Vol_Threshold'],
                            c=top_30['Calmar_Ratio'],
                            cmap='viridis',
                            s=top_30['Calmar_Ratio'] * 50,  # Size based on Calmar ratio
                            alpha=0.7
                        )
                        ax2.set_xlabel('Z-Score Threshold')
                        ax2.set_ylabel('High Volatility Threshold')
                        ax2.set_title('Parameter Impact on Calmar Ratio')
                        ax2.grid(True)
                        fig.colorbar(scatter, ax=ax2, label='Calmar Ratio')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Best parameters section
                    st.subheader("Best Parameters")
                    
                    best_params_col1, best_params_col2 = st.columns(2)
                    
                    with best_params_col1:
                        st.metric("Best Z-Score Threshold", f"{best_params['Z_Threshold']:.2f}")
                        st.metric("Best High Volatility Threshold", f"{best_params['High_Vol_Threshold']:.2f}")
                        st.metric("Best Low Volatility Threshold", f"{best_params['Low_Vol_Threshold']:.2f}")
                    
                    with best_params_col2:
                        st.metric("Expected Sharpe Ratio", f"{best_params['Sharpe_Ratio']:.2f}")
                        st.metric("Expected Total Return", f"{best_params['Total_Return']:.2f}%")
                        st.metric("Expected Maximum Drawdown", f"{best_params['Max_Drawdown']:.2f}%")
                    
                    # Button to apply the best parameters
                    if st.button("Apply Best Parameters"):
                        # Update session state with best parameters
                        st.session_state['z_threshold'] = float(best_params['Z_Threshold'])
                        st.session_state['high_vol_threshold'] = float(best_params['High_Vol_Threshold'])
                        st.session_state['low_vol_threshold'] = float(best_params['Low_Vol_Threshold'])
                        st.success(f"Best parameters applied! Z-Score: {st.session_state['z_threshold']:.2f}, " + 
                                 f"High Vol: {st.session_state['high_vol_threshold']:.2f}, " + 
                                 f"Low Vol: {st.session_state['low_vol_threshold']:.2f}")
                        # Force a rerun to update the UI
                        st.experimental_rerun()
                else:
                    st.error("Optimization failed or did not find any valid parameter combinations.")

else:
    st.warning("Please select at least one ticker to begin analysis.") 