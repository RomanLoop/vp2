import numpy as np

#https://www.codearmo.com/blog/sharpe-sortino-and-calmar-ratios-python

N_TRADING_DAYS = 252
RISK_FREE_RATE = 0.01


def portfolio_return(df, assets):
    """" """
    df_returns = df.pct_change()
    df_returns.dropna(axis=0, inplace=True)

    assert df_returns.shape[0] + 1 == df.shape[0]
    assert df_returns.shape[1] == df.shape[1]

    returns = df_returns.loc[:,assets].mean(axis=1)

    return returns, returns.cumsum()


def drawdown(series):
    """ """
    roll_max = series.rolling(window=len(series), min_periods=1).max()
    return series / roll_max - 1.0


def drawdown_pct(return_series):
    rs_max = return_series.rolling(
        window=len(return_series), 
        min_periods=1
        ).max()
    return return_series - rs_max


def sharpe_ratio(return_series, N=N_TRADING_DAYS, rf=RISK_FREE_RATE):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma


def sortino_ratio(return_series, N=N_TRADING_DAYS, rf=RISK_FREE_RATE):
    mean = return_series.mean() * N -rf
    std_neg = return_series[return_series<0].std()*np.sqrt(N)
    return mean/std_neg


def max_drawdown(return_series):
    comp_ret = (return_series+1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()


def calmar_ratio(return_series, N=N_TRADING_DAYS, rf=RISK_FREE_RATE):
    mean = return_series.mean() * N -rf
    mdd = max_drawdown(return_series)
    return mean/mdd