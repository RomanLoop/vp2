import numpy as np
import pandas as pd
from typing import Tuple

#https://www.codearmo.com/blog/sharpe-sortino-and-calmar-ratios-python

N_TRADING_DAYS = 252
RISK_FREE_RATE = 0.01


def annualized_sharpe_ratio(df_returns:pd.DataFrame, risk_free_rate:float, name:str) -> pd.DataFrame:
    """ Returns a dataframe with annualized sharpe ratios.
        Args:
            df (pd.DataFrame): Dataframe with datetime index and asset returns as columns
            risk_free_rate (float): Risk free rate according to sharpe ratio formula
        Returns:
            pd.DataFrame: Annualized Sharpe Ratios 
    """
    df_mean = df_returns.groupby(df_returns.index.year).mean()
    df_count = df_returns.groupby(df_returns.index.year).count()
    df_std = df_returns.groupby(df_returns.index.year).std()

    mean = df_mean * df_count - risk_free_rate
    sigma = df_std * df_count.apply(lambda x: np.sqrt(x))
    s = mean / sigma
    s.name = name
    return s


def annualized_sortino_ratio(df_returns:pd.DataFrame, risk_free_rate:float, name:str) -> pd.DataFrame:
    """ Returns a dataframe with annualized sortino ratios.
        Args:
            df (pd.DataFrame): Dataframe with datetime index and asset returns as columns
            risk_free_rate (float): Risk free rate according to sortion ratio formula
            name (str): name for the returned series
        Returns:
            pd.DataFrame: Annualized Sortino Ratios 
    """
    df_mean = df_returns.groupby(df_returns.index.year).mean()
    df_count = df_returns.groupby(df_returns.index.year).count()
    df_returns_neg = df_returns[df_returns < 0]
    df_returns_neg = df_returns_neg.groupby(df_returns_neg.index.year).std()

    mean = df_mean * df_count - risk_free_rate
    std_neg = df_returns_neg * df_count.apply(lambda x: np.sqrt(x))
    s = mean / std_neg
    s.name = name
    return s


def max_drawdown_series(df_returns:pd.DataFrame):
    """ Returns a dataframe with max drawdowns.
        Args:
            df (pd.DataFrame): Dataframe with datetime index and asset returns as columns
        Returns:
            pd.DataFrame: Maximal Drawdowns 
    """
    df_returns_cum = (df_returns + 1).cumprod() - 1
    df_peak = df_returns_cum.expanding(min_periods=1).max()
    df_drawdown = df_returns_cum - df_peak
    return df_drawdown


def annualized_max_drawdown(df_returns:pd.DataFrame, name:str) -> pd.DataFrame:
    """ Returns a dataframe with annualized max drawdowns.
        Args:
            df (pd.DataFrame): Dataframe with datetime index and asset returns as columns
            name (str): name for the returned series
        Returns:
            pd.DataFrame: Annualized Maximal Drawdowns 
    """
    df_returns_cum = (df_returns + 1).cumprod() - 1
    df_peak = df_returns_cum.expanding(min_periods=1).max()
    df_drawdown = df_returns_cum - df_peak
    s = df_drawdown.groupby(df_drawdown.index.year).min()
    s.name = name
    return s


def annualized_calmar_ratio(df_returns:pd.DataFrame, risk_free_rate:float, name:str) -> pd.DataFrame:
    """ Returns a dataframe with annualized calmar ratios.
        Args:
            df (pd.DataFrame): Dataframe with datetime index and asset returns as columns
            name (str): name for the returned series
        Returns:
            pd.DataFrame: Annualized Calmar Ratios 
    """
    df_mean = df_returns.groupby(df_returns.index.year).mean()
    df_count = df_returns.groupby(df_returns.index.year).count()
    
    mean = df_mean * df_count - risk_free_rate
    mdd = annualized_max_drawdown(df_returns, name='MDD')
    s = mean / (mdd * -1)
    s.name = name
    return s


def portfolio_return(df:pd.DataFrame, assets:list) -> Tuple[pd.Series, pd.Series]:
    """" Caclculates the equally weighted returns of a portfolio given.
        
        This function returns the daily portfolio returns and the its cumulated returns.
        `assets` must be list of values which are also in `df` as columns.

        Args:
            df (pd.DataFrame): Dataframe with datetime index and asset prices as columns
            assets (list): Assets wich are in the portfolio
        Returns:
            Tuple(pd.Series): daily returns, cumulated returns
    """
    df_returns = df.pct_change()
    df_returns.dropna(axis=0, inplace=True)


    returns = df_returns.loc[:,assets].mean(axis=1)

    return returns, (returns + 1).cumprod() -1


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