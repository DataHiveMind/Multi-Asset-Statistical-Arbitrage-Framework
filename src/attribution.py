"""
===============================================================================================================================================
Title: attribution.py
Author: Kenneth LeGare
Date: 2024-06-15

Description:
    Breaks down PnL into alpha sources, factor exposures, and risk contributions. Produces attribution reports for institutional polish.

Dependencies:
    - pandas
    - numpy
    - statsmodels
    - matplotlib
    - seaborn
===============================================================================================================================================
"""

# Third-party libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

def calculate_factor_exposures(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates factor exposures using linear regression.

    args:
        returns (pd.DataFrame): DataFrame of asset returns.
        factors (pd.DataFrame): DataFrame of factor returns.

    Returns:
        pd.DataFrame: DataFrame of factor exposures.
    """
    exposures = {}
    for asset in returns.columns:
        y = returns[asset].dropna()
        X = factors.loc[y.index]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        exposures[asset] = model.params
    return pd.DataFrame(exposures).T

def attribute_pnl(returns: pd.DataFrame, factors: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    """
    Attributes PnL to factors based on exposures.

    args:
        returns (pd.DataFrame): DataFrame of asset returns.
        factors (pd.DataFrame): DataFrame of factor returns.
        exposures (pd.DataFrame): DataFrame of factor exposures.

    Returns:
        pd.DataFrame: DataFrame of attributed PnL.
    """
    attributed_pnl = pd.DataFrame(index=returns.index, columns=returns.columns)
    for asset in returns.columns:
        if asset in exposures.index:
            attributed_pnl[asset] = factors.dot(exposures.loc[asset].drop('const'))
    return attributed_pnl

def plot_attribution(attributed_pnl: pd.DataFrame, asset: str):
    """
    Plots the attribution of PnL for a given asset.

    args:
        attributed_pnl (pd.DataFrame): DataFrame of attributed PnL.
        asset (str): The asset to plot.
    """
    plt.figure(figsize=(10, 6))
    attributed_pnl[asset].cumsum().plot()
    plt.title(f'PnL Attribution for {asset}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Attributed PnL')
    plt.grid(True)
    plt.show()

def generate_attribution_report(attributed_pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary report of the attribution.

    args:
        attributed_pnl (pd.DataFrame): DataFrame of attributed PnL.

    Returns:
        pd.DataFrame: Summary report of total attributed PnL per factor.
    """
    report = attributed_pnl.sum().to_frame(name='Total Attributed PnL')
    return report

def main(returns: pd.DataFrame, factors: pd.DataFrame):
    """
    Main function to perform attribution analysis.

    args:
        returns (pd.DataFrame): DataFrame of asset returns.
        factors (pd.DataFrame): DataFrame of factor returns.
    """
    exposures = calculate_factor_exposures(returns, factors)
    attributed_pnl = attribute_pnl(returns, factors, exposures)
    
    for asset in returns.columns:
        plot_attribution(attributed_pnl, asset)
    
    report = generate_attribution_report(attributed_pnl)
    print(report)

if __name__ == "__main__":
    # Example usage with dummy data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    assets = ['Asset1', 'Asset2']
    factors = ['Factor1', 'Factor2']

    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(100, 2) / 100, index=dates, columns=assets)
    factor_returns = pd.DataFrame(np.random.randn(100, 2) / 100, index=dates, columns=factors)

    main(returns, factor_returns)