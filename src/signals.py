"""
=============================================================================================
Title: signals.py
Author: Kenneth LeGare
Date: 2023-10-01

Description:
    Implements microstructure features (e.g., order book imbalance, ETF–constituent dislocations).

Dependencies:
    - pandas
    - numpy
    - scipy
=============================================================================================
"""

# Third-party libraries
import pandas as pd
import numpy as np
from scipy.stats import zscore

def order_book_imbalance(bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> np.ndarray:
    """
    Calculates the order book imbalance given bid and ask sizes.

    args:
        bid_sizes (np.ndarray): Array of bid sizes.
        ask_sizes (np.ndarray): Array of ask sizes.

    Returns:
        np.ndarray: Array of order book imbalance values.
    """
    imbalance = (bid_sizes - ask_sizes) / (bid_sizes + ask_sizes)
    return imbalance

def zscore_normalize(series: pd.Series) -> pd.Series:
    """
    Applies z-score normalization to a pandas Series.

    args:
        series (pd.Series): The input Series to be normalized.

    Returns:
        pd.Series: The z-score normalized Series.
    """
    return pd.Series(zscore(series), index=series.index)


def etf_constituent_dislocation(etf_price: pd.Series, constituent_prices: pd.DataFrame) -> pd.Series:
    """
    Calculates the ETF-constituent dislocation.

    args:
        etf_price (pd.Series): Series of ETF prices.
        constituent_prices (pd.DataFrame): DataFrame of constituent prices with the same index as etf_price.

    Returns:
        pd.Series: Series of dislocation values.
    """
    etf_value = constituent_prices.sum(axis=1)
    dislocation = (etf_price - etf_value) / etf_value
    return dislocation

def main():
    # Example usage
    bid_sizes = np.array([100, 150, 200])
    ask_sizes = np.array([80, 120, 160])
    imbalance = order_book_imbalance(bid_sizes, ask_sizes)
    print("Order Book Imbalance:", imbalance)

    series = pd.Series([1, 2, 3, 4, 5])
    normalized_series = zscore_normalize(series)
    print("Z-score Normalized Series:\n", normalized_series)

    etf_price = pd.Series([100, 102, 101, 103])
    constituent_prices = pd.DataFrame({
        'StockA': [30, 31, 30.5, 32],
        'StockB': [40, 41, 40.5, 42],
        'StockC': [25, 26, 25.5, 26.5]
    })
    dislocation = etf_constituent_dislocation(etf_price, constituent_prices)
    print("ETF-Constituent Dislocation:\n", dislocation)

if __name__ == "__main__":
    main()