"""
===================================================================================================================================================
Title: Data Pipeline
Author: Kenneth LeGare
Date: 2023-10-01

Description: 
    This module implements a data pipeline for processing financial data. 
    Handles ingestion, cleaning, and transformation of tick/quote data. Connects to KDB+/q for efficient time-series management.

Dependencies:
    - pandas
    - numpy
    - yfinance
    - kdb
==================================================================================================================================================
"""
# Third-party libraries
import pandas as pd, numpy as np, yfinance as yf

def download_raw_data(ticker: str, start_date: str, end_date: str)-> pd.DataFrame:
    """
    Downloads raw historical data for a given ticker from Yahoo Finance.

    args:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date for the data download (YYYY-MM-DD).
        end_date (str): The end date for the data download (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame containing the raw historical data.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw data DataFrame, including cleaning.

    args:
        df (pd.DataFrame): The raw data DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the preprocessed data.
    """
    # Handle duplicates
    df = df[~df.index.duplicated(keep='first')]

    # Handle missing values
    df = df.ffill().bfill()

    # Handle outliers
    if 'Return' in df.columns:
        df = df[(np.abs(df['Return']) < 3 * df['Return'].std())]

    return df

def technical_indicators(df: pd.DataFrame, long: int, short: int) -> pd.DataFrame:
    """
    Computes technical indicators for the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing price data.
        long (int): The long window for moving averages.
        short (int): The short window for moving averages.

    Returns:
        pd.DataFrame: A DataFrame containing the original data along with the computed indicators.
    """
    df['SMA_{long}'] = df['Close'].rolling(window=long).mean()
    df['SMA_{short}'] = df['Close'].rolling(window=short).mean()
    df['Signal'] = 0
    df['Signal'][short:] = np.where(df['SMA_{short}'][short:] > df['SMA_{long}'][short:], 1, 0)
    df['Position'] = df['Signal'].diff()
    return df

def connect_kdb() -> None:
    """
    Connects to the KDB+ database for data storage and retrieval.
    """
    pass

def main() -> None:
    """
    The main entry point for the data pipeline.
    """
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    raw_data = download_raw_data(ticker, start_date, end_date)
    print(raw_data.head(8))
    preprocessed_data = preprocess_data(raw_data)
    print(preprocessed_data.head(8))
    print(preprocessed_data.head(8))

    technical_data = technical_indicators(preprocessed_data, long=40, short=20)
    print(technical_data.head(8))

if __name__ == "__main__":
    main()
