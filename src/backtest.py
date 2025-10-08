"""
============================================================================================
Title: backtest.py
Author: Kenneth LeGare
Date: 2023-10-01

Description:
    Runs walk forward simulations,
    applies risk controls (factor neutrality, volatility targeting), and evaluates performance.

Dependencies:
    - pandas
    - numpy
    - matplotlib
============================================================================================
"""

# Third-party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def walk_forward_split(df: pd.DataFrame, n_splits: int, test_size: int) -> list:
    """
    Splits the DataFrame into training and testing sets using walk-forward validation.

    args:
        df (pd.DataFrame): The input DataFrame to be split.
        n_splits (int): The number of splits to create.
        test_size (int): The size of the test set for each split.

    Returns:
        list: A list of tuples, each containing a training and testing DataFrame.
    """
    splits = []
    total_size = len(df)
    train_size = (total_size - n_splits * test_size) // n_splits

    for i in range(n_splits):
        start_train = i * (train_size + test_size)
        end_train = start_train + train_size
        start_test = end_train
        end_test = start_test + test_size

        train_df = df.iloc[start_train:end_train]
        test_df = df.iloc[start_test:end_test]

        splits.append((train_df, test_df))

    return splits

def factor_neutralize(df: pd.DataFrame, factors: list) -> pd.DataFrame:
    """
    Neutralizes the DataFrame against specified factors.

    args:
        df (pd.DataFrame): The input DataFrame to be neutralized.
        factors (list): A list of column names representing the factors.

    Returns:
        pd.DataFrame: The factor-neutralized DataFrame.
    """
    from sklearn.linear_model import LinearRegression

    # Make a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Check if all factors exist in the DataFrame
    missing_factors = [f for f in factors if f not in df_copy.columns]
    if missing_factors:
        print(f"Warning: Factors {missing_factors} not found in DataFrame. Skipping factor neutralization.")
        df_copy['Neutralized Returns'] = df_copy['Returns']
        return df_copy

    X = df_copy[factors]
    y = df_copy['Returns']

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    df_copy['Neutralized Returns'] = y - predictions
    return df_copy

def volatility_targeting(df: pd.DataFrame, target_volatility: float) -> pd.DataFrame:
    """
    Applies volatility targeting to the returns in the DataFrame.

    args:
        df (pd.DataFrame): The input DataFrame containing returns.
        target_volatility (float): The target volatility level.

    Returns:
        pd.DataFrame: The DataFrame with volatility-targeted returns.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    rolling_vol = df_copy['Returns'].rolling(window=21).std()
    scaling_factor = target_volatility / rolling_vol
    df_copy['Volatility Targeted Returns'] = df_copy['Returns'] * scaling_factor
    df_copy['Volatility Targeted Returns'] = df_copy['Volatility Targeted Returns'].fillna(0)
    return df_copy


def run_backtest(df: pd.DataFrame, initial_capital: float = 100000.0) -> pd.DataFrame:
    """
    Runs a backtest on the given DataFrame.

    args:
        df (pd.DataFrame): The input DataFrame containing returns.
        initial_capital (float): The initial capital for the backtest.

    Returns:
        pd.DataFrame: A DataFrame containing the portfolio value over time.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    df_copy['Portfolio Value'] = initial_capital * (1 + df_copy['Volatility Targeted Returns'].cumsum())
    return df_copy

def plot_performance(df: pd.DataFrame):
    """
    Plots the portfolio performance over time.

    args:
        df (pd.DataFrame): DataFrame containing the portfolio value over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Portfolio Value'], label='Portfolio Value')
    plt.title('Portfolio Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_performance(df: pd.DataFrame) -> dict:
    """
    Evaluates the performance of the backtest.

    args:
        df (pd.DataFrame): DataFrame containing the portfolio value over time.

    Returns:
        dict: A dictionary containing performance metrics.
    """
    total_return = df['Portfolio Value'].iloc[-1] / df['Portfolio Value'].iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1
    rolling_max = df['Portfolio Value'].cummax()
    drawdown = (df['Portfolio Value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    performance_metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Max Drawdown': max_drawdown
    }

    return performance_metrics

def save_backtest_results(df: pd.DataFrame, filename: str) -> None:
    """
    Saves the backtest results to a CSV file.

    args:
        df (pd.DataFrame): DataFrame containing the backtest results.
        filename (str): The filename to save the results to.
    """
    df.to_csv(filename, index=True)


def main():
    # Sample data with factor columns
    np.random.seed(42)  # For reproducible results
    df = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=500, freq='D'),
        'Returns': np.random.randn(500) / 100,
        'Factor1': np.random.randn(500),  # Market factor
        'Factor2': np.random.randn(500)   # Size factor
    })

    splits = walk_forward_split(df, n_splits=5, test_size=60)

    all_results = []

    for train_df, test_df in splits:
        train_df = factor_neutralize(train_df, factors=['Factor1', 'Factor2'])
        train_df = volatility_targeting(train_df, target_volatility=0.02)

        test_df = factor_neutralize(test_df, factors=['Factor1', 'Factor2'])
        test_df = volatility_targeting(test_df, target_volatility=0.02)

        combined_df = pd.concat([train_df, test_df])
        backtest_df = run_backtest(combined_df)

        all_results.append(backtest_df)

    final_results = pd.concat(all_results)
    plot_performance(final_results)
    performance_metrics = evaluate_performance(final_results)
    print(performance_metrics)
    save_backtest_results(final_results, 'results/backtest_results.csv')

if __name__ == "__main__":
    main()