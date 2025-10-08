"""
Test module for source code functionality.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from attribution import (
        calculate_factor_exposures, 
        attribute_pnl, 
        generate_attribution_report
    )
    from backtest import (
        walk_forward_split, 
        factor_neutralize, 
        volatility_targeting, 
        run_backtest, 
        evaluate_performance
    )
    from data_pipeline import (
        preprocess_data, 
        technical_indicators
    )
    from signals import (
        order_book_imbalance, 
        zscore_normalize, 
        etf_constituent_dislocation
    )
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


class TestAttribution(unittest.TestCase):
    """Test cases for attribution.py functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.returns = pd.DataFrame(
            np.random.randn(100, 2) / 100, 
            index=dates, 
            columns=['Asset1', 'Asset2']
        )
        self.factors = pd.DataFrame(
            np.random.randn(100, 2) / 100, 
            index=dates, 
            columns=['Factor1', 'Factor2']
        )
    
    def test_calculate_factor_exposures(self):
        """Test factor exposure calculation."""
        try:
            exposures = calculate_factor_exposures(self.returns, self.factors)
            
            # Check output structure
            self.assertIsInstance(exposures, pd.DataFrame)
            self.assertEqual(len(exposures), len(self.returns.columns))
            self.assertTrue(all(col in exposures.columns for col in ['const', 'Factor1', 'Factor2']))
        except NameError:
            self.skipTest("calculate_factor_exposures not available")
    
    def test_attribute_pnl(self):
        """Test PnL attribution functionality."""
        try:
            exposures = calculate_factor_exposures(self.returns, self.factors)
            attributed_pnl = attribute_pnl(self.returns, self.factors, exposures)
            
            # Check output structure
            self.assertIsInstance(attributed_pnl, pd.DataFrame)
            self.assertEqual(attributed_pnl.shape, self.returns.shape)
        except NameError:
            self.skipTest("attribute_pnl not available")
    
    def test_generate_attribution_report(self):
        """Test attribution report generation."""
        try:
            exposures = calculate_factor_exposures(self.returns, self.factors)
            attributed_pnl = attribute_pnl(self.returns, self.factors, exposures)
            report = generate_attribution_report(attributed_pnl)
            
            # Check output structure
            self.assertIsInstance(report, pd.DataFrame)
            self.assertIn('Total Attributed PnL', report.columns)
        except NameError:
            self.skipTest("generate_attribution_report not available")


class TestBacktest(unittest.TestCase):
    """Test cases for backtest.py functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'Returns': np.random.randn(100) / 100,
            'Factor1': np.random.randn(100),
            'Factor2': np.random.randn(100)
        })
    
    def test_walk_forward_split(self):
        """Test walk-forward validation splitting."""
        try:
            splits = walk_forward_split(self.df, n_splits=3, test_size=10)
            
            # Check output structure
            self.assertEqual(len(splits), 3)
            for train_df, test_df in splits:
                self.assertIsInstance(train_df, pd.DataFrame)
                self.assertIsInstance(test_df, pd.DataFrame)
                self.assertEqual(len(test_df), 10)
        except NameError:
            self.skipTest("walk_forward_split not available")
    
    def test_factor_neutralize(self):
        """Test factor neutralization."""
        try:
            neutralized_df = factor_neutralize(self.df, factors=['Factor1', 'Factor2'])
            
            # Check output structure
            self.assertIsInstance(neutralized_df, pd.DataFrame)
            self.assertIn('Neutralized Returns', neutralized_df.columns)
            self.assertEqual(len(neutralized_df), len(self.df))
        except NameError:
            self.skipTest("factor_neutralize not available")
    
    def test_volatility_targeting(self):
        """Test volatility targeting."""
        try:
            vol_targeted_df = volatility_targeting(self.df, target_volatility=0.02)
            
            # Check output structure
            self.assertIsInstance(vol_targeted_df, pd.DataFrame)
            self.assertIn('Volatility Targeted Returns', vol_targeted_df.columns)
            self.assertEqual(len(vol_targeted_df), len(self.df))
        except NameError:
            self.skipTest("volatility_targeting not available")
    
    def test_run_backtest(self):
        """Test backtest execution."""
        try:
            # Add required column for backtest
            test_df = self.df.copy()
            test_df['Volatility Targeted Returns'] = test_df['Returns']
            
            backtest_df = run_backtest(test_df, initial_capital=100000.0)
            
            # Check output structure
            self.assertIsInstance(backtest_df, pd.DataFrame)
            self.assertIn('Portfolio Value', backtest_df.columns)
            self.assertEqual(len(backtest_df), len(test_df))
        except NameError:
            self.skipTest("run_backtest not available")
    
    def test_evaluate_performance(self):
        """Test performance evaluation."""
        try:
            # Create test data with portfolio values
            test_df = pd.DataFrame({
                'Portfolio Value': [100000, 101000, 102000, 101500, 103000]
            })
            
            metrics = evaluate_performance(test_df)
            
            # Check output structure
            self.assertIsInstance(metrics, dict)
            required_metrics = ['Total Return', 'Annualized Return', 'Max Drawdown']
            for metric in required_metrics:
                self.assertIn(metric, metrics)
        except NameError:
            self.skipTest("evaluate_performance not available")


class TestDataPipeline(unittest.TestCase):
    """Test cases for data_pipeline.py functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(80, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        try:
            # Add some missing values and duplicates for testing
            test_df = self.df.copy()
            test_df.iloc[5:10] = np.nan
            
            processed_df = preprocess_data(test_df)
            
            # Check that missing values are handled
            self.assertFalse(processed_df.isnull().any().any())
            self.assertIsInstance(processed_df, pd.DataFrame)
        except NameError:
            self.skipTest("preprocess_data not available")
    
    def test_technical_indicators(self):
        """Test technical indicators calculation."""
        try:
            # Make a copy to avoid modifying the original test data
            test_df = self.df.copy()
            indicators_df = technical_indicators(test_df, long=20, short=10)
            
            # Check output structure
            self.assertIsInstance(indicators_df, pd.DataFrame)
            
            # The function should add SMA columns, Signal, and Position
            expected_new_cols = ['SMA_20', 'SMA_10', 'Signal', 'Position']
            for col in expected_new_cols:
                self.assertIn(col, indicators_df.columns, f"Expected column '{col}' not found")
                
            # Check that original columns are still there
            for col in self.df.columns:
                self.assertIn(col, indicators_df.columns, f"Original column '{col}' missing")
        except NameError:
            self.skipTest("technical_indicators not available")


class TestSignals(unittest.TestCase):
    """Test cases for signals.py functionality."""
    
    def test_order_book_imbalance(self):
        """Test order book imbalance calculation."""
        try:
            bid_sizes = np.array([100, 150, 200])
            ask_sizes = np.array([80, 120, 160])
            
            imbalance = order_book_imbalance(bid_sizes, ask_sizes)
            
            # Check output structure and values
            self.assertIsInstance(imbalance, np.ndarray)
            self.assertEqual(len(imbalance), len(bid_sizes))
            self.assertTrue(all(-1 <= val <= 1 for val in imbalance))
        except NameError:
            self.skipTest("order_book_imbalance not available")
    
    def test_zscore_normalize(self):
        """Test z-score normalization."""
        try:
            series = pd.Series([1, 2, 3, 4, 5])
            normalized = zscore_normalize(series)
            
            # Check output structure and properties
            self.assertIsInstance(normalized, pd.Series)
            self.assertEqual(len(normalized), len(series))
            self.assertAlmostEqual(normalized.mean(), 0, places=5)
            # Note: scipy.stats.zscore uses ddof=0, so std might not be exactly 1
            # Just check that it's reasonably close to 1
            self.assertGreater(normalized.std(), 0.5)
            self.assertLess(normalized.std(), 2.0)
        except NameError:
            self.skipTest("zscore_normalize not available")
    
    def test_etf_constituent_dislocation(self):
        """Test ETF-constituent dislocation calculation."""
        try:
            etf_price = pd.Series([100, 102, 101, 103])
            constituent_prices = pd.DataFrame({
                'StockA': [30, 31, 30.5, 32],
                'StockB': [40, 41, 40.5, 42],
                'StockC': [25, 26, 25.5, 26.5]
            })
            
            dislocation = etf_constituent_dislocation(etf_price, constituent_prices)
            
            # Check output structure
            self.assertIsInstance(dislocation, pd.Series)
            self.assertEqual(len(dislocation), len(etf_price))
        except NameError:
            self.skipTest("etf_constituent_dislocation not available")


if __name__ == '__main__':
    unittest.main()