"""
Test module for configuration handling and settings validation.
"""

import unittest
import yaml
import os
from pathlib import Path


class TestConfig(unittest.TestCase):
    """Test cases for configuration handling."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config_path = Path(__file__).parent.parent / "configs" / "settings.yaml"
    
    def test_config_file_exists(self):
        """Test that the configuration file exists."""
        self.assertTrue(self.config_path.exists(), "Configuration file should exist")
    
    def test_config_file_valid_yaml(self):
        """Test that the configuration file is valid YAML."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.assertIsInstance(config, dict, "Config should be a dictionary")
        except yaml.YAMLError as e:
            self.fail(f"Configuration file is not valid YAML: {e}")
    
    def test_required_config_sections(self):
        """Test that all required configuration sections are present."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        required_sections = ['data', 'paths', 'backtest', 'strategy']
        for section in required_sections:
            self.assertIn(section, config, f"Required section '{section}' missing from config")
    
    def test_data_config_structure(self):
        """Test the structure of data configuration section."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        data_config = config.get('data', {})
        required_fields = ['source', 'tickers', 'start_date', 'end_date']
        
        for field in required_fields:
            self.assertIn(field, data_config, f"Required field '{field}' missing from data config")
        
        # Test data types
        self.assertIsInstance(data_config['tickers'], list, "Tickers should be a list")
        self.assertTrue(len(data_config['tickers']) > 0, "At least one ticker should be specified")
    
    def test_paths_config_structure(self):
        """Test the structure of paths configuration section."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        paths_config = config.get('paths', {})
        required_paths = ['raw_data_paths', 'processed_data_paths', 'notebooks', 'tests']
        
        for path_key in required_paths:
            self.assertIn(path_key, paths_config, f"Required path '{path_key}' missing from paths config")
    
    def test_backtest_config_structure(self):
        """Test the structure of backtest configuration section."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        backtest_config = config.get('backtest', {})
        required_fields = ['start_date', 'end_date', 'initial_capital']
        
        for field in required_fields:
            self.assertIn(field, backtest_config, f"Required field '{field}' missing from backtest config")
        
        # Test data types and ranges
        self.assertIsInstance(backtest_config['initial_capital'], (int, float), 
                             "Initial capital should be numeric")
        self.assertGreater(backtest_config['initial_capital'], 0, 
                          "Initial capital should be positive")
    
    def test_strategy_config_structure(self):
        """Test the structure of strategy configuration section."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        strategy_config = config.get('strategy', {})
        required_fields = ['type', 'lookback_period', 'max_positions']
        
        for field in required_fields:
            self.assertIn(field, strategy_config, f"Required field '{field}' missing from strategy config")
        
        # Test constraints
        self.assertGreater(strategy_config['lookback_period'], 0, 
                          "Lookback period should be positive")
        self.assertGreater(strategy_config['max_positions'], 0, 
                          "Max positions should be positive")


class TestDirectoryStructure(unittest.TestCase):
    """Test cases for project directory structure."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.project_root = Path(__file__).parent.parent
    
    def test_required_directories_exist(self):
        """Test that all required directories exist."""
        required_dirs = ['src', 'tests', 'configs', 'data', 'notebooks', 'results']
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            self.assertTrue(dir_path.exists(), f"Required directory '{dir_name}' should exist")
            self.assertTrue(dir_path.is_dir(), f"'{dir_name}' should be a directory")
    
    def test_data_subdirectories_exist(self):
        """Test that data subdirectories exist."""
        data_subdirs = ['raw', 'processed']
        
        for subdir in data_subdirs:
            subdir_path = self.project_root / 'data' / subdir
            self.assertTrue(subdir_path.exists(), f"Data subdirectory '{subdir}' should exist")


if __name__ == '__main__':
    unittest.main()