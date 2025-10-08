"""
Test module for notebook functionality and validation.
"""

import unittest
import json
import os
from pathlib import Path


class TestNotebooks(unittest.TestCase):
    """Test cases for Jupyter notebooks."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.notebooks_dir = Path(__file__).parent.parent / "notebooks"
        self.expected_notebooks = ["01_EDA.ipynb", "02_result_analysis.ipynb"]
    
    def test_notebooks_directory_exists(self):
        """Test that the notebooks directory exists."""
        self.assertTrue(self.notebooks_dir.exists(), "Notebooks directory should exist")
        self.assertTrue(self.notebooks_dir.is_dir(), "Notebooks path should be a directory")
    
    def test_expected_notebooks_exist(self):
        """Test that expected notebook files exist."""
        for notebook in self.expected_notebooks:
            notebook_path = self.notebooks_dir / notebook
            self.assertTrue(notebook_path.exists(), f"Notebook {notebook} should exist")
    
    def test_notebooks_are_valid_json(self):
        """Test that notebook files are valid JSON."""
        for notebook in self.expected_notebooks:
            notebook_path = self.notebooks_dir / notebook
            if notebook_path.exists():
                try:
                    with open(notebook_path, 'r', encoding='utf-8') as f:
                        notebook_data = json.load(f)
                    self.assertIsInstance(notebook_data, dict, f"{notebook} should be a valid JSON object")
                except json.JSONDecodeError as e:
                    self.fail(f"Notebook {notebook} is not valid JSON: {e}")
    
    def test_notebook_structure(self):
        """Test that notebooks have the expected structure."""
        for notebook in self.expected_notebooks:
            notebook_path = self.notebooks_dir / notebook
            if notebook_path.exists():
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                # Check required top-level keys
                required_keys = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
                for key in required_keys:
                    self.assertIn(key, notebook_data, f"Notebook {notebook} missing required key: {key}")
                
                # Check that cells is a list
                self.assertIsInstance(notebook_data['cells'], list, f"Cells in {notebook} should be a list")
    
    def test_notebook_cells_structure(self):
        """Test that notebook cells have the expected structure."""
        for notebook in self.expected_notebooks:
            notebook_path = self.notebooks_dir / notebook
            if notebook_path.exists():
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                cells = notebook_data.get('cells', [])
                for i, cell in enumerate(cells):
                    # Check required cell keys
                    required_cell_keys = ['cell_type', 'metadata', 'source']
                    for key in required_cell_keys:
                        self.assertIn(key, cell, f"Cell {i} in {notebook} missing required key: {key}")
                    
                    # Check cell type is valid
                    valid_cell_types = ['code', 'markdown', 'raw']
                    self.assertIn(cell['cell_type'], valid_cell_types, 
                                f"Cell {i} in {notebook} has invalid cell_type: {cell['cell_type']}")
    
    def test_notebook_has_content(self):
        """Test that notebooks are not empty."""
        for notebook in self.expected_notebooks:
            notebook_path = self.notebooks_dir / notebook
            if notebook_path.exists():
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                cells = notebook_data.get('cells', [])
                # Check that there's at least one cell with content
                has_content = False
                for cell in cells:
                    source = cell.get('source', [])
                    if source and any(line.strip() for line in source):
                        has_content = True
                        break
                
                self.assertTrue(has_content, f"Notebook {notebook} should have at least one cell with content")
    
    def test_code_cells_syntax(self):
        """Test that code cells have valid Python syntax (basic check)."""
        for notebook in self.expected_notebooks:
            notebook_path = self.notebooks_dir / notebook
            if notebook_path.exists():
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                cells = notebook_data.get('cells', [])
                for i, cell in enumerate(cells):
                    if cell.get('cell_type') == 'code':
                        source = cell.get('source', [])
                        if source:
                            # Join source lines into a single string
                            code = ''.join(source)
                            # Skip empty cells or cells with only comments/whitespace
                            if code.strip() and not all(line.strip().startswith('#') or not line.strip() 
                                                       for line in code.split('\n')):
                                try:
                                    compile(code, f"<notebook {notebook} cell {i}>", "exec")
                                except SyntaxError as e:
                                    self.fail(f"Syntax error in {notebook} cell {i}: {e}")


class TestNotebookDependencies(unittest.TestCase):
    """Test cases for notebook dependencies and imports."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.notebooks_dir = Path(__file__).parent.parent / "notebooks"
        self.requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists for dependency management."""
        self.assertTrue(self.requirements_file.exists(), "requirements.txt should exist")
    
    def test_common_imports_available(self):
        """Test that common data science libraries are importable."""
        common_imports = [
            'pandas',
            'numpy', 
            'matplotlib',
            'matplotlib.pyplot'
        ]
        
        for module_name in common_imports:
            try:
                __import__(module_name)
            except ImportError:
                self.fail(f"Required module {module_name} is not available")
    
    def test_src_modules_accessible(self):
        """Test that src modules can be imported from notebooks context."""
        # This simulates the notebook environment where src modules should be accessible
        import sys
        src_path = str(Path(__file__).parent.parent / "src")
        
        # Check if src path exists
        self.assertTrue(Path(src_path).exists(), "src directory should exist")
        
        # Test that src modules exist
        src_modules = ['attribution.py', 'backtest.py', 'data_pipeline.py', 'signals.py']
        for module_file in src_modules:
            module_path = Path(src_path) / module_file
            self.assertTrue(module_path.exists(), f"Source module {module_file} should exist")


class TestNotebookOutput(unittest.TestCase):
    """Test cases for notebook execution and output."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.notebooks_dir = Path(__file__).parent.parent / "notebooks"
        self.results_dir = Path(__file__).parent.parent / "results"
    
    def test_results_directory_exists(self):
        """Test that results directory exists for notebook outputs."""
        self.assertTrue(self.results_dir.exists(), "Results directory should exist")
    
    def test_notebook_can_access_data_directory(self):
        """Test that notebooks can access the data directory."""
        data_dir = Path(__file__).parent.parent / "data"
        self.assertTrue(data_dir.exists(), "Data directory should exist")
        
        # Check subdirectories
        raw_data_dir = data_dir / "raw"
        processed_data_dir = data_dir / "processed"
        self.assertTrue(raw_data_dir.exists(), "Raw data directory should exist")
        self.assertTrue(processed_data_dir.exists(), "Processed data directory should exist")
    
    def test_notebook_can_access_configs(self):
        """Test that notebooks can access configuration files."""
        configs_dir = Path(__file__).parent.parent / "configs"
        settings_file = configs_dir / "settings.yaml"
        
        self.assertTrue(configs_dir.exists(), "Configs directory should exist")
        self.assertTrue(settings_file.exists(), "Settings file should exist")


if __name__ == '__main__':
    unittest.main()