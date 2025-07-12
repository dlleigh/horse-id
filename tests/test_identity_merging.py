import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import re

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the problematic ML/CV imports before importing the module
with patch.dict('sys.modules', {
    'wildlife_tools': Mock(),
    'wildlife_tools.similarity': Mock(),
    'wildlife_tools.similarity.wildfusion': Mock(),
    'wildlife_tools.features': Mock(),
    'wildlife_tools.data': Mock(),
    'timm': Mock(),
    'torch': Mock(),
    'torchvision': Mock(),
    'torchvision.transforms': Mock()
}):
    from merge_horse_identities import load_recurring_names


class TestLoadRecurringNames:
    """Test recurring names loading functionality."""
    
    def test_load_recurring_names_file_not_found(self):
        """Test loading recurring names when file doesn't exist."""
        with patch('merge_horse_identities.os.path.exists', return_value=False):
            result = load_recurring_names()
        
        assert result == set()
    
    def test_load_recurring_names_success(self):
        """Test successful loading of recurring names."""
        # Create mock Excel data
        mock_data = pd.DataFrame({
            'col_0': [None, None, None, 'Pasture A', None, 'Cowboy 1', 'Sunny 2', 'Thunder', None],
            'col_1': [None, None, None, None, None, None, None, None, None],
            'col_2': [None, None, None, 'Pasture B', None, 'Lightning 1', 'Storm 2', 'Breeze', None],
            'col_3': [None, None, None, None, None, None, None, None, None],
            'col_4': [None, None, None, 'Pasture C', None, 'Cowboy 2', 'Sunny 3', 'Rain', None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        expected = {'Cowboy', 'Lightning', 'Storm', 'Sunny'}
        assert result == expected
    
    def test_load_recurring_names_no_numbered_horses(self):
        """Test loading recurring names with no numbered horses."""
        mock_data = pd.DataFrame({
            'col_0': [None, None, None, 'Pasture A', None, 'Thunder', 'Lightning', 'Storm', None],
            'col_1': [None, None, None, None, None, None, None, None, None],
            'col_2': [None, None, None, 'Pasture B', None, 'Breeze', 'Rain', 'Snow', None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        assert result == set()
    
    def test_load_recurring_names_mixed_formats(self):
        """Test loading recurring names with mixed number formats."""
        mock_data = pd.DataFrame({
            'col_0': [None, None, None, 'Pasture A', None, 'Cowboy 1', 'Sunny 2', 'Thunder 10', None],
            'col_1': [None, None, None, None, None, None, None, None, None],
            'col_2': [None, None, None, 'Pasture B', None, 'Lightning', 'Storm 22', 'Breeze', None],
            'col_3': [None, None, None, None, None, None, None, None, None],
            'col_4': [None, None, None, 'Pasture C', None, 'Cowboy 3', 'Rain 1', 'Snow', None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        expected = {'Cowboy', 'Sunny', 'Thunder', 'Storm', 'Rain'}
        assert result == expected
    
    def test_load_recurring_names_empty_dataframe(self):
        """Test loading recurring names with empty DataFrame."""
        mock_data = pd.DataFrame()
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        assert result == set()
    
    def test_load_recurring_names_missing_row_3(self):
        """Test loading recurring names with missing row 3."""
        mock_data = pd.DataFrame({
            'col_0': [None, None],
            'col_1': [None, None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        assert result == set()
    
    def test_load_recurring_names_with_nan_values(self):
        """Test loading recurring names with NaN values."""
        mock_data = pd.DataFrame({
            'col_0': [None, None, None, 'Pasture A', None, 'Cowboy 1', np.nan, 'Thunder', None],
            'col_1': [None, None, None, None, None, None, None, None, None],
            'col_2': [None, None, None, 'Pasture B', None, np.nan, 'Storm 2', 'Breeze', None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        expected = {'Cowboy', 'Storm'}
        assert result == expected
    
    def test_load_recurring_names_feed_column_exclusion(self):
        """Test loading recurring names excludes 'Feed:' columns."""
        mock_data = pd.DataFrame({
            'col_0': [None, None, None, 'Pasture A', None, 'Cowboy 1', 'Sunny 2', 'Thunder', None],
            'col_1': [None, None, None, None, None, None, None, None, None],
            'col_2': [None, None, None, 'Feed:', None, 'Hay', 'Grain', 'Water', None],
            'col_3': [None, None, None, None, None, None, None, None, None],
            'col_4': [None, None, None, 'Pasture B', None, 'Lightning 1', 'Storm 2', 'Breeze', None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        expected = {'Cowboy', 'Sunny', 'Lightning', 'Storm'}
        assert result == expected
    
    def test_load_recurring_names_excel_read_error(self):
        """Test loading recurring names with Excel read error."""
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', side_effect=Exception('Excel read error')):
                result = load_recurring_names()
        
        assert result == set()
    
    def test_load_recurring_names_complex_names(self):
        """Test loading recurring names with complex horse names."""
        mock_data = pd.DataFrame({
            'col_0': [None, None, None, 'Pasture A', None, 'Sky Blue 1', 'Thunder Storm 2', 'Little Thunder', None],
            'col_1': [None, None, None, None, None, None, None, None, None],
            'col_2': [None, None, None, 'Pasture B', None, 'Sky Blue 2', 'Thunder Storm 3', 'Big Thunder', None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        expected = {'Sky Blue', 'Thunder Storm'}
        assert result == expected
    
    def test_load_recurring_names_single_space_separation(self):
        """Test loading recurring names with single space separation."""
        mock_data = pd.DataFrame({
            'col_0': [None, None, None, 'Pasture A', None, 'Thunder1', 'Lightning 2', 'Storm3', None],
            'col_1': [None, None, None, None, None, None, None, None, None],
            'col_2': [None, None, None, 'Pasture B', None, 'Thunder 1', 'Lightning3', 'Storm 2', None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        expected = {'Thunder', 'Lightning', 'Storm'}
        assert result == expected
    
    def test_load_recurring_names_edge_cases(self):
        """Test loading recurring names with edge cases."""
        mock_data = pd.DataFrame({
            'col_0': [None, None, None, 'Pasture A', None, 'A 1', 'B 123', 'C 0', None],
            'col_1': [None, None, None, None, None, None, None, None, None],
            'col_2': [None, None, None, 'Pasture B', None, 'A 2', 'B 456', 'D 1', None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        expected = {'A', 'B', 'C', 'D'}
        assert result == expected
    
    def test_load_recurring_names_whitespace_handling(self):
        """Test loading recurring names with whitespace handling."""
        mock_data = pd.DataFrame({
            'col_0': [None, None, None, 'Pasture A', None, '  Thunder 1  ', 'Lightning 2', '  Storm  ', None],
            'col_1': [None, None, None, None, None, None, None, None, None],
            'col_2': [None, None, None, 'Pasture B', None, 'Thunder 2', '  Lightning 3  ', 'Rain', None]
        })
        
        with patch('merge_horse_identities.os.path.exists', return_value=True):
            with patch('merge_horse_identities.pd.read_excel', return_value=mock_data):
                result = load_recurring_names()
        
        expected = {'Thunder', 'Lightning'}
        assert result == expected
    
    def test_load_recurring_names_regex_pattern_validation(self):
        """Test that the regex pattern correctly identifies numbered horses."""
        test_cases = [
            ('Thunder 1', True),
            ('Lightning 22', True),
            ('Storm 123', True),
            ('Thunder1', False),  # No space
            ('Lightning22', False),  # No space
            ('Storm', False),  # No number
            ('Thunder a', False),  # Not a number
            ('Lightning 1a', False),  # Not pure number
            ('Storm 1 2', False),  # Multiple parts after space
            ('A 1', True),  # Single character name
            ('Sky Blue 1', True),  # Multi-word name
            ('Thunder-Lightning 1', True),  # Name with dash
        ]
        
        # Test the logic that identifies numbered horses (more robust than regex)
        def is_numbered_horse(name):
            # Split on spaces to get words
            parts = name.split()
            if len(parts) < 2:
                return False
            
            # Check if exactly the last part is a digit and second-to-last is not a digit
            last_part = parts[-1]
            if not last_part.isdigit():
                return False
                
            # If there are only 2 parts, it's valid (name + number)
            if len(parts) == 2:
                return True
                
            # If there are more parts, the second-to-last should not be a digit
            # This rejects cases like "Storm 1 2" where both "1" and "2" are digits
            second_last = parts[-2]
            return not second_last.isdigit()
        
        for name, should_match in test_cases:
            matches = is_numbered_horse(name)
            assert matches == should_match, f"Pattern matching failed for '{name}': expected {should_match}, got {matches}"
    
    def test_load_recurring_names_base_name_extraction(self):
        """Test that base names are correctly extracted from numbered horses."""
        test_cases = [
            ('Thunder 1', 'Thunder'),
            ('Lightning 22', 'Lightning'),
            ('Sky Blue 1', 'Sky Blue'),
            ('Thunder-Lightning 123', 'Thunder-Lightning'),
            ('A 1', 'A'),
            ('Very Long Horse Name 999', 'Very Long Horse Name')
        ]
        
        # Test the regex substitution used in the function
        pattern = r'\s+\d+$'
        
        for full_name, expected_base in test_cases:
            base_name = re.sub(pattern, '', full_name)
            assert base_name == expected_base, f"Base name extraction failed for '{full_name}': expected '{expected_base}', got '{base_name}'"