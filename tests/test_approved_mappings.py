#!/usr/bin/env python3
"""
Unit tests for approved mappings functionality in normalize_horse_names.py

Tests the approved mappings cache system including:
- Loading approved mappings from JSON file
- Saving approved mappings with optimization (only save when normalization occurred)
- File I/O error handling
- Directory creation
- Edge cases and error conditions
"""

import unittest
import sys
import os
import tempfile
import json
import shutil
from unittest.mock import patch, mock_open, MagicMock

# Add the parent directory to the path so we can import the script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from normalize_horse_names.py
from normalize_horse_names import (
    load_approved_mappings,
    save_approved_mapping
)


class TestLoadApprovedMappings(unittest.TestCase):
    """Test loading approved mappings from JSON file."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_approved_mappings.json')
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_load_approved_mappings_file_exists_valid_json(self):
        """Test loading approved mappings when file exists with valid JSON."""
        # Create test data with both normalizations and identity mappings
        test_data = {
            "Goodwill": "Good Will",
            "OHalon": "O'Halon",
            "Mandy": "Mandy",  # Identity mapping (should be loaded if present)
            "davinci": "DaVinci"
        }
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            result = load_approved_mappings()
        
        self.assertEqual(result, test_data)
    
    def test_load_approved_mappings_file_not_found(self):
        """Test loading approved mappings when file doesn't exist."""
        nonexistent_file = '/nonexistent/path/mappings.json'
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', nonexistent_file):
            result = load_approved_mappings()
        
        self.assertEqual(result, {})
    
    def test_load_approved_mappings_invalid_json(self):
        """Test loading approved mappings when file has invalid JSON."""
        # Create file with invalid JSON
        with open(self.test_file, 'w') as f:
            f.write('{ invalid json content }')
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            result = load_approved_mappings()
        
        self.assertEqual(result, {})
    
    def test_load_approved_mappings_empty_file(self):
        """Test loading approved mappings when file is empty."""
        # Create empty file
        with open(self.test_file, 'w') as f:
            f.write('')
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            result = load_approved_mappings()
        
        self.assertEqual(result, {})
    
    def test_load_approved_mappings_file_permission_error(self):
        """Test loading approved mappings when file can't be read due to permissions."""
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file), \
             patch('builtins.open', side_effect=PermissionError("Permission denied")):
            result = load_approved_mappings()
        
        self.assertEqual(result, {})


class TestSaveApprovedMapping(unittest.TestCase):
    """Test saving approved mappings with optimization."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_approved_mappings.json')
        self.approved_mappings = {}
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_save_approved_mapping_normalization_occurred(self):
        """Test that actual normalizations ARE saved to cache."""
        # Test case where normalization occurred
        original_name = "Goodwill"
        normalized_name = "Good Will"
        master_list = ["Good Will", "Ben", "Thunder"]  # Doesn't matter for normalizations
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            save_approved_mapping(original_name, normalized_name, self.approved_mappings, master_list)
        
        # Should be added to in-memory dict
        self.assertIn(original_name, self.approved_mappings)
        self.assertEqual(self.approved_mappings[original_name], normalized_name)
        
        # Should be saved to file
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, {original_name: normalized_name})
    
    def test_save_approved_mapping_identity_mapping_in_master_list_not_saved(self):
        """Test that identity mappings for names IN master list are NOT saved to cache."""
        # Test case where no normalization occurred and name is in master list
        original_name = "Mandy"
        normalized_name = "Mandy"  # Same as original
        master_list = ["Mandy", "Ben", "Thunder"]  # Name IS in master list
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            save_approved_mapping(original_name, normalized_name, self.approved_mappings, master_list)
        
        # Should NOT be added to in-memory dict
        self.assertNotIn(original_name, self.approved_mappings)
        
        # File should NOT be created
        self.assertFalse(os.path.exists(self.test_file))
    
    def test_save_approved_mapping_identity_mapping_not_in_master_list_saved(self):
        """Test that identity mappings for names NOT in master list ARE saved to cache."""
        # Test case where no normalization occurred but name is NOT in master list
        original_name = "Midnight Express"
        normalized_name = "Midnight Express"  # Same as original
        master_list = ["Mandy", "Ben", "Thunder"]  # Name is NOT in master list
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            save_approved_mapping(original_name, normalized_name, self.approved_mappings, master_list)
        
        # Should be added to in-memory dict (user decision to keep non-standard name)
        self.assertIn(original_name, self.approved_mappings)
        self.assertEqual(self.approved_mappings[original_name], normalized_name)
        
        # Should be saved to file
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, {original_name: normalized_name})
    
    def test_save_approved_mapping_mixed_cases_smart_caching(self):
        """Test saving multiple mappings with smart caching logic."""
        # Start with existing mappings
        self.approved_mappings = {"Existing": "ExistingNorm"}
        
        # Master list contains some of these names
        master_list = ["Good Will", "Ben", "Thunder", "DaVinci"]
        
        test_cases = [
            ("Goodwill", "Good Will"),      # Should save (normalization)
            ("Mandy", "Mandy"),             # Should save (identity, NOT in master list)
            ("OHalon", "O'Halon"),          # Should save (normalization)  
            ("Ben", "Ben"),                 # Should NOT save (identity, in master list)
            ("davinci", "DaVinci"),         # Should save (normalization)
            ("Midnight Express", "Midnight Express"),  # Should save (identity, NOT in master list)
            ("Thunder", "Thunder")          # Should NOT save (identity, in master list)
        ]
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            for original, normalized in test_cases:
                save_approved_mapping(original, normalized, self.approved_mappings, master_list)
        
        # Check in-memory dict - should have normalizations and non-standard identity mappings
        expected_mappings = {
            "Existing": "ExistingNorm",
            "Goodwill": "Good Will",        # Normalization
            "Mandy": "Mandy",               # Identity, but NOT in master list
            "OHalon": "O'Halon",            # Normalization
            "davinci": "DaVinci",           # Normalization
            "Midnight Express": "Midnight Express"  # Identity, but NOT in master list
            # "Ben", "Thunder" should NOT be saved (identity, in master list)
        }
        self.assertEqual(self.approved_mappings, expected_mappings)
        
        # Check file content - should match in-memory dict
        with open(self.test_file, 'r') as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data, expected_mappings)
    
    def test_save_approved_mapping_directory_creation(self):
        """Test that directory is created if it doesn't exist."""
        nested_dir = os.path.join(self.test_dir, 'nested', 'dir')
        nested_file = os.path.join(nested_dir, 'mappings.json')
        master_list = ["TestNorm", "Other"]
        
        # Ensure directory doesn't exist initially
        self.assertFalse(os.path.exists(nested_dir))
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', nested_file):
            save_approved_mapping("Test", "TestNorm", self.approved_mappings, master_list)
        
        # Directory should be created
        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.exists(nested_file))
    
    def test_save_approved_mapping_backward_compatibility_no_master_list(self):
        """Test that the function works without master_list parameter (backward compatibility)."""
        # Test case where master_list is None (old behavior)
        original_name = "Mandy"
        normalized_name = "Mandy"  # Same as original
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            save_approved_mapping(original_name, normalized_name, self.approved_mappings, None)
        
        # Without master_list, identity mappings should NOT be saved (old optimization behavior)
        self.assertNotIn(original_name, self.approved_mappings)
        self.assertFalse(os.path.exists(self.test_file))
    
    def test_save_approved_mapping_file_write_error(self):
        """Test handling of file write errors."""
        master_list = ["TestNorm", "Other"]
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file), \
             patch('json.dump', side_effect=IOError("Disk full")):
            
            # Should not raise exception, just print warning
            save_approved_mapping("Test", "TestNorm", self.approved_mappings, master_list)
            
            # In-memory dict should still be updated
            self.assertEqual(self.approved_mappings["Test"], "TestNorm")
    
    def test_save_approved_mapping_directory_creation_error(self):
        """Test handling of directory creation errors."""
        master_list = ["TestNorm", "Other"]
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file), \
             patch('os.makedirs', side_effect=OSError("Permission denied")):
            
            # Should not raise exception, just handle gracefully
            save_approved_mapping("Test", "TestNorm", self.approved_mappings, master_list)
            
            # In-memory dict should still be updated
            self.assertEqual(self.approved_mappings["Test"], "TestNorm")
    
    def test_save_approved_mapping_preserves_existing_data(self):
        """Test that saving new mappings preserves existing data in file."""
        # Create initial file with some data
        initial_data = {"Existing1": "ExistingNorm1", "Existing2": "ExistingNorm2"}
        with open(self.test_file, 'w') as f:
            json.dump(initial_data, f)
        
        # Start with the existing data in memory
        self.approved_mappings.update(initial_data)
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            # Add new mapping
            master_list = ["NewHorseNorm", "Other"]
            save_approved_mapping("NewHorse", "NewHorseNorm", self.approved_mappings, master_list)
        
        # Check that both old and new data are preserved
        expected_data = {
            "Existing1": "ExistingNorm1",
            "Existing2": "ExistingNorm2", 
            "NewHorse": "NewHorseNorm"
        }
        
        with open(self.test_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, expected_data)
        self.assertEqual(self.approved_mappings, expected_data)


class TestApprovedMappingsEdgeCases(unittest.TestCase):
    """Test edge cases for approved mappings functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_approved_mappings.json')
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_save_approved_mapping_empty_strings(self):
        """Test handling of empty strings."""
        approved_mappings = {}
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            # Empty original name with non-empty normalized name
            save_approved_mapping("", "SomeName", approved_mappings)
            self.assertEqual(approved_mappings, {"": "SomeName"})
            
            # Non-empty original name with empty normalized name  
            approved_mappings.clear()
            save_approved_mapping("SomeName", "", approved_mappings)
            self.assertEqual(approved_mappings, {"SomeName": ""})
            
            # Both empty (identity case)
            approved_mappings.clear()
            save_approved_mapping("", "", approved_mappings)
            self.assertEqual(approved_mappings, {})  # Should not save identity mapping
    
    def test_save_approved_mapping_whitespace_handling(self):
        """Test handling of names with whitespace."""
        approved_mappings = {}
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            # Names with leading/trailing whitespace
            save_approved_mapping("  Mandy  ", "Mandy", approved_mappings)
            self.assertEqual(approved_mappings, {"  Mandy  ": "Mandy"})
            
            # Identity case with whitespace
            approved_mappings.clear()
            save_approved_mapping("  Mandy  ", "  Mandy  ", approved_mappings)
            self.assertEqual(approved_mappings, {})  # Should not save identity mapping
    
    def test_save_approved_mapping_unicode_characters(self):
        """Test handling of unicode characters in horse names."""
        approved_mappings = {}
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            # Unicode characters
            save_approved_mapping("Naïve", "Naive", approved_mappings)
            self.assertEqual(approved_mappings, {"Naïve": "Naive"})
            
            # Check file can handle unicode
            with open(self.test_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data, {"Naïve": "Naive"})


class TestApprovedMappingsIntegration(unittest.TestCase):
    """Test integration scenarios for approved mappings."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_approved_mappings.json')
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cache_optimization_reduces_file_size(self):
        """Test that the optimization significantly reduces cache file size."""
        approved_mappings = {}
        
        # Simulate a realistic normalization session with many identity mappings
        test_cases = [
            # Actual normalizations (should be saved)
            ("Goodwill", "Good Will"),
            ("OHalon", "O'Halon"),
            ("davinci", "DaVinci"),
            ("Issac", "Isaac"),
            ("Guiness", "Guinness"),
            
            # Identity mappings (should NOT be saved)
            ("Absinthe", "Absinthe"),
            ("Ace", "Ace"),
            ("Ben", "Ben"),
            ("Finnick", "Finnick"),
            ("Thunder", "Thunder"),
            ("Lightning", "Lightning"),
            ("Storm", "Storm"),
            ("Breeze", "Breeze"),
            ("Rain", "Rain"),
            ("Snow", "Snow"),
            ("Cloudy", "Cloudy"),
            ("Sunny", "Sunny"),
            ("Windy", "Windy"),
            ("Misty", "Misty"),
            ("Frosty", "Frosty")
        ]
        
        with patch('normalize_horse_names.APPROVED_MAPPINGS_FILE', self.test_file):
            for original, normalized in test_cases:
                save_approved_mapping(original, normalized, approved_mappings)
        
        # Check that only normalizations are saved (not identity mappings)
        expected_cache = {
            "Goodwill": "Good Will",
            "OHalon": "O'Halon",
            "davinci": "DaVinci", 
            "Issac": "Isaac",
            "Guiness": "Guinness"
        }
        
        self.assertEqual(approved_mappings, expected_cache)
        
        # Check file contents
        with open(self.test_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, expected_cache)
        
        # Verify we saved only 5 entries instead of 20 (75% reduction)
        self.assertEqual(len(saved_data), 5)
        
        # Calculate the theoretical size reduction
        total_entries = len(test_cases)
        saved_entries = len(saved_data)
        reduction_percentage = (1 - saved_entries / total_entries) * 100
        
        self.assertGreater(reduction_percentage, 70)  # At least 70% reduction
        print(f"Cache optimization achieved {reduction_percentage:.1f}% size reduction")


if __name__ == '__main__':
    unittest.main()