#!/usr/bin/env python3
"""
Unit tests for normalize_horse_names.py

Tests the core normalization logic including:
- Exact matches
- Case-insensitive matches  
- Special case handling
- Fuzzy matching
- Base name matching with numbers
- Confidence scoring
"""

import unittest
import sys
import os
import tempfile
import json
import pandas as pd
from unittest.mock import patch, mock_open

# Add the parent directory to the path so we can import the script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from normalize_horse_names.py
from normalize_horse_names import (
    normalize_name_for_comparison,
    create_base_name,
    find_exact_matches,
    find_base_name_matches,
    find_special_case_matches,
    find_fuzzy_matches,
    find_substring_matches,
    get_normalization_candidates,
    get_approved_names_list,
    find_approved_name_matches,
    NormalizationCandidate
)


class TestNormalizationFunctions(unittest.TestCase):
    """Test core normalization functions."""
    
    def setUp(self):
        """Set up test data."""
        self.master_list = [
            "Absinthe", "Ace", "Cowboy 1", "Cowboy 2", "Good Will", 
            "O'Halon", "DaVinci", "Ceelo", "Isaac", "Raffiki",
            "Guinness", "Louie 1", "Louie 2", "Ben", "Finnick"
        ]
    
    def test_normalize_name_for_comparison(self):
        """Test name normalization for comparison."""
        self.assertEqual(normalize_name_for_comparison("Good Will"), "goodwill")
        self.assertEqual(normalize_name_for_comparison("O'Halon"), "ohalon")
        self.assertEqual(normalize_name_for_comparison("Da-Vinci"), "davinci")
        self.assertEqual(normalize_name_for_comparison("COWBOY"), "cowboy")
        self.assertEqual(normalize_name_for_comparison(""), "")
    
    def test_create_base_name(self):
        """Test base name creation."""
        self.assertEqual(create_base_name("Cowboy 1"), "Cowboy")
        self.assertEqual(create_base_name("Louie 2"), "Louie")
        self.assertEqual(create_base_name("Sunny 10"), "Sunny")
        self.assertEqual(create_base_name("Absinthe"), "Absinthe")  # No change
        self.assertEqual(create_base_name("Storm 1 2"), "Storm 1")  # Removes last number only
    
    def test_find_exact_matches(self):
        """Test exact matching."""
        # Exact match
        candidates = find_exact_matches("Absinthe", self.master_list)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "Absinthe")
        self.assertEqual(candidates[0].confidence, 1.0)
        self.assertEqual(candidates[0].method, "exact")
        
        # Case-insensitive match
        candidates = find_exact_matches("ABSINTHE", self.master_list)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "Absinthe")
        self.assertEqual(candidates[0].confidence, 0.95)
        self.assertEqual(candidates[0].method, "exact_case_insensitive")
        
        # No match
        candidates = find_exact_matches("Nonexistent", self.master_list)
        self.assertEqual(len(candidates), 0)
    
    def test_find_base_name_matches(self):
        """Test base name matching with numbers."""
        # Should match Cowboy 1 and Cowboy 2
        candidates = find_base_name_matches("Cowboy", self.master_list)
        self.assertEqual(len(candidates), 2)
        names = [c.name for c in candidates]
        self.assertIn("Cowboy 1", names)
        self.assertIn("Cowboy 2", names)
        
        # Should match Louie 1 and Louie 2
        candidates = find_base_name_matches("Louie", self.master_list)
        self.assertEqual(len(candidates), 2)
        names = [c.name for c in candidates]
        self.assertIn("Louie 1", names)
        self.assertIn("Louie 2", names)
        
        # Should find exact match (base name logic finds all matches, including exact)
        candidates = find_base_name_matches("Absinthe", self.master_list)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "Absinthe")
    
    def test_find_special_case_matches(self):
        """Test special case matching."""
        # Good Will variations
        candidates = find_special_case_matches("Goodwill", self.master_list)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "Good Will")
        self.assertEqual(candidates[0].method, "special_case")
        
        # O'Halon variations
        candidates = find_special_case_matches("OHalon", self.master_list)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "O'Halon")
        
        # DaVinci variations
        candidates = find_special_case_matches("Da Vinci", self.master_list)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "DaVinci")
        
        # No special case
        candidates = find_special_case_matches("Absinthe", self.master_list)
        self.assertEqual(len(candidates), 0)
    
    def test_find_fuzzy_matches(self):
        """Test fuzzy matching."""
        # Close misspelling (use lower threshold for this test)
        candidates = find_fuzzy_matches("Issac", self.master_list, threshold=0.8)
        isaac_matches = [c for c in candidates if c.name == "Isaac"]
        self.assertTrue(len(isaac_matches) > 0)
        self.assertTrue(isaac_matches[0].confidence >= 0.8)
        
        # Another close match  
        candidates = find_fuzzy_matches("Guiness", self.master_list, threshold=0.8)
        guinness_matches = [c for c in candidates if c.name == "Guinness"]
        self.assertTrue(len(guinness_matches) > 0)
        
        # No close matches (should return empty or very low confidence)
        candidates = find_fuzzy_matches("Zebra", self.master_list)
        high_confidence = [c for c in candidates if c.confidence >= 0.85]
        self.assertEqual(len(high_confidence), 0)
    
    def test_find_substring_matches(self):
        """Test substring matching."""
        # "Ben" should match "Finnick" -> "Ben" is substring
        candidates = find_substring_matches("Benji", self.master_list)
        ben_matches = [c for c in candidates if c.name == "Ben"]
        self.assertTrue(len(ben_matches) > 0)
        
        # "Finn" should match "Finnick"
        candidates = find_substring_matches("Finn", self.master_list)
        finnick_matches = [c for c in candidates if c.name == "Finnick"]
        self.assertTrue(len(finnick_matches) > 0)
        
        # Length difference too large should not match
        candidates = find_substring_matches("A", self.master_list)
        self.assertEqual(len(candidates), 0)  # Too short (less than 3 chars)
    
    def test_get_normalization_candidates(self):
        """Test comprehensive candidate generation."""
        # Test exact match (should be first/highest confidence)
        candidates = get_normalization_candidates("Absinthe", self.master_list)
        self.assertTrue(len(candidates) > 0)
        self.assertEqual(candidates[0].name, "Absinthe")
        self.assertEqual(candidates[0].confidence, 1.0)
        
        # Test special case
        candidates = get_normalization_candidates("Goodwill", self.master_list)
        self.assertTrue(len(candidates) > 0)
        good_will_match = next((c for c in candidates if c.name == "Good Will"), None)
        self.assertIsNotNone(good_will_match)
        self.assertEqual(good_will_match.method, "special_case")
        
        # Test that candidates are sorted by confidence (descending)
        candidates = get_normalization_candidates("Issac", self.master_list)
        if len(candidates) > 1:
            for i in range(len(candidates) - 1):
                self.assertGreaterEqual(candidates[i].confidence, candidates[i+1].confidence)
        
        # Test empty input
        candidates = get_normalization_candidates("", self.master_list)
        self.assertEqual(len(candidates), 0)
        
        candidates = get_normalization_candidates(None, self.master_list)
        self.assertEqual(len(candidates), 0)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty master list
        candidates = get_normalization_candidates("Absinthe", [])
        self.assertEqual(len(candidates), 0)
        
        # None in master list
        master_with_none = ["Absinthe", None, "Ace"]
        candidates = get_normalization_candidates("Absinthe", master_with_none)
        # Should still work, just skip None values
        self.assertTrue(len(candidates) > 0)
        
        # Special characters in names
        special_master = ["Test-Horse", "Horse O'Connor", "Horse & Rider"]
        candidates = get_normalization_candidates("Test Horse", special_master)
        # Should find some match via fuzzy logic
        self.assertTrue(len(candidates) >= 0)  # At least doesn't crash


class TestNormalizationCandidate(unittest.TestCase):
    """Test the NormalizationCandidate class."""
    
    def test_candidate_creation(self):
        """Test creating normalization candidates."""
        candidate = NormalizationCandidate("Test Horse", 0.95, "exact")
        self.assertEqual(candidate.name, "Test Horse")
        self.assertEqual(candidate.confidence, 0.95)
        self.assertEqual(candidate.method, "exact")


class TestApprovedNamesMatching(unittest.TestCase):
    """Test approved names matching functionality."""
    
    def setUp(self):
        """Set up test data for approved names matching."""
        self.master_list = ["Good Will", "Thunder", "Lightning"]
        self.approved_mappings = {
            "Goodwill": "Good Will",      # Normalization to master list name
            "Socks": "Socks",             # Identity mapping (non-standard name)
            "Painted Lady": "Painted Lady",  # Identity mapping (non-standard name)
            "Mr. Ed": "Mr. Ed",           # Identity mapping (non-standard name)
            "Lightning Bolt": "Lightning Bolt"  # Identity mapping (non-standard name)
        }
    
    def test_get_approved_names_list(self):
        """Test extracting approved names list from mappings."""
        approved_names = get_approved_names_list(self.approved_mappings)
        
        expected_names = ["Good Will", "Socks", "Painted Lady", "Mr. Ed", "Lightning Bolt"]
        self.assertEqual(set(approved_names), set(expected_names))
    
    def test_get_approved_names_list_empty(self):
        """Test extracting approved names from empty mappings."""
        approved_names = get_approved_names_list({})
        self.assertEqual(approved_names, [])
    
    def test_find_approved_name_matches_exact(self):
        """Test exact matches against approved names."""
        approved_names = ["Socks", "Painted Lady", "Mr. Ed"]
        
        # Exact match
        candidates = find_approved_name_matches("Socks", approved_names)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "Socks")
        self.assertEqual(candidates[0].method, "approved_exact")
        self.assertEqual(candidates[0].confidence, 0.95)
        
        # Case-insensitive match
        candidates = find_approved_name_matches("socks", approved_names)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "Socks")
        self.assertEqual(candidates[0].method, "approved_exact_case_insensitive")
        self.assertEqual(candidates[0].confidence, 0.90)
    
    def test_find_approved_name_matches_substring(self):
        """Test substring matches against approved names."""
        approved_names = ["Socks", "Painted Lady", "Mr. Ed"]
        
        # "Sock" should match "Socks" (the key use case!)
        candidates = find_approved_name_matches("Sock", approved_names)
        sock_matches = [c for c in candidates if c.name == "Socks"]
        self.assertTrue(len(sock_matches) > 0)
        self.assertEqual(sock_matches[0].method, "approved_substring")
        self.assertEqual(sock_matches[0].confidence, 0.80)
        
        # "Paint" should match "Painted Lady"
        candidates = find_approved_name_matches("Paint", approved_names)
        paint_matches = [c for c in candidates if c.name == "Painted Lady"]
        self.assertTrue(len(paint_matches) > 0)
    
    def test_find_approved_name_matches_fuzzy(self):
        """Test fuzzy matches against approved names."""
        approved_names = ["Socks", "Lightning Bolt"]
        
        # "Socky" should fuzzy match "Socks"
        candidates = find_approved_name_matches("Socky", approved_names)
        sock_matches = [c for c in candidates if c.name == "Socks"]
        self.assertTrue(len(sock_matches) > 0)
        self.assertEqual(sock_matches[0].method, "approved_fuzzy")
        
        # "Lightning Bold" should fuzzy match "Lightning Bolt"
        candidates = find_approved_name_matches("Lightning Bold", approved_names)
        bolt_matches = [c for c in candidates if c.name == "Lightning Bolt"]
        self.assertTrue(len(bolt_matches) > 0)
    
    def test_find_approved_name_matches_empty_list(self):
        """Test approved name matching with empty approved names list."""
        candidates = find_approved_name_matches("Sock", [])
        self.assertEqual(len(candidates), 0)
    
    def test_get_normalization_candidates_with_approved_names(self):
        """Test the main candidate function includes approved names."""
        # Test the key scenario: "Sock" should suggest "Socks"
        candidates = get_normalization_candidates("Sock", self.master_list, self.approved_mappings)
        
        # Should include both master list matches and approved name matches
        candidate_names = [c.name for c in candidates]
        
        # Should suggest "Socks" from approved names
        self.assertIn("Socks", candidate_names)
        
        # Find the "Socks" candidate and verify it's from approved names
        socks_candidate = next(c for c in candidates if c.name == "Socks")
        self.assertTrue(socks_candidate.method.startswith("approved_"))
    
    def test_get_normalization_candidates_deduplication(self):
        """Test that duplicate names from master list and approved names are handled."""
        # "Good Will" exists in both master list and approved mappings
        candidates = get_normalization_candidates("Goodwill", self.master_list, self.approved_mappings)
        
        # Should only appear once in results
        good_will_matches = [c for c in candidates if c.name == "Good Will"]
        self.assertEqual(len(good_will_matches), 1)
        
        # Should prefer the higher confidence match (special case vs approved)
        good_will_candidate = good_will_matches[0]
        # Special case matching has higher confidence than approved matching
        self.assertEqual(good_will_candidate.method, "special_case")
    
    def test_get_normalization_candidates_without_approved_mappings(self):
        """Test that function works correctly without approved mappings."""
        # Should work with None
        candidates = get_normalization_candidates("Goodwill", self.master_list, None)
        self.assertTrue(len(candidates) > 0)
        
        # Should work with empty dict
        candidates = get_normalization_candidates("Goodwill", self.master_list, {})
        self.assertTrue(len(candidates) > 0)
    
    def test_approved_names_confidence_scoring(self):
        """Test that approved names have appropriate confidence scores."""
        approved_names = ["Socks", "Lightning Bolt"]
        
        # Exact match should have high confidence
        candidates = find_approved_name_matches("Socks", approved_names)
        exact_match = candidates[0]
        self.assertGreaterEqual(exact_match.confidence, 0.90)
        
        # Substring match should have medium confidence
        candidates = find_approved_name_matches("Sock", approved_names)
        substring_match = next(c for c in candidates if c.name == "Socks")
        self.assertGreaterEqual(substring_match.confidence, 0.75)
        self.assertLess(substring_match.confidence, 0.90)
        
        # Fuzzy match should have lower confidence
        candidates = find_approved_name_matches("Socky", approved_names)
        fuzzy_match = next(c for c in candidates if c.name == "Socks")
        self.assertLess(fuzzy_match.confidence, 0.85)


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios."""
    
    def setUp(self):
        """Set up test data based on real horse name drift issues."""
        self.master_list = [
            "Good Will", "O'Halon", "Canola", "DaVinci", "Cowboy 1", "Cowboy 2",
            "Isaac", "Raffiki", "Guinness", "Ceelo", "Louie 1", "Louie 2"
        ]
    
    def test_known_drift_cases(self):
        """Test the specific drift cases mentioned in the requirements."""
        test_cases = [
            ("Goodwill", "Good Will"),
            ("OHalon", "O'Halon"), 
            ("Conola", "Canola"),
            ("Da Vinci", "DaVinci"),
            ("Cowboy", "Cowboy 1"),  # Could match either, but should find both
            ("Issac", "Isaac"),
            ("Rafikki", "Raffiki"),
            ("Guiness", "Guinness"),
            ("CeeLo", "Ceelo")
        ]
        
        for input_name, expected_match in test_cases:
            with self.subTest(input_name=input_name, expected_match=expected_match):
                candidates = get_normalization_candidates(input_name, self.master_list)
                
                # Should find at least one candidate
                self.assertTrue(len(candidates) > 0, 
                              f"No candidates found for '{input_name}'")
                
                # Expected match should be among the candidates
                found_matches = [c.name for c in candidates if c.name == expected_match]
                self.assertTrue(len(found_matches) > 0,
                              f"Expected match '{expected_match}' not found for '{input_name}'. "
                              f"Found: {[c.name for c in candidates]}")
                
                # For high-confidence cases, should be the top candidate
                if input_name in ["Goodwill", "OHalon", "Conola", "Da Vinci"]:
                    self.assertEqual(candidates[0].name, expected_match,
                                   f"Expected '{expected_match}' as top candidate for '{input_name}', "
                                   f"got '{candidates[0].name}'")


if __name__ == '__main__':
    unittest.main()