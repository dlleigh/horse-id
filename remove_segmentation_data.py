#!/usr/bin/env python3
"""
Remove segmentation and bounding box data from horse ID manifest files.

This script removes the following columns that are no longer needed:
- bbox_x, bbox_y, bbox_width, bbox_height
- segmentation_mask  
- size_ratio

Based on analysis in README_MASKING_ANALYSIS.md, segmentation and cropping data
hurts performance and is not used for reprocessing detection.
"""

import os
import pandas as pd
import yaml
from datetime import datetime

def load_config():
    """Load configuration from config.yml"""
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_root = os.path.expanduser(config['paths']['data_root'])
    detected_file = config['paths']['detected_manifest_file'].format(data_root=data_root)
    merged_file = config['paths']['merged_manifest_file'].format(data_root=data_root)
    
    return detected_file, merged_file

def backup_file(filepath):
    """Create a timestamped backup of the original file"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    
    # Copy file to backup
    df = pd.read_csv(filepath)
    df.to_csv(backup_path, index=False)
    print(f"Created backup: {backup_path}")
    return backup_path

def remove_segmentation_columns(df):
    """Remove segmentation and bbox columns from dataframe"""
    columns_to_remove = [
        'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
        'segmentation_mask', 'size_ratio'
    ]
    
    removed_cols = []
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed_cols.append(col)
    
    return df, removed_cols

def get_file_size_mb(filepath):
    """Get file size in MB"""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0

def clean_manifest_file(filepath, file_description):
    """Clean a single manifest file"""
    print(f"\n=== Processing {file_description} ===")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Get original file size
    original_size = get_file_size_mb(filepath)
    print(f"Original file size: {original_size:.2f} MB")
    
    # Create backup
    backup_path = backup_file(filepath)
    if not backup_path:
        return
    
    # Load the manifest
    print(f"Loading manifest: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Remove segmentation columns
    cleaned_df, removed_cols = remove_segmentation_columns(df)
    
    if removed_cols:
        print(f"Removed columns: {', '.join(removed_cols)}")
        print(f"Remaining columns: {len(cleaned_df.columns)}")
        
        # Save cleaned file
        cleaned_df.to_csv(filepath, index=False)
        new_size = get_file_size_mb(filepath)
        
        print(f"New file size: {new_size:.2f} MB")
        print(f"Size reduction: {original_size - new_size:.2f} MB ({((original_size - new_size) / original_size * 100):.1f}%)")
    else:
        print("No segmentation columns found to remove")

def main():
    """Main function to clean manifest files"""
    print("Horse ID Segmentation Data Removal Tool")
    print("=" * 50)
    
    # Load configuration
    try:
        detected_file, merged_file = load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Clean detected manifest
    clean_manifest_file(detected_file, "Detected Manifest")
    
    # Clean merged manifest  
    clean_manifest_file(merged_file, "Merged Manifest")
    
    print(f"\n=== Cleanup Complete ===")
    print("Segmentation and bounding box data removed from manifest files.")
    print("Original files backed up with timestamp.")
    print("The system will continue to work normally for reprocessing detection.")

if __name__ == "__main__":
    main()