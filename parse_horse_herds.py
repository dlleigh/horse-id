#!/usr/bin/env python3
"""
Parse Master Horse-Location List.xlsx and create CSV with horse names and herds.
"""

import pandas as pd
import csv
import sys
import os
import yaml
from pathlib import Path

def load_config():
    """Load configuration from config.yml"""
    try:
        from config_utils import load_config as load_cfg
        return load_cfg()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def setup_paths(config):
    """Setup input and output paths from configuration"""
    try:
        from config_utils import get_data_root
        data_root = get_data_root(config)
        excel_file = config['herd_parser']['master_horse_location_file'].format(data_root=data_root)
        output_file = config['paths']['horse_herds_file'].format(data_root=data_root)
        return excel_file, output_file
    except KeyError as e:
        print(f"Error: Missing path configuration for '{e}' in 'config.yml'.")
        sys.exit(1)

def parse_horse_herds(excel_file=None, output_file=None):
    """
    Parse the Excel file and extract horse names with their corresponding herds.
    For horses with shared names, creates entries for each herd where that name appears.
    
    Args:
        excel_file (str): Path to the Excel file (optional, uses config if None)
        output_file (str): Path to output CSV file (optional, uses config if None)
    """
    
    # Use config paths if not provided
    if excel_file is None or output_file is None:
        config = load_config()
        excel_file, output_file = setup_paths(config)
    
    # Check if input file exists
    if not Path(excel_file).exists():
        print(f"Error: File '{excel_file}' not found")
        sys.exit(1)
    
    # Read the Excel file
    df = pd.read_excel(excel_file, sheet_name=0)
    
    # Find herd headers and their positions
    herd_info = []
    
    # Row 3 contains herd names and counts
    row_3 = df.iloc[3]
    
    # Parse herd information from row 3
    for i in range(0, len(row_3), 2):
        herd_name = row_3.iloc[i]
        if pd.notna(herd_name) and herd_name != "Feed:":
            herd_count = row_3.iloc[i + 1] if i + 1 < len(row_3) else 0
            herd_info.append({
                'name': herd_name,
                'count': herd_count,
                'col_index': i
            })
    
    print(f"Found {len(herd_info)} herds:")
    for herd in herd_info:
        print(f"  - {herd['name']}: {herd['count']} horses")
    
    # Extract horse names for each herd
    horse_data = []
    
    def create_base_name(name: str) -> str:
        """Create base name by removing trailing numbers."""
        import re
        # Remove trailing space + number pattern
        base = re.sub(r'\s+\d+$', '', name)
        return base.strip()
    
    for herd in herd_info:
        col_index = herd['col_index']
        herd_name = herd['name']
        
        # Start from row 5 (index 4) to skip header rows
        for row_idx in range(5, len(df)):
            horse_name = df.iloc[row_idx, col_index]
            
            # If we find a valid horse name (not NaN and not empty)
            if pd.notna(horse_name) and str(horse_name).strip():
                horse_name_str = str(horse_name).strip()
                horse_data.append({
                    'horse_name': horse_name_str,
                    'herd': herd_name,
                    'basename': create_base_name(horse_name_str)
                })
    
    # Remove duplicates while preserving order (keep multiple herds for same horse name)
    seen = set()
    unique_horses = []
    for horse in horse_data:
        key = (horse['horse_name'], horse['herd'])
        if key not in seen:
            seen.add(key)
            unique_horses.append(horse)
    
    print(f"\nExtracted {len(unique_horses)} unique horse-herd combinations")
    
    # Analyze horses appearing in multiple herds
    horse_name_counts = {}
    for horse in unique_horses:
        name = horse['horse_name']
        if name not in horse_name_counts:
            horse_name_counts[name] = []
        horse_name_counts[name].append(horse['herd'])
    
    multi_herd_horses = {name: herds for name, herds in horse_name_counts.items() if len(herds) > 1}
    if multi_herd_horses:
        print(f"\nHorses found in multiple herds ({len(multi_herd_horses)} horses):")
        for name, herds in sorted(multi_herd_horses.items()):
            print(f"  - {name}: {', '.join(sorted(herds))}")
    else:
        print("\nNo horses found in multiple herds.")
    
    # Find horses with names in format "name number" (e.g., "George 2", "Sunny 1")
    import re
    numbered_horses = []
    for horse in unique_horses:
        horse_name = horse['horse_name']
        # Match pattern: word characters followed by space and number
        if re.match(r'^.+\s+\d+$', horse_name):
            numbered_horses.append(horse)
    
    if numbered_horses:
        print(f"\nHorses with numbered names ({len(numbered_horses)} found):")
        # Sort alphabetically by horse name
        numbered_horses.sort(key=lambda x: x['horse_name'])
        for horse in numbered_horses:
            print(f"  - {horse['horse_name']} ({horse['herd']})")
    else:
        print("\nNo horses found with numbered names.")
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['horse_name', 'herd', 'basename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for horse in unique_horses:
            writer.writerow(horse)
    
    print(f"Successfully created '{output_file}' with {len(unique_horses)} horse-herd combinations")
    
    # Display summary by herd
    herd_counts = {}
    for horse in unique_horses:
        herd = horse['herd']
        herd_counts[herd] = herd_counts.get(herd, 0) + 1
    
    print("\nHorse-herd combinations per herd:")
    for herd, count in sorted(herd_counts.items()):
        print(f"  - {herd}: {count} horses")
    
    print(f"\nTotal unique horse names: {len(horse_name_counts)}")
    print(f"Total horse-herd combinations: {len(unique_horses)}")
    
    return unique_horses

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse horse herds from Excel file")
    parser.add_argument("--input", "-i", default=None, 
                       help="Input Excel file path (uses config.yml if not specified)")
    parser.add_argument("--output", "-o", default=None, 
                       help="Output CSV file path (uses config.yml if not specified)")
    
    args = parser.parse_args()
    
    # If no arguments provided, use config paths
    if args.input is None and args.output is None:
        print("Using paths from config.yml...")
        parse_horse_herds()
    else:
        parse_horse_herds(args.input, args.output)