import os
import yaml
import pickle
import pandas as pd
import numpy as np
import torch
import timm
from datetime import datetime
import csv # For logging similarity scores
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import torchvision.transforms as T
import re

from wildlife_tools.features import DeepFeatures, SuperPointExtractor, AlikedExtractor, DiskExtractor, SiftExtractor
from wildlife_tools.similarity import MatchLOFTR, CosineSimilarity, MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.data import ImageDataset

# --- Load Configuration ---
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# --- Use the config values ---
DATA_ROOT = os.path.expanduser(config['paths']['data_root'])
IMAGE_DIR = config['paths']['dataset_dir'].format(data_root=DATA_ROOT)
# Reads from the detector's output, writes to the final merged file
INPUT_MANIFEST_FILE = config['detection']['detected_manifest_file'].format(data_root=DATA_ROOT)
OUTPUT_MANIFEST_FILE = config['paths']['merged_manifest_file'].format(data_root=DATA_ROOT)
CALIBRATION_DIR = config['paths']['calibration_dir'].format(data_root=DATA_ROOT) # Keep for WildFusion
MERGE_RESULTS_FILE = config['paths']['merge_results_file'].format(data_root=DATA_ROOT) # New name
SIMILARITY_THRESHOLD = config['similarity']['merge_threshold']
MASTER_HORSE_LOCATION_FILE = config['similarity']['master_horse_location_file'].format(data_root=DATA_ROOT)


def load_recurring_names():
    """
    Load the master horse location file and identify horses with recurring names
    (names that end with a number like 'Cowboy 1', 'Cowboy 2').
    Returns a set of base names that have recurring horses.
    """
    if not os.path.exists(MASTER_HORSE_LOCATION_FILE):
        print(f"Warning: Master horse location file not found at {MASTER_HORSE_LOCATION_FILE}")
        return set()
    
    try:
        print(f"Loading master horse location file: {MASTER_HORSE_LOCATION_FILE}")
        # Read the Excel file
        df = pd.read_excel(MASTER_HORSE_LOCATION_FILE, sheet_name=0)
        
        # Find herd headers and their positions (following parse_horse_herds.py logic)
        herd_info = []
        row_3 = df.iloc[3]  # Row 3 contains herd names and counts
        
        # Parse herd information from row 3
        for i in range(0, len(row_3), 2):
            herd_name = row_3.iloc[i]
            if pd.notna(herd_name) and herd_name != "Feed:":
                herd_info.append({
                    'name': herd_name,
                    'col_index': i
                })
        
        # Extract all horse names
        all_horse_names = []
        for herd in herd_info:
            col_index = herd['col_index']
            
            # Start from row 5 (index 4) to skip header rows
            for row_idx in range(5, len(df)):
                horse_name = df.iloc[row_idx, col_index]
                
                # If we find a valid horse name (not NaN and not empty)
                if pd.notna(horse_name) and str(horse_name).strip():
                    all_horse_names.append(str(horse_name).strip())
        
        # Find horses with numbered names (e.g., "Cowboy 1", "Sunny 2")
        numbered_horses = []
        for horse_name in all_horse_names:
            # Match pattern: any characters followed by space and number
            if re.match(r'^.+\s+\d+$', horse_name):
                numbered_horses.append(horse_name)
        
        # Extract base names (everything before the last space and number)
        recurring_base_names = set()
        for horse_name in numbered_horses:
            # Remove the last space and number to get the base name
            base_name = re.sub(r'\s+\d+$', '', horse_name)
            recurring_base_names.add(base_name)
        
        print(f"Found {len(numbered_horses)} horses with numbered names")
        print(f"Identified {len(recurring_base_names)} recurring base names: {sorted(recurring_base_names)}")
        
        return recurring_base_names
        
    except Exception as e:
        print(f"Error loading master horse location file: {e}")
        return set()


def is_recurring_name(horse_name, recurring_base_names):
    """
    Check if a horse name belongs to a recurring name pattern.
    
    Args:
        horse_name (str): The horse name to check
        recurring_base_names (set): Set of base names that have recurring horses
    
    Returns:
        bool: True if the horse name is part of a recurring pattern
    """
    # Check if the name itself ends with a number (e.g., "Cowboy 1")
    if re.match(r'^.+\s+\d+$', horse_name):
        base_name = re.sub(r'\s+\d+$', '', horse_name)
        return base_name in recurring_base_names
    
    # Check if the base name (without number) is in recurring names
    return horse_name in recurring_base_names


def auto_merge_non_recurring_names(output_df, recurring_base_names):
    """
    Automatically merge horses with non-recurring names without detailed analysis.
    
    Args:
        output_df (DataFrame): The output dataframe to modify
        recurring_base_names (set): Set of base names that have recurring horses
    
    Returns:
        DataFrame: Modified dataframe with auto-merged non-recurring names
    """
    # Filter for only single horses
    df_single = output_df[output_df['num_horses_detected'] == 'SINGLE'].copy()
    
    # Group by horse name
    grouped_by_name = df_single.groupby('horse_name')
    
    merge_count = 0
    for name, group in grouped_by_name:
        # Skip if this is a recurring name - it will be handled by detailed analysis
        if is_recurring_name(name, recurring_base_names):
            continue
            
        # Skip if only one message (nothing to merge)
        unique_message_ids = group['message_id'].unique()
        if len(unique_message_ids) <= 1:
            continue
            
        print(f"Auto-merging non-recurring horse '{name}' across {len(unique_message_ids)} messages")
        
        # Get all canonical IDs for this name
        canonical_ids = group['canonical_id'].unique()
        
        if len(canonical_ids) > 1:
            # Choose the lowest ID as the final canonical ID
            final_id = min(canonical_ids)
            ids_to_merge = [cid for cid in canonical_ids if cid != final_id]
            
            # Update all rows with this horse name to use the final canonical ID
            merge_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rows_to_update_mask = (output_df['horse_name'] == name) & (output_df['canonical_id'].isin(ids_to_merge))
            
            output_df.loc[rows_to_update_mask, 'canonical_id'] = final_id
            output_df.loc[rows_to_update_mask, 'last_merged_timestamp'] = pd.to_datetime(merge_timestamp)
            
            merge_count += 1
            print(f"  Merged IDs {ids_to_merge} into {final_id}")
    
    if merge_count > 0:
        print(f"Auto-merged {merge_count} non-recurring horse names")
    else:
        print("No non-recurring horse names required merging")
    
    return output_df


def load_wildfusion_system():
    """Loads the pre-trained and calibrated WildFusion system components."""
    print("Loading WildFusion system components...")

    calibrated_matchers_dict = {
        'lightglue_superpoint': SimilarityPipeline(
            matcher = MatchLightGlue(features='superpoint'),
            extractor = SuperPointExtractor(),
            transform = T.Compose([
                T.Resize([512, 512]),
                T.ToTensor()
            ]),
            calibration = IsotonicCalibration()
        ),

        # 'lightglue_aliked': SimilarityPipeline(
        #     matcher = MatchLightGlue(features='aliked'),
        #     extractor = AlikedExtractor(),
        #     transform = T.Compose([
        #         T.Resize([512, 512]),
        #         T.ToTensor()
        #     ]),
        #     calibration = IsotonicCalibration()
        # ),

        # 'lightglue_disk': SimilarityPipeline(
        #     matcher = MatchLightGlue(features='disk'),
        #     extractor = DiskExtractor(),
        #     transform = T.Compose([
        #         T.Resize([512, 512]),
        #         T.ToTensor()
        #     ]),
        #     calibration = IsotonicCalibration()
        # ),

        # 'lightglue_sift': SimilarityPipeline(
        #     matcher = MatchLightGlue(features='sift'),
        #     extractor = SiftExtractor(),
        #     transform = T.Compose([
        #         T.Resize([512, 512]),
        #         T.ToTensor()
        #     ]),
        #     calibration = IsotonicCalibration()
        # ),

        'DeepFeatures': SimilarityPipeline(
            matcher=CosineSimilarity(),
            extractor=DeepFeatures(
                model=timm.create_model('hf-hub:BVRA/wildlife-mega-L-384', num_classes=0, pretrained=True)
            ),
            transform=T.Compose([
                T.Resize(size=(384, 384)), T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        ),
    }

    # Load pre-computed calibrations
    for name, pipeline in calibrated_matchers_dict.items():
        if name == 'DeepFeatures':
            continue
        cal_file = os.path.join(CALIBRATION_DIR, f"{name}.pkl")
        if os.path.exists(cal_file):
            print(f"  Loading calibration for {name} from: {cal_file}")
            with open(cal_file, 'rb') as f:
                pipeline.calibration = pickle.load(f)
                pipeline.calibration_done = True
        else:
            raise FileNotFoundError(f"Calibration file not found for {name}: {cal_file}. Please run training first.")

    priority_matcher = SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(
            model=timm.create_model('hf-hub:BVRA/wildlife-mega-L-384', num_classes=0, pretrained=True)
        ),
        transform=T.Compose([
            T.Resize(size=(384, 384)), T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
    )

    # return WildFusion(
    #     calibrated_pipelines=list(calibrated_matchers_dict.values()),
    #     priority_pipeline=priority_matcher
    # )
    return WildFusion(
        calibrated_pipelines=calibrated_matchers_dict.values(),
        #priority_pipeline=priority_matcher
    )

def check_similarity(wildfusion_system, df_a, df_b):
    """
    Checks if two groups of images are a match using a more robust
    similarity check. Returns (is_match, max_similarity_score).
    """
    overall_max_similarity = np.nan
    if df_a.empty or df_b.empty:
        return False, overall_max_similarity

    dataset_a = ImageDataset(df_a, col_label='canonical_id', root=IMAGE_DIR, col_path='filename')
    dataset_b = ImageDataset(df_b, col_label='canonical_id', root=IMAGE_DIR, col_path='filename')
    try:
        similarity_matrix = wildfusion_system(dataset_a, dataset_b)
        if similarity_matrix is None or similarity_matrix.size == 0:
            return False, overall_max_similarity

        # --- NEW LOGIC: Match if any single pair is above threshold ---
        if similarity_matrix.size > 0: # Ensure matrix is not empty before calling np.max
            overall_max_similarity = np.max(similarity_matrix)
            print(f"    Max similarity found across all pairs: {overall_max_similarity:.4f} (threshold: {SIMILARITY_THRESHOLD})")
            return overall_max_similarity >= SIMILARITY_THRESHOLD, overall_max_similarity
        else:
            # overall_max_similarity remains np.nan, is_match remains False
            print(f"    Similarity matrix is empty, no match.")
            return False, overall_max_similarity
    except Exception as e:
        print(f"    Error during similarity check: {e}")
        return False, overall_max_similarity

def find_connected_components(graph):
    """Finds all connected components in a graph using BFS."""
    visited = set()
    components = []
    for node in graph:
        if node not in visited:
            component = []
            q = [node]
            visited.add(node)
            while q:
                current = q.pop(0)
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            components.append(component)
    return components

def main():
    """Merges horse identities in the manifest based on photo similarity."""
    print("Starting horse identity merging process...")
    
    # Load recurring names from master horse location file
    recurring_base_names = load_recurring_names()
    
    wildfusion = load_wildfusion_system()

    # INPUT_MANIFEST_FILE is config['detection']['detected_manifest_file']
    # OUTPUT_MANIFEST_FILE is config['paths']['merged_manifest_file'] (this script's output)
    print(f"Reading detected manifest: {INPUT_MANIFEST_FILE}")
    try:
        detected_df = pd.read_csv(INPUT_MANIFEST_FILE, dtype={'message_id': str, 'filename': str})
    except FileNotFoundError:
        print(f"Error: Detected manifest file not found at {INPUT_MANIFEST_FILE}. Cannot proceed.")
        return

    output_df = detected_df.copy()

    # Ensure original_canonical_id exists. It's crucial for initializing canonical_id.
    if 'original_canonical_id' not in output_df.columns:
        print("Error: 'original_canonical_id' column missing. Initializing from 'canonical_id' in detected manifest.")
        # This assumes 'canonical_id' from detected_df is effectively the original if 'original_canonical_id' is missing.
        output_df['original_canonical_id'] = output_df['canonical_id']

    # Initialize 'canonical_id' from 'original_canonical_id' for all rows.
    # Previous merge results will overwrite this for relevant rows.
    output_df['canonical_id'] = output_df['original_canonical_id']
    if 'last_merged_timestamp' not in output_df.columns:
        output_df['last_merged_timestamp'] = pd.NaT
    else:
        output_df['last_merged_timestamp'] = pd.to_datetime(output_df['last_merged_timestamp'], errors='coerce').fillna(pd.NaT)

    # Load previous merge results if the output file exists
    if os.path.exists(OUTPUT_MANIFEST_FILE):
        print(f"Loading previous merge results from: {OUTPUT_MANIFEST_FILE}")
        previous_merges_df = pd.read_csv(OUTPUT_MANIFEST_FILE, dtype={'message_id': str, 'filename': str})

        if not previous_merges_df.empty:
            # Merge previous canonical_id and last_merged_timestamp into output_df
            cols_to_carry = ['filename']
            if 'canonical_id' in previous_merges_df.columns: cols_to_carry.append('canonical_id')
            if 'last_merged_timestamp' in previous_merges_df.columns: cols_to_carry.append('last_merged_timestamp')
            
            temp_df = pd.merge(output_df,
                               previous_merges_df[cols_to_carry],
                               on='filename',
                               how='left',
                               suffixes=('', '_prev'))
            
            # If a previous canonical_id exists for a file, use it. Otherwise, keep the one from original_canonical_id.
            if 'canonical_id_prev' in temp_df.columns:
                output_df['canonical_id'] = temp_df['canonical_id_prev'].fillna(temp_df['canonical_id'])
            if 'last_merged_timestamp_prev' in temp_df.columns:
                temp_df['last_merged_timestamp_prev'] = pd.to_datetime(temp_df['last_merged_timestamp_prev'], errors='coerce').fillna(pd.NaT)
                output_df['last_merged_timestamp'] = temp_df['last_merged_timestamp_prev'].fillna(temp_df['last_merged_timestamp'])

    # Ensure correct dtypes after potential merges
    output_df['canonical_id'] = pd.to_numeric(output_df['canonical_id'], errors='coerce').astype('Int64')
    output_df['original_canonical_id'] = pd.to_numeric(output_df['original_canonical_id'], errors='coerce').astype('Int64')
    output_df['last_merged_timestamp'] = pd.to_datetime(output_df['last_merged_timestamp'], errors='coerce').fillna(pd.NaT)
    
    # Auto-merge non-recurring horse names without detailed analysis
    print("\n=== Auto-merging non-recurring horse names ===")
    output_df = auto_merge_non_recurring_names(output_df, recurring_base_names)

    # --- Load existing merge results ---
    comparison_results_cache = {} # Key: frozenset({cid_a, cid_b}), Value: dict of log entry
    if os.path.exists(MERGE_RESULTS_FILE):
        print(f"Loading existing merge results from: {MERGE_RESULTS_FILE}")
        try:
            existing_results_df = pd.read_csv(MERGE_RESULTS_FILE, dtype={'canonical_id_a': str, 'canonical_id_b': str})
            for _, row in existing_results_df.iterrows():
                try:
                    cid_a = int(float(row['canonical_id_a'])) # Handle potential float strings
                    cid_b = int(float(row['canonical_id_b']))
                    pair_key = frozenset({cid_a, cid_b})
                    # Store the entire row as a dict, ensuring latest timestamp wins if duplicates somehow exist
                    # (though ideally, keys should be unique in the CSV)
                    current_timestamp = pd.to_datetime(row['timestamp'])
                    if pair_key not in comparison_results_cache or \
                       pd.to_datetime(comparison_results_cache[pair_key]['timestamp']) < current_timestamp:
                        comparison_results_cache[pair_key] = row.to_dict()
                except (ValueError, TypeError) as e:
                    print(f"  Warning: Skipping row in merge_results.csv due to parsing error: {row.to_dict()}, Error: {e}")
        except Exception as e:
            print(f"  Warning: Could not load or parse {MERGE_RESULTS_FILE}: {e}")

    merge_results_headers = ['timestamp', 'horse_name', 'canonical_id_a', 'canonical_id_b', 'message_id_a', 'message_id_b',
                   'max_similarity', 'is_match'] # Updated log headers

    # Filter for only single horses, as multiple/none are ambiguous
    df_single = output_df[output_df['num_horses_detected'] == 'SINGLE'].copy()

    # Group by the non-unique horse name
    grouped_by_name = df_single.groupby('horse_name')
    
    print(f"\n=== Detailed analysis for recurring horse names ===")

    for name, group in tqdm(grouped_by_name, desc="Processing names"):
        # Only perform detailed analysis for recurring names
        if not is_recurring_name(name, recurring_base_names):
            continue  # Skip non-recurring names as they were already auto-merged
        # CHANGED: The unit of comparison is now message_id, not canonical_id.
        unique_message_ids = group['message_id'].unique()

        if len(unique_message_ids) <= 1:
            continue # No other messages to compare for this name

        print(f"\nProcessing '{name}' with {len(unique_message_ids)} potential identities across different messages.")
        # The graph still connects canonical_ids, as they are the entities we want to merge.
        graph = defaultdict(list)
        
        # CHANGED: Create pairs of message_ids to compare.
        message_id_pairs = combinations(unique_message_ids, 2)

        # Build the graph of similar identities by comparing message_id image sets
        for msg_id_a, msg_id_b in message_id_pairs:
            # CHANGED: Get all images for each message_id.
            df_a = group[group['message_id'] == msg_id_a]
            df_b = group[group['message_id'] == msg_id_b]

            # Get the CURRENT canonical_id for these message groups from output_df
            # These IDs might already reflect previous merges.
            canonical_id_a = df_a['canonical_id'].iloc[0]
            canonical_id_b = df_b['canonical_id'].iloc[0]

            # CHANGED: If they already have the same ID, no need to compare or merge.
            if canonical_id_a == canonical_id_b:
                continue

            # Ensure CIDs are integers for cache key
            try:
                cid_a_int = int(canonical_id_a)
                cid_b_int = int(canonical_id_b)
            except (ValueError, TypeError):
                print(f"  Warning: Invalid canonical IDs ({canonical_id_a}, {canonical_id_b}) for pair ('{msg_id_a}', '{msg_id_b}'). Skipping this pair.")
                continue
            
            pair_key = frozenset({cid_a_int, cid_b_int})
            is_match_for_graph = False

            if pair_key in comparison_results_cache:
                cached_result = comparison_results_cache[pair_key]
                is_match_for_graph = str(cached_result.get('is_match', 'false')).lower() == 'true'
                score_str = cached_result.get('max_similarity', 'N/A')
                print(f"  Using cached result for CIDs ({cid_a_int}, {cid_b_int}) for messages ('{msg_id_a}', '{msg_id_b}'): {'Match' if is_match_for_graph else 'No Match'}, Score: {score_str}")
            else:
                print(f"  Comparing message '{msg_id_a}' (CID: {cid_a_int}) vs message '{msg_id_b}' (CID: {cid_b_int})...")
                is_match_computed, score_computed_float = check_similarity(wildfusion, df_a, df_b)
                is_match_for_graph = is_match_computed
                score_str = f"{score_computed_float:.4f}" if not np.isnan(score_computed_float) else 'N/A'

                # Update cache with new result
                new_log_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'horse_name': name,
                    'canonical_id_a': cid_a_int,
                    'canonical_id_b': cid_b_int,
                    'message_id_a': msg_id_a,
                    'message_id_b': msg_id_b,
                    'max_similarity': score_str,
                    'is_match': is_match_computed
                }
                comparison_results_cache[pair_key] = new_log_entry

            if is_match_for_graph:
                print(f"    --> Graph edge added between canonical IDs {cid_a_int} and {cid_b_int} based on messages '{msg_id_a}' & '{msg_id_b}'.")
                graph[cid_a_int].append(cid_b_int)
                graph[cid_b_int].append(cid_a_int)

        # This part remains the same: find clusters of connected canonical_ids and merge them.
        components = find_connected_components(graph)
        if not components:
            print(f"  No merges found for '{name}'.")
            continue

        print(f"  Found {len(components)} component(s) to merge for '{name}'.")
        for component in components:
            if len(component) > 1:
                # Choose the lowest ID as the new canonical ID for the whole group
                final_id = min(component)
                ids_to_merge = [i for i in component if i != final_id]
                print(f"    Merging IDs {ids_to_merge} into {final_id}.")
                 # --- NEW: Update audit columns ---
                merge_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                rows_to_update_mask = output_df['canonical_id'].isin(ids_to_merge)
                # Update the canonical_id for the merged rows
                output_df.loc[rows_to_update_mask, 'canonical_id'] = final_id
                # Set the timestamp for the merged rows
                output_df.loc[rows_to_update_mask, 'last_merged_timestamp'] = pd.to_datetime(merge_timestamp)

    # --- Save all comparison results (old and new) to merge_results.csv ---
    print(f"\nSaving all comparison results to: {MERGE_RESULTS_FILE}")
    if comparison_results_cache: # Check if there's anything to save
        results_to_save_df = pd.DataFrame(list(comparison_results_cache.values()))
        results_to_save_df.to_csv(MERGE_RESULTS_FILE, index=False, columns=merge_results_headers)
    else: # If cache is empty (e.g. first run and no comparisons made, or file was empty)
        with open(MERGE_RESULTS_FILE, 'w', newline='') as f: # Create empty file with headers
            writer = csv.DictWriter(f, fieldnames=merge_results_headers)
            writer.writeheader()

    # Save the final, merged manifest
    print(f"\nSaving merged manifest to: {OUTPUT_MANIFEST_FILE}")
    output_df.to_csv(OUTPUT_MANIFEST_FILE, index=False)
    print("Process complete.")


if __name__ == '__main__':
    main()