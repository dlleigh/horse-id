import os
import yaml
import pickle
import pandas as pd
import numpy as np
import torch
import timm
from datetime import datetime
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import torchvision.transforms as T

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
CALIBRATION_DIR = config['paths']['calibration_dir'].format(data_root=DATA_ROOT)
SIMILARITY_THRESHOLD = config['similarity']['threshold']


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
    }

    # Load pre-computed calibrations
    for name, pipeline in calibrated_matchers_dict.items():
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
        #calibrated_pipelines=calibrated_matchers_dict.values()
        calibrated_pipelines=[priority_matcher]
    )

def check_similarity(wildfusion_system, df_a, df_b):
    """
    Checks if two groups of images are a match using a more robust
    bidirectional similarity check.
    """
    if df_a.empty or df_b.empty:
        return False

    dataset_a = ImageDataset(df_a, col_label='canonical_id', root=IMAGE_DIR, col_path='filename')
    dataset_b = ImageDataset(df_b, col_label='canonical_id', root=IMAGE_DIR, col_path='filename')

    try:
        similarity_matrix = wildfusion_system(dataset_a, dataset_b)
        if similarity_matrix is None or similarity_matrix.size == 0:
            return False

        # --- IMPROVED LOGIC: Bidirectional Check ---

        # 1. Calculate similarity from A -> B
        # For each image in A, find its best match in B, then average those scores.
        avg_max_sim_ab = np.mean(np.max(similarity_matrix, axis=1))

        # 2. Calculate similarity from B -> A
        # For each image in B, find its best match in A, then average those scores.
        avg_max_sim_ba = np.mean(np.max(similarity_matrix, axis=0))

        # 3. Combine the scores. Taking the mean is a good general-purpose approach.
        # Taking the minimum (np.min) would be even stricter.
        final_similarity = (avg_max_sim_ab + avg_max_sim_ba) / 2

        print(f"    Sim A->B: {avg_max_sim_ab:.4f}, Sim B->A: {avg_max_sim_ba:.4f}")
        print(f"    Combined similarity: {final_similarity:.4f} (threshold: {SIMILARITY_THRESHOLD})")

        return final_similarity >= SIMILARITY_THRESHOLD
    except Exception as e:
        print(f"    Error during similarity check: {e}")
        return False

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
    wildfusion = load_wildfusion_system()

    print(f"Reading manifest: {INPUT_MANIFEST_FILE}")
    # Read the manifest and ensure message_id is present and of a consistent type
    df = pd.read_csv(INPUT_MANIFEST_FILE, dtype={'message_id': str})

    # Filter for only single horses, as multiple/none are ambiguous
    df_single = df[df['num_horses_detected'] == 'SINGLE'].copy()

    # Group by the non-unique horse name
    grouped_by_name = df_single.groupby('horse_name')
    df_updated = df.copy() # We'll update this dataframe

    for name, group in tqdm(grouped_by_name, desc="Processing names"):
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

            # CHANGED: Get the canonical_id associated with each message.
            # It's guaranteed that all images in a message have the same canonical_id.
            canonical_id_a = df_a['canonical_id'].iloc[0]
            canonical_id_b = df_b['canonical_id'].iloc[0]

            # CHANGED: If they already have the same ID, no need to compare or merge.
            if canonical_id_a == canonical_id_b:
                continue

            print(f"  Comparing message '{msg_id_a}' (ID: {canonical_id_a}) vs message '{msg_id_b}' (ID: {canonical_id_b})...")

            if check_similarity(wildfusion, df_a, df_b):
                print(f"    --> Match found between canonical IDs {canonical_id_a} and {canonical_id_b}!")
                # CHANGED: Add an edge between the respective canonical_ids, not the message_ids.
                graph[canonical_id_a].append(canonical_id_b)
                graph[canonical_id_b].append(canonical_id_a)

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
                rows_to_update_mask = df_updated['canonical_id'].isin(ids_to_merge)
                # Update the canonical_id for the merged rows
                df_updated.loc[rows_to_update_mask, 'canonical_id'] = final_id
                # Set the timestamp for the merged rows
                df_updated.loc[rows_to_update_mask, 'last_merged_timestamp'] = merge_timestamp

    # Save the final, merged manifest
    print(f"\nSaving merged manifest to: {OUTPUT_MANIFEST_FILE}")
    df_updated.to_csv(OUTPUT_MANIFEST_FILE, index=False)
    print("Process complete.")


if __name__ == '__main__':
    main()