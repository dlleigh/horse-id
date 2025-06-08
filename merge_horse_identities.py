import os
import yaml
import pickle
import pandas as pd
import numpy as np
import torch
import timm
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import torchvision.transforms as T

from wildlife_tools.features import DeepFeatures, SuperPointExtractor, AlikedExtractor, DiskExtractor, SiftExtractor
from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue
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
            matcher=MatchLightGlue(features='superpoint'), extractor=SuperPointExtractor(),
            transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
            calibration=IsotonicCalibration()
        ),
        'lightglue_aliked': SimilarityPipeline(
            matcher=MatchLightGlue(features='aliked'), extractor=AlikedExtractor(),
            transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
            calibration=IsotonicCalibration()
        ),
    }

    # Load pre-computed calibrations
    for name, pipeline in calibrated_matchers_dict.items():
        cal_file = os.path.join(CALIBRATION_DIR, f"{name}.pkl")
        if os.path.exists(cal_file):
            print(f"  Loading calibration for {name} from: {cal_file}")
            with open(cal_file, 'rb') as f:
                pipeline.calibration = pickle.load(f)
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

    return WildFusion(
        calibrated_pipelines=list(calibrated_matchers_dict.values()),
        priority_pipeline=priority_matcher
    )

def check_similarity(wildfusion_system, df_a, df_b):
    """
    Checks if two groups of images (dataframes) are likely the same horse.
    """
    if df_a.empty or df_b.empty:
        return False

    dataset_a = ImageDataset(df_a, root=IMAGE_DIR)
    dataset_b = ImageDataset(df_b, root=IMAGE_DIR)

    try:
        # B is a performance parameter for WildFusion, can be tuned
        similarity_matrix = wildfusion_system(dataset_a, dataset_b, B=10)
        if similarity_matrix is None or similarity_matrix.size == 0:
            return False

        # Average the maximum similarity score for each query image
        avg_max_sim = np.mean(np.max(similarity_matrix, axis=1))
        return avg_max_sim >= SIMILARITY_THRESHOLD
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
    df = pd.read_csv(INPUT_MANIFEST_FILE)

    # Filter for only single horses, as multiple/none are ambiguous
    df_single = df[df['num_horses_detected'] == 'SINGLE'].copy()

    # Group by the non-unique horse name
    grouped_by_name = df_single.groupby('horse_name')
    df_updated = df.copy() # We'll update this dataframe

    for name, group in tqdm(grouped_by_name, desc="Processing names"):
        unique_ids = group['canonical_id'].unique()
        if len(unique_ids) <= 1:
            continue # No merging needed for this name

        print(f"\nProcessing '{name}' with {len(unique_ids)} potential identities.")
        graph = defaultdict(list)
        id_pairs = combinations(unique_ids, 2)

        # Build the graph of similar identities
        for id_a, id_b in id_pairs:
            print(f"  Comparing ID {id_a} vs ID {id_b}...")
            df_a = group[group['canonical_id'] == id_a]
            df_b = group[group['canonical_id'] == id_b]

            if check_similarity(wildfusion, df_a, df_b):
                print(f"    --> Match found between ID {id_a} and ID {id_b}!")
                graph[id_a].append(id_b)
                graph[id_b].append(id_a)

        # Find the clusters of connected IDs
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
                # Update the main dataframe
                df_updated.loc[df_updated['canonical_id'].isin(ids_to_merge), 'canonical_id'] = final_id

    # Save the final, merged manifest
    print(f"\nSaving merged manifest to: {OUTPUT_MANIFEST_FILE}")
    df_updated.to_csv(OUTPUT_MANIFEST_FILE, index=False)
    print("Process complete.")


if __name__ == '__main__':
    main()