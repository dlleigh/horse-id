# extract_features.py
import os
import pandas as pd
import pickle
import yaml
import sys

# Assume WildlifeDataset and ImageDataset are available or copy/import them
from wildlife_datasets import datasets
from wildlife_tools.data import ImageDataset
from wildlife_tools.features import DeepFeatures
import torchvision.transforms as T
import timm

# --- Configuration ---
CONFIG_FILE = 'config.yml' # Same config as horse_id.py

# --- Horses Class (Copied from notebook, ensure it's identical) ---
class Horses(datasets.WildlifeDataset):
    def __init__(self, root_dir, manifest_file_path):
        self.manifest_file_path = manifest_file_path
        super().__init__(root_dir)

    def create_catalogue(self) -> pd.DataFrame:
        """Create catalogue from manifest file"""
        manifest_df = pd.read_csv(self.manifest_file_path)
        
        rows = []
        for _, row in manifest_df.iterrows():
            if 'status' in row and row['status'] == 'EXCLUDE':
                continue
            if 'num_horses_detected' in row and row['num_horses_detected'] in ['NONE', 'MULTIPLE']:
                continue

            # Parse segmentation mask from "x1 y1;x2 y2;..." into a list of tuples.
            # This format is expected by wildlife_tools.data.ImageDataset.
            segmentation_data = None
            mask_str = row.get('segmentation_mask')
            if mask_str and pd.notna(mask_str) and isinstance(mask_str, str):
                try:
                    points = [
                        (float(p.split()[0]), float(p.split()[1]))
                        for p in mask_str.split(';') if ' ' in p
                    ]
                    if points:
                        segmentation_data = points
                except (ValueError, IndexError):
                    # If parsing fails, segmentation_data remains None, and the full image is used.
                    pass

            rows.append({
                'image_id': row['filename'],
                'identity': row['canonical_id'],
                'horse_name': row['horse_name'],
                'path': row['filename'],
                'date': pd.to_datetime(str(row['email_date']), format='%Y%m%d'),
                'segmentation': segmentation_data,
            })
        
        df = pd.DataFrame(rows)
        result = self.finalize_catalogue(df)
        return result

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        print("Please ensure it's in the same directory as the script or provide the correct path.")
        sys.exit(1)
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_paths(config):
    try:
        data_root = os.path.expanduser(config['paths']['data_root'])
        image_dir = config['paths']['dataset_dir'].format(data_root=data_root)
        manifest_file = config['paths']['merged_manifest_file'].format(data_root=data_root)
        # Add a path for saving features
        features_dir = config['paths']['features_dir'].format(data_root=data_root)
        os.makedirs(features_dir, exist_ok=True)
        return image_dir, manifest_file, features_dir
    except KeyError as e:
        print(f"Error: Missing path configuration for '{e}' in '{CONFIG_FILE}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up paths from config: {e}")
        sys.exit(1)

def extract_and_save_features(image_dir, manifest_file, features_dir):
    print("Initializing Horse dataset for feature extraction...")
    horses_dataset_obj = Horses(image_dir, manifest_file_path=manifest_file)
    horses_df_all = horses_dataset_obj.create_catalogue()
    transform = T.Compose([T.Resize([384, 384]), T.ToTensor()])
    image_dataset = ImageDataset(
        horses_df_all,
        image_dir,
        transform=transform,
        segmentation='segmentation',
        segmentation_kind='polygon'
    )

    print("Extracting features for all database images...")

    backbone = timm.create_model('hf-hub:BVRA/wildlife-mega-L-384', num_classes=0, pretrained=True)
    extractor = DeepFeatures(backbone)
    features = extractor(image_dataset)

    features_output_path = os.path.join(features_dir, 'database_deep_features.pkl')

    print(f"Saving extracted features to {features_output_path}")
    with open(features_output_path, 'wb') as f:
        pickle.dump(features, f)

    print("Feature extraction complete")

if __name__ == "__main__":
    config = load_config()
    image_dir, manifest_file, features_dir = setup_paths(config)
    
    # Ensure necessary directories exist
    os.makedirs(features_dir, exist_ok=True)
    
    extract_and_save_features(image_dir, manifest_file, features_dir)