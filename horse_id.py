import argparse
import os
import pandas as pd
import numpy as np
import requests
from PIL import Image
import io
import tempfile
import pickle
import yaml
import sys

from wildlife_datasets import datasets
from wildlife_tools.inference import TopkClassifier
import torchvision.transforms as T
import timm
from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import ImageDataset
from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline
from wildlife_tools.similarity.calibration import IsotonicCalibration

# --- Configuration ---
CONFIG_FILE = 'config.yml'

# --- Horses Class (Copied from notebook) ---
class Horses(datasets.WildlifeDataset):
    def __init__(self, root_dir, manifest_file_path):
        self.manifest_file_path = manifest_file_path
        super().__init__(root_dir, check_files=False)

    def create_catalogue(self) -> pd.DataFrame:
        """Create catalogue from manifest file"""
        manifest_df = pd.read_csv(self.manifest_file_path)
        
        rows = []
        for _, row in manifest_df.iterrows():
            if 'status' in row and row['status'] == 'EXCLUDE':
                continue
            if 'num_horses_detected' in row and row['num_horses_detected'] in ['NONE', 'MULTIPLE']:
                continue
            rows.append({
                'image_id': row['filename'],
                #'identity': row['canonical_id'],
                'identity': row['horse_name'],
                'path': row['filename'],
                'date': pd.to_datetime(str(row['email_date']), format='%Y%m%d')
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
        calibration_dir = config['paths']['calibration_dir'].format(data_root=data_root)
        features_dir = config['paths']['features_dir'].format(data_root=data_root)
        return image_dir, manifest_file, calibration_dir, features_dir
    except KeyError as e:
        print(f"Error: Missing path configuration for '{e}' in '{CONFIG_FILE}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up paths from config: {e}")
        sys.exit(1)

def identify_horse(image_url):
    config = load_config()
    image_dir, manifest_file, calibration_dir, features_dir = setup_paths(config)

    if not os.path.isdir(image_dir):
        print(f"Error: IMAGE_DIR '{image_dir}' not found or not a directory.")
        sys.exit(1)
    if not os.path.isfile(manifest_file):
        print(f"Error: MANIFEST_FILE '{manifest_file}' not found.")
        sys.exit(1)
    if not os.path.isdir(calibration_dir):
        print(f"Error: CALIBRATION_DIR '{calibration_dir}' not found or not a directory.")
        sys.exit(1)
    if not os.path.isdir(features_dir):
        print(f"Error: FEATURES_DIR '{features_dir}' not found or not a directory.")
        sys.exit(1)

    print("Initializing Horse dataset...")
    horses_dataset_obj = Horses(None, manifest_file_path=manifest_file)
    horses_df_all = horses_dataset_obj.create_catalogue()
    if horses_df_all.empty:
        print("Error: The horse catalogue is empty. Check manifest file and Horses class.")
        sys.exit(1)

    dataset_database = ImageDataset(horses_df_all, horses_dataset_obj.root)

    print(f"Downloading image from {image_url}...")
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img_bytes = io.BytesIO(response.content)
        # Validate if the downloaded content can be opened as an image by PIL
        try:
            Image.open(img_bytes).verify() # verify() checks for corruption
            img_bytes.seek(0) # Reset stream position after verify
        except (IOError, SyntaxError) as e:
            print(f"Error: Downloaded content from {image_url} is not a valid image or is corrupted: {e}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(img_bytes.getvalue())
        temp_image_path = tmp_file.name
    
    print(f"Temporary image saved to: {temp_image_path}")

    try:
        query_df = pd.DataFrame([{
            'path': os.path.basename(temp_image_path), # ImageDataset needs relative path from its root
            'identity': -1, # Dummy identity for query
            'image_id': 'query_image',
            'horse_name': 'query_horse', # Dummy name
            'date': pd.Timestamp.now() # Dummy date
        }])
        transform = T.Compose([T.Resize([384, 384]), T.ToTensor()])
        dataset_query_single = ImageDataset(query_df, os.path.dirname(temp_image_path), transform=transform)

        print("Extracting features for query image...")

        backbone = timm.create_model('hf-hub:BVRA/wildlife-mega-L-384', num_classes=0, pretrained=True)
        extractor = DeepFeatures(backbone)
        query_features = extractor(dataset_query_single)

        features_output_path = os.path.join(features_dir, 'database_deep_features.pkl')
        print(f"Loading database features from {features_output_path}...")
        with open(features_output_path, 'rb') as f:
            database_features = pickle.load(f)

        print("Calculating similarity...")
        similarity_function = CosineSimilarity()
        similarity = similarity_function(query_features, database_features)
        
        db_labels = dataset_database.labels_string

        classifier = TopkClassifier(k=5, database_labels=db_labels, return_all=True)
        predictions,scores,idx = classifier(similarity)

        # Load the confidence threshold from the configuration
        CONFIDENCE_THRESHOLD = config['similarity']['inference_threshold']

        print("\n--- Predictions above Confidence Threshold ---")
        found_above_threshold = False
        for pred, score in zip(predictions[0], scores[0]):
            if score > CONFIDENCE_THRESHOLD:
                print(f"  Predicted identity: {pred}, Score: {score:.4f}")
                found_above_threshold = True
        if not found_above_threshold:
            print(f"  No predictions found above the confidence threshold of {CONFIDENCE_THRESHOLD}.")
        print("--------------------------------------------")

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print(f"Cleaned up temporary image: {temp_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify a horse from an image URL.")
    parser.add_argument("image_url", type=str, help="URL of the horse image to identify.")
    
    args = parser.parse_args()
    
    identify_horse(args.image_url)
    os._exit(0)
    #sys.exit(0)
