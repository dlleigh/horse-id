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
import boto3
from botocore.exceptions import ClientError

from wildlife_datasets import datasets
from wildlife_tools.inference import TopkClassifier
import torchvision.transforms as T
import timm
from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import ImageDataset
from wildlife_tools.similarity import CosineSimilarity

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
        data_root = os.path.join(os.getcwd(),'horse-id-data')
        manifest_file = config['paths']['merged_manifest_file'].format(data_root=data_root)
        features_dir = config['paths']['features_dir'].format(data_root=data_root)
        s3_bucket_name = config['s3']['bucket_name']
        return manifest_file, features_dir, s3_bucket_name
    except KeyError as e:
        print(f"Error: Missing path configuration for '{e}' in '{CONFIG_FILE}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up paths from config: {e}")
        sys.exit(1)

def download_from_s3(s3_client, bucket_name, s3_key, local_path):
    """Downloads a file from S3 if it doesn't exist locally."""
    if os.path.exists(local_path):
        print(f"  File already exists locally: {local_path}")
        return True

    print(f"  Downloading s3://{bucket_name}/{s3_key} to {local_path}...")
    try:
        # Ensure local directory exists
        local_dir = os.path.dirname(local_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir)

        s3_client.download_file(bucket_name, s3_key, local_path)
        print("  Download complete.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"    ERROR: The file was not found in S3: s3://{bucket_name}/{s3_key}")
        else:
            print(f"    ERROR: Failed to download file from S3. Reason: {e}")
        return False

def identify_horse(image_url):
    config = load_config()
    manifest_file, features_dir, s3_bucket_name = setup_paths(config)

    # --- S3 Download Logic ---
    print("\nChecking for required files from S3...")
    s3_client = boto3.client('s3')

    # 1. Download manifest file
    manifest_s3_key = os.path.basename(manifest_file)
    if not download_from_s3(s3_client, s3_bucket_name, manifest_s3_key, manifest_file):
        print("Exiting: Could not retrieve manifest file.")
        sys.exit(1)

    # 2. Download features file
    features_local_path = os.path.join(features_dir, 'database_deep_features.pkl')
    # Construct S3 key based on upload script logic: features_dir_basename/filename
    features_s3_key = os.path.join(os.path.basename(features_dir), 'database_deep_features.pkl').replace("\\", "/")
    if not download_from_s3(s3_client, s3_bucket_name, features_s3_key, features_local_path):
        print("Exiting: Could not retrieve features file.")
        sys.exit(1)
    print("--------------------------------------------")

    if not os.path.isfile(manifest_file):
        print(f"Error: MANIFEST_FILE '{manifest_file}' not found.")
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
            for pred, score in zip(predictions[0], scores[0]):
                print(f"  identity: {pred}, Score: {score:.4f}")
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
